"""Tempo, beat and downbeat detection for a whole track (offline mode)."""

from __future__ import annotations

import numpy as np

from .features import robust_norm


def onset_envelope(feats: dict[str, np.ndarray], frame_rate: float) -> np.ndarray:
    """One number per frame: 'how much did something just hit'. Kick-weighted, because EDM."""
    env = (
        0.6 * robust_norm(feats["kick_flux"])
        + 0.25 * robust_norm(feats["snare_flux"])
        + 0.15 * robust_norm(feats["flux"])
    )
    # subtract a ~0.5 s moving average so slow loudness changes don't count as hits
    k = max(3, int(0.5 * frame_rate))
    local = np.convolve(env, np.ones(k) / k, mode="same")
    return np.maximum(env - local, 0.0)


def phase_envelope(feats: dict[str, np.ndarray], frame_rate: float) -> np.ndarray:
    """Kick + snare only. Used to decide WHERE the beat is: hats and basslines often
    sit on the offbeat (house, dubstep) and would pull the grid half a beat off."""
    env = 0.5 * robust_norm(feats["kick_flux"]) + 0.5 * robust_norm(feats["snare_flux"])
    k = max(3, int(0.5 * frame_rate))
    return np.maximum(env - np.convolve(env, np.ones(k) / k, mode="same"), 0.0)


def estimate_tempo(env: np.ndarray, frame_rate: float, bpm_range=(90.0, 190.0), prior_bpm=128.0) -> float:
    """Autocorrelation tempo estimate, nudged toward typical EDM tempos."""
    x = env - env.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0] + 1e-12

    lag_min = int(60 * frame_rate / bpm_range[1])
    lag_max = min(n - 1, int(60 * frame_rate / bpm_range[0]) + 1)
    lags = np.arange(max(lag_min, 1), lag_max)
    if len(lags) == 0:
        return prior_bpm
    bpms = 60 * frame_rate / lags
    # comb: a real beat period also repeats at 2, 3 and 4 beats (= a bar). This is what
    # stops half-time patterns (kick on 1 and the "and" of 2) reading as 2/3 tempo.
    score = sum(acf[np.minimum(lags * k, n - 1)] for k in (1, 2, 3, 4)) / 4
    score *= np.exp(-0.5 * (np.log2(bpms / prior_bpm) / 0.8) ** 2)
    i = int(np.argmax(score))
    lag = float(lags[i])
    if 0 < i < len(lags) - 1:  # parabolic interpolation for sub-frame accuracy
        a, b, c = score[i - 1], score[i], score[i + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
            lag += 0.5 * (a - c) / denom
    return _rerank_ratios(acf, frame_rate, 60 * frame_rate / lag, bpm_range)


def _acf_at(acf: np.ndarray, lag: float) -> float:
    i = int(lag)
    if i + 1 >= len(acf):
        return 0.0
    f = lag - i
    return float(acf[i] * (1 - f) + acf[i + 1] * f)


def _rerank_ratios(acf: np.ndarray, frame_rate: float, bpm: float, bpm_range) -> float:
    """Syncopated beats (trap 808s, dubstep "kick on 1 and the and of 2", DnB breaks)
    repeat every 1.5 beats or so, which fools a beat-level autocorrelation into 2/3 or
    3/4 tempo. What they can't fake is the BAR: the whole pattern repeats every 4 beats.
    So among tempos related by simple ratios, pick the one whose 1-, 2- and 4-bar
    periods line up best."""
    n = len(acf)
    cands = [bpm * r for r in (1 / 2, 2 / 3, 3 / 4, 4 / 5, 1, 5 / 4, 4 / 3, 3 / 2, 2)]
    cands = [c for c in cands if bpm_range[0] <= c <= bpm_range[1]]

    def score(b):
        L = 60 * frame_rate / b
        bars = [_acf_at(acf, k * L) for k in (4, 8, 16) if k * L < 0.6 * n]
        if not bars:
            return -1.0
        beat = (_acf_at(acf, L) + _acf_at(acf, 2 * L)) / 2
        prior = np.exp(-0.5 * (np.log2(b / 135) / 1.0) ** 2)
        return (0.65 * float(np.mean(bars)) + 0.35 * beat) * (0.85 + 0.15 * prior)

    best = max(cands, key=score, default=bpm)
    return best if score(best) > score(bpm) + 1e-9 else bpm


def track_beats(env: np.ndarray, frame_rate: float, bpm: float, tightness: float = 300.0) -> np.ndarray:
    """Dynamic-programming beat tracker (Ellis 2007). Returns beat frame indices."""
    period = 60 * frame_rate / bpm
    n = len(env)
    onset = env / (env.std() + 1e-9)
    score = onset.copy()
    back = np.full(n, -1, np.int64)
    lo_off, hi_off = int(round(2 * period)), int(round(period / 2))
    offsets = np.arange(hi_off, lo_off + 1)
    cost = -tightness * np.log(offsets / period) ** 2
    for t in range(hi_off, n):
        prev = t - offsets
        ok = prev >= 0
        cand = score[prev[ok]] + cost[ok]
        j = int(np.argmax(cand))
        if cand[j] > 0:
            score[t] = onset[t] + cand[j]
            back[t] = prev[ok][j]
    # start from the best-scoring frame in the last beat period, then walk back
    tail = max(0, n - int(period))
    t = tail + int(np.argmax(score[tail:]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    beats = np.array(beats[::-1], np.int64)
    return _fill_gaps(beats, period, n)


def fold_phase(env: np.ndarray, period: float, offset: float = 0.0) -> tuple[float, float]:
    """Where do hits land within a beat of `period` frames? Returns (phase 0..1, sharpness).

    Circular mean around the strongest bin, so the answer is continuous (no bin dead zone).
    """
    n = len(env)
    ph = ((np.arange(n) - offset) / period) % 1.0
    bins = 64
    hist = np.bincount((ph * bins).astype(np.int64) % bins, weights=env, minlength=bins)
    hist = hist + np.roll(hist, 1) + np.roll(hist, -1)
    peak = (np.argmax(hist) + 0.5) / bins
    sharp = hist.max() / (hist.mean() + 1e-12)
    d = (ph - peak + 0.5) % 1.0 - 0.5
    near = np.abs(d) < 0.2
    w = env[near]
    if w.sum() <= 0:
        return peak, sharp
    return (peak + float((d[near] * w).sum() / w.sum())) % 1.0, sharp


def fit_grid(env: np.ndarray, frame_rate: float, bpm: float, search=0.008,
             phase_env: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Fit one rigid beat grid to the whole track (EDM tracks don't change tempo).

    For each candidate period, fold the onset envelope modulo the period: the right
    period gives one sharp peak, which also gives the phase. Much more robust than
    a beat-by-beat tracker through snare rolls and offbeat basslines.
    Returns (beat_frames, sharpness) where sharpness is peak/mean of the fold.
    """
    n = len(env)
    base = 60 * frame_rate / bpm
    idx = np.arange(n)
    best = (-1.0, base, 0.0)
    bins = 160
    for period in base * (1 + np.linspace(-search, search, 81)):
        pos = (idx % period) / period * bins
        hist = np.bincount(pos.astype(np.int64) % bins, weights=env, minlength=bins)
        hist = hist + np.roll(hist, 1) + np.roll(hist, -1)  # small circular smoothing
        j = int(np.argmax(hist))
        sharp = hist[j] / (hist.mean() + 1e-12)
        if sharp > best[0]:
            best = (sharp, period, (j + 0.5) / bins * period)
    sharp, period, phase = best
    if phase_env is not None:
        phase = fold_phase(phase_env, period)[0] * period
    frames = np.arange(phase, n, period)
    return np.clip(np.round(frames), 0, n - 1).astype(np.int64), sharp


def tempo_is_steady(env: np.ndarray, frame_rate: float, bpm: float, bpm_range, tol=0.012) -> bool:
    """Checks each quarter of the track has the same tempo (a DJ mix might not)."""
    n = len(env)
    if n < frame_rate * 60:
        return True
    for q in range(4):
        part = env[q * n // 4:(q + 1) * n // 4]
        if part.std() < 1e-6:
            continue
        local = estimate_tempo(part, frame_rate, bpm_range, prior_bpm=bpm)
        ratio = local / bpm
        if min(abs(ratio - 1), abs(ratio * 2 - 1), abs(ratio / 2 - 1)) > tol:
            return False
    return True


def _fill_gaps(beats: np.ndarray, period: float, n: int) -> np.ndarray:
    """Quiet intros/breakdowns can leave gaps. Keep the grid going through them."""
    if len(beats) == 0:
        return np.arange(0, n, period).astype(np.int64)
    out = []
    # extend backwards to the start of the track
    t = beats[0] - period
    while t >= -period * 0.25:
        out.append(t)
        t -= period
    out = out[::-1]
    for a, b in zip(beats[:-1], beats[1:]):
        out.append(a)
        gap = b - a
        k = int(round(gap / period))
        if k > 1:
            out.extend(a + gap * np.arange(1, k) / k)
    out.append(beats[-1])
    t = beats[-1] + period
    while t < n:
        out.append(t)
        t += period
    return np.clip(np.round(np.array(out)), 0, n - 1).astype(np.int64)


def find_downbeat_phase(feats: dict[str, np.ndarray], beat_frames: np.ndarray) -> int:
    """Which beat (0-3) of the first bar is the '1'?

    Three clues, z-scored and added up per candidate phase:
    * spectral change: chords, bass notes, crashes and sections change on the 1
    * kick ON the beat: every style has a kick on the 1
    * snare ON the beat counts against: snares/claps sit on 2 & 4 (four-on-the-floor)
      or on 3 (half-time dubstep/trap), never on the 1. Without this, half-time
      tracks tie between 1 and 3 and lights end up a half bar off.
    """
    if len(beat_frames) < 9:
        return 0
    coarse = feats["coarse"]
    segs = [coarse[a:b].mean(axis=0) for a, b in zip(beat_frames[:-1], beat_frames[1:]) if b > a]
    sync = np.array(segs)
    novelty = np.r_[0.0, np.linalg.norm(np.diff(sync, axis=0), axis=1)]
    n = len(novelty)

    def at_beats(x):
        x = robust_norm(x)
        return np.array([x[max(0, f - 3): f + 4].max() for f in beat_frames[:n]])

    def z(v):
        return (v - v.mean()) / (v.std() + 1e-9)

    score = z(novelty) + z(at_beats(feats["kick_flux"])) - z(at_beats(feats["snare_flux"]))
    return int(np.argmax(downbeat_scores(feats, beat_frames, score)))


def downbeat_scores(feats, beat_frames, _score=None) -> np.ndarray:
    """Per-phase evidence (4 numbers) that beat p of the list is a '1'. See find_downbeat_phase."""
    if _score is None:
        if len(beat_frames) < 9:
            return np.zeros(4)
        coarse = feats["coarse"]
        sync = np.array([coarse[a:b].mean(axis=0) for a, b in zip(beat_frames[:-1], beat_frames[1:]) if b > a])
        novelty = np.r_[0.0, np.linalg.norm(np.diff(sync, axis=0), axis=1)]
        n = len(novelty)

        def at_beats(x):
            x = robust_norm(x)
            return np.array([x[max(0, f - 3): f + 4].max() for f in beat_frames[:n]])

        def z(v):
            return (v - v.mean()) / (v.std() + 1e-9)

        _score = z(novelty) + z(at_beats(feats["kick_flux"])) - z(at_beats(feats["snare_flux"]))
    return np.array([_score[p::4].mean() for p in range(4)])
