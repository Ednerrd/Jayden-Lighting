"""Finds the structure of a track: intro / groove / breakdown / build / drop / outro.

Works bar by bar. EDM is built from 4/8/16-bar phrases, so boundaries are only
allowed on a 4-bar grid. The rules are simple on purpose so they are easy to tune:

* a "kick" bar has a strong four-on-the-floor kick
* the loudest kick sections are drops, quieter ones are grooves
* no-kick sections are breakdowns
* the stretch right before a drop where tension (hats, noise risers, snare rolls,
  brightness) climbs is the build
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..state import BREAKDOWN, BUILD, DROP, GROOVE, INTRO, OUTRO
from .features import robust_norm

PHRASE = 4  # bars


@dataclass
class Section:
    kind: str
    start_bar: int
    end_bar: int  # exclusive


def bar_features(feats, beat_frames: np.ndarray, downbeat_phase: int, onsets: dict[str, np.ndarray]):
    """Aggregate frame features into one row per bar. Returns (bar_start_beats, matrix dict)."""
    n_frames = len(feats["rms"])
    kick_flux = feats["kick_flux"]
    p95 = np.percentile(kick_flux, 95) + 1e-9
    # strongest kick hit anywhere in each beat (half-time kicks land on the "and")
    ends = np.r_[beat_frames[1:], n_frames]
    beat_kick = np.array([kick_flux[max(0, a - 3): max(a - 2, b - 3)].max() / p95 for a, b in zip(beat_frames, ends)])
    beat_kick = np.clip(beat_kick, 0, 1.5)

    low = robust_norm(feats["kick_e"])
    energy = robust_norm(20 * np.log10(feats["rms"] + 1e-6))
    high = robust_norm(feats["hat_e"] * 0.6 + feats["snare_e"] * 0.4)
    bright = robust_norm(np.log(feats["centroid"] + 1))
    noise = robust_norm(feats["flatness"])

    bar_starts = np.arange(downbeat_phase, len(beat_frames) - 4, 4)
    rows = {k: [] for k in ("kick", "low", "energy", "high", "tension", "density")}
    hits = np.sort(np.concatenate([onsets["snare"], onsets["hat"]]))
    for b in bar_starts:
        f0, f1 = beat_frames[b], beat_frames[min(b + 4, len(beat_frames) - 1)]
        f1 = max(f1, f0 + 1)
        f1 = min(f1, n_frames)
        # 2nd-strongest beat of the bar: high for four-on-the-floor AND half-time
        # (kick on 1 + kick/snare thump later), low for snare rolls / pads
        rows["kick"].append(np.sort(beat_kick[b:b + 4])[-2])
        rows["low"].append(low[f0:f1].mean())
        rows["energy"].append(energy[f0:f1].mean())
        rows["high"].append(high[f0:f1].mean())
        rows["tension"].append((high[f0:f1].mean() + bright[f0:f1].mean() + noise[f0:f1].mean()) / 3)
        rows["density"].append(np.count_nonzero((hits >= f0) & (hits < f1)))
    m = {k: np.asarray(v, float) for k, v in rows.items()}
    if len(m["density"]):
        m["density"] = m["density"] / max(1.0, np.percentile(m["density"], 95))
    return bar_starts, m


def _segment(m: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    n = len(m["kick"])
    if n == 0:
        return []
    X = np.stack([m["kick"], m["low"], m["energy"], m["high"], m["density"]], axis=1)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    def novelty(b, w=PHRASE):
        if b - 1 < 0 or b + 1 > n:
            return 0.0
        a = X[max(0, b - w): b].mean(axis=0)
        c = X[b: min(n, b + w)].mean(axis=0)
        return float(np.linalg.norm(a - c))

    # phrase grid offset: usually 0, but some tracks start with pickup bars
    offset = max(range(PHRASE), key=lambda o: sum(novelty(b) for b in range(o + PHRASE, n, PHRASE)))
    cands = list(range(offset + PHRASE if offset == 0 else offset, n, PHRASE))
    nov = {b: novelty(b) for b in cands}
    # keep every phrase boundary where things change noticeably. Over-splitting is
    # fine: neighbouring segments that end up with the same label get merged later.
    bounds = [0] + [b for b in cands if nov[b] > 1.0]
    bounds.append(n)
    bounds = sorted(set(bounds))
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def _split_threshold(x: np.ndarray, min_gap: float = 0.15):
    """Otsu split of per-bar values into two groups. None if they aren't clearly two groups."""
    v = np.sort(x)
    if len(v) < 4:
        return None
    best, thr = -1.0, None
    for i in range(1, len(v)):
        a, b = v[:i], v[i:]
        between = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if between > best:
            best, thr = between, (v[i - 1] + v[i]) / 2
            gap = b.mean() - a.mean()
    lo, hi = v[v <= thr], v[v > thr]
    if len(lo) == 0 or len(hi) == 0 or hi.mean() - lo.mean() < min_gap:
        return None
    return thr


def _runs(flags: np.ndarray, min_len: int = 2) -> list[list]:
    """[start, end, flag] runs; runs shorter than min_len bars get absorbed by a neighbour."""
    runs = []
    for i, f in enumerate(flags):
        if runs and runs[-1][2] == f:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1, bool(f)])
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, r in enumerate(runs):
            # keep a 1-bar empty gap between two full runs: the silence right before a drop
            pre_drop_gap = (not r[2] and 0 < i < len(runs) - 1 and runs[i + 1][1] - runs[i + 1][0] >= 4)
            if r[1] - r[0] < min_len and not pre_drop_gap:
                j = i - 1 if i > 0 else i + 1
                runs[j][0], runs[j][1] = min(runs[j][0], r[0]), max(runs[j][1], r[1])
                del runs[i]
                merged = []
                for q in runs:  # re-merge neighbours with the same flag
                    if merged and merged[-1][2] == q[2]:
                        merged[-1][1] = q[1]
                    else:
                        merged.append(q)
                runs = merged
                changed = True
                break
    return runs


def label_sections(m: dict[str, np.ndarray]) -> list[Section]:
    """Label bars by whether the floor is shaking.

    The most reliable "is this a drop?" signal across house, dubstep, trap and DnB is
    sub-bass level (35-120 Hz): drops are full, breakdowns and builds aren't. A kick
    detector gets fooled by wobble basses, 808 tails and heavy limiting; the sub level
    doesn't. So:
      1. bars are FULL or EMPTY by a per-track threshold on sub level
      2. a full stretch right after an empty one is a DROP (all of it - no stripes)
      3. the end of an empty stretch where tension (hats, risers, snare rolls) climbs
         is the BUILD; the rest is a BREAKDOWN
      4. a big energy jump inside a full stretch also counts as a drop (builds where
         the kick never stops), and a clear step down at the end is the OUTRO
    """
    low = m["low"]
    n = len(low)
    if n == 0:
        return []
    # FULL = within reach of drop-level sub. Intros and builds often keep a soft kick,
    # so "louder than the breakdown" isn't enough: it has to be near the loudest bars.
    p90, p10 = np.percentile(low, [90, 10])
    thr = p90 - 0.15 if p90 - p10 > 0.15 else None
    full = low > thr if thr is not None else np.ones(n, bool)
    power = m["energy"] + 0.5 * m["low"] + 0.5 * m["high"]
    tension = m["tension"] + 0.5 * m["density"]

    # split full runs at big upward energy jumps (kick-through builds -> drop)
    pieces = []  # [a, b, full, jumped_in]
    for a, b, f in _runs(full):
        cuts = [a]
        if f and b - a >= 12:
            for c in range(a + 4, b - 3, PHRASE):
                before = power[max(cuts[-1], c - 8):c].mean()
                after = power[c:min(b, c + 8)].mean()
                if after - before > 0.2:
                    cuts.append(c)
        cuts.append(b)
        for i, (x, y) in enumerate(zip(cuts[:-1], cuts[1:])):
            pieces.append([x, y, f, i > 0])

    full_power = [power[a:b].mean() for a, b, f, _ in pieces if f]
    top = max(full_power) if full_power else 1.0
    kinds = []
    groove_power = None  # loudness of the first groove, once we've heard one
    for i, (a, b, f, jumped) in enumerate(pieces):
        if not f:
            kinds.append(BREAKDOWN)
            continue
        after_gap = i > 0 and (not pieces[i - 1][2] or jumped)
        # right after the intro, a full section is usually the groove; it only counts as
        # a drop if it's as big as the biggest part of the track
        margin = 0.08 if i == 1 else 0.2
        pw = power[a:b].mean()
        is_drop = after_gap and pw >= top - margin
        # a section that comes back at groove level (closer to the groove than to the
        # biggest drop) is the groove again, even right after a dip...
        # ...but only after a short dip (<= 8 bars). A real breakdown runs 16+ bars, and
        # whatever comes after one is a drop even when it's no louder than the groove
        short_dip = i > 0 and not pieces[i - 1][2] and pieces[i - 1][1] - pieces[i - 1][0] <= 8
        if is_drop and short_dip and groove_power is not None and pw < (groove_power + top) / 2:
            is_drop = False
        if not is_drop and groove_power is None:
            groove_power = pw
        kinds.append(DROP if is_drop else GROOVE)
    # a short bass dip that keeps the kick going and leads back into the groove is groove
    for i in range(len(pieces) - 1):
        a, b = pieces[i][0], pieces[i][1]
        if kinds[i] == BREAKDOWN and kinds[i + 1] == GROOVE and i > 0 and b - a <= 8 and m["kick"][a:b].mean() >= 0.6:
            kinds[i] = GROOVE

    out: list[Section] = []
    for i, ((a, b, f, _), kind) in enumerate(zip(pieces, kinds)):
        nxt = kinds[i + 1] if i + 1 < len(kinds) else None
        if nxt != DROP or kind == DROP:
            out.append(Section(kind, a, b))
            continue
        # look back from the drop over this piece (and the one before, if it isn't a
        # drop): builds often start while the kick is still going
        lo = a
        if out and out[-1].kind != DROP:
            lo = out[-1].start_bar
        t = tension[a:b]
        length = b - a
        if not f and length <= 8:
            # a short dip (<= 2 phrases) right before a drop IS the build, even when the
            # last bar is a near-silent gap that makes tension look like it's falling
            blen = length
        else:
            search = t[:-2] if length > 4 else t
            floor = search.min() + 0.25 * (t.max() - search.min())
            calm = np.flatnonzero(search <= floor)
            last_calm = int(calm[-1]) if len(calm) else 0
            blen = length - last_calm - 1
            if f and _slope(t[-min(8, length):]) <= 0.01:
                blen = 0  # kick section that doesn't build up: no build
        if blen > 0:
            # builds are whole phrases counted back from the drop. A 5-bar fade means the
            # build really started 3 bars earlier, with the kick still going.
            blen = int(np.clip(np.ceil(blen / PHRASE) * PHRASE, PHRASE, 32))
        start = max(lo, b - blen)
        # drops and grooves come in whole 8-bar phrases. If this build would leave the
        # section before it on a half phrase (say 12 bars), the build really started 4 bars
        # earlier, with the kick still going (it sounds the same, only the phrase tells)
        if out and out[-1].kind in (DROP, GROOVE) and out[-1].end_bar >= start:
            dlen = start - out[-1].start_bar
            if dlen >= 12 and dlen % 8 == 4:
                start -= 4
                out[-1].end_bar = start
        # trim whatever we're eating into
        while out and out[-1].start_bar >= start and out[-1].kind != DROP:
            out.pop()
        if out and out[-1].end_bar > start and out[-1].kind != DROP:
            out[-1].end_bar = start
        elif out and out[-1].end_bar > start:
            start = out[-1].end_bar
        if start > a and (not out or out[-1].end_bar < a):
            out.append(Section(kind, a, start))
        elif start > a:
            out.append(Section(kind, max(a, out[-1].end_bar), start))
        if start < b:
            out.append(Section(BUILD, start, b))

    # outro: a clear, lasting step down in energy at the end of the last drop
    last_drop = max((i for i, s in enumerate(out) if s.kind == DROP), default=None)
    if last_drop is not None:
        d = out[last_drop]
        best_c, best_gain = None, 0.12
        for c in range(d.start_bar + 8, d.end_bar - 3, PHRASE):
            gain = power[d.start_bar:c].mean() - power[c:d.end_bar].mean()
            if gain > best_gain:
                best_c, best_gain = c, gain
        if best_c is not None:
            out.insert(last_drop + 1, Section(OUTRO, best_c, d.end_bar))
            d.end_bar = best_c
        n_bars = out[-1].end_bar
        for s in out[last_drop + 1:]:
            if s.kind in (GROOVE, BREAKDOWN) and s.start_bar >= 0.75 * n_bars:
                s.kind = OUTRO
    if out and out[0].kind in (GROOVE, BREAKDOWN):
        out[0].kind = INTRO

    merged: list[Section] = []
    for s in out:
        if merged and merged[-1].kind == s.kind:
            merged[-1].end_bar = s.end_bar
        else:
            merged.append(s)
    return merged
