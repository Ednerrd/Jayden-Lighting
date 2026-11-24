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


def label_sections(m: dict[str, np.ndarray]) -> list[Section]:
    segs = _segment(m)
    if not segs:
        return []
    kick = m["kick"]
    lo, hi = np.percentile(kick, [15, 90])
    kick_thr = (lo + hi) / 2 if hi - lo > 0.2 else -1.0  # kick all the way through? then all bars count

    info = []
    for a, b in segs:
        info.append({
            "a": a, "b": b,
            "kick": float(np.mean(kick[a:b] > kick_thr)) > 0.5,
            "energy": float(m["energy"][a:b].mean() + 0.5 * m["low"][a:b].mean()),
        })
    kick_e = [s["energy"] for s in info if s["kick"]]
    top = max(kick_e) if kick_e else 1.0
    for i, s in enumerate(info):
        if s["kick"]:
            # drops come right after a build/breakdown. A loud kick section without
            # one has to be clearly the loudest thing in the track to count.
            after_break = i > 0 and not info[i - 1]["kick"]
            s["kind"] = DROP if s["energy"] >= top - (0.2 if after_break else 0.05) else GROOVE
        else:
            s["kind"] = BREAKDOWN

    # merge neighbours with the same label, so a breakdown split in two is one run
    runs = []
    for s in info:
        if runs and runs[-1]["kind"] == s["kind"]:
            r = runs[-1]
            w0, w1 = r["b"] - r["a"], s["b"] - s["a"]
            r["energy"] = (r["energy"] * w0 + s["energy"] * w1) / (w0 + w1)
            r["b"] = s["b"]
        else:
            runs.append(dict(s))
    info = runs

    # carve builds out of whatever comes right before each drop
    out: list[Section] = []
    for i, s in enumerate(info):
        nxt = info[i + 1] if i + 1 < len(info) else None
        a, b, kind = s["a"], s["b"], s["kind"]
        if nxt and nxt["kind"] == DROP and kind != DROP:
            tension = m["tension"][a:b] + 0.5 * m["density"][a:b]
            length = b - a
            if not s["kick"]:
                if length <= 8:
                    build_start = a
                else:
                    # build starts after the last bar where tension is still near its
                    # floor, measured back from the drop in whole phrases
                    search = tension[: length - PHRASE]
                    floor = search.min() + 0.25 * (tension.max() - search.min())
                    last_calm = int(np.flatnonzero(search <= floor)[-1])
                    blen = int(np.clip(round((length - last_calm - 1) / PHRASE) * PHRASE, PHRASE, 16))
                    build_start = b - blen
            else:
                tail = min(8, length)
                rising = _slope(tension[-tail:]) > 0.02
                build_start = b - tail if rising else b
            if build_start > a:
                out.append(Section(kind, a, build_start))
            if build_start < b:
                out.append(Section(BUILD, build_start, b))
        else:
            out.append(Section(kind, a, b))

    # intro / outro
    if out and out[0].kind in (GROOVE, BREAKDOWN):
        out[0].kind = INTRO
    last_drop = max((i for i, s in enumerate(out) if s.kind == DROP), default=None)
    if last_drop is not None:
        n_bars = out[-1].end_bar
        for s in out[last_drop + 1:]:
            if s.kind in (GROOVE, BREAKDOWN) and s.start_bar >= 0.75 * n_bars:
                s.kind = OUTRO

    merged: list[Section] = []
    for s in out:
        if merged and merged[-1].kind == s.kind:
            merged[-1].end_bar = s.end_bar
        else:
            merged.append(s)
    return merged
