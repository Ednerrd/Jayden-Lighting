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
FAKEOUT_MAX = 12        # a "drop" shorter than this (bars) can be a fake-out
FAKEOUT_GAP = 16        # ...if the real drop comes within this many bars after it
FAKEOUT_BRIGHTER = 0.1  # ...and is this much brighter (tension)
DEEP_DIP = 0.5          # a sub dip this far under the loudest bars is a breakdown, not a groove dip
TOP_COLLAPSE = 0.2      # a dip whose tension falls this far under the groove's is a breakdown
RISER_SLOPE = 0.03      # a groove's last 8 bars climb at least this fast (tension + density/2 per bar)...
RISER_RISE = 0.25       # ...and end this far over their calm level: that's a build into the break
LONG_INTRO = 20         # an intro with the kick running for more than this many bars is a breakdown


@dataclass
class Section:
    kind: str
    start_bar: int
    end_bar: int  # exclusive
    stage2_bar: int | None = None  # BUILD only: the bar where the 2nd, more intense build stage starts


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


def _deep_dip(low: np.ndarray, p90: float) -> bool:
    """The sub really leaves (two bars in a row at least DEEP_DIP under the loudest bars), as opposed to a
    groove dip that only thins it. Not Even Love 1:45 bottoms out at 0.55 under a 0.86 top (a groove
    dip); Breathe It Out Slowly 1:49 fades to 0.28 under a 0.97 top (a breakdown)."""
    return len(low) >= 2 and bool((np.convolve(low, [0.5, 0.5], "valid") < p90 - DEEP_DIP).any())


def _top_collapse(t: np.ndarray, groove_t: np.ndarray) -> bool:
    """The top end falls away in the dip (two bars in a row TOP_COLLAPSE under the groove before it): a
    breakdown even with the kick running (Breathe It Out Slowly 2:40). A groove dip keeps its tension
    (Not Even Love 1:45: 0.43 vs 0.45)."""
    return len(t) >= 2 and bool((np.convolve(t, [0.5, 0.5], "valid") < groove_t.mean() - TOP_COLLAPSE).any())


def _riser_builds(secs: list[Section], m: dict[str, np.ndarray], tension: np.ndarray) -> list[Section]:
    """A groove that ends in a climbing riser (tension and hi-hat density rising, kick and sub still in)
    and then loses its sub has a build at the end of it, even when what follows isn't a drop
    (Breathe It Out Slowly 2:26 to 2:40: the build runs into a short breakdown, then the fake-out)."""
    out = [Section(s.kind, s.start_bar, s.end_bar) for s in secs]
    for i in range(len(out) - 1, -1, -1):
        s = out[i]
        if s.kind != GROOVE or i + 1 >= len(out) or out[i + 1].kind != BREAKDOWN or s.end_bar - s.start_bar < 12:
            continue
        t = tension[s.start_bar:s.end_bar]
        if _slope(t[-8:]) < RISER_SLOPE:
            continue
        search = t[:-2]
        floor = search.min() + 0.25 * (t.max() - search.min())
        calm = np.flatnonzero(search <= floor)
        start = s.start_bar + (int(calm[-1]) if len(calm) else 0) + 1
        if s.end_bar - start >= PHRASE and t[-4:].mean() - floor >= RISER_RISE:
            out.insert(i + 1, Section(BUILD, start, s.end_bar))
            s.end_bar = start
    return out


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


def _label_sections(m: dict[str, np.ndarray]) -> list[Section]:
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
    # trap: the hook can be as loud as the drop (Love Sosa). The real drops are the ones
    # that land after a big build, a long stretch with the sub cut out. A section that only
    # follows a short dip, isn't louder than those drops and is much busier (hi-hat rolls
    # over the hook vs a half-time 808 drop) is the hook, i.e. groove.
    big = [i for i in range(1, len(pieces)) if kinds[i] == DROP and not pieces[i - 1][2]
           and pieces[i - 1][1] - pieces[i - 1][0] >= 6]
    hooks = []
    if big:
        ref_pw = max(power[pieces[i][0]:pieces[i][1]].mean() for i in big)
        ref_d = max(m["density"][pieces[i][0]:pieces[i][1]].mean() for i in big)
        for i in range(1, len(pieces)):
            a, b = pieces[i][0], pieces[i][1]
            pa, pb, pf, _ = pieces[i - 1]
            if (kinds[i] == DROP and i not in big and not pf and pb - pa <= 2
                    and power[a:b].mean() <= ref_pw and m["density"][a:b].mean() >= ref_d + 0.25):
                kinds[i] = GROOVE
                hooks.append((a, b))
    # melodic dubstep: a short "drop" that leads straight into a build, and then a drop that is clearly
    # brighter, was a fake-out (Seven Lions, Breathe It Out Slowly 2:46 to 2:59: the loudest thing so far,
    # right after a build, 8 bars long, then build 2 and the real drop at 3:19). It is groove, and the
    # bars after it up to the real drop are the build. Real drops run 16+ bars (Chicken Soup's are
    # exactly 16), so only a drop under FAKEOUT_MAX bars can be one.
    fake_end: dict[int, int] = {}   # real-drop piece index -> bar where its build starts
    for i in range(len(pieces)):
        if kinds[i] != DROP or pieces[i][1] - pieces[i][0] >= FAKEOUT_MAX:
            continue
        j = next((k for k in range(i + 1, len(pieces)) if kinds[k] == DROP), None)
        if j is None:
            continue
        a, b = pieces[i][0], pieces[i][1]
        c = pieces[j][0]
        if (c - b <= FAKEOUT_GAP and pieces[j][1] - c >= 12
                and m["tension"][c:pieces[j][1]].mean() >= m["tension"][a:b].mean() + FAKEOUT_BRIGHTER):
            kinds[i] = GROOVE
            if i + 1 < j and not pieces[i + 1][2] and pieces[i + 1][1] - pieces[i + 1][0] <= 2:
                pieces[i][1] = pieces[i + 1][1]   # the 1-bar gap after it belongs to the groove
                pieces[i + 1][0] = pieces[i + 1][1]
            fake_end[j] = pieces[i][1]
    # a 1-2 bar hole between two alike drop halves (a vocal cut) is part of the drop. If the
    # part before it was still climbing (a snare roll, busier than what follows), that was
    # the build and the hole is the silence right before the real drop: leave it.
    for i in range(1, len(pieces) - 1):
        if kinds[i] == BREAKDOWN and kinds[i - 1] == DROP and kinds[i + 1] == DROP and pieces[i][1] - pieces[i][0] <= 2:
            (a0, a1), (c0, c1) = pieces[i - 1][:2], pieces[i + 1][:2]
            if (_slope(tension[a0:a1]) <= 0.015 and abs(power[a0:a1].mean() - power[c0:c1].mean()) < 0.1
                    and abs(m["density"][a0:a1].mean() - m["density"][c0:c1].mean()) < 0.2):
                kinds[i] = DROP
    # the kick slamming back in after the sub was cut is the drop, even while the sub is
    # still swelling back up for that first bar
    for i in range(1, len(pieces)):
        a = pieces[i][0]
        if (kinds[i] == DROP and not pieces[i - 1][2] and a - pieces[i - 1][0] >= 5
                and m["kick"][a - 1] >= 1.2 and low[a - 1] >= 0.5 and (low[a - 4:a - 1] < 0.2).all()):
            pieces[i][0] -= 1
            pieces[i - 1][1] -= 1
    # a short bass dip that keeps the kick going and leads back into the groove is groove
    for i in range(len(pieces) - 1):
        a, b = pieces[i][0], pieces[i][1]
        if (kinds[i] == BREAKDOWN and kinds[i + 1] == GROOVE and i > 0 and 0 < b - a <= 8 and m["kick"][a:b].mean() >= 0.6
                and not _deep_dip(low[a:b], p90)
                and not (pieces[i - 1][2] and _top_collapse(m["tension"][a:b], m["tension"][pieces[i - 1][0]:pieces[i - 1][1]]))):
            kinds[i] = GROOVE

    out: list[Section] = []
    for i, ((a, b, f, _), kind) in enumerate(zip(pieces, kinds)):
        if a >= b:
            continue
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
        if i + 1 in fake_end:
            start = max(lo, min(start, fake_end[i + 1]))   # after a fake-out, everything up to the real drop is the build
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
    if out and out[0].kind == BREAKDOWN and out[0].end_bar - out[0].start_bar > LONG_INTRO:
        # an intro is a few bars of set-up. If the kick starts early and then runs for 20+ bars with the
        # sub well under the first groove, the melody/vocal part is a breakdown (Seven Lions, Breathe It
        # Out Slowly 0:06 to 0:56). Intros that are silent or sub-less and short stay intros.
        e = out[0].end_bar
        run = np.flatnonzero(m["kick"][:e] >= 1.0)
        s0 = int(run[0]) if len(run) else None
        if s0 is not None and 2 <= s0 <= 8 and m["kick"][s0:e].mean() >= 1.0:
            out[0].start_bar = s0
            out.insert(0, Section(INTRO, 0, s0))
    if out and out[0].kind in (GROOVE, BREAKDOWN):
        out[0].kind = INTRO

    merged: list[Section] = []
    for s in out:
        if merged and merged[-1].kind == s.kind:
            merged[-1].end_bar = s.end_bar
        else:
            merged.append(s)
    merged = _riser_builds(merged, m, tension)
    secs = _bright_drops(merged, m)
    grid = _phrase_grid(secs)
    if grid is None:
        return secs
    return _snap_to_grid(_hook_builds(secs, hooks, power, grid), grid)


# --- build stage 2 -------------------------------------------------------------------
# Trap builds come in two stages: a build, then a more intense 2nd build. The section
# boundaries never change; this only marks the bar where stage 2 starts. Two cues, both
# learned from real tracks (bar features in tests/real_tracks):
#  * the sub CUTS OUT: two bars with the sub (low) under STAGE2_CUT and well below the
#    build's own earlier bars (Love Sosa 0:55 and 2:37: low 0.43 -> 0.09, 0.03, 0.03...).
#    A 1-bar cut is the pause before the drop (Not Even Love 1:14, 2:28), not stage 2.
#  * the sub DIPS for 2+ bars (thinned out, still ~0.6), then the full kick and sub come
#    back with a jump in loudness (Chicken Soup 0:36 and 1:36: low 0.61 -> 0.92, kick 1.5,
#    energy +0.22).
# Stage 2 needs at least STAGE2_MIN_BARS of build before it, so short builds have none.
STAGE2_MIN_BARS = 4
STAGE2_CUT = 0.35        # sub level of a real cut
STAGE2_DROP = 0.3        # ...and how far under the build's earlier bars it has to be
STAGE2_RETURN = 0.25     # dip -> comeback: sub rises by this much
STAGE2_KICK = 1.2        # ...with the kick pinned
STAGE2_LOUDER = 0.15     # ...and the whole mix this much louder


def stage2_bar(m: dict[str, np.ndarray], a: int, b: int) -> int | None:
    """First bar of a build [a, b) where stage 2 starts, or None."""
    low, kick, energy = m["low"], m["kick"], m["energy"]
    for x in range(a + STAGE2_MIN_BARS, b):
        before = low[a:x]
        if (x + 1 < b and low[x] < STAGE2_CUT and low[x + 1] < STAGE2_CUT
                and low[x] < before.mean() - STAGE2_DROP):
            return x
        # a 2-bar dip, then the comeback
        if (before.max() - max(low[x - 2], low[x - 1]) >= STAGE2_DROP
                and low[x] >= max(low[x - 2], low[x - 1]) + STAGE2_RETURN
                and kick[x] >= STAGE2_KICK and energy[x] >= energy[x - 1] + STAGE2_LOUDER):
            return x
    return None


def label_sections(m: dict[str, np.ndarray]) -> list[Section]:
    """Label the track (see _label_sections), then mark where each build's 2nd stage starts."""
    secs = _label_sections(m)
    for s in secs:
        if s.kind == BUILD:
            s.stage2_bar = stage2_bar(m, s.start_bar, s.end_bar)
    return secs


def _phrase_grid(secs: list[Section]):
    """Which bar (mod 4) phrases start on, taken from the drops: they're the most
    reliable boundaries, and EDM phrases are 4 bars. Only when every drop agrees: if they
    don't, at least one of them is off and the grid can't be trusted."""
    starts = {s.start_bar % PHRASE for s in secs if s.kind == DROP}
    return starts.pop() if len(starts) == 1 else None


def _hook_builds(secs: list[Section], hooks, power: np.ndarray, grid: int) -> list[Section]:
    """Trap: once the hook ends, everything up to the drop is the build (Love Sosa: hook,
    then a kick-pinned snare climb, then the sub cuts, then the drop). The build starts on
    the phrase line where the hook's power falls away."""
    if not hooks:
        return secs
    labels = []
    for s in secs:
        labels += [s.kind] * (s.end_bar - s.start_bar)
    changed = False
    for ha, hb in hooks:
        drop = next((s for s in secs if s.kind == DROP and s.start_bar >= hb), None)
        if drop is None or DROP in labels[hb:drop.start_bar]:
            continue
        hook_pw = power[ha:hb].mean()
        # first phrase line after which no bar is as loud as the hook any more
        lines = [c for c in range(ha + 2, min(hb + 3, drop.start_bar)) if (c - grid) % PHRASE == 0
                 and power[c:c + 4].max() < hook_pw - 0.05]
        if not lines or drop.start_bar - lines[0] > 28:
            continue
        start = lines[0]
        if any(power[c:c + 4].mean() >= hook_pw - 0.1 for c in range(start, drop.start_bar - 3)):
            continue  # something as big as the hook in between: not one long build
        for i in range(start, drop.start_bar):
            labels[i] = BUILD
        changed = True
    if not changed:
        return secs
    return _from_labels(labels)


def _snap_to_grid(secs: list[Section], grid: int) -> list[Section]:
    """A boundary one bar off the drops' phrase grid moves onto it (a sub that fades a bar
    early, a 1-bar pause before the groove). Drop starts stay put."""
    out = [Section(s.kind, s.start_bar, s.end_bar) for s in secs]
    for i in range(1, len(out)):
        x = out[i].start_bar
        if out[i].kind == DROP or (x - grid) % PHRASE not in (1, PHRASE - 1):
            continue
        y = x + 1 if (x - grid) % PHRASE == PHRASE - 1 else x - 1
        if out[i - 1].start_bar < y < out[i].end_bar:
            out[i - 1].end_bar = out[i].start_bar = y
    return out


def _from_labels(labels: list[str]) -> list[Section]:
    out: list[Section] = []
    for i, lab in enumerate(labels):
        if out and out[-1].kind == lab:
            out[-1].end_bar = i + 1
        else:
            out.append(Section(lab, i, i + 1))
    return out


def _bright_drops(secs: list[Section], m: dict[str, np.ndarray]) -> list[Section]:
    """Drops that hit with the kick and the growl instead of the sub.

    Trap / bass-house drops are often a gritty mid-range bass with LESS sub than the
    hook-y groove before them (Chicken Soup: the groove has the most sub in the track).
    The sub rule misses those. What still gives them away: the kick is pinned for the
    whole stretch and it's much brighter / noisier than the groove. So an 8+ bar stretch
    like that, which isn't already a drop, becomes one, and the build runs back to the
    short gap that ended the groove before it.
    """
    grooves = [s for s in secs if s.kind == GROOVE]
    if not grooves:
        return secs
    g = grooves[0]
    gt = m["tension"][g.start_bar:g.end_bar].mean()
    gl = m["low"][g.start_bar:g.end_bar].mean()
    n = len(m["kick"])
    flags = (m["kick"] >= 1.4) & (m["tension"] >= gt + 0.08)
    thr = np.percentile(m["low"], 90) - 0.15
    labels = []
    for s in secs:
        labels += [s.kind] * (s.end_bar - s.start_bar)
    changed = False
    # bridge gaps of <= 2 bars (a wobble that lets go for a bar mid-drop)
    stretches = []
    for i in np.flatnonzero(flags):
        if stretches and i - stretches[-1][1] <= 2:
            stretches[-1][1] = i + 1
        else:
            stretches.append([int(i), int(i) + 1])
    for a, b in stretches:
        # a drop starts solid: 3 slammed bars in a row (kick hits inside a snare roll don't count)
        while a + 3 <= b and not flags[a:a + 3].all():
            a += 1
        # a 1-2 bar dip at the very end that isn't anything else yet belongs to the drop
        k = 0
        while b + k < n and k < 2 and labels[b + k] == BREAKDOWN:
            k += 1
        if k and (b + k >= n or labels[b + k] != BREAKDOWN):
            b += k
        if b - a < 8 or a < g.end_bar or DROP in labels[a:min(n, b + 2)]:
            continue  # too short, still the intro, or the run-up to a drop we already have
        if m["low"][a:b].mean() >= gl:
            continue  # as much sub as the groove: the sub rule already had its say
        if _slope(m["tension"][a:b] + 0.5 * m["density"][a:b]) > 0.015:
            continue  # still climbing (snare roll speeding up): that's a kick build, not a drop
        # the pause at the peak of the build: the bar where the sub JUST cut out after a heavy
        # build bar. The drop hits right after it. If the sub was already out the bar before,
        # that was the pause and this bar is the drop (some drops open on a sub-less bar).
        if m["low"][a] < thr and 0 < a and m["low"][a - 1] >= thr and a + 1 < b and m["low"][a + 1] >= thr:
            a += 1
        # build: back to the last energy gap (a 1-bar dip) in the 12 bars before the drop
        lo = max(0, a - 12)
        gaps = [i for i in range(lo, a - 2) if m["energy"][i] < 0.5]
        start = gaps[-1] + 1 if gaps else max(0, a - 8)
        start = max(start, max((i + 1 for i in range(start) if labels[i] == DROP), default=0))
        # whatever old build/drop guess the new build/drop cuts through becomes groove
        for s in secs:
            if s.kind in (BUILD, DROP) and s.start_bar < b and s.end_bar > start:
                for i in range(s.start_bar, s.end_bar):
                    labels[i] = GROOVE
        for i in range(start, a):
            labels[i] = BUILD
        for i in range(a, b):
            labels[i] = DROP
        changed = True
    if not changed:
        return secs
    return _from_labels(labels)
