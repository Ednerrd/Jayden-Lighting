"""Causal bar-by-bar section machine for live mode.

Offline labelling (sections.py) sees the whole track: it can normalise against the
track's own p90, look ahead to see whether a dip led into a drop, and snap to a phrase
grid. Live can't. This machine sees ONE bar at a time (the six bar features of the bar
that just ended) and only the past, so every rule is the causal twin of an offline
lesson (see HANDOFF "Section rules"):

* levels are measured against RUNNING references (the top of the sub so far), never p90
* FULL = sub within 0.15 of the running top; EMPTY = sub cut (0.42+ under the top)
* DROP is instant: the sub / kick slams back after a real cut. Everything else needs the
  current section to have held for 2 bars (no flicker)
* a build is called as soon as the evidence shows (a sub dip with tension climbing, or the
  sub cut right after dipping bars). It can't be called any earlier than that
* a 1-2 bar hole inside a drop is a vocal cut, not a breakdown

Pure python lists and floats on purpose: web/engine.js has a line-for-line twin
(LiveSections) and tests check that both give identical labels.
"""

from __future__ import annotations

from ..state import BREAKDOWN, BUILD, DROP, GROOVE, INTRO
from .sections import Section

FULL_WITHIN = 0.15      # full = sub within this of the running top
EMPTY_BELOW = 0.42      # empty = sub this far under the running top (the cut)
FLOOR = 0.35            # ...and never "full" below this, whatever the running top says
TENSION_UP = 0.10       # tension this far over the groove's = a build is climbing
HOLD = 2                # bars a section holds before it may change (entering DROP is exempt)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


class LiveSections:
    def __init__(self):
        self.kind = INTRO
        self.run = 0                 # bars the current section has held
        self.rows: list[dict] = []
        self.kinds: list[str] = []
        self.full: list[bool] = []
        self.power: list[float] = []
        self.low_ref = 0.0           # running top of the sub (2-bar minimum, so one spike can't set it)
        self.pw_ref = 0.0            # running top of the overall power (same trick)
        self.g_te = None             # tension of the groove (full bars outside drops)
        self.g_n = 0                 # how many bars it has learned from
        self.g_pw = None             # power of the groove
        self.empty_run = 0           # consecutive empty bars up to and including the last one
        self.full_run = 0            # consecutive full bars
        self.dip_run = 0             # consecutive non-full bars
        self.build_cut = 0           # longest empty run since this build began
        self._cur_cut = 0
        self.drop_ref = None         # [low, energy, kick] sums over the first bars of the drop
        self.drop_n = 0
        self.sag_run = 0
        self.slack_run = 0
        self.since_drop = 99         # bars since the last drop ended
        self.k_te_min = 9.0          # lowest tension since the breakdown began

    # ------------------------------------------------------------------ helpers
    def dips4(self) -> int:
        """Non-full bars among the last 4 (a build that really dips the sub)."""
        return sum(1 for f in self.full[-4:] if not f)

    def _te_up(self, te):
        return self.g_n >= 8 and self.g_te is not None and te >= self.g_te + TENSION_UP

    def _enter(self, kind):
        if kind != self.kind:
            self.kind = kind
            if kind == BREAKDOWN:
                self.k_te_min = min(r["tension"] for r in self.rows[-3:])   # the bars that sent us here count
            if kind == BUILD:
                self.build_cut = self._cur_cut
            if kind == DROP:
                self.drop_ref, self.drop_n = [0.0, 0.0, 0.0], 0
                self.sag_run = self.slack_run = 0

    def push(self, row: dict, slam: bool = False) -> str:
        """Feed the features of the bar that just ended. Returns the section it belongs to.

        slam: the frame-level detector heard the sub slam back in at the start of this bar (a
        low end that hadn't been heard for 4 bars). That's strong evidence the bar is a drop.
        """
        low, en, hi = row["low"], row["energy"], row["high"]
        te, kick = row["tension"], row["kick"]
        p = en + 0.5 * low + 0.5 * hi
        i = len(self.rows)
        prev_low = self.rows[-1]["low"] if i else low
        prev_p = self.power[-1] if i else p
        self.low_ref = max(min(low, prev_low), self.low_ref - 0.001)
        self.pw_ref = max(min(p, prev_p), self.pw_ref - 0.003)
        full = low >= max(self.low_ref - FULL_WITHIN, FLOOR)
        empty = self.low_ref >= 0.5 and low < self.low_ref - EMPTY_BELOW

        # the cut that came before this bar: how long, and what the bar before it was like
        er_prev = self.empty_run
        dips4 = self.dips4()
        j = i - 1 - er_prev
        build_like = j >= 0 and (self.kinds[j] == BUILD or not self.full[j])
        heavy_before = j >= 0 and self.full[j]       # the pause cut the sub out of a full bar
        self.empty_run = self.empty_run + 1 if empty else 0
        self._cur_cut = self.empty_run
        self.full_run = self.full_run + 1 if full else 0
        self.dip_run = 0 if full else self.dip_run + 1
        if self.kind == BUILD:
            self.build_cut = max(self.build_cut, self.empty_run)

        self.rows.append(row)
        self.power.append(p)
        self.full.append(full)
        was = self.kind

        # ---- DROP: instant, from anything but a drop ----
        if self.kind != DROP and i >= 1:
            cut = er_prev >= 1 and (build_like or er_prev >= 3)
            hit = False
            if cut and full and self.g_te is not None and te >= self.g_te - 0.1:
                hit = True       # the sub is back at full after a real cut (Not Even Love 1:15, 2:29).
                #                  The soft "boom" bar after a gap has the tension dropped away (Chicken Soup 1:33)
            elif cut and low >= self.low_ref - 0.3 and kick >= 1.2 and p >= 0.82 * self.pw_ref:
                hit = True       # kick slams back, sub still swelling in (Love Sosa 1:01); a quieter
                #                  comeback is the 2nd build stage, not the drop (Love Sosa 2:37)
            elif (1 <= er_prev <= 2 and build_like and heavy_before and kick >= 1.4
                  and self.g_te is not None and te >= self.g_te + 0.08):
                hit = True       # bright drop: kick pinned right after the pause, sub still out
                #                  (Chicken Soup 0:43). A drop can open on a sub-less bar
            elif (self.kind == BUILD and self.build_cut >= 3 and full and not self.full[i - 1]
                  and p >= 0.9 * self.pw_ref):
                hit = True       # build with a real cut in it, then the sub is full again (Love Sosa 2:51)
            elif (self.kind in (BUILD, BREAKDOWN) and self.run >= 2 and full and low >= prev_low + 0.15
                  and p >= _mean(self.power[-3:-1]) + 0.3):
                hit = True       # an established build / breakdown, then the sub and the whole mix jump up together
                #                  (a bar-old build that returns at groove loudness is not a drop: Chicken Soup 0:36)
            elif (slam and (self.kind in (BUILD, BREAKDOWN) or dips4 >= 2 or er_prev >= 1) and low >= self.low_ref - 0.3
                  and p >= 0.75 * self.pw_ref):
                hit = True       # the sub slammed back in after dipping bars, as loud as the track gets
            if hit:
                self._enter(DROP)

        self.k_te_min = min(self.k_te_min, te)
        if self.kind in (INTRO, GROOVE) and self.g_pw is not None and p >= self.g_pw + 0.35:
            self.g_te, self.g_pw, self.g_n = None, None, 0   # a much bigger groove arrived (Love Sosa 0:26): learn it afresh
        if self.kind == DROP:
            self._in_drop(row, full)
        else:
            self._others(row)
        # groove baselines learn from full bars that aren't drops
        if full and self.kind in (INTRO, GROOVE):
            self.g_te = te if self.g_te is None else 0.75 * self.g_te + 0.25 * te
            self.g_pw = p if self.g_pw is None else 0.75 * self.g_pw + 0.25 * p
            self.g_n += 1
        self.run = self.run + 1 if self.kind == was else 1
        self.since_drop = 0 if self.kind == DROP else self.since_drop + 1
        self.kinds.append(self.kind)
        return self.kind

    # ------------------------------------------------------------------ inside a drop
    def _in_drop(self, row, full):
        if self.drop_n < 4:
            self.drop_ref[0] += row["low"]
            self.drop_ref[1] += row["energy"]
            self.drop_ref[2] += row["tension"]
            self.drop_n += 1
        n = self.drop_n
        d_en, d_te = self.drop_ref[1] / n, self.drop_ref[2] / n
        # sag: the floor and the loudness fall clear of where the drop started (Not Even Love 1:45)
        sag = row["energy"] <= d_en - 0.15
        self.sag_run = self.sag_run + 1 if sag and self.empty_run == 0 else 0
        # the top end falling away (tension collapses to the groove's) ends the drop (Chicken Soup 1:11)
        slack = row["tension"] <= d_te - 0.2 and self.empty_run == 0
        self.slack_run = self.slack_run + 1 if slack else 0
        if not (self.kinds and self.kinds[-1] == DROP and self.run >= HOLD):
            return
        if self.empty_run >= 3:
            # the sub is really gone (a 1-2 bar hole is just the vocal cut). Tension up = the build
            tes = [r["tension"] for r in self.rows[-3:]]
            self._enter(BUILD if self._te_up(_mean(tes)) else BREAKDOWN)
        elif self.sag_run >= 2 or self.slack_run >= 2:
            self._enter(GROOVE)

    # ------------------------------------------------------------------ everything else
    def _others(self, row):
        if self.run < HOLD and self.kind != INTRO:
            return
        k = self.kind
        # the bars right after a drop are the groove settling back in: the sub still pumps and
        # the tension is still coming down, which looks like a dip but isn't a build (Chicken Soup 1:15)
        settling = self.since_drop < 8
        if k == INTRO:
            if self.full_run >= 3:           # (an intro is at least 2 bars: no 1-bar sections)
                self._enter(GROOVE)
        elif k == GROOVE:
            n = len(self.rows)
            tes2 = _mean([r["tension"] for r in self.rows[-2:]])
            g = self.g_te
            if (self.empty_run == 1 and n >= 2 and not self.full[n - 2] and not settling
                    and g is not None and tes2 >= g - 0.05):
                self._enter(BUILD)           # the sub cut right after a dipping bar, tension not falling (Not Even Love 1:14)
            elif self.empty_run >= 3 or (self.empty_run == 2 and self._te_up(tes2)):
                self._enter(BUILD if self._te_up(_mean([r["tension"] for r in self.rows[-3:]])) else BREAKDOWN)
            elif self.dip_run >= 2 and self._te_up(tes2) and not settling:
                self._enter(BUILD)           # sub dipping while the tension climbs (Chicken Soup 0:32)
            elif self.dip_run >= 2 and g is not None and tes2 <= g - 0.15:
                self._enter(BREAKDOWN)       # sub dipping AND the top end falling away (a groove dip keeps its tension: Not Even Love 1:45)
            elif (n >= 8 and self.g_n >= 8 and self.full_run >= 2 and g is not None and tes2 >= g + 0.15 and not settling
                  and tes2 >= _mean([r["tension"] for r in self.rows[-8:-4]]) + 0.1):
                self._enter(BUILD)           # kick-through build: the sub never leaves, the tension just climbs
        elif k == BREAKDOWN:
            tes2 = _mean([r["tension"] for r in self.rows[-2:]])
            if self._te_up(tes2) or tes2 >= self.k_te_min + 0.2:
                self._enter(BUILD)           # the tension climbs back out of the breakdown
            elif self.full_run >= 3:
                self._enter(GROOVE)
        elif k == BUILD:
            te4 = _mean([r["tension"] for r in self.rows[-4:]])
            quiet = self.g_te is not None and te4 < self.g_te + 0.06
            if self.run >= 16 and self.full_run >= 8 or (self.run >= 8 and self.full_run >= 8 and quiet):
                self._enter(GROOVE)          # a build that never dropped


def simulate_live_bars(m: dict) -> list[Section]:
    """Push a track's per-bar features through LiveSections one bar at a time and collapse
    the per-bar kinds into Sections (same dataclass as the offline labeller)."""
    n = len(m["low"])
    ls = LiveSections()
    kinds = []
    for i in range(n):
        kinds.append(ls.push({k: float(m[k][i]) for k in ("kick", "low", "energy", "high", "tension", "density")}))
    out: list[Section] = []
    for i, k in enumerate(kinds):
        if out and out[-1].kind == k:
            out[-1].end_bar = i + 1
        else:
            out.append(Section(k, i, i + 1))
    return out
