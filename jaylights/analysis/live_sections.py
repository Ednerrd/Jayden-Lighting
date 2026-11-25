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
from .sections import STAGE2_DROP, STAGE2_KICK, STAGE2_LOUDER, STAGE2_RETURN, Section

FULL_WITHIN = 0.15      # full = sub within this of the running top
EMPTY_BELOW = 0.42      # empty = sub this far under the running top (the cut)
FLOOR = 0.35            # ...and never "full" below this, whatever the running top says
TENSION_UP = 0.10       # tension this far over the groove's = a build is climbing
STAGE2_HARD_CUT = 0.2   # a 1-bar cut this deep is already the 2nd build stage (Love Sosa), a shallower one is the pause
BACK_MIN = 4            # ...but a dip of 1-3 bars is the pause before a drop, not a fade
BACK_DIP = 8            # a dip this short (bars) that returns at groove loudness is the groove again, not a drop
D_RELAX = 0.15          # ...after it had relaxed this far under the drop's own tension
D_CLIMB = 0.12          # inside a drop, tension this far over its own low point (after it relaxed) = a build
FALSE_START = 3         # a "drop" that cuts its sub again after this many bars was the build's climax, not the drop
SILENT = 0.1            # a bar quieter than this is the silence after the track, not a build
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
        self._back = False
        self.false_start = False     # the last "drop" was cut after 3 bars: this build is about to drop for real
        self.d_te_min = 9.0          # lowest 2-bar tension inside the current drop (after its first 4 bars)
        self.stage2 = False          # this build has reached its 2nd, more intense stage (never changes a label)

    # ------------------------------------------------------------------ helpers
    def dips4(self) -> int:
        """Non-full bars among the last 4 (a build that really dips the sub)."""
        return sum(1 for f in self.full[-4:] if not f)

    def _te_up(self, te):
        return self.g_n >= 8 and self.g_te is not None and te >= self.g_te + TENSION_UP

    def _enter(self, kind):
        if kind != self.kind:
            self.kind = kind
            self.stage2 = False
            self.false_start = False
            if kind == BREAKDOWN:
                self.k_te_min = min(r["tension"] for r in self.rows[-3:])   # the bars that sent us here count
            if kind == BUILD:
                self.build_cut = self._cur_cut
            if kind == DROP:
                self.drop_ref, self.drop_n = [0.0, 0.0, 0.0], 0
                self.sag_run = self.slack_run = 0
                self.d_te_min = 9.0

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
        dip_prev = self.dip_run
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
        # a short dip that comes back at groove loudness (closer to the groove than to the biggest bars so
        # far) is the groove again, even right after a build-like fade (Breathe It Out Slowly 2:01, the sub
        # returns after an 8-bar fade at the first groove's loudness). Offline twin: the "short dip" rule.
        back = (BACK_MIN <= dip_prev <= BACK_DIP and full and self.g_pw is not None and p < (self.g_pw + self.pw_ref) / 2)

        # ---- DROP: instant, from anything but a drop ----
        if self.kind != DROP and i >= 1:
            cut = er_prev >= 1 and (build_like or er_prev >= 3)
            hit = False
            if not back and cut and full and self.g_te is not None and te >= self.g_te - 0.1:
                hit = True       # the sub is back at full after a real cut (Not Even Love 1:15, 2:29).
                #                  The soft "boom" bar after a gap has the tension dropped away (Chicken Soup 1:33)
            elif not back and cut and low >= self.low_ref - 0.3 and kick >= 1.2 and p >= 0.82 * self.pw_ref:
                hit = True       # kick slams back, sub still swelling in (Love Sosa 1:01); a quieter
                #                  comeback is the 2nd build stage, not the drop (Love Sosa 2:37)
            elif (1 <= er_prev <= 2 and build_like and heavy_before and kick >= 1.4
                  and self.g_te is not None and te >= self.g_te + 0.08):
                hit = True       # bright drop: kick pinned right after the pause, sub still out
                #                  (Chicken Soup 0:43). A drop can open on a sub-less bar
            elif (not back and self.kind == BUILD and self.build_cut >= 3 and full and not self.full[i - 1]
                  and p >= 0.9 * self.pw_ref):
                hit = True       # build with a real cut in it, then the sub is full again (Love Sosa 2:51)
            elif self.kind == BUILD and self.false_start and full and not self.full[i - 1] and p >= 0.9 * self.pw_ref:
                hit = True       # the pause after a false start, then the sub slams back: now it is the drop (Breathe 3:19)
            elif (not back and self.kind in (BUILD, BREAKDOWN) and self.run >= 2 and full and low >= prev_low + 0.15
                  and p >= _mean(self.power[-3:-1]) + 0.3):
                hit = True       # an established build / breakdown, then the sub and the whole mix jump up together
                #                  (a bar-old build that returns at groove loudness is not a drop: Chicken Soup 0:36)
            elif (slam and (self.kind in (BUILD, BREAKDOWN) or dips4 >= 2 or er_prev >= 1) and low >= self.low_ref - 0.3
                  and p >= 0.75 * self.pw_ref):
                hit = True       # the sub slammed back in after dipping bars, as loud as the track gets
            if hit:
                self._enter(DROP)
        self._back = back

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
        if self.kind == BUILD and was == BUILD and not self.stage2:
            self.stage2 = self._stage2_cue(low, kick, en)
        self.run = self.run + 1 if self.kind == was else 1
        self.since_drop = 0 if self.kind == DROP else self.since_drop + 1
        self.kinds.append(self.kind)
        return self.kind

    def _stage2_cue(self, low, kick, en) -> bool:
        """Causal twin of sections.stage2_bar: this bar starts the build's 2nd stage when the sub
        cuts out (Love Sosa 0:55, 2:37) or comes back full after a 2+ bar dip (Chicken Soup 0:36).
        A single empty bar that isn't a near-total cut is the pause before the drop (Not Even
        Love), so that waits for a 2nd empty bar."""
        if self.empty_run >= 2 or (self.empty_run == 1 and low < STAGE2_HARD_CUT):
            return True
        if len(self.rows) >= 3 and self.empty_run == 0:
            a, b = self.rows[-3], self.rows[-2]
            dip = max(a["low"], b["low"])
            return (self.low_ref - dip >= STAGE2_DROP and low >= dip + STAGE2_RETURN
                    and kick >= STAGE2_KICK and en >= b["energy"] + STAGE2_LOUDER)
        return False

    # ------------------------------------------------------------------ inside a drop
    def _in_drop(self, row, full):
        # the pause on the last bar of a phrase, 3 bars into a "drop": the sub ducks out well under where
        # this drop's own sub was, and it is a drop that only just began (Breathe 3:12 to 3:18 is the
        # build's climax, the sub cuts again on the 4th bar and the real drop lands at 3:19)
        if (self.drop_n == FALSE_START and self.run == FALSE_START and row["kick"] >= 1.2
                and row["low"] < self.drop_ref[0] / self.drop_n - 0.2):
            self._enter(BUILD)
            self.false_start = True
            return
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
        if self.run >= 4 and self.empty_run == 0:
            te2 = _mean([r["tension"] for r in self.rows[-2:]])
            self.d_te_min = min(self.d_te_min, te2)
            # kick-through build inside a long drop: the tension relaxed, then climbs again
            # (Breathe 2:59 to 3:08, the build after the fake-out drop)
            if (self.run >= 8 and self.d_te_min <= d_te - D_RELAX and te2 >= self.d_te_min + D_CLIMB
                    and self.kinds[-1] == DROP):
                self._enter(BUILD)
                return
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
            if (self._te_up(tes2) or tes2 >= self.k_te_min + 0.2) and row["energy"] >= SILENT:
                self._enter(BUILD)           # the tension climbs back out of the breakdown (not out of the silence after the end)
            elif self.full_run >= 3:
                self._enter(GROOVE)
        elif k == BUILD:
            te4 = _mean([r["tension"] for r in self.rows[-4:]])
            quiet = self.g_te is not None and te4 < self.g_te + 0.06
            if self.run >= 16 and self.full_run >= 8 or (self.run >= 8 and self.full_run >= 8 and quiet):
                self._enter(GROOVE)          # a build that never dropped
            elif self._back and not self.false_start:
                self._enter(GROOVE)          # the sub came back after a short fade at groove loudness


def simulate_live_bars(m: dict) -> list[Section]:
    """Push a track's per-bar features through LiveSections one bar at a time and collapse
    the per-bar kinds into Sections (same dataclass as the offline labeller)."""
    n = len(m["low"])
    ls = LiveSections()
    kinds = []
    stage2 = []
    for i in range(n):
        kinds.append(ls.push({k: float(m[k][i]) for k in ("kick", "low", "energy", "high", "tension", "density")}))
        stage2.append(ls.stage2)
    out: list[Section] = []
    for i, k in enumerate(kinds):
        if out and out[-1].kind == k:
            out[-1].end_bar = i + 1
        else:
            out.append(Section(k, i, i + 1))
        if k == BUILD and stage2[i] and out[-1].stage2_bar is None:
            out[-1].stage2_bar = i
    return out
