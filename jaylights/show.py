"""The show engine: MusicState in, a FixtureState per light out.

Each section has its own look, like a lighting designer would program it:

  intro / groove  pulse on the kick, two-colour alternation that swaps every bar,
                  movers doing slow figure-8s
  breakdown       slow colour waves, low intensity, movers drifting
  build           chase speeds up (1/4 -> 1/8 -> 1/16 -> 1/32) with the snare roll,
                  colour drains to white, movers aim up, strobes ramp in,
                  then BLACKOUT on the last beat before the drop (pre-analyzed mode)
  drop            white impact flash, then one of the DROP_STYLES (pump, chase, split,
                  strobe_hell, fan_blast, rainbow_chase, blackout_stabs, sweep,
                  half_time_heavy). "auto" rotates a genre-suited subset every 8 bars;
                  Controls.drop_style locks one.

Genre flavour changes palettes and how hard it hits.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .fixtures import DIMMER, MOVER, STROBE, WASH, Fixture, FixtureState, hsv, mix, wave
from .state import BREAKDOWN, BUILD, DROP, INTRO, OUTRO, MusicState

WHITE = (1.0, 1.0, 1.0)

# hue sets (degrees). Each palette is a list of colours the looks pick from.
PALETTES = {
    "hot": [[0, 300], [330, 30], [280, 0], [0, 45], [300, 200]],
    "cool": [[220, 270], [190, 240], [260, 300], [170, 210]],
    "acid": [[100, 290], [80, 0], [120, 300], [0, 280]],
    "warm": [[30, 330], [15, 45], [340, 20], [40, 280]],
    "mixed": [[300, 180], [0, 220], [270, 30], [160, 320], [200, 60]],
}

GENRES = {
    #              bpm range        drop palette  calm palette  strobe  snare hits
    "auto":        ((90, 190),     "mixed",      "cool",       1.0,    0.8),
    "house":       ((115, 132),    "warm",       "cool",       0.6,    0.6),
    "progressive": ((118, 132),    "mixed",      "cool",       0.7,    0.6),
    "techno":      ((120, 150),    "cool",       "cool",       0.9,    0.7),
    "trance":      ((125, 145),    "mixed",      "cool",       0.9,    0.7),
    "dubstep":     ((134, 156),    "acid",       "cool",       1.0,    1.0),
    "trap":        ((125, 165),    "hot",        "cool",       1.0,    1.0),
    "dnb":         ((160, 182),    "acid",       "cool",       1.0,    0.9),
}


# Drop styles. The first beat of every drop is always the white impact; after that the
# style takes over. "auto" rotates through a genre-appropriate list every 8 bars (32 beats).
DROP_STYLES = ("pump", "chase", "split", "strobe_hell", "fan_blast", "rainbow_chase",
               "blackout_stabs", "sweep", "half_time_heavy")
DROP_ROTATION = {
    "auto":        ("pump", "chase", "split", "fan_blast", "rainbow_chase", "sweep"),
    "house":       ("pump", "rainbow_chase", "sweep", "chase"),
    "progressive": ("sweep", "rainbow_chase", "pump", "chase"),
    "techno":      ("pump", "strobe_hell", "chase", "blackout_stabs"),
    "trance":      ("sweep", "rainbow_chase", "pump", "fan_blast"),
    "dubstep":     ("half_time_heavy", "strobe_hell", "blackout_stabs", "fan_blast"),
    "trap":        ("half_time_heavy", "blackout_stabs", "strobe_hell", "fan_blast"),
    "dnb":         ("fan_blast", "chase", "strobe_hell", "pump"),
}


@dataclass
class Controls:
    master: float = 1.0
    blackout: bool = False
    strobe_enabled: bool = True
    max_strobe_hz: float = 10.0
    movement: float = 1.0     # 0 = movers stay home
    drop_style: str = "auto"  # "auto" (genre rotation) or one of DROP_STYLES to lock it


class ShowEngine:
    def __init__(self, fixtures: list[Fixture], genre: str = "auto", seed: int = 1):
        self.fixtures = fixtures
        self.genre = genre if genre in GENRES else "auto"
        _, self.drop_pal_name, self.calm_pal_name, self.strobe_amt, self.snare_amt = GENRES[self.genre]
        self.rng = random.Random(seed)
        self.section_index = -1
        self.palette = [hsv(0), hsv(240)]
        self._smooth = [FixtureState() for _ in fixtures]
        self._first = True
        self._style = "pump"
        self._hz_cap = 10.0

    # ------------------------------------------------------------------
    def _pick_palette(self, kind: str):
        name = self.drop_pal_name if kind == DROP else self.calm_pal_name if kind == BREAKDOWN else "mixed"
        hues = self.rng.choice(PALETTES[name])
        base = [hsv(h) for h in hues]
        if kind == BREAKDOWN:
            base = [hsv(h, 0.9, 0.9) for h in hues]
        self.palette = base

    def drop_style_at(self, section_beats: float, setting: str = "auto") -> str:
        if setting in DROP_STYLES:
            return setting
        rot = DROP_ROTATION[self.genre]
        return rot[int(section_beats // 32) % len(rot)]

    def render(self, st: MusicState, ctl: Controls) -> list[FixtureState]:
        if st.section_index != self.section_index:
            self.section_index = st.section_index
            if st.section != BUILD:  # builds keep the previous colours and drain them to white
                self._pick_palette(st.section)
        self._hz_cap = ctl.max_strobe_hz if ctl.strobe_enabled else 2.0   # wash flashing obeys the strobe safety too
        if st.section == DROP:
            self._style = self.drop_style_at(st.section_beats, ctl.drop_style)
        out = [FixtureState(pan=f.pan_center, tilt=f.tilt_center) for f in self.fixtures]
        look = {
            DROP: self._drop, BUILD: self._build, BREAKDOWN: self._breakdown,
        }.get(st.section, self._groove)
        for f, fs in zip(self.fixtures, out):
            look(st, f, fs)
        return self._post(st, ctl, out)

    # ------------------------------------------------------------------
    def _move(self, f: Fixture, fs: FixtureState, dpan: float, dtilt: float):
        fs.pan = f.pan_center + dpan * f.pan_span
        fs.tilt = f.tilt_center + dtilt * f.tilt_span * f.tilt_up

    def _groove(self, st: MusicState, f: Fixture, fs: FixtureState):
        A, B = self.palette[0], self.palette[-1]
        calm = 0.7 if st.section in (INTRO, OUTRO) else 1.0
        swap = st.bar % 2
        if f.kind in (WASH, DIMMER):
            fs.set_color(A if (f.index + swap) % 2 == 0 else B)
            fs.intensity = calm * (0.2 + 0.6 * st.kick + 0.2 * st.energy)
        elif f.kind == MOVER:
            fs.set_color(B if f.index % 2 else A)
            fs.intensity = calm * (0.5 + 0.3 * st.hat)
            ph = f.index * 0.25
            self._move(f, fs, math.sin(2 * math.pi * (st.beat / 16 + ph)) * 0.8,
                       math.sin(2 * math.pi * (st.beat / 8 + ph)) * 0.5)
        elif f.kind == STROBE:
            fs.intensity = 0.35 * st.snare * self.snare_amt if st.energy > 0.6 else 0.0

    def _breakdown(self, st: MusicState, f: Fixture, fs: FixtureState):
        A, B = self.palette[0], self.palette[-1]
        breathe = 0.6 + 0.4 * wave(st.beat / 16)
        level = 0.1 + 0.5 * max(st.mid, st.energy * 0.8)
        if f.kind in (WASH, DIMMER):
            fs.set_color(mix(A, B, wave(f.x * 0.8 + st.beat / 32)))
            fs.intensity = level * breathe
        elif f.kind == MOVER:
            fs.set_color(mix(B, A, wave(st.beat / 32 + f.index * 0.3)))
            fs.intensity = 0.35 * breathe
            self._move(f, fs, math.sin(2 * math.pi * (st.beat / 32 + f.index * 0.37)) * 0.6, 0.6)
        else:
            fs.intensity = 0.0

    def _build(self, st: MusicState, f: Fixture, fs: FixtureState):
        p = st.section_progress
        if st.next_section == DROP and st.beats_to_next is not None:
            if st.beats_to_next <= 1.0:
                fs.intensity = 0.0       # the breath before the drop
                self._move(f, fs, 0.0, 1.0)
                return
        div = 1 if p < 0.25 else 2 if p < 0.5 else 4 if p < 0.75 else 8
        color = mix(self.palette[0], WHITE, p ** 1.5)
        sub = st.beat * div
        gate = 1.0 - (sub % 1.0)            # sharp attack, decays until the next subdivision
        if f.kind in (WASH, DIMMER):
            fs.set_color(color)
            if p < 0.5:
                # chase across the group, faster as the build goes
                lit = int(sub) % max(1, f.group_size)
                on = 1.0 if lit == f.index else 0.15
                fs.intensity = (0.35 + 0.65 * p) * on * (0.4 + 0.6 * gate)
            else:
                fs.intensity = (0.35 + 0.65 * p) * gate
            fs.intensity = max(fs.intensity, st.snare * (0.4 + 0.6 * p))
        elif f.kind == MOVER:
            fs.set_color(color)
            fs.intensity = 0.4 + 0.6 * p
            speed = 1 + 3 * p
            self._move(f, fs, math.sin(2 * math.pi * (st.beat / 8 * speed + f.index * 0.25)) * (1 - p),
                       0.3 + 0.7 * p)
        elif f.kind == STROBE:
            if p > 0.6:
                fs.intensity = (p - 0.6) / 0.4 * self.strobe_amt
                fs.strobe_hz = st.bpm / 60 * div
            else:
                fs.intensity = 0.6 * st.snare * p

    def _drop(self, st: MusicState, f: Fixture, fs: FixtureState):
        sb = st.section_beats
        if sb < 1.0:
            # IMPACT (every style): everything white, full, decaying over the first beat
            hit = 1.0 - 0.5 * sb
            fs.set_color(WHITE)
            fs.intensity = hit if f.kind != STROBE else 1.0 - sb
            self._move(f, fs, 0.0, -0.4)
            return
        getattr(self, "_d_" + self._style)(st, f, fs)

    # ---- drop styles. Each handles every fixture kind, so any rig subset looks good:
    #      washes/dimmers carry the look on their own, movers and strobes add to it.
    def _flash_div(self, st: MusicState, want: int) -> int:
        """Largest flash subdivision <= want that stays under the strobe-rate cap."""
        d = want
        while d > 1 and st.bpm / 60 * d > self._hz_cap:
            d //= 2
        return d

    def _d_pump(self, st, f, fs):
        """whole group hits on the kick, colours swap every beat; movers fan-sweep in mirrored pairs"""
        A, B = self.palette[0], self.palette[-1]
        bi, k = int(st.beat), st.kick
        if f.kind in (WASH, DIMMER):
            fs.set_color(A if (bi + f.index) % 2 == 0 else B)
            fs.intensity = max(0.15 + 0.85 * k, st.snare * self.snare_amt * 0.9)
        elif f.kind == MOVER:
            fs.set_color(B if f.index % 2 else A)
            fs.intensity = 0.5 + 0.5 * k
            mirror = -1 if f.index % 2 else 1
            self._move(f, fs, mirror * math.sin(2 * math.pi * st.beat / 4), -0.5 + 0.8 * k)
        elif f.kind == STROBE:
            fs.intensity = self.strobe_amt * max(st.snare ** 2 * self.snare_amt, (k ** 4) * 0.5)
            fs.set_color(WHITE)

    def _d_chase(self, st, f, fs):
        """1/2-beat bouncing chase across each group; movers circle, one bar per lap"""
        A, B = self.palette[0], self.palette[-1]
        bi, k = int(st.beat), st.kick
        if f.kind in (WASH, DIMMER):
            n = max(1, f.group_size)
            step = int(st.beat * 2) % (2 * n - 2 if n > 2 else n)
            pos = step if step < n else 2 * n - 2 - step
            fs.set_color(A if (bi // 2) % 2 == 0 else B)
            fs.intensity = max(1.0 if pos == f.index else 0.1, 0.5 * k, st.snare * self.snare_amt * 0.9)
        elif f.kind == MOVER:
            fs.set_color(B if f.index % 2 else A)
            fs.intensity = 0.5 + 0.5 * k
            a = 2 * math.pi * (st.beat / 4 + f.index * 0.25)
            self._move(f, fs, math.sin(a), math.cos(a) * 0.8)
        elif f.kind == STROBE:
            fs.intensity = self.strobe_amt if st.beat_in_bar < 0.15 else 0.0   # downbeats only
            fs.set_color(WHITE)

    def _d_split(self, st, f, fs):
        """left vs right, swapping each beat; movers snap into a crossing X"""
        A, B = self.palette[0], self.palette[-1]
        bi, k = int(st.beat), st.kick
        if f.kind in (WASH, DIMMER):
            left = f.x < 0.5
            on = left == (bi % 2 == 0)
            fs.set_color(A if on else B)
            fs.intensity = max((0.3 + 0.7 * k) if on else 0.15 * k, st.snare * self.snare_amt * 0.9)
        elif f.kind == MOVER:
            fs.set_color(B if f.index % 2 else A)
            fs.intensity = 0.5 + 0.5 * k
            self._move(f, fs, (1 if (bi + f.index) % 2 else -1) * 0.9, 0.2)
        elif f.kind == STROBE:
            fs.intensity = self.strobe_amt * max(st.snare ** 2 * self.snare_amt, (k ** 4) * 0.5)
            fs.set_color(WHITE)

    def _d_strobe_hell(self, st, f, fs):
        """strobes + washes flash on every 8th (16ths on the last bar of each 4), colours alternating"""
        A, B = self.palette[0], self.palette[-1]
        k = st.kick
        div = self._flash_div(st, 4 if st.bar % 4 == 3 else 2)
        sub = st.beat * div
        n, on = int(sub), (sub % 1.0) < 0.5
        if f.kind in (WASH, DIMMER):
            fs.set_color(A if n % 2 == 0 else B)
            fs.intensity = 1.0 if on else 0.04 + 0.2 * k
        elif f.kind == MOVER:
            fs.set_color(WHITE if on and n % 4 == 0 else (B if f.index % 2 else A))
            fs.intensity = 1.0 if on else 0.2
            side = 1 if (n + f.index) % 2 else -1
            self._move(f, fs, side * 0.7, 0.3)
        elif f.kind == STROBE:
            fs.set_color(WHITE)
            fs.intensity = max(self.strobe_amt, 0.8) * (0.6 + 0.4 * k)
            fs.strobe_hz = st.bpm / 60 * div

    def _d_fan_blast(self, st, f, fs):
        """movers fan out from centre on each kick then snap back; washes burst outward in full colour"""
        A, B = self.palette[0], self.palette[-1]
        bi, k = int(st.beat), st.kick
        c = A if bi % 2 == 0 else B
        if f.kind in (WASH, DIMMER):
            dist = abs(f.x - 0.5) * 2
            fan = min(1.0, max(0.0, k * 1.6 - dist * 0.9 + 0.25))
            fs.set_color(mix(c, WHITE, 0.6 * st.snare))
            fs.intensity = max(0.08, fan, st.snare * self.snare_amt * 0.9)
        elif f.kind == MOVER:
            fs.set_color(c if f.index % 2 == 0 else (B if bi % 2 == 0 else A))
            fs.intensity = 0.4 + 0.6 * k
            self._move(f, fs, (f.x - 0.5) * 2 * k * 1.1, -0.2 + 0.6 * k)
        elif f.kind == STROBE:
            fs.set_color(WHITE)
            fs.intensity = self.strobe_amt * max(st.snare ** 2 * self.snare_amt, (k ** 6) * 0.6)

    def _d_rainbow_chase(self, st, f, fs):
        """hue rolls across the rig by x position while a comet chases on 1/2 beats"""
        k = st.kick
        hue = f.x * 300 + st.beat * 40
        if f.kind in (WASH, DIMMER):
            n = max(1, f.group_size)
            head = int(st.beat * 2) % n
            d = (head - f.index) % n                       # steps behind the comet head
            tail = max(0.0, 1.0 - d / max(2.0, n / 2))
            fs.set_color(hsv(hue))
            fs.intensity = max(0.12 + 0.88 * tail * (0.65 + 0.35 * k), st.snare * self.snare_amt * 0.8)
        elif f.kind == MOVER:
            fs.set_color(hsv(hue + 120))
            fs.intensity = 0.6 + 0.4 * k
            ph = f.x * 0.5
            self._move(f, fs, math.sin(2 * math.pi * (st.beat / 8 + ph)) * 0.8,
                       0.1 + 0.4 * math.sin(2 * math.pi * (st.beat / 4 + ph)))
        elif f.kind == STROBE:
            fs.set_color(WHITE)
            fs.intensity = self.strobe_amt * (1.0 if st.beat_in_bar < 0.15 else 0.5 * st.snare ** 2)

    def _d_blackout_stabs(self, st, f, fs):
        """everything dark except hard cut-off stabs on every kick and snare"""
        A, B = self.palette[0], self.palette[-1]
        bi, k, sn = int(st.beat), st.kick, st.snare
        v = max(k, sn)
        v = v if v > 0.4 else 0.0
        if f.kind in (WASH, DIMMER):
            fs.set_color(mix(B, WHITE, 0.7) if sn > k else A)
            fs.intensity = v
        elif f.kind == MOVER:
            fs.set_color(WHITE if sn > k else (B if f.index % 2 else A))
            fs.intensity = v
            self._move(f, fs, (1 if (bi + f.index) % 2 else -1) * 0.9, 0.3)
        elif f.kind == STROBE:
            fs.set_color(WHITE)
            fs.intensity = self.strobe_amt if sn > 0.5 else 0.0

    def _d_sweep(self, st, f, fs):
        """smooth and slow: movers sweep wide arcs while washes crossfade colour in a travelling wave"""
        A, B = self.palette[0], self.palette[-1]
        k = st.kick
        if f.kind in (WASH, DIMMER):
            fs.set_color(mix(A, B, wave(f.x * 0.6 + st.beat / 16)))
            fs.intensity = max(0.4 + 0.35 * wave(st.beat / 8 - f.x * 0.5) + 0.25 * k, st.snare * 0.5)
        elif f.kind == MOVER:
            fs.set_color(mix(B, A, wave(st.beat / 16 + f.index * 0.25)))
            fs.intensity = 0.6 + 0.3 * k
            self._move(f, fs, math.sin(2 * math.pi * (st.beat / 16 + f.index * 0.125)) * 0.9,
                       0.1 + 0.5 * math.sin(2 * math.pi * (st.beat / 32 + f.index * 0.2)))
        elif f.kind == STROBE:
            fs.set_color(WHITE)
            fs.intensity = self.strobe_amt * 0.6 * max(st.snare ** 2, k ** 4)

    def _d_half_time_heavy(self, st, f, fs):
        """kick = solid hit, snare = huge white-ish slam with strobes; movers move slowly and snap up on the snare"""
        A, B = self.palette[0], self.palette[-1]
        k, sn = st.kick, st.snare
        if f.kind in (WASH, DIMMER):
            fs.set_color(mix(B, WHITE, 0.5 * sn) if sn > k else A)
            fs.intensity = max(0.12 + 0.5 * k ** 1.5, sn * self.snare_amt)
        elif f.kind == MOVER:
            fs.set_color(B if f.index % 2 else A)
            fs.intensity = 0.4 + 0.6 * max(k, sn)
            self._move(f, fs, math.sin(2 * math.pi * (st.beat / 16 + f.index * 0.25)) * 0.7, 0.2 + 0.7 * sn)
        elif f.kind == STROBE:
            fs.set_color(WHITE)
            fs.intensity = self.strobe_amt * sn ** 2
            fs.strobe_hz = 14.0 if sn > 0.5 else 0.0

    # ------------------------------------------------------------------
    def _post(self, st: MusicState, ctl: Controls, out: list[FixtureState]) -> list[FixtureState]:
        smooth_move = 0.08 if st.section == BREAKDOWN else 0.3
        for f, fs, sm in zip(self.fixtures, out, self._smooth):
            # movement: scale by the movement control, then smooth (saves the motors)
            fs.pan = f.pan_center + (fs.pan - f.pan_center) * ctl.movement
            fs.tilt = f.tilt_center + (fs.tilt - f.tilt_center) * ctl.movement
            if self._first:
                sm.pan, sm.tilt = fs.pan, fs.tilt
            sm.pan += (fs.pan - sm.pan) * smooth_move
            sm.tilt += (fs.tilt - sm.tilt) * smooth_move
            fs.pan = min(1.0, max(0.0, sm.pan))
            fs.tilt = min(1.0, max(0.0, sm.tilt))
            # safety + master
            if not ctl.strobe_enabled:
                fs.strobe_hz = 0.0
                if f.kind == STROBE:
                    fs.intensity *= 0.3
            fs.strobe_hz = min(fs.strobe_hz, ctl.max_strobe_hz)
            fs.intensity = 0.0 if ctl.blackout else min(1.0, max(0.0, fs.intensity)) * ctl.master
        self._first = False
        return out
