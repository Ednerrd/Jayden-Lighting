"""The show engine: MusicState in, a FixtureState per light out.

Each section has its own look, like a lighting designer would program it:

  intro / groove  pulse on the kick, two-colour alternation that swaps every bar,
                  movers doing slow figure-8s
  breakdown       slow colour waves, low intensity, movers drifting
  build           chase speeds up (1/4 -> 1/8 -> 1/16 -> 1/32) with the snare roll,
                  colour drains to white, movers aim up, strobes ramp in,
                  then BLACKOUT on the last beat before the drop (pre-analyzed mode)
  drop            white impact flash, then hard kick hits, fast colour swaps, big
                  mover sweeps, strobes on the snares. Rotates through 3 variations
                  every 8 bars so it doesn't get stale.

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


@dataclass
class Controls:
    master: float = 1.0
    blackout: bool = False
    strobe_enabled: bool = True
    max_strobe_hz: float = 10.0
    movement: float = 1.0     # 0 = movers stay home


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

    # ------------------------------------------------------------------
    def _pick_palette(self, kind: str):
        name = self.drop_pal_name if kind == DROP else self.calm_pal_name if kind == BREAKDOWN else "mixed"
        hues = self.rng.choice(PALETTES[name])
        base = [hsv(h) for h in hues]
        if kind == BREAKDOWN:
            base = [hsv(h, 0.9, 0.9) for h in hues]
        self.palette = base

    def render(self, st: MusicState, ctl: Controls) -> list[FixtureState]:
        if st.section_index != self.section_index:
            self.section_index = st.section_index
            if st.section != BUILD:  # builds keep the previous colours and drain them to white
                self._pick_palette(st.section)
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
        A, B = self.palette[0], self.palette[-1]
        if sb < 1.0:
            # IMPACT: everything white, full, decaying over the first beat
            hit = 1.0 - 0.5 * sb
            fs.set_color(WHITE)
            fs.intensity = hit if f.kind != STROBE else 1.0 - sb
            self._move(f, fs, 0.0, -0.4)
            return
        var = int(sb // 32) % 3                 # new variation every 8 bars
        beat_i = int(st.beat)
        k = st.kick
        if f.kind in (WASH, DIMMER):
            if var == 0:      # pump: whole group hits on the kick, colours swap every beat
                fs.set_color(A if (beat_i + f.index) % 2 == 0 else B)
                fs.intensity = 0.15 + 0.85 * k
            elif var == 1:    # chase: 1/2-beat chase, bouncing
                n = max(1, f.group_size)
                step = int(st.beat * 2) % (2 * n - 2 if n > 2 else n)
                pos = step if step < n else 2 * n - 2 - step
                fs.set_color(A if (beat_i // 2) % 2 == 0 else B)
                fs.intensity = max(1.0 if pos == f.index else 0.1, 0.5 * k)
            else:             # split: left vs right, swapping each beat
                left = f.x < 0.5
                fs.set_color(A if left == (beat_i % 2 == 0) else B)
                fs.intensity = (0.3 + 0.7 * k) if left == (beat_i % 2 == 0) else 0.15 * k
            fs.intensity = max(fs.intensity, st.snare * self.snare_amt * 0.9)
        elif f.kind == MOVER:
            fs.set_color(B if f.index % 2 else A)
            fs.intensity = 0.5 + 0.5 * k
            mirror = -1 if f.index % 2 else 1
            if var == 0:      # fan sweeps, mirrored pairs
                self._move(f, fs, mirror * math.sin(2 * math.pi * st.beat / 4), -0.5 + 0.8 * k)
            elif var == 1:    # circles, one bar per lap
                a = 2 * math.pi * (st.beat / 4 + f.index * 0.25)
                self._move(f, fs, math.sin(a), math.cos(a) * 0.8)
            else:             # crossing X, snaps each beat
                side = 1 if (beat_i + f.index) % 2 else -1
                self._move(f, fs, side * 0.9, 0.2)
        elif f.kind == STROBE:
            amt = self.strobe_amt
            if var == 1:
                fs.intensity = amt if st.beat_in_bar < 0.15 else 0.0   # downbeats only
            else:
                fs.intensity = amt * max(st.snare ** 2 * self.snare_amt, (k ** 4) * 0.5)
            fs.set_color(WHITE)

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
