"""Live mode: listen to audio as it plays (DJ mixer -> line in, or loopback) and
work out beats + sections in real time.

It can't see the future, so it works the way a human LD would:
* tempo from the last ~8 s of hits, phase locked to the kick (then *predicted*, so
  lights fire on the beat instead of ~30 ms after it)
* kick disappears -> breakdown; tension rising with no kick -> build;
  kick slams back in after a build/breakdown -> DROP
You can always override from the visualizer (force BUILD / DROP, tap the "1").
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..state import BREAKDOWN, BUILD, DROP, GROOVE, INTRO, MusicState
from .beats import downbeat_scores, estimate_tempo, fold_phase
from .features import FeatureExtractor
from .live_sections import LiveSections


class AutoGain:
    """Tracks a running floor/ceiling of a signal so it can be scaled to 0..1."""

    def __init__(self, frames_to_adapt: float, lo=None, hi=None):
        self.rate = 1.0 / max(1.0, frames_to_adapt)
        self.lo, self.hi = lo, hi

    def __call__(self, x: float) -> float:
        if self.lo is None:
            self.lo, self.hi = x, x + 1e-6
        span = max(self.hi - self.lo, 1e-6)
        self.hi = max(x, self.hi - span * self.rate)
        self.lo = min(x, self.lo + span * self.rate)
        return min(1.0, max(0.0, (x - self.lo) / max(self.hi - self.lo, 1e-6)))


class PeakNorm:
    """x / (slowly decaying peak). Unlike AutoGain it doesn't freak out when the
    signal is perfectly steady (a kick on every beat)."""

    def __init__(self, frames_to_halve: float):
        self.decay = 0.5 ** (1.0 / max(1.0, frames_to_halve))
        self.peak = 1e-9

    def __call__(self, x: float) -> float:
        self.peak = max(x, self.peak * self.decay)
        return min(1.0, x / self.peak)


class OnsetDetector:
    """Adaptive-threshold peak picker, one frame of latency."""

    def __init__(self, frame_rate: float, min_gap_s: float, k: float = 1.5):
        self.a = 1.0 / (frame_rate * 0.8)
        self.k = k
        self.gap = int(min_gap_s * frame_rate)
        self.mean = 0.0
        self.var = 0.0
        self.prev = [0.0, 0.0]
        self.since = 10 ** 6
        self.peak = 1e-6

    def __call__(self, x: float) -> float:
        """Returns the strength (0..1) of an onset detected at the previous frame, else 0."""
        p2, p1 = self.prev
        self.prev = [p1, x]
        self.since += 1
        self.peak = max(x, self.peak * 0.9995)
        hit = 0.0
        thr = self.mean + self.k * math.sqrt(self.var) + 0.05 * self.peak
        if p1 > thr and p1 >= p2 and p1 > x and self.since >= self.gap:
            hit = min(1.0, p1 / self.peak)
            self.since = 0
        d = x - self.mean
        self.mean += self.a * d
        self.var = (1 - self.a) * (self.var + self.a * d * d)
        return hit


def _ratio_related(a: float, b: float, tol: float = 0.015) -> bool:
    """Half / double / 2:3 / 3:4 tempo jumps are almost always the tracker getting
    fooled by a rhythm pattern, not a real tempo change."""
    r = a / b
    return any(abs(r / q - 1) < tol for q in (0.5, 2 / 3, 0.75, 4 / 3, 1.5, 2.0))


def _octave(a: float, b: float, tol: float = 0.015) -> bool:
    """Exactly half or double the tempo."""
    r = a / b
    return abs(r / 0.5 - 1) < tol or abs(r / 2.0 - 1) < tol


class BarFeatures:
    """The six per-bar features of sections.bar_features(), but computed causally.

    Offline normalises every band against the whole track's percentiles. Live only has the
    last ~3 minutes of frames, so the same robust_norm (5th..98th percentile) is taken over a
    rolling window of the frames heard so far (the bar itself included), and each bar is
    scaled with the percentiles as they stand when the bar ends.
    """

    KEYS = ("low", "energy", "high", "bright", "noise", "kflux")

    def __init__(self, frame_rate: float, window_s: float = 180.0):
        n = int(window_s * frame_rate)
        self.win = {k: deque(maxlen=n) for k in self.KEYS}
        self.chunks: deque = deque(maxlen=4)   # the last 4 beats
        self.cur = self._new()
        self.hit_hist: list[int] = []

    @staticmethod
    def _new():
        return {"low": [], "energy": [], "high": [], "bright": [], "noise": [], "kmax": 0.0, "hits": 0}

    def add(self, f, hits: int):
        c = self.cur
        vals = {"low": float(f["kick_e"]), "energy": 20 * math.log10(float(f["rms"]) + 1e-6),
                "high": float(f["hat_e"]) * 0.6 + float(f["snare_e"]) * 0.4, "bright": math.log(float(f["centroid"]) + 1),
                "noise": float(f["flatness"]), "kflux": float(f["kick_flux"])}
        for k in self.KEYS:
            self.win[k].append(vals[k])
        for k in ("low", "energy", "high", "bright", "noise"):
            c[k].append(vals[k])
        c["kmax"] = max(c["kmax"], vals["kflux"])
        c["hits"] += hits

    def end_beat(self):
        self.chunks.append(self.cur)
        self.cur = self._new()

    def ready(self) -> bool:
        return len(self.chunks) == 4

    def row(self) -> dict:
        ch = list(self.chunks)
        p95 = float(np.percentile(np.fromiter(self.win["kflux"], float), 95)) + 1e-9
        beat_kick = sorted(min(max(c["kmax"] / p95, 0.0), 1.5) for c in ch)

        def norm(key):
            w = np.fromiter(self.win[key], float)
            lo, hi = np.percentile(w, [5, 98])
            x = np.concatenate([np.asarray(c[key], float) for c in ch])
            if hi - lo < 1e-9 or len(x) == 0:
                return 0.0
            return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0).mean())

        high, bright, noise = norm("high"), norm("bright"), norm("noise")
        hits = sum(c["hits"] for c in ch)
        self.hit_hist.append(hits)
        dens = hits / max(1.0, float(np.percentile(self.hit_hist, 95)))
        return {"kick": beat_kick[-2], "low": norm("low"), "energy": norm("energy"), "high": high,
                "tension": (high + bright + noise) / 3, "density": dens}


class LiveAnalyzer:
    def __init__(self, sr: int, bpm_range=(90.0, 190.0), latency: float = 0.0):
        self.fx = FeatureExtractor(sr)
        fr = self.fx.frame_rate
        self.fr = fr
        self.bpm_range = bpm_range
        self.latency = latency  # extra seconds to predict ahead (DMX + fixture lag)

        self.env_buf = deque(maxlen=int(16 * fr))    # all hits: for tempo (16 s = enough to see whole bars)
        self.phase_buf = deque(maxlen=int(6 * fr))   # kick + snare: for where the beat is
        self._ever_locked = False
        self._phase_good = False
        self._flip_votes = 0
        self._n_kick, self._n_snare = PeakNorm(fr * 30), PeakNorm(fr * 30)
        self.kick_det = OnsetDetector(fr, 0.15, 1.5)
        self.snare_det = OnsetDetector(fr, 0.06, 1.5)
        self.hat_det = OnsetDetector(fr, 0.05, 1.5)
        adapt = fr * 20
        self.g_energy, self.g_low, self.g_mid, self.g_high = (AutoGain(adapt) for _ in range(4))
        self.g_kick = PeakNorm(60)  # per beat: halves after ~60 beats of no kick
        self.g_tension = AutoGain(fr * 30)
        self.g_env = AutoGain(fr * 10)

        self.frame = 0
        self.bpm = 128.0
        self.beat_pos = 0.0          # continuous beat counter
        self.locked = False
        self._bpm_votes = 0
        self._pending_bpm = None
        self.downbeat_offset = 0     # which beat number mod 4 is the "1"

        # per-beat measurements for the section logic
        self._beat_acc = self._new_acc()
        self.hist = deque(maxlen=64)
        self.section = GROOVE
        self.section_index = 0
        self.section_start_beat = 0.0
        self.no_kick_beats = 0
        self.forced: str | None = None
        self._low_slow = -9.0
        self._kick_ref = None
        self._pre_drop = None   # set while a drop is still provisional
        self.low_ref = None          # running "drop level" of the sub (log10 power, per-beat median)
        self.peak_ref = None         # same, for the per-beat loudest frame (kick hits)
        self._cooldown_until = -1.0  # no instant drops until this beat (after a false alarm)
        self._kick_beat, self._prev_kick_e = 0.0, -9.0
        self._full = True            # is the floor shaking right now (with hysteresis)
        self._empty_beats = 0
        self._freeze = False         # the floor is out (build / breakdown): don't chase hits, free-run the grid
        self._freeze_min = 0.0
        # the bar-level section machine (causal twin of the offline labeller) and what it saw
        self.bars = BarFeatures(fr)
        self.sections = LiveSections()
        self.bar_trace: list[dict] = []   # per bar: its six causal features, section, lock (for Copy analysis)
        # rolling window for finding the "1" with the same method as offline mode
        self._fbuf = deque(maxlen=int(16 * fr))   # (coarse, kick_flux, snare_flux) per frame
        self._beat_frames = deque(maxlen=48)      # (frame number, beat number) at each beat
        self._db_score = np.zeros(4)              # evidence per (beat number mod 4)
        self._db_hold_until = -1.0                # after a tap, trust the user for a while

        self.state = MusicState(locked=False)
        self._kick_env = self._snare_env = self._hat_env = 0.0
        self._decay = {"kick": math.exp(-1 / (0.12 * fr)), "snare": math.exp(-1 / (0.09 * fr)),
                       "hat": math.exp(-1 / (0.05 * fr))}

    # ---------------- controls (from the visualizer) ----------------
    def force_section(self, kind: str | None):
        """None = back to automatic."""
        self.forced = kind
        if kind:
            self._set_section(kind)

    def tap_downbeat(self):
        """User says 'this beat is the 1'."""
        b = int(round(self.beat_pos))
        self.downbeat_offset = b % 4
        self.beat_pos = float(b) + (self.beat_pos - round(self.beat_pos))
        self._db_hold_until = self.beat_pos + 64

    def nudge_half_beat(self):
        """Lights on the offbeat? Hit this."""
        self.beat_pos += 0.5
        self._flip_votes = 0

    # ---------------- audio in ----------------
    def push(self, samples: np.ndarray) -> MusicState:
        feats = self.fx.push(samples)
        if feats is not None:
            for i in range(len(feats["rms"])):
                self._frame({k: v[i] for k, v in feats.items()})
        return self.state

    def _frame(self, f):
        fr = self.fr
        self.frame += 1

        # hits
        k = self.kick_det(f["kick_flux"])
        s = self.snare_det(f["snare_flux"])
        h = self.hat_det(f["hat_flux"])
        self._kick_env = max(k, self._kick_env * self._decay["kick"])
        self._snare_env = max(s, self._snare_env * self._decay["snare"])
        self._hat_env = max(h, self._hat_env * self._decay["hat"])

        # onset envelope for tempo (same recipe as offline, kick-weighted)
        env = 0.6 * f["kick_flux"] + 0.25 * f["snare_flux"] + 0.15 * f["flux"]
        self.env_buf.append(self.g_env(env))
        self.phase_buf.append(0.5 * self._n_kick(f["kick_flux"]) + 0.5 * self._n_snare(f["snare_flux"]))

        # tempo + phase: re-estimate ~5x per second
        if self.frame % 16 == 0 and len(self.env_buf) > fr * 6:
            self._update_tempo()

        prev_beat = self.beat_pos
        self.beat_pos += self.bpm / 60.0 / fr

        # THE DROP: the sub slams back in after a build / breakdown / pre-drop gap.
        # Fires on the first hit instead of waiting a beat or two.
        frac = self.beat_pos - math.floor(self.beat_pos)
        # triggered by the sub level itself crossing up (onset detectors get deafened by
        # risers and snare rolls right when it matters)
        prev_low, self._prev_kick_e = self._prev_kick_e, f["kick_e"]
        # armed in a build, or in a breakdown / a groove whose sub is dipping or cut
        # (the bar machine decides later whether the slam was real)
        dipping = self.section == GROOVE and (self.sections.dip_run >= 2 or self.sections.empty_run >= 1)
        floor_out = self._empty_beats >= 2 and self.section != DROP
        if (not self.forced and (self.section in (BUILD, BREAKDOWN) or dipping or floor_out) and self._pre_drop is None
                and self.peak_ref is not None and len(self.hist) >= 16
                and self.beat_pos > self._cooldown_until
                # from a breakdown: only after 2+ bars and once tension is up (a kick coming
                # back at the START of a build isn't a drop). From a build: any time.
                and (self.section == BUILD or (self.beat_pos - self.section_start_beat >= 8
                     and np.mean([x["tension"] for x in list(self.hist)[-4:]]) >= 0.6))):
            # compare peak with peak: this frame's kick vs the loudest kicks of the track
            hist = list(self.hist)
            recent = float(np.mean([x["peak"] for x in hist[-2:]]))
            # a drop is a low end we haven't heard for at least 4 bars
            unheard = max(x["peak"] for x in hist[-16:]) + 0.25
            level = max(self.peak_ref - 0.3, recent + 0.35, unheard)
            # ...or the very first hit after a silent gap (kick-through builds never leave the sub
            # "unheard", but a real gap before the slam is just as clear a cue)
            after_gap = (self.section == BUILD and self._empty_beats >= 2 and self.sections.run >= 2
                         and f["kick_e"] > self.peak_ref - 0.3 and prev_low < self.peak_ref - 0.6)
            if (f["kick_e"] > level >= prev_low and (self._empty_beats >= 1 or recent < self.peak_ref - 0.35)) or after_gap:
                self._kick_beat = self._beat_at_latency(-0.5 / fr)
                b = round(self._kick_beat)
                self._pre_drop = (self.section, self.section_start_beat, self.downbeat_offset, b)
                self.beat_pos += b - self._kick_beat       # the drop is ON the beat
                prev_beat = self.beat_pos - 1e-6
                self.downbeat_offset = b % 4              # ...and on the 1
                self._db_score[b % 4] += 3.0
                self._set_section(DROP)
                self._kick_env = 1.0

        # per-beat accumulation
        acc = self._beat_acc
        acc["kick"] = max(acc["kick"], f["kick_flux"])
        if frac < 0.15:   # right on the beat: which drum hits here? (for finding the "1")
            acc["kick_on"] = max(acc["kick_on"], f["kick_flux"])
            acc["snare_on"] = max(acc["snare_on"], f["snare_flux"])
        acc["energy"].append(20 * math.log10(f["rms"] + 1e-6))
        acc["low"].append(f["kick_e"])
        bright = math.log(f["centroid"] + 1)
        acc["tension"].append(f["hat_e"] + 2.0 * bright + 3.0 * f["flatness"])
        if s > 0:
            acc["snares"] += 1
        self.bars.add(f, (1 if s > 0 else 0) + (1 if h > 0 else 0))
        self._fbuf.append((f["coarse"], f["kick_flux"], f["snare_flux"]))
        if math.floor(self.beat_pos) > math.floor(prev_beat):
            self._on_beat()

        # outputs
        st = self.state
        lvl_e = self.g_energy(20 * math.log10(f["rms"] + 1e-6))
        st.energy = 0.8 * st.energy + 0.2 * lvl_e
        st.low = 0.7 * st.low + 0.3 * self.g_low(f["kick_e"] * 0.6 + f["bass_e"] * 0.4)
        st.mid = 0.7 * st.mid + 0.3 * self.g_mid(f["mid_e"])
        st.high = 0.7 * st.high + 0.3 * self.g_high(f["hat_e"] * 0.6 + f["snare_e"] * 0.4)
        self._publish()

    def _beat_at_latency(self, dt: float) -> float:
        return self.beat_pos + dt * self.bpm / 60.0

    def _kick_running(self, beats: int = 4) -> bool:
        """Is there a real kick pattern? Four-on-the-floor or half-time (dubstep/trap)."""
        if len(self.hist) < 4:
            return True
        k = sorted(x["kick"] for x in list(self.hist)[-beats:])
        return k[-1] > 0.5 and k[-2] > 0.3

    def _update_tempo(self):
        env = np.fromiter(self.env_buf, float)
        if env.std() < 1e-4:
            self.locked = False
            return
        if self.locked and (not (self._kick_running() or self._full) or self._freeze):
            # breakdown / build: nothing solid to follow (snare rolls lie), and the
            # tempo doesn't change mid-track, so just let the grid free-run
            return

        # 1. coarse tempo (autocorrelation): to acquire, or to notice a new track's tempo
        cand = estimate_tempo(env, self.fr, self.bpm_range, prior_bpm=self.bpm if self._ever_locked else 128.0)
        if not self._ever_locked:
            self.bpm = cand
        elif abs(cand / self.bpm - 1) >= 0.03 and (
                not _ratio_related(cand, self.bpm)
                or (_octave(cand, self.bpm) and (self.section == DROP or self.frame < 45 * self.fr))
                or (self.section == DROP and self._full)):
            # octave jumps (x2, x1/2) are only trusted early on, or in a drop (full drums = best
            # evidence); otherwise they're the tracker getting fooled. 2:3 / 3:4 jumps are never a free
            # pass in the first 45 s: they need a drop with the sub in (melodic dubstep at 150 slipped to
            # 100 in its kick-running melody part, Breathe It Out Slowly 0:33 and 2:20)
            # a different tempo has to win for ~3 s straight before we believe it
            if self._pending_bpm and abs(cand / self._pending_bpm - 1) < 0.03:
                self._bpm_votes += 1
            else:
                self._pending_bpm, self._bpm_votes = cand, 1
            if self._bpm_votes >= 15:
                self.bpm = cand
                self._bpm_votes = 0
        else:
            self._bpm_votes = 0

        # 2. fine tempo + phase, measured directly from the last 6 s of kick/snare hits:
        # try tempos within +-2% and keep the one where hits line up sharpest.
        recent = np.fromiter(self.phase_buf, float)
        recent = np.maximum(recent - np.convolve(recent, np.ones(21) / 21, mode="same"), 0)
        if recent.sum() <= 0:
            return
        ages = np.arange(len(recent))[::-1]  # frames ago
        periods = 60 * self.fr / self.bpm * (1 + np.linspace(-0.02, 0.02, 41))
        ph = (-ages[None, :] / periods[:, None]) % 1.0   # beat phase of each hit, relative to now
        bins = 64
        idx = (ph * bins).astype(np.int64) % bins
        hist = np.zeros((len(periods), bins))
        np.add.at(hist, (np.repeat(np.arange(len(periods)), len(recent)), idx.ravel()), np.tile(recent, len(periods)))
        hist = hist + np.roll(hist, 1, axis=1) + np.roll(hist, -1, axis=1)
        sharp = hist.max(axis=1) / (hist.mean(axis=1) + 1e-9)
        j = int(np.argmax(sharp))
        period = periods[j]
        if 0 < j < len(periods) - 1:  # parabolic refine
            a, b, c = sharp[j - 1], sharp[j], sharp[j + 1]
            d = a - 2 * b + c
            if abs(d) > 1e-9:
                period = period + 0.5 * (a - c) / d * (periods[1] - periods[0])
        # circular mean of the hits around the peak = where the beat is, relative to now
        peak = (np.argmax(hist[j]) + 0.5) / bins
        dphi = (ph[j] - peak + 0.5) % 1.0 - 0.5
        near = np.abs(dphi) < 0.2
        hit_phase = (peak + (dphi[near] * recent[near]).sum() / max(recent[near].sum(), 1e-9)) % 1.0
        self.locked = sharp[j] > 1.6
        if not self.locked:
            return
        self._ever_locked = True
        self.bpm += 0.3 * (60 * self.fr / period - self.bpm)
        # a hit `age` frames ago folded to phase -age/period; hits are ON beats, so the
        # true phase right now is -hit_phase. Compare with where our grid thinks we are.
        err = ((self.beat_pos % 1.0) + hit_phase + 0.5) % 1.0 - 0.5
        if abs(err) > 0.3 and self._phase_good:
            # looks like the offbeat (basslines / open hats). Only believe it if it
            # keeps saying so for ~1 s.
            self._flip_votes += 1
            if self._flip_votes < 5:
                return
        self._flip_votes = 0
        self._phase_good = True
        self.beat_pos -= 0.3 * err if abs(err) < 0.3 else err

    @staticmethod
    def _new_acc():
        return {"kick": 0.0, "kick_on": 0.0, "snare_on": 0.0, "tension": [], "energy": [], "low": [], "snares": 0}

    def _on_beat(self):
        """Called when we cross into a new beat: score the beat that just ENDED."""
        acc = self._beat_acc
        ended = int(math.floor(self.beat_pos)) - 1

        # downbeat: every bar, run the offline "1" finder (spectral change + kick on
        # the beat - snare on the beat) over the last ~16 s. Change only on 2 votes.
        self._beat_frames.append((self.frame, ended + 1))
        if (ended + 1) % 4 == 0 and len(self._beat_frames) >= 17 and self.beat_pos > self._db_hold_until:
            first = self.frame - len(self._fbuf) + 1
            bf = [(fn - first, bn) for fn, bn in self._beat_frames if fn - first >= 0]
            if len(bf) >= 13:
                buf = list(self._fbuf)
                feats = {"coarse": np.array([x[0] for x in buf]),
                         "kick_flux": np.array([x[1] for x in buf]),
                         "snare_flux": np.array([x[2] for x in buf])}
                sc = downbeat_scores(feats, np.array([x[0] for x in bf]))
                self._db_score *= 0.85
                for ph in range(4):
                    self._db_score[bf[ph][1] % 4] += sc[ph]
                best = int(np.argmax(self._db_score))
                if (best != self.downbeat_offset and self._pre_drop is None
                        and self._db_score[best] > self._db_score[self.downbeat_offset] + 1.0):
                    self.downbeat_offset = best

        kick = self.g_kick(acc["kick"])
        energy = float(np.mean(acc["energy"])) if acc["energy"] else -60.0
        tension = self.g_tension(float(np.mean(acc["tension"]))) if acc["tension"] else 0.0
        low = float(np.median(acc["low"])) if acc["low"] else -9.0
        peak = float(np.max(acc["low"])) if acc["low"] else -9.0
        self.hist.append({"kick": kick, "energy": energy, "tension": tension, "snares": acc["snares"],
                          "low": low, "peak": peak})
        self.peak_ref = peak if self.peak_ref is None else max(peak, self.peak_ref - 0.0015)
        self._beat_acc = self._new_acc()

        # floor level: running drop-level reference + full/empty with hysteresis
        self.low_ref = low if self.low_ref is None else max(low, self.low_ref - 0.0015)
        if low > self.low_ref - 0.3:
            self._full = True
        elif low < self.low_ref - 0.8:
            self._full = False
        self._empty_beats = 0 if self._full else self._empty_beats + 1
        if self._empty_beats >= 2:
            if not self._freeze:
                self._freeze, self._freeze_min = True, self.beat_pos + 6
        elif self._freeze and self._full and self.beat_pos >= self._freeze_min:
            self._freeze = False
        self.bars.end_beat()
        if (int(math.floor(self.beat_pos)) - self.downbeat_offset) % 4 == 0 and self.bars.ready():
            self._on_bar(self.bars.row())

    def _on_bar(self, row: dict):
        """A bar just ended: hand its causal features to the section machine. The machine
        decides the section; a provisional drop (the frame-level trigger that fired mid-bar on
        the first sub hit) is confirmed if the machine agrees, taken back if it doesn't."""
        kind = self.sections.push(row, slam=self._pre_drop is not None)
        self.bar_trace.append(dict(row, section=kind, locked=bool(self.locked), t=self.frame / self.fr))
        if self.forced:
            self._pre_drop = None
            return
        if self._pre_drop is not None:
            if kind != DROP:
                prev, start, offset, _ = self._pre_drop
                self.section, self.section_start_beat, self.downbeat_offset = prev, start, offset
                self.section_index += 1
                self._cooldown_until = self.beat_pos + 16  # don't cry wolf again right away
            self._pre_drop = None
        if kind != self.section:
            self._set_section(kind)

    def _set_section(self, kind: str):
        if kind != self.section:
            self.section = kind
            self.section_index += 1
            self.section_start_beat = float(round(self.beat_pos))

    def _publish(self):
        st = self.state
        beat = self._beat_at_latency(self.latency)
        st.t = self.frame / self.fr
        st.bpm = self.bpm
        st.beat = beat
        rel = beat - self.downbeat_offset
        st.beat_in_bar = rel % 4
        st.bar = int(math.floor(rel / 4))
        st.section = self.section
        st.section_index = self.section_index
        st.section_beats = beat - self.section_start_beat
        if self.section == BUILD:
            tension = self.hist[-1]["tension"] if self.hist else 0
            st.section_progress = float(min(0.98, 0.5 * st.section_beats / 32 + 0.5 * tension))
        else:
            st.section_progress = float(min(1.0, st.section_beats / 64))
        st.build_stage = (2 if self.sections.stage2 else 1) if self.section == BUILD else 0
        st.next_section = None
        st.beats_to_next = None
        st.locked = self.locked
        frac = beat - math.floor(beat)
        kick_on = self._full and self.section in (GROOVE, DROP, INTRO)
        if self.locked and kick_on:
            # predicted pulse: fires exactly on the beat instead of after we hear it
            st.kick = math.exp(-frac * 60.0 / self.bpm / 0.12)
        else:
            st.kick = self._kick_env
        st.snare = self._snare_env
        st.hat = self._hat_env
