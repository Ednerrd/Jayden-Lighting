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

from ..state import BREAKDOWN, BUILD, DROP, GROOVE, MusicState
from .beats import estimate_tempo, fold_phase
from .features import FeatureExtractor


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


class LiveAnalyzer:
    def __init__(self, sr: int, bpm_range=(90.0, 190.0), latency: float = 0.0):
        self.fx = FeatureExtractor(sr)
        fr = self.fx.frame_rate
        self.fr = fr
        self.bpm_range = bpm_range
        self.latency = latency  # extra seconds to predict ahead (DMX + fixture lag)

        self.env_buf = deque(maxlen=int(8 * fr))     # all hits: for tempo
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
        self._phase_nov = np.zeros(4)
        self._last_coarse_beat = None
        self._beat_coarse = []

        # per-beat measurements for the section logic
        self._beat_acc = {"kick": 0.0, "tension": [], "energy": [], "low": [], "snares": 0}
        self.hist = deque(maxlen=64)
        self.section = GROOVE
        self.section_index = 0
        self.section_start_beat = 0.0
        self.no_kick_beats = 0
        self.drop_energy = 0.0
        self.forced: str | None = None
        self._low_slow = -9.0
        self._kick_ref = None
        self._pre_drop = None   # set while a drop is still provisional
        self._pre_drop_low = 0.0
        self._drop_low = None

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
        self._phase_nov[:] = 0
        self._phase_nov[self.downbeat_offset] = 5.0

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
        if self.frame % 16 == 0 and len(self.env_buf) > fr * 3:
            self._update_tempo()

        prev_beat = self.beat_pos
        self.beat_pos += self.bpm / 60.0 / fr

        # THE DROP: first big kick after a build/breakdown. Don't wait for more beats.
        self._low_slow += (f["kick_e"] - self._low_slow) / (2.0 * fr)
        if k > 0 and self.section not in (BUILD, BREAKDOWN):
            # remember how hard this track's real kicks hit
            self._kick_ref = f["kick_e"] if self._kick_ref is None else 0.9 * self._kick_ref + 0.1 * f["kick_e"]
        if (k > 0.3 and not self.forced and self.section in (BUILD, BREAKDOWN)
                and self.no_kick_beats >= 4 and f["kick_e"] > self._low_slow + 0.8
                and (self._kick_ref is None or f["kick_e"] > self._kick_ref - 0.4)):
            b = round(self._beat_at_latency(-1.0 / fr))
            self._pre_drop = (self.section, self.section_start_beat, self.downbeat_offset, b)
            self.beat_pos = b + self.bpm / 60.0 / fr   # the drop is ON the beat
            prev_beat = self.beat_pos - 1e-6
            self.downbeat_offset = b % 4              # ...and on the 1
            self._set_section(DROP)
            self.no_kick_beats = 0
            self.drop_energy = -120.0   # measured over the next beats
            self._pre_drop_low = self._low_slow
            self._kick_env = 1.0

        # per-beat accumulation
        acc = self._beat_acc
        acc["kick"] = max(acc["kick"], f["kick_flux"])
        acc["energy"].append(20 * math.log10(f["rms"] + 1e-6))
        acc["low"].append(f["kick_e"])
        bright = math.log(f["centroid"] + 1)
        acc["tension"].append(f["hat_e"] + 2.0 * bright + 3.0 * f["flatness"])
        if s > 0:
            acc["snares"] += 1
        self._beat_coarse.append(f["coarse"])
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
        if self.locked and (not self._kick_running() or self.section in (BUILD, BREAKDOWN)):
            # breakdown / build: nothing solid to follow (snare rolls lie), and the
            # tempo doesn't change mid-track, so just let the grid free-run
            return

        # 1. coarse tempo (autocorrelation): to acquire, or to notice a new track's tempo
        cand = estimate_tempo(env, self.fr, self.bpm_range, prior_bpm=self.bpm if self._ever_locked else 128.0)
        if not self._ever_locked:
            self.bpm = cand
        elif abs(cand / self.bpm - 1) >= 0.03 and (self.section == DROP or not _ratio_related(cand, self.bpm)):
            # (ratio jumps are only trusted during a drop: full drums = best evidence)
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

    def _on_beat(self):
        acc = self._beat_acc
        beat_n = int(math.floor(self.beat_pos))

        # downbeat: bars change on the 1, so track which phase has the most spectral change
        if self._beat_coarse:
            c = np.mean(self._beat_coarse, axis=0)
            if self._last_coarse_beat is not None:
                nov = float(np.linalg.norm(c - self._last_coarse_beat))
                ph = beat_n % 4
                self._phase_nov *= 0.995
                self._phase_nov[ph] += nov
                best = int(np.argmax(self._phase_nov))
                if best != self.downbeat_offset and self._phase_nov[best] > 1.15 * self._phase_nov[self.downbeat_offset]:
                    self.downbeat_offset = best
            self._last_coarse_beat = c
            self._beat_coarse = []

        kick = self.g_kick(acc["kick"])
        energy = float(np.mean(acc["energy"])) if acc["energy"] else -60.0
        tension = self.g_tension(float(np.mean(acc["tension"]))) if acc["tension"] else 0.0
        low = float(np.mean(acc["low"])) if acc["low"] else -9.0
        self.hist.append({"kick": kick, "energy": energy, "tension": tension, "snares": acc["snares"], "low": low})
        self._beat_acc = {"kick": 0.0, "tension": [], "energy": [], "low": [], "snares": 0}
        if not self.forced:
            self._sections()

    def _sections(self):
        h = list(self.hist)
        if len(h) < 8:
            return
        # "full" = kick pattern going, OR the low end is still at drop level (dubstep
        # wobbles and trap 808s can smear the kick, but the floor is still shaking)
        kick_now = self._kick_running() or (self._drop_low is not None and h[-1]["low"] > self._drop_low - 0.25)
        if self.section == DROP and self._kick_running():
            self._drop_low = h[-1]["low"] if self._drop_low is None else 0.95 * self._drop_low + 0.05 * h[-1]["low"]
        beats_in = self.beat_pos - self.section_start_beat
        energy = np.mean([x["energy"] for x in h[-4:]])
        tension = [x["tension"] for x in h[-16:]]
        snares = [x["snares"] for x in h[-16:]]
        rising = len(tension) >= 8 and (
            np.polyfit(np.arange(len(tension)), tension, 1)[0] > 0.01
            or np.mean(snares[-4:]) > np.mean(snares[:4]) + 1
        )
        self.no_kick_beats = 0 if kick_now else self.no_kick_beats + 1
        sec = self.section

        if self._pre_drop is not None:
            # provisional drop: needs the kick to keep going, or we take it back
            if beats_in >= 3:
                # confirmed only if the kick pattern keeps going AND the low end stays up
                low_up = np.mean([x["low"] for x in h[-3:]]) > self._pre_drop_low + 0.5
                if not (self._kick_running(3) and low_up):
                    prev, start, offset, _ = self._pre_drop
                    self.section, self.section_start_beat, self.downbeat_offset = prev, start, offset
                    self.section_index += 1
                    self.no_kick_beats = 4
                self._pre_drop = None
            return
        if beats_in < 8:
            return  # hold every section for at least 2 bars, no flicker

        if kick_now:
            if sec in (BUILD, BREAKDOWN):
                # kick came back without the instant detector firing (soft re-entry)
                self._set_section(GROOVE if sec == BREAKDOWN and energy < self.drop_energy - 6 else DROP)
                if self.section == DROP:
                    self.drop_energy = energy
            elif sec == DROP:
                self.drop_energy = max(self.drop_energy - 0.02, float(np.mean([x["energy"] for x in h[-8:]])))
                earlier = float(np.mean([x["energy"] for x in h[-8:-4]]))
                if energy < self.drop_energy - 6 and earlier < self.drop_energy - 6 and beats_in > 32:
                    self._set_section(GROOVE)
            elif sec == GROOVE and rising and np.mean(snares[-4:]) >= 3 and beats_in > 32:
                self._set_section(BUILD)  # build with the kick still going
        else:
            if sec in (DROP, GROOVE) and self.no_kick_beats >= 2:
                self._set_section(BUILD if rising else BREAKDOWN)
            elif sec == BREAKDOWN and rising:
                self._set_section(BUILD)

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
        st.next_section = None
        st.beats_to_next = None
        st.locked = self.locked
        frac = beat - math.floor(beat)
        kick_on = self._kick_running() and self.section not in (BUILD, BREAKDOWN)
        if self.locked and kick_on:
            # predicted pulse: fires exactly on the beat instead of after we hear it
            st.kick = math.exp(-frac * 60.0 / self.bpm / 0.12)
        else:
            st.kick = self._kick_env
        st.snare = self._snare_env
        st.hat = self._hat_env
