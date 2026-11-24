"""Pre-analyzed mode: analyze a whole track once, save a timeline, then look up
the exact musical state at any moment during playback (so drops land perfectly)."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from ..state import BUILD, DROP, MusicState
from .beats import (estimate_tempo, find_downbeat_phase, fit_grid, onset_envelope, phase_envelope,
                    tempo_is_steady, track_beats)
from .features import FeatureExtractor, HOP, robust_norm
from .sections import Section, bar_features, label_sections

TIMELINE_VERSION = 1
ENV_DECIM = 4  # store energy curves at ~21 Hz


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load any file libsndfile can read (wav, flac, mp3, ogg, aiff) as mono float32."""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def pick_onsets(x: np.ndarray, frame_rate: float, min_gap_s: float, k: float = 1.0) -> np.ndarray:
    """Peak picking with an adaptive threshold. Returns frame indices."""
    w = max(3, int(1.0 * frame_rate))
    kern = np.ones(w) / w
    mean = np.convolve(x, kern, mode="same")
    std = np.sqrt(np.maximum(np.convolve(x * x, kern, mode="same") - mean ** 2, 0))
    floor = 0.1 * np.percentile(x, 99)
    thr = np.maximum(mean + k * std, floor)
    is_peak = (x > thr) & (x >= np.roll(x, 1)) & (x >= np.roll(x, -1))
    idx = np.flatnonzero(is_peak)
    gap = int(min_gap_s * frame_rate)
    out, last = [], -10 ** 9
    for i in idx:
        if i - last >= gap:
            out.append(i)
            last = i
    return np.asarray(out, np.int64)


@dataclass
class Timeline:
    duration: float
    bpm: float
    beats: np.ndarray            # beat times (s)
    downbeat_phase: int          # index of the first downbeat in `beats`
    sections: list[dict]         # {kind, start, end, start_beat, end_beat}
    env_rate: float              # samples per second of the curves below
    curves: dict[str, np.ndarray]  # energy, low, mid, high (0..1)
    onsets: dict[str, np.ndarray]  # kick, snare, hat times (s)
    source: str = ""
    _sec_starts: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self._sec_starts = np.array([s["start"] for s in self.sections]) if self.sections else np.zeros(1)

    # ---------- persistence ----------
    def to_json(self) -> dict:
        return {
            "version": TIMELINE_VERSION,
            "source": self.source,
            "duration": round(self.duration, 3),
            "bpm": round(self.bpm, 3),
            "downbeat_phase": self.downbeat_phase,
            "beats": [round(float(b), 4) for b in self.beats],
            "sections": self.sections,
            "env_rate": self.env_rate,
            "curves": {k: [round(float(v), 3) for v in arr] for k, arr in self.curves.items()},
            "onsets": {k: [round(float(v), 4) for v in arr] for k, arr in self.onsets.items()},
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_json(), f)

    @classmethod
    def load(cls, path: str) -> "Timeline":
        with open(path) as f:
            d = json.load(f)
        if d.get("version") != TIMELINE_VERSION:
            raise ValueError("old timeline format, re-analyze")
        return cls(
            duration=d["duration"], bpm=d["bpm"], beats=np.array(d["beats"]),
            downbeat_phase=d["downbeat_phase"], sections=d["sections"], env_rate=d["env_rate"],
            curves={k: np.array(v) for k, v in d["curves"].items()},
            onsets={k: np.array(v) for k, v in d["onsets"].items()}, source=d.get("source", ""),
        )

    # ---------- lookup ----------
    def beat_at(self, t: float) -> float:
        """Continuous beat position at time t (extrapolates past the ends)."""
        b = self.beats
        if len(b) < 2:
            return t * self.bpm / 60
        if t <= b[0]:
            return (t - b[0]) / (b[1] - b[0])
        if t >= b[-1]:
            return len(b) - 1 + (t - b[-1]) / (b[-1] - b[-2])
        i = int(np.searchsorted(b, t, side="right")) - 1
        return i + (t - b[i]) / (b[i + 1] - b[i])

    def _env(self, name: str, t: float, tau: float) -> float:
        on = self.onsets.get(name)
        if on is None or len(on) == 0:
            return 0.0
        i = int(np.searchsorted(on, t, side="right")) - 1
        if i < 0:
            return 0.0
        return math.exp(-(t - on[i]) / tau)

    def _curve(self, name: str, t: float) -> float:
        arr = self.curves[name]
        x = t * self.env_rate
        i = int(x)
        if i < 0:
            return float(arr[0])
        if i >= len(arr) - 1:
            return float(arr[-1])
        f = x - i
        return float(arr[i] * (1 - f) + arr[i + 1] * f)

    def state_at(self, t: float) -> MusicState:
        beat = self.beat_at(t)
        rel = beat - self.downbeat_phase
        st = MusicState(t=t, beat=beat)
        st.beat_in_bar = rel % 4
        st.bar = int(math.floor(rel / 4))
        i = max(0, int(np.searchsorted(self.beats, t)) - 1)
        if 0 < i < len(self.beats) - 1:
            st.bpm = 60.0 / (self.beats[i + 1] - self.beats[i])
        else:
            st.bpm = self.bpm
        if self.sections:
            si = max(0, int(np.searchsorted(self._sec_starts, t, side="right")) - 1)
            sec = self.sections[si]
            st.section = sec["kind"]
            st.section_index = si
            st.section_beats = beat - sec["start_beat"]
            length = max(1e-6, sec["end_beat"] - sec["start_beat"])
            st.section_progress = float(np.clip(st.section_beats / length, 0, 1))
            if si + 1 < len(self.sections):
                st.next_section = self.sections[si + 1]["kind"]
                st.beats_to_next = sec["end_beat"] - beat
        for k in ("energy", "low", "mid", "high"):
            setattr(st, k, self._curve(k, t))
        st.kick = self._env("kick", t, 0.12)
        st.snare = self._env("snare", t, 0.09)
        st.hat = self._env("hat", t, 0.05)
        st.playing = 0 <= t <= self.duration
        return st

    def summary(self) -> str:
        lines = [f"BPM {self.bpm:.1f} | {len(self.beats)} beats | {self.duration / 60:.1f} min"]
        for s in self.sections:
            bars = (s["end_beat"] - s["start_beat"]) / 4
            m0, s0 = divmod(s["start"], 60)
            lines.append(f"  {int(m0):2d}:{s0:05.2f}  {s['kind']:<10} {bars:5.0f} bars")
        return "\n".join(lines)


def analyze_audio(audio: np.ndarray, sr: int, bpm_range=(90.0, 190.0), source: str = "") -> Timeline:
    fx = FeatureExtractor(sr)
    feats = fx.push(audio)
    fr = fx.frame_rate
    duration = len(audio) / sr

    env = onset_envelope(feats, fr)
    bpm = estimate_tempo(env, fr, bpm_range)
    if tempo_is_steady(env, fr, bpm, bpm_range):
        beat_frames, _ = fit_grid(env, fr, bpm, phase_env=phase_envelope(feats, fr))
    else:  # tempo changes (DJ mix, live recording): follow it beat by beat
        beat_frames = track_beats(env, fr, bpm)
    beat_frames = beat_frames[beat_frames < len(env)]
    beat_times = fx.frame_time(beat_frames)
    if len(beat_times) > 8:
        # least-squares fit of the beat grid: far more precise than one frame (11 ms)
        bpm = 60.0 / np.polyfit(np.arange(len(beat_times)), beat_times, 1)[0]
    phase = find_downbeat_phase(feats, beat_frames)

    onset_frames = {
        "kick": pick_onsets(feats["kick_flux"], fr, 0.15, 1.2),
        "snare": pick_onsets(feats["snare_flux"], fr, 0.06, 1.0),
        "hat": pick_onsets(feats["hat_flux"], fr, 0.05, 1.0),
    }

    bar_starts, m = bar_features(feats, beat_frames, phase, onset_frames)
    secs: list[Section] = label_sections(m)
    sections = []
    for s in secs:
        sb = int(bar_starts[s.start_bar]) if s.start_bar < len(bar_starts) else len(beat_times) - 1
        eb = int(bar_starts[s.end_bar]) if s.end_bar < len(bar_starts) else len(beat_times) - 1
        if s.start_bar == 0:
            sb = 0  # include any pickup beats in the first section
        sections.append({
            "kind": s.kind,
            "start": round(float(beat_times[sb]) if sb > 0 else 0.0, 4),
            "end": round(float(beat_times[eb]) if eb < len(beat_times) - 1 else duration, 4),
            "start_beat": sb if sb > 0 else 0,
            "end_beat": eb,
        })

    dec = slice(None, None, ENV_DECIM)
    curves = {
        "energy": robust_norm(20 * np.log10(feats["rms"] + 1e-6))[dec],
        "low": robust_norm(feats["kick_e"] * 0.6 + feats["bass_e"] * 0.4)[dec],
        "mid": robust_norm(feats["mid_e"])[dec],
        "high": robust_norm(feats["hat_e"] * 0.6 + feats["snare_e"] * 0.4)[dec],
    }
    env_rate = fr / ENV_DECIM
    return Timeline(
        duration=duration, bpm=float(bpm), beats=beat_times, downbeat_phase=phase,
        sections=sections, env_rate=env_rate, curves=curves,
        onsets={k: fx.frame_time(v) for k, v in onset_frames.items()}, source=os.path.basename(source),
    )


def cache_path(path: str) -> str:
    return path + ".jaylights.json"


def analyze_file(path: str, bpm_range=(90.0, 190.0), use_cache=True) -> Timeline:
    cp = cache_path(path)
    if use_cache and os.path.exists(cp) and os.path.getmtime(cp) >= os.path.getmtime(path):
        try:
            return Timeline.load(cp)
        except (ValueError, KeyError, json.JSONDecodeError):
            pass
    audio, sr = load_audio(path)
    tl = analyze_audio(audio, sr, bpm_range, source=path)
    try:
        tl.save(cp)
    except OSError:
        pass
    return tl


__all__ = ["Timeline", "analyze_audio", "analyze_file", "load_audio", "BUILD", "DROP", "HOP"]
