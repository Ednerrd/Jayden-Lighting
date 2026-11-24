"""Frame-by-frame audio features, shared by the offline and live analyzers.

Works in a streaming fashion: push() any amount of audio and it returns features
for every complete frame. Offline analysis just pushes the whole track at once.
"""

from __future__ import annotations

import numpy as np

FRAME = 2048
HOP = 512

# Frequency bands (Hz) that roughly isolate the parts of an EDM track.
BANDS = {
    "kick": (35, 120),
    "bass": (120, 300),
    "mid": (300, 2000),
    "snare": (2000, 5000),
    "hat": (7000, 16000),
}
BAND_NAMES = tuple(BANDS)
N_COARSE = 24  # coarse log-spaced spectrum, used for downbeat detection


class FeatureExtractor:
    def __init__(self, sr: int):
        self.sr = sr
        self.frame_rate = sr / HOP
        self.window = np.hanning(FRAME).astype(np.float32)
        freqs = np.fft.rfftfreq(FRAME, 1.0 / sr)
        self.freqs = freqs
        self.band_masks = [(freqs >= lo) & (freqs < min(hi, sr / 2)) for lo, hi in BANDS.values()]
        edges = np.geomspace(40, min(16000, sr / 2 - 1), N_COARSE + 1)
        self.coarse_idx = np.clip(np.searchsorted(freqs, edges), 1, len(freqs) - 1)
        self._buf = np.zeros(0, np.float32)
        self._prev_log = None
        self.frames_done = 0

    def frame_time(self, index) -> np.ndarray | float:
        """Center time (s) of a frame index."""
        return (np.asarray(index) * HOP + FRAME / 2) / self.sr

    def push(self, samples: np.ndarray) -> dict[str, np.ndarray] | None:
        """Feed mono float audio. Returns features for any newly completed frames."""
        self._buf = np.concatenate([self._buf, np.asarray(samples, np.float32)])
        if len(self._buf) < FRAME:
            return None
        n = 1 + (len(self._buf) - FRAME) // HOP
        frames = np.lib.stride_tricks.sliding_window_view(self._buf, FRAME)[::HOP][:n]
        feats = self._compute(frames)
        self._buf = self._buf[n * HOP:]
        feats["index"] = np.arange(self.frames_done, self.frames_done + n)
        self.frames_done += n
        return feats

    def _compute(self, frames: np.ndarray) -> dict[str, np.ndarray]:
        spec = np.abs(np.fft.rfft(frames * self.window, axis=1)).astype(np.float32)
        power = spec ** 2
        logmag = np.log1p(spec)

        prev = self._prev_log if self._prev_log is not None else logmag[:1]
        diff = np.diff(np.vstack([prev, logmag]), axis=0)
        flux = np.maximum(diff, 0.0)
        self._prev_log = logmag[-1:]

        out: dict[str, np.ndarray] = {}
        for name, mask in zip(BAND_NAMES, self.band_masks):
            out[name + "_e"] = np.log10(power[:, mask].mean(axis=1) + 1e-9)
            out[name + "_flux"] = flux[:, mask].mean(axis=1)
        out["flux"] = flux.mean(axis=1)
        out["rms"] = np.sqrt((frames ** 2).mean(axis=1))
        total = power.sum(axis=1) + 1e-12
        out["centroid"] = (power * self.freqs).sum(axis=1) / total
        # Spectral flatness: close to 1 for noise (risers / white-noise sweeps), ~0 for tones.
        hi = power[:, self.freqs > 1000] + 1e-12
        out["flatness"] = np.exp(np.log(hi).mean(axis=1)) / hi.mean(axis=1)
        cs = np.cumsum(power, axis=1)
        ci = self.coarse_idx
        coarse = (cs[:, ci[1:]] - cs[:, ci[:-1]]) / np.maximum(ci[1:] - ci[:-1], 1)
        out["coarse"] = np.log10(coarse + 1e-9)
        return out


def concat_features(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    chunks = [c for c in chunks if c is not None]
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}


def robust_norm(x: np.ndarray, lo_pct=5, hi_pct=98) -> np.ndarray:
    """Scale to 0..1 using percentiles, so one loud spike doesn't squash everything."""
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)
