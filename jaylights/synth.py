"""Generates a fake EDM track with a known structure.

Used by `jaylights demo` (so you can see the show with no music files) and by the
tests, which check that the analyzer finds the builds and drops where we put them.
"""

from __future__ import annotations

import numpy as np

SR = 44100

# (section, bars). 128 BPM, 4/4.
DEFAULT_STRUCTURE = [
    ("intro", 8),
    ("groove", 8),
    ("build", 8),
    ("drop", 16),
    ("breakdown", 8),
    ("build", 8),
    ("drop", 16),
    ("outro", 8),
]


def _env(n, attack, decay, sr=SR):
    t = np.arange(n) / sr
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-t / decay)


def _kick(sr=SR):
    n = int(0.35 * sr)
    t = np.arange(n) / sr
    freq = 45 + 110 * np.exp(-t / 0.03)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return np.sin(phase) * _env(n, 0.001, 0.12) * 0.9


def _noise_hit(rng, dur, decay, sr=SR):
    n = int(dur * sr)
    return rng.standard_normal(n) * _env(n, 0.0005, decay)


def _highpass(x, amount=0.95):
    # crude high-pass (blend of first difference and the raw signal) so noise
    # hits sound like hats/claps instead of rumble
    d = np.diff(x, prepend=x[:1])
    return (amount * d + (1 - amount) * x).astype(np.float32)


def _add(out, sig, start):
    start = int(start)
    if start >= len(out):
        return
    end = min(len(out), start + len(sig))
    out[start:end] += sig[: end - start]


def make_track(bpm=128.0, structure=None, seed=7, sr=SR, halftime=False):
    """Returns (audio, sections) where sections is a list of (kind, start_s, end_s).

    halftime=True makes a dubstep/trap groove: kick on 1 and the 'and' of 2, snare on 3.
    """
    structure = structure or DEFAULT_STRUCTURE
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    total_bars = sum(b for _, b in structure)
    out = np.zeros(int(total_bars * 4 * beat * sr) + sr, np.float32)

    kick = _kick(sr)
    hat = _highpass(_highpass(_noise_hit(rng, 0.05, 0.012, sr), 1.0), 1.0) * 0.12  # 2nd-order: mostly >6 kHz
    clap = _highpass(_noise_hit(rng, 0.2, 0.06, sr), 1.0) * 0.5
    chord_roots = [55.0, 43.65, 49.0, 41.2]  # A, F, G, E (bass notes, Hz)

    sections = []
    bar0 = 0
    for kind, bars in structure:
        s0 = bar0 * 4 * beat
        sections.append((kind, s0, (bar0 + bars) * 4 * beat))
        for b in range(bars):
            bar_start = (bar0 + b) * 4 * beat
            prog = (b + 1) / bars
            root = chord_roots[(bar0 + b) // 2 % 4]
            for q in range(4):
                bt = bar_start + q * beat
                pos = int(bt * sr)
                kick_gain = 1.0 if kind == "drop" else 0.75
                if kind in ("intro", "groove", "drop", "outro"):
                    if not halftime:
                        _add(out, kick * kick_gain, pos)
                    elif q == 0:
                        _add(out, kick * kick_gain, pos)
                    elif q == 1:
                        _add(out, kick * kick_gain, pos + int(beat * 0.5 * sr))
                if kind in ("groove", "drop", "outro") or (kind == "intro" and b >= 4):
                    _add(out, hat * (1.3 if kind == "drop" else 0.8), pos + int(beat * 0.5 * sr))
                if kind in ("groove", "drop") and (q == 2 if halftime else q in (1, 3)):
                    _add(out, clap * (1.0 if kind == "drop" else 0.6), pos)
                if kind == "build":
                    # snare roll that speeds up: quarters -> eighths -> 16ths -> 32nds
                    div = 1 if prog <= 0.25 else 2 if prog <= 0.5 else 4 if prog <= 0.75 else 8
                    for k in range(div):
                        _add(out, clap * (0.3 + 0.7 * prog), pos + int(k * beat / div * sr))
            # half-time (dubstep/trap): sustained wobble bass instead of offbeat stabs
            if halftime and kind in ("groove", "drop"):
                n = int(4 * beat * sr)
                tt = np.arange(n) / sr
                saw = 2 * ((tt * root * 2) % 1.0) - 1
                wobble = 0.5 + 0.5 * np.sin(2 * np.pi * tt / beat * (2 if b % 2 else 1))
                _add(out, saw * wobble * (0.25 if kind == "drop" else 0.12), int(bar_start * sr))
            # bass line (offbeat 8ths) in groove / drop
            elif kind in ("groove", "drop"):
                n = int(beat * 0.45 * sr)
                tt = np.arange(n) / sr
                # low-passed saw (first 5 harmonics), like a real mixed bass
                saw = sum(np.sin(2 * np.pi * root * 2 * h * tt) / h for h in range(1, 6))
                note = saw * _env(n, 0.01, 0.15) * (0.22 if kind == "drop" else 0.14)
                for q in range(4):
                    _add(out, note, int((bar_start + q * beat + beat / 2) * sr))
            # pads in breakdown / build / intro
            if kind in ("breakdown", "build", "intro"):
                n = int(4 * beat * sr)
                tt = np.arange(n) / sr
                pad = sum(np.sin(2 * np.pi * root * 4 * r * tt) for r in (1, 1.26, 1.5)) / 3
                _add(out, pad * 0.12 * np.minimum(1, tt / 0.3), int(bar_start * sr))
        if kind == "build":
            # white-noise riser over the whole build
            n = int(bars * 4 * beat * sr)
            ramp = np.linspace(0, 1, n) ** 2
            riser = _highpass(rng.standard_normal(n).astype(np.float32), 0.97) * ramp * 0.25
            _add(out, riser, int(s0 * sr))
        bar0 += bars

    out /= max(1e-6, np.abs(out).max()) / 0.9
    return out.astype(np.float32), sections


def write_track(path, bpm=128.0):
    import soundfile as sf

    audio, sections = make_track(bpm)
    sf.write(path, audio, SR)
    return sections
