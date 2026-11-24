"""Realistic, varied synthetic EDM for stress-testing the analyzers.

jaylights/synth.py makes very clean tracks. This module makes tracks that are
closer to what DJs actually play: long DJ-friendly intros, kick-through builds
(the kick gets high-passed and fades), gaps before drops, breakdowns that still
have hats/percussion/pads/vocal-ish formant tones, drops in several genres
(house/progressive/techno/melodic, dubstep half-time with sub that masks the
kick, trap 808s with hat rolls, DnB breakbeats with reese bass), a heavily
compressed/limited master, pickup bars / leading silence and hat swing.

    audio, truth = make_realistic_track(spec_or_name, seed)

`truth` has: sections [(kind, start_s, end_s)], beats (s), downbeats (s), bpm,
genre, name, duration, spec.  `suite_specs(quick)` is the list used by
tools/eval_analysis.py.  `python tools/stress_synth.py --list` prints it and
`--wav NAME out.wav` renders one for listening.
"""

from __future__ import annotations

import copy
import math

import numpy as np

SR = 44100
TAU = 2 * np.pi
GENRES = ("house", "progressive", "techno", "melodic", "dubstep", "trap", "dnb")
FOUR = ("house", "progressive", "techno", "melodic")


# --------------------------------------------------------------------------- dsp helpers
def fftfilt(x, sr, lo=None, hi=None, order=2):
    """Zero-phase high/low-pass via FFT (no scipy needed)."""
    n = len(x)
    N = 1 << int(np.ceil(np.log2(max(n, 16))))  # power-of-2 FFT (odd sizes are very slow)
    X = np.fft.rfft(x, N)
    f = np.fft.rfftfreq(N, 1.0 / sr)[: len(X)] + 1e-9
    g = np.ones(len(X), np.float32)
    if lo:
        g *= 1 / np.sqrt(1 + (lo / f) ** (2 * order))
    if hi:
        g *= 1 / np.sqrt(1 + (f / hi) ** (2 * order))
    return np.fft.irfft(X * g, N)[:n].astype(np.float32)


def env(n, attack, decay, sr=SR):
    t = np.arange(n) / sr
    return (np.clip(t / max(attack, 1e-4), 0, 1) * np.exp(-t / decay)).astype(np.float32)


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))) + 1e-9)


def reverb(x, sr, rt=1.3, rng=None, decim=4, hi=5000.0):
    """Cheap convolution reverb (noise IR), done at 1/decim rate."""
    rng = rng or np.random.default_rng(0)
    n = len(x)
    m = n // decim
    xd = x[: m * decim].reshape(m, decim).mean(axis=1)
    sr2 = sr / decim
    L = int(rt * 1.2 * sr2)
    t = np.arange(L) / sr2
    ir = rng.standard_normal(L) * np.exp(-t / (rt / 6.9)) * np.minimum(1, t / 0.01)
    ir = fftfilt(ir.astype(np.float32), sr2, lo=150, hi=min(hi, sr2 / 2.5))
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
    nfft = 1 << int(np.ceil(np.log2(m + L)))
    y = np.fft.irfft(np.fft.rfft(xd, nfft) * np.fft.rfft(ir, nfft), nfft)[:m].astype(np.float32)
    up = np.interp(np.arange(n) / decim, np.arange(m), y, right=0).astype(np.float32)
    return up


# --------------------------------------------------------------------------- instruments
class Kit:
    """All one-shots for a track (deterministic given the rng)."""

    def __init__(self, genre, rng, sr=SR, tune=None):
        self.sr, self.rng, self.genre = sr, rng, genre
        self.kicks = [self._kick(genre, hp) for hp in (None, 160.0, 380.0)]
        self.clap = self._clap()
        self.snare = self._snare()
        self.hat_c = self._hat(0.018, 7500)
        self.hat_o = self._hat(0.13, 6500)
        self.ride = self._hat(0.35, 5000)
        self.shaker = self._shaker()
        self.percs = [self._perc(f, d) for f, d in ((190, 0.10), (310, 0.08), (460, 0.06), (140, 0.14))]
        self.crash = self._crash()
        self.boom = self._kick("trap", None, root=45.0)
        self.rim = self._rim()
        self._cache = {}
        self._kick808 = {}

    def kick808(self, f):
        key = round(f)
        if key not in self._kick808:
            self._kick808[key] = self._kick("trap", None, root=float(key))
        return self._kick808[key]

    def _kick(self, genre, hp, root=48.0):
        sr = self.sr
        if genre == "techno":
            dur, f0, f1, tau, dec, click = 0.42, 190, 50, 0.028, 0.20, 0.25
        elif genre == "dubstep":
            dur, f0, f1, tau, dec, click = 0.26, 150, 52, 0.022, 0.09, 0.5
        elif genre == "trap":
            dur, f0, f1, tau, dec, click = 1.0, 160, root, 0.012, 0.42, 0.15
        elif genre == "dnb":
            dur, f0, f1, tau, dec, click = 0.2, 190, 58, 0.014, 0.07, 0.5
        else:
            dur, f0, f1, tau, dec, click = 0.34, 150, 47, 0.03, 0.13, 0.3
        n = int(dur * sr)
        t = np.arange(n) / sr
        freq = f1 + (f0 - f1) * np.exp(-t / tau)
        ph = TAU * np.cumsum(freq) / sr
        k = np.sin(ph) * env(n, 0.001, dec, sr)
        k += click * np.random.default_rng(1).standard_normal(n) * env(n, 0.0003, 0.004, sr) * 0.5
        if genre == "trap":
            k = np.tanh(1.6 * k)  # 808 saturation -> harmonics so it's audible on small speakers
        k = k.astype(np.float32) * 0.95
        if hp:
            k = fftfilt(k, sr, lo=hp, order=2) * 1.2
        return k

    def _clap(self):
        sr, rng = self.sr, self.rng
        n = int(0.28 * sr)
        x = np.zeros(n, np.float32)
        noise = rng.standard_normal(n).astype(np.float32)
        for off, g in ((0, 0.6), (0.011, 0.7), (0.023, 0.8), (0.036, 1.0)):
            s = int(off * sr)
            x[s:] += noise[: n - s] * env(n - s, 0.0005, 0.012 if g < 1 else 0.075, sr) * g
        return fftfilt(x, sr, lo=800, hi=6500) * 0.9

    def _snare(self):
        sr, rng = self.sr, self.rng
        n = int(0.3 * sr)
        nz = fftfilt(rng.standard_normal(n).astype(np.float32), sr, lo=1200, hi=11000) * env(n, 0.0004, 0.07, sr)
        t = np.arange(n) / sr
        body = np.sin(TAU * (170 + 90 * np.exp(-t / 0.02)) * t) * env(n, 0.0004, 0.05, sr)
        return (0.8 * nz + 0.7 * body).astype(np.float32)

    def _hat(self, decay, hp):
        sr = self.sr
        n = int(max(decay * 6, 0.05) * sr)
        x = self.rng.standard_normal(n).astype(np.float32) * env(n, 0.0004, decay, sr)
        return fftfilt(x, sr, lo=hp, order=3) * 0.55

    def _shaker(self):
        sr = self.sr
        n = int(0.09 * sr)
        x = self.rng.standard_normal(n).astype(np.float32) * env(n, 0.012, 0.03, sr)
        return fftfilt(x, sr, lo=3500, hi=12000) * 0.7

    def _perc(self, f, d):
        sr = self.sr
        n = int(0.4 * sr)
        t = np.arange(n) / sr
        return (np.sin(TAU * (f + 0.4 * f * np.exp(-t / 0.015)) * t) * env(n, 0.0005, d, sr) * 0.7).astype(np.float32)

    def _rim(self):
        sr = self.sr
        n = int(0.06 * sr)
        t = np.arange(n) / sr
        x = (np.sin(TAU * 1700 * t) + 0.6 * np.sin(TAU * 2450 * t) + 0.3 * self.rng.standard_normal(n)) * env(n, 0.0002, 0.008, sr)
        return (x * 0.35).astype(np.float32)

    def _crash(self):
        sr = self.sr
        n = int(1.8 * sr)
        x = self.rng.standard_normal(n).astype(np.float32) * env(n, 0.002, 0.5, sr)
        return fftfilt(x, sr, lo=3500, order=2) * 0.7

    # ---- tonal one-shots, cached by their musical parameters
    def get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]


def saw_stack(f, n, sr, harm=8, detune=(1.0,), wt=None):
    t = np.arange(n) / sr
    y = np.zeros(n, np.float32)
    for d in detune:
        for h in range(1, harm + 1):
            if f * d * h < sr / 2.2:
                y += (np.sin(TAU * f * d * h * t + h) / h).astype(np.float32)
    return y / len(detune)


def pad_chord(root, n, sr, kind="min"):
    iv = {"min": (0, 3, 7, 10), "maj": (0, 4, 7, 11), "sus": (0, 5, 7, 12)}[kind]
    y = np.zeros(n, np.float32)
    for s in iv:
        f = root * 2 ** (s / 12)
        y += saw_stack(f, n, sr, harm=7, detune=(0.997, 1.003))
    t = np.arange(n) / sr
    a = np.minimum(1, t / 0.35) * np.minimum(1, (n / sr - t) / 0.25 + 0.0).clip(0, 1)
    y = fftfilt(y * a.astype(np.float32), sr, hi=2800, order=2)
    return y / (_rms(y) * 6.0)


VOWELS = {"a": (730, 1090, 2440), "o": (570, 840, 2410), "e": (530, 1840, 2480), "u": (300, 870, 2240), "i": (270, 2290, 3010)}


def vocal_note(f0, dur, sr, vowel="a", seed=0, vib=5.0, breathy=0.05):
    """Formant-synthesised, voice-like tone (harmonics shaped by 3 formants + vibrato)."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = f0 * (1 + 0.012 * np.sin(TAU * vib * t) * np.minimum(1, t / 0.2))
    ph = TAU * np.cumsum(f) / sr
    fm = VOWELS[vowel]
    y = np.zeros(n, np.float32)
    for h in range(1, 40):
        fh = f0 * h
        if fh > 5200 or fh > sr / 2.2:
            break
        amp = sum(math.exp(-0.5 * (math.log(fh / c) / 0.22) ** 2) * w for c, w in zip(fm, (1.0, 0.6, 0.3)))
        y += (np.sin(h * ph) * amp / (h ** 0.5)).astype(np.float32)
    y += breathy * np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.2
    a = np.minimum(1, t / 0.04) * np.minimum(1, (dur - t) / 0.08).clip(0, 1)
    return (y * a.astype(np.float32)) / (_rms(y) * 9.0)


def wobble_note(root, dur, sr, beat, lfo_per_beat, seed=0):
    """Dubstep growl: FM saw-ish bass with an LFO on the modulation index + big sub."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = root * 2
    lfo = 0.5 + 0.5 * np.sin(TAU * lfo_per_beat / beat * t - 1.2)
    beta = 0.8 + 5.5 * lfo
    ph = TAU * f * t
    y = np.sin(ph + beta * np.sin(ph * 1.0)) * (0.55 + 0.45 * lfo)
    y += 1.0 * np.sin(TAU * root * t)  # sub
    y = np.tanh(1.5 * y)
    a = np.minimum(1, t / 0.01) * np.minimum(1, (dur - t) / 0.02).clip(0, 1)
    return (y * a).astype(np.float32)


def reese_note(root, dur, sr, seed=0):
    n = int(dur * sr)
    f = root * 2
    y = saw_stack(f, n, sr, harm=14, detune=(0.994, 1.006)) * 0.5
    y = fftfilt(y.astype(np.float32), sr, hi=1800, order=2)
    t = np.arange(n) / sr
    y += 0.9 * np.sin(TAU * root * t).astype(np.float32)
    a = np.minimum(1, t / 0.01) * np.minimum(1, (dur - t) / 0.03).clip(0, 1)
    return (y * a).astype(np.float32) * 0.8


def sub_bass_note(root, dur, sr, bright=0.5):
    n = int(dur * sr)
    t = np.arange(n) / sr
    y = sum(np.sin(TAU * root * h * t) / (h ** 1.5) * (bright if h > 1 else 1.0) for h in range(1, 6))
    return (y * env(n, 0.008, dur * 0.7, sr)).astype(np.float32)


def pluck(f, dur, sr):
    n = int(dur * sr)
    t = np.arange(n) / sr
    y = saw_stack(f, n, sr, harm=9, detune=(0.998, 1.002))
    y = y * np.exp(-t / (dur * 0.5)).astype(np.float32)
    return fftfilt(y, sr, hi=4500, order=2) * 0.5


# --------------------------------------------------------------------------- specs
def _structure(genre, rng, bpm, variant):
    bar_s = 240.0 / bpm
    if genre in ("dubstep", "trap"):
        intro = int(rng.choice([8, 16, 16, 16]))
        drop_a = int(rng.choice([8, 16, 16, 32]))
        outro = int(rng.choice([8, 16]))
        bd = int(rng.choice([8, 16]))
    elif genre == "dnb":
        intro = int(rng.choice([16, 32, 32]))
        drop_a = int(rng.choice([16, 32, 32]))
        outro = int(rng.choice([16, 32]))
        bd = int(rng.choice([16, 16, 32]))
    else:
        intro = int(rng.choice([16, 24, 32, 32]))
        drop_a = int(rng.choice([16, 32, 32]))
        outro = int(rng.choice([16, 32, 32]))
        bd = int(rng.choice([16, 16, 24, 32]))
    b1 = int(rng.choice([4, 8, 8, 8, 16]))
    b2 = int(rng.choice([8, 8, 16]))
    drop_b = drop_a if rng.random() < 0.55 else drop_a + int(rng.choice([8, 16]))
    st = [("intro", intro)]
    if rng.random() < 0.3:
        st.append(("groove", int(rng.choice([8, 16]))))
    st += [("build", b1), ("drop", drop_a), ("breakdown", bd), ("build", b2), ("drop", drop_b)]
    if variant.get("third_drop"):
        st += [("breakdown", 8), ("build", 8), ("drop", 16)]
    st.append(("outro", outro))
    # keep tracks around <= ~5 minutes so the whole suite runs in reasonable time
    def dur(s):
        return sum(b for _, b in s) * bar_s
    while dur(st) > 300:
        for i, (k, b) in enumerate(st):
            if k in ("intro", "outro", "drop", "breakdown") and b >= 24:
                st[i] = (k, 16)
                break
        else:
            break
    return st


_DEFAULT_BPM = {"house": (122, 128), "progressive": (122, 130), "techno": (126, 138), "melodic": (122, 130),
                "dubstep": (138, 150), "trap": (130, 160), "dnb": (170, 176)}


def random_spec(genre, seed, **over):
    """Randomised spec for a genre. Anything in `over` replaces the drawn value."""
    rng = np.random.default_rng(seed * 7919 + GENRES.index(genre) * 31 + 5)
    lo, hi = _DEFAULT_BPM[genre]
    bpm = float(over.pop("bpm", None) or round(rng.uniform(lo, hi) * 2) / 2)
    variant = {k: over.pop(k) for k in list(over) if k in ("third_drop",)}
    spec = dict(
        name=over.pop("name", f"{genre}_{seed}"), genre=genre, bpm=bpm, seed=seed,
        structure=over.pop("structure", None) or _structure(genre, rng, bpm, variant),
        swing=float(rng.choice([0.0, 0.1, 0.2, 0.3])),
        silence=float(rng.choice([0.0, 0.0, 0.0, 0.25, 0.8])),
        pickup_beats=int(rng.choice([0, 0, 0, 0, 2, 4, 8])),
        drop_level=1.0,
        groove_level=float(rng.uniform(0.75, 0.9)),
        intro_level=float(rng.uniform(0.6, 0.85)),
        build=dict(
            kick_through=bool(rng.random() < 0.65), kick_hp=bool(rng.random() < 0.7), kick_fade=float(rng.choice([0.0, 0.4, 0.7])),
            kick_stop_bars=int(rng.choice([0, 0, 1])), gap_beats=int(rng.choice([0, 0, 1, 2, 4])),
            roll_start=float(rng.choice([0.0, 0.25, 0.5])), riser=bool(rng.random() < 0.85), riser_gain=float(rng.uniform(0.25, 0.6)),
            pitch_riser=bool(rng.random() < 0.5), vocal=bool(rng.random() < 0.4), drums_off=bool(rng.random() < 0.15),
        ),
        breakdown=dict(hats=bool(rng.random() < 0.9), perc=bool(rng.random() < 0.6), soft_kick=bool(rng.random() < 0.3),
                       vocal=bool(rng.random() < 0.6), pad_gain=float(rng.uniform(0.7, 1.2)), bass=bool(rng.random() < 0.3)),
        comp=float(rng.uniform(0.25, 0.65)), drive=float(rng.uniform(1.5, 3.5)),
        mix=dict(kick=float(rng.uniform(0.85, 1.15)), hat=float(rng.uniform(0.7, 1.3)), bass=float(rng.uniform(0.8, 1.2)),
                 pad=float(rng.uniform(0.7, 1.3)), lead=float(rng.uniform(0.7, 1.3))),
        key=int(rng.integers(0, 8)), humanize_ms=float(rng.choice([0.0, 2.0, 4.0])),
        wobble_rate=float(rng.choice([1.0, 1.5, 2.0, 3.0])),
    )
    if genre == "melodic":
        spec["drop_level"] = float(rng.uniform(0.92, 1.0))
        spec["groove_level"] = float(rng.uniform(0.85, 0.95))
    if genre == "progressive":
        spec["drop_level"] = float(rng.uniform(0.95, 1.0))
        spec["groove_level"] = float(rng.uniform(0.8, 0.92))
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(spec.get(k), dict):
            spec[k].update(v)
        else:
            spec[k] = v
    return spec


def suite_specs(quick=False):
    """Hand-picked recipe list (each is also randomised by seed)."""
    R = random_spec
    full = [
        R("house", 1, name="house_classic", bpm=124, build={"kick_through": True, "kick_hp": True, "gap_beats": 1}),
        R("house", 2, name="house_nokick_build", bpm=126, build={"kick_through": False, "gap_beats": 0, "roll_start": 0.25}),
        R("house", 3, name="house_pickup_gap4", bpm=128, pickup_beats=4, silence=0.8, build={"gap_beats": 4, "kick_through": True, "kick_fade": 0.7}),
        R("house", 4, name="house_soft_kickbreak", bpm=122, breakdown={"soft_kick": True, "hats": True}, swing=0.3, third_drop=True),
        R("progressive", 5, name="prog_modest_drop", bpm=126, drop_level=0.95, groove_level=0.92, build={"kick_through": True, "kick_hp": False, "roll_start": 0.5}),
        R("progressive", 6, name="prog_longintro", bpm=124, comp=0.7, build={"kick_through": True, "kick_stop_bars": 1}),
        R("melodic", 7, name="melodic_soft", bpm=124, drop_level=0.93, groove_level=0.9, breakdown={"vocal": True, "hats": True, "perc": True}),
        R("melodic", 8, name="melodic_pickup8", bpm=128, pickup_beats=8, build={"kick_through": False, "pitch_riser": True, "gap_beats": 2}),
        R("techno", 9, name="techno_driving", bpm=132, build={"kick_through": True, "kick_hp": True, "riser": False, "roll_start": 0.5}),
        R("techno", 10, name="techno_135_nobuild_gap", bpm=135, build={"kick_through": True, "kick_fade": 0.0, "gap_beats": 1, "roll_start": 0.0}),
        R("dubstep", 11, name="dubstep_halftime", bpm=140, build={"drums_off": False, "kick_through": False, "gap_beats": 1}),
        R("dubstep", 12, name="dubstep_masking_sub", bpm=145, build={"kick_through": False, "gap_beats": 4, "drums_off": True}, third_drop=True),
        R("dubstep", 13, name="dubstep_riddim_150", bpm=150, wobble_rate=3.0, build={"kick_through": True, "kick_hp": True, "gap_beats": 2}),
        R("trap", 14, name="trap_808_140", bpm=140, build={"kick_through": False, "roll_start": 0.0, "gap_beats": 1}),
        R("trap", 15, name="trap_150_gap", bpm=150, build={"kick_through": False, "gap_beats": 2, "riser": True}),
        R("trap", 16, name="trap_130_swing", bpm=130, swing=0.3, silence=0.25),
        R("dnb", 17, name="dnb_174_classic", bpm=174, build={"kick_through": False, "gap_beats": 1, "roll_start": 0.25}),
        R("dnb", 18, name="dnb_172_kickbuild", bpm=172, build={"kick_through": True, "kick_hp": True, "gap_beats": 0}),
        R("dnb", 19, name="dnb_175_pickup", bpm=175, pickup_beats=4, build={"gap_beats": 2}, breakdown={"hats": True, "perc": True}),
        R("house", 20, name="house_heavy_limit", bpm=125, comp=0.85, drive=4.5),
        R("progressive", 21, name="prog_nogroove_hp", bpm=128, build={"kick_through": True, "kick_hp": True, "kick_fade": 0.7, "gap_beats": 0}),
        R("techno", 22, name="techno_pickup_silence", bpm=130, pickup_beats=2, silence=0.8, comp=0.8),
        R("melodic", 23, name="melodic_130_gap1", bpm=130, build={"gap_beats": 1, "kick_through": True, "kick_hp": True}, third_drop=True),
        R("house", 24, name="house_120_swing", bpm=120, swing=0.35, humanize_ms=4.0),
    ]
    if not quick:
        return full
    pick = ["house_classic", "house_pickup_gap4", "prog_modest_drop", "melodic_pickup8", "dubstep_masking_sub", "trap_808_140",
            "dnb_174_classic", "techno_135_nobuild_gap"]
    return [s for s in full if s["name"] in pick]


# --------------------------------------------------------------------------- renderer
def make_realistic_track(spec, seed=None, sr=SR):
    """Render a spec (dict or suite name or genre string). Returns (audio float32 mono, truth dict)."""
    if isinstance(spec, str):
        named = {s["name"]: s for s in suite_specs()}
        spec = named[spec] if spec in named else random_spec(spec, 0 if seed is None else seed)
    spec = copy.deepcopy(spec)
    if seed is not None:
        spec["seed"] = seed
    genre, bpm = spec["genre"], float(spec["bpm"])
    rng = np.random.default_rng(spec["seed"] * 104729 + 17)
    kit = Kit(genre, rng, sr)
    beat = 60.0 / bpm
    bar = 4 * beat
    structure = spec["structure"]
    pickup = int(spec["pickup_beats"])
    t0 = spec["silence"] + pickup * beat
    total_bars = sum(b for _, b in structure)
    music_end = t0 + total_bars * bar
    n = int((music_end + 3.0) * sr)
    names = ("kick", "drums", "hats", "perc", "bass", "pad", "lead", "fx")
    L = {k: np.zeros(n, np.float32) for k in names}
    send = np.zeros(n, np.float32)  # reverb send
    duck = np.ones(n, np.float32)
    mix = spec["mix"]
    swing, hum = spec["swing"], spec["humanize_ms"] / 1000.0
    base = 48.99 * 2 ** (spec["key"] / 12)
    chord_semis = [0, -4, 3, -2]
    half = genre in ("dubstep", "trap")
    bset = spec["build"]
    bd = spec["breakdown"]

    def put(layer, smp, t, g=1.0, snd=0.0, jitter=0.0):
        if jitter:
            t += rng.normal(0, jitter)
        s = int(round(t * sr))
        if s < 0 or s >= n:
            return
        e = min(n, s + len(smp))
        L[layer][s:e] += smp[: e - s] * g
        if snd:
            send[s:e] += smp[: e - s] * (g * snd)

    def duck_at(t, depth, tau=0.11):
        s = int(round(t * sr))
        if s < 0 or s >= n:
            return
        m = int(0.4 * sr)
        e = min(n, s + m)
        sh = (1 - depth * np.exp(-np.arange(e - s) / (tau * sr))).astype(np.float32)
        np.minimum(duck[s:e], sh, out=duck[s:e])

    # --- ground truth sections
    truth_secs, b0 = [], 0
    for kind, nb in structure:
        truth_secs.append([kind, t0 + b0 * bar, t0 + (b0 + nb) * bar])
        b0 += nb
    truth_secs[0][1] = 0.0
    truth_secs[-1][2] = music_end
    first_bar_t = t0
    drop_starts = [s for k, s, _ in truth_secs if k == "drop"]
    gap_masks = []  # (start_s, end_s) to silence before each drop

    kick_t = []  # times of audible kicks (for sidechain)
    prog_roots = [base * 2 ** (s / 12) for s in chord_semis]

    # --- pickup material (hats + swelling noise, no kick)
    for i in range(pickup):
        tt = spec["silence"] + i * beat
        put("hats", kit.hat_c, tt, 0.5 * mix["hat"])
        put("hats", kit.hat_o, tt + beat / 2, 0.35 * mix["hat"])
    if pickup >= 4:
        m = int(pickup * beat * sr)
        sw = fftfilt(rng.standard_normal(m).astype(np.float32), sr, lo=2000) * (np.linspace(0, 1, m) ** 2).astype(np.float32) * 0.15
        put("fx", sw, spec["silence"], 1.0, 0.3)

    gbar = 0  # global bar index
    for si, (kind, nb) in enumerate(structure):
        sec_t0 = t0 + gbar * bar
        lvl = {"intro": spec["intro_level"], "groove": spec["groove_level"], "drop": spec["drop_level"],
               "outro": spec["groove_level"], "breakdown": 1.0, "build": 1.0}[kind]
        for bi in range(nb):
            bar_t = t0 + (gbar + bi) * bar
            p = (bi + 0.5) / nb  # progress through the section
            root = prog_roots[((gbar + bi) // 2) % 4]
            chord_root = root * 4
            ev_kick = []  # (pos, gain, hp_level)
            ev = []       # (layer, pos, sample, gain, send)
            g_out = 1.0 - 0.55 * p if kind == "outro" else 1.0
            first_bar = bi == 0

            # ================== DRUMS ==================
            if genre in FOUR:
                K4 = (0, 1, 2, 3)
                if kind in ("intro", "outro", "groove", "drop"):
                    kg = {"intro": 0.9, "outro": 0.9, "groove": 0.95, "drop": 1.0}[kind] * (0.9 + 0.1 * lvl)
                    for q in K4:
                        ev_kick.append((q, kg * g_out, 0))
                    # hats: intro starts with just kick (DJ-friendly); percussion and hats creep in
                    hats_on = kind != "intro" or bi >= nb // 4
                    if hats_on:
                        for q in range(4):
                            ev.append(("hats", q + 0.5, kit.hat_o, 0.55 * lvl * g_out, 0))
                    if kind in ("groove", "drop") or (kind == "intro" and bi >= nb // 2) or (kind == "outro" and bi < nb // 2):
                        for q in range(8):
                            pos = q * 0.5 + 0.25
                            ev.append(("hats", pos, kit.hat_c, (0.28 if kind == "drop" else 0.2) * g_out, 0))
                    if kind in ("groove", "drop"):
                        cg = (0.85 if kind == "drop" else 0.55)
                        smp = kit.rim if genre == "techno" and kind == "groove" else kit.clap
                        for q in (1, 3):
                            ev.append(("drums", q, smp, cg * lvl, 0.18))
                    if kind in ("intro", "outro", "drop", "groove") and (bi % 2 == 1 or kind == "drop"):
                        if rng.random() < 0.6 and (kind != "intro" or bi >= nb // 3):
                            pp = kit.percs[int(rng.integers(0, 4))]
                            ev.append(("perc", float(rng.choice([0.75, 1.25, 2.75, 3.25, 3.5])), pp, 0.35, 0.1))
                    if kind == "drop" and genre == "techno":
                        for q in range(4):
                            ev.append(("hats", q + 0.5, kit.ride, 0.25, 0.15))
                elif kind == "breakdown":
                    if bd["hats"]:
                        for q in range(8):
                            ev.append(("hats", q * 0.5, kit.hat_c, 0.22 * (1 + 0.3 * (q % 2)), 0))
                    if bd["perc"] and bi % 2 == 0:
                        ev.append(("perc", float(rng.choice([0.5, 1.5, 2.5, 3.0])), kit.percs[int(rng.integers(0, 4))], 0.35, 0.2))
                        ev.append(("hats", 2.0, kit.shaker, 0.5, 0))
                    if bd["soft_kick"]:
                        for q in (K4 if rng.random() < 0.7 else (0, 2)):
                            ev_kick.append((q, 0.38, 1))
                elif kind == "build":
                    hats_g = 0.2 + 0.4 * p
                    if not bset["drums_off"]:
                        if bset["kick_through"] and not (bset["kick_stop_bars"] and bi >= nb - bset["kick_stop_bars"]):
                            hp = (1 if p > 0.35 else 0) if bset["kick_hp"] else 0
                            hp = (2 if p > 0.7 else hp) if bset["kick_hp"] else 0
                            kg = (1 - bset["kick_fade"] * p) * 0.9
                            for q in K4:
                                ev_kick.append((q, kg, hp))
                        for q in range(8):
                            ev.append(("hats", q * 0.5 + (0.5 if p < 0.3 else 0.25), kit.hat_c, hats_g * 0.6, 0))
                    # snare/clap roll: subdivisions double as the build progresses
                    for q in range(4):
                        pq = (bi + (q + 0.5) / 4) / nb
                        if pq < bset["roll_start"] or bset["roll_start"] >= 0.99:
                            continue
                        rel = (pq - bset["roll_start"]) / max(1e-3, 1 - bset["roll_start"])
                        div = 1 if rel < 0.3 else 2 if rel < 0.6 else 4 if rel < 0.85 else 8
                        for k in range(div):
                            ev.append(("drums", q + k / div, kit.clap if (bi + q) % 2 else kit.snare,
                                       (0.25 + 0.75 * rel) * 0.85, 0.2))
                elif kind == "intro":
                    pass
            elif genre == "dubstep":
                if kind in ("intro", "groove", "drop", "outro"):
                    kpos = [0.0, 2.5] if (bi % 4) != 3 else [0.0, 1.75, 2.5]
                    kg = 1.0 if kind == "drop" else 0.85
                    for q in kpos:
                        ev_kick.append((q, kg * g_out, 0))
                    sg = {"drop": 1.1, "groove": 0.8, "intro": 0.5, "outro": 0.6}[kind]
                    ev.append(("drums", 2.0, kit.snare, sg * g_out, 0.25))
                    if kind in ("drop", "groove"):
                        ev.append(("drums", 2.0, kit.clap, sg * 0.6, 0.25))
                    for q in range(8):
                        if kind != "intro" or bi >= nb // 2:
                            ev.append(("hats", q * 0.5 + 0.5 * (q % 2 == 1) * 0, kit.hat_c, 0.3 * g_out, 0))
                    if kind == "drop":
                        for q in range(4):
                            ev.append(("hats", q + 0.5, kit.hat_o, 0.3, 0))
                elif kind == "breakdown":
                    if bd["hats"]:
                        for q in range(4):
                            ev.append(("hats", q * 1.0 + 0.5, kit.hat_c, 0.2, 0))
                    if bd["perc"] and bi % 2 == 1:
                        ev.append(("perc", 3.5, kit.percs[1], 0.4, 0.2))
                    if bd["soft_kick"]:
                        ev_kick.append((0, 0.4, 1))
                elif kind == "build":
                    if not bset["drums_off"]:
                        if bset["kick_through"] and not (bset["kick_stop_bars"] and bi >= nb - bset["kick_stop_bars"]):
                            for q in (0.0, 2.5):
                                ev_kick.append((q, (1 - bset["kick_fade"] * p) * 0.9, 1 if (bset["kick_hp"] and p > 0.4) else 0))
                        for q in range(8):
                            ev.append(("hats", q * 0.5, kit.hat_c, 0.15 + 0.4 * p, 0))
                    for q in range(4):
                        pq = (bi + (q + 0.5) / 4) / nb
                        if pq < max(bset["roll_start"], 0.25):
                            continue
                        rel = (pq - max(bset["roll_start"], 0.25)) / (1 - max(bset["roll_start"], 0.25) + 1e-3)
                        div = 1 if rel < 0.3 else 2 if rel < 0.6 else 4 if rel < 0.85 else 8
                        for k in range(div):
                            ev.append(("drums", q + k / div, kit.snare, (0.25 + 0.75 * rel) * 0.9, 0.25))
            elif genre == "trap":
                if kind in ("intro", "groove", "drop", "outro"):
                    var = (gbar + bi) % 4
                    kpos = [(0.0, 1.0), (1.75, 0.8), (2.5, 0.9), (3.5 if var in (1, 3) else 3.25, 0.7)]
                    if var == 2:
                        kpos = [(0.0, 1.0), (0.75, 0.7), (2.5, 0.9)]
                    for q, g in kpos:
                        ev_kick.append((q, g * (1.0 if kind == "drop" else 0.8) * g_out, 3))
                    ev.append(("drums", 2.0, kit.snare, (1.0 if kind != "intro" else 0.5) * g_out, 0.3))
                    ev.append(("drums", 2.0, kit.clap, 0.7 * g_out if kind != "intro" else 0.0, 0.3))
                    # 16th hats with velocity wiggle + rolls
                    if kind != "intro" or bi >= nb // 3:
                        for q in range(16):
                            vel = (0.18 + 0.2 * ((q % 4) == 0) + 0.05 * ((q * 7) % 3)) * g_out
                            if (q >= 14 and (bi % 2 == 1)) or (q in (6, 7) and bi % 4 == 3):
                                for k in range(2):
                                    ev.append(("hats", q * 0.25 + k * 0.125, kit.hat_c, vel * (0.7 + 0.5 * k), 0))
                            else:
                                ev.append(("hats", q * 0.25, kit.hat_c, vel, 0))
                        if kind == "drop" and bi % 2 == 0:
                            ev.append(("hats", 3.5, kit.hat_o, 0.35, 0))
                elif kind == "breakdown":
                    if bd["hats"]:
                        for q in range(8):
                            ev.append(("hats", q * 0.5, kit.hat_c, 0.18, 0))
                    if bd["perc"] and bi % 2 == 1:
                        ev.append(("perc", 2.0, kit.rim, 0.8, 0.3))
                    if bd["soft_kick"]:
                        ev_kick.append((0, 0.5, 3))
                elif kind == "build":
                    if not bset["drums_off"]:
                        if bset["kick_through"]:
                            ev_kick.append((0.0, 0.8 * (1 - 0.5 * p), 3))
                        for q in range(16):
                            ev.append(("hats", q * 0.25, kit.hat_c, 0.1 + 0.4 * p, 0))
                    for q in range(4):
                        pq = (bi + (q + 0.5) / 4) / nb
                        if pq < bset["roll_start"]:
                            continue
                        rel = (pq - bset["roll_start"]) / (1 - bset["roll_start"] + 1e-3)
                        div = 1 if rel < 0.3 else 2 if rel < 0.6 else 4 if rel < 0.85 else 8
                        for k in range(div):
                            ev.append(("drums", q + k / div, kit.snare, (0.25 + 0.75 * rel) * 0.9, 0.3))
            elif genre == "dnb":
                if kind in ("intro", "groove", "drop", "outro"):
                    kg = 1.0 if kind == "drop" else 0.9
                    kpos = [0.0, 2.5] if bi % 4 != 3 else [0.0, 0.75, 2.5]
                    for q in kpos:
                        ev_kick.append((q, kg * g_out, 0))
                    sg = 1.0 if kind == "drop" else 0.7
                    if kind != "intro" or bi >= nb // 3:
                        for q in (1.0, 3.0):
                            ev.append(("drums", q, kit.snare, sg * g_out, 0.15))
                        ev.append(("drums", 1.75, kit.snare, 0.22 * g_out, 0.1))
                        ev.append(("drums", 3.5, kit.snare, 0.25 * g_out, 0.1))
                    for q in range(8):
                        ev.append(("hats", q * 0.5 + 0.0, kit.hat_c, (0.45 if q % 2 == 0 else 0.3) * g_out, 0))
                    for q in range(4):
                        ev.append(("hats", q + 0.5, kit.hat_o, 0.28 * g_out, 0))
                    if kind == "drop":
                        ev.append(("hats", 0.0, kit.ride, 0.3, 0.2))
                elif kind == "breakdown":
                    if bd["hats"]:
                        for q in range(8):
                            ev.append(("hats", q * 0.5, kit.hat_c, 0.2, 0))
                    if bd["perc"]:
                        ev.append(("perc", 1.0, kit.rim, 0.8, 0.3))
                    if bd["soft_kick"]:
                        ev_kick.append((0, 0.45, 1))
                elif kind == "build":
                    if not bset["drums_off"]:
                        if bset["kick_through"]:
                            for q in (0.0, 2.5):
                                ev_kick.append((q, 0.85 * (1 - bset["kick_fade"] * p), 1 if bset["kick_hp"] and p > 0.4 else 0))
                        for q in range(8):
                            ev.append(("hats", q * 0.5, kit.hat_c, 0.15 + 0.35 * p, 0))
                    for q in range(4):
                        pq = (bi + (q + 0.5) / 4) / nb
                        if pq < bset["roll_start"]:
                            continue
                        rel = (pq - bset["roll_start"]) / (1 - bset["roll_start"] + 1e-3)
                        div = 1 if rel < 0.25 else 2 if rel < 0.55 else 4 if rel < 0.85 else 8
                        for k in range(div):
                            ev.append(("drums", q + k / div, kit.snare, (0.25 + 0.75 * rel) * 0.85, 0.2))

            # shared: crash on the first bar of drops, and sometimes at phrase starts
            if first_bar and kind == "drop":
                ev.append(("fx", 0.0, kit.crash, 0.8, 0.4))
                ev.append(("fx", 0.0, kit.boom, 0.9, 0.0))
            elif kind in ("groove", "drop") and bi % 8 == 0 and rng.random() < 0.6:
                ev.append(("fx", 0.0, kit.crash, 0.4, 0.4))

            # ================== BASS ==================
            bg = mix["bass"]
            if kind in ("groove", "drop", "outro") or (kind == "breakdown" and bd["bass"]) or (kind == "intro" and False):
                bl = (1.0 if kind == "drop" else 0.75 if kind == "groove" else 0.6) * lvl * g_out * (0.5 if kind == "breakdown" else 1.0)
                if genre in ("house", "melodic"):
                    smp = kit.get(("sub", round(root), 0.45), lambda: sub_bass_note(root, 0.45 * beat, sr, 0.5))
                    for q in range(4):
                        ev.append(("bass", q + 0.5, smp, 0.42 * bl * bg, 0))
                elif genre in ("progressive", "techno"):
                    smp = kit.get(("sub2", round(root), 0.2), lambda: sub_bass_note(root, 0.22 * beat, sr, 0.8))
                    for q in range(4):
                        for off in (0.25, 0.5, 0.75):
                            ev.append(("bass", q + off, smp, 0.38 * bl * bg, 0))
                elif genre == "dubstep" and kind in ("drop", "groove"):
                    rate = spec["wobble_rate"] if kind == "drop" else 1.0
                    for hb in (0, 1):
                        r = rate * (2 if (bi + hb) % 2 else 1)
                        key = ("wob", round(root), r, hb)
                        smp = kit.get(key, lambda: wobble_note(root, 1.8 * beat, sr, beat, r, hb))
                        ev.append(("bass", hb * 2.0, smp, (0.75 if kind == "drop" else 0.35) * bg, 0))
                elif genre == "dnb":
                    for q, d in ((0.0, 1.4), (1.5, 0.45), (2.0, 1.4), (3.25, 0.6)):
                        smp = kit.get(("reese", round(root), d), lambda: reese_note(root, d * beat, sr))
                        ev.append(("bass", q, smp, 0.45 * bl * bg, 0))
                # trap: the 808 kicks are the bass

            # ================== PADS / LEAD / VOCAL ==================
            pg = 0.0
            if kind == "breakdown":
                pg = 0.55 * bd["pad_gain"]
            elif kind == "build":
                pg = 0.2 + 0.3 * p
            elif kind == "intro" and bi >= nb // 2:
                pg = 0.18
            elif kind == "outro":
                pg = 0.25 * (1 - p)
            elif kind == "drop":
                pg = 0.3 if genre in ("melodic", "progressive") else 0.12
            elif kind == "groove":
                pg = 0.15
            if pg > 0 and (bi % 2 == 0 or kind in ("breakdown", "build", "drop")):
                nbar = 2 if kind in ("breakdown", "build") else 1
                if bi % nbar == 0:
                    smp = kit.get(("pad", round(chord_root), nbar), lambda: pad_chord(chord_root, int(nbar * bar * sr), sr))
                    put("pad", smp, bar_t, pg * mix["pad"], 0.55)
            if kind == "breakdown" and bd["vocal"] and bi % 2 == 0:
                f0 = chord_root * 2 * float(rng.choice([1.0, 1.25, 1.5]))
                vs = vocal_note(f0, 1.7 * bar, sr, vowel=str(rng.choice(list(VOWELS))), seed=bi)
                put("lead", vs, bar_t + (0.5 * beat if bi % 4 == 0 else 0), 0.55 * mix["lead"], 0.5)
            if kind == "build" and bset["vocal"] and bi % 2 == 1:
                f0 = chord_root * 2 * (1.0 + 0.5 * p)
                vs = vocal_note(f0, 0.9 * beat, sr, vowel="a", seed=bi)
                for q in (0, 1, 2, 3) if p > 0.5 else (0, 2):
                    put("lead", vs, bar_t + q * beat, 0.4 * mix["lead"], 0.4)
            if kind == "drop":
                if genre in ("melodic", "progressive", "house") and (genre != "house" or (bi // 4) % 2 == 1):
                    notes = [chord_root * 2 * 2 ** (s / 12) for s in (0, 7, 12, 15)]
                    for i16 in range(16):
                        f = notes[(i16 * 3 + bi) % 4]
                        smp = kit.get(("pl", round(f)), lambda: pluck(f, 0.22 * beat * 2, sr))
                        put("lead", smp, bar_t + i16 * beat / 4 + (swing * beat * 0.25 if i16 % 2 else 0), 0.42 * mix["lead"], 0.3)
                elif genre in ("dubstep", "trap", "dnb") and bi % 4 == 3:
                    vs = vocal_note(chord_root * 2, 0.45 * beat, sr, vowel="o", seed=bi)
                    put("lead", vs, bar_t + 2 * beat, 0.5 * mix["lead"], 0.3)

            # ================== PLACE EVENTS ==================
            for pos, g, hp in ev_kick:
                if genre == "trap":
                    smp = kit.kick808(root * 2 if root < 45 else root)
                else:
                    smp = kit.kicks[hp]
                tt = bar_t + pos * beat
                put("kick", smp, tt, g * mix["kick"])
                kick_t.append((tt, g))
            for layer, pos, smp, g, sn in ev:
                tt = bar_t + pos * beat
                sixteenth = round(pos * 4)
                if layer in ("hats", "perc") and abs(pos * 4 - sixteenth) < 1e-6 and sixteenth % 2 == 1:
                    tt += swing * beat * 0.25
                gm = mix["hat"] if layer == "hats" else 1.0
                put(layer, smp, tt, g * gm, sn, jitter=hum if layer in ("hats", "perc") else 0.0)
        gbar += nb

        # ---- section-wide builds: risers + gap
        if kind == "build":
            tb0, tb1 = sec_t0, sec_t0 + nb * bar
            m = int((tb1 - tb0) * sr)
            tt = np.arange(m) / sr
            ramp = (np.arange(m) / m)
            if bset["riser"]:
                nz = rng.standard_normal(m).astype(np.float32)
                lowband = fftfilt(nz, sr, lo=400, hi=2500) * (ramp ** 2.5)
                hiband = fftfilt(nz, sr, lo=3500) * (ramp ** 1.6)
                put("fx", (lowband * 0.7 + hiband * 0.5).astype(np.float32), tb0, bset["riser_gain"] * 0.6, 0.25)
            if bset["pitch_riser"]:
                fr_ = 180 * 2 ** (ramp * 4.0)  # 180 Hz -> 2.9 kHz
                ph = TAU * np.cumsum(fr_) / sr
                sw = (np.sin(ph) + 0.3 * np.sin(2 * ph)).astype(np.float32) * (ramp ** 1.5).astype(np.float32) * 0.22
                put("fx", sw, tb0, 1.0, 0.35)
            if bset["gap_beats"] > 0:
                gap_masks.append((tb1 - bset["gap_beats"] * beat, tb1))
            # sub-bass swell when drums are off (dubstep builds)
            if genre in ("dubstep", "trap") and bset["drums_off"]:
                put("fx", (np.sin(TAU * 0.5 * root * 2 * tt) * ramp * 0.3).astype(np.float32), tb0, 1.0)

    # --- sidechain: pads/bass/lead duck on every kick-ish event. Ghost kicks (4x4) even in breakdowns for 4x4 genres.
    for tt, g in kick_t:
        duck_at(tt, 0.55 * min(1.0, g))
    if genre in FOUR:
        for kind, s, e in truth_secs:
            if kind in ("breakdown", "build") and not bset["kick_through"] or kind == "breakdown":
                for tt in np.arange(max(s, first_bar_t), e, beat):
                    duck_at(float(tt), 0.28)

    # --- mix
    bus = L["kick"] * 1.0 + L["drums"] * 0.9 + L["hats"] * 0.8 + L["perc"] * 0.8
    bus += L["bass"] * duck ** 1.0 + L["pad"] * duck + L["lead"] * (duck ** 0.5) + L["fx"]
    tail = reverb(send * duck ** 0.3, sr, rt=1.4, rng=rng)
    bus += tail * 0.45
    # silence ("gap") right before drops: everything cut, only the reverb tail rings
    if gap_masks:
        mask = np.ones(n, np.float32)
        for a, b in gap_masks:
            s0, s1 = int(a * sr), int(b * sr)
            mask[max(0, s0 - 100): s0] = np.linspace(1, 0, min(100, s0))
            mask[s0:s1] = 0.0
        dry = bus - tail * 0.45
        bus = dry * mask + tail * 0.45 * np.minimum(1, mask + 0.55)

    # --- mastering: slow compression to flatten section loudness, then soft clip / limit
    w = int(0.25 * sr)
    c = np.cumsum(np.abs(bus), dtype=np.float64)
    envl = np.empty(n, np.float32)
    envl[w:] = (c[w:] - c[:-w]) / w
    envl[:w] = c[:w] / (np.arange(w) + 1)
    envl = np.maximum(envl, 0.03)
    comp = spec["comp"]
    gain = (0.12 / envl) ** comp
    gain = np.minimum(gain, 4.0).astype(np.float32)
    # smooth gain (avoid zipper)
    gc = np.cumsum(gain, dtype=np.float64)
    w2 = int(0.05 * sr)
    gs = np.empty(n, np.float32)
    gs[w2:] = (gc[w2:] - gc[:-w2]) / w2
    gs[:w2] = gain[:w2]
    out = bus * gs
    d = spec["drive"]
    out = np.tanh(d * out / max(1e-6, np.percentile(np.abs(out), 99.5))) / np.tanh(d)
    out = fftfilt(out.astype(np.float32), sr, hi=17000, order=2)  # "mp3-ish" top end
    out = out + rng.standard_normal(n).astype(np.float32) * 3e-4  # dither / noise floor
    if spec["silence"] > 0:
        out[: int(spec["silence"] * sr)] = rng.standard_normal(int(spec["silence"] * sr)).astype(np.float32) * 3e-4
    out = out / max(1e-6, np.abs(out).max()) * 0.92

    nbt = int(math.floor((music_end - spec["silence"]) / beat + 1e-9))
    beats = spec["silence"] + np.arange(nbt) * beat
    downbeats = t0 + np.arange(total_bars) * bar
    dur = len(out) / sr
    truth = dict(
        name=spec["name"], genre=genre, bpm=bpm, sections=[(k, round(s, 4), round(min(e, dur), 4)) for k, s, e in truth_secs],
        beats=beats, downbeats=downbeats, duration=dur, spec=spec, first_bar=t0, music_end=music_end,
        drop_starts=drop_starts, gaps=gap_masks,
    )
    return out.astype(np.float32), truth


# --------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--wav", nargs=2, metavar=("NAME", "OUT.wav"))
    a = ap.parse_args()
    if a.list:
        for s in suite_specs():
            print(f"{s['name']:28s} {s['genre']:12s} {s['bpm']:6.1f}  " + " ".join(f"{k[0]}{b}" for k, b in s["structure"]))
    if a.wav:
        import soundfile as sf

        t = time.time()
        audio, tr = make_realistic_track(a.wav[0])
        sf.write(a.wav[1], audio, SR)
        print(f"{len(audio) / SR:.1f}s audio in {time.time() - t:.1f}s")
        for k, s, e in tr["sections"]:
            print(f"  {s:7.2f} {e:7.2f} {k}")
