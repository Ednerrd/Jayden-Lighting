"""Run with: python -m pytest -q"""

import numpy as np
import pytest

from jaylights.analysis.live import LiveAnalyzer
from jaylights.analysis.offline import analyze_audio
from jaylights.dmx import ArtNet, EnttecPro, build_universes
from jaylights.fixtures import BUILTIN_PROFILES, Fixture, FixtureState
from jaylights.rig import EXAMPLE_RIG, load_rig, parse_rig
from jaylights.show import DROP_ROTATION, DROP_STYLES, GENRES, Controls, ShowEngine
from jaylights.state import BUILD, DROP
from jaylights.synth import SR, make_track


@pytest.fixture(scope="module", params=[(128, False), (174, False), (140, True)], ids=["house128", "dnb174", "dubstep140"])
def track(request):
    bpm, halftime = request.param
    audio, sections = make_track(bpm=bpm, halftime=halftime)
    return bpm, halftime, audio, sections


def _starts(sections, kind):
    return [s for k, s, _ in sections if k == kind]


def test_offline_finds_tempo_builds_and_drops(track):
    bpm, _, audio, truth = track
    tl = analyze_audio(audio, SR)
    assert abs(tl.bpm - bpm) < 0.3
    for kind in (BUILD, DROP):
        got = [s["start"] for s in tl.sections if s["kind"] == kind]
        want = _starts(truth, kind)
        assert len(got) == len(want), (kind, got, want)
        assert np.allclose(got, want, atol=0.1), (kind, got, want)


def test_offline_beats_are_on_the_kick(track):
    bpm, _, audio, _ = track
    tl = analyze_audio(audio, SR)
    grid = np.arange(0, len(audio) / SR, 60 / bpm)
    err = [np.min(np.abs(grid - b)) for b in tl.beats]
    assert np.median(err) < 0.015


def test_live_catches_drops_fast_and_stays_in_phase(track):
    bpm, _, audio, truth = track
    la = LiveAnalyzer(SR)
    drops_seen, last, errs = [], None, []
    drops = [(s, e) for k, s, e in truth if k == DROP]
    for i in range(0, len(audio), 512):
        st = la.push(audio[i:i + 512])
        t = (i + 512) / SR
        if st.section == DROP and last != DROP:
            drops_seen.append(t)
        last = st.section
        tc = t - 1024 / SR
        if any(s + 4 < tc < e for s, e in drops):
            errs.append(((st.beat - tc * bpm / 60 + 0.5) % 1 - 0.5) * 60 / bpm)
    for want in _starts(truth, DROP):
        assert any(0 <= got - want < 0.6 for got in drops_seen), (want, drops_seen)
    assert abs(np.median(errs)) < 0.025  # lights within 25 ms of the beat


def test_enttec_pro_packet():
    pkt = EnttecPro.packet(bytes([1, 2, 3]))
    assert pkt[0] == 0x7E and pkt[1] == 6 and pkt[-1] == 0xE7
    assert pkt[2] | (pkt[3] << 8) == 513
    assert pkt[4] == 0 and pkt[5:8] == bytes([1, 2, 3])


def test_artnet_packet():
    pkt = ArtNet.packet(3, bytes(512), seq=7)
    assert pkt[:8] == b"Art-Net\x00"
    assert pkt[8:10] == b"\x00\x50" and pkt[12] == 7
    assert pkt[14] == 3 and pkt[16:18] == b"\x02\x00"


def test_fixture_rendering():
    par = Fixture("p", BUILTIN_PROFILES["rgb_par_7ch"], address=1)
    assert par.render(FixtureState(intensity=1, r=1, g=0, b=0.5), 0) == [255, 255, 0, 128, 0, 0, 0]
    assert par.render(FixtureState(intensity=1, strobe_hz=20), 0)[4] == 255

    mover = Fixture("m", BUILTIN_PROFILES["mover_wash_14ch"], address=1)
    v = mover.render(FixtureState(intensity=1, pan=0.5, tilt=1.0), 0)
    assert v[0] == 128 and v[1] == 0 and v[2] == 255 and v[3] == 255  # 16-bit pan/tilt

    spot = Fixture("s", BUILTIN_PROFILES["mover_spot_9ch"], address=1)
    assert spot.render(FixtureState(intensity=1, r=0, g=0, b=1), 0)[2] == 50   # blue slot
    assert spot.render(FixtureState(intensity=1, r=1, g=1, b=1), 0)[2] == 0    # white slot
    assert spot.render(FixtureState(intensity=1), 0)[4] == 255                 # shutter open

    # no dimmer channel: intensity scales RGB; no strobe channel: software strobe
    rgb = Fixture("r", BUILTIN_PROFILES["rgb_3ch"], address=1)
    assert rgb.render(FixtureState(intensity=0.5, r=1, g=1, b=1), 0) == [128, 128, 128]
    assert rgb.render(FixtureState(intensity=1, strobe_hz=10), 0.05) == [0, 0, 0]


def test_rig_files():
    rig = parse_rig(EXAMPLE_RIG)
    assert len(rig.fixtures) == 7 and not rig.warnings
    demo = load_rig(None)
    assert not demo.warnings
    custom = parse_rig("""
[profile.bar]
kind = "wash"
channels = ["red", "green", "blue", "mode=42"]
[[fixture]]
profile = "bar"
address = 1
[[fixture]]
profile = "bar"
address = 3
""")
    assert custom.warnings  # overlapping addresses
    u = build_universes(custom.fixtures[:1], [FixtureState(intensity=1)], 0)
    assert list(u[0][:4]) == [255, 255, 255, 42]


def test_show_engine_whole_track():
    audio, _ = make_track(bpm=128)
    tl = analyze_audio(audio, SR)
    rig = load_rig(None)
    eng = ShowEngine(rig.fixtures, "house")
    ctl = Controls()
    drop_t = next(s["start"] for s in tl.sections if s["kind"] == DROP)
    for t in np.arange(0, tl.duration, 0.05):
        states = eng.render(tl.state_at(float(t)), ctl)
        for s in states:
            assert 0 <= s.intensity <= 1 and 0 <= s.pan <= 1 and 0 <= s.tilt <= 1
            assert s.strobe_hz <= ctl.max_strobe_hz
        if drop_t - 0.3 < t < drop_t - 0.05:
            assert all(s.intensity == 0 for s in states), "blackout right before the drop"
        if drop_t + 0.02 < t < drop_t + 0.1:
            assert max(s.intensity for s in states) > 0.85, "impact on the drop"
    ctl.blackout = True
    assert all(s.intensity == 0 for s in eng.render(tl.state_at(drop_t + 1), ctl))


@pytest.mark.parametrize("style", DROP_STYLES)
@pytest.mark.parametrize("rig_kinds", [None, "wash", "mover", "strobe"])
def test_every_drop_style_renders_a_whole_track(style, rig_kinds):
    audio, _ = make_track(bpm=128)
    tl = analyze_audio(audio, SR)
    fixtures = load_rig(None).fixtures
    if rig_kinds:                       # degrade gracefully: rigs with only one kind of light
        fixtures = [f for f in fixtures if f.kind == rig_kinds]
        assert fixtures
    eng = ShowEngine(fixtures, "dubstep")
    ctl = Controls(drop_style=style, max_strobe_hz=8.0)
    drop_t = next(s["start"] for s in tl.sections if s["kind"] == DROP)
    drop_end = next(s["end"] for s in tl.sections if s["kind"] == DROP)
    peak = 0.0
    for t in np.arange(0, tl.duration, 0.05):
        states = eng.render(tl.state_at(float(t)), ctl)
        for s in states:
            assert 0 <= s.intensity <= 1 and 0 <= s.pan <= 1 and 0 <= s.tilt <= 1
            assert 0 <= s.strobe_hz <= ctl.max_strobe_hz
            assert all(0 <= c <= 1 for c in (s.r, s.g, s.b))
        if drop_t - 0.3 < t < drop_t - 0.05:
            assert all(s.intensity == 0 for s in states), "blackout right before the drop"
        if drop_t + 0.02 < t < drop_t + 0.1:
            assert max(s.intensity for s in states) > (0.7 if rig_kinds == "strobe" else 0.85), "impact on the drop"
        if drop_t + 1.0 < t < drop_end:
            peak = max(peak, max(s.intensity for s in states))
    assert peak > 0.5, "style does something strong on this rig"


def test_drop_style_rotation_and_lock():
    for genre in GENRES:
        eng = ShowEngine(load_rig(None).fixtures, genre)
        assert set(DROP_ROTATION[genre]) <= set(DROP_STYLES)
        seen = {eng.drop_style_at(sb) for sb in range(0, 32 * len(DROP_ROTATION[genre]), 32)}
        assert seen == set(DROP_ROTATION[genre])
        assert eng.drop_style_at(70, "sweep") == "sweep"
        assert eng.drop_style_at(70, "nonsense") in DROP_ROTATION[genre]


def test_strobe_safety_holds_for_every_style():
    audio, _ = make_track(bpm=174)
    tl = analyze_audio(audio, SR)
    drop_t = next(s["start"] for s in tl.sections if s["kind"] == DROP)
    for style in DROP_STYLES:
        eng = ShowEngine(load_rig(None).fixtures, "dnb")
        ctl = Controls(drop_style=style, strobe_enabled=False)
        for t in np.arange(drop_t - 1, drop_t + 6, 0.02):
            assert all(s.strobe_hz == 0 for s in eng.render(tl.state_at(float(t)), ctl))


# ---------------------------------------------------------------- audience blinders + crowd aiming
def _blinders(fixtures):
    return [i for i, f in enumerate(fixtures) if f.kind == "blinder"]


def test_blinder_profiles_and_effective_colour():
    rig = load_rig(None)
    bl = [f for f in rig.fixtures if f.kind == "blinder"]
    assert len(bl) == 2 and sorted(f.x for f in bl) == [0.3, 0.7]
    for name in ("blinder_1ch", "blinder_4ch"):
        assert BUILTIN_PROFILES[name].kind == "blinder"
    four = Fixture("b", BUILTIN_PROFILES["blinder_4ch"], address=1)
    assert four.render(FixtureState(intensity=1, r=1, g=0, b=0), 0) == [255, 255, 255, 255]
    e = bl[0].effective(FixtureState(intensity=1, r=1, g=0, b=0), 0)
    assert (e["r"], e["g"], e["b"]) == (1.0, 0.85, 0.6)


@pytest.mark.parametrize("style", DROP_STYLES)
def test_blinders_valid_and_dark_outside_drops(style):
    audio, _ = make_track(bpm=128)
    tl = analyze_audio(audio, SR)
    rig = load_rig(None)
    bi = _blinders(rig.fixtures)
    eng = ShowEngine(rig.fixtures, "house")
    ctl = Controls(drop_style=style)
    drop_t = next(s["start"] for s in tl.sections if s["kind"] == DROP)
    calm, swell, impact = [], [], []
    for t in np.arange(0, tl.duration, 0.05):
        st = tl.state_at(float(t))
        states = eng.render(st, ctl)
        for i in bi:
            s = states[i]
            assert 0 <= s.intensity <= 1 and s.strobe_hz == 0
            assert (s.r, s.g, s.b) == (1.0, 0.85, 0.6)
            if st.section in ("intro", "groove", "breakdown", "outro"):
                calm.append(s.intensity)
            if st.section == BUILD and st.beats_to_next is not None:
                if st.beats_to_next <= 1:
                    assert s.intensity == 0, "dark on the final pre-drop beat"
                elif st.beats_to_next <= 4:
                    swell.append(s.intensity)
                else:
                    assert s.intensity == 0
            if drop_t + 0.0 < t < drop_t + 0.06:
                impact.append(s.intensity)
    assert calm and np.mean(calm) < 0.1
    assert swell and 0 < max(swell) <= 0.5
    assert impact and min(impact) > 0.85, "full blinder hit on the drop impact"


def test_blinders_flag_blackout_and_strobe_independence():
    audio, _ = make_track(bpm=128)
    tl = analyze_audio(audio, SR)
    rig = load_rig(None)
    bi = _blinders(rig.fixtures)
    drop_t = next(s["start"] for s in tl.sections if s["kind"] == DROP)
    st = tl.state_at(drop_t + 0.05)
    assert max(ShowEngine(rig.fixtures, "house").render(st, Controls())[i].intensity for i in bi) > 0.85
    # strobes disabled must NOT disable blinders
    assert max(ShowEngine(rig.fixtures, "house").render(st, Controls(strobe_enabled=False))[i].intensity for i in bi) > 0.85
    off = Controls(blinders=False)
    eng = ShowEngine(rig.fixtures, "house")
    for t in np.arange(drop_t - 5, drop_t + 6, 0.05):
        assert all(eng.render(tl.state_at(float(t)), off)[i].intensity == 0 for i in bi)
    assert all(ShowEngine(rig.fixtures, "house").render(st, Controls(blackout=True))[i].intensity == 0 for i in bi)
    assert all(abs(ShowEngine(rig.fixtures, "house").render(st, Controls(master=0.5))[i].intensity - 0.5 * ShowEngine(rig.fixtures, "house").render(st, Controls())[i].intensity) < 1e-9 for i in bi)


def test_live_mode_has_no_blinder_swell():
    from jaylights.state import MusicState
    rig = load_rig(None)
    eng = ShowEngine(rig.fixtures, "house")
    for beat in np.arange(0, 32, 0.25):
        st = MusicState(section=BUILD, section_index=0, beat=beat, beat_in_bar=beat % 4, section_progress=beat / 32,
                        next_section=None, beats_to_next=None)
        assert all(eng.render(st, Controls())[i].intensity == 0 for i in _blinders(rig.fixtures))


def test_crowd_sweep_blinders_hit_downbeats():
    from jaylights.state import MusicState
    rig = load_rig(None)
    eng = ShowEngine(rig.fixtures, "house")
    mk = lambda bib, kick: MusicState(section=DROP, section_index=1, section_beats=10 + bib, beat=10 + bib,
                                      beat_in_bar=bib, bar=2, kick=kick)
    ctl = Controls(drop_style="crowd_sweep")
    on = eng.render(mk(0.0, 1.0), ctl)
    off = eng.render(mk(2.0, 0.0), ctl)
    for i in _blinders(rig.fixtures):
        assert abs(on[i].intensity - 0.6) < 1e-9 and off[i].intensity == 0
    for i, f in enumerate(rig.fixtures):       # movers are aimed at the crowd (dtilt <= -0.8)
        if f.kind == "mover":
            assert (on[i].tilt - f.tilt_center) * f.tilt_up < 0


def test_movers_snap_to_the_crowd_on_drop_impact_and_aim_up_in_builds():
    audio, _ = make_track(bpm=128)
    tl = analyze_audio(audio, SR)
    rig = load_rig(None)
    movers = [(i, f) for i, f in enumerate(rig.fixtures) if f.kind == "mover"]
    drop_t = next(s["start"] for s in tl.sections if s["kind"] == DROP)
    for style in DROP_STYLES:
        eng = ShowEngine(rig.fixtures, "house")
        ctl = Controls(drop_style=style)
        for t in np.arange(drop_t - 2, drop_t + 0.45, 0.05):
            states = eng.render(tl.state_at(float(t)), ctl)
            if t < drop_t - 0.4:
                for i, f in movers:   # building: aimed up
                    assert (states[i].tilt - f.tilt_center) * f.tilt_up > 0
        for i, f in movers:           # ~0.4 s into the impact: crowd side of home
            assert (states[i].tilt - f.tilt_center) * f.tilt_up < 0, style


def test_crowd_sweep_in_rotations_and_no_blinders_rig_safety():
    for genre in ("auto", "house", "progressive", "trance", "techno", "dubstep", "trap"):
        assert "crowd_sweep" in DROP_ROTATION[genre]
    rig = parse_rig(EXAMPLE_RIG + "\n[[fixture]]\nname='B'\nprofile='blinder_1ch'\naddress=70\n")
    assert rig.blinders_enabled
    rig.safety["blinders"] = False
    assert not rig.blinders_enabled


# ---------------------------------------------------------------- real tracks
# Per-bar features pasted from the web page's "Copy analysis" button (values x100),
# with the structure the user heard. Keeps real-world fixes from regressing.
import os as _os


def _load_bars(name):
    m = {}
    with open(_os.path.join(_os.path.dirname(__file__), "real_tracks", name)) as f:
        for line in f:
            k, v = line.split(":")
            m[k.strip()] = np.array([float(x) for x in v.split(",")]) / 100
    return m


def test_real_not_even_love_builds():
    from jaylights.analysis.sections import label_sections

    t0, bar = 0.27, 1.846  # first downbeat (s), bar length (s) at 130 BPM
    secs = label_sections(_load_bars("not_even_love.txt"))
    builds = [t0 + s.start_bar * bar for s in secs if s.kind == "build"]
    drops = [t0 + s.start_bar * bar for s in secs if s.kind == "drop"]
    # user: build 1:08 -> drop 1:15 ; build 2:15 -> drop 2:29
    assert any(abs(b - 68) < 2 for b in builds), builds
    assert any(abs(b - 134) < 2 for b in builds), builds
    assert any(abs(d - 75) < 2 for d in drops), drops
    assert any(abs(d - 149) < 2 for d in drops), drops
    # user: 1:45 -> 2:15 is the groove (bass dips, kick keeps going), not a drop/breakdown
    for s in secs:
        a, b = t0 + s.start_bar * bar, t0 + s.end_bar * bar
        if a < 133 and b > 107:  # overlaps 1:47 - 2:13
            assert s.kind == "groove", (s, round(a), round(b))
