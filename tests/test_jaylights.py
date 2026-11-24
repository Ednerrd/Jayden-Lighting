"""Run with: python -m pytest -q"""

import numpy as np
import pytest

from jaylights.analysis.live import LiveAnalyzer
from jaylights.analysis.offline import analyze_audio
from jaylights.dmx import ArtNet, EnttecPro, build_universes
from jaylights.fixtures import BUILTIN_PROFILES, Fixture, FixtureState
from jaylights.rig import EXAMPLE_RIG, load_rig, parse_rig
from jaylights.show import Controls, ShowEngine
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
