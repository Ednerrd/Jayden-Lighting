"""Loads rig.toml: what lights you have, where they're patched, how DMX goes out."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field

from .fixtures import BUILTIN_PROFILES, Fixture, profile_from_dict

EXAMPLE_RIG = r'''# ---------------------------------------------------------------
#  Jaylights rig file
#  List every light, its DMX start address, and its profile.
#  Run `jaylights profiles` to see the built-in profiles.
#  Then `jaylights test-rig --rig rig.toml` to flash each light one by one.
# ---------------------------------------------------------------

[output]
# "enttec_pro"  - Enttec DMX USB Pro (or clones that say "Pro" / "DMXking ultraDMX")
# "open_dmx"    - cheap FTDI USB->DMX dongles (Enttec Open DMX & most $20 Amazon ones)
# "artnet"      - network DMX (Art-Net node / wireless DMX box / other software)
# "none"        - no hardware, just the visualizer
type = "none"
port = "COM3"              # enttec_pro / open_dmx: see Device Manager -> Ports (COM & LPT)
# host = "255.255.255.255" # artnet: node IP (or broadcast)
# universe = 0             # artnet: first universe
fps = 40

[safety]
strobe = true          # false = never strobe (photosensitive guests, kids around)
max_strobe_hz = 10     # cap on strobe speed
master = 1.0           # overall brightness cap 0..1

# ------------------------------ lights ------------------------------
# Optional per-fixture settings:
#   x = 0.0..1.0       position across the room (left->right). Auto-spaced if left out.
#   group = "front"    lights with the same group chase together
#   universe = 0       which DMX universe (Art-Net only, for big rigs)
# Movers also take:
#   pan_center / tilt_center = 0..1   home position (0.5 = middle of the range)
#   pan_span / tilt_span     = 0..1   how far the show may move them from home
#   tilt_up = -1 or 1                 which tilt direction points at the ceiling
#   invert_pan / invert_tilt = true   mirror a fixture (hung the other way round)

[[fixture]]
name = "Par 1"
profile = "rgb_par_7ch"
address = 1

[[fixture]]
name = "Par 2"
profile = "rgb_par_7ch"
address = 8

[[fixture]]
name = "Par 3"
profile = "rgb_par_7ch"
address = 15

[[fixture]]
name = "Par 4"
profile = "rgb_par_7ch"
address = 22

[[fixture]]
name = "Mover L"
profile = "mover_spot_9ch"
address = 40
x = 0.25

[[fixture]]
name = "Mover R"
profile = "mover_spot_9ch"
address = 49
x = 0.75
invert_pan = true

[[fixture]]
name = "Strobe"
profile = "strobe_2ch"
address = 60
x = 0.5

# ------------------------- custom profiles -------------------------
# Got a light that doesn't match? Copy its channel chart from the manual
# (for the DMX mode you set on the fixture) into a profile like this:
#
# [profile.my_bar]
# kind = "wash"                     # wash | mover | strobe | dimmer
# channels = ["dimmer", "red", "green", "blue", "strobe", "mode=0"]
# strobe_range = [10, 255]          # DMX values for slow..fast strobe
# strobe_off = 0                    # value meaning "no strobe" (some movers use 255 = shutter open)
#
# [profile.my_spot]
# kind = "mover"
# channels = ["pan", "tilt", "color_wheel", "gobo=0", "strobe", "dimmer", "speed=0"]
# color_wheel = [[0, "white"], [10, "red"], [20, "green"], [30, "blue"], [40, "yellow"]]
'''

DEMO_RIG = r'''
[output]
type = "none"
[safety]
strobe = true
max_strobe_hz = 10
''' + "\n".join(
    f'[[fixture]]\nname = "Par {i + 1}"\nprofile = "rgbw_par_8ch"\naddress = {1 + i * 8}\ngroup = "front"\ny = 0.0\n'
    for i in range(8)
) + "\n".join(
    f'[[fixture]]\nname = "Mover {i + 1}"\nprofile = "mover_wash_14ch"\naddress = {100 + i * 14}\ngroup = "movers"\n'
    f'x = {0.2 + i * 0.2:.2f}\ny = 0.0\ninvert_pan = {"true" if i % 2 else "false"}\npan_span = 0.12\ntilt_span = 0.18\n'
    for i in range(4)
) + '''
[[fixture]]
name = "Strobe L"
profile = "strobe_2ch"
address = 200
x = 0.1
y = 0.0
[[fixture]]
name = "Strobe R"
profile = "strobe_2ch"
address = 202
x = 0.9
y = 0.0
''' + "\n".join(
    f'[[fixture]]\nname = "Floor {i + 1}"\nprofile = "rgb_par_7ch"\naddress = {300 + i * 7}\ngroup = "floor"\ny = 1.0\n'
    for i in range(6)
)


@dataclass
class Rig:
    fixtures: list[Fixture]
    output: dict = field(default_factory=lambda: {"type": "none"})
    safety: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def strobe_enabled(self) -> bool:
        return bool(self.safety.get("strobe", True))

    @property
    def max_strobe_hz(self) -> float:
        return float(self.safety.get("max_strobe_hz", 10))

    @property
    def master(self) -> float:
        return float(self.safety.get("master", 1.0))

    def layout(self) -> list[dict]:
        return [{"name": f.name, "kind": f.kind, "x": f.x, "y": f.y, "group": f.group,
                 "address": f.address, "universe": f.universe, "profile": f.profile.name}
                for f in self.fixtures]


def load_rig(path: str | None) -> Rig:
    if path is None:
        return parse_rig(DEMO_RIG)
    with open(path, "rb") as f:
        return parse_rig(f.read().decode("utf-8"))


def parse_rig(text: str) -> Rig:
    d = tomllib.loads(text)
    profiles = dict(BUILTIN_PROFILES)
    for name, pd in d.get("profile", {}).items():
        profiles[name] = profile_from_dict(name, pd)

    fixtures: list[Fixture] = []
    known = set(Fixture.__dataclass_fields__) - {"profile", "index", "group_size", "extra"}
    for i, fd in enumerate(d.get("fixture", [])):
        pname = fd.get("profile")
        if pname not in profiles:
            raise ValueError(f"fixture {fd.get('name', i)}: unknown profile '{pname}'. "
                             f"Built-in: {', '.join(BUILTIN_PROFILES)}")
        kwargs = {k: v for k, v in fd.items() if k in known}
        kwargs.setdefault("name", f"{pname} {i + 1}")
        if "address" not in kwargs:
            raise ValueError(f"fixture {kwargs['name']}: missing address")
        fixtures.append(Fixture(profile=profiles[pname], **kwargs,
                                extra={k: v for k, v in fd.items() if k not in known and k != "profile"}))

    # groups: default = one group per fixture kind
    groups: dict[str, list[Fixture]] = {}
    for f in fixtures:
        if not f.group:
            f.group = f.kind
        groups.setdefault(f.group, []).append(f)
    for members in groups.values():
        auto = [f for f in members if f.x is None]
        for j, f in enumerate(auto):
            f.x = (j + 0.5) / len(auto)
        members.sort(key=lambda f: f.x)
        for j, f in enumerate(members):
            f.index, f.group_size = j, len(members)

    warnings = []
    used: dict[tuple[int, int], str] = {}
    for f in fixtures:
        for c in range(f.address, f.address + f.profile.footprint):
            if c > 512:
                warnings.append(f"{f.name}: runs past channel 512")
                break
            if (f.universe, c) in used:
                warnings.append(f"{f.name}: channel {c} overlaps {used[(f.universe, c)]}")
                break
            used[(f.universe, c)] = f.name
    return Rig(fixtures=fixtures, output=d.get("output", {"type": "none"}), safety=d.get("safety", {}),
               warnings=warnings)
