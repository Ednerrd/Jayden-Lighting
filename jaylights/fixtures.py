"""Fixtures: what kind of light it is, and how its DMX channels are laid out.

The show engine thinks in "intents" (FixtureState: brightness, colour, strobe,
pan/tilt). A Profile turns that into the actual DMX bytes for a specific light.

Channel tokens you can use in a custom profile (rig.toml):
    dimmer  red  green  blue  white  amber  uv  strobe
    pan  pan_fine  tilt  tilt_fine  color_wheel
    name=VALUE   -> always send VALUE (e.g. "mode=0", "speed=0", "gobo=0")
    -            -> unused channel, sends 0
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field

WASH, MOVER, STROBE, DIMMER, BLINDER = "wash", "mover", "strobe", "dimmer", "blinder"

# Audience blinders (2/4/8-lite warm white lamps aimed at the crowd) are white-only; the visualizer
# shows them in this warm white whatever colour the engine hands them.
BLINDER_COLOR = (1.0, 0.85, 0.6)

# AIMING CONVENTION (movers): the show engine thinks in "dtilt" -1..+1 around the home position:
#   tilt = tilt_center + dtilt * tilt_span * tilt_up
# `tilt_up` is the direction (+1/-1 of the raw tilt value) that points at the CEILING, so
#   dtilt > 0  -> aim UP (the classic build-up look)
#   dtilt < 0  -> aim at the AUDIENCE (drop impact, crowd_sweep, snare slams); -1 = straight at the crowd
# Home (tilt_center) is the default "out over the room" position.


@dataclass
class FixtureState:
    intensity: float = 0.0
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    strobe_hz: float = 0.0
    pan: float = 0.5      # 0..1 of the fixture's full pan range
    tilt: float = 0.5     # 0..1 of the full tilt range

    def set_color(self, rgb):
        self.r, self.g, self.b = rgb


@dataclass
class Profile:
    name: str
    kind: str
    channels: list[str]
    strobe_range: tuple[int, int] = (10, 255)  # DMX values for slowest..fastest strobe
    strobe_off: int = 0                         # DMX value for "no strobe" (some movers: 255 = open)
    strobe_max_hz: float = 20.0                 # how fast the fixture strobes at the top of the range
    color_wheel: list[tuple[int, tuple[float, float, float]]] | None = None
    description: str = ""

    @property
    def has(self) -> set[str]:
        return {c.split("=")[0] for c in self.channels}

    @property
    def footprint(self) -> int:
        return len(self.channels)


def _hex(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


NAMED_COLORS = {
    "white": "#ffffff", "red": "#ff0000", "orange": "#ff8000", "yellow": "#ffff00",
    "green": "#00ff00", "cyan": "#00ffff", "blue": "#0000ff", "magenta": "#ff00ff",
    "pink": "#ff60c0", "purple": "#8000ff", "uv": "#4000ff", "amber": "#ffb000",
    "lightblue": "#80c0ff", "lime": "#a0ff00",
}

_GENERIC_WHEEL = [(0, "white"), (10, "red"), (20, "orange"), (30, "yellow"), (40, "green"),
                  (50, "blue"), (60, "cyan"), (70, "magenta")]

BUILTIN_PROFILES: dict[str, Profile] = {
    p.name: p for p in [
        Profile("dimmer_1ch", DIMMER, ["dimmer"], description="Single dimmer channel (PAR cans on a dimmer pack, blinders)"),
        Profile("rgb_3ch", WASH, ["red", "green", "blue"], description="Plain RGB, 3 channels (also: one segment of a pixel bar)"),
        Profile("rgb_par_7ch", WASH, ["dimmer", "red", "green", "blue", "strobe", "mode=0", "speed=0"],
                description="Cheap RGB LED par in 7CH mode: dim, R, G, B, strobe, mode, speed"),
        Profile("rgbw_par_8ch", WASH, ["dimmer", "red", "green", "blue", "white", "strobe", "mode=0", "speed=0"],
                description="RGBW LED par in 8CH mode"),
        Profile("rgbwauv_par_10ch", WASH, ["dimmer", "red", "green", "blue", "white", "amber", "uv", "strobe", "mode=0", "speed=0"],
                description="RGBWA+UV (6-in-1) LED par in 10CH mode"),
        Profile("rgbw_4ch", WASH, ["red", "green", "blue", "white"], description="RGBW, 4 channels, no dimmer"),
        Profile("blinder_1ch", BLINDER, ["dimmer"],
                description="Audience blinder (2/4/8-lite warm white), single dimmer channel"),
        Profile("blinder_4ch", BLINDER, ["dimmer", "dimmer", "dimmer", "dimmer"],
                description="4-cell audience blinder, one dimmer per cell (all cells driven together)"),
        Profile("strobe_2ch", STROBE, ["dimmer", "strobe"], description="LED strobe: dimmer + strobe speed"),
        Profile("strobe_1ch", STROBE, ["dimmer"], description="Strobe/blinder with only a dimmer (strobing done in software)"),
        Profile("mover_spot_9ch", MOVER,
                ["pan", "tilt", "color_wheel", "gobo=0", "strobe", "dimmer", "speed=0", "auto=0", "reset=0"],
                strobe_off=255, strobe_range=(16, 250),
                color_wheel=[(v, _hex(NAMED_COLORS[c])) for v, c in _GENERIC_WHEEL],
                description="Cheap mini moving-head spot (colour wheel) in 9CH mode"),
        Profile("mover_spot_11ch", MOVER,
                ["pan", "pan_fine", "tilt", "tilt_fine", "speed=0", "dimmer", "strobe", "color_wheel", "gobo=0", "auto=0", "reset=0"],
                strobe_off=0, strobe_range=(10, 255),
                color_wheel=[(v, _hex(NAMED_COLORS[c])) for v, c in _GENERIC_WHEEL],
                description="Moving-head spot with 16-bit pan/tilt and colour wheel, 11CH mode"),
        Profile("mover_wash_14ch", MOVER,
                ["pan", "pan_fine", "tilt", "tilt_fine", "speed=0", "dimmer", "strobe", "red", "green", "blue", "white", "color_macro=0", "auto=0", "reset=0"],
                description="RGBW moving-head wash/beam, 14CH mode"),
    ]
}


def profile_from_dict(name: str, d: dict) -> Profile:
    wheel = None
    if "color_wheel" in d:
        wheel = []
        for entry in d["color_wheel"]:
            v, c = entry
            rgb = _hex(NAMED_COLORS.get(c, c)) if isinstance(c, str) else tuple(x / 255 for x in c)
            wheel.append((int(v), rgb))
    return Profile(
        name=name,
        kind=d.get("kind", WASH),
        channels=list(d["channels"]),
        strobe_range=tuple(d.get("strobe_range", (10, 255))),
        strobe_off=int(d.get("strobe_off", 0)),
        strobe_max_hz=float(d.get("strobe_max_hz", 20.0)),
        color_wheel=wheel,
        description=d.get("description", "custom"),
    )


@dataclass
class Fixture:
    name: str
    profile: Profile
    address: int              # 1-based DMX start address
    universe: int = 0
    x: float | None = None    # position across the stage, 0 = left, 1 = right
    y: float = 0.0            # 0 = up on the truss, 1 = on the floor (visualizer only)
    group: str = ""
    invert_pan: bool = False
    invert_tilt: bool = False
    pan_center: float = 0.5   # home position (0..1 of full range)
    tilt_center: float = 0.5
    pan_span: float = 0.15    # how far the engine may move it from home
    tilt_span: float = 0.15
    tilt_up: float = -1.0     # which tilt direction points "up" toward the ceiling (+1 or -1)
    index: int = 0            # position in its group (filled in by the rig loader)
    group_size: int = 1
    extra: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.profile.kind

    def render(self, st: FixtureState, t: float) -> list[int]:
        """FixtureState -> DMX channel values for this fixture."""
        p = self.profile
        has = p.has
        inten = max(0.0, min(1.0, st.intensity))
        hw_strobe = "strobe" in has
        if st.strobe_hz > 0 and not hw_strobe:
            # software strobe: chop the intensity (short flashes)
            if (t * st.strobe_hz) % 1.0 > 0.3:
                inten = 0.0
        r, g, b = (max(0.0, min(1.0, c)) for c in (st.r, st.g, st.b))
        if "dimmer" not in has:
            r, g, b = r * inten, g * inten, b * inten
        w = min(r, g, b)
        vals = {
            "dimmer": inten, "red": r, "green": g, "blue": b,
            "white": w * 0.9,
            "amber": max(0.0, min(r, g * 1.4) - b) * 0.6,
            "uv": max(0.0, b - g) * min(1.0, r * 2 + 0.3) * 0.7,
        }
        if "white" in has:
            # let the white LED carry the white part so colours stay saturated
            vals["red"], vals["green"], vals["blue"] = r - w * 0.5, g - w * 0.5, b - w * 0.5
        pan = 1 - st.pan if self.invert_pan else st.pan
        tilt = 1 - st.tilt if self.invert_tilt else st.tilt
        pan16 = int(max(0, min(65535, round(pan * 65535))))
        tilt16 = int(max(0, min(65535, round(tilt * 65535))))

        out = []
        for ch in p.channels:
            if "=" in ch:
                out.append(int(ch.split("=", 1)[1]))
            elif ch == "-":
                out.append(0)
            elif ch == "strobe":
                if st.strobe_hz > 0:
                    lo, hi = p.strobe_range
                    f = min(1.0, st.strobe_hz / p.strobe_max_hz)
                    out.append(int(lo + (hi - lo) * f))
                else:
                    out.append(p.strobe_off)
            elif ch == "pan":
                out.append(pan16 >> 8)
            elif ch == "pan_fine":
                out.append(pan16 & 0xFF)
            elif ch == "tilt":
                out.append(tilt16 >> 8)
            elif ch == "tilt_fine":
                out.append(tilt16 & 0xFF)
            elif ch == "color_wheel":
                out.append(self._wheel_value(st.r, st.g, st.b))
            elif ch in vals:
                out.append(int(round(max(0.0, min(1.0, vals[ch])) * 255)))
            else:
                out.append(0)
        return out

    def _wheel_value(self, r, g, b) -> int:
        wheel = self.profile.color_wheel or [(0, (1.0, 1.0, 1.0))]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        def dist(c):
            h2, s2, _ = colorsys.rgb_to_hsv(*c)
            if s < 0.25 or s2 < 0.25:
                return abs(s - s2) * 2
            dh = min(abs(h - h2), 1 - abs(h - h2))
            return dh * 3 + abs(s - s2)

        return min(wheel, key=lambda e: dist(e[1]))[0]

    def effective(self, st: FixtureState, t: float) -> dict:
        """What this fixture looks like right now (for the visualizer)."""
        inten = max(0.0, min(1.0, st.intensity))
        if st.strobe_hz > 0 and (t * st.strobe_hz) % 1.0 > 0.3:
            inten = 0.0
        r, g, b = st.r, st.g, st.b
        if self.kind == BLINDER:
            r, g, b = BLINDER_COLOR
        elif self.profile.color_wheel:
            v = self._wheel_value(r, g, b)
            r, g, b = next(c for val, c in self.profile.color_wheel if val == v)
        elif not ({"red", "green", "blue"} & self.profile.has):
            r = g = b = 1.0  # dimmer / white strobe
        return {"i": round(inten, 3), "r": round(r, 3), "g": round(g, 3), "b": round(b, 3),
                "pan": round(st.pan, 4), "tilt": round(st.tilt, 4), "strobe": round(st.strobe_hz, 2)}


def hsv(h_deg: float, s: float = 1.0, v: float = 1.0) -> tuple[float, float, float]:
    return colorsys.hsv_to_rgb((h_deg % 360) / 360.0, s, v)


def mix(a, b, f: float):
    f = max(0.0, min(1.0, f))
    return tuple(x + (y - x) * f for x, y in zip(a, b))


def wave(x: float) -> float:
    """0..1 sine wave with period 1."""
    return 0.5 + 0.5 * math.sin(2 * math.pi * x)
