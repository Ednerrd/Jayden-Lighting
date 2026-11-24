"""The shared description of "what the music is doing right now".

Both the offline (pre-analyzed) and live analyzers produce a MusicState, and the
show engine only ever looks at a MusicState. That way every look works the same
in both modes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

# Section kinds, in rough order of intensity.
INTRO = "intro"
GROOVE = "groove"
BREAKDOWN = "breakdown"
BUILD = "build"
DROP = "drop"
OUTRO = "outro"

SECTIONS = (INTRO, GROOVE, BREAKDOWN, BUILD, DROP, OUTRO)


@dataclass
class MusicState:
    t: float = 0.0                 # seconds
    bpm: float = 128.0
    beat: float = 0.0              # continuous beat count since start (fraction = phase)
    beat_in_bar: float = 0.0       # 0..4, 0 = downbeat
    bar: int = 0                   # bar counter
    section: str = GROOVE
    section_index: int = 0         # increments every time the section changes
    section_beats: float = 0.0     # beats since this section started
    section_progress: float = 0.0  # 0..1 through the section (builds: how close the drop is)
    next_section: str | None = None
    beats_to_next: float | None = None   # known in pre-analyzed mode only
    energy: float = 0.0            # overall loudness 0..1 (relative to the track)
    low: float = 0.0               # kick + bass 0..1
    mid: float = 0.0
    high: float = 0.0
    kick: float = 0.0              # hit envelopes: 1.0 right on the hit, decaying to 0
    snare: float = 0.0
    hat: float = 0.0
    locked: bool = True            # live mode: is the beat tracker confident?
    playing: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if hasattr(v, "item"):  # numpy scalar -> plain python
                v = v.item()
            if isinstance(v, float):
                v = round(v, 4)
            d[k] = v
        return d
