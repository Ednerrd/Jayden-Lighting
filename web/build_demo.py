"""Builds the demo data for web/: two synthetic tracks (mp3) + meta.json with their known structure.
The page analyzes the mp3s in the browser (engine.js) and renders the show from the music states,
so the drop style can be changed live. Usage: python web/build_demo.py web
"""
import sys, json
import numpy as np, soundfile as sf
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from jaylights.synth import make_track, SR
from jaylights.analysis.offline import analyze_audio
from jaylights.rig import load_rig
from jaylights.state import SECTIONS

OUT = sys.argv[1]
FPS = 30
rig = load_rig(None)

sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))), "tools"))
import stress_synth  # noqa: E402  (realistic trap/dubstep test tracks with ground truth)

meta = {"fps": FPS, "fixtures": rig.layout(), "sections": list(SECTIONS), "tracks": {}}


def clean_house():
    audio, truth = make_track(bpm=126)
    return audio, 126, truth


def stress(name):
    audio, truth = stress_synth.make_realistic_track(name)
    return audio, truth["bpm"], [tuple(x) for x in truth["sections"]]


for key, maker in (("house", clean_house), ("dubstep", lambda: stress("dubstep_halftime")),
                   ("trap", lambda: stress("trap_808_140"))):
    audio, bpm, truth = maker()
    dur = len(audio) / SR
    sf.write(f"{OUT}/{key}.mp3", audio, SR, format="MP3")
    tl = analyze_audio(audio, SR)
    meta["tracks"][key] = {
        "bpm": bpm, "duration": round(dur, 3), "frames": int(dur * FPS),
        "truth": [{"kind": k, "start": round(s, 2), "end": round(e, 2)} for k, s, e in truth],
        "offline_sections": [{"kind": s["kind"], "start": s["start"], "end": s["end"]} for s in tl.sections],
        "live_sections": [],
        "detected_bpm": round(tl.bpm, 2),
    }
    print(key, round(dur), 's bpm', round(tl.bpm, 2), 'offline', [(s['kind'][:5], round(s['start'], 1)) for s in tl.sections])
with open(f"{OUT}/meta.json", "w") as f:
    json.dump(meta, f)
