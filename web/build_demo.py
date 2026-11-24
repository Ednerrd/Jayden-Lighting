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

meta = {"fps": FPS, "fixtures": rig.layout(), "sections": list(SECTIONS), "tracks": {}}
for key, bpm, ht in (("house", 126, False), ("dubstep", 140, True)):
    audio, truth = make_track(bpm=bpm, halftime=ht)
    dur = len(audio) / SR
    sf.write(f"{OUT}/{key}.mp3", audio, SR, format="MP3")
    tl = analyze_audio(audio, SR)
    meta["tracks"][key] = {
        "bpm": bpm, "duration": round(dur, 3), "frames": int(dur * FPS),
        "truth": [{"kind": k, "start": round(s, 2), "end": round(e, 2)} for k, s, e in truth],
        # the page replaces these with its in-browser analysis once the track is loaded
        "offline_sections": [{"kind": s["kind"], "start": s["start"], "end": s["end"]} for s in tl.sections],
        "live_sections": [],
        "detected_bpm": round(tl.bpm, 2),
    }
    print(key, 'bpm', round(tl.bpm, 2), 'offline', [(s['kind'][:5], s['start']) for s in tl.sections])
with open(f"{OUT}/meta.json", "w") as f:
    json.dump(meta, f)
