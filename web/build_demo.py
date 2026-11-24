"""Builds the demo data for web/: two synthetic tracks + their rendered shows.
Usage: python web/build_demo.py web
"""
import sys, json, base64, dataclasses
import numpy as np, soundfile as sf
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from jaylights.synth import make_track, SR
from jaylights.analysis.offline import analyze_audio
from jaylights.analysis.live import LiveAnalyzer
from jaylights.rig import load_rig
from jaylights.show import ShowEngine, Controls
from jaylights.state import SECTIONS

OUT = sys.argv[1]
FPS = 30
rig = load_rig(None)
SEC = {k: i for i, k in enumerate(SECTIONS)}

def pack(states_seq):
    fx, st = bytearray(), bytearray()
    for t, ms, fstates in states_seq:
        for f, s in zip(rig.fixtures, fstates):
            e = f.effective(s, t)
            fx += bytes([round(e['i']*255), round(e['r']*255), round(e['g']*255), round(e['b']*255),
                         round(min(1, max(0, e['pan']))*255), round(min(1, max(0, e['tilt']))*255)])
        bpm10 = int(round(ms.bpm * 10))
        st += bytes([SEC.get(ms.section, 1), int(ms.beat_in_bar * 60) % 240, bpm10 >> 8, bpm10 & 255,
                     round(ms.section_progress*255), round(min(1,ms.kick)*255), round(min(1,ms.snare)*255),
                     round(min(1,ms.hat)*255), round(ms.low*255), round(ms.mid*255), round(ms.high*255),
                     round(ms.energy*255), int(ms.locked)])
    return base64.b64encode(bytes(fx)).decode(), base64.b64encode(bytes(st)).decode()

meta = {"fps": FPS, "fixtures": rig.layout(), "sections": list(SECTIONS), "tracks": {}}
for key, bpm, ht, genre in (("house", 126, False, "house"), ("dubstep", 140, True, "dubstep")):
    audio, truth = make_track(bpm=bpm, halftime=ht)
    dur = len(audio) / SR
    sf.write(f"{OUT}/{key}.mp3", audio, SR, format="MP3")
    tl = analyze_audio(audio, SR)
    times = np.arange(0, dur, 1 / FPS)
    # pre-analyzed
    eng = ShowEngine(rig.fixtures, genre); ctl = Controls()
    seq = []
    for t in times:
        ms = tl.state_at(float(t) + 0.03); ms.t = float(t)
        seq.append((float(t), ms, eng.render(ms, ctl)))
    fx_o, st_o = pack(seq)
    # live (fed in real-time-sized blocks, sampled at each frame time)
    la = LiveAnalyzer(SR, (134, 156) if ht else (115, 132), latency=0.03)
    eng = ShowEngine(rig.fixtures, genre); seq = []; pos = 0
    live_secs = []; last = None
    for t in times:
        target = int(t * SR) + 1024
        while pos + 512 <= min(target, len(audio)):
            la.push(audio[pos:pos+512]); pos += 512
        ms = dataclasses.replace(la.state); ms.t = float(t)
        if ms.section != last:
            live_secs.append({"kind": ms.section, "start": round(float(t), 2)}); last = ms.section
        seq.append((float(t), ms, eng.render(ms, ctl)))
    fx_l, st_l = pack(seq)
    for i, s in enumerate(live_secs):
        s["end"] = live_secs[i+1]["start"] if i + 1 < len(live_secs) else round(dur, 2)
    meta["tracks"][key] = {
        "bpm": bpm, "duration": round(dur, 3), "frames": len(times),
        "truth": [{"kind": k, "start": round(s, 2), "end": round(e, 2)} for k, s, e in truth],
        "offline_sections": [{"kind": s["kind"], "start": s["start"], "end": s["end"]} for s in tl.sections],
        "live_sections": live_secs,
        "detected_bpm": round(tl.bpm, 2),
    }
    with open(f"{OUT}/{key}_show.json", "w") as f:
        json.dump({"pre": {"fx": fx_o, "st": st_o}, "live": {"fx": fx_l, "st": st_l}}, f)
    print(key, 'bpm', round(tl.bpm,2), 'offline', [(s['kind'][:5], s['start']) for s in tl.sections])
    print('   live', [(s['kind'][:5], s['start']) for s in live_secs])
with open(f"{OUT}/meta.json", "w") as f:
    json.dump(meta, f)
