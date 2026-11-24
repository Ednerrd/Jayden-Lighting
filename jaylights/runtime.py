"""The main show loop: music state -> show engine -> DMX + visualizer."""

from __future__ import annotations

import time
import webbrowser

from .dmx import build_universes, make_output
from .rig import Rig
from .show import Controls, ShowEngine
from .viz.server import VizServer


def run_show(source, rig: Rig, genre: str = "auto", port: int = 8765, open_browser: bool = True,
             fps: float | None = None, title: str = "", max_seconds: float | None = None,
             frame_hook=None):
    engine = ShowEngine(rig.fixtures, genre)
    ctl = Controls(master=rig.master, strobe_enabled=rig.strobe_enabled, max_strobe_hz=rig.max_strobe_hz)
    out = make_output(rig.output)
    fps = fps or float(rig.output.get("fps", 40))

    info = {"mode": source.mode, "title": title, "genre": genre, "output": out.name,
            "fixtures": rig.layout(), "warnings": rig.warnings}
    if source.mode == "file":
        tl = source.timeline
        info["duration"] = tl.duration
        info["sections"] = [{"kind": s["kind"], "start": s["start"], "end": s["end"]} for s in tl.sections]
        info["bpm"] = tl.bpm

    def on_control(msg: dict):
        a, v = msg.get("action"), msg.get("value")
        if a == "blackout":
            ctl.blackout = not ctl.blackout if v is None else bool(v)
        elif a == "strobe":
            ctl.strobe_enabled = not ctl.strobe_enabled if v is None else bool(v)
        elif a == "master":
            ctl.master = max(0.0, min(1.0, float(v)))
        elif a == "movement":
            ctl.movement = max(0.0, min(1.0, float(v)))
        elif a == "pause" and source.mode == "file":
            source.toggle_pause()
        elif a == "seek" and source.mode == "file":
            source.seek(float(v))
        elif source.mode == "live":
            source.control(a, v)

    viz = VizServer(port, info, on_control)
    viz.start()
    url = f"http://localhost:{viz.port}/"
    print(f"\n  Visualizer: {url}")
    print(f"  Output: {out.name}   Fixtures: {len(rig.fixtures)}   Genre: {genre}")
    for w in rig.warnings:
        print(f"  ! {w}")
    print("  Ctrl+C to stop.\n")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    source.start()
    period = 1.0 / fps
    t_start = time.perf_counter()
    last_viz = 0.0
    try:
        while True:
            t0 = time.perf_counter()
            st = source.state()
            states = engine.render(st, ctl)
            unis = build_universes(rig.fixtures, states, t0)
            out.send(unis)
            if t0 - last_viz >= 1 / 30:
                last_viz = t0
                viz.publish({
                    "s": st.to_dict(),
                    "f": [f.effective(s, t0) for f, s in zip(rig.fixtures, states)],
                    "c": {"blackout": ctl.blackout, "strobe": ctl.strobe_enabled, "master": ctl.master,
                          "movement": ctl.movement,
                          "forced": getattr(getattr(source, "analyzer", None), "forced", None)},
                })
            if frame_hook:
                frame_hook(st, states, unis)
            if source.finished() or (max_seconds and t0 - t_start > max_seconds):
                break
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        out.send({u: bytearray(512) for u in {f.universe for f in rig.fixtures} or {0}})  # lights off
        time.sleep(0.1)
        out.close()
        source.stop()
        viz.stop()
        print("\n  Show over. Lights out. ✌")
