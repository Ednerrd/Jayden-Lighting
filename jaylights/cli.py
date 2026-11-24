"""Command line: `jaylights <command>`. Run `jaylights -h` for help."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

from .show import DROP_STYLES, GENRES


def _genre_range(args):
    if getattr(args, "bpm_range", None):
        lo, hi = args.bpm_range
        return (float(lo), float(hi))
    return GENRES[args.genre][0]


def cmd_analyze(args):
    from .analysis.offline import analyze_file

    for path in args.files:
        t0 = time.time()
        tl = analyze_file(path, _genre_range(args), use_cache=not args.fresh)
        print(f"\n{os.path.basename(path)}  ({time.time() - t0:.1f}s)")
        print(tl.summary())
        if args.out:
            tl.save(args.out)
            print(f"saved {args.out}")


def _rig(args):
    from .rig import load_rig

    rig = load_rig(args.rig)
    if args.output:
        rig.output["type"] = args.output
    if args.port_name:
        rig.output["port"] = args.port_name
    if args.no_strobe:
        rig.safety["strobe"] = False
    return rig


def cmd_play(args):
    from .analysis.offline import analyze_file
    from .runtime import run_show
    from .sources import FileSource

    print(f"Analyzing {args.file} ...")
    tl = analyze_file(args.file, _genre_range(args), use_cache=not args.fresh)
    print(tl.summary())
    src = FileSource(args.file, tl, play_audio=not args.no_audio, start=args.start,
                     light_offset=args.offset / 1000)
    run_show(src, _rig(args), args.genre, args.http_port, not args.no_browser, title=os.path.basename(args.file),
             max_seconds=args.seconds, drop_style=args.drop_style)


def cmd_live(args):
    from .runtime import run_show
    from .sources import LiveSource

    src = LiveSource(device=args.device, loopback=args.loopback, sim_file=args.file,
                     bpm_range=_genre_range(args), latency=args.offset / 1000, play_sim=not args.no_audio)
    title = "simulated live: " + os.path.basename(args.file) if args.file else ("loopback" if args.loopback else f"input {args.device or 'default'}")
    run_show(src, _rig(args), args.genre, args.http_port, not args.no_browser, title=title, max_seconds=args.seconds,
             drop_style=args.drop_style)


def cmd_demo(args):
    from .synth import make_track, SR
    import soundfile as sf

    halftime = args.genre in ("dubstep", "trap")
    bpm = {"dubstep": 140, "trap": 145, "dnb": 174, "techno": 132, "trance": 138}.get(args.genre, 126)
    path = os.path.join(tempfile.gettempdir(), f"jaylights_demo_{args.genre}_{bpm}.wav")
    if not os.path.exists(path):
        print("Synthesizing a demo track ...")
        audio, _ = make_track(bpm=bpm, halftime=halftime)
        sf.write(path, audio, SR)
    args.file = path
    if args.live:
        args.device, args.loopback = None, False
        cmd_live(args)
    else:
        args.start, args.fresh = 0.0, False
        cmd_play(args)


def cmd_devices(_args):
    try:
        import sounddevice as sd

        print("Audio inputs (use the number with --device):")
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"  [{i}] {d['name']}  ({sd.query_hostapis(d['hostapi'])['name']})")
    except (ImportError, OSError) as e:
        print(f"  audio: unavailable ({e})")
    try:
        import soundcard as sc

        print("\nLoopback-able outputs (use with --loopback --device NAME):")
        for s in sc.all_speakers():
            print(f"  {s.name}")
    except Exception:  # noqa: BLE001
        print("\n(install 'soundcard' for loopback: pip install soundcard)")
    try:
        from serial.tools import list_ports

        print("\nSerial ports (DMX dongles show up here, e.g. 'USB Serial Port (COM3)'):")
        for p in list_ports.comports():
            print(f"  {p.device}  {p.description}")
    except ImportError:
        pass


def cmd_profiles(_args):
    from .fixtures import BUILTIN_PROFILES

    for p in BUILTIN_PROFILES.values():
        print(f"\n{p.name}  [{p.kind}, {p.footprint} ch]  {p.description}")
        print("   " + "  ".join(f"{i + 1}:{c}" for i, c in enumerate(p.channels)))


def cmd_init_rig(args):
    from .rig import EXAMPLE_RIG

    if os.path.exists(args.path) and not args.force:
        sys.exit(f"{args.path} already exists (use --force to overwrite)")
    with open(args.path, "w", encoding="utf-8") as f:
        f.write(EXAMPLE_RIG)
    print(f"wrote {args.path} - edit it to match your lights")


def cmd_test_rig(args):
    """Flash each fixture in turn so you can check the patch."""
    from .dmx import build_universes, make_output
    from .fixtures import MOVER, FixtureState

    rig = _rig(args)
    out = make_output(rig.output)
    print(f"Output: {out.name}. Each light: white, red, green, blue (+ movers sweep). Ctrl+C to stop.")
    try:
        while True:
            for f in rig.fixtures:
                print(f"  -> {f.name:<16} {f.profile.name:<18} addr {f.address}" + (f" uni {f.universe}" if f.universe else ""))
                for color, dur in (((1, 1, 1), 1.0), ((1, 0, 0), 0.6), ((0, 1, 0), 0.6), ((0, 0, 1), 0.6)):
                    t_end = time.time() + dur
                    while time.time() < t_end:
                        states = [FixtureState(intensity=0.0, pan=x.pan_center, tilt=x.tilt_center) for x in rig.fixtures]
                        s = states[rig.fixtures.index(f)]
                        s.intensity, (s.r, s.g, s.b) = 1.0, color
                        if f.kind == MOVER:
                            ph = 1 - (t_end - time.time()) / dur
                            s.pan = f.pan_center + (ph - 0.5) * 2 * f.pan_span
                        out.send(build_universes(rig.fixtures, states, time.time()))
                        time.sleep(1 / 30)
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        out.send({u: bytearray(512) for u in {f.universe for f in rig.fixtures} or {0}})
        out.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jaylights", description="Auto light shows from music. EDM-tuned.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, show=True):
        p.add_argument("--genre", choices=list(GENRES), default="auto",
                       help="tunes the tempo range + look (dubstep/trap = half-time aware)")
        p.add_argument("--bpm-range", nargs=2, metavar=("LO", "HI"), help="override the tempo search range")
        if show:
            p.add_argument("--rig", help="rig.toml (default: built-in virtual demo rig)")
            p.add_argument("--output", help="override output type: enttec_pro | open_dmx | artnet | none")
            p.add_argument("--port-name", help="override serial port, e.g. COM4")
            p.add_argument("--http-port", type=int, default=8765, help="visualizer port")
            p.add_argument("--no-browser", action="store_true")
            p.add_argument("--no-strobe", action="store_true", help="disable strobing entirely")
            p.add_argument("--no-audio", action="store_true", help="don't play sound (show runs on the clock)")
            p.add_argument("--offset", type=float, default=30, help="fire lights this many ms early (DMX/LED lag)")
            p.add_argument("--drop-style", choices=["auto", *DROP_STYLES], default="auto",
                           help="drop look: auto rotates a genre-suited set every 8 bars, or lock one")
            p.add_argument("--seconds", type=float, help="stop after N seconds")

    p = sub.add_parser("analyze", help="analyze track(s) and print BPM + structure")
    p.add_argument("files", nargs="+")
    p.add_argument("-o", "--out", help="save timeline JSON here")
    p.add_argument("--fresh", action="store_true", help="ignore cached analysis")
    common(p, show=False)
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("play", help="play a track with a pre-analyzed light show")
    p.add_argument("file")
    p.add_argument("--start", type=float, default=0.0, help="start at N seconds")
    p.add_argument("--fresh", action="store_true")
    common(p)
    p.set_defaults(fn=cmd_play)

    p = sub.add_parser("live", help="live mode: listen to an audio input (DJ sets)")
    p.add_argument("--device", help="input device number or name (see `jaylights devices`)")
    p.add_argument("--loopback", action="store_true", help="listen to what the PC is playing (Windows)")
    p.add_argument("--file", help="simulate live mode from an audio file")
    common(p)
    p.set_defaults(fn=cmd_live)

    p = sub.add_parser("demo", help="no music? synthesizes a track and runs the show on virtual lights")
    p.add_argument("--live", action="store_true", help="run the demo through live mode instead")
    common(p)
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("devices", help="list audio inputs and serial ports")
    p.set_defaults(fn=cmd_devices)

    p = sub.add_parser("profiles", help="list built-in fixture profiles")
    p.set_defaults(fn=cmd_profiles)

    p = sub.add_parser("init-rig", help="write an example rig.toml to edit")
    p.add_argument("path", nargs="?", default="rig.toml")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init_rig)

    p = sub.add_parser("test-rig", help="flash each light in turn to check addresses")
    p.add_argument("--rig", required=True)
    p.add_argument("--output")
    p.add_argument("--port-name")
    p.add_argument("--loop", action="store_true")
    p.set_defaults(fn=cmd_test_rig, no_strobe=False)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
