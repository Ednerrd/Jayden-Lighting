#!/usr/bin/env python
"""Evaluate the Jaylights analyzers on a suite of realistic synthetic EDM tracks.

    python tools/eval_analysis.py [--quick] [--no-js] [--genre X] [--ranges auto,match]
                                  [--jobs N] [--detail] [--no-live]

For every track in tools/stress_synth.suite_specs() it runs
  * offline Python  (jaylights.analysis.offline.analyze_audio)
  * live Python     (LiveAnalyzer streamed in 512-sample blocks)
  * offline + live JS (web/engine.js via tools/js_runner.js, needs `node`)
once with the full "auto" tempo range (90-190) and once with the range of the track's own
genre (jaylights.show.GENRES), and scores them against the generator's ground truth.

--genre X only keeps suite tracks of that genre (house/progressive/techno/melodic/dubstep/trap/dnb).
Results go to tools/eval_results.json.

Metric definitions
  tempo     est vs true BPM. "ok" = within 1%; otherwise classified x2, x0.5, x3/2, x2/3, x4/3, x3/4, other.
  beatF     F-measure of detected beats vs true beats, +-70 ms (whole track offline; after 12 s live).
  dbAcc     fraction of true downbeats that have a detected downbeat within +-70 ms (offline);
            live: fraction of beat crossings whose beat-in-bar matches the truth's bar position.
  builds / drops
            offline: hit = a detected section of that kind starts within +-1 beat of the true start;
            live drops: hit = DROP state entered 0..+0.6 s after the true drop start;
            live builds: hit = BUILD state covers >= 30% of the true build.
            FP = detected drop/build start that matches no true one (live: entered outside any true
            section of that kind, 0.6 s early tolerance).
  frame     fraction of 0.25 s frames with the exactly right section label; frame* merges
            intro/groove/outro into one "base" class (their naming is arbitrary); dropF1 = frame-level
            F1 of the "drop" class.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from stress_synth import SR, make_realistic_track, suite_specs  # noqa: E402

BEAT_TOL = 0.07
LIVE_SKIP = 12.0  # seconds of lock-in ignored for live beat metrics
GENRE_KEY = {"melodic": "progressive"}  # stress genre -> jaylights.show.GENRES key
ANALYZERS = ("off_py", "live_py", "off_js", "live_js")
LABEL = {"off_py": "OFFLINE python", "live_py": "LIVE python", "off_js": "OFFLINE js", "live_js": "LIVE js"}


# ----------------------------------------------------------------------------- metrics
def bpm_range(genre_key):
    from jaylights.show import GENRES
    return tuple(float(v) for v in GENRES[genre_key][0])


def tempo_class(est, true):
    r = est / true
    if abs(r - 1) < 0.01:
        return "ok"
    for name, q in (("x2", 2.0), ("x0.5", 0.5), ("x3/2", 1.5), ("x2/3", 2 / 3), ("x4/3", 4 / 3), ("x3/4", 0.75)):
        if abs(r / q - 1) < 0.02:
            return name
    return "other"


def match_beats(det, true, tol=BEAT_TOL):
    """Greedy one-to-one nearest matching. Returns list of (det_idx, true_idx)."""
    det, true = np.asarray(det, float), np.asarray(true, float)
    if len(det) == 0 or len(true) == 0:
        return []
    out, used = [], set()
    idx = np.searchsorted(true, det)
    for i, (d, j) in enumerate(zip(det, idx)):
        best, bj = tol + 1, None
        for c in (j - 1, j):
            if 0 <= c < len(true) and c not in used and abs(true[c] - d) < best:
                best, bj = abs(true[c] - d), c
        if bj is not None and best <= tol:
            used.add(bj)
            out.append((i, bj))
    return out


def f_measure(det, true, tol=BEAT_TOL):
    if len(det) == 0 or len(true) == 0:
        return 0.0
    tp = len(match_beats(det, true, tol))
    p, r = tp / len(det), tp / len(true)
    return 0.0 if tp == 0 else 2 * p * r / (p + r)


def phase_offset(det, true_beat_s):
    """Median signed offset of detected beats from the nearest true beat, in beats (-0.5..0.5)."""
    if len(det) == 0:
        return float("nan")
    ph = ((np.asarray(det) / true_beat_s + 0.5) % 1.0) - 0.5
    return float(np.median(ph))


def label_at(secs, t):
    for s in secs:
        if s["start"] <= t < s["end"]:
            return s["kind"]
    return secs[-1]["kind"] if secs and t >= secs[-1]["end"] else "none"


BASE = {"intro": "base", "groove": "base", "outro": "base"}


def frame_scores(det_secs, true_secs, duration, step=0.25):
    ts = np.arange(step / 2, duration, step)
    tl = [label_at(true_secs, t) for t in ts]
    dl = [label_at(det_secs, t) for t in ts]
    exact = float(np.mean([a == b for a, b in zip(tl, dl)]))
    coarse = float(np.mean([BASE.get(a, a) == BASE.get(b, b) for a, b in zip(tl, dl)]))
    tp = sum(a == "drop" and b == "drop" for a, b in zip(tl, dl))
    fp = sum(a != "drop" and b == "drop" for a, b in zip(tl, dl))
    fn = sum(a == "drop" and b != "drop" for a, b in zip(tl, dl))
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return exact, coarse, float(f1)


def kind_starts(secs, kind):
    return [s["start"] for s in secs if s["kind"] == kind]


def section_scores_offline(det_secs, true_secs, beat_s):
    out = {}
    for kind in ("build", "drop"):
        want = [s for k, s, _ in true_secs if k == kind]
        got = kind_starts(det_secs, kind)
        hit = [any(abs(g - w) <= beat_s for g in got) for w in want]
        loose = [any(abs(g - w) <= 16 * beat_s for g in got) for w in want]
        fp = [g for g in got if not any(abs(g - w) <= beat_s for w in want)]
        out[kind] = dict(n=len(want), hit=int(sum(hit)), loose=int(sum(loose)), fp=len(fp), det=len(got),
                         hit_flags=[bool(h) for h in hit], fp_times=[round(float(f), 2) for f in fp])
    return out


def section_scores_live(det_secs, true_secs, beat_s, duration):
    out = {}
    # drops: DROP entry within 0..+0.6 s of the true start
    entries = [s["start"] for s in det_secs if s["kind"] == "drop"]
    want = [(s, e) for k, s, e in true_secs if k == "drop"]
    hits, lat = [], []
    for s, e in want:
        c = [t - s for t in entries if 0 <= t - s <= 0.6]
        hits.append(bool(c))
        near = [t - s for t in entries if -3 <= t - s <= 6]
        lat.append(round(min(near, key=abs), 3) if near else None)
    fp = [t for t in entries if not any(s - 0.6 <= t <= e for s, e in want)]
    flap = [t for t in entries if any(s + 0.6 < t <= e for s, e in want)]
    out["drop"] = dict(n=len(want), hit=int(sum(hits)), loose=int(sum(l is not None for l in lat)), fp=len(fp), det=len(entries),
                       flaps=len(flap), latency=lat, hit_flags=hits, fp_times=[round(t, 2) for t in fp])
    # builds: BUILD state covers >= 30% of the true build
    bwant = [(s, e) for k, s, e in true_secs if k == "build"]
    bsec = [s for s in det_secs if s["kind"] == "build"]
    bhit = []
    for s, e in bwant:
        cov = sum(max(0.0, min(e, x["end"]) - max(s, x["start"])) for x in bsec)
        bhit.append(cov >= 0.3 * (e - s))
    bfp = [x["start"] for x in bsec if not any(s - 2 * beat_s <= x["start"] <= e for s, e in bwant)]
    out["build"] = dict(n=len(bwant), hit=int(sum(bhit)), loose=int(sum(bhit)), fp=len(bfp), det=len(bsec), hit_flags=[bool(h) for h in bhit],
                        fp_times=[round(t, 2) for t in bfp])
    return out


def downbeat_offline(beats, phase, truth):
    beats = np.asarray(beats, float)
    det_db = beats[phase % 4::4] if len(beats) else beats
    # detected downbeats also before `phase` when phase >= 4 are implicit in the %4 slicing
    tdb = truth["downbeats"]
    acc = float(np.mean([np.any(np.abs(det_db - d) <= BEAT_TOL) for d in tdb])) if len(det_db) else 0.0
    # beat-position shift (grid right but "1" on the wrong beat?)
    m = match_beats(beats, truth["beats"])
    t_first = int(round((truth["first_bar"] - truth["beats"][0]) / (60 / truth["bpm"])))
    shifts = Counter(((j - t_first) - (i - phase)) % 4 for i, j in m)
    shift = shifts.most_common(1)[0][0] if shifts else None
    return acc, shift


def eval_offline(res, truth):
    beat_s = 60.0 / truth["bpm"]
    secs = res["sections"]
    out = dict(bpm=res["bpm"], tempo_err_pct=100 * (res["bpm"] / truth["bpm"] - 1), tempo_class=tempo_class(res["bpm"], truth["bpm"]))
    out["beatF"] = f_measure(res["beats"], truth["beats"])
    out["beat_phase"] = phase_offset(res["beats"], beat_s)  # NB uses true grid origin below
    # phase relative to the true grid (beats[0] is the grid origin)
    out["beat_phase"] = phase_offset(np.asarray(res["beats"]) - truth["beats"][0], beat_s)
    acc, shift = downbeat_offline(res["beats"], res["downbeat_phase"], truth)
    out["dbAcc"], out["db_shift"] = acc, shift
    out["sections"] = section_scores_offline(secs, truth["sections"], beat_s)
    ex, co, f1 = frame_scores(secs, [dict(kind=k, start=s, end=e) for k, s, e in truth["sections"]], truth["duration"])
    out.update(frame=ex, frame_coarse=co, dropF1=f1)
    out["det_sections"] = [(s["kind"], round(float(s["start"]), 2), round(float(s["end"]), 2)) for s in secs]
    return out


def eval_live(res, truth):
    beat_s = 60.0 / truth["bpm"]
    secs = res["sections"]
    bt = np.array([b["t"] for b in res["beats"]])
    bib = np.array([b["bib"] for b in res["beats"]])
    sel = bt >= LIVE_SKIP
    tb = truth["beats"][truth["beats"] >= LIVE_SKIP]
    out = {}
    out["beatF"] = f_measure(bt[sel], tb)
    out["beat_phase"] = phase_offset(bt[sel] - truth["beats"][0], beat_s)
    # live tempo: median spacing of the crossings in the second half
    half = bt[len(bt) // 2:]
    est = 60.0 / float(np.median(np.diff(half))) if len(half) > 3 else float("nan")
    out["bpm"], out["tempo_err_pct"], out["tempo_class"] = est, 100 * (est / truth["bpm"] - 1), tempo_class(est, truth["bpm"])
    m = match_beats(bt[sel], tb)
    t_first = int(round((truth["first_bar"] - truth["beats"][0]) / beat_s))
    tidx_all = np.flatnonzero(truth["beats"] >= LIVE_SKIP)
    ok, shifts = [], Counter()
    bsel = bib[sel]
    for i, j in m:
        tpos = (tidx_all[j] - t_first) % 4
        ok.append(bsel[i] == tpos)
        shifts[(tpos - bsel[i]) % 4] += 1
    out["dbAcc"] = float(np.sum(ok) / max(1, len(tb))) if ok else 0.0
    out["db_shift"] = shifts.most_common(1)[0][0] if shifts else None
    out["sections"] = section_scores_live(secs, truth["sections"], beat_s, truth["duration"])
    ex, co, f1 = frame_scores(secs, [dict(kind=k, start=s, end=e) for k, s, e in truth["sections"]], truth["duration"])
    out.update(frame=ex, frame_coarse=co, dropF1=f1)
    out["det_sections"] = [(s["kind"], round(float(s["start"]), 2), round(float(s["end"]), 2)) for s in secs]
    return out


# ----------------------------------------------------------------------------- running analyzers
def run_off_py(audio, rng_):
    from jaylights.analysis.offline import analyze_audio
    tl = analyze_audio(audio, SR, rng_)
    return dict(bpm=float(tl.bpm), beats=[float(b) for b in tl.beats], downbeat_phase=int(tl.downbeat_phase),
                sections=[dict(kind=s["kind"], start=float(s["start"]), end=float(s["end"])) for s in tl.sections])


def run_live_py(audio, rng_):
    from jaylights.analysis.live import LiveAnalyzer
    la = LiveAnalyzer(SR, rng_)
    secs, beats = [], []
    last, prev_beat, prev_t, last_floor = None, 0.0, 0.0, None
    n = len(audio)
    for i in range(0, n, 512):
        st = la.push(audio[i:i + 512])
        t = (i + 512) / SR          # wall time the state is available
        if st.section != last:
            secs.append(dict(kind=st.section, start=t))
            last = st.section
        fl = int(np.floor(st.beat))
        if last_floor is not None and fl > last_floor:
            k = max(fl, last_floor + 1)
            f = min(1.0, max(0.0, (k - prev_beat) / (st.beat - prev_beat))) if st.beat > prev_beat else 1.0
            tc = (prev_t + f * (t - prev_t)) - 1024 / SR   # audio-time axis, as in tests/test_jaylights.py
            beats.append(dict(t=float(tc), bib=int(round(st.beat_in_bar)) % 4, locked=bool(st.locked)))
        last_floor, prev_beat, prev_t = fl, st.beat, t
    for k, s in enumerate(secs):
        s["end"] = secs[k + 1]["start"] if k + 1 < len(secs) else n / SR
    return dict(sections=secs, beats=beats)


def run_js(path, genre_key):
    p = subprocess.run(["node", os.path.join(HERE, "js_runner.js"), path, str(SR), genre_key], capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-800:])
    return json.loads(p.stdout)


def work(args):
    spec, ranges, do_js, do_live, tmp = args
    t0 = time.time()
    audio, truth = make_realistic_track(spec)
    gkey = GENRE_KEY.get(truth["genre"], truth["genre"])
    path = os.path.join(tmp, truth["name"] + ".f32")
    audio.astype("<f4").tofile(path)
    res = {"name": truth["name"], "genre": truth["genre"], "bpm": truth["bpm"], "duration": truth["duration"],
           "sections_true": truth["sections"], "pickup_beats": truth["spec"]["pickup_beats"], "silence": truth["spec"]["silence"],
           "build_style": truth["spec"]["build"], "gaps": [(round(a, 2), round(b, 2)) for a, b in truth["gaps"]],
           "swing": truth["spec"]["swing"], "analyses": {}}
    for rname in ranges:
        rng_ = (90.0, 190.0) if rname == "auto" else bpm_range(gkey)
        gname = "auto" if rname == "auto" else gkey
        a = res["analyses"].setdefault(rname, {})
        try:
            a["off_py"] = eval_offline(run_off_py(audio, rng_), truth)
        except Exception as e:  # keep going, report the crash
            a["off_py"] = {"error": repr(e)}
        if do_live:
            try:
                a["live_py"] = eval_live(run_live_py(audio, rng_), truth)
            except Exception as e:
                a["live_py"] = {"error": repr(e)}
        if do_js:
            try:
                js = run_js(path, gname)
                a["off_js"] = eval_offline(js, truth)
                if do_live:
                    a["live_js"] = eval_live(js["live"], truth)
                a["_js_raw_off"] = dict(bpm=js["bpm"], sections=js["sections"])
                # python vs js offline disagreement flags
                a["js_vs_py"] = disagreement(a.get("off_py"), a.get("off_js"))
            except Exception as e:
                a["off_js"] = {"error": repr(e)}
    res["secs"] = round(time.time() - t0, 1)
    os.remove(path)
    return res


def disagreement(py, js):
    if not py or not js or "error" in py or "error" in js:
        return []
    why = []
    if abs(py["bpm"] / js["bpm"] - 1) > 0.005:
        why.append(f"bpm {py['bpm']:.2f} vs {js['bpm']:.2f}")
    if abs(py["beatF"] - js["beatF"]) > 0.1:
        why.append(f"beatF {py['beatF']:.2f} vs {js['beatF']:.2f}")
    for kind in ("build", "drop"):
        a, b = py["sections"][kind], js["sections"][kind]
        if (a["hit"], a["fp"], a["det"]) != (b["hit"], b["fp"], b["det"]):
            why.append(f"{kind}s hit/fp/det {a['hit']}/{a['fp']}/{a['det']} vs {b['hit']}/{b['fp']}/{b['det']}")
    if abs(py["frame"] - js["frame"]) > 0.08:
        why.append(f"frame {py['frame']:.2f} vs {js['frame']:.2f}")
    return why


# ----------------------------------------------------------------------------- reporting
def fmt_row(r, m, live):
    if "error" in m:
        return f"{r['name']:<26} ERROR {m['error'][:80]}"
    d, b = m["sections"]["drop"], m["sections"]["build"]
    lat = ""
    if live:
        ls = [x for x in d["latency"] if x is not None]
        lat = f"{np.median(ls):+5.2f}" if ls else "  n/a"
    return (f"{r['name']:<26} {r['bpm']:6.1f} {m['bpm']:6.1f} {m['tempo_err_pct']:+6.1f}% {m['tempo_class']:<5} "
            f"{m['beatF']:5.2f} {m['dbAcc']:5.2f} {str(m['db_shift']) if m['db_shift'] is not None else '-':>2} "
            f"{b['hit']}/{b['n']}+{b['fp']:<2} {d['hit']}/{d['n']}+{d['fp']:<2} {m['frame']:5.2f} {m['frame_coarse']:5.2f} {m['dropF1']:5.2f}"
            + (f" {lat}" if live else ""))


HEAD = (f"{'track':<26} {'bpm':>6} {'est':>6} {'err':>7} {'cls':<5} {'beatF':>5} {'dbAcc':>5} {'sh':>2} "
        f"{'build+fp':<7} {'drop+fp':<7} {'frame':>5} {'fr*':>5} {'dropF1':>5}")


def aggregate(results, rname, an):
    ms = [r["analyses"][rname][an] for r in results if an in r["analyses"].get(rname, {}) and "error" not in r["analyses"][rname][an]]
    if not ms:
        return None
    g = lambda f: float(np.nanmean([f(m) for m in ms]))
    tot = lambda kind, key: sum(m["sections"][kind][key] for m in ms)
    cls = Counter(m["tempo_class"] for m in ms)
    return dict(n=len(ms), tempo_ok=cls["ok"] / len(ms), tempo_cls=dict(cls), beatF=g(lambda m: m["beatF"]), dbAcc=g(lambda m: m["dbAcc"]),
                drop_recall=tot("drop", "hit") / max(1, tot("drop", "n")), drop_fp=tot("drop", "fp") / len(ms),
                build_recall=tot("build", "hit") / max(1, tot("build", "n")), build_fp=tot("build", "fp") / len(ms),
                frame=g(lambda m: m["frame"]), frame_coarse=g(lambda m: m["frame_coarse"]), dropF1=g(lambda m: m["dropF1"]))


def print_report(results, ranges, detail, do_js, do_live):
    ans = [a for a in ANALYZERS if (do_js or "js" not in a) and (do_live or "live" not in a)]
    print("\n==== AGGREGATE (mean over tracks) " + "=" * 60)
    print(f"{'analyzer':<16}{'range':<7}{'n':>3} {'tempoOK':>8} {'beatF':>6} {'dbAcc':>6} {'dropRec':>8} {'dropFP/t':>9} {'bldRec':>7} {'bldFP/t':>8} {'frame':>6} {'fr*':>6} {'dropF1':>7}  tempo errors")
    agg = {}
    for an in ans:
        for rn in ranges:
            a = aggregate(results, rn, an)
            if not a:
                continue
            agg[f"{an}/{rn}"] = a
            errs = ", ".join(f"{k}:{v}" for k, v in sorted(a["tempo_cls"].items()) if k != "ok")
            print(f"{LABEL[an]:<16}{rn:<7}{a['n']:>3} {a['tempo_ok']:8.0%} {a['beatF']:6.2f} {a['dbAcc']:6.2f} {a['drop_recall']:8.0%} {a['drop_fp']:9.2f} "
                  f"{a['build_recall']:7.0%} {a['build_fp']:8.2f} {a['frame']:6.2f} {a['frame_coarse']:6.2f} {a['dropF1']:7.2f}  {errs}")
    # by genre (offline python, auto)
    for an in ("off_py", "live_py"):
        if an not in ans:
            continue
        print(f"\n---- {LABEL[an]} (range={ranges[0]}) by genre")
        print(f"{'genre':<12}{'n':>3} {'tempoOK':>8} {'beatF':>6} {'dbAcc':>6} {'dropRec':>8} {'dropFP/t':>9} {'bldRec':>7} {'frame':>6} {'dropF1':>7}")
        for g in sorted({r["genre"] for r in results}):
            sub = [r for r in results if r["genre"] == g]
            a = aggregate(sub, ranges[0], an)
            if a:
                print(f"{g:<12}{a['n']:>3} {a['tempo_ok']:8.0%} {a['beatF']:6.2f} {a['dbAcc']:6.2f} {a['drop_recall']:8.0%} {a['drop_fp']:9.2f} {a['build_recall']:7.0%} {a['frame']:6.2f} {a['dropF1']:7.2f}")
    for an in ans:
        for rn in (ranges if detail else ranges[:1]):
            print(f"\n==== {LABEL[an]}  range={rn}   (build/drop columns: hits/true+falsePos)" + ("   last col: median drop latency s" if "live" in an else ""))
            print(HEAD + (" lat" if "live" in an else ""))
            for r in results:
                m = r["analyses"].get(rn, {}).get(an)
                if m is not None:
                    print(fmt_row(r, m, "live" in an))
    if do_js:
        print("\n==== PYTHON vs JS offline disagreements")
        n = 0
        for r in results:
            for rn in ranges:
                w = r["analyses"].get(rn, {}).get("js_vs_py")
                if w:
                    n += 1
                    print(f"  {r['name']:<26} [{rn}] " + "; ".join(w))
        if not n:
            print("  none")
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="8 tracks instead of 24")
    ap.add_argument("--no-js", action="store_true")
    ap.add_argument("--no-live", action="store_true", help="skip the (slow) live analyzers")
    ap.add_argument("--genre", help="only tracks of this genre")
    ap.add_argument("--ranges", default="auto,match", help="comma list of: auto (90-190) / match (the track's genre range)")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--detail", action="store_true", help="per-track tables for every range, not just the first")
    ap.add_argument("--only", help="comma list of track names")
    ap.add_argument("--out", default=os.path.join(HERE, "eval_results.json"))
    a = ap.parse_args()

    specs = suite_specs(a.quick)
    if a.genre:
        specs = [s for s in specs if s["genre"] == a.genre]
    if a.only:
        keep = set(a.only.split(","))
        specs = [s for s in specs if s["name"] in keep]
    if not specs:
        sys.exit("no tracks selected")
    do_js = not a.no_js
    if do_js:
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
        except Exception:
            print("node not found: skipping the JS port", file=sys.stderr)
            do_js = False
    ranges = [r for r in a.ranges.split(",") if r]
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        jobs = [(s, ranges, do_js, not a.no_live, tmp) for s in specs]
        results = []
        with mp.Pool(max(1, min(a.jobs, len(jobs)))) as pool:
            for r in pool.imap(work, jobs):
                results.append(r)
                print(f"  done {r['name']:<26} {r['secs']:5.1f}s", file=sys.stderr, flush=True)
    agg = print_report(results, ranges, a.detail, do_js, not a.no_live)
    print(f"\n{len(results)} tracks, {time.time() - t0:.0f}s total")
    for r in results:
        for rn in r["analyses"]:
            r["analyses"][rn].pop("_js_raw_off", None)
    with open(a.out, "w") as f:
        json.dump({"aggregate": agg, "tracks": results}, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
