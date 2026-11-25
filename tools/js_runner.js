// Usage: node tools/js_runner.js <raw_float32_file> <sample_rate> <genre>
// Runs the JS port (web/engine.js) offline + live on mono float32 audio and prints JSON:
// {bpm, beats[], downbeat_phase, sections[], live:{sections[], beats[]}}.
// The live part streams the features frame by frame exactly like analyzeTrack() does,
// but skips the lighting-show rendering (not needed for analysis).
const fs = require("fs");
const path = require("path");
const J = require(path.join(__dirname, "..", "web", "engine.js"));

(async () => {
  const [file, srS, genre] = process.argv.slice(2);
  const sr = parseInt(srS, 10);
  const buf = fs.readFileSync(file);
  const x = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  const range = (J.GENRES[genre] || J.GENRES.auto)[0];
  const F = await J.extractFeatures(x, sr, () => {});
  const tl = J.analyzeOffline(F, range);
  // live replay: wall time of a frame = its centre + FRAME/2 (the frame is complete only then)
  const la = new J.LiveAnalyzer(sr, range, 0);
  const secs = [], beats = [];
  let last = null, lastB = null, prevBeat = 0, prevT = 0;
  for (let i = 0; i < F.n; i++) {
    la.frameIn(F, i);
    const st = la.st, tc = J.frameTime(i, sr), wall = tc + 1024 / sr;
    if (st.section !== last) { secs.push({ kind: st.section, start: wall }); last = st.section; }
    const fb = Math.floor(st.beat);
    if (lastB !== null && fb !== lastB) {
      // interpolate crossing inside this frame; time on the audio axis (= test's t - 1024/sr)
      const k = Math.max(fb, lastB + 1), f = st.beat > prevBeat ? Math.min(1, Math.max(0, (k - prevBeat) / (st.beat - prevBeat))) : 1;
      beats.push({ t: prevT + f * (tc - prevT), bib: Math.round(st.beat_in_bar) % 4, locked: st.locked });
    }
    lastB = fb; prevBeat = st.beat; prevT = tc;
  }
  secs.forEach((s, i) => { s.end = i + 1 < secs.length ? secs[i + 1].start : x.length / sr; });
  const out = {
    bpm: tl.bpm, beats: Array.from(tl.beats), downbeat_phase: tl.phase,
    sections: tl.sections.map(s => ({ kind: s.kind, start: s.start, end: s.end })),
    live: { sections: secs, beats, bars: la.barTrace },
  };
  process.stdout.write(JSON.stringify(out));
})().catch(e => { console.error(e.stack || e); process.exit(1); });
