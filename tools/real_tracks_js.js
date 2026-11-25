// Prints the JS engine's section labels for every tests/real_tracks/*.txt as JSON
// (one line per track), so the Python tests can check Python == JS on real songs:
// "sections" = offline labelSections, "live" = the causal bar-by-bar LiveSections simulation.
// Only the six per-bar feature lines are read; anything else (header lines, the live_* trace
// lines the page's Copy analysis adds) is ignored.
const fs = require("fs"), path = require("path");
const J = require(path.join(__dirname, "..", "web", "engine.js"));
const dir = path.join(__dirname, "..", "tests", "real_tracks");
const KEYS = ["kick", "low", "energy", "high", "tension", "density"];
for (const f of fs.readdirSync(dir).filter(f => f.endsWith(".txt")).sort()) {
  const M = {};
  for (const line of fs.readFileSync(path.join(dir, f), "utf8").split("\n")) {
    const at = line.indexOf(":");
    if (at < 0) continue;
    const k = line.slice(0, at).trim();
    if (!KEYS.includes(k)) continue;
    const v = line.slice(at + 1).split(",").map(x => parseFloat(x) / 100);
    if (v.length && v.every(x => Number.isFinite(x))) M[k] = v;
  }
  console.log(JSON.stringify({
    name: f,
    sections: J.labelSections(M).map(s => [s.kind, s.start, s.end]),
    live: J.simulateLiveBars(M).map(s => [s.kind, s.start, s.end]),
  }));
}
