// Prints the JS engine's section labels for every tests/real_tracks/*.txt as JSON
// (one line per track), so the Python tests can check Python == JS on real songs.
const fs = require("fs"), path = require("path");
const J = require(path.join(__dirname, "..", "web", "engine.js"));
const dir = path.join(__dirname, "..", "tests", "real_tracks");
for (const f of fs.readdirSync(dir).filter(f => f.endsWith(".txt")).sort()) {
  const M = {};
  for (const line of fs.readFileSync(path.join(dir, f), "utf8").split("\n")) {
    if (!line.includes(":")) continue;
    const [k, v] = line.split(":");
    M[k.trim()] = v.split(",").map(x => parseFloat(x) / 100);
  }
  console.log(JSON.stringify({ name: f, sections: J.labelSections(M).map(s => [s.kind, s.start, s.end]) }));
}
