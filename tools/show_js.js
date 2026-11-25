// Usage: node tools/show_js.js <json file>
// Input: {fixtures:[{kind,index,group_size,x,pan_center,tilt_center,pan_span,tilt_span,tilt_up}], genre, ctl:{maxStrobe,strobeEnabled,blinders,dropStyle},
//         states:[MusicState dicts]}. Renders every state through ONE ShowEngine (like a playback) and prints
// [[ [i, r, g, b, hz, pan, tilt] per fixture ] per state ] as JSON, so tests can compare show.py with web/engine.js.
const fs = require("fs"), path = require("path");
const J = require(path.join(__dirname, "..", "web", "engine.js"));
const d = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const eng = new J.ShowEngine(d.fixtures, d.genre);
const out = d.states.map(st => eng.render(st, d.ctl).map(o => [o.i, o.c[0], o.c[1], o.c[2], o.hz, o.pan, o.tilt]));
process.stdout.write(JSON.stringify(out));
