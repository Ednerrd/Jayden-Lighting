# Handoff: Jaylights

Context for the next session picking this project up.

## The project
Automatic EDM light shows for **Edward's home DJ setup**. Edward is the shop foreman at a production company, lives in SF and is on Windows. They play house, dubstep, trap, progressive and so on, for fun. The software listens to music, finds beats and sections (intro, groove, breakdown, build, **drop**, outro), and drives DMX lights.

- **Branch:** `ccr-a9f2d96c-j8t85e`. Everything is committed and pushed. No PR has been opened.
- **Browser test page:** https://claude.ai/artifact/TuE4DvjziUAr8DNTuH1uyE
  - It is published from the files in `web/` plus the demo assets built by `python web/build_demo.py <dir>`.
  - It has three demo tracks (clean house 126, plus realistic dubstep 140 and trap 140), and you can drop in your own track, which is analyzed in the browser.
  - It has pre-analyzed and live modes, a drop-style picker, a crowd or side view, and a **Copy analysis** button.
- **Hardware:** Edward hasn't given fixture models or a USB-DMX dongle model yet. Ask for them.

## Layout
| Path | What |
|---|---|
| `jaylights/analysis/features.py` | Per-frame band features (kick/bass/mid/snare/hat, flux, centroid, flatness, coarse spectrum) |
| `jaylights/analysis/beats.py` | Tempo (ACF plus **bar-level ratio re-rank**), rigid beat grid, kick/snare-aware **downbeat** |
| `jaylights/analysis/sections.py` | Section labeling from per-bar features (see the rules below) |
| `jaylights/analysis/offline.py` | Pre-analyzed timeline, `state_at(t)` |
| `jaylights/analysis/live.py` | Live analyzer: tempo/phase fit, sub-level sections, level-crossing drop trigger |
| `jaylights/show.py` | Show engine: looks per section, 10 drop styles, blinders, crowd aiming |
| `jaylights/fixtures.py`, `rig.py`, `dmx.py` | Profiles, rig.toml, Enttec Pro / Open DMX / Art-Net |
| `web/engine.js` | **JS port of all of the above.** It must stay identical to the Python; `tools/eval_analysis.py` checks this. |
| `web/index.html` | Test page (crowd 3D view, side view, lanes, Copy analysis) |
| `tools/stress_synth.py`, `tools/eval_analysis.py` | 24-track realistic synthetic suite and harness. Run `python tools/eval_analysis.py --ranges auto --jobs 4` (about 4 minutes; `--no-live` is faster). |
| `tests/` | pytest suite (73 tests) plus `tests/real_tracks/`, which holds real songs pasted from Copy analysis |

## Section rules (sections.py) that came from real tracks
- A bar is **full** if its sub level (kick band, 35–120 Hz) is above p90 − 0.15. The kick detector alone gets fooled by 808s, wobble bass and limiting.
- A full run right after an empty run is a **drop**, and the whole run stays a drop.
- A **short dip of 8 bars or less right before a drop is the build**, even when its last bar is a silent gap. This came from "Not Even Love", build 1:08 to 1:15.
- **Drops and grooves come in whole 8-bar phrases.** If a build would leave the previous drop or groove at 12 bars, 20 bars and so on, the build starts 4 bars earlier, with the kick still going. This came from "Not Even Love", build 2:15 to 2:29.
- **After a short dip (8 bars or less), a section at groove loudness is the groove again, not a drop.** Its loudness has to sit closer to the first groove than to the biggest drop. A short dip that keeps the kick going and leads back into the groove counts as groove too. After a long breakdown (16+ bars), the next full section is still a drop. This came from "Not Even Love", groove 1:45 to 2:15.
- **Bright drops (trap / bass house): a drop can have LESS sub than the groove.** It's an 8+ bar stretch with the kick pinned (kick 1.4 or more) and tension at least 0.08 above the first groove, with gaps of 2 bars or less bridged. It must start with 3 solid bars, come after the first groove, not be the run-up to a drop already found, have less sub than the groove, and not keep climbing (a slope of tension + 0.5 × density above 0.015 means a kick build). Its build runs back to the last 1-bar energy gap. This lives in `_bright_drops` / `brightDrops`. It came from "Chicken Soup" (Skrillex & Habstrakt, Part Native Remix): build 0:28, a 2nd more intense build 0:36 (still build, NOT a drop), the beat pauses at the peak 0:41, drop 0:43 to 1:11, groove 1:11 to 1:33, build 1:33 (pause ~1:45.7), drop ~1:47.5 to 2:15, outro. The second half repeats the first exactly, 36 bars later.
- **Pause before the drop:** the pause is the bar where the sub *just* cut out after a heavy build bar, and the drop hits the bar after it. If the sub was already out the bar before, that bar was the pause and this one IS the drop: some drops open on a sub-less bar (Chicken Soup 0:43). Two-stage builds (slow, then more intense with the kick and sub in) are one build.
- Real tracks are checked Python against JS by `tools/real_tracks_js.js` (in a test).

## Scores (stress suite, auto range)
- **Offline, Python = JS:** tempo 100%, beat F 0.99, downbeat 1.00, drop recall 90%, 0.17 false drops per track, build recall 71%.
- **Live:** tempo 100%, beat F 0.91, downbeat 0.55, drop recall 71%, 0.42 false drops per track, build recall 84%.

## Open items / next ideas
1. Drops land about 1 s after where Edward hears them on "Not Even Love" (1:16 vs 1:15). Check the beat-grid and downbeat offset on real tracks.
2. Live downbeat is only 0.55. Live builds produce about 1.6 false builds per track.
3. Keep collecting real tracks: Edward pastes Copy analysis, you add it to `tests/real_tracks/` with the timestamps Edward heard, then fix and add a test.
4. Get Edward's actual fixtures and dongle, write exact profiles, then run `test-rig`.
5. Ideas: Rekordbox/Serato sync, sACN, pixel bars, haze, user-saved looks.

## Working style Edward likes
Be a homie: casual, funny, roast a little, but keep the technical side real. Edward is on their phone a lot (Claude app). The test page has no audience any more (Edward asked); the front view button is labelled "Front". Their son Jayden is 21 months old, so keep the strobe/blinder safety notes (`--no-strobe`, `--no-blinders`).
