# Handoff: Jaylights

Context for the next session picking this project up.

## The project
Automatic EDM light shows for **Edward's home DJ setup**. Edward is the shop foreman at a production company, lives in SF and is on Windows. They play house, dubstep, trap, progressive and so on, for fun. The software listens to music, finds beats and sections (intro, groove, breakdown, build, **drop**, outro), and drives DMX lights.

- **Branch:** `claude/jaylights-handoff-wuia4d`. Everything is committed and pushed. No PR has been opened.
- **Browser test page:** https://claude.ai/artifact/TuE4DvjziUAr8DNTuH1uyE
  - It is published from the files in `web/` plus the demo assets built by `python web/build_demo.py <dir>`.
  - It has three demo tracks (clean house 126, plus realistic dubstep 140 and trap 140), and you can drop in your own track, which is analyzed in the browser.
  - It has pre-analyzed and live modes, a drop-style picker, a Front (3D, from the dance floor, no audience) or Side view, and a **Copy analysis** button.
  - To republish: publish `web/index.html` with `engine.js` as a supporting file to that same URL (the mp3s and meta.json already on it stay).
- **Hardware:** Edward hasn't given fixture models or a USB-DMX dongle model yet. Ask for them.

## Layout
| Path | What |
|---|---|
| `jaylights/analysis/features.py` | Per-frame band features (kick/bass/mid/snare/hat, flux, centroid, flatness, coarse spectrum) |
| `jaylights/analysis/beats.py` | Tempo (ACF plus **bar-level ratio re-rank**), rigid beat grid, kick/snare-aware **downbeat** |
| `jaylights/analysis/sections.py` | Section labeling from per-bar features (see the rules below) |
| `jaylights/analysis/offline.py` | Pre-analyzed timeline, `state_at(t)` |
| `jaylights/analysis/live.py` | Live analyzer: tempo/phase fit, causal `BarFeatures` (rolling 180 s percentiles), level-crossing drop trigger (mid-bar, snaps the grid) |
| `jaylights/analysis/live_sections.py` | `LiveSections`: causal bar-by-bar section machine used by live mode; `simulate_live_bars(m)` replays pasted bars through it |
| `jaylights/show.py` | Show engine: looks per section, 10 drop styles, blinders, crowd aiming |
| `jaylights/fixtures.py`, `rig.py`, `dmx.py` | Profiles, rig.toml, Enttec Pro / Open DMX / Art-Net |
| `web/engine.js` | **JS port of all of the above.** It must stay identical to the Python; `tools/eval_analysis.py` checks this. |
| `web/index.html` | Test page (Front 3D view, Side view, lanes, Copy analysis) |
| `tools/stress_synth.py`, `tools/eval_analysis.py` | 24-track realistic synthetic suite and harness. Run `python tools/eval_analysis.py --ranges auto --jobs 4` (about 4 minutes; `--no-live` is faster). |
| `tests/` | pytest suite (76 tests) plus `tests/real_tracks/`, which holds real songs pasted from Copy analysis (Not Even Love, Chicken Soup, Love Sosa) |

## Section rules (sections.py) that came from real tracks
- A bar is **full** if its sub level (kick band, 35–120 Hz) is above p90 − 0.15. The kick detector alone gets fooled by 808s, wobble bass and limiting.
- A full run right after an empty run is a **drop**, and the whole run stays a drop.
- A **short dip of 8 bars or less right before a drop is the build**, even when its last bar is a silent gap. This came from "Not Even Love", build 1:08 to 1:15.
- **Drops and grooves come in whole 8-bar phrases.** If a build would leave the previous drop or groove at 12 bars, 20 bars and so on, the build starts 4 bars earlier, with the kick still going. This came from "Not Even Love", build 2:15 to 2:29.
- **After a short dip (8 bars or less), a section at groove loudness is the groove again, not a drop.** Its loudness has to sit closer to the first groove than to the biggest drop. A short dip that keeps the kick going and leads back into the groove counts as groove too. After a long breakdown (16+ bars), the next full section is still a drop. This came from "Not Even Love", groove 1:45 to 2:15.
- **Bright drops (trap / bass house): a drop can have LESS sub than the groove.** It's an 8+ bar stretch with the kick pinned (kick 1.4 or more) and tension at least 0.08 above the first groove, with gaps of 2 bars or less bridged. It must start with 3 solid bars, come after the first groove, not be the run-up to a drop already found, have less sub than the groove, and not keep climbing (a slope of tension + 0.5 × density above 0.015 means a kick build). Its build runs back to the last 1-bar energy gap. This lives in `_bright_drops` / `brightDrops`. It came from "Chicken Soup" (Skrillex & Habstrakt, Part Native Remix): build 0:28, a 2nd more intense build 0:36 (still build, NOT a drop), the beat pauses at the peak 0:41, drop 0:43 to 1:11, groove 1:11 to 1:33, build 1:33 (pause ~1:45.7), drop ~1:47.5 to 2:15, outro. The second half repeats the first exactly, 36 bars later.
- **Pause before the drop:** the pause is the bar where the sub *just* cut out after a heavy build bar, and the drop hits the bar after it. If the sub was already out the bar before, that bar was the pause and this one IS the drop: some drops open on a sub-less bar (Chicken Soup 0:43). Two-stage builds (slow, then more intense with the kick and sub in) are one build.
- **Hook vs drop (trap):** the hook can be as loud as the drop (Love Sosa, RL Grime Remix / Acyan Flip: vocal + hi-hat rolls). The real drops land after a big build, a 6+ bar stretch with the sub cut. A "drop" that only follows a 1-2 bar dip, isn't louder than those and is busier by 0.25+ density is the hook = groove. Once the hook ends, everything up to the drop is the build (`_hook_builds`), starting on the first phrase line where no bar is as loud as the hook. Love Sosa: intro to 0:26, groove 0:26, build 0:41 (2nd build 0:55), drop 1:01 to 1:29, groove 1:29 to 2:10, build 2:10 (2nd build 2:37), drop 2:51 to 3:20, outro.
- **Vocal cut inside a drop:** a 1-2 bar hole between two alike drop halves stays drop, unless the half before it was still climbing (that was a kick build and the hole is the pre-drop silence).
- **Kick slams back after the sub cut = the drop**, even if the sub swells in for that first bar (Love Sosa 1:01).
- **Phrase grid snap:** when every drop starts on the same bar mod 4, any other boundary 1 bar off that grid snaps onto it (`_phrase_grid`, `_snap_to_grid`). All of Edward's timestamps sit on that grid.
- Real tracks are checked Python against JS by `tools/real_tracks_js.js` (in a test).

## Scores (stress suite, auto range)
- **Offline, Python = JS:** tempo 100%, beat F 0.99, downbeat 1.00, drop recall 90%, 0.17 false drops per track, build recall 71%.
- **Live (Python):** tempo 100%, beat F 0.92, downbeat 0.58, drop recall 78%, 0.25 false drops per track, build recall 84%, 0.21 false builds per track (was 71% / 0.42 / 1.62 before LiveSections).
- **Live on real tracks (bar sim):** all 6 heard drops called within a bar, no false drops, 8-9 sections per track. Builds come late (live can't see ahead). Known miss: Not Even Love's 3rd drop (3:14), its sub only dips to 0.45 vs the 0.46 cut.

## ▶ NEXT TASK: ask Edward which of these three to do first
Edward asked for these three to be the next task. Start the new chat by asking which one they want (they may say "all"). Recommend 1 if they don't care.

### 1. Live mode on real tracks (the weak spot)
Live is what Edward would actually use mid-set without pre-analyzing, and it's way off on real songs:
- **Chicken Soup live:** groove from 0:00, first drop at 1:18 (real: 0:43).
- **Love Sosa live:** 19 sections with "breakdowns" every few seconds (`groove@0:00 breakdown@0:11 build@0:13 drop@0:27 breakdown@0:41 drop@1:01 breakdown@1:02 groove@1:03 ...`).
- Code: `jaylights/analysis/live.py` (`LiveAnalyzer._sections`, the drop trigger around "THE DROP: the sub slams back in"), mirrored in `web/engine.js`. Stress suite live: downbeat 0.55, 1.6 false builds per track.
- **The catch:** the pasted bar features are offline features, and live mode needs audio, so you can't re-run live from the `tests/real_tracks/*.txt` files as they are. Two ways to get around it (pick one, or both):
  a. Add a causal "live from bars" simulation: run the live section logic bar by bar over the pasted features, seeing only past bars. Fast to test against Edward's timestamps.
  b. Extend **Copy analysis** to also dump what the live analyzer saw per bar (its sub level, full/empty flag, section, lock), so Edward's next pastes show where live goes wrong.
- Likely fixes: hysteresis (a section must hold ≥ 1-2 bars before it can change again; kills the breakdown/groove flicker), and reuse the offline lessons causally (a drop needs the sub to come back after a real sub cut; a 1-2 bar dip mid-drop isn't a breakdown; a busy hook at drop loudness isn't a drop).

### 2. Lights: make the "2nd build" step up
Edward hears trap builds in two stages: a build, then a 2nd more intense build when the sub cuts out (Chicken Soup 0:36, Love Sosa 0:55 and 2:37). Sections treat it as one build, so the lights just ramp smoothly.
- Code: `Show._build` in `jaylights/show.py` (~line 165; it uses `section_progress` for the chase speed: div 1/2/4/8, and the colour draining to white), mirrored in `web/engine.js`.
- Idea: when `st.low` drops to near zero during a BUILD (the sub cut), jump a level: subdivisions straight to 8ths/16ths, colour to near-white, movers to a tight fan, strobes (if enabled) start ticking. Optionally mark it in the analysis (a build "stage 2" start bar) so pre-analyzed mode can hit it exactly on the bar.
- **Safety:** Jayden is 21 months. Any new strobe/blinder use must respect `--no-strobe` / `--no-blinders` and the page's Strobes toggle.

### 3. More real tracks
Ask Edward for tracks that would trip it up, each with Copy analysis + their timestamps + the genre:
- a **fake-out drop** (build -> silence/vocal/2nd build instead of the drop)
- **DnB** (170+ BPM; risk: tempo read at half speed)
- **melodic/progressive** with a long breakdown (Illenium, Seven Lions, ODESZA, Lane 8)
- **house/tech house** with a soft drop (Fisher, Chris Lake, John Summit, Dom Dolla)
- **dubstep half-time** drop (Excision, Subtronics, Virtual Riot)
Workflow: save to `tests/real_tracks/<name>.txt`, map their times to bars (time = first bar + bar × bar length, both in the paste header), fix, add a `test_real_<name>` test, run the full eval (scores must not drop), check Python = JS, republish the page.

## Other open items
1. "Not Even Love" drop lands ~1 s late (1:16 vs 1:15). Chicken Soup and Love Sosa drops are now on the exact bar Edward heard, so this may be a grid offset on that one track.
2. Live downbeat is only 0.55 (part of task 1).
3. Get Edward's actual fixtures and dongle, write exact profiles, then run `test-rig`.
4. Ideas: Rekordbox/Serato sync, sACN, pixel bars, haze, user-saved looks.

## Working style Edward likes
Be a homie: casual, funny, roast a little, but keep the technical side real. Edward is on their phone a lot (Claude app), so keep replies tight and use tables for timelines. The test page has no audience any more (Edward asked). Their son Jayden is 21 months old, so keep the strobe/blinder safety notes (`--no-strobe`, `--no-blinders`).
