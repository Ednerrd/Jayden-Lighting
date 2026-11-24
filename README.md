# Jaylights 💡🎧

Automatic light shows for EDM. It listens to the music, works out the tempo, the
kick, the snares and where you are in the track (groove, breakdown, **build**,
**DROP**), and runs your DMX lights to match. Tuned for house, progressive,
dubstep, trap, techno, trance and DnB (half-time aware).

Two ways to run it:

| Mode | Use it for | How it knows about drops |
|---|---|---|
| **Live** (`jaylights live`) | DJ sets: plug the mixer into the PC | Hears the kick slam back after a build and fires instantly (and corrects itself if it was a false alarm). Beat is *predicted*, so lights land on the beat, not after it. |
| **Pre-analyzed** (`jaylights play`) | Playing tracks or a recorded mix from a file | Analyzes the whole file first, so it knows exactly where every drop is. It blacks out on the beat *before* the drop, like a real LD. |

There's a browser **visualizer** with virtual lights, so you can try it with no
lights plugged in at all.

---

## 1. Setup (Windows)

1. Install **Python 3.12** from python.org. On the first screen, **tick "Add python.exe to PATH"**.
2. Get this folder onto your PC (GitHub → Code → Download ZIP, then unzip it, or use `git clone`).
3. Double-click **`setup.bat`**. It creates a private Python environment and installs everything.
4. Double-click **`demo.bat`**. A browser tab opens with a virtual light show running on a synthesized track. 🎉

From here on, open a terminal in this folder (in Explorer, click the address bar, type `cmd`, then Enter) and use `jaylights.bat <command>`:

```bat
jaylights.bat demo --genre dubstep        :: virtual demo, half-time dubstep track
jaylights.bat demo --live                 :: same, but through the live analyzer
jaylights.bat play "C:\Music\track.mp3"   :: pre-analyzed show for a track
jaylights.bat live --loopback             :: live: listen to whatever the PC is playing
jaylights.bat devices                     :: list audio inputs + COM ports
```

Audio formats: WAV, MP3, FLAC, OGG and AIFF. Convert M4A/AAC files first.

---

## 2. Hooking up real lights

### What you need
- **A USB→DMX interface.** Any of these:
  - **Enttec DMX USB Pro** or a "Pro-compatible" box like DMXking ultraDMX: `type = "enttec_pro"`. This is the most reliable option, because the box handles the DMX timing itself.
  - **A cheap FTDI dongle** (Enttec Open DMX, and most budget "USB to DMX" cables): `type = "open_dmx"`. It works, but the PC handles the timing, so you may see the odd flicker.
  - **An Art-Net node** or wireless DMX box on your network: `type = "artnet"`.
- **DMX cables** daisy-chained light → light, ideally with a **terminator** plugged into the last light.

### Set up your rig file
```bat
jaylights.bat init-rig            :: writes rig.toml, open it in Notepad
jaylights.bat profiles            :: built-in fixture types and their channel layouts
```
For each light:
1. On the light's own menu, set its **DMX mode**, e.g. `7CH`, and its **start address**, e.g. `d001`.
2. Add a `[[fixture]]` block to `rig.toml` with the same address and a matching `profile`.
3. If none of the built-in profiles match, copy the channel chart from the manual into a `[profile.xxx]` block. The bottom of `rig.toml` shows how.

Put the COM port in `[output]`. Find it with `jaylights.bat devices`, or in Device Manager under **Ports (COM & LPT)**. Then check your patch:
```bat
jaylights.bat test-rig --rig rig.toml
```
Each light flashes white, red, green and blue in turn, and movers sweep. If the wrong light flashes, that light's address is wrong.

### Run the show
```bat
jaylights.bat live --rig rig.toml --loopback --genre house
jaylights.bat play "C:\Music\track.mp3" --rig rig.toml --genre dubstep
```

---

## 3. Live mode for DJ sets: getting the audio in

Pick one of these:
- **Loopback (easiest).** `--loopback` listens to whatever Windows is playing. If your DJ controller is its own sound card (most are), name it, e.g. `--loopback --device "DDJ"`. Run `jaylights.bat devices` to see the names.
- **Line in.** Run a cable from the mixer's **REC / BOOTH / MASTER 2** out to your PC's line-in or audio interface, then pass `--device <number>`.

Leave it on `auto` for a minute at the start. The status shows **● LOCKED** once it has the beat.

### Visualizer controls (keyboard shortcuts in brackets)
| Button | What it does |
|---|---|
| **BLACKOUT** `B` | All lights off |
| **STROBES** `S` | Strobes on/off |
| **BLINDERS** `L` | Audience blinders on/off |
| MASTER / MOVE sliders | Overall brightness / how much the movers move |
| **AUTO / GROOVE / BREAK / BUILD / DROP** `0-4` | Override the section (live mode). Hit DROP yourself if you want it perfect. |
| **DROP STYLE** dropdown | Lock one drop look, or leave on `auto` (see below) |
| **TAP THE 1** `T` | Press on the first beat of a bar if the bar count is off |
| **½ BEAT** `H` | Use this if the lights are pulsing on the offbeat |
| PLAY/PAUSE `Space`, click the timeline to seek | Pre-analyzed mode |

### Genres
`--genre` sets the tempo range and the vibe: `house`, `progressive`, `techno`, `trance`,
`dubstep`, `trap`, `dnb`, or `auto`. For **dubstep and trap, set the genre**. The kick
pattern in a half-time intro is ambiguous, and the genre range stops it reading 140 as 93.

### Drop styles
Every drop starts with the same white impact flash on the first beat. After that the look is a
*drop style*. `--drop-style auto` (the default, also on `play`, `live` and `demo`) rotates through a set that
suits your `--genre` every 8 bars. Or lock one with `--drop-style NAME`, or with the visualizer's
**DROP STYLE** dropdown (changes instantly, mid-song).

| Style | Looks like |
|---|---|
| `pump` | whole rig hits on every kick, colours swap each beat, movers fan in mirrored pairs |
| `chase` | bouncing 1/2-beat chase across the lights, movers circling |
| `split` | left vs right halves swap each beat, movers snap into a crossing X |
| `strobe_hell` | strobes + washes flash on every 8th (16ths on every 4th bar), alternating colours |
| `fan_blast` | on each kick the movers fan out from the middle and snap back, washes burst outward in full colour |
| `rainbow_chase` | a rainbow rolls across the rig by position while a comet chases on 1/2 beats |
| `blackout_stabs` | everything dark except hard cut-off stabs on every kick and snare |
| `sweep` | slow, smooth mover sweeps and colour crossfades (melodic / progressive drops) |
| `half_time_heavy` | solid hit on the kick, huge white-ish slam + strobe on the snare, slow movers (dubstep / trap) |
| `crowd_sweep` | movers aimed into the crowd sweeping in mirrored pairs (1 bar/cycle), washes pump on the kick, blinders on every downbeat, strobes on snares |

Auto rotations: house `pump, crowd_sweep, rainbow_chase, sweep, chase`; progressive `sweep, crowd_sweep, rainbow_chase, pump, chase`;
techno `pump, crowd_sweep, strobe_hell, chase, blackout_stabs`; trance `sweep, crowd_sweep, rainbow_chase, pump, fan_blast`;
dubstep `half_time_heavy, strobe_hell, crowd_sweep, blackout_stabs, fan_blast`; trap `half_time_heavy, blackout_stabs, crowd_sweep, strobe_hell, fan_blast`;
dnb `fan_blast, chase, strobe_hell, pump`. Every style works with any mix of lights (pars only, movers, strobes, floor
uplights): each fixture type has its own part, so a mover-heavy style still hits hard on pars alone. Flash rates obey
`max_strobe_hz`, and `--no-strobe` also stops the wash flashing from running faster than 2 per second.

### Blinders and crowd aiming
Lights that face the audience, like a real EDM show.

- **Blinders** (`kind = "blinder"`, profiles `blinder_1ch` / `blinder_4ch`; 2/4/8-lite warm-white lamps aimed at the
  crowd) are used sparingly: a full hit on the drop impact (decaying over a beat), a swell up to ~50% in the last bar
  of a build (pre-analyzed mode only, where the drop is known; dark again for the final blackout beat), short hits on
  big snares in `half_time_heavy` / `blackout_stabs`, and on every downbeat in `crowd_sweep`. Otherwise they stay off.
- **Movers** aim *up* during builds and snap *at the crowd* on the drop impact (`dtilt < 0` = toward the audience,
  `> 0` = toward the ceiling; set `tilt_up` per fixture so "up" is right for how it hangs).
- Turn blinders off with `--no-blinders`, `[safety] blinders = false` in `rig.toml`, or the BLINDERS button. This is
  independent of `--no-strobe`.

⚠️ **Blinders are very bright.** Don't point them at people's eyes at close range at home, and keep `master` low.

### Tuning
- Lights a bit late? Raise `--offset` (default 30 ms). It fires the lights earlier to cover DMX and LED lag.
- Too much strobe? Lower `max_strobe_hz` in `rig.toml`, or use `--no-strobe`.

⚠️ **Strobe safety:** strobes can trigger seizures in photosensitive people. Use
`--no-strobe` (or `strobe = false` in `rig.toml`) when kids or guests who might be
sensitive are around.

---

## How it works (short version)

- **Features** (`analysis/features.py`): splits the audio into kick, bass, mid, snare and hat bands about 86 times a second, and measures loudness, brightness and how "noisy" it sounds (risers are noisy).
- **Pre-analyzed** (`analysis/offline.py`, `beats.py`, `sections.py`): fits one rigid beat grid to the whole track, since EDM tempos don't drift. It finds "the 1" from where the chords change. Then it labels 4-bar phrases: kick plus top energy is a drop, no kick is a breakdown, and rising tension before a drop is a build.
- **Live** (`analysis/live.py`): every 0.2 s it measures tempo and phase from the last 6 s of kicks and snares. It freezes the tempo during builds, because snare rolls lie. The first big kick after a build means DROP, which gets confirmed or taken back over the next 3 beats.
- **Show** (`show.py`): one look per section, with genre palettes. The drop picks a drop style (nine of them, auto-rotating by genre every 8 bars, or locked with `--drop-style`).
- **DMX** (`dmx.py`, `fixtures.py`): turns each look into channel values per fixture profile.

Run the tests with `python -m pytest -q`. They generate synthetic house, DnB and dubstep tracks
with known structure and check that the analyzers find every build and drop.

## Status / what's next
- It's tuned on synthetic tracks, so expect some threshold tuning on real music. If a
  track gets mislabeled, run `jaylights analyze` on it and send the output.
- Ideas: Rekordbox/Serato track sync, sACN output, pixel-mapped LED bars, haze control,
  saving your own looks.
