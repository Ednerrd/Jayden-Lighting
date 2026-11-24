# Browser test rig

`engine.js` is a straight JavaScript port of the analyzers and show engine
(same algorithms and constants as the Python package; checked to give identical
BPM, downbeat and sections). `index.html` plays a track on the virtual rig and
lets you drop in your own file, which gets analyzed in the browser.

To run it locally:

    python web/build_demo.py web      # makes the two demo tracks (mp3) + meta.json with their known structure
    python -m http.server -d web 8000 # then open http://localhost:8000

The page analyzes every track (demo and your own) in the browser, keeps the per-frame music
states, and renders the lights from them with `Jaylights.renderShow(states, genre, dropStyle)`,
so the **Drop style** picker re-renders the show instantly without re-analyzing.
