"""Where the music comes from.

FileSource  - plays a track (pre-analyzed mode). Knows the whole song in advance.
LiveSource  - listens to an input (line-in / mic / DJ controller) or, on Windows, a
              "loopback" of what the PC is playing. Or simulates live from a file.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
import time

import numpy as np

from .analysis.live import LiveAnalyzer
from .analysis.offline import Timeline, load_audio
from .state import MusicState


def _sounddevice():
    try:
        import sounddevice as sd  # noqa: F401
        return sd
    except (ImportError, OSError):
        return None


class FileSource:
    mode = "file"

    def __init__(self, path: str, timeline: Timeline, play_audio: bool = True, start: float = 0.0,
                 light_offset: float = 0.03):
        self.timeline = timeline
        self.light_offset = light_offset  # fire lights a bit early to cover DMX + LED lag
        self.play_audio = play_audio
        self._pos = int(0)
        self._paused = False
        self._stream = None
        self._latency = 0.0
        if play_audio:
            import soundfile as sf

            self.audio, self.sr = sf.read(path, dtype="float32", always_2d=True)
        else:
            self.audio, self.sr = None, 44100
        self._wall0 = None
        self._start = start
        self.seek(start)

    @property
    def duration(self) -> float:
        return self.timeline.duration

    def start(self):
        sd = _sounddevice() if self.play_audio else None
        if sd is None:
            if self.play_audio:
                print("! no audio output available, running the show on the clock (silent)")
            self.play_audio = False
            self._wall0 = time.perf_counter() - self._start
            return
        ch = self.audio.shape[1]

        def cb(outdata, frames, _t, _status):
            if self._paused:
                outdata.fill(0)
                return
            chunk = self.audio[self._pos:self._pos + frames]
            outdata[:len(chunk)] = chunk
            outdata[len(chunk):] = 0
            self._pos += frames

        self._stream = sd.OutputStream(samplerate=self.sr, channels=ch, callback=cb, blocksize=512)
        self._stream.start()
        self._latency = float(self._stream.latency or 0.0)

    def time(self) -> float:
        if self.play_audio:
            return self._pos / self.sr - self._latency
        if self._paused:
            return self._pause_t
        return time.perf_counter() - self._wall0

    def state(self) -> MusicState:
        t = self.time()
        st = self.timeline.state_at(t + self.light_offset)
        st.t = t
        st.playing = not self._paused and t <= self.duration
        return st

    def toggle_pause(self):
        if not self._paused:
            self._pause_t = self.time()
        elif not self.play_audio:
            self._wall0 = time.perf_counter() - self._pause_t
        self._paused = not self._paused

    def seek(self, t: float):
        t = max(0.0, min(self.duration, t))
        self._pos = int(t * self.sr)
        self._pause_t = t
        if self._wall0 is not None or not self.play_audio:
            self._wall0 = time.perf_counter() - t

    def finished(self) -> bool:
        return self.time() > self.duration + 1.0

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


class LiveSource:
    mode = "live"

    def __init__(self, device=None, loopback: bool = False, sim_file: str | None = None,
                 bpm_range=(90.0, 190.0), latency: float = 0.03, play_sim: bool = True):
        self.device = device
        self.loopback = loopback
        self.sim_file = sim_file
        self.play_sim = play_sim
        self.bpm_range = bpm_range
        self.latency = latency
        self.analyzer: LiveAnalyzer | None = None
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue(maxsize=200)
        self._run = True
        self._threads: list[threading.Thread] = []
        self._stream = None
        self.input_name = ""
        self._done = False

    # ---- start ----
    def start(self):
        if self.sim_file:
            self._start_sim()
        elif self.loopback:
            self._start_loopback()
        else:
            self._start_input()
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()
        self._threads.append(t)

    def _make_analyzer(self, sr):
        self.analyzer = LiveAnalyzer(int(sr), self.bpm_range, latency=self.latency)

    def _start_input(self):
        sd = _sounddevice()
        if sd is None:
            raise RuntimeError("sounddevice/PortAudio not available - `pip install sounddevice`")
        dev = self.device
        if isinstance(dev, str) and dev.isdigit():
            dev = int(dev)
        info = sd.query_devices(dev, "input")
        sr = int(info["default_samplerate"])
        ch = max(1, min(2, int(info["max_input_channels"])))
        self.input_name = info["name"]
        self._make_analyzer(sr)

        def cb(indata, frames, _t, _status):
            try:
                self._q.put_nowait(indata.mean(axis=1).copy())
            except queue.Full:
                pass

        self._stream = sd.InputStream(device=dev, samplerate=sr, channels=ch, callback=cb, blocksize=512)
        self._stream.start()

    def _start_loopback(self):
        try:
            import soundcard as sc
        except ImportError as e:
            raise RuntimeError("loopback needs the 'soundcard' package: pip install soundcard") from e
        spk = sc.default_speaker() if not self.device else next(
            s for s in sc.all_speakers() if str(self.device).lower() in s.name.lower())
        mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
        self.input_name = f"loopback: {spk.name}"
        sr = 48000
        self._make_analyzer(sr)

        def loop():
            with mic.recorder(samplerate=sr, blocksize=512) as rec:
                while self._run:
                    data = rec.record(numframes=512)
                    try:
                        self._q.put_nowait(data.mean(axis=1).astype(np.float32))
                    except queue.Full:
                        pass

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self._threads.append(t)

    def _start_sim(self):
        """Pretend a file is a live input: feed it in real time (and play it if we can)."""
        mono, sr = load_audio(self.sim_file)
        self._make_analyzer(sr)
        self.input_name = f"simulated live: {self.sim_file}"
        sd = _sounddevice() if self.play_sim else None
        if sd is not None:
            import soundfile as sf

            stereo, _ = sf.read(self.sim_file, dtype="float32", always_2d=True)
            pos = [0]

            def cb(outdata, frames, _t, _status):
                chunk = stereo[pos[0]:pos[0] + frames]
                outdata[:len(chunk)] = chunk
                outdata[len(chunk):] = 0
                try:
                    self._q.put_nowait(mono[pos[0]:pos[0] + frames].copy())
                except queue.Full:
                    pass
                pos[0] += frames
                if pos[0] >= len(mono):
                    self._done = True

            self._stream = sd.OutputStream(samplerate=sr, channels=stereo.shape[1], callback=cb, blocksize=512)
            self._stream.start()
            return

        def loop():
            t0 = time.perf_counter()
            for i in range(0, len(mono), 512):
                if not self._run:
                    return
                self._q.put(mono[i:i + 512])
                ahead = (i + 512) / sr - (time.perf_counter() - t0)
                if ahead > 0:
                    time.sleep(ahead)
            self._done = True

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self._threads.append(t)

    def _worker(self):
        while self._run:
            try:
                block = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._lock:
                self.analyzer.push(block)

    # ---- show loop API ----
    def state(self) -> MusicState:
        if self.analyzer is None:
            return MusicState(locked=False)
        with self._lock:
            return dataclasses.replace(self.analyzer.state)

    def control(self, action: str, value=None):
        if self.analyzer is None:
            return
        with self._lock:
            if action == "force":
                self.analyzer.force_section(value or None)
            elif action == "tap_one":
                self.analyzer.tap_downbeat()
            elif action == "half_beat":
                self.analyzer.nudge_half_beat()

    def finished(self) -> bool:
        return self._done and self._q.empty()

    def stop(self):
        self._run = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
