"""DMX outputs.

* EnttecPro  - Enttec DMX USB Pro and compatibles (DMXking ultraDMX, "USB Pro" clones).
               Easiest + most reliable: the box does the DMX timing itself.
* OpenDmx    - cheap FTDI-chip dongles (Enttec Open DMX and most $20-40 "USB to DMX"
               cables). The computer does the timing, so it can flicker on a busy PC.
* ArtNet     - DMX over the network, to an Art-Net node or wireless DMX box.
* NullOutput - nothing (visualizer only).
"""

from __future__ import annotations

import socket
import struct
import threading
import time


class DmxOutput:
    name = "none"

    def send(self, universes: dict[int, bytearray]):
        pass

    def close(self):
        pass


class NullOutput(DmxOutput):
    pass


class EnttecPro(DmxOutput):
    name = "enttec_pro"

    def __init__(self, port: str):
        import serial  # pyserial

        self.ser = serial.Serial(port, baudrate=57600, timeout=1)

    @staticmethod
    def packet(data: bytes) -> bytes:
        payload = b"\x00" + bytes(data[:512]).ljust(512, b"\x00")  # start code 0 + 512 slots
        n = len(payload)
        return bytes([0x7E, 6, n & 0xFF, n >> 8]) + payload + b"\xE7"

    def send(self, universes):
        self.ser.write(self.packet(universes.get(0, bytearray(512))))

    def close(self):
        self.ser.close()


class OpenDmx(DmxOutput):
    """Bit-banged DMX through an FTDI serial chip: 250 kbaud 8N2 with a break before each frame.

    Runs in its own thread so a slow write never stalls the show."""

    name = "open_dmx"

    def __init__(self, port: str, fps: float = 30):
        import serial

        self.ser = serial.Serial(port, baudrate=250000, bytesize=8, parity="N", stopbits=2, timeout=1)
        self._data = bytearray(512)
        self._lock = threading.Lock()
        self._run = True
        self._period = 1.0 / fps
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def send(self, universes):
        with self._lock:
            self._data[:] = universes.get(0, bytearray(512))[:512].ljust(512, b"\x00")

    def _loop(self):
        while self._run:
            t0 = time.perf_counter()
            with self._lock:
                frame = b"\x00" + bytes(self._data)
            try:
                self.ser.break_condition = True
                time.sleep(0.0001)           # >= 88 us break
                self.ser.break_condition = False
                time.sleep(0.00001)          # mark after break
                self.ser.write(frame)
                self.ser.flush()
            except Exception:
                time.sleep(0.5)
            dt = time.perf_counter() - t0
            if dt < self._period:
                time.sleep(self._period - dt)

    def close(self):
        self._run = False
        self._t.join(timeout=1)
        self.ser.close()


class ArtNet(DmxOutput):
    name = "artnet"
    PORT = 6454

    def __init__(self, host: str = "255.255.255.255", first_universe: int = 0):
        self.host = host
        self.first = first_universe
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.seq = 0

    @staticmethod
    def packet(universe: int, data: bytes, seq: int = 0) -> bytes:
        data = bytes(data[:512])
        if len(data) % 2:
            data += b"\x00"
        return (b"Art-Net\x00" + struct.pack("<H", 0x5000) + struct.pack(">H", 14)
                + bytes([seq & 0xFF, 0]) + struct.pack("<H", universe & 0x7FFF)
                + struct.pack(">H", len(data)) + data)

    def send(self, universes):
        self.seq = self.seq % 255 + 1
        for u, data in universes.items():
            try:
                self.sock.sendto(self.packet(self.first + u, data, self.seq), (self.host, self.PORT))
            except OSError:
                pass

    def close(self):
        self.sock.close()


def make_output(cfg: dict) -> DmxOutput:
    kind = cfg.get("type", "none")
    if kind == "enttec_pro":
        return EnttecPro(cfg.get("port", "COM3"))
    if kind == "open_dmx":
        return OpenDmx(cfg.get("port", "COM3"), fps=min(30, cfg.get("fps", 30)))
    if kind == "artnet":
        return ArtNet(cfg.get("host", "255.255.255.255"), int(cfg.get("universe", 0)))
    if kind in ("none", "", None):
        return NullOutput()
    raise ValueError(f"unknown output type '{kind}' (use enttec_pro, open_dmx, artnet or none)")


def build_universes(fixtures, states, t: float) -> dict[int, bytearray]:
    unis: dict[int, bytearray] = {}
    for f, st in zip(fixtures, states):
        buf = unis.setdefault(f.universe, bytearray(512))
        vals = f.render(st, t)
        start = f.address - 1
        for i, v in enumerate(vals):
            if 0 <= start + i < 512:
                buf[start + i] = max(0, min(255, v))
    return unis
