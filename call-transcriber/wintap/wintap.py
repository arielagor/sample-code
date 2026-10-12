# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pyaudiowpatch",
# ]
# ///
"""Windows audio capture for the call transcriber: the counterpart of audiotap on macOS.

Streams one WASAPI device to stdout as raw 16-bit mono PCM at the device's own rate,
and prints `streaming rate=N` to stderr once it starts, the same contract audiotap
follows, so call_transcriber.py treats both helpers alike.

    python wintap.py --loopback        # whatever Windows plays (the call): the Them side
    python wintap.py --mic             # the default microphone: the Me side
    python wintap.py --mic "Headset"   # the first microphone whose name contains this
    python wintap.py --list            # list microphones and outputs

WASAPI loopback delivers nothing while Windows is silent, so silence is padded in to
keep the stream real-time. If the default device changes (headphones plugged in, a
Bluetooth headset connecting), the helper exits so the transcriber restarts it on the
new device, at the new device's rate.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio

PAD_EVERY = 0.02  # seconds between silence checks
PAD_BEHIND = 0.04  # pad up to this far behind real time, as audiotap does
CHECK_EVERY = 5.0  # seconds between default-device checks


def log(msg: str) -> None:
    print(f"wintap: {msg}", file=sys.stderr, flush=True)


def wasapi_devices(pa: pyaudio.PyAudio) -> list[dict]:
    return list(pa.get_device_info_generator_by_host_api(host_api_type=pyaudio.paWASAPI))


def find_device(pa: pyaudio.PyAudio, loopback: bool, name: str | None) -> dict:
    if loopback:
        return pa.get_default_wasapi_loopback()
    if not name:
        return pa.get_default_wasapi_device(d_out=False)
    for d in wasapi_devices(pa):
        if d["maxInputChannels"] and not d.get("isLoopbackDevice") and name.lower() in d["name"].lower():
            return d
    sys.exit(f"wintap: no microphone matching {name!r}; see --list")


def default_name(loopback: bool) -> str:
    """The current default device's name. PortAudio only enumerates devices when it
    starts, so a fresh process is the only way to see a change."""
    pa = pyaudio.PyAudio()
    try:
        return find_device(pa, loopback, None)["name"]
    finally:
        pa.terminate()


def list_devices() -> None:
    pa = pyaudio.PyAudio()
    try:
        for d in wasapi_devices(pa):
            if d.get("isLoopbackDevice"):
                continue
            kind = "mic   " if d["maxInputChannels"] else "output"
            print(f"{kind}  {d['name']}  ({int(d['defaultSampleRate'])} Hz)")
    finally:
        pa.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    side = parser.add_mutually_exclusive_group(required=True)
    side.add_argument("--loopback", action="store_true", help="capture what Windows plays")
    side.add_argument("--mic", nargs="?", const="", metavar="NAME", help="capture a microphone")
    side.add_argument("--list", action="store_true", help="list devices")
    side.add_argument("--default-name", choices=["loopback", "mic"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    # Device names carry characters like (R) and TM; keep them readable in any console.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    if args.list:
        return list_devices()
    if args.default_name:
        return print(default_name(args.default_name == "loopback"))

    loopback = args.loopback
    pa = pyaudio.PyAudio()
    dev = find_device(pa, loopback, args.mic or None)
    rate = int(dev["defaultSampleRate"])
    chans = max(1, dev["maxInputChannels"])
    log(f"device={dev['name']!r} rate={rate} channels={chans}")

    out = sys.stdout.buffer
    lock = threading.Lock()
    written = 0  # frames written so far
    last_io = 0.0
    dead = threading.Event()

    def emit(samples: np.ndarray) -> None:
        nonlocal written
        try:
            out.write(samples.tobytes())
            out.flush()
        except (BrokenPipeError, OSError):
            dead.set()  # the transcriber went away
        written += len(samples)

    def callback(in_data, frame_count, time_info, status):
        nonlocal last_io
        frames = np.frombuffer(in_data, dtype=np.int16).reshape(-1, chans)
        mono = frames.mean(axis=1).astype(np.int16)  # mix every channel down to mono
        with lock:
            last_io = time.monotonic()
            emit(mono)
        return None, pyaudio.paContinue

    stream = pa.open(format=pyaudio.paInt16, channels=chans, rate=rate, input=True,
                     input_device_index=dev["index"], frames_per_buffer=rate // 50,
                     stream_callback=callback)
    started = time.monotonic()
    log(f"streaming rate={rate}")

    # While nothing plays, pad silence up to PAD_BEHIND behind real time.
    silence = np.zeros(rate, dtype=np.int16)
    next_check = started + CHECK_EVERY
    checker: subprocess.Popen | None = None
    try:
        while not dead.is_set():
            time.sleep(PAD_EVERY)
            now = time.monotonic()
            with lock:
                if now - last_io > 0.06:
                    missing = int((now - started - PAD_BEHIND) * rate) - written
                    while missing > 0:
                        n = min(missing, len(silence))
                        emit(silence[:n])
                        missing -= n
            if not stream.is_active():
                log("device stopped (unplugged?)")
                sys.exit(1)
            # Check for a new default device in a child process, without waiting on it.
            if checker and checker.poll() is not None:
                name = checker.stdout.read().decode(errors="replace").strip()
                checker = None
                if name and name != dev["name"] and not args.mic:
                    log(f"default device changed to {name!r}; restarting")
                    sys.exit(1)
            if now >= next_check and checker is None and not args.mic:
                next_check = now + CHECK_EVERY
                checker = subprocess.Popen(
                    [sys.executable, __file__, "--default-name", "loopback" if loopback else "mic"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW)
    except KeyboardInterrupt:
        pass
    finally:
        if checker:
            checker.kill()
        stream.close()
        pa.terminate()
        log("stopped")


if __name__ == "__main__":
    main()
