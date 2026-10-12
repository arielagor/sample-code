"""Local Whisper speech-to-text with faster-whisper. No API key; audio stays on this machine.

Needs the faster-whisper package, which this engine does not install for you:

    uv run --with faster-whisper call_transcriber.py start --engine whisper

Optional: WHISPER_MODEL (default small.en; base.en is about 2.5x faster on a CPU, tiny.en
faster still), WHISPER_DEVICE (default auto: CUDA if available, else CPU),
WHISPER_COMPUTE (default int8), WHISPER_LANGUAGE (default en; empty to auto-detect, which
needs a multilingual model such as small). The model downloads on first use.

Whisper is not a streaming model, so this engine finds phrases itself: a phrase starts
when the audio gets louder than the room and ends after PAUSE seconds of quiet (or at
MAX_PHRASE). Each phrase is transcribed in a worker thread so audio keeps flowing.
There is no diarization, so every far-side voice is speaker 0 (`Them`).
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager

from engines import Utterance

RATE = 16000  # what Whisper expects
PAUSE = 0.5  # seconds of quiet that end a phrase
MAX_PHRASE = 15.0  # seconds; a longer phrase is cut and transcribed in pieces
PRE_ROLL = 0.3  # seconds of audio kept from before the phrase was detected
MIN_SPEECH = 0.3  # seconds of loud audio needed before a phrase is worth transcribing
MIN_LEVEL = 0.004  # RMS floor (about -48 dBFS) below which nothing counts as speech
NO_SPEECH = 0.6  # Whisper's no-speech probability above which a segment is dropped

_models: dict[tuple, object] = {}  # one model per config, shared by both sides
_models_lock = threading.Lock()


class Engine:
    name = "Whisper (local)"

    def __init__(self) -> None:
        try:
            import faster_whisper  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            raise SystemExit("faster-whisper is not installed: "
                             "uv run --with faster-whisper call_transcriber.py ... --engine whisper")
        self.model = os.environ.get("WHISPER_MODEL", "small.en")
        self.device = os.environ.get("WHISPER_DEVICE", "auto")
        self.compute = os.environ.get("WHISPER_COMPUTE", "int8")
        self.language = os.environ.get("WHISPER_LANGUAGE", "en") or None

    def load_model(self):
        key = (self.model, self.device, self.compute)
        with _models_lock:
            if key not in _models:
                from faster_whisper import WhisperModel
                print(f"loading Whisper model {self.model} (downloads on first use)", flush=True)
                _models[key] = WhisperModel(self.model, device=self.device,
                                            compute_type=self.compute, num_workers=2)
            return _models[key]

    @asynccontextmanager
    async def connect(self, rate: int, diarize: bool):
        model = await asyncio.to_thread(self.load_model)
        session = Session(model, rate, self.language)
        try:
            yield session
        finally:
            session.close()


class Session:
    def __init__(self, model, rate: int, language: str | None) -> None:
        import numpy as np

        self.np = np
        self.model = model
        self.rate = rate
        self.language = language
        self.phrases: asyncio.Queue = asyncio.Queue()  # (started, float32 audio at RATE) or None
        self.recent: list = []  # pre-roll chunks while quiet
        self.phrase: list = []  # chunks of the phrase being spoken
        self.started = 0.0
        self.loud = 0.0  # seconds of loud audio in the current phrase
        self.quiet = 0.0  # seconds of quiet since the last loud chunk
        self.floor = MIN_LEVEL  # running estimate of the room's noise level

    def close(self) -> None:
        self.phrases.put_nowait(None)

    async def send(self, pcm: bytes) -> None:
        np = self.np
        chunk = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
        if not len(chunk):
            return
        secs = len(chunk) / self.rate
        level = float(np.sqrt(np.mean(chunk * chunk)))
        loud = level > max(MIN_LEVEL, self.floor * 3)
        if not loud:  # follow the noise floor slowly, and only while quiet
            self.floor = 0.95 * self.floor + 0.05 * max(level, MIN_LEVEL / 3)
        if not self.phrase:
            if not loud:
                self.recent = (self.recent + [chunk])[-max(1, round(PRE_ROLL / secs)):]
                return
            pre = sum(len(c) for c in self.recent) / self.rate
            self.started = time.time() - secs - pre  # wall clock, like the other engines
            self.phrase, self.recent = self.recent + [chunk], []
            self.loud, self.quiet = secs, 0.0
            return
        self.phrase.append(chunk)
        if loud:
            self.loud += secs
            self.quiet = 0.0
        else:
            self.quiet += secs
        length = sum(len(c) for c in self.phrase) / self.rate
        if self.quiet >= PAUSE or length >= MAX_PHRASE:
            if self.loud >= MIN_SPEECH:
                self.phrases.put_nowait((self.started, self.resample(np.concatenate(self.phrase))))
            self.phrase = []
            self.recent = []

    def resample(self, audio):
        if self.rate == RATE:
            return audio
        np = self.np
        n = round(len(audio) * RATE / self.rate)
        return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)

    def transcribe(self, audio) -> str:
        segments, _ = self.model.transcribe(
            audio, language=self.language, beam_size=1, vad_filter=True,
            condition_on_previous_text=False)
        # Whisper invents text for noise ("Thank you.", "You"); its own no-speech score
        # and the VAD filter above catch most of it.
        return " ".join(s.text.strip() for s in segments if s.no_speech_prob < NO_SPEECH).strip()

    async def results(self):
        while (item := await self.phrases.get()) is not None:
            started, audio = item
            if text := await asyncio.to_thread(self.transcribe, audio):
                yield Utterance(text, started, 0)
