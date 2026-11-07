"""Push-to-talk speech to text on this laptop.

Spot's access point has no internet, so speech is transcribed locally with
faster-whisper. The browser sends 16 kHz mono 16-bit PCM; the text then goes
through the same command parser and target steps as typing.
"""
import importlib.util
import threading

import numpy as np

SAMPLE_RATE = 16000
MAX_SECONDS = 15
MODEL = "base.en"
INSTALL_HINT = "Voice needs the optional package: .venv/bin/pip install 'faster-whisper>=1.1,<2'"


def available():
    return importlib.util.find_spec("faster_whisper") is not None


class Transcriber:
    def __init__(self, model_name=MODEL, factory=None):
        self.model_name = model_name
        self.factory = factory
        self.model = None
        self.lock = threading.Lock()

    def _load(self):
        if self.factory is not None:
            return self.factory()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ValueError(INSTALL_HINT) from exc
        try:
            # Prefer the cached copy so loading works on Spot's Wi-Fi.
            return WhisperModel(self.model_name, device="cpu", compute_type="int8",
                                local_files_only=True)
        except Exception:
            return WhisperModel(self.model_name, device="cpu", compute_type="int8")

    def transcribe(self, pcm16):
        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.
        if audio.size < SAMPLE_RATE * .3:
            raise ValueError("Recording too short; hold the button while speaking")
        if audio.size > SAMPLE_RATE * MAX_SECONDS:
            raise ValueError(f"Recording longer than {MAX_SECONDS} s")
        if float(np.sqrt(np.mean(audio ** 2))) < .002:
            raise ValueError("No speech heard; check the microphone")
        with self.lock:
            if self.model is None:
                self.model = self._load()
            segments, _ = self.model.transcribe(
                audio, language="en", beam_size=1, condition_on_previous_text=False,
                without_timestamps=True)
            # Whisper invents text on silence; drop segments it rates as non-speech.
            text = " ".join(s.text.strip() for s in segments if s.no_speech_prob < .6)
        return " ".join(text.split())
