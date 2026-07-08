"""
Smoke tests for the transcription pipeline.
Does not require mlx_whisper or any audio files — uses stdlib only.
"""

import sys
import wave
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio.transcribe import load_audio, _extract_words_and_pauses


def _make_synthetic_wav(path: Path, duration: float = 3.0, framerate: int = 16000) -> None:
    t = np.linspace(0, duration, int(framerate * duration), endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(samples.tobytes())


def test_load_audio():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = Path(f.name)

    try:
        _make_synthetic_wav(wav_path, duration=3.0)
        audio, framerate = load_audio(wav_path)

        assert framerate == 16000, f"expected 16000, got {framerate}"
        assert len(audio) == 3 * 16000, f"expected {3 * 16000} samples, got {len(audio)}"
        assert audio.dtype == np.float32, f"expected float32, got {audio.dtype}"
        assert audio.max() <= 1.0 and audio.min() >= -1.0, "audio out of [-1, 1] range"
        print("✓ load_audio")
    finally:
        wav_path.unlink(missing_ok=True)


def test_extract_words_and_pauses():
    mock_result = {
        "segments": [
            {
                "words": [
                    {"word": " the",    "start": 0.1, "end": 0.3, "probability": 0.99},
                    {"word": " cookie", "start": 0.4, "end": 0.8, "probability": 0.95},
                    # gap of 0.7s here — should be detected as a pause
                    {"word": " jar",    "start": 1.5, "end": 1.9, "probability": 0.97},
                ]
            }
        ]
    }

    words, pauses = _extract_words_and_pauses(mock_result, min_pause=0.25)

    assert len(words) == 3, f"expected 3 words, got {len(words)}"
    assert words[0]["word"] == "the"
    assert words[1]["word"] == "cookie"
    assert words[2]["word"] == "jar"

    assert len(pauses) == 1, f"expected 1 pause, got {len(pauses)}"
    assert pauses[0]["duration"] == round(1.5 - 0.8, 3), \
        f"expected pause duration {round(1.5 - 0.8, 3)}, got {pauses[0]['duration']}"
    print("✓ _extract_words_and_pauses")


if __name__ == "__main__":
    test_load_audio()
    test_extract_words_and_pauses()
    print("\n✅ All smoke tests passed")
