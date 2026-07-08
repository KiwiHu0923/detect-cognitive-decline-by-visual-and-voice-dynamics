"""
Audio transcription pipeline.

Ported and simplified from KiwiHu0923/Multimodal-Video-Content-Segmentation-System/backend/ollama_audio.py.
Strips content-segmentation logic; keeps ffmpeg extraction + mlx_whisper with word timestamps.
"""

import json
import subprocess
import wave
from pathlib import Path

import numpy as np

MIN_PAUSE_SECONDS = 0.25


def extract_audio(source_path: Path, audio_path: Path) -> None:
    """Extract 16kHz mono WAV from a video file using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(source_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def load_audio(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load WAV as float32 numpy array. Returns (array, framerate)."""
    with wave.open(str(wav_path), "rb") as wf:
        framerate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, framerate


def _run_whisper(audio_path: Path, model: str) -> dict:
    import mlx_whisper  
    return mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        word_timestamps=True,
    )


def _extract_words_and_pauses(
    whisper_result: dict,
    min_pause: float = MIN_PAUSE_SECONDS,
) -> tuple[list[dict], list[dict]]:
    """
    Flatten whisper segments into a word list and derive inter-word pauses.

    Returns:
        words  — [{"word", "start", "end", "probability"}]
        pauses — [{"start", "end", "duration"}]
    """
    words = []
    for seg in whisper_result.get("segments", []):
        for w in seg.get("words", []):
            words.append(
                {
                    "word": w["word"].strip(),
                    "start": round(float(w["start"]), 3),
                    "end": round(float(w["end"]), 3),
                    "probability": round(float(w.get("probability", 0.9)), 3),
                }
            )

    pauses = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= min_pause:
            pauses.append(
                {
                    "start": round(words[i - 1]["end"], 3),
                    "end": round(words[i]["start"], 3),
                    "duration": round(gap, 3),
                }
            )

    return words, pauses


def transcribe_file(
    source_path: str | Path,
    output_path: str | Path | None = None,
    model: str = "mlx-community/whisper-base-mlx",
    min_pause: float = MIN_PAUSE_SECONDS,
) -> dict:
    """
    End-to-end: video or audio path → transcription dict.

    If source is a video file, audio is extracted to a temp WAV and deleted
    after transcription. Saves JSON to output_path if provided.

    Returns dict with keys:
        source, duration_seconds, full_transcript, words, pauses, segments
    """
    source_path = Path(source_path)
    needs_conversion = source_path.suffix.lower() != ".wav"

    if needs_conversion:
        audio_path = source_path.with_suffix("._temp.wav")
        extract_audio(source_path, audio_path)
    else:
        audio_path = source_path

    try:
        audio_array, framerate = load_audio(audio_path)
        duration = round(len(audio_array) / framerate, 2)

        whisper_result = _run_whisper(audio_path, model)
        words, pauses = _extract_words_and_pauses(whisper_result, min_pause)
        full_transcript = " ".join(w["word"] for w in words).strip()

        result = {
            "source": str(source_path),
            "duration_seconds": duration,
            "full_transcript": full_transcript,
            "words": words,
            "pauses": pauses,
            "segments": whisper_result.get("segments", []),
        }

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(result, f, indent=2)

        return result

    finally:
        if needs_conversion and audio_path.exists():
            audio_path.unlink()
