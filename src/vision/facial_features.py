"""Extract per-frame OpenFace AU intensities and presence flags from a video.

Uses OpenFace 2.0 via Docker (`algebr/openface`) — the same extractor UFNet trained
on, so the AU intensity scale (0-5) and presence flags match the training
distribution exactly.

AU column order is locked to the canonical UFNet list:
    AU01, AU06, AU12, AU14, AU25, AU26, AU45
matching the AU order in `eval/models/smile_pd_columns.json`.

Frames where OpenFace's tracking `confidence` < 0.75 or the `success` flag is 0
are NaN-masked (AU_r) / zero-masked (AU_c). Downstream `aggregate.py` uses the
active-frame-only rule from Islam et al. 2023 ("Unmasking Parkinson's Disease
with Smile", arxiv 2308.02588 §2): mean and variance are computed only over
frames where AU_c == 1.

Docker Desktop (or colima) must be running. On Apple Silicon, the image runs
under Rosetta 2 emulation — expect ~20-30s per 5-second clip.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

AUS: list[str] = ["AU01", "AU06", "AU12", "AU14", "AU25", "AU26", "AU45"]

CONFIDENCE_MIN: float = 0.75
QUALITY_GATE_MIN: float = 0.80

OPENFACE_IMAGE: str = "algebr/openface:latest"
OPENFACE_BINARY: str = "./build/bin/FeatureExtraction"
DOCKER_TIMEOUT_SEC: int = 300


def _cut_segment(
    video_path: Path, out_path: Path, t0: float | None, t1: float | None
) -> None:
    """ffmpeg-cut `[t0, t1]` from input to out_path; copy verbatim if no window."""
    if t0 is None and t1 is None:
        shutil.copy(video_path, out_path)
        return
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path)]
    if t0 is not None:
        cmd += ["-ss", str(t0)]
    if t1 is not None:
        duration = (t1 - t0) if t0 is not None else t1
        cmd += ["-t", str(duration)]
    # Re-encode so segment boundaries land on a keyframe; drop audio (OpenFace ignores it).
    cmd += [
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-an",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)


def _run_openface(work_dir: Path, video_name: str) -> Path:
    """Run OpenFace FeatureExtraction inside Docker; return path to the CSV."""
    out_dir = work_dir / "out"
    out_dir.mkdir(exist_ok=True)
    base_cmd = [
        "docker", "run", "--rm",
        "--platform", "linux/amd64",
        "--entrypoint", "",
        "-v", f"{work_dir}:/data",
        OPENFACE_IMAGE,
        OPENFACE_BINARY,
        "-f", f"/data/{video_name}",
        "-out_dir", "/data/out",
    ]
    # Try with the AU-only output flag first for speed. If OpenFace rejects it
    # (unlikely but not guaranteed on all builds), retry without: OpenFace then
    # emits every output type, which is slower but always produces AU columns.
    try:
        subprocess.run(
            base_cmd + ["-aus"], check=True, capture_output=True, timeout=DOCKER_TIMEOUT_SEC
        )
    except subprocess.CalledProcessError:
        subprocess.run(base_cmd, check=True, capture_output=True, timeout=DOCKER_TIMEOUT_SEC)
    csv_path = out_dir / (Path(video_name).stem + ".csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"OpenFace produced no CSV at {csv_path}")
    return csv_path


def _parse_openface_csv(csv_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    df = pd.read_csv(csv_path)
    # OpenFace CSV columns are sometimes emitted with a leading space.
    df.columns = [c.strip() for c in df.columns]
    n_total = len(df)
    warnings: list[str] = []

    if n_total == 0:
        return (
            {"au_r": np.zeros((0, len(AUS))), "au_c": np.zeros((0, len(AUS)), dtype=np.int8)},
            {
                "fps": 0.0,
                "n_frames_total": 0,
                "n_frames_used": 0,
                "detection_rate": 0.0,
                "quality_gate_pass": False,
                "warnings": ["OpenFace produced empty CSV — no face detected in clip"],
            },
        )

    ts = df["timestamp"].to_numpy() if "timestamp" in df.columns else np.array([])
    if ts.size > 1 and ts[-1] > ts[0]:
        fps = float((n_total - 1) / (ts[-1] - ts[0]))
    else:
        fps = 0.0
        warnings.append("could not infer fps from timestamps")

    success = df["success"].to_numpy().astype(bool)
    confidence = df["confidence"].to_numpy()
    kept = success & (confidence >= CONFIDENCE_MIN)
    n_used = int(kept.sum())
    detection_rate = float(n_used / n_total) if n_total else 0.0

    au_r = np.full((n_total, len(AUS)), np.nan)
    au_c = np.zeros((n_total, len(AUS)), dtype=np.int8)
    for j, au in enumerate(AUS):
        r_col, c_col = f"{au}_r", f"{au}_c"
        if r_col in df.columns:
            # Pandas 3.0's `.to_numpy()` can return a read-only view of the Arrow
            # backing storage — force a writable copy before masking.
            r_vals = df[r_col].to_numpy(dtype=float).copy()
            r_vals[~kept] = np.nan
            au_r[:, j] = r_vals
        else:
            warnings.append(f"OpenFace CSV missing column {r_col}")
        if c_col in df.columns:
            c_vals = df[c_col].to_numpy(dtype=np.int8).copy()
            c_vals[~kept] = 0
            au_c[:, j] = c_vals
        else:
            warnings.append(f"OpenFace CSV missing column {c_col}")

    quality_gate_pass = detection_rate >= QUALITY_GATE_MIN
    if not quality_gate_pass:
        warnings.append(
            f"detection rate {detection_rate:.2%} below {QUALITY_GATE_MIN:.0%} quality gate; "
            "downstream score reliability is degraded"
        )

    meta = {
        "fps": fps,
        "n_frames_total": n_total,
        "n_frames_used": n_used,
        "detection_rate": detection_rate,
        "quality_gate_pass": quality_gate_pass,
        "warnings": warnings,
    }
    return {"au_r": au_r, "au_c": au_c}, meta


def extract_smile_features(
    video_path: str | Path,
    t0: float | None = None,
    t1: float | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Extract per-frame AU intensity + presence via OpenFace Docker.

    Returns:
      arrays: {
        'au_r': np.ndarray[N_frames, 7],  # AU intensity 0-5, NaN on dropped frames
        'au_c': np.ndarray[N_frames, 7],  # AU presence 0/1, 0 on dropped frames
      }
      meta: {'fps', 'n_frames_total', 'n_frames_used', 'detection_rate',
             'quality_gate_pass', 'warnings'}
    AU order matches AUS constant (AU01, AU06, AU12, AU14, AU25, AU26, AU45).
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    with tempfile.TemporaryDirectory(prefix="parkscreen_of_") as tmp:
        work = Path(tmp)
        clip_name = "clip.mp4"
        _cut_segment(video_path, work / clip_name, t0, t1)
        csv_path = _run_openface(work, clip_name)
        return _parse_openface_csv(csv_path)


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", help="Path to a video containing the smile task.")
    p.add_argument("--t0", type=float, default=None, help="Start of smile segment (seconds).")
    p.add_argument("--t1", type=float, default=None, help="End of smile segment (seconds).")
    args = p.parse_args()

    arrays, meta = extract_smile_features(args.video, args.t0, args.t1)

    print(f"fps: {meta['fps']:.2f}")
    print(f"frames total: {meta['n_frames_total']}")
    print(f"frames used:  {meta['n_frames_used']} ({meta['detection_rate']:.1%})")
    print(f"quality gate pass: {meta['quality_gate_pass']}")
    for w in meta["warnings"]:
        print(f"  ⚠ {w}")
    print()
    print("Per-AU active-frame stats (active-frame-only, per Islam 2023):")
    au_r, au_c = arrays["au_r"], arrays["au_c"]
    for j, au in enumerate(AUS):
        active = au_c[:, j] == 1
        n_active = int(active.sum())
        if n_active == 0:
            print(f"  {au}: never active")
            continue
        vals = au_r[active, j]
        print(
            f"  {au}: active {n_active:4d}/{meta['n_frames_used']:<4d} frames  "
            f"mean(active) = {np.nanmean(vals):.3f}  var(active) = {np.nanvar(vals):.3f}"
        )


if __name__ == "__main__":
    _main()
