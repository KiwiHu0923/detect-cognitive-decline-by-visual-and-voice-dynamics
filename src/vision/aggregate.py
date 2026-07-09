"""Aggregate per-frame OpenFace outputs into a session-level feature vector.

Implements the "active-frame-only" aggregation rule from Islam et al. 2023
("Unmasking Parkinson's Disease with Smile", arxiv 2308.02588 §2): for each
Action Unit, mean and variance are computed only over frames where the AU's
binary presence flag `AU_c == 1`, not over every frame in the clip.

Zero-fill convention for degenerate cases (paper is silent — see CLAUDE.md
Facial Features section — this is our documented convention that matches the
observed distribution of `smile_*_var` == 0 values in UFNet's released CSV):

  * No active frames for an AU: `mean = 0.0`, `var = 0.0`.
  * Exactly one active frame: `mean = that_value`, `var = 0.0`.
  * Two or more active frames: standard `np.nanmean` / `np.nanvar`.

Column order in the returned vector follows the caller-supplied `columns` list,
which should be `eval/models/smile_pd_columns.json` (14 entries, interleaved
AU01_mean, AU01_var, AU06_mean, AU06_var, ..., AU45_var).
"""

from __future__ import annotations

import numpy as np

from src.vision.facial_features import AUS as EXTRACTED_AUS


def _parse_column(col: str) -> tuple[str, str]:
    """`smile_AU12_mean` -> ('AU12', 'mean'). Raises on unexpected format."""
    parts = col.split("_")
    if len(parts) != 3 or parts[0] != "smile" or not parts[1].startswith("AU"):
        raise ValueError(
            f"unsupported column name {col!r}; expected `smile_AU<NN>_<stat>` "
            f"(e.g. `smile_AU12_mean`). Geometric or entropy columns are not "
            f"supported by this 14-feature aggregator."
        )
    stat = parts[2]
    if stat not in {"mean", "var"}:
        raise ValueError(f"unsupported statistic {stat!r} in column {col!r}")
    return parts[1], stat


def aggregate(
    arrays: dict[str, np.ndarray],
    columns: list[str],
) -> tuple[np.ndarray, dict]:
    """Reduce per-frame AU intensity/presence arrays to a 14-dim feature vector.

    Args:
      arrays: {'au_r': np.ndarray[N_frames, K], 'au_c': np.ndarray[N_frames, K]}
        where the AU dimension is ordered as `facial_features.AUS`.
      columns: canonical output column order (typically loaded from
        `smile_pd_columns.json`, 14 entries).

    Returns:
      vector: np.ndarray[len(columns)], one value per output column.
      meta: {
        'per_au_active_frames': {'AU01': int, 'AU06': int, ...},
        'zero_filled_columns': list[str],   # output columns that were zero-filled
                                            # because the AU never fired
      }
    """
    au_r = arrays["au_r"]
    au_c = arrays["au_c"]
    if au_r.shape != au_c.shape:
        raise ValueError(f"au_r shape {au_r.shape} != au_c shape {au_c.shape}")
    if au_r.shape[1] != len(EXTRACTED_AUS):
        raise ValueError(
            f"array AU dimension {au_r.shape[1]} != len(EXTRACTED_AUS) {len(EXTRACTED_AUS)}"
        )

    au_to_idx = {au: i for i, au in enumerate(EXTRACTED_AUS)}

    per_au_active_frames: dict[str, int] = {}
    per_au_stats: dict[str, dict[str, float]] = {}
    never_active_aus: set[str] = set()

    for au in EXTRACTED_AUS:
        j = au_to_idx[au]
        active = au_c[:, j] == 1
        n_active = int(active.sum())
        per_au_active_frames[au] = n_active
        if n_active == 0:
            per_au_stats[au] = {"mean": 0.0, "var": 0.0}
            never_active_aus.add(au)
        else:
            vals = au_r[active, j]
            # `np.nanmean` / `np.nanvar` are defensive: NaN slots should already
            # be filtered by au_c==1 (facial_features.py zeroes au_c on dropped
            # frames), but the paper is silent so we don't want a stray NaN to
            # poison the vector.
            per_au_stats[au] = {
                "mean": float(np.nanmean(vals)),
                "var": float(np.nanvar(vals)),  # ddof=0, matches numpy default
            }

    vector = np.zeros(len(columns), dtype=float)
    zero_filled_columns: list[str] = []
    for i, col in enumerate(columns):
        au, stat = _parse_column(col)
        if au not in per_au_stats:
            raise ValueError(
                f"column {col!r} references AU {au!r}, not present in the extracted "
                f"AU set {EXTRACTED_AUS}. Did facial_features.AUS change?"
            )
        vector[i] = per_au_stats[au][stat]
        if au in never_active_aus:
            zero_filled_columns.append(col)

    if not np.isfinite(vector).all():
        bad = [columns[i] for i, v in enumerate(vector) if not np.isfinite(v)]
        raise ValueError(f"aggregate produced non-finite value in columns: {bad}")

    return vector, {
        "per_au_active_frames": per_au_active_frames,
        "zero_filled_columns": zero_filled_columns,
    }
