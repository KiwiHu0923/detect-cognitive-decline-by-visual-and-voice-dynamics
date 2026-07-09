"""Day 2 QA — feature distribution plots + cross-check vs NeuroVoz's shipped features.

Two jobs, run in sequence:

1. Boxplot HC vs PD for 6 phonation + 6 DDK features (3x4 grid) →
   eval/results/figures/feature_distributions.png. Print a per-feature
   NaN/inf/min/max table.

2. Load NeuroVoz's shipped audio_features.csv, discover its schema, join
   against our per-file phonation features on the filename, and report
   Pearson correlations on overlapping feature names. r > 0.9 = compatible
   extraction; 0.7–0.9 = same feature but different params; < 0.7 = investigate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PHONATION_PLOT = [
    "jitter_local",
    "shimmer_local",
    "hnr_mean_db",
    "f0_mean_hz",
    "f0_std_hz",
    "voicing_fraction",
]
DDK_PLOT = [
    "ddk_rate_hz",
    "isi_cv",
    "amp_mean",
    "amp_cv",
    "amp_decrement",
    "duration_s",
]


def _qa_table(df: pd.DataFrame, features: list[str], label: str) -> None:
    print(f"\n{label} (n={len(df)}):")
    header = f"  {'feature':<20} {'NaN':>5} {'Inf':>5} {'min':>12} {'max':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for f in features:
        col = df[f]
        n_nan = int(col.isna().sum())
        finite = col[np.isfinite(col)]
        n_inf = int((~np.isfinite(col.dropna())).sum())
        vmin = float(finite.min()) if len(finite) else float("nan")
        vmax = float(finite.max()) if len(finite) else float("nan")
        print(f"  {f:<20} {n_nan:>5} {n_inf:>5} {vmin:>12.4f} {vmax:>12.4f}")


def plot_distributions(
    phon_df: pd.DataFrame, ddk_df: pd.DataFrame, out_path: Path
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    features = [("phonation", f, phon_df) for f in PHONATION_PLOT] + [
        ("ddk", f, ddk_df) for f in DDK_PLOT
    ]
    for i, (channel, feat, src) in enumerate(features):
        ax = axes[i]
        hc = src.loc[src["group"] == "HC", feat].dropna()
        pd_ = src.loc[src["group"] == "PD", feat].dropna()
        ax.boxplot([hc, pd_], tick_labels=["HC", "PD"], showfliers=True)
        ax.set_title(f"{channel}: {feat}", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Feature distributions by group (analysis cohort: 49 PD × 46 HC)",
        fontsize=14,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


FEATURE_MAP = {
    # our column → NeuroVoz column (best-effort mapping)
    "jitter_local": "rJitter",   # relative jitter, unitless
    "jitter_rap": "RAP",         # relative average perturbation
    "jitter_ppq5": "rPPQ",       # 5-point period perturbation quotient
    "shimmer_local": "rShimmer", # relative shimmer, unitless
    "hnr_mean_db": "HNR",        # HNR — note: their sign convention may differ
}


def cross_check_neurovoz(
    our_perfile: pd.DataFrame, neurovoz_csv: Path
) -> None:
    if not neurovoz_csv.exists():
        print(f"\nNeuroVoz audio_features.csv not found at {neurovoz_csv} — skipping.")
        return

    # Their CPP column contains per-row numpy-array text dumps with embedded
    # newlines that confuse the parser. Drop it at read time with usecols to
    # avoid the quoting mess entirely.
    keep_cols = [c for c in ["AudioPath", *FEATURE_MAP.values()]]
    nv = pd.read_csv(neurovoz_csv, usecols=keep_cols)
    print(
        f"\nNeuroVoz shipped audio_features.csv: {nv.shape[0]} rows × {nv.shape[1]} cols (CPP dropped)"
    )

    nv["basename"] = nv["AudioPath"].astype(str).str.rsplit("/", n=1).str[-1]
    our = our_perfile.copy()
    # Reconstruct NeuroVoz-style basename from subject_id + task (per_file
    # doesn't carry file_path; NeuroVoz names files {group}_{task}_{id:04d}.wav).
    group_id = our["subject_id"].str.split("_", n=1, expand=True)
    our["basename"] = (
        group_id[0] + "_" + our["task"].astype(str) + "_"
        + group_id[1].astype(int).astype(str).str.zfill(4) + ".wav"
    )

    merged = our.merge(nv, on="basename", how="inner", suffixes=("_ours", "_nv"))
    print(f"Joined on filename: {len(merged)} matching rows (of {len(our)} ours, {len(nv)} theirs)")

    if not len(merged):
        print("No overlap — inspect basename samples:")
        print("  ours:", our["basename"].head(3).tolist())
        print("  theirs:", nv["basename"].head(3).tolist())
        return

    print(
        f"\nPearson correlation (n={len(merged)}), our column ↔ NeuroVoz column:"
    )
    print(f"  {'our_col':<18} {'nv_col':<12} {'r':>7}  {'ours mean±sd':>18}  {'theirs mean±sd':>18}")
    print("  " + "-" * 82)
    for our_col, nv_col in FEATURE_MAP.items():
        a = merged[our_col]
        b = merged[nv_col]
        mask = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            print(f"  {our_col:<18} {nv_col:<12} {'--':>7} (n<3 valid)")
            continue
        r = float(np.corrcoef(a[mask], b[mask])[0, 1])
        ours_s = f"{a[mask].mean():.4f}±{a[mask].std():.4f}"
        nv_s = f"{b[mask].mean():.4f}±{b[mask].std():.4f}"
        print(f"  {our_col:<18} {nv_col:<12} {r:>7.3f}  {ours_s:>18}  {nv_s:>18}")

    print(
        "\nInterpretation: r > 0.9 = compatible extraction; 0.7–0.9 = same feature, "
        "different Praat params; < 0.7 = investigate."
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    paths = yaml.safe_load((repo_root / "configs/paths.yaml").read_text())

    phon_per_subject = pd.read_csv(
        repo_root / paths["phonation_features_dir"] / "per_subject.csv"
    )
    phon_per_file = pd.read_csv(
        repo_root / paths["phonation_features_dir"] / "per_file.csv"
    )
    ddk_per_subject = pd.read_csv(
        repo_root / paths["ddk_features_dir"] / "per_subject.csv"
    )

    # Job 1: QA tables + boxplot
    _qa_table(phon_per_subject, PHONATION_PLOT, "Phonation per-subject QA")
    _qa_table(ddk_per_subject, DDK_PLOT, "DDK per-subject QA")

    fig_path = repo_root / paths["figures_dir"] / "feature_distributions.png"
    plot_distributions(phon_per_subject, ddk_per_subject, fig_path)

    # Job 2: NeuroVoz cross-check (per-file, since we join on filename)
    nv_csv = repo_root / "data/raw/neurovoz/data/audio_features/audio_features.csv"
    cross_check_neurovoz(phon_per_file, nv_csv)  # per_file has file_path column


if __name__ == "__main__":
    main()
