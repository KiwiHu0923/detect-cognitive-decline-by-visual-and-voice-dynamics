"""Day-3 figures: ROC curves + coefficient magnitudes.

Reads the CSVs written by `eval.ablation` and produces:
  - eval/results/figures/roc_curves.png
  - eval/results/figures/coefficients.png

Pure matplotlib, no seaborn dependency. Idempotent — safe to rerun.

Run:
    .venv/bin/python -m eval.make_plots
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

from src.fusion.statistical import (
    default_channel_specs,
    load_analysis_cohort,
    load_channel_matrix,
    make_pipeline_l1,
)

# sklearn 1.8 flags penalty= as deprecated in favour of the l1_ratio API. We keep
# liblinear+penalty="l1" for reliability at small N — the deprecation is not
# scheduled to break until 1.10. Silence both the FutureWarning and the paired
# UserWarning ("penalty=l1 with l1_ratio=0.0").
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_paths() -> dict:
    with open(REPO_ROOT / "configs" / "paths.yaml") as f:
        return yaml.safe_load(f)


def refit_phonation_l1(paths: dict, C: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Refit phonation LogReg with L1 penalty for coefficient interpretation.

    Fits on the full analysis cohort (same data as the L2 deployment model,
    per-subject aggregated features). Returns a coefficients frame in the
    same shape as `coefficients.csv` so the plot can drop it in.
    """
    cohort = load_analysis_cohort(paths)
    spec = default_channel_specs(paths)["phonation"]
    X, y, _ = load_channel_matrix(spec, cohort)
    pipe = make_pipeline_l1(C=C, seed=seed)
    pipe.fit(X, y)
    coef = pipe.named_steps["clf"].coef_[0]
    n_nonzero = int((coef != 0).sum())
    print(f"[L1 phonation] C={C}: {n_nonzero}/{len(coef)} features retained")
    return pd.DataFrame({
        "channel": "phonation",
        "feature": spec.feature_cols,
        "coefficient": coef,
    })


def plot_roc_curves(oof: pd.DataFrame, out_path: Path) -> None:
    """Three-line ROC comparison — the visual argument for the ablation story.

    Baseline phonation (dashed) → DDK-only (mid) → best fusion (thick).
    Chance line as reference. AUC + 95% CI in the legend.
    """
    y = oof["y_true"].to_numpy()

    # (label, prob_col, style) — order controls legend order
    lines = [
        ("Phonation-only (per-subject mean)", "phonation_prob",
         {"color": "#888888", "linestyle": "--", "linewidth": 1.5}),
        ("DDK-only (PATAKA)", "ddk_prob",
         {"color": "#1f77b4", "linestyle": "-", "linewidth": 2.0}),
        ("Phonation(per-file) + DDK (AUC-excess weighted)",
         "fusion_per_file_weighted_prob",
         {"color": "#d62728", "linestyle": "-", "linewidth": 2.6}),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for label, col, style in lines:
        prob = oof[col].to_numpy()
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        ax.plot(fpr, tpr, label=f"{label}   AUC={auc:.3f}", **style)

    # Chance diagonal
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linestyle=":", linewidth=1.0,
            label="Chance (AUC=0.500)")

    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("Subject-level LOSO ROC — NeuroVoz analysis cohort (49 PD × 46 HC)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path.relative_to(REPO_ROOT)}")


def plot_coefficients(coef_df: pd.DataFrame, out_path: Path) -> None:
    """Two horizontal-bar subplots — phonation on the left, DDK on the right.

    Bars sorted by |coefficient| descending. Colour encodes direction:
      red   = positive coefficient → higher feature value pushes toward PD
      blue  = negative coefficient → higher feature value pushes toward HC

    Phonation is expected to be refit with L1 penalty upstream — see
    `refit_phonation_l1`. DDK uses the L2 deployment coefficients (8 features,
    no material multicollinearity).
    """
    channels = ["phonation", "ddk"]
    titles = {
        "phonation": "Phonation channel — L1 penalty (feature selection)",
        "ddk": "DDK channel — L2 penalty (deployment model)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                             gridspec_kw={"width_ratios": [3, 2]})

    for ax, ch in zip(axes, channels):
        sub = coef_df[coef_df["channel"] == ch].copy()
        sub["abs"] = sub["coefficient"].abs()
        sub = sub.sort_values("abs", ascending=True)  # ascending → largest on top
        colors = [
            "#d62728" if c > 0 else "#1f77b4" if c < 0 else "#dddddd"
            for c in sub["coefficient"]
        ]
        ax.barh(sub["feature"], sub["coefficient"], color=colors, edgecolor="black",
                linewidth=0.4)
        ax.axvline(0, color="#333333", linewidth=0.6)
        ax.set_title(titles[ch], fontsize=11)
        ax.set_xlabel("LogReg coefficient (standardized features)")
        ax.grid(axis="x", alpha=0.3)

    # Shared legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#d62728", edgecolor="black", label="Positive → pushes toward PD"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="Negative → pushes toward HC"),
        Patch(facecolor="#dddddd", edgecolor="black", label="Zero (L1 dropped by regularization)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.suptitle("Per-channel LogReg coefficients — full analysis cohort (49 PD × 46 HC)",
                 fontsize=12, y=1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    paths = _load_paths()
    results_dir = REPO_ROOT / paths["results_dir"]
    figures_dir = REPO_ROOT / paths["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(results_dir / "loso_oof_probs.csv")
    coef_df = pd.read_csv(results_dir / "coefficients.csv")

    # Swap phonation L2 coefficients for L1-refit coefficients (viz-only).
    # C=0.3 tunes L1 sparsity: at C=1.0 both shimmer_apq5 and shimmer_apq11
    # survive with opposite signs (collinearity artefact — same 4-shimmer
    # multicollinearity problem the plot exists to escape). Tighter C forces
    # the L1 loss to pick one representative from each collinear cluster.
    l1_phon = refit_phonation_l1(paths, C=0.3)
    l1_out = results_dir / "coefficients_l1_phonation.csv"
    l1_phon.to_csv(l1_out, index=False)
    print(f"Saved {l1_out.relative_to(REPO_ROOT)}")
    coef_df = pd.concat(
        [coef_df[coef_df["channel"] != "phonation"], l1_phon],
        ignore_index=True,
    )

    plot_roc_curves(oof, figures_dir / "roc_curves.png")
    plot_coefficients(coef_df, figures_dir / "coefficients.png")


if __name__ == "__main__":
    main()
