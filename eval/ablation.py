"""
Produces the table shown to evaluators. Four rows:
  1. Phonation-only
  2. DDK-only
  3. Phonation + DDK (unweighted average fusion) — the honest baseline
  4. Phonation + DDK (weighted, weights from configs/model.yaml) — the deployed setting

LOSO details:
- Analysis cohort: subjects with `in_analysis_cohort == True` in cohort.csv
  (≈ 49 PD × 46 HC after HC age ≥ 50 filter ∩ has PATAKA ∩ has ≥1 vowel).
- Per fold: hold out one subject; scale + fit on the remaining N-1; predict prob for
  the held-out subject. Every subject gets exactly one out-of-fold probability.
- Metrics on the assembled OOF vector: AUC, F1, sensitivity, specificity.
- Subject-level bootstrap 95% CI (1000 resamples) on AUC.

Also fits final classifiers on the full cohort and saves them to `eval/models/`
for Layer-2 demo consumption. LOSO folds are for evaluation; the demo uses the
full-cohort refit.

Run:
    .venv/bin/python -m eval.ablation
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.fusion.statistical import (
    REPO_ROOT,
    default_channel_specs,
    fit_final_channel_model,
    fuse_probs,
    load_analysis_cohort,
    load_channel_matrix,
    load_model_config,
    load_paths,
    load_phonation_per_file,
    make_pipeline,
    save_channel_model,
)


def loso_probs(
    X: np.ndarray, y: np.ndarray, C: float = 1.0, seed: int = 42
) -> np.ndarray:
    """Return per-subject out-of-fold PD probabilities via leave-one-subject-out CV."""
    n = len(y)
    probs = np.zeros(n, dtype=float)
    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        pipe = make_pipeline(C=C, seed=seed)
        pipe.fit(X[train_mask], y[train_mask])
        probs[i] = pipe.predict_proba(X[i:i + 1])[0, 1]
    return probs


def loso_probs_grouped(
    X: np.ndarray, y: np.ndarray, groups: list[str],
    C: float = 1.0, seed: int = 42,
) -> np.ndarray:
    """LeaveOneGroupOut LOSO: each fold holds out all rows sharing a group id.

    Used for per-file phonation where the group is subject_id — all of one
    subject's vowel files are held out together (no within-subject leakage),
    but each held-out file receives its own OOF probability.
    """
    probs = np.zeros(len(y), dtype=float)
    groups_arr = np.asarray(groups)
    for g in pd.unique(groups_arr):
        test_mask = groups_arr == g
        train_mask = ~test_mask
        pipe = make_pipeline(C=C, seed=seed)
        pipe.fit(X[train_mask], y[train_mask])
        probs[test_mask] = pipe.predict_proba(X[test_mask])[:, 1]
    return probs


def aggregate_per_file_to_subject(
    file_probs: np.ndarray, file_subject_ids: list[str],
    subject_order: list[str],
) -> np.ndarray:
    """Mean of per-file probs, one row per subject in `subject_order`.

    Turns a per-file OOF vector (466 entries) into a per-subject OOF vector
    (95 entries) that lines up with the existing per-subject baseline for
    fusion and per-subject-eval AUC.
    """
    df = pd.DataFrame({"subject_id": file_subject_ids, "prob": file_probs})
    per_subject = df.groupby("subject_id")["prob"].mean()
    return per_subject.reindex(subject_order).to_numpy()


def metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "n_pd": int((y_true == 1).sum()),
        "n_hc": int((y_true == 0).sum()),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def bootstrap_auc_ci(
    y_true: np.ndarray, y_prob: np.ndarray,
    n_bootstrap: int = 1000, seed: int = 42, alpha: float = 0.05,
) -> tuple[float, float]:
    """Subject-level bootstrap 95% CI on AUC.

    Resamples subjects with replacement. Skips resamples that end up with a
    single class (AUC undefined).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    lo = float(np.quantile(aucs, alpha / 2))
    hi = float(np.quantile(aucs, 1 - alpha / 2))
    return lo, hi


def _print_row(name: str, m: dict, ci: tuple[float, float]) -> None:
    print(
        f"[{name:38s}] n={m['n']:3d} (PD={m['n_pd']:2d}/HC={m['n_hc']:2d})  "
        f"AUC={m['auc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  "
        f"F1={m['f1']:.3f}  sens={m['sensitivity']:.3f}  spec={m['specificity']:.3f}"
    )


def run(
    C: float = 1.0, seed: int = 42, n_bootstrap: int = 1000, save: bool = True,
) -> pd.DataFrame:
    paths = load_paths()
    model_cfg = load_model_config()
    fusion_weights = model_cfg["fusion"]["weights"]

    cohort = load_analysis_cohort(paths)
    specs = default_channel_specs(paths)

    print(f"Analysis cohort: {len(cohort)} subjects "
          f"({(cohort['group']=='PD').sum()} PD / {(cohort['group']=='HC').sum()} HC)")
    print(f"LOSO folds: {len(cohort)} per model. Bootstrap resamples: {n_bootstrap}.\n")

    channel_data = {}
    channel_probs = {}
    for ch_name, spec in specs.items():
        X, y, subject_ids = load_channel_matrix(spec, cohort)
        print(f"[{ch_name}] X shape {X.shape}, {int(y.sum())} PD / {int((y==0).sum())} HC — running LOSO...")
        probs = loso_probs(X, y, C=C, seed=seed)
        channel_data[ch_name] = {
            "X": X, "y": y, "subject_ids": subject_ids,
            "feature_cols": spec.feature_cols,
        }
        channel_probs[ch_name] = probs

    # Sanity: all channels share the same subject ordering (by cohort merge order).
    ref_ids = channel_data["phonation"]["subject_ids"]
    for ch, d in channel_data.items():
        assert d["subject_ids"] == ref_ids, f"Subject order mismatch in {ch}"
    y = channel_data["phonation"]["y"]

    rows = []
    print()

    # Single-channel rows
    for ch in ("phonation", "ddk"):
        m = metrics(y, channel_probs[ch])
        ci = bootstrap_auc_ci(y, channel_probs[ch], n_bootstrap=n_bootstrap, seed=seed)
        _print_row(f"{ch}-only", m, ci)
        rows.append({
            "model": f"{ch}-only",
            **m,
            "auc_ci_low": ci[0], "auc_ci_high": ci[1],
        })

    # Fusion: unweighted average
    fused_unw = fuse_probs(channel_probs, weights=None)
    m = metrics(y, fused_unw)
    ci = bootstrap_auc_ci(y, fused_unw, n_bootstrap=n_bootstrap, seed=seed)
    _print_row("phonation + DDK (unweighted avg)", m, ci)
    rows.append({
        "model": "phonation+ddk_unweighted",
        **m,
        "auc_ci_low": ci[0], "auc_ci_high": ci[1],
    })

    # Fusion: weighted (configs/model.yaml, restricted to speech channels)
    speech_weights = {k: fusion_weights[k] for k in ("phonation", "ddk")}
    fused_w = fuse_probs(channel_probs, weights=speech_weights)
    m = metrics(y, fused_w)
    ci = bootstrap_auc_ci(y, fused_w, n_bootstrap=n_bootstrap, seed=seed)
    label = f"phonation+ddk_weighted (p={speech_weights['phonation']}, d={speech_weights['ddk']})"
    _print_row(label, m, ci)
    rows.append({
        "model": "phonation+ddk_weighted",
        **m,
        "auc_ci_low": ci[0], "auc_ci_high": ci[1],
        "weight_phonation": speech_weights["phonation"],
        "weight_ddk": speech_weights["ddk"],
    })

    # --- Per-file phonation variant: subject-level LOSO on 466 vowel files ---
    # Tests two things at once: (a) does removing mean-over-vowels aggregation
    # recover signal? (b) how much does the eval unit (per-file vs per-subject)
    # inflate the AUC?
    X_pf, y_pf, subject_ids_pf, _tasks_pf = load_phonation_per_file(paths, cohort)
    print(f"\n[phonation-per-file] X shape {X_pf.shape}, {int(y_pf.sum())} PD-files / "
          f"{int((y_pf==0).sum())} HC-files — running subject-level LOSO...")
    probs_pf = loso_probs_grouped(X_pf, y_pf, subject_ids_pf, C=C, seed=seed)

    # (a) Per-file evaluation — comparable to NeuroVoz paper baseline
    m = metrics(y_pf, probs_pf)
    ci = bootstrap_auc_ci(y_pf, probs_pf, n_bootstrap=n_bootstrap, seed=seed)
    _print_row("phonation per-file (per-file eval)", m, ci)
    rows.append({
        "model": "phonation_per-file_per-file-eval",
        **m,
        "auc_ci_low": ci[0], "auc_ci_high": ci[1],
    })

    # (b) Aggregate per-file probs back to per-subject via mean, then evaluate.
    # This isolates the training-regime effect from the eval-unit effect.
    probs_pf_agg = aggregate_per_file_to_subject(probs_pf, subject_ids_pf, ref_ids)
    m = metrics(y, probs_pf_agg)
    ci = bootstrap_auc_ci(y, probs_pf_agg, n_bootstrap=n_bootstrap, seed=seed)
    _print_row("phonation per-file (per-subject eval)", m, ci)
    rows.append({
        "model": "phonation_per-file_per-subject-eval",
        **m,
        "auc_ci_low": ci[0], "auc_ci_high": ci[1],
    })

    # Fusion with per-file-trained, per-subject-aggregated phonation + DDK
    fused_unw_pf = fuse_probs(
        {"phonation": probs_pf_agg, "ddk": channel_probs["ddk"]}, weights=None,
    )
    m = metrics(y, fused_unw_pf)
    ci = bootstrap_auc_ci(y, fused_unw_pf, n_bootstrap=n_bootstrap, seed=seed)
    _print_row("phonation(per-file) + DDK (unweighted avg)", m, ci)
    rows.append({
        "model": "phonation-per-file+ddk_unweighted",
        **m,
        "auc_ci_low": ci[0], "auc_ci_high": ci[1],
    })

    fused_w_pf = fuse_probs(
        {"phonation": probs_pf_agg, "ddk": channel_probs["ddk"]}, weights=speech_weights,
    )
    m = metrics(y, fused_w_pf)
    ci = bootstrap_auc_ci(y, fused_w_pf, n_bootstrap=n_bootstrap, seed=seed)
    label_pf = f"phonation(per-file)+ddk_weighted (p={speech_weights['phonation']}, d={speech_weights['ddk']})"
    _print_row(label_pf, m, ci)
    rows.append({
        "model": "phonation-per-file+ddk_weighted",
        **m,
        "auc_ci_low": ci[0], "auc_ci_high": ci[1],
        "weight_phonation": speech_weights["phonation"],
        "weight_ddk": speech_weights["ddk"],
    })

    table = pd.DataFrame(rows)

    if save:
        results_dir = REPO_ROOT / paths["results_dir"]
        results_dir.mkdir(parents=True, exist_ok=True)
        table_path = results_dir / "ablation_table.csv"
        table.to_csv(table_path, index=False)
        print(f"\nSaved ablation table -> {table_path.relative_to(REPO_ROOT)}")

        # Persist per-subject OOF probabilities for downstream plots / notebook.
        # Per-file phonation is stored in its aggregated-to-subject form so every
        # column here shares the same per-subject index.
        oof = pd.DataFrame({
            "subject_id": ref_ids,
            "y_true": y,
            "phonation_prob": channel_probs["phonation"],
            "phonation_per_file_prob": probs_pf_agg,
            "ddk_prob": channel_probs["ddk"],
            "fusion_unweighted_prob": fused_unw,
            "fusion_weighted_prob": fused_w,
            "fusion_per_file_unweighted_prob": fused_unw_pf,
            "fusion_per_file_weighted_prob": fused_w_pf,
        })
        oof_path = results_dir / "loso_oof_probs.csv"
        oof.to_csv(oof_path, index=False)
        print(f"Saved OOF probabilities -> {oof_path.relative_to(REPO_ROOT)}")

        # Fit final full-cohort deployment models for Layer 2.
        models_dir = REPO_ROOT / paths["models_dir"]
        for ch_name, spec in specs.items():
            pipe, meta = fit_final_channel_model(spec, cohort, C=C, seed=seed)
            model_path, _ = save_channel_model(pipe, meta, models_dir)
            print(f"Saved deployment model [{ch_name}] -> {model_path.relative_to(REPO_ROOT)}")

        # Coefficient dump for the coefficients.png plot + interpretability.
        coefs = []
        for ch_name, spec in specs.items():
            pipe, _ = fit_final_channel_model(spec, cohort, C=C, seed=seed)
            clf = pipe.named_steps["clf"]
            for feat, coef in zip(spec.feature_cols, clf.coef_[0]):
                coefs.append({"channel": ch_name, "feature": feat, "coefficient": float(coef)})
        coef_df = pd.DataFrame(coefs)
        coef_path = results_dir / "coefficients.csv"
        coef_df.to_csv(coef_path, index=False)
        print(f"Saved coefficient dump -> {coef_path.relative_to(REPO_ROOT)}")

        summary = {
            "C": C,
            "seed": seed,
            "n_bootstrap": n_bootstrap,
            "cohort_n": int(len(cohort)),
            "cohort_pd": int((cohort['group']=='PD').sum()),
            "cohort_hc": int((cohort['group']=='HC').sum()),
            "fusion_weights_used": speech_weights,
            "models": rows,
        }
        summary_path = results_dir / "ablation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved summary JSON -> {summary_path.relative_to(REPO_ROOT)}")

    return table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--C", type=float, default=1.0, help="LogReg inverse regularization strength")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()
    run(C=args.C, seed=args.seed, n_bootstrap=args.n_bootstrap, save=not args.no_save)


if __name__ == "__main__":
    main()
