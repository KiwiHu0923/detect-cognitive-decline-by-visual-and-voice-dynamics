"""Build a denormalized labels.csv from NeuroVoz's per-file metadata CSVs.

One row per audio file. HC subjects with age < 50 or missing age are dropped
(strict age-matching against PD floor). subject_id is namespaced by group
because HC and PD share the same integer ID space.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def parse_task_from_audio_path(audio_rel_path: str) -> str:
    """`data/audios/HC_PAN_VINO_0034.wav` → `PAN_VINO`.

    Filenames are `{group}_{task}_{subject}.wav`; task may itself contain
    underscores (PAN_VINO, PATATA_BLANDA, PETACA_BLANCA), so a naive split
    is wrong — the task is everything between the first and last token.
    """
    stem = Path(audio_rel_path).stem
    parts = stem.split("_")
    return "_".join(parts[1:-1])


def _prepare(csv_path: Path, group: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    # NeuroVoz metadata quirk: the spontaneous-speech task is labelled
    # "ESPONTANEA" in Audio paths but the on-disk files are "FREE". Fix at
    # ingest so the file_path column always points to a real file.
    df["Audio"] = df["Audio"].astype(str).str.replace(
        "_ESPONTANEA_", "_FREE_", regex=False
    )
    df["subject_id"] = f"{group}_" + df["ID"].astype(str)
    df["task"] = df["Audio"].apply(parse_task_from_audio_path)
    df["file_path"] = "data/raw/neurovoz/" + df["Audio"]
    return df


def build_labels(
    hc_csv: Path, pd_csv: Path, hc_age_min: float = 50.0
) -> tuple[pd.DataFrame, dict]:
    hc = _prepare(hc_csv, "HC")
    pdf = _prepare(pd_csv, "PD")

    common = ["subject_id", "Group", "Age", "Sex", "task", "file_path"]
    pd_extra = ["UPDRS scale", "H-Y Stadium", "Time Disease (years)", "Medication status"]

    hc_sub = hc[common].copy()
    for c in pd_extra:
        hc_sub[c] = pd.NA
    pd_sub = pdf[common + pd_extra].copy()

    combined = pd.concat([hc_sub, pd_sub], ignore_index=True)

    hc_mask = combined["Group"] == "HC"
    dropped_by_age = int((hc_mask & (combined["Age"] < hc_age_min)).sum())
    dropped_no_age = int((hc_mask & combined["Age"].isna()).sum())

    keep = (combined["Group"] == "PD") | (hc_mask & (combined["Age"] >= hc_age_min))
    combined = combined[keep].copy()

    combined = combined.rename(
        columns={
            "Group": "group",
            "Age": "age",
            "Sex": "sex",
            "UPDRS scale": "updrs",
            "H-Y Stadium": "hy_stage",
            "Time Disease (years)": "disease_years",
            "Medication status": "medication_status",
        }
    )

    stats = {
        "hc_dropped_by_age": dropped_by_age,
        "hc_dropped_no_age": dropped_no_age,
        "rows_out": len(combined),
    }
    return combined, stats


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    paths = yaml.safe_load((repo_root / "configs/paths.yaml").read_text())

    hc_csv = repo_root / paths["neurovoz_metadata_hc"]
    pd_csv = repo_root / paths["neurovoz_metadata_pd"]
    out_path = repo_root / paths["neurovoz_labels"]

    model_cfg = yaml.safe_load((repo_root / "configs/model.yaml").read_text())
    hc_age_min = float(model_cfg["cohort"]["hc_age_min"])

    labels, stats = build_labels(hc_csv, pd_csv, hc_age_min=hc_age_min)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_path, index=False)

    n_pd = labels[labels["group"] == "PD"]["subject_id"].nunique()
    n_hc = labels[labels["group"] == "HC"]["subject_id"].nunique()
    print(f"Wrote {stats['rows_out']} rows → {out_path.relative_to(repo_root)}")
    print(f"Subjects: {n_pd} PD, {n_hc} HC")
    print(
        f"HC filter (age >= {hc_age_min:g}): dropped "
        f"{stats['hc_dropped_by_age']} rows below age, "
        f"{stats['hc_dropped_no_age']} rows with missing age"
    )

    missing = [p for p in labels["file_path"] if not (repo_root / p).exists()]
    if missing:
        print(f"\nWARNING: {len(missing)} rows point to non-existent files")
        for p in missing[:5]:
            print(f"  {p}")
    else:
        print("\nAll file_path entries resolve to existing files ✓")

    coverage = labels.pivot_table(
        index="task",
        columns="group",
        values="subject_id",
        aggfunc="nunique",
        fill_value=0,
    )
    print("\nUnique subjects per task:")
    print(coverage.to_string())


if __name__ == "__main__":
    main()
