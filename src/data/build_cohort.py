"""Build the subject-level analysis cohort from labels.csv.

One row per subject. Downstream feature-extraction / modelling code joins this
with labels.csv on subject_id to recover per-task file paths. Subjects failing
the analysis-cohort filter (no PATAKA or no vowel files) are kept in the CSV
with `in_analysis_cohort = False` — preserves the audit trail.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

VOWEL_RE = re.compile(r"^[AEIOU][123]$")
NON_READING_TASKS = {"PATAKA", "FREE"}


def _summarise_subject(g: pd.DataFrame) -> pd.Series:
    tasks = set(g["task"])
    vowel_tasks = sorted(t for t in tasks if VOWEL_RE.match(t))
    reading_tasks = [
        t for t in tasks if not VOWEL_RE.match(t) and t not in NON_READING_TASKS
    ]
    first = g.iloc[0]
    return pd.Series(
        {
            "group": first["group"],
            "age": first["age"],
            "sex": first["sex"],
            "updrs": first["updrs"],
            "hy_stage": first["hy_stage"],
            "disease_years": first["disease_years"],
            "medication_status": first["medication_status"],
            "has_pataka": "PATAKA" in tasks,
            "vowel_tasks": ",".join(vowel_tasks),
            "n_vowel_files": len(vowel_tasks),
            "has_free": "FREE" in tasks,
            "n_reading_files": len(reading_tasks),
        }
    )


def build_cohort(labels: pd.DataFrame) -> pd.DataFrame:
    cohort = (
        labels.groupby("subject_id", sort=False)
        .apply(_summarise_subject, include_groups=False)
        .reset_index()
    )
    cohort["in_analysis_cohort"] = cohort["has_pataka"] & (cohort["n_vowel_files"] >= 1)
    return cohort


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    paths = yaml.safe_load((repo_root / "configs/paths.yaml").read_text())

    labels = pd.read_csv(repo_root / paths["neurovoz_labels"])
    cohort = build_cohort(labels)

    out_path = repo_root / paths["cohort_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out_path, index=False)

    n = len(cohort)
    in_c = cohort["in_analysis_cohort"]
    n_pd = int(((cohort["group"] == "PD") & in_c).sum())
    n_hc = int(((cohort["group"] == "HC") & in_c).sum())
    print(f"Wrote {n} subject rows → {out_path.relative_to(repo_root)}")
    print(f"Analysis cohort: {int(in_c.sum())} subjects ({n_pd} PD, {n_hc} HC)")
    print("  target: ≈ 49 PD × 46 HC (per earlier task-coverage estimate)")

    excluded = cohort[~in_c]
    if len(excluded):
        print(f"\nExcluded {len(excluded)} subjects:")
        for _, row in excluded.iterrows():
            reasons = []
            if not row["has_pataka"]:
                reasons.append("no PATAKA")
            if row["n_vowel_files"] == 0:
                reasons.append("no vowels")
            age = row["age"]
            age_str = f"age {age:.0f}" if pd.notna(age) else "age NA"
            print(f"  {row['subject_id']:<8} ({row['group']}, {age_str}): {', '.join(reasons)}")

    print("\nVowel coverage across analysis cohort (subjects, per vowel task):")
    kept = cohort[in_c]
    all_vowels = sorted(
        {v for row in kept["vowel_tasks"] for v in row.split(",") if v}
    )
    for v in all_vowels:
        n_pd_v = int(kept[kept["group"] == "PD"]["vowel_tasks"].str.contains(rf"\b{v}\b").sum())
        n_hc_v = int(kept[kept["group"] == "HC"]["vowel_tasks"].str.contains(rf"\b{v}\b").sum())
        print(f"  {v:<3} PD={n_pd_v:<3} HC={n_hc_v}")


if __name__ == "__main__":
    main()
