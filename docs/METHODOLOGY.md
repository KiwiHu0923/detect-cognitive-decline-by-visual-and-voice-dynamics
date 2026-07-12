# ParkScreen — Methodology and Cohort Audit

This document is the detailed methodology + reproducibility record for ParkScreen. It covers data lineage, feature extraction, model training, cross-validation protocol, sensitivity analyses, cohort audit (bias / confounder checks), and known limitations. Anyone reproducing our numbers or evaluating our claims should read this end to end.

For the higher-level project overview, positioning, and scope rules see `CLAUDE.md`. For the deliverables checklist see `TODO.md`.

---

## 1. Executive Summary

ParkScreen fuses two motor-speech channels (phonation from sustained vowels, articulation from PATAKA DDK) and one facial channel (smile-task classifier + hypomimia narrative) into a late-fusion PD risk score. The scientific validation spine (Layer 1) runs on the NeuroVoz corpus with subject-level leave-one-subject-out cross-validation. The product layer (Layer 2) scores user-uploaded task-matched clips using the fitted Layer-1 classifiers and passes both structured scores and narrative markers to Claude for report generation.

**Headline result (NeuroVoz, subject-level LOSO):**

| Model | AUC [95% CI] | F1 | Sens | Spec |
|-------|--------------|----|------|------|
| Phonation-only (per-subject) | 0.567 [0.455, 0.678] | 0.545 | 0.490 | 0.674 |
| DDK-only (PATAKA) | 0.740 [0.632, 0.845] | 0.701 | 0.694 | 0.696 |
| Phonation(per-file) + DDK (AUC-excess weighted) | **0.758 [0.662, 0.859]** | 0.667 | 0.633 | 0.717 |

For rendered figures (ROC curves, coefficient bar plots) and executable reproduction, see [`notebooks/ablation_results.ipynb`](../notebooks/ablation_results.ipynb).

**Key caveats up front:**
1. PD training subjects were recorded ON medication (2–5h post-dose). Numbers apply to medicated PD vs age-matched HC.
2. PD subjects are 5 years older than HC on average (70.8 vs 65.8, Cohen's d = 0.518, p = 0.013) despite the HC age ≥ 50 filter. Age is a known confounder for voice-quality features.
3. PD cohort is mild-to-moderate severity (H-Y stage 2 dominant, UPDRS median 14, max 47). AUC does not extrapolate to severe PD.
4. Screening decision aid, NOT a diagnosis.

---

## 2. Data Lineage

### 2.1 Raw source

**NeuroVoz** (Spanish PD voice corpus, Nature Sci Data 2024, Zenodo record — DOI: https://doi.org/10.1038/s41597-024-04186-z, DUA cleared 2026-07-08). See [`DATASETS.md`](DATASETS.md) for full citation.

- ~2,976 WAV files across 53 PD + 55 HC subjects
- Three task families: sustained vowels (A/E/I/O/U, up to 3 reps each), PATAKA DDK (/pa-ta-ka/), 20+ Spanish word/sentence tasks + FREE spontaneous speech
- Location: `data/raw/neurovoz/data/{audios,metadata,transcriptions,grbas,audio_features}/`
- Subject metadata at `data/raw/neurovoz/data/metadata/{metadata_pd,metadata_hc}.csv`
- Per-file GRBAS perceptual ratings at `data/raw/neurovoz/data/grbas/` (used for narrative, not training)

### 2.2 Cohort construction

Two filters applied in order:

1. **Age filter (HC only): HC age ≥ 50.** Drops 4 HC subjects: 2 missing age, 2 aged 31 and 38.
   → 53 PD + 51 age-matched HC (raw age-matched cohort)
2. **Task-intersection filter: subject must have PATAKA AND ≥ 1 vowel file.**
   → **49 PD + 46 HC = 95 subjects (analysis cohort)**

Excluded 9 subjects: 8 missing PATAKA (5 HC, 3 PD), 1 (PD_108) missing both PATAKA and all vowels.

Excluded subjects are kept in `data/processed/cohort.csv` with `in_analysis_cohort=False` for audit trail.

Built by `python -m src.data.build_cohort` from `data/raw/neurovoz/labels.csv` (which itself is built by `python -m src.data.build_labels`).

**Data quirk fixed at ingest:** NeuroVoz metadata references task name "ESPONTANEA" but files are named "FREE". The labels loader rewrites `_ESPONTANEA_` → `_FREE_`.

### 2.3 Per-channel filters within the cohort

**Phonation channel (post 2026-07-10 refactor):**
- Vowel filter: only /i/, /o/, /u/ used (drops /a/ and /e/ per Li et al. arxiv 2606.19125 Table IV, where /a/=0.58 and /e/=0.63 person-AUC are barely above chance vs /i/=0.77, /o/=0.77, /u/=0.83).
- Voicing floor: files with < 1.0s voiced audio are skipped.
- Steady-window preprocessing: middle 60% of each vowel used (drops onset/offset glottal transients).
- After preprocessing 6 files fell below the 1.0s voicing floor (was 466, now 460).
- Coverage: [I] 49 PD / 46 HC (mean 2.12 reps/subject), [O] 49/46 (2.11), [U] **45/46** (1.26 — 4 PD subjects have no /u/ file, are scored on {I, O} via weight renormalization).

**DDK channel:**
- One PATAKA file per subject; 95 files total (one per analysis-cohort subject).
- Silent-envelope guard: peak count < 3 → file discarded (0 discarded in practice).

**Facial channel (not on NeuroVoz):**
- Smile classifier trained on ROC-HCI UFNet released feature dataset (1,361 subjects, MIT license, external), NOT on NeuroVoz.
- Deployed via OpenFace 2.0 Docker (`algebr/openface`), the same extractor UFNet trained on.

---

## 3. Feature Extraction

### 3.1 Phonation — sustained vowels (Parselmouth)

**Preprocessing:**
- Steady-window crop to middle 60% of each vowel WAV (skip if duration < 0.5s).

**Features (12 per file):**

Definitions below are the *quantity* the number represents (not the API used to compute it). The Parselmouth call producing each number is listed at the end.

| Feature | Definition |
|---------|------------|
| `jitter_local` | mean absolute difference between consecutive pitch periods, divided by the mean period. Cycle-to-cycle frequency perturbation, dimensionless ratio. |
| `jitter_rap` | Relative Average Perturbation — three-point running-average variant of jitter (each period compared to the mean of itself and its two neighbors). Less sensitive to isolated outliers than `jitter_local`. |
| `jitter_ppq5` | Five-point Period Perturbation Quotient — jitter averaged over a five-period window. Smoother still. |
| `shimmer_local` | mean absolute difference between consecutive period amplitudes, divided by the mean amplitude. Cycle-to-cycle amplitude perturbation, dimensionless ratio. |
| `shimmer_apq5` | Five-point Amplitude Perturbation Quotient — shimmer averaged over a five-period window. |
| `shimmer_apq11` | Eleven-point APQ. Even smoother; picks up longer-scale amplitude drift rather than jitter-scale flicker. |
| `shimmer_dda` | Difference of Differences of Amplitudes — second-order amplitude change, mathematically equivalent to 3 × `shimmer_apq3`. Sensitive to abrupt cycle-to-cycle amplitude swings. |
| `hnr_mean_db` | Harmonics-to-Noise Ratio — 10·log₁₀(harmonic energy / noise energy), averaged across voiced frames. Higher = cleaner voice; typical adult HNR is 15–25 dB. |
| `f0_mean_hz` | Mean of the frame-wise fundamental frequency in Hz, over voiced frames only. |
| `f0_std_hz` | Standard deviation of frame-wise F0 in Hz. Pitch stability during the sustained vowel — PD often shows elevated F0 std. |
| `f0_range_hz` | Max minus min F0 across voiced frames, in Hz. Coarser stability marker than std; sensitive to isolated glitches. |
| `voicing_fraction` | Proportion of frames in the analyzed window for which pitch tracking returned a valid F0. Quality-check feature — very low values signal breathy or aperiodic phonation. |

**Parselmouth API used (implementation, not definition):**

| Group | Call |
|-------|------|
| jitter (all three variants) | `praat.call(point_process, "Get jitter (local|rap|ppq5)", 0, 0, 1e-4, 0.02, 1.3)` on the pitch-derived `PointProcess` |
| shimmer (all four variants) | `praat.call([sound, point_process], "Get shimmer (local|apq5|apq11|dda)", 0, 0, 1e-4, 0.02, 1.3, 1.6)` |
| HNR | `snd.to_harmonicity_cc()` then `harm.values[harm.values != -200].mean()` (Praat sets undefined frames to −200) |
| F0 | `snd.to_pitch_cc(pitch_floor=75, pitch_ceiling=500).selected_array["frequency"]`, then mean/std/range on nonzero values |
| voicing fraction | fraction of pitch frames with `frequency > 0` |

**Hyperparameters (`configs/model.yaml → phonation`):**
- F0 range: 75–500 Hz
- Min voicing duration: 1.0s
- Praat "To Pitch (cc)" full 10-arg signature

**Aggregation to subject-level:**
1. Per subject × per vowel: mean over reps (e.g. `I1`, `I2`, `I3` → one /i/ vector).
2. Cross-vowel weighted average with weights I=0.310, O=0.310, U=0.379 (Li et al. Table IV AUC-excess, renormalized after dropping /a/ and /e/). Renormalize per subject over vowels actually present.

Batch entry point: `python -m src.audio.phonation` → `data/processed/phonation_features/{per_file,per_subject_per_vowel,per_subject}.csv`.

### 3.2 Articulation / DDK — PATAKA (envelope peak-picking, no ASR)

**Method:** rectify → moving-average smooth (linear amplitude, not dB) → `scipy.signal.find_peaks` on the intensity envelope.

**Features (8 per file):**

| Feature | Method |
|---------|--------|
| `n_peaks` | count of detected peaks |
| `duration_s` | clip duration in seconds |
| `ddk_rate_hz` | peaks / duration |
| `isi_mean_s` | mean of inter-syllable intervals |
| `isi_cv` | CV of ISI (timing regularity — lower = more regular) |
| `amp_mean` | mean peak amplitude |
| `amp_cv` | CV of peak amplitude |
| `amp_decrement` | slope of linear fit `amp ~ time` |

**Hyperparameters (`configs/model.yaml → ddk`):**
- Peak prominence: 0.15 × env_max
- Minimum ISI: 0.05s
- Envelope smoothing: 30 ms moving average
- `amp_decrement` requires ≥ 8 peaks (else NaN)
- Files with < 3 peaks discarded

**Note on `duration_s` and `n_peaks` as features:** Both scale with recording length. See §7.4 for the confounder audit — model does not rely on them for its dominant discrimination.

**Aggregation:** One PATAKA per subject → per_file = per_subject.

Batch entry point: `python -m src.audio.ddk` → `data/processed/ddk_features/{per_file,per_subject}.csv`.

### 3.3 Facial — smile classifier + hypomimia narrative

**Two outputs from one OpenFace Docker run per clip.** OpenFace called with `-aus -pose` flags (both AU_r/AU_c and pose_Tx/Ty/Tz columns). The Docker call: `algebr/openface:latest` with `--platform linux/amd64` for Rosetta 2 on Apple Silicon.

**Output 1: 14-dim classifier feature vector (session-level, active-frame-only aggregation per Islam et al. 2023 §2)**

7 AUs used: AU01, AU06, AU12 (lip corner puller — dominant predictor, coefficient rank 1), AU14, AU25, AU26, AU45 (blink).

For each AU:
- `mean(AU_r | AU_c == 1)` — intensity when the AU is firing
- `var(AU_r | AU_c == 1)` — variance when firing
- Zero-fill: if AU never fires → mean = var = 0, recorded in `zero_filled_columns` metadata

Quality gates:
- Face detection confidence ≥ 0.75 per frame (else masked)
- Session-level `detection_rate < 0.80` → score withheld (`score=None` with reason)
- ≥ 3 AUs never firing → "smile task may not have been performed" warning

**Output 2: hypomimia narrative JSON (for Claude report)**

Computed by `src/vision/summarize.py` from the same OpenFace CSV (no second Docker run). Five markers:

| Marker | Method |
|--------|--------|
| `mean_AU12` | kept-frame mean of AU12_r |
| `AU12_amplitude_on_smile_cue` | **max** of AU12_r on active frames (peak, not mean — clinical "amplitude" convention, non-redundant with the classifier's active-frame mean) |
| `expression_variance` | mean of per-AU temporal std across kept frames (higher = more expressive) |
| `blink_rate_per_min` | AU45_c 0→1 rising edges / (kept frames / fps / 60) |
| `head_movement_std` | mean of std(pose_Tx), std(pose_Ty), std(pose_Tz) on kept frames |

**Design decision:** No composite `hypomimia_score`. A composite requires an arbitrary normalization anchor and duplicates the synthesis Claude does. Raw markers give Claude flexibility; a composite would be a false-precision anchor.

**Insufficient-detection short-circuit:** `< 10` kept frames → all fields None with reason in `warnings`.

---

## 4. Model Training

### 4.1 Model class per channel

**Phonation, DDK: `sklearn.LogisticRegression` (l2, C=1.0)** inside `Pipeline([StandardScaler, LogisticRegression])`.

Rationale:
- N ≈ 95 subjects — deep learning is overparameterized for this regime
- LogReg gives calibrated `predict_proba` directly (needed by score-level fusion + Claude report)
- Interpretable — coefficients dumped to `eval/results/coefficients.csv`
- Matches the smile classifier's model class

C=1.0 was not tuned via inner CV. The bottleneck at this cohort size is aggregation, not regularization (verified empirically — see §5).

**Facial (smile classifier):** Same model class (`LogisticRegression + StandardScaler + SMOTE`) trained on UFNet's `ID`-column split, external-validated on YouTubePD. UFNet's ShallowANN is mathematically a LogReg over the session-level feature vector, so we use LogReg + no torch dependency.

### 4.2 Fusion mechanism

**Late fusion of per-channel probabilities** (not feature-level concatenation).

Reasons for late fusion:
- Voice-quality and articulatory-timing features live on different scales and come from different tasks
- Concatenation would require imputation for missing tasks (a demo user may skip a task)
- Late fusion preserves modularity: any channel can be N/A at inference and the fusion renormalizes over the present channels

**Weight scheme: AUC-excess-over-chance.** `w_c ∝ max(0, AUC_c − 0.5)`, normalized over channels present.

Concretely (from Day 3 LOSO): phonation excess 0.130, DDK excess 0.240 → phonation weight 0.35, DDK weight 0.65. Facial weight 0.15 (paper 0.812 test AUROC → excess 0.312; deprecated in favor of a smaller fixed weight because the facial extractor is domain-transferred at demo time and our confidence in its calibration is lower than the audio channels).

### 4.3 Cross-validation protocol

**Subject-level leave-one-subject-out (LOSO)** on the 95-subject analysis cohort. Two eval regimes both run and reported:

1. **Per-subject LOSO (95 folds, 95 datapoints)** — one prediction per subject. This is the honest deployment metric ("will this work on a new patient?"). Headline number.
2. **Per-file with `LeaveOneGroupOut(subject_id)` (95 folds, 466 vowel datapoints)** — literature-comparable to NeuroVoz paper's per-file baselines. Auxiliary number. No leakage (still splits by subject).

**No scaler leakage:** `StandardScaler` inside the `Pipeline` re-fits per fold.

**Metrics:** AUC (bootstrap 95% CI, 1000 resamples over subjects), F1, sensitivity, specificity at threshold 0.5. All computed on out-of-fold predictions.

Ablation runner: `python -m eval.ablation` → `eval/results/{ablation_table.csv, ablation_summary.json, loso_oof_probs.csv, coefficients.csv}`.

### 4.4 Deployment classifiers

For Layer 2 (demo-time inference), each channel is refit on the **full analysis cohort** (not LOSO). LOSO is eval-only. Deployment classifiers saved to `eval/models/{phonation,ddk}.joblib` + `_meta.json`. Smile classifier at `eval/models/smile_pd_{lr,scaler,columns,metrics}.*`.

---

## 5. Sensitivity Analyses

Each analysis below tested a specific hypothesis. Both wins and rejected hypotheses are logged — the negatives are as important as the positives for methodological transparency.

### 5.1 Aggregation: per-subject mean vs per-file training (Day 3, 2026-07-09)

**Hypothesis:** Averaging vowel files into one 12-dim per-subject vector loses signal.

**Test:** Per-file LogReg with `LeaveOneGroupOut(subject_id)`, then per-subject AUC computed by averaging per-file OOF probabilities per subject.

**Result:**
- Per-subject baseline (mean-over-vowels): phonation AUC = 0.567
- Per-file training, per-subject eval: phonation AUC = **0.630 (+0.063)**

**Conclusion:** Per-file training is strictly better for phonation at this cohort size. Deployed configuration uses per-file training.

### 5.2 Fusion weights: unweighted vs AUC-excess (Day 3)

**Test:**
- Unweighted average of phonation + DDK probs: AUC = 0.756
- AUC-excess weighted (p=0.35, d=0.65): AUC = **0.758**
- Old CogniScreen-era prior weights (p=0.50, d=0.35): AUC = 0.705 (**fusion loses to DDK-only**)

**Conclusion:** AUC-excess weighting is the principled default. Manual priors that upweight the weaker channel are actively harmful.

### 5.3 Steady-window preprocessing (Day 4, 2026-07-10)

**Hypothesis:** Cropping vowels to middle 60% removes onset/offset glottal transients that inflate jitter/shimmer.

**Test on NeuroVoz LOSO:** All |ΔAUC| ≤ 0.004 (within CI). No effect.

**Test on demo audio (raw, not NeuroVoz-preprocessed):** Decisive.

| | Un-trimmed F0 std | Post-trim F0 std | PD prob (un-trimmed) | PD prob (post-trim) |
|---|---|---|---|---|
| HC demo | 10.68 Hz | 5.78 Hz | 0.163 | 0.466 |
| PD demo | 8.93 Hz | 4.98 Hz | 0.240 | **0.530** |

**Explanation:** NeuroVoz training data was silently pre-trimmed by the AVCA-ByO pipeline before Zenodo release. Raw demo audio was not. Un-trimmed demo F0 std (9–11 Hz) is completely outside NeuroVoz training range (3–5 Hz); the classifier extrapolates. Steady window aligns feature distributions.

**Conclusion:** `apply_steady_window: true` is **mandatory on the demo path**. LOSO does not surface this because NeuroVoz is already pre-trimmed.

**Meta-lesson:** Training-time hidden preprocessing is a first-class validity concern. Any future feature-engineering change must be re-benchmarked on both NeuroVoz LOSO AND demo audio because the two evaluation surfaces disagree systematically.

### 5.4 Vowel filter + paper-weighted aggregation (Day 4)

**Hypothesis:** Dropping low-signal vowels (/a/, /e/ per Li et al. Table IV) improves phonation.

**Test on NeuroVoz LOSO:** Neutral (Δ ≤ 0.006, within CI).

**Test on demo:** PD phonation +0.083 (0.530 → 0.613), fused +0.025 (0.679 → 0.704). HC essentially unchanged (0.466 → 0.464).

### 5.5 Rejected hypotheses (documented for provenance)

**A. Log-transform + RobustScaler on phonation** (Day 3). Hypothesis: right-skewed jitter/shimmer outliers dilate StandardScaler std. Result: phonation-only AUC 0.567 → 0.534 (worse). Hypothesis rejected. Code reverted.

**B. Learned meta-combiner (stacking)** (Day 3). Rejected before implementation: N=95 out-of-fold probs is too little to fit a stable second-stage classifier.

**C. Composite `hypomimia_score`** (Day 4). Rejected in design review: requires an arbitrary normalization anchor, duplicates work Claude does at report synthesis.

---

## 6. Facial Channel — External Validation

The facial channel does NOT use NeuroVoz. Trained on UFNet (ROC-HCI, AAAI 2025), MIT-licensed.

**In-distribution test AUROC (UFNet held-out participant split):** **0.812**
- Reference: paper's smile-only 0.830 (SVM ensemble). Gap is within noise given our model-class simplification (LogReg not SVM ensemble) and aggressive feature reduction for reproducibility (dropped MediaPipe geometric signals + entropy stats — see below).

**External validation on YouTubePD (informational only, DO NOT quote as headline):**
- Full released CSV (251 clips, 58 PD): AUROC **0.602**
- UFNet's designated `splits/test_yt_pd.txt` subset (178 of 184 IDs matched, 21 PD): AUROC **0.708**
- Paper does NOT publish smile-only YouTubePD (only Smile+Speech fusion 0.838), so there is no direct baseline.
- Interpretation: The 73 clips in the full CSV but excluded from UFNet's subset are 51% PD vs 12% PD in the retained subset. Non-random exclusion (likely low-quality / off-task clips UFNet's own model also failed on). The 0.708 subset number is a fairer measure of smile-only wild-transfer.

**Neither YouTubePD number is demo-representative.** The demo is task-matched (Islam 2023 smile ×3 protocol) and should sit near the in-distribution 0.812.

### Three subtleties we hit precisely to get comparable numbers

1. **Splits key off the `ID` column, not `Participant_ID`.** UFNet's `unimodal_smile_baal.py:136` uses `IDs = df['ID']`. Using `Participant_ID` matches only ~43% of test IDs, silently leaking the rest into train.
2. **NaN in the `pd` column is PD=1**, per their `lambda x: 0 if str(x) in ['no','0'] else 1` rule. In our data, dropping the 20 NaN rows changes test AUROC by 0.001.
3. **AU aggregation is active-frame-only** (Islam 2023 §2). Full-frame aggregation dilutes toward 0.

### Features: 14 = 7 AUs × {mean, var}, active-frame-only

**Dropped from UFNet's 42-feature original set:**
- 7 MediaPipe geometric signals × {mean, var, entropy} = 21 features. Reason: Islam 2023 does not publish landmark indices → non-reproducible. Ablation cost: -0.026 test AUROC (0.839 → 0.812).
- Entropy statistic (all 14 base signals × 1 entropy = 14 features). Reason: histogram binning, log base, range unpublished → non-reproducible. Ablation cost: -0.028 test AUROC (0.837 → 0.812).

**Retained AU45** (blink) — could be dropped with negligible cost (<0.005 AUROC) but kept for column-alignment with UFNet CSV schema.

---

## 7. Cohort Audit (Confounders and Biases)

### 7.1 Age

**Finding: PD 5 years older than HC on average, despite the HC age ≥ 50 filter.**

| Group | n | mean | std | median | min | max |
|-------|---|------|-----|--------|-----|-----|
| HC | 46 | 65.83 | 8.79 | 66.0 | 53 | 86 |
| PD | 49 | 70.84 | 10.47 | 71.0 | 41 | 88 |

Welch t=2.532, p=0.0130, Cohen's d = **0.518** (medium effect size).

**Implication:** Voice ages — F0 typically drops in adult men and can rise in adult women; jitter and shimmer both trend up with age. Some fraction of the 0.740 DDK-only AUC and 0.758 fusion AUC is measuring age, not PD. This is not correctable without a larger cohort (we can't just drop PD subjects because we don't have replacements at those ages).

**Mitigation reporting:** We report the age gap in LIMITATIONS.md and any external comms. We do NOT age-adjust because with N=95 an age-regression step would eat too much variance to leave stable classifier training data.

**Independent evidence age is not doing all the work:** Misclassified subjects (see §7.5) have essentially the same age distribution as correctly classified subjects (67.3 vs 68.9, no significant difference). If age were the dominant signal, we'd expect the oldest subjects (regardless of group) to be classified PD and the youngest HC — but the age distributions are indistinguishable. Age is a real confounder but not the primary driver.

### 7.2 Sex

Sex is encoded 0/1 in NeuroVoz raw metadata with **no legend published**. This is a NeuroVoz reproducibility gap, not ours.

Distribution in analysis cohort:

| Group | sex=0 | sex=1 |
|-------|-------|-------|
| HC | 24 | 22 |
| PD | 20 | 29 |

Slight imbalance (HC leans toward 0, PD leans toward 1) but not dramatic. Sex is a known covariate for phonation (F0 differs ~1 octave M/F, jitter/shimmer differ). Without a legend we cannot audit the direction of any bias. Documented as a known-unknown in LIMITATIONS.

### 7.3 Severity distribution (PD subjects only)

**UPDRS:**
- n=43 with UPDRS reported (6 PD subjects have UPDRS missing)
- mean 15.8, std 9.3, median 14.0, IQR 9.5–20, max 47

**Hoehn-Yahr stage:**

| H-Y stage | Count | Description |
|-----------|-------|-------------|
| 1 | 4 | unilateral |
| 2 | 32 | bilateral, no balance impairment |
| 3 | 11 | bilateral with postural instability |
| 5 | 1 | wheelchair-bound |

**Disease duration:** median 6 years, IQR 3.75–9, max 30.

**Interpretation:** This is a **mild-to-moderate PD cohort**. 65% are H-Y stage 2. Only 1 subject at severe stage. Our AUC numbers apply to this severity range and should not be extrapolated to early-stage (H-Y 1) or severe (H-Y 4–5) PD.

---

## 8. Known Limitations Summary

For the user-facing / evaluator-facing version see `LIMITATIONS.md`. Full list from a methodology perspective:

1. **ON-medication training bias.** All PD training subjects recorded 2–5h post-dose. Model calibration valid for ON-state PD only.
2. **Age imbalance.** PD 5 years older than HC (p=0.013, d=0.518). Documented, not corrected.
3. **Mild-to-moderate PD cohort.** H-Y stage 2 dominant. AUC does not extrapolate to H-Y stage 4–5.
4. **Sex encoding unknown.** NeuroVoz 0/1 legend not published; audit incomplete.
5. **Small N.** 49 PD × 46 HC = 95 subjects. Bootstrap 95% CI on AUC ≈ ±0.10 wide. Any single-percentage-point AUC comparison is inside CI noise.
6. **Language: Spanish only.** Phonation + DDK features are language-neutral; cross-language transfer is untested.
7. **No cross-corpus training.** Deliberate — mixing tasks/corpora at small N would create dataset-bias shortcuts. Also means our AUC has not been validated on any external audio corpus.
8. **Facial channel domain gap.** Trained on UFNet (US clinical + YouTube). No task-matched Spanish facial validation set exists in this project.
9. **Facial extraction dependency.** OpenFace 2.0 via Docker (`algebr/openface`). If Docker or platform changes, extraction is not portable.
10. **DDK `duration_s` and `n_peaks` are duration-scaled features.** Coefficient audit says the model doesn't rely on them, but this has not been proven with a leave-out ablation.
11. **Screening decision aid, NOT a clinical diagnosis.** Restated because it is the single most important framing constraint.

---

## 9. Reproducibility Pointers

- **Numbers in this document are reproducible from the repo state at commit d935913** (Day 4+5 Claude report layer). Run `python -m eval.ablation` to rebuild the ablation table; `python -m src.audio.phonation` and `python -m src.audio.ddk` to rebuild per-channel features.
- **Cohort snapshot** at `data/processed/cohort.csv` (checked into repo — 104 rows, 95 in analysis cohort).
- **Deployment models** at `eval/models/*.joblib`.
- **Per-subject OOF probabilities** at `eval/results/loso_oof_probs.csv` (used for §7.5 misclassification analysis).
- **Config snapshot** at `eval/results/ablation_summary.json` — hyperparameters at the time of the run.
- **Coefficients** at `eval/results/coefficients.csv` and `eval/results/coefficients_l1_phonation.csv` (L1 viz-only).

For dataset citations and licenses see `DATASETS.md`.
