# ParkScreen — Hackathon TODO

**Dates:** July 7–13, 2026 | **Track:** Development

Pivoted from CogniScreen (dementia) → ParkScreen (PD) on 2026-07-07. See CLAUDE.md Changelog for scientific rationale.

---

## Weekly Overview

| Day | Theme | Must-ship | Can-cut |
|-----|-------|-----------|---------|
| Day 1 (Mon 7/7) | Foundation + pivot | Repo + env + Whisper pipeline running; NeuroVoz downloaded; Parselmouth sanity-checked | — |
| Day 2 (Tue 7/8) | Dataset swap + feature extraction | IPVS → NeuroVoz swap; Phonation + DDK features for all NeuroVoz samples (age-matched split) | — |
| Day 3 (Wed 7/9) | **Ablation (never cut)** | Train 3 models (phonation, DDK, phonation+DDK late fusion), subject-level LOSO on NeuroVoz, produce `ablation_table.csv` with real numbers | fancy plots |
| Day 4 (Thu 7/10) | Facial + Claude layer | ~~facial classifier~~ (done Day 2 bonus); **still needed:** OpenFace-derived hypomimia JSON + Claude 3-channel report + `llm_fusion` context builder | UI polish |
| Day 5 (Fri 7/11) | **End-to-end demo (never cut)** | Gradio app: task-matched video upload → full report rendered; held-out NeuroVoz PD sample also runs | advanced UI |
| Day 6 (Sat 7/12) | Freeze + polish | README, limitations page (PD framing), two deliverables ready | new features |
| Day 7 (Sun 7/13) | Submit | Demo video + pitch deck + submission | — |

---

## Day 1 — Foundation + Pivot (July 7)

### Admin (post-pivot)
- [x] Update CLAUDE.md to ParkScreen (PD, phonation+DDK late fusion)
- [x] Update TODO.md to ParkScreen
- [x] ~~Download IPVS (593 MB zip → `data/`); extracted to `data/raw/italian_pd/` preserving group folders. 831 WAVs, all subject counts match paper (28 PD + 22 elderly HC + 15 young HC).~~ **SUPERSEDED by NeuroVoz swap (2026-07-08) — see Day 2 admin.**
- [x] ~~Decode `FILE CODES.xlsx`: vowels VA1/2..VU1/2 (phonation), D1=/pa/ + D2=/ta/ (DDK).~~ **SUPERSEDED — NeuroVoz filenames self-describe task, no separate decode step needed.**
- [x] ~~Plan adjustment applied:~~ **SUPERSEDED — IPVS-specific age-matching note. NeuroVoz uses explicit HC age ≥ 50 filter.**

### Repo Setup
- [x] `git init` repo at `/Users/kiwi/Documents/anthropic_hackathon/`
- [x] Create folder structure: `src/audio/`, `src/vision/`, `src/fusion/`, `src/report/`, `eval/results/figures/`, `demo/assets/`, `data/processed/`, `data/samples/`, `notebooks/`, `configs/`, `tests/`
- [x] ~~Rename `data/raw/adresso/` → `data/raw/italian_pd/`; add `vowels/ ddk/ reading/` subdirs~~ **SUPERSEDED — NeuroVoz uses `data/raw/neurovoz/data/audios/` (flat).** `data/raw/backup/` → `data/raw/external/` still applies.
- [x] Create `data/processed/phonation_features/`, `data/processed/ddk_features/`, `data/processed/facial_features/`, `data/processed/transcripts/`, `eval/models/`
- [x] Add `.gitignore` (excludes `data/`, `.env`, `__pycache__`, `*.wav`, `*.mp4`)
- [x] Create `.env.example` with `ANTHROPIC_API_KEY=`
- [x] Create `requirements.txt`
- [x] Create Python 3.12 venv at `.venv/` — activate: `source .venv/bin/activate`
- [x] Update `requirements.txt`: fix `feat` → `py-feat` typo, drop `librosa` / `opensmile` / `transformers` / `torch`, add `matplotlib`
- [x] Install all requirements: `pip install -r requirements.txt` (+ `brew install libomp` for xgboost/py-feat); added `openpyxl` for reading IPVS metadata sheets
- [ ] Sanity-check `praat-parselmouth` on a real vowel: load a sustained /a/ from NeuroVoz (e.g. `HC_A1_0034.wav`), print jitter+shimmer+HNR — verifies the toolchain end-to-end before Day 2
- [x] Verify `ffmpeg` available (`ffmpeg 8.0.1` ✓)

### Config Files
- [x] `configs/paths.yaml` — all data/output paths
- [x] ~~Update `configs/paths.yaml` for IPVS layout (vowels/ ddk/ reading/)~~ **SUPERSEDED — updated to NeuroVoz layout 2026-07-08** (`neurovoz_audios_dir`, `neurovoz_metadata_pd`, etc.) + processed feature dirs + `eval/models/` + cohort manifests
- [x] `configs/model.yaml` — Whisper model, pause threshold, fusion weights, Claude model (`claude-opus-4-7`)
- [x] Update `configs/model.yaml`: add `phonation:` (F0 75–500 Hz, min voicing 1.0s, jitter/shimmer params), `ddk:` (peak prominence, min ISI 0.05s, envelope smoothing), per-channel `fusion:` weights (phonation 0.50 / DDK 0.35 / facial 0.15), `cohort:` (primary vs secondary)

### Whisper Pipeline (`src/audio/transcribe.py`) — supporting role now, not primary
- [x] `extract_audio()` — ffmpeg WAV extraction from video
- [x] `load_audio()` — WAV → float32 numpy array
- [x] `_run_whisper()` — mlx_whisper wrapper (private, isolated for testability)
- [x] `_extract_words_and_pauses()` — flatten segments → word list + pause list (gaps ≥ 0.25s)
- [x] `transcribe_file()` — public entry point, handles video/audio, cleans up temp WAV

### Tests
- [x] `tests/test_pipeline.py` — smoke tests for `load_audio` and `_extract_words_and_pauses` (no MLX/audio file needed)
- [x] Smoke tests passing: `✅ All smoke tests passed`

### Verification
- [x] Run `transcribe_file()` on a real audio sample — `recording.m4a` (48.23s, 101 words, 4 pauses) ✅
- [ ] Commit Day 1 + pivot work to git (`chore: pivot CogniScreen → ParkScreen`)

---

## Day 2 — Dataset Swap + Feature Extraction (July 8)

### Dataset Swap Admin (IPVS → NeuroVoz)
- [x] Download NeuroVoz (976 MB zip → `data/neurovoz_v3.zip`); extracted to `data/raw/neurovoz/data/` (audios/, metadata/, transcriptions/, grbas/, audio_features/). 2976 WAVs, 53 PD + 55 HC per metadata.
- [x] Update CLAUDE.md (data section, changelog sub-entry, demo protocol PATAKA, feature tables) — see 2026-07-08 sub-changelog.
- [x] Update `configs/paths.yaml` for NeuroVoz layout; update `configs/model.yaml` cohort block (`pd_n: 53`, `hc_age_min: 50`, `require_task_intersection: [PATAKA]`); rename DDK comment /pa/+/ta/ → PATAKA.
- [x] Delete IPVS data: `data/raw/italian_pd/` (812 MB) + `data/Italian Parkinson's Voice and speech.zip` (565 MB) — done 2026-07-08. Kept `data/neurovoz_v3.zip` as extraction safety net.
- [x] Build subject-level `labels.csv` at `data/raw/neurovoz/labels.csv` via `python -m src.data.build_labels` — 2791 rows, 53 PD + 51 HC (after HC age ≥ 50 filter, dropping 2 young + 2 no-age subjects). Fixed a data quirk: NeuroVoz metadata references task "ESPONTANEA" but files are named "FREE" — the loader rewrites `_ESPONTANEA_` → `_FREE_` at ingest.
- [x] Build analysis cohort at `data/processed/cohort.csv` via `python -m src.data.build_cohort` — 104 total subject rows (one per subject), 95 in analysis cohort (**49 PD × 46 HC** — hit target exactly). 9 excluded: 8 for missing PATAKA (5 HC, 3 PD), 1 (PD_108) missing both PATAKA and all vowels. cohort.csv keeps excluded subjects with `in_analysis_cohort=False` for audit.
- [x] **Vowel-balance decision:** restricted to the balanced 5-vowel set (A1, A2, I1, O2, U1). `src/audio/phonation.py::BALANCED_VOWEL_SET` is the source of truth.

### ~~Task Segmentation (`src/audio/segment.py`)~~ — **DROPPED 2026-07-09**
- Every corpus we use (NeuroVoz, IPVS, UFNet/PARK@Home) stores tasks in separate files; ParkScreen collects three separate uploads at demo time and skips runtime segmentation entirely. No heuristic split, no ASR-driven boundary detection. Task alignment guaranteed by construction. See CLAUDE.md → "Demo Protocol".

### Phonation Features (`src/audio/phonation.py`)
- [x] `extract_phonation_features(wav_path, t0=None, t1=None, cfg=None) -> dict` — 12 features via Parselmouth (jitter local/rap/ppq5, shimmer local/apq5/apq11/dda, HNR mean dB, F0 mean/std/range, voicing fraction). Praat "To Pitch (cc)" verified with the full 10-arg signature; failures return `{}`, log warning.
- [x] Guardrails: skip files with < 1.0s voiced audio (one file hit this: HC_A2_0056, 0.53s voiced). Exception handling per-file, batch continues.
- [x] Batch script `python -m src.audio.phonation` → `data/processed/phonation_features/per_file.csv` (466 rows) + `per_subject.csv` (95 rows, 49 PD × 46 HC). Restricted to `BALANCED_VOWEL_SET = (A1, A2, I1, O2, U1)`. Per-subject = arithmetic mean over available vowels.
- [x] Sanity check: group-mean directions match PD literature — PD > HC for jitter/shimmer/F0-std/F0-range, PD < HC for HNR. Signal is present.

### Articulation / DDK Features (`src/audio/ddk.py`)
- [x] `extract_ddk_features(wav_path, t0=None, t1=None, cfg=None) -> dict` — 8 features: `n_peaks, duration_s, ddk_rate_hz, isi_mean_s, isi_cv, amp_mean, amp_cv, amp_decrement`. `duration_s` retained as a feature because PATAKA clip length varies wildly across subjects (3–20+s) and could confound ISI/amp stats.
- [x] Rectify-and-smooth envelope (linear, not dB) → `scipy.signal.find_peaks` with prominence 0.15×env_max, min ISI 0.05s. `amp_decrement` slope requires ≥ 8 peaks (else NaN); files with < 3 peaks total return `{}`.
- [x] Batch script `python -m src.audio.ddk` → `data/processed/ddk_features/per_file.csv` (95 rows) + `per_subject.csv` (95 rows, 49 PD × 46 HC — one PATAKA per subject so they're identical).
- [x] Sanity check: group-mean directions match PD literature — PD lower DDK rate (5.75 vs 6.49 syl/s), higher ISI mean/CV, lower amp_mean, steeper amp_decrement. Signal is present.

### Bonus (ahead of schedule) — Facial classifier bootstrapped from UFNet
- [x] Recon ROC-HCI UFNet repo (https://github.com/ROC-HCI/UFNet, AAAI 2025, MIT). Decoded the 42 features (14 base × mean/var/entropy = 7 OpenFace AUs + 7 MediaPipe geometric signals). Confirmed their smile-only ShallowANN is mathematically a LogReg. Chose reuse their CSV to train, not their pretrained weights.
- [x] Copy UFNet CSVs + participant splits + pretrained-for-reference into `data/raw/ufnet_smile/` (gitignored). Install `mediapipe` + `imblearn`. Update `requirements.txt` + `configs/paths.yaml`.
- [x] Write `src/vision/train_smile_pd.py` — StandardScaler + SMOTE + LogReg on UFNet's `ID`-column splits, external validate on YoutubePD CSV. Two bugs caught: (a) splits key off `ID` not `Participant_ID` (using wrong column silently leaked 151/267 test subjects into train); (b) UFNet's `pd`-column rule treats NaN as PD=1. `--nan-as` and `--feature-set` CLI switches expose both alternatives for reproducibility.
- [x] Ran 7-way feature-set ablation on UFNet's test split (see `train_smile_pd.py` FEATURE_SETS). Full 42 features → 0.837; no_entropy 28 → 0.839; au_only 21 → 0.803; **au_mean_var 14 → 0.812**; au5_mean_var 10 → 0.811. Entropy adds ~0 AUROC; MediaPipe geometric adds ~0.026 but its landmark indices are unpublished; AU45 costs <0.005 to drop.
- [x] Read Islam et al. 2023 "Unmasking Parkinson's Disease with Smile" (arxiv 2308.02588 — the smile-only predecessor to UFNet) for methodology. **Critical finding: they aggregate mean/variance only on frames where the AU is active (`AU_c == 1`), not over all frames.** Full-frame aggregation dilutes toward zero and would replicate the py-feat scale-gap failure mode.
- [x] Design decision after paper read + ablation: **14 features (7 AUs × mean+var), active-frame-only aggregation, OpenFace 2.0 via Docker `algebr/openface` (matches UFNet's training extractor exactly → zero AU scale gap).** MediaPipe geometric dropped (landmark indices unpublished). Entropy dropped (histogram binning unpublished). AU45 retained for column-alignment with UFNet CSV schema. Expected retraining AUROC on UFNet test split: 0.812.
- [x] Retrain classifier at 14 features: `python -m src.vision.train_smile_pd --feature-set au_mean_var` → overwrites `smile_pd_{lr,scaler,columns,metrics}.*`. **Verified: test AUROC 0.812** (matches ablation). Coefficient sanity check confirms AU12_mean dominance (|coef| 2.59, rank 1, matches Islam 2023 Table 2).
- [x] Write `src/vision/facial_features.py` — OpenFace 2.0 via Docker `algebr/openface:latest` subprocess (`--platform linux/amd64` for Rosetta 2 on Apple Silicon), returns per-frame `au_r` + `au_c` for 7 AUs. Confidence < 0.75 or `success == 0` → NaN on `au_r` / 0 on `au_c`. Detection-rate meta exposed to callers. Fixed a Pandas 3.0 read-only Arrow view bug (`.to_numpy()` returns immutable view; `.copy()` before mask). CLI: `python -m src.vision.facial_features <video> [--t0 X --t1 Y]`.
- [x] Write `src/vision/aggregate.py` — active-frame-only mean+var per Islam 2023 §2. Zero-fill convention for degenerate cases (never-active AU → mean=var=0; single-active-frame → mean=that_value, var=0). `zero_filled_columns` meta tracks which output columns came from a never-fired AU. Column order locked to `smile_pd_columns.json`.
- [x] Write `src/vision/predict_smile_pd.py` — video → PD prob end-to-end. Quality gate: detection_rate < 0.80 → `score=None` with reason. Zero-fill warning: ≥ 3 AUs never fired → "smile task may not have been performed as instructed". Interpretability: top-3 PD-ward + top-3 healthy-ward contributor ranking (`clf.coef_ * scaled_features`). CLI: `python -m src.vision.predict_smile_pd <video> [--t0 X --t1 Y --min-detection-rate 0.80]`.
- [x] End-to-end smoke test on `data/samples/Personal Video Essays copy.mp4` (non-smile self-introduction video, 5-second window). Docker + ffmpeg + OpenFace + parser + aggregator + classifier all executed; 5-second clip completed in 5s real time (Rosetta 2 much faster than predicted). Correctly triggered zero-fill warning (AU01/AU06/AU12 never fired during speech) and produced garbage-in-garbage-out score 0.968, validating both the pipeline and the quality-gate design.

### Data Sanity
- [x] ~~Verify subject-level file grouping~~ — implicitly done by build_cohort's `has_pataka & n_vowel_files ≥ 1` filter (95 in-cohort subjects all pass).
- [x] ~~Build analysis cohort manifest~~ — duplicate of the cohort item at line 77.
- [x] Plot feature distributions per group via `python -m src.data.qa_features` → `eval/results/figures/feature_distributions.png` (3×4 boxplot grid, 6 phonation + 6 DDK features, HC vs PD). **QA table: zero NaN, zero Inf across all 12 features on 95 subjects.** Ranges plausible (F0 100–265 Hz spans male-adult to female-adult; DDK rate 1.95–10.58 syl/sec; HNR 7.7–31.7 dB).
- [x] ~~Save cohort manifest: `data/processed/cohort.csv`~~ — completed by build_cohort.
- [x] Cross-checked our Parselmouth features against NeuroVoz's shipped `audio_features.csv` (join on filename, 466 rows matched). **Findings:** (a) `shimmer_local` ↔ `rShimmer` r=0.824 — compatible, differ only by ~100× scale (they use percent). (b) Three jitter variants ↔ `rJitter/RAP/rPPQ` r=0.46–0.50 — moderate agreement, same family with different Praat parameter presets; acceptable at hackathon quality. (c) `hnr_mean_db` ↔ NeuroVoz `HNR` r=−0.18 — their column name is misleading; theirs averages −10.8 (likely a noise-to-harmonic inverse metric; their CSV also carries `NNE, CHNR, GNE`), ours averages +23.1 dB which is the standard Praat `harmonicity_cc` output. Ours is correct. Cross-check gives our phonation extraction a green light.

---

## Day 3 — Ablation Study (July 9) — NEVER CUT

### `src/fusion/statistical.py`
- [x] Load per-channel feature matrices + subject-level labels + cohort manifest (`load_channel_matrix` for per-subject, `load_phonation_per_file` for per-file). Pipeline = `StandardScaler → LogisticRegression(C=1.0, l2)`, re-fit per fold via `sklearn.Pipeline` so no scaler leakage.
- [x] Per-channel model: **LogReg (l2, C=1.0)** chosen over linear SVM. LogReg gives calibrated `predict_proba` directly (needed by fusion + Claude report), matches the smile classifier's model class, and stays interpretable via `coefficients.csv`. Did NOT tune C via inner CV — `C=1.0` was strong enough on both channels; the bottleneck was aggregation, not regularization.
- [x] Late fusion: per-channel probabilities → **average (unweighted)** AND **AUC-excess weighted** (`w_c ∝ max(0, AUC_c − 0.5)`, normalized over channels present). Rejected a learned meta-combiner because N=95 out-of-fold probs is too little to fit a stable second-stage classifier. Weights land in `configs/model.yaml` — phonation 0.35, DDK 0.65 (see model.yaml comment block for derivation).
- [x] Save fitted classifiers (fit on full analysis cohort — LOSO is eval-only) to `eval/models/{phonation,ddk}.joblib` + `_meta.json` for Layer 2 demo-time loading.

### `eval/ablation.py`
- [x] Subject-level LOSO on NeuroVoz analysis cohort (**49 PD × 46 HC = 95 subjects**). Two eval regimes both run:
  - Per-subject LOSO (95 folds, 95 datapoints) — the honest deployment metric
  - Per-file with `LeaveOneGroupOut(subject_id)` (95 folds, 466 vowel datapoints) — literature-comparable, no leakage
  Each subject's ALL files (vowels + PATAKA) held out together.
- [x] Metrics: AUC, F1, sensitivity, specificity per model + subject-level bootstrap 95% CI on AUC (1000 resamples)
- [x] Rows produced (8 total — see CLAUDE.md → "Ablation table" for numbers):
  1. Phonation-only (per-subject mean-over-vowels baseline) — AUC 0.567
  2. DDK-only (PATAKA) — AUC 0.740
  3. Phonation + DDK (unweighted avg) — AUC 0.722
  4. Phonation + DDK (AUC-excess weighted) — AUC 0.736
  5. Phonation per-file (per-file eval, n=466) — AUC 0.603
  6. Phonation per-file (per-subject eval) — AUC 0.630
  7. Phonation(per-file) + DDK (unweighted avg) — AUC 0.756
  8. **Phonation(per-file) + DDK (AUC-excess weighted) — AUC 0.758 (best)**
- [x] Save `eval/results/ablation_table.csv` (+ `ablation_summary.json` config snapshot, `loso_oof_probs.csv` per-subject OOF probs, `coefficients.csv` per-feature weights)
- [x] ROC curve plot via `eval/make_plots.py` — 3 headline models + chance line → `eval/results/figures/roc_curves.png`. Fusion curve dominates the clinically-relevant low-FPR zone.
- [x] Coefficient-magnitude plot per channel → `eval/results/figures/coefficients.png`. **Phonation refit with L1 (`liblinear`, C=0.3) for viz-only** — 4 shimmer variants + 3 jitter variants are collinear at L2; L1 forces sparsity → `jitter_local` dominates (+0.46), all 4 shimmer variants zeroed. DDK stays L2 (no material collinearity at 8 features); `ddk_rate_hz` and `amp_mean` dominate. L2 deployment classifiers in `eval/models/` unchanged. L1 coefficients persisted at `eval/results/coefficients_l1_phonation.csv`.
- [x] `notebooks/03_ablation_results.ipynb` — 10 cells, executed end-to-end via `jupyter nbconvert --execute`. Renders ablation table (winning row highlighted), embeds both figures, condensed key findings + link back to CLAUDE.md for full detail.

### Optional / stretch (only if ahead of schedule)
- [ ] **Per-vowel modeling for phonation** — 5 separate LogRegs, one per vowel (A1, A2, I1, O2, U1); per-subject soft vote over the 5 probs → per-subject AUC. Preserves per-vowel differences (PD may crash on /i/ but not /a/). Est. +45–60 min work; hypothesized to add another +0.02–0.05 phonation AUC. Skip if Day 4 (facial + Claude layer) is behind.
- [ ] **XGBoost head-to-head with LogReg on per-file phonation** — a fair per-file comparison against NeuroVoz's XGBoost baseline. Only if `configs/model.yaml` can accommodate a per-channel `model_class` field cleanly. Skip if pushing schedule.

### Diagnostic experiments run (documented for provenance; code reverted)
- [x] **log + RobustScaler on phonation** — hypothesis "jitter/shimmer outliers dilate StandardScaler std". Result: phonation-only AUC went 0.567 → 0.534 (worse). Hypothesis rejected. Code reverted to keep `src/fusion/statistical.py` clean. See CLAUDE.md → "Day 3 findings" finding #5.

---

## Day 4 — Facial Path + Claude Layer (July 10)

### Phonation preprocessing — steady-window extraction (added Day 4)
- [x] Add `apply_steady_window`, `steady_window_frac: 0.6`, `steady_window_min_duration_s: 0.5` to `configs/model.yaml → phonation:`.
- [x] Add `_apply_steady_window()` helper to `src/audio/phonation.py`; wire into `extract_phonation_features` after the optional t0/t1 crop. Short-clip guard preserves originals below 0.5s.
- [x] Back up Day 3 baselines: `per_file_day3_baseline.csv`, `per_subject_day3_baseline.csv`, `ablation_table_day3_baseline.csv`.
- [x] Re-run `python -m src.audio.phonation` → 460 rows (6 files fell below 1s voicing floor after cropping); all 95 analysis-cohort subjects retained.
- [x] Re-run `python -m eval.ablation` → NeuroVoz LOSO essentially unchanged (all |ΔAUC| ≤ 0.004). Reason: NeuroVoz files were already pre-trimmed by AVCA-ByO before release.
- [x] A/B test on raw demo audio (`data/samples/{hc,pd}_demo/vowel/`): steady window rescued PD demo from misclassification (0.240 → 0.530). Un-trimmed demo F0_std (9–11 Hz) is completely outside NeuroVoz training range (3–5 Hz); cropping aligns feature distributions. Full multimodal `quick_score` results: HC fused 0.158 (HC ✓), PD fused 0.679 (PD ✓).
- [x] Update CLAUDE.md → Phonation Features + Day 4 findings section with the empirical justification and demo scores.

### Phonation vowel filter + paper-weighted aggregation (added Day 4, 2026-07-10)
- [x] Add `included_vowels: [I, O, U]` + `vowel_weights: {I: 0.310, O: 0.310, U: 0.379}` to `configs/model.yaml → phonation:` with derivation comment (AUC-excess from Li et al. arxiv 2606.19125 Table IV, renormalized after dropping [a]/[e]).
- [x] Retire `BALANCED_VOWEL_SET = (A1, A2, I1, O2, U1)` (Day 2 bug — treated A1/A2 as separate tasks and double-counted /a/). Replace with `INCLUDED_VOWELS_DEFAULT = ("I", "O", "U")` on the vowel-identity axis.
- [x] Rewrite `batch_extract` in `src/audio/phonation.py`: (1) filter by `task[0] ∈ included_vowels`; (2) per-subject × per-vowel mean-of-reps; (3) `apply_vowel_weights` cross-vowel weighted average with per-subject renormalization for missing vowels.
- [x] Add `apply_vowel_weights` helper (exported for `quick_score.py` demo use).
- [x] Back up steady-window baseline CSVs: `per_file_day4_all5tasks.csv`, `per_subject_day4_meanall.csv`, `ablation_table_day4_steady_pre_vowelfix.csv`.
- [x] Re-run `python -m src.audio.phonation` → per-file 516 rows, per-subject-per-vowel 281 rows, per-subject 95 rows. Coverage: [I] 49/46 (2.12 reps/subject), [O] 49/46 (2.11), [U] 45/46 (1.26) — 4 PD subjects have no /u/ file and are scored on IO only. Cohort unchanged at 49+46.
- [x] Update `src/fusion/quick_score.py::_score_phonation` to mirror training-time aggregation: filename → vowel letter parse, per-vowel rep averaging, `apply_vowel_weights` cross-vowel combination. Print skipped-file reasons per vowel.
- [x] Re-run `python -m eval.ablation`: LOSO neutral (phonation-only 0.569 → 0.560; weighted fusion 0.740 → 0.746, +0.006 within CI). Per-file per-subject-eval dropped 0.033 (0.629 → 0.596) — expected side effect of /u/'s lower rep coverage.
- [x] Re-run `quick_score` on both demos: **HC fused 0.158 → 0.157 (HC ✓)**, **PD fused 0.679 → 0.704 (PD ✓)**. PD demo phonation channel gained +0.083 (0.530 → 0.613), which is the target improvement — the LOSO metric under-detects it because training data is pre-trimmed while demo is out-of-distribution.
- [x] Update CLAUDE.md → Phonation Features intro + Day 4 findings #9/#10/#11/#12 with results.

### `src/vision/facial_features.py` — per-frame OpenFace extraction — **DONE as Day 2 bonus (2026-07-08)**
- [x] `extract_smile_features(video_path, t0=None, t1=None)` returning per-frame `au_r` + `au_c` arrays for 7 AUs + detection meta
- [x] Docker subprocess wrapper (`algebr/openface:latest`, `--platform linux/amd64`); confidence < 0.75 or `success == 0` masked; `[t0, t1]` pre-cut via ffmpeg before Docker
- [x] Meta exposes `fps, n_frames_total, n_frames_used, detection_rate, quality_gate_pass, warnings`

### `src/vision/aggregate.py` — session-level 14-dim vector (active-frame-only) — **DONE as Day 2 bonus**
- [x] Active-frame-only mean+var per Islam 2023 §2; zero-fill for never-active AUs with `zero_filled_columns` in meta
- [x] Column order matches `smile_pd_columns.json` exactly

### `src/vision/predict_smile_pd.py` — video → PD probability — **DONE as Day 2 bonus**
- [x] End-to-end video → PD score with detection-rate quality gate, zero-fill warning, and top-3 contributor ranking
- [x] Smoke-tested on `Personal Video Essays copy.mp4` — correctly withheld / warned per design

### `src/vision/summarize.py` — hypomimia narrative JSON (companion to classifier)
- [x] `summarize_facial_features(openface_csv_path_or_df, min_confidence=0.75) -> dict` — reads the OpenFace CSV that `facial_features.extract_smile_features` already produced (path OR pre-loaded DataFrame). No second Docker run, no py-feat.
- [x] Fields: `mean_AU12` (kept-frame mean of AU12_r), `AU12_amplitude_on_smile_cue` (**max** of AU12_r on active frames — clinical "amplitude" = peak, and non-redundant with the classifier's active-frame mean feature), `expression_variance` (mean of per-AU temporal std across kept frames — higher = more expressive), `blink_rate_per_min` (AU45_c 0→1 rising edges / kept duration in minutes), `head_movement_std` (mean of std across `pose_Tx/Ty/Tz`), `detection_rate`, `warnings`.
- [x] **Dropped `hypomimia_score`** per Day 4 design review: composite requires an arbitrary normalization anchor and duplicates work Claude does anyway — leaving the raw markers is more defensible and gives Claude the flexibility to weave them into narrative.
- [x] Emotion/gaze narrative dropped (OpenFace doesn't emit them, non-core to hypomimia).
- [x] Per-minute blink rate uses `n_kept_frames / fps / 60` denominator; low-detection warning (`< 0.80`) surfaced; insufficient-detection short-circuit (`< 10` kept frames) returns all-None schema with reason in warnings.
- [x] Smoke tests: synthetic 60-frame smile arc → AU12 peak recovered to 3 sig figs, blink rate 90/min matches 3 events over 2 s; no-smile CSV → "AU12 never active" warning; low-detection CSV → all fields None + insufficient-detection warning.

### `src/fusion/llm_fusion.py`
- [x] Split into two functions: `fuse_scores(scores, weights, agreement_threshold) -> dict` (canonical late-fusion mechanic + pairwise agreement flags) and `build_claude_context(channels, facial_summary, fusion_result) -> str` (XML-tagged Claude prompt block). One-way import: `quick_score.py` uses `fuse_scores`; nothing imports `build_claude_context` yet (pipeline.py will).
- [x] Weight renormalization + N/A handling: any channel with `None` score drops out, remaining weights renormalize to sum=1. Behaviour identical to old `quick_score._fuse` (validated: 3-channel demo case → `fused=0.704`, matches CLAUDE.md Day 4 table exactly).
- [x] Agreement flags: `speech_channels_agree` = `|phon - ddk| < threshold`; `facial_agrees_with_speech` = `|facial - mean(speech_present)| < threshold`; `any_flag_for_review` = OR of `not agree`. Undefined comparisons return None (e.g. only one speech channel present → `speech_channels_agree=None`, no flag). Threshold defaults to `configs/model.yaml → fusion.agreement_threshold` (0.30).
- [x] **Unit conversion isolated to `build_claude_context`** (NeuroVoz cross-check rule): callers pass ratio-unit features (`jitter_local=0.00425`); the function × 100 the seven ratio keys (jitter/shimmer variants) and renames to `_percent` suffix (`jitter_local_percent: 0.425`). HNR/F0/DDK features already in canonical units. Feature curation: only the clinician-facing subset (5 phonation + 4 DDK + 5 hypomimia markers) forwarded to Claude — classifier still sees the full vector.
- [x] XML tag format (`<phonation status="present">...</phonation>`, `<facial status="omitted" reason="no smile upload"/>`, `<fusion>...</fusion>`) — Claude parses tags reliably and N/A channels are explicit rather than silently missing. `hypomimia_score` composite intentionally not added (see summarize.py rationale).
- [x] `load_fusion_config()` convenience: `configs/model.yaml → fusion.weights` + `fusion.agreement_threshold` → `(dict, float)` tuple. Saves callers from repeating the yaml parse.
- [x] Refactor: removed `_fuse` from `quick_score.py`; call site now uses `fuse_scores` + also prints agreement flags. CLI verified (help + import), behaviour unchanged.
- [x] Smoke tests: 0/1/2/3-channel cases all produce sane fusion + agreement dicts; end-to-end context render on synthetic PD-demo inputs produces the XML block shown in CLAUDE.md example JSON schema (with unit conversion applied).

### `src/report/claude_client.py`
- [x] `generate_report(context_str) -> str` — Anthropic SDK, `claude-opus-4-7`, non-streaming (`max_tokens=2048`, well under the ~16K streaming threshold), `thinking={"type": "disabled"}` (Opus 4.7 default; set explicitly so future default flips can't silently start thinking on us), no sampling params (Opus 4.7 400s on `temperature`/`top_p`/`top_k`).
- [x] Frozen system prompt (~1143 tokens, `src/report/claude_client.py::SYSTEM_PROMPT`) with `cache_control: {"type": "ephemeral"}` — marker is a no-op today (Opus 4.7 needs ≥ 4096-token prefix to cache) but self-activates if we ever grow the prompt past 4K. Usage-log to stderr when cache_read or cache_create is non-zero, so we'll see it turn on without touching code.
- [x] Report structure enforced in system prompt: `## Risk Level` (Low/Moderate/Elevated, explicitly NOT a diagnosis) → `## Phonation` → `## Articulation (DDK)` → `## Facial Expression` → `## Cross-Channel Consistency` → `## Recommended Next Steps` → `## Disclaimers`. Channel sections gated on `status="present"` — omitted channels are skipped in the narrative and called out in the Consistency section.
- [x] **Both mandatory disclaimers baked into the system prompt verbatim** — not-a-diagnosis + ON-med training caveat. Prompt uses "include BOTH sentences verbatim, in this order" so Claude doesn't paraphrase them into something weaker. Language matches CLAUDE.md → "Training-distribution bias" section.
- [x] **Consistency-flag rule wired into the system prompt**: "you MUST flag this for clinical review, never silently reconcile" — matches the CLAUDE.md "never silently override" invariant for cross-channel disagreement.
- [x] Feature-unit reminders in the system prompt (jitter/shimmer already in percent, HNR in dB, DDK rate in syl/s, F0 in Hz) so Claude quotes numbers with correct units. Also includes clinical typical-value anchors (jitter ~0.3–0.5%, DDK rate ≥ 6, blink rate ~12–20/min) so hedged language like "consistent with PD-typical perturbation" has a reference point.
- [x] Tone constraint: hedged language whitelist ("consistent with", "may suggest", "within the range typical of"), diagnostic-language blacklist ("indicates PD", "shows Parkinson's"), 2–4 sentences per channel section (keeps output well under `max_tokens=2048`).
- [x] `load_claude_config() -> dict` helper — matches `llm_fusion.load_fusion_config` pattern, saves callers from repeating the yaml parse.
- [x] `.env` picked up automatically via `python-dotenv` `load_dotenv()` at import time (both `anthropic` and `python-dotenv` already in `requirements.txt`).
- [x] CLI (`python -m src.report.claude_client <context_file>` or `-` for stdin) for smoke tests without wiring the whole pipeline. Missing-key path exits cleanly with a one-line message, not a stack trace.
- [x] Smoke tests: import + config load pass; system prompt token count = 1143 (matches design assumption); missing-key CLI path exits with "ANTHROPIC_API_KEY not set — export it or add to .env at repo root." **End-to-end API call not yet run** (requires user to supply `ANTHROPIC_API_KEY` in `.env` or shell env).

---

## Day 5 — End-to-End Demo (July 11) — NEVER CUT

### `src/pipeline.py`
- [x] `run_pipeline(vowel_dir, pataka_dir, smile_dir, call_claude=True) -> dict` — three task-matched upload dirs (matches Demo Protocol; runtime segmentation dropped 2026-07-09). Any dir → N/A on that channel and fusion renormalizes. Returns dict of `{scores, channels, channel_meta, facial_summary, fusion, claude_context, report}`.
- [x] Orchestrates: phonation (`_score_phonation` from `quick_score.py`) → DDK (`_score_ddk`) → facial (new local `_score_facial` uses `predict_and_summarize` for single-Docker-run per clip) → `fuse_scores` → `build_claude_context` → `generate_report`. All shared with `quick_score.py` for one canonical fusion + one canonical per-audio-channel scorer.
- [x] `--no-claude` flag returns context XML only (debug / API-key-less runs). `--out-dir` writes `context.xml`, `report.md`, `result.json`. `--label PD|HC` prints correct/incorrect after fusion.
- [x] Bug fixed on the way: `facial_features._run_openface` was calling `-aus` (AU-only output); `summarize.py` needs pose columns for `head_movement_std`. Switched to `-aus -pose`. See Day 5 finding #15.
- [x] Added `predict_and_summarize` to `predict_smile_pd.py` and `return_dataframe=True` option to `extract_smile_features` — one Docker run per clip yields both the classifier score and the hypomimia summary from the same CSV. Preserves the "one OpenFace run per demo" invariant on the actual code path (previously a spec statement only).
- [x] Smoke test on `data/samples/hc_demo/` end-to-end (phonation + DDK + facial → fused 0.157 → Claude Low-risk report with disagreement flag + both disclaimers verbatim). See Day 5 finding #14.
- [x] Rerun end-to-end on `data/samples/pd_demo/` through `pipeline.py`: **fused 0.704 → PD ✓, Elevated report**. Reproduces Day 4 finding #10's per-channel numbers exactly (phonation 0.613 / DDK 0.881 / facial 0.148 / fused 0.704), confirming pipeline.py doesn't silently drift from quick_score.py. Facial ≠ speech mean → correctly flagged. Full artifacts at `out/pd_demo/`. See Day 5 finding #17.

### `demo/app.py` — Gradio UI
- [x] Mock UI shell (683 lines): upload → 6-stage progress → 2-tier report (Vocal + Facial cards with collapsible per-channel breakdown), risk grade A/B/C, `New scan` button, MOCK_REPORT fixture
- [x] `demo/report_pdf.py` (433 lines) — PDF export for the demo report (bonus, ahead of scope)
- [ ] Wire `analyze()` to real `src/pipeline.py` (currently returns MOCK_REPORT)
- [ ] Task-recording instruction panel referencing the Demo Protocol in CLAUDE.md (sustained /a/ + PATAKA + smile ×3)
- [ ] Off-task channel handling in the UI: render "N/A — task not detected" when `pipeline` returns `score=None` on a channel

### Demo Materials
- [ ] Record three task-matched clips per Demo Protocol: (a) sustained /a/ ~5 s (audio ok), (b) /pa-ta-ka/ ~5 s (audio ok), (c) 8–12 s of smile ×3 alternating with neutral (video, face visible, per Islam 2023 protocol). Save to `data/samples/self_demo/{vowel,pataka,smile}.{wav|mp4}`.
- [ ] Prepare held-out NeuroVoz PD sample: pick one PD subject, verify it was the LOSO-held-out fold in Day 3 (no leakage), package their vowel + PATAKA files as the audio channels. Facial channel returns N/A (no video in NeuroVoz). Save to `data/samples/neurovoz_holdout_demo/`.
- [ ] End-to-end test on both demo cases; screenshot each report

---

## Day 6 — Freeze + Polish (July 12)

- [ ] README: setup, how to run demo, how to run ablation, NeuroVoz download instructions (Zenodo)
- [ ] Limitations page:
  - Moderate N (analysis cohort ≈ 49 PD vs 46 age-matched HC, from a raw 53 PD × 51 HC after HC age ≥ 50 filter)
  - **ON-medication training bias:** NeuroVoz PD subjects were ALL recorded 2–5h post-dose (paper §Data records). AUC / sensitivity numbers apply to the ON-state cohort — extrapolation to OFF-state PD is not supported by these results.
  - Spanish training data — task-matching (vowels + PATAKA) is language-neutral, but reading/free-speech channel carries a cross-language caveat
  - Task-matching is mandatory; off-task inputs return N/A on speech channels
  - Facial channel = smile classifier (7 AUs × mean+var, active-frame-only, in-distribution test AUROC 0.812 on UFNet split vs Islam 2023 smile-only 0.830 SVM ensemble) + hypomimia narrative JSON. OpenFace extraction matches training pipeline (zero AU domain gap). MediaPipe geometric + entropy statistics dropped as non-reproducible from paper. Demo-time detection rate < 80% → score withheld with warning.
  - Screening decision-aid, NOT a clinical diagnosis
  - No external corpora merged into training (dataset-bias avoidance)
- [ ] Roadmap: ParkCeleb cross-lingual robustness check if access is granted post-hackathon; second in-language corpus (IPVS Italian) for cross-lingual sensitivity analysis; paired audio+video PD corpora; larger multi-site validation
- [ ] Both deliverables verified working (ablation_table.csv + end-to-end demo)
- [ ] Feature freeze at 6pm

---

## Day 7 — Submit (July 13)

- [ ] Record demo video (screen capture: upload task-matched video → pipeline → report; then upload held-out NeuroVoz PD sample → report)
- [ ] Write 2–3 paragraph pitch summary — lead with the pivot rationale (motor-speech disorder → motor-speech features; open-access data → reproducible)
- [ ] Final submission

---

## Snapshot — What's left after Day 3 close (as of 2026-07-09)

Remaining critical-path work, roughly in dependency order:

1. ~~**`src/vision/summarize.py`**~~ — **DONE 2026-07-10.** ~180 lines, five markers (mean_AU12, AU12_amplitude_on_smile_cue, expression_variance, blink_rate_per_min, head_movement_std) + detection_rate + warnings; `hypomimia_score` dropped as arbitrary/redundant. See Day 4 checklist for design notes.
2. **`src/pipeline.py`** — end-to-end orchestrator: three uploads (vowel WAV/mp4, PATAKA WAV/mp4, smile mp4) → per-channel score via already-fitted Layer-1 classifiers → weighted vote → Claude context → report. Single entry point for `demo/app.py` to call. Any missing upload → that channel returns N/A.
3. **Wire `demo/app.py`** — replace `MOCK_REPORT` + mocked progress with real `pipeline.run_pipeline()` calls. UI change: **three upload widgets** (vowel, PATAKA, smile) instead of one, matching how every training corpus stores tasks. Keep the existing 2-tier report layout.
4. ~~**`src/fusion/llm_fusion.py`**~~ — **DONE 2026-07-10.** Two functions: `fuse_scores` (canonical late-fusion + agreement flags; also used by `quick_score.py`) and `build_claude_context` (XML-tagged prompt block, jitter/shimmer × 100 inside the function). See Day 4 checklist.
5. ~~**`src/report/claude_client.py`**~~ — **DONE 2026-07-10.** Non-streaming Anthropic SDK call to `claude-opus-4-7`, frozen ~1143-token system prompt with both mandatory disclaimers baked in, `thinking={"type": "disabled"}`, cache marker on system block (no-op today, self-activates past 4K tokens). **End-to-end API call validated 2026-07-10** on the synthetic PD demo case: fused=0.704 → Elevated report, both disclaimers verbatim, disagreement flag triggered exactly as specified. Pure narrative layer — does not affect AUC.
6. **Demo materials** — record three task-matched self-clips (vowel + PATAKA + smile); package one held-out NeuroVoz PD subject (verified LOSO-held, audio-only → facial N/A) as a second demo case.
7. **Day 6 polish** — README, limitations page (see Day 6 checklist), Day 7 demo screen-capture + pitch.

**Bonus deliverables from Day 4 end-to-end validation (2026-07-10):**
- `demo/assets/example_context.xml` — canonical Claude input (the XML block that goes into `generate_report`)
- `demo/assets/example_report.md` — canonical Claude output for the synthetic PD demo case
- Useful for pitch-deck illustration ("input → output at a glance") and for regression-checking that future prompt/config changes don't silently alter report format.

**Dropped from scope 2026-07-09:**
- `src/audio/segment.py` — three separate uploads make runtime segmentation unnecessary. See Day 2 dropped section + CLAUDE.md Demo Protocol.
- `py-feat` dependency — OpenFace CSV already contains AU_r/AU_c + head-pose columns needed for the hypomimia narrative.

Nice-to-have (skip if behind): per-vowel phonation modeling, XGBoost head-to-head, ParkCeleb cross-lingual check (see ParkCeleb discussion below), in-the-wild YouTube clip demo.
