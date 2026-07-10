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
| Day 4 (Thu 7/10) | Facial + Claude layer | py-feat hypomimia extraction + Claude 3-channel report | UI polish |
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

### Task Segmentation (`src/audio/segment.py`)
- [ ] `segment_tasks(wav_path) -> {"vowel": (t0, t1), "ddk": (t0, t1), "reading": (t0, t1) | None}`
- [ ] Heuristics: vowel = long continuously-voiced region with low F0 variance; DDK = regular high-rate intensity-peak train; reading region isolated via Whisper timestamps (from `transcribe.py`)
- [ ] For NeuroVoz batch processing, task is known from filename (`{group}_{task}_{subject}.wav`) — segmentation is only needed at demo time. Provide both code paths: `segment_from_manifest(labels_csv)` (NeuroVoz) and `segment_from_audio(wav_path)` (demo).

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

### `src/vision/facial_features.py` — per-frame OpenFace extraction
- [ ] `extract_smile_features(video_path, t0=None, t1=None) -> tuple[dict[str, np.ndarray], dict]`
- [ ] Returns arrays `{'au_r': np.ndarray[N, 7], 'au_c': np.ndarray[N, 7]}` for 7 AUs (AU01, AU06, AU12, AU14, AU25, AU26, AU45) — intensity 0–5 and presence 0/1
- [ ] Subprocess wrapper around `docker run --rm --platform linux/amd64 --entrypoint="" -v <tmp>:/data algebr/openface:latest ./build/bin/FeatureExtraction -f /data/clip.mp4 -aus -out_dir /data/out` — falls back to no `-aus` flag if that subflag is rejected
- [ ] Confidence < 0.75 or `success == 0` frames → NaN masked (per-column NaN, aggregate.py handles). This is defense-in-depth vs Islam 2023 which specifies no explicit frame filter.
- [ ] Meta: `{fps, n_frames_total, n_frames_used, detection_rate, quality_gate_pass (>= 0.80), warnings[]}` — detection_rate exposed to Claude context per feedback (dropped frames are non-random; must be flagged)
- [ ] `[t0, t1]` handled by `ffmpeg` pre-cutting a temp clip before Docker invocation

### `src/vision/aggregate.py` — session-level 14-dim vector (active-frame-only)
- [ ] `aggregate(arrays: dict) -> tuple[np.ndarray[14], dict]`
- [ ] For each of 7 AUs: `active = au_c == 1; mean = au_r[active].mean(); var = au_r[active].var()` — active-frame-only per Islam 2023 §2.
- [ ] If no active frame for a given AU → fill 0 for both mean and var, add column name to `zero_filled_columns` in meta.
- [ ] Column order must match `eval/models/smile_pd_columns.json` exactly (retrained at 14 features): `smile_AU01_mean, smile_AU01_var, smile_AU06_mean, ..., smile_AU45_var`
- [ ] Per-minute features (blink rate etc. for hypomimia summary) use `n_frames_used / fps` as denominator, never `t1 - t0` — dropping frames must not inflate rates.

### `src/vision/predict_smile_pd.py` — video → PD probability
- [ ] `predict(video_path, t0=None, t1=None, min_detection_rate=0.80) -> dict{"score", "detection_rate", "quality_gate_pass", "features", "warnings"}`
- [ ] Load `eval/models/smile_pd_lr.joblib` + `smile_pd_scaler.joblib` + `smile_pd_columns.json`
- [ ] Pipeline: `facial_features.extract_smile_features` → `aggregate.aggregate` → `scaler.transform` → `clf.predict_proba`
- [ ] If `quality_gate_pass == False`: return `score=None` with detection_rate reason string. Do not silently emit a score on low-quality input.

### `src/vision/summarize.py` — hypomimia narrative JSON (companion to classifier)
- [ ] `summarize_facial_features(video_path) -> dict`
- [ ] Uses py-feat (separate from OpenFace, to avoid double-Docker cost) for AU + head-pose + emotion outputs → `mean_AU12`, `AU12_amplitude_on_smile_cue`, `expression_variance`, `hypomimia_score` (composite), `blink_rate_per_min`, `head_movement_std`, `dominant_emotion`, `emotion_variability`
- [ ] Passed to Claude alongside the classifier score for narrative colour, not as a second classifier.
- [ ] Per-minute rates use kept-duration denominator; hypomimia_score narrative includes detection_rate warning when < 0.80.

### `src/fusion/llm_fusion.py`
- [ ] `build_claude_context(phonation_score, ddk_score, facial_score, facial_summary, transcript_excerpt_or_none) -> str`
- [ ] Compute weighted vote: phonation weight highest, DDK next, facial (from smile classifier) mid-low; support N/A on any channel (in-the-wild off-task inputs)
- [ ] Compute per-channel agreement flag (do speech channels agree with each other? does facial agree with speech consensus?)
- [ ] **Unit conversion for Claude context (NeuroVoz cross-check finding, 2026-07-08):** jitter and shimmer are stored in ratio units (0.005, 0.044) but clinical convention is percent (0.5%, 4.4%). Convert `× 100` before passing to Claude, and label the units explicitly in the prompt (`"jitter_local_percent"` not `"jitter_local"`). Otherwise Claude may narrate "jitter 0.005 is below detection threshold" — wrong.

### `src/report/claude_client.py`
- [ ] `generate_report(context_str) -> str` — Anthropic SDK, `claude-opus-4-7`, prompt caching on the system prompt
- [ ] Report sections: risk level (Low/Moderate/Elevated — NOT diagnosis), per-channel narrative (phonation, DDK, facial), consistency flag, recommended next steps, explicit disclaimer (screening decision-aid, not diagnosis)
- [ ] **Mandatory ON-medication caveat in every report** (NeuroVoz paper §Data records): PD training subjects were all recorded 2–5h post-dose. System prompt must instruct Claude to include a sentence noting this training bias, separately from the not-a-diagnosis disclaimer. See CLAUDE.md → "Training-distribution bias" section for exact language.
- [ ] Handle N/A channels gracefully in the prompt (e.g. off-task input skips phonation/DDK narrative sections)

---

## Day 5 — End-to-End Demo (July 11) — NEVER CUT

### `src/pipeline.py`
- [ ] `run_pipeline(video_path) -> report_dict`
- [ ] Orchestrates: extract_audio → segment_tasks → phonation → ddk → facial extract → facial summarize → load fitted Layer-1 classifiers → score per channel → weighted vote → build Claude context → generate report
- [ ] Return report + per-channel scores + agreement flags (for UI display)

### `demo/app.py`
- [ ] Gradio UI: single video upload + a clear "record these tasks" instruction panel referencing the Demo Protocol in CLAUDE.md
- [ ] Progress indicators per pipeline stage
- [ ] Report display: risk level, per-channel scores, consistency flag, narrative, disclaimer
- [ ] Off-task channel handling: show "N/A — task not detected" rather than a bogus score

### Demo Materials
- [ ] Record task-matched demo video: sustained /a/ ~5 s → /pa-ta-ka/ ~5 s → (optional) short reading → **8–12 s of smile ×3 alternating with neutral (~2–3s smile + ~1–2s neutral per cycle, per Islam 2023 protocol)** → face visible throughout. Save to `data/samples/self_demo.mp4`.
- [ ] Prepare held-out NeuroVoz PD sample: pick one PD subject, verify it was the LOSO-held-out fold in Day 3 (no leakage), package their vowel + PATAKA files as a "clip" for the demo. Save to `data/samples/neurovoz_holdout_demo/`.
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
