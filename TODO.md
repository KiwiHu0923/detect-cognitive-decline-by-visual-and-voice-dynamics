# ParkScreen — Hackathon TODO

**Dates:** July 7–13, 2026 | **Track:** Development

Pivoted from CogniScreen (dementia) → ParkScreen (PD) on 2026-07-07. See CLAUDE.md Changelog for scientific rationale.

---

## Weekly Overview

| Day | Theme | Must-ship | Can-cut |
|-----|-------|-----------|---------|
| Day 1 (Mon 7/7) | Foundation + pivot | Repo + env + Whisper pipeline running; IPVS downloaded; Parselmouth sanity-checked | — |
| Day 2 (Tue 7/8) | Feature extraction | Phonation + DDK features for all IPVS samples (age-matched split) | Prosody channel |
| Day 3 (Wed 7/9) | **Ablation (never cut)** | Train 3 models (phonation, DDK, phonation+DDK late fusion), subject-level LOSO on IPVS, produce `ablation_table.csv` with real numbers | fancy plots |
| Day 4 (Thu 7/10) | Facial + Claude layer | py-feat hypomimia extraction + Claude 3-channel report | UI polish |
| Day 5 (Fri 7/11) | **End-to-end demo (never cut)** | Gradio app: task-matched video upload → full report rendered; held-out IPVS PD sample also runs | advanced UI |
| Day 6 (Sat 7/12) | Freeze + polish | README, limitations page (PD framing), two deliverables ready | new features |
| Day 7 (Sun 7/13) | Submit | Demo video + pitch deck + submission | — |

---

## Day 1 — Foundation + Pivot (July 7)

### Admin (post-pivot)
- [x] Update CLAUDE.md to ParkScreen (PD, IPVS, phonation+DDK late fusion)
- [x] Update TODO.md to ParkScreen
- [x] Download IPVS (593 MB zip → `data/`); extracted to `data/raw/italian_pd/` preserving group folders. 831 WAVs, all subject counts match paper (28 PD + 22 elderly HC + 15 young HC).
- [x] Decode `FILE CODES.xlsx`: vowels VA1/2..VU1/2 (phonation), D1=/pa/ + D2=/ta/ (DDK), B1/B2/FB1/PR1 (reading/prosody). DDK confirmed present for all 28 PD + 22 elderly HC.
- [ ] Build subject-level `labels.csv` at `data/raw/italian_pd/labels.csv` — columns: `subject_id, group (PD/HC-elderly/HC-young), age (from TAB 5.xlsx / Tab 3.xlsx / 15 YHC.xlsx), task, file_path`.
- [x] **Plan adjustment applied:** the 15 young HC subjects have NO vowel or DDK files, so the "secondary row on full HC set" was factually impossible. Removed from CLAUDE.md and TODO.md; cohort is now 28 PD vs 22 elderly HC only. `data/processed/cohort_full.csv` also removed from paths.yaml (not needed).

### Repo Setup
- [x] `git init` repo at `/Users/kiwi/Documents/anthropic_hackathon/`
- [x] Create folder structure: `src/audio/`, `src/vision/`, `src/fusion/`, `src/report/`, `eval/results/figures/`, `demo/assets/`, `data/processed/`, `data/samples/`, `notebooks/`, `configs/`, `tests/`
- [x] Rename `data/raw/adresso/` → `data/raw/italian_pd/`; `data/raw/backup/` → `data/raw/external/`; add `vowels/ ddk/ reading/` subdirs
- [x] Create `data/processed/phonation_features/`, `data/processed/ddk_features/`, `data/processed/prosody_features/`, `data/processed/facial_features/`, `data/processed/transcripts/`, `eval/models/`
- [x] Add `.gitignore` (excludes `data/`, `.env`, `__pycache__`, `*.wav`, `*.mp4`)
- [x] Create `.env.example` with `ANTHROPIC_API_KEY=`
- [x] Create `requirements.txt`
- [x] Create Python 3.12 venv at `.venv/` — activate: `source .venv/bin/activate`
- [x] Update `requirements.txt`: fix `feat` → `py-feat` typo, drop `librosa` / `opensmile` / `transformers` / `torch`, add `matplotlib`
- [x] Install all requirements: `pip install -r requirements.txt` (+ `brew install libomp` for xgboost/py-feat); added `openpyxl` for reading IPVS metadata sheets
- [ ] Sanity-check `praat-parselmouth` on a real vowel: load a sustained /a/ from IPVS (e.g. `VA1*.wav` from Anna B), print jitter+shimmer+HNR — verifies the toolchain end-to-end before Day 2
- [x] Verify `ffmpeg` available (`ffmpeg 8.0.1` ✓)

### Config Files
- [x] `configs/paths.yaml` — all data/output paths
- [x] Update `configs/paths.yaml` for IPVS layout (vowels/ ddk/ reading/) + processed feature dirs + `eval/models/` + cohort manifests
- [x] `configs/model.yaml` — Whisper model, pause threshold, fusion weights, Claude model (`claude-opus-4-7`)
- [x] Update `configs/model.yaml`: add `phonation:` (F0 75–500 Hz, min voicing 1.0s, jitter/shimmer params), `ddk:` (peak prominence, min ISI 0.05s, envelope smoothing), `prosody:` (min pause 0.25s), per-channel `fusion:` weights (phonation 0.50 / DDK 0.35 / facial 0.15 / prosody 0.00 unless enabled), `cohort:` (primary vs secondary)

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

## Day 2 — Feature Extraction (July 8)

### Task Segmentation (`src/audio/segment.py`)
- [ ] `segment_tasks(wav_path) -> {"vowel": (t0, t1), "ddk": (t0, t1), "reading": (t0, t1) | None}`
- [ ] Heuristics: vowel = long continuously-voiced region with low F0 variance; DDK = regular high-rate intensity-peak train; reading region isolated via Whisper timestamps (from `transcribe.py`)
- [ ] For IPVS batch processing, task is known from folder — segmentation is only needed at demo time. Provide both code paths: `segment_from_folder(subject_dir)` (IPVS) and `segment_from_audio(wav_path)` (demo).

### Phonation Features (`src/audio/phonation.py`)
- [ ] `extract_phonation_features(wav_path, t0=None, t1=None) -> dict`
- [ ] Parselmouth: jitter (local, ppq5, rap), shimmer (local, apq5, apq11, dda), HNR (mean, dB), F0 mean/std/range, voicing fraction
- [ ] Guardrails: skip files < 1 s of continuous voicing; return `{}` on failure (log warning, don't crash batch)
- [ ] Batch script: all IPVS vowel files → `data/processed/phonation_features/{subject_id}.npy` + column-name manifest

### Articulation / DDK Features (`src/audio/ddk.py`)
- [ ] `extract_ddk_features(wav_path, t0=None, t1=None) -> dict`
- [ ] Intensity envelope (Parselmouth `Intensity` or Hilbert magnitude via scipy) → `scipy.signal.find_peaks` with min prominence + min ISI from config
- [ ] Features: DDK rate (syllables/sec), ISI mean, ISI CV (regularity — lower = more regular), peak-amplitude mean, peak-amplitude CV, amplitude decrement (linear slope of peak amplitudes over time)
- [ ] Batch script: all IPVS DDK files (/pa/ + /ta/ separately, then averaged per subject) → `data/processed/ddk_features/{subject_id}.npy`

### (Optional) Prosody Features (`src/audio/prosody.py`) — cut first if behind
- [ ] `extract_prosody_features(whisper_json, wav_path, t0, t1) -> dict`
- [ ] Speech rate (words/sec), mean pause duration, pause rate/min (from `transcribe_file()` output), F0 variability (SD, semitones — monotonicity marker), F0 range voiced, intensity mean/SD
- [ ] Batch script: IPVS reading files → `data/processed/prosody_features/{subject_id}.npy`

### Data Sanity
- [ ] Verify subject-level file grouping (each subject has both vowel + DDK files present)
- [ ] Build age-matched cohort manifest (28 PD vs 22 elderly HC); young HCs auto-excluded — no vowel or DDK files exist for them
- [ ] Plot feature distributions per group (jitter/shimmer/HNR/DDK rate); check for NaN/inf, remove or impute per feature
- [ ] Save cohort manifest: `data/processed/cohort_primary.csv`

---

## Day 3 — Ablation Study (July 9) — NEVER CUT

### `src/fusion/statistical.py`
- [ ] Load per-channel feature matrices + subject-level labels + cohort manifest
- [ ] Per-channel model: LogReg (l2, C tuned via inner CV) OR linear SVM with probability calibration — pick the more robust of the two on primary cohort
- [ ] Late fusion: per-channel probabilities → average (baseline) OR logistic combiner fitted on per-channel probs
- [ ] Save fitted classifiers + scalers to `eval/models/{channel}.joblib` for Layer 2 to load at demo time

### `eval/ablation.py`
- [ ] Subject-level LOSO on IPVS cohort (28 PD vs 22 elderly HC — age-matched, enforced by the data since young HCs have no vowel/DDK files). Each subject's ALL files (vowel + DDK) held out together — no within-subject leakage.
- [ ] Metrics: AUC, F1, sensitivity, specificity per model + 95% CI (bootstrap over subjects)
- [ ] Rows produced:
  1. Phonation-only
  2. DDK-only
  3. Phonation + DDK (late fusion)
  4. (Optional) Phonation + DDK + Prosody
- [ ] Save `eval/results/ablation_table.csv`
- [ ] ROC curve plot (one line per model, primary cohort) → `eval/results/figures/roc_curves.png`
- [ ] Coefficient-magnitude plot per channel (models are linear → interpretable) → `eval/results/figures/coefficients.png`
- [ ] `notebooks/03_ablation_results.ipynb` with rendered table + figures

---

## Day 4 — Facial Path + Claude Layer (July 10)

### `src/vision/extract.py`
- [ ] `extract_facial_features(video_path) -> dict` using py-feat
- [ ] Per-frame AU timeseries (esp. AU12) + head pose + gaze + emotion probs

### `src/vision/summarize.py`
- [ ] `summarize_facial_features(timeseries_json) -> dict`
- [ ] Hypomimia-focused output: `mean_AU12`, `AU12_amplitude_on_smile_cue`, `expression_variance`, `hypomimia_score` (composite), `blink_rate_per_min`, `head_movement_std`, `dominant_emotion`, `emotion_variability`
- [ ] Facial channel is qualitative — no classifier fitted, no LOSO. Passed through raw to Claude.

### `src/fusion/llm_fusion.py`
- [ ] `build_claude_context(phonation_score, ddk_score, prosody_score_or_none, facial_summary, transcript_excerpt_or_none) -> str`
- [ ] Compute weighted vote: phonation weight highest, DDK next, facial lowest; support N/A on any channel (in-the-wild off-task inputs)
- [ ] Compute per-channel agreement flag (do speech channels agree with each other? does facial agree with speech consensus?)

### `src/report/claude_client.py`
- [ ] `generate_report(context_str) -> str` — Anthropic SDK, `claude-opus-4-7`, prompt caching on the system prompt
- [ ] Report sections: risk level (Low/Moderate/Elevated — NOT diagnosis), per-channel narrative (phonation, DDK, facial), consistency flag, recommended next steps, explicit disclaimer (screening decision-aid, not diagnosis)
- [ ] Handle N/A channels gracefully in the prompt (e.g. off-task input skips phonation/DDK narrative sections)

---

## Day 5 — End-to-End Demo (July 11) — NEVER CUT

### `src/pipeline.py`
- [ ] `run_pipeline(video_path) -> report_dict`
- [ ] Orchestrates: extract_audio → segment_tasks → phonation → ddk → (optional prosody) → facial extract → facial summarize → load fitted Layer-1 classifiers → score per channel → weighted vote → build Claude context → generate report
- [ ] Return report + per-channel scores + agreement flags (for UI display)

### `demo/app.py`
- [ ] Gradio UI: single video upload + a clear "record these tasks" instruction panel referencing the Demo Protocol in CLAUDE.md
- [ ] Progress indicators per pipeline stage
- [ ] Report display: risk level, per-channel scores, consistency flag, narrative, disclaimer
- [ ] Off-task channel handling: show "N/A — task not detected" rather than a bogus score

### Demo Materials
- [ ] Record task-matched demo video: sustained /a/ ~5 s → /pa/ ~5 s → /ta/ ~5 s → (optional) short reading → face visible throughout with a smile cue near end. Save to `data/samples/self_demo.mp4`.
- [ ] Prepare held-out IPVS PD sample: pick one PD subject, verify it was the LOSO-held-out fold in Day 3 (no leakage), package their vowel + DDK files as a "clip" for the demo. Save to `data/samples/ipvs_holdout_demo/`.
- [ ] End-to-end test on both demo cases; screenshot each report

---

## Day 6 — Freeze + Polish (July 12)

- [ ] README: setup, how to run demo, how to run ablation, IPVS download instructions
- [ ] Limitations page:
  - Small N (28 PD vs 22 elderly HC)
  - Italian training data — task-matching (vowels + DDK) is language-neutral, but reading channel carries a cross-language caveat
  - Task-matching is mandatory; off-task inputs return N/A on speech channels
  - Facial channel is qualitative, not a validated classifier
  - Screening decision-aid, NOT a clinical diagnosis
  - No external corpora merged into training (dataset-bias avoidance)
- [ ] Roadmap: NeuroVoz / ParkCeleb cross-lingual robustness check if access is granted post-hackathon; paired audio+video PD corpora; larger multi-site validation
- [ ] Both deliverables verified working (ablation_table.csv + end-to-end demo)
- [ ] Feature freeze at 6pm

---

## Day 7 — Submit (July 13)

- [ ] Record demo video (screen capture: upload task-matched video → pipeline → report; then upload held-out IPVS PD sample → report)
- [ ] Write 2–3 paragraph pitch summary — lead with the pivot rationale (motor-speech disorder → motor-speech features; open-access data → reproducible)
- [ ] Final submission
