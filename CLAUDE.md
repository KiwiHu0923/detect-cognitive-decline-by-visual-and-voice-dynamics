# ParkScreen — Built with Claude: Life Sciences Hackathon

## Working Protocol

Before every non-trivial code change:
1. **Brief the user** — explain each function/block in plain language (algorithm, non-obvious decisions)
2. **Wait for "go"** before touching any files
3. **After implementation** — mark TODO.md items `[x]` and update CLAUDE.md if architecture/schemas changed

## Project Overview

Multimodal screening tool that detects motor-speech and facial signs of Parkinson's disease (PD) from three task-matched uploads. Fuses **phonation** signals (sustained-vowel voice quality) and **articulation** signals (DDK syllable rate/regularity) with **facial dynamics** (hypomimia), routed through a Claude integration layer that generates a clinical-style report.

**Positioning:** a supporting-evidence / screening decision-aid, explicitly **not a diagnosis**.

**Hackathon:** Built with Claude: Life Sciences (Anthropic × Gladstone × Cerebral Valley), July 7–13 2026. Development Track.

**Related docs (read these when relevant, don't duplicate them here):**
- `docs/METHODOLOGY.md` — full methodology, sensitivity analyses (Day 3/4/5 findings), cohort audit (age/sex/severity/duration confounder checks), reproducibility pointers.
- `docs/LIMITATIONS.md` — evaluator-facing caveats.
- `docs/DATASETS.md` — dataset citations, DUA compliance, license summary.
- `TODO.md` — deliverables checklist.

---

## Project Origin (One-liner)

Pivoted from CogniScreen (dementia) to ParkScreen (PD) on 2026-07-07 to remove a dataset-access blocker and target a scientifically better fit: PD is a motor-speech disorder, so motor-speech features (jitter/shimmer/HNR + DDK) are direct evidence. Dataset swapped again IPVS → NeuroVoz on 2026-07-08 when NeuroVoz's DUA cleared. See git log for full pivot history.

---

## Two-Layer Architecture

### Layer 1 — Scientific Validation Spine (offline, NeuroVoz)

**Score-level (late) fusion** of two speech channels from the same subjects but different tasks. Purpose: rigorous ablation table showing fusion > single-channel. Answers "does fusion actually add signal?".

- **Phonation channel** — sustained vowels /i/o/u/ (dropped /a/e/ per Li et al. 2606.19125) → jitter, shimmer, HNR, F0 stats (Parselmouth).
- **Articulation channel (DDK)** — /pa-ta-ka/ (PATAKA task) → DDK rate + timing/amplitude regularity via intensity-envelope peak picking (no ASR, language-neutral).
- **Not concatenated.** Each channel trains its own LogReg; probabilities fused at score level. Different task, different scale, different noise floor.
- **Model class:** LogReg + StandardScaler (Pipeline, re-fit per fold). Small + interpretable. N ≈ 95 subjects — deep learning is overparameterized.
- **Eval:** subject-level LOSO; metrics AUC/F1/Sens/Spec + bootstrap 95% CI. See `docs/METHODOLOGY.md` §4 for full protocol.

### Layer 2 — Product Layer (real-time, task-matched uploads)

User uploads three task-matched dirs → pipeline extracts features + scores each channel with the already-fitted Layer-1 classifiers + facial classifier → fuses at score level with weighted voting → passes structured scores + facial hypomimia JSON as context into a Claude prompt → Claude generates a clinical-style report with confidence flags.

No training at demo time.

**Fusion weights (AUC-excess-over-chance):** `w_c ∝ max(0, AUC_c − 0.5)`, renormalized over channels present. Current: phonation 0.35, DDK 0.65, facial 0.15 (see `configs/model.yaml` for the derivation comment). Rationale: DDK carries the largest AUC-excess in Day 3 LOSO; phonation second; facial deprecated slightly because it's domain-transferred (UFNet → user webcam).

**Channel agreement flags:** If per-channel scores disagree beyond `agreement_threshold` (0.30), report flags for clinical review — **never silently reconciled**.

---

## Data

### Primary Dataset: NeuroVoz (Spanish)

- **Source:** Zenodo (DUA cleared 2026-07-08). Spanish PD voice corpus, ~2900 audio files across 53 PD + 55 HC subjects.
- **Three task families:**
  1. Sustained vowels /a/e/i/o/u/, up to 3 reps each → phonation channel (we use only I/O/U — see METHODOLOGY §3.1).
  2. **PATAKA** — canonical clinical DDK task → articulation channel.
  3. Word/sentence tasks + FREE spontaneous speech — present but **not used** (prosody channel dropped 2026-07-09).
- **Language note:** vowels + PATAKA are language-neutral; demo remains cross-language compatible.
- **Analysis cohort:** age-matched (HC ≥ 50) + PATAKA + ≥ 1 vowel → **49 PD × 46 HC = 95 subjects**. Full lineage in METHODOLOGY §2.
- **Bonus clinical labels** (used by report layer, not classifier training): UPDRS, Hoehn-Yahr stage, disease duration, med ON/OFF, per-file GRBAS ratings, hypophonic-voice/tremor/dysphagia flags.
- **Pre-extracted NeuroVoz eGeMAPS features** available at `data/raw/neurovoz/data/audio_features/audio_features.csv` — used as a sanity check only (our Parselmouth features cross-check r=0.46–0.82 vs their columns; their `HNR` column has an undocumented sign convention — do not use it — ours is standard Praat).
- Path: `data/raw/neurovoz/data/`. Filename encodes group_task_subject (e.g. `PD_PATAKA_0004.wav`).

**Training-distribution bias — mandatory disclaimer:** PD subjects were ALL recorded ON medication (2–5h post-dose per NeuroVoz paper §Data records). Model calibration is valid for ON-state PD vs age-matched HC only. This must appear in every Claude report (see §Claude Integration below).

### External corpora (facial only)

- **UFNet** (ROC-HCI, AAAI 2025, MIT license) — smile-task feature CSV for training the facial classifier. 1,361 subjects. At `data/raw/ufnet_smile/`.
- **YouTubePD** — informational external validation only, per UFNet's shipped subset. Not used for training.
- **No external corpora merged into NeuroVoz training** — dataset-bias avoidance.

### Demo Data

Task-matched user uploads → three separate dirs (`vowel/`, `pataka/`, `smile/`), any file type ffmpeg can decode. At `data/samples/`.

---

## Demo Protocol (task-matching is mandatory)

Demo tasks MUST match training tasks — otherwise features are out-of-distribution and the score is meaningless. Every corpus we use stores tasks as separate files, so ParkScreen collects **three separate upload dirs** at demo time and skips runtime segmentation (no `src/audio/segment.py`).

Three upload dirs (any of `.wav`, `.m4a`, `.mp4`, `.mov`):

1. **Sustained vowels /i/, /o/, /u/** — up to 3 reps per vowel, ~3–5s each. Filenames follow `<group>_<VOWEL><REP>.<ext>` (e.g. `PD_I1.m4a`). At least one of {I, O, U} required; weights renormalize over what's actually recorded. Audio-only OK.
2. **Rapid /pa-ta-ka/** — 1+ reps of ~5s each, as fast and steady as possible. Features averaged across reps. Audio-only OK.
3. **8–12 seconds of smile ×3 alternating with a neutral face** — must be video, face clearly visible, per Islam 2023 protocol. Multiple clips allowed; pipeline max-pools the classifier score over clips.

Any channel whose upload dir is empty → pipeline returns `N/A` for that channel; Claude report is generated with remaining channels; fusion renormalizes.

**Data-leakage guard:** If a held-out NeuroVoz PD sample is used as a demo, it MUST be the subject left out in its LOSO fold — never a subject the classifier was fit on. The self-recorded volunteer sample is out-of-training by construction.

**Optional in-the-wild demo** (e.g. YouTube PD clip): task mismatch → speech channels return N/A, only facial channel scored. Illustrative only, never counted toward AUC.

---

## Tech Stack

| Component | Library | Notes |
|-----------|---------|-------|
| Audio extraction | ffmpeg via subprocess | 16kHz mono WAV |
| Phonation features | `praat-parselmouth` | jitter, shimmer, HNR, F0 — interpretable core |
| Articulation / DDK features | `praat-parselmouth` + `scipy.signal` | intensity-envelope peak picking, no ASR, language-neutral |
| Facial AU extraction | OpenFace 2.0 via Docker `algebr/openface` | matches UFNet training extractor; one Docker run per demo clip yields classifier vector + hypomimia summary |
| Statistical fusion | `scikit-learn` | LogReg per channel + score-level fusion |
| LLM / report | `anthropic` SDK | `claude-opus-4-7`, non-streaming, thinking disabled, ~1143-token frozen system prompt |
| UI | `gradio` | wired to `run_pipeline` |

---

## Repository Structure

```
parkscreen/
├── data/                          # gitignored
│   ├── raw/
│   │   ├── neurovoz/data/         # NeuroVoz: audios/ metadata/ transcriptions/ grbas/ audio_features/
│   │   ├── ufnet_smile/           # UFNet training + YouTubePD validation CSVs
│   │   └── external/              # (bonus) ParkCeleb if approved
│   ├── processed/
│   │   ├── phonation_features/    # per_file.csv + per_subject.csv + per_subject_per_vowel.csv
│   │   ├── ddk_features/          # per_file.csv + per_subject.csv (identical — one PATAKA per subject)
│   │   └── cohort.csv             # 104 subject-level rows, 95 in analysis cohort
│   └── samples/                   # demo dirs (hc_demo/, pd_demo/, neurovoz_holdout_demo/, ...)
│
├── src/
│   ├── audio/
│   │   ├── transcribe.py          # Whisper wrapper — NOT on demo path (kept for tests + future debug)
│   │   ├── phonation.py           # Parselmouth jitter/shimmer/HNR/F0 on vowels
│   │   └── ddk.py                 # intensity-envelope peak picking → DDK rate + regularity
│   ├── vision/
│   │   ├── train_smile_pd.py      # trains 14-feature smile classifier on UFNet CSV
│   │   ├── facial_features.py     # OpenFace Docker → per-frame AU_r + AU_c + pose
│   │   ├── aggregate.py           # per-frame → 14-dim session vector (active-frame-only)
│   │   ├── predict_smile_pd.py    # video → PD score + detection-rate gate; `predict_and_summarize` for one-run classifier+summary
│   │   └── summarize.py           # OpenFace CSV → hypomimia JSON for Claude
│   ├── fusion/
│   │   ├── statistical.py         # Layer 1: per-channel LogReg + score-level late fusion
│   │   ├── llm_fusion.py          # canonical `fuse_scores` + `build_claude_context`
│   │   └── quick_score.py         # debug CLI — same fusion as pipeline.py
│   ├── report/
│   │   └── claude_client.py       # Anthropic SDK + frozen system prompt (~1143 tokens)
│   ├── data/
│   │   ├── build_labels.py        # NeuroVoz metadata → per-file labels.csv
│   │   ├── build_cohort.py        # per-file → per-subject cohort.csv (with in_analysis_cohort flag)
│   │   └── qa_features.py         # standalone QA: per-group feature-distribution plots + NaN/Inf check
│   └── pipeline.py                # single entry point: three upload dirs → dict + report
│
├── eval/
│   ├── ablation.py                # subject-level LOSO across 8 model rows (+ inline AUC/F1/sens/spec + bootstrap CI)
│   ├── make_plots.py              # ROC curves, coefficient bar plots (L1 phonation viz)
│   ├── eval_smile_yt_subset.py    # facial external-validation on UFNet's YouTubePD subset
│   ├── models/                    # deployment classifiers (LogReg + Scaler joblib)
│   └── results/                   # ablation_table.csv, coefficients.csv, loso_oof_probs.csv, figures/
│
├── demo/
│   ├── app.py                     # Gradio UI (wired to src/pipeline.run_pipeline); launch: `python -m demo.app`
│   ├── report_pdf.py              # PDF export
│   └── assets/                    # example_context.xml, example_report.md (canonical example artifacts)
│
├── docs/
│   ├── METHODOLOGY.md             # data lineage + features + training + sensitivity + audit + reproducibility
│   ├── LIMITATIONS.md             # evaluator-facing caveats
│   └── DATASETS.md                # citations, licenses, DUA compliance
│
├── notebooks/
│   └── ablation_results.ipynb     # executable visual companion to METHODOLOGY §4-5
├── configs/                       # model.yaml (hyperparams, fusion weights), paths.yaml
├── tests/                         # test_pipeline.py smoke tests
├── README.md
├── LICENSE
├── requirements.txt
├── TODO.md
└── CLAUDE.md
```

---

## Phonation Features (sustained vowels, Parselmouth)

**Preprocessing:** steady-window crop to middle 60% (`configs/model.yaml → phonation.apply_steady_window: true`). Onset/offset carry glottal transients that inflate jitter/shimmer/F0-std. Mandatory on demo path — see METHODOLOGY §5.3 for why.

**Vowel filter + weighted aggregation:** use only I/O/U (drop A/E per Li et al. 2606.19125). Per-subject × per-vowel mean of reps, then cross-vowel weighted average (I=0.310, O=0.310, U=0.379, renormalized over vowels present).

**Feature list (12 per file):** jitter (local/rap/ppq5), shimmer (local/apq5/apq11/dda), HNR mean dB, F0 mean/std/range, voicing fraction. See METHODOLOGY §3.1.

## Articulation / DDK Features (PATAKA, no ASR)

**Method:** intensity-envelope peak picking (rectify → moving-average → `scipy.signal.find_peaks` with prominence 0.15×env_max, min ISI 0.05s). Language-neutral.

**Feature list (8 per file):** `n_peaks`, `duration_s`, `ddk_rate_hz`, `isi_mean_s`, `isi_cv`, `amp_mean`, `amp_cv`, `amp_decrement`. See METHODOLOGY §3.2. Duration audit (§7.4) confirms model is not primarily using recording length.

## Facial Features — Smile Classifier + Hypomimia Summary

**Two outputs from one OpenFace Docker run per clip.** OpenFace called with `-aus -pose` flags (so `pose_Tx/Ty/Tz` present for `head_movement_std`).

### 1. Smile classifier score

Trained on UFNet CSV, 14 features = 7 AUs × {mean, var}, active-frame-only aggregation per Islam 2023 §2. In-distribution test AUROC on UFNet split = **0.812** vs paper's smile-only 0.830 SVM ensemble.

- Model: LogReg + StandardScaler + SMOTE
- 7 AUs: AU01, AU06, AU12 (dominant), AU14, AU25, AU26, AU45
- Task at demo: 8–12s smile ×3 alternating with neutral, per Islam 2023 protocol
- Quality gate: detection_rate < 0.80 → score withheld with reason

See METHODOLOGY §3.3 + §6 for external validation on YouTubePD and reproducibility subtleties (UFNet's `ID` vs `Participant_ID` gotcha, NaN=PD rule, active-frame-only aggregation).

Saved artifacts: `eval/models/smile_pd_{lr,scaler,columns,metrics}.*`.

### 2. Hypomimia summary (Claude narrative)

Computed by `src/vision/summarize.py` from the **same** OpenFace CSV — no second Docker run.

Fields: `mean_AU12`, `AU12_amplitude_on_smile_cue` (max on active frames), `expression_variance`, `blink_rate_per_min`, `head_movement_std`, `detection_rate`, `warnings`.

**No composite `hypomimia_score`** — a composite requires an arbitrary normalization anchor and duplicates work Claude does at report synthesis.

---

## Claude Integration (Layer 2)

Four files, clear separation of concerns:

- **`src/vision/summarize.py`** — hypomimia narrative JSON (see above).
- **`src/fusion/llm_fusion.py`** — canonical `fuse_scores` (also used by `quick_score.py` and `pipeline.py`) + `build_claude_context` (composes XML block for Claude; ratio→percent unit conversion isolated here).
- **`src/report/claude_client.py`** — Anthropic SDK call to `claude-opus-4-7`. Non-streaming, `max_tokens=2048`, `thinking={"type": "disabled"}`, no sampling params (Opus 4.7 400s on `temperature`/`top_p`/`top_k`). Frozen ~1143-token `SYSTEM_PROMPT`; `cache_control: {"type": "ephemeral"}` marker set but no-op today (Opus 4.7 cacheable-prefix floor is 4096 tokens; self-activates if prompt grows past 4K).
- **`src/pipeline.py`** — end-to-end orchestrator. `run_pipeline(vowel_dir, pataka_dir, smile_dir, call_claude=True) -> dict`. Reuses `_score_phonation` and `_score_ddk` from `quick_score.py` (one canonical audio scorer); facial uses `predict_and_summarize` for single-Docker-run per clip.

**System prompt non-negotiables** (enforced at generation time):
- Report structure: Risk Level → Phonation → DDK → Facial → Cross-Channel Consistency → Recommended Next Steps → Disclaimers
- Omitted channels skipped narratively AND called out in Consistency
- `any_flag_for_review=true` → explicit "Per-channel disagreement — flagged for clinical review", never silently reconciled
- Both mandatory disclaimers verbatim (screening-decision-aid + ON-medication caveat)
- Feature units labelled (jitter/shimmer percent, HNR dB, DDK syl/sec, F0 Hz) + clinical anchor ranges for hedged language
- Hedged-tone whitelist, diagnostic-language blacklist, 2–4 sentences per channel section

Canonical example artifacts at `demo/assets/example_context.xml` (input) and `demo/assets/example_report.md` (output).

---

## Evaluation — Headline Ablation

Subject-level LOSO on NeuroVoz analysis cohort (49 PD × 46 HC = 95 subjects). LogReg (l2, C=1.0) inside `Pipeline([StandardScaler, LogisticRegression])`, re-fit per fold. Bootstrap 95% CI on AUC (1000 resamples).

| Model | AUC [95% CI] | F1 | Sens | Spec |
|-------|--------------|----|------|------|
| Phonation-only (per-subject) | 0.567 [0.455, 0.678] | 0.545 | 0.490 | 0.674 |
| DDK-only (PATAKA) | **0.740** [0.632, 0.845] | 0.701 | 0.694 | 0.696 |
| Phonation + DDK (AUC-excess weighted) | 0.736 [0.634, 0.842] | 0.660 | 0.633 | 0.696 |
| Phonation(per-file) + DDK (AUC-excess weighted) | **0.758** [0.662, 0.859] | 0.667 | 0.633 | 0.717 |

Full 8-row table + sensitivity analyses + cohort audit in `docs/METHODOLOGY.md` §4–§7.

---

## Scope Rules (Hackathon Constraints)

**Never cut:**
- Ablation table (phonation-only, DDK-only, phonation+DDK late fusion, subject-level LOSO on NeuroVoz)
- End-to-end demo on task-matched uploads

**Honest framing (must appear in reports and in `docs/LIMITATIONS.md`):**
- Channel inconsistency → flag for review, never silently override
- Demo score only meaningful if input is task-matched
- Screening decision-aid, NOT a clinical diagnosis
- ON-medication training caveat: PD training subjects all recorded 2–5h post-dose
- Age imbalance: PD 5 years older than HC on average (p=0.013). See METHODOLOGY §7.1.
- No external corpora merged into training

---

## Environment

- Platform: macOS (Apple Silicon)
- `mlx_whisper` requires Apple Silicon; swap to `openai-whisper` for CI/server
- API key: `ANTHROPIC_API_KEY` in `.env`
- Python: 3.12 (via `/opt/homebrew/opt/python@3.12/bin/python3.12`)
- Venv: `.venv/` at repo root — activate with `source .venv/bin/activate`
- Claude model: `claude-opus-4-7`
- Docker: required for OpenFace (`algebr/openface:latest`, run with `--platform linux/amd64` on Apple Silicon)

---

## Deployment (Google Cloud Run)

Public demo runs on Google Cloud Run's Always Free tier. Chosen because HF Spaces removed free Docker Spaces (2026-07) and free micro-VMs elsewhere lack the RAM to run OpenFace.

### Architecture: three-stage self-contained image

Local dev spins up a second container (`docker run algebr/openface ...`) from inside the Python process. Cloud Run does not allow Docker-in-Docker, so the deployed image **bakes OpenFace into the app container itself**.

Naive base of `algebr/openface:latest` fails: it's Ubuntu 14.04 (glibc 2.19) whose apt archives are broken and whose g++ 4.8 can't build any 2026-era scientific Python wheel (all require glibc ≥ 2.28 / C++17). So the runtime base is `python:3.12-slim` (Debian 12, glibc 2.36) with OpenFace bundled in as a self-contained payload from a builder stage.

**Stage 1 — `tools` (debian:bookworm-slim):** Downloads John Van Sickle's statically-linked ffmpeg tarball. Static ffmpeg has no shared-lib dependencies, works anywhere.

**Stage 2 — `openface-bundle` (algebr/openface:latest):** Used **only** to extract files, never as runtime. Copies:
- `/home/openface-build/build/` — the whole OpenFace tree (binary + models + classifiers + AU predictors)
- Every non-glibc-family `.so` linked by `FeatureExtraction` (via `ldd`) — this is OpenCV 2.4, Boost 1.54, dlib, TBB, libbsd etc. Excludes libc/libpthread/libm/libdl/librt/libgcc_s/libstdc++/libresolv/libnss_\*/libutil/libanl/libcrypt/libthread_db/libmvec/libnsl/libBrokenLocale — all glibc family. Bundling those causes `symbol lookup error ... undefined symbol: h_errno, version GLIBC_PRIVATE` because `GLIBC_PRIVATE` symbols are strictly versioned across glibc releases; Debian 12's native stubs (backward-compatible for all public symbols) work fine.

**Stage 3 — runtime (python:3.12-slim):**
- `apt-install patchelf` — one binary needed for RPATH rewrites
- COPY the Stage 1 ffmpeg + Stage 2 OpenFace bundle
- `patchelf --set-rpath /openface-libs` on the binary AND every bundled `.so` — the RPATH is baked into the ELF header so only these files use the Ubuntu 14.04 libs. System-wide linker path is untouched.
- Build-time `ldd` sanity check: `RUN` fails loudly if any dep is `not found`
- COPY `uv` (Astral) from `ghcr.io/astral-sh/uv:0.5` and `uv venv --python 3.12` — auto-downloads a python-build-standalone Python 3.12 (targets glibc 2.17, compatible with Debian 12's 2.36)
- `uv pip install -r requirements-deploy.txt` — no version pins needed, all modern wheels install cleanly
- COPY app code

### Dual-mode facial extraction

`src/vision/facial_features.py::_find_local_openface_binary` detects mode at runtime:
- **Direct mode** (Cloud Run): reads `OPENFACE_BIN` env var (set by Dockerfile), calls the binary — no `docker run` layer
- **Docker mode** (local Mac dev): env var absent, falls back to `docker run algebr/openface`, unchanged

This keeps local Mac dev working without any changes.

### Environment / port handling

- `demo/app.py` reads `GRADIO_SHARE` env var — Dockerfile sets `false` (Cloud Run already provides HTTPS; the Gradio share tunnel would hang on Cloud Run's egress)
- Port is read from `$PORT` (Cloud Run injects it, default 8080) via `sh -c 'GRADIO_SERVER_PORT=${PORT:-8080} python -m demo.app'`

### Runtime files

- `Dockerfile` — three-stage build described above
- `requirements-deploy.txt` — pruned runtime deps (drops mlx-whisper, py-feat, mediapipe, imbalanced-learn, openpyxl, tqdm)
- `.gcloudignore` — excludes datasets, notebooks, docs, tests, `.venv/`. **Keeps** `eval/models/` (trained classifiers) and `data/samples/hc_demo/` (UI example widgets)

### Deploy commands

```bash
# One-time setup
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com
echo -n "sk-ant-..." | gcloud secrets create anthropic-key --data-file=-
# Grant Cloud Run runtime SA access to the secret
PROJECT_NUMBER=$(gcloud projects describe parkscreen --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding anthropic-key \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

# Every deploy (Cloud Build builds from --source . on Google's side)
gcloud run deploy parkscreen \
    --source . \
    --region us-central1 \
    --memory 4Gi --cpu 2 --cpu-boost \
    --timeout 300 --max-instances 3 \
    --allow-unauthenticated \
    --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest
```

`git push` to GitHub does **not** trigger a deploy — no CI/CD wired. Deploys are manual via the command above.

First build ~10–15 min (Stage 2's `apt-get` on Ubuntu 14.04's `old-releases` mirrors is the bottleneck; can be flaky). Subsequent builds reuse layer cache from prior successful builds — code-only edits typically rebuild in 3–5 min.

`--cpu-boost` triggers Cloud Run's "startup CPU boost" — the container gets 4× normal CPU during boot, cutting cold start from ~5s to ~2–3s. Free (boost time doesn't count toward monthly quota).

### No keepalive workflow — by design

Unlike HF Spaces (binary sleep/awake with 48h threshold), Cloud Run charges per GB-second of container instance time. Frequent keepalive pings would keep the container permanently warm and burn through the 360k GB-second/month free tier. Instead: accept a 10–15s cold start on first request after idle (Cloud Run stays warm ~15 min after last request). For a scheduled demo, manually `curl` the URL once beforehand to pre-warm:

```bash
curl -s $(gcloud run services describe parkscreen --region us-central1 --format='value(status.url)') > /dev/null
```
