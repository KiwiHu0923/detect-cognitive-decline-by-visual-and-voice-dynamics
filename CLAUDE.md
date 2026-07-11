# ParkScreen — Built with Claude: Life Sciences Hackathon

## Working Protocol

Before every non-trivial code change:
1. **Brief the user** — explain each function/block in plain language (algorithm, non-obvious decisions)
2. **Wait for "go"** before touching any files
3. **After implementation** — mark TODO.md items `[x]` and update CLAUDE.md if architecture/schemas changed

## Project Overview

Multimodal screening tool that detects motor-speech and facial signs of Parkinson's disease (PD) from a single uploaded video. Fuses **phonation** signals (sustained-vowel voice quality) and **articulation** signals (DDK syllable rate/regularity) with **facial dynamics** (hypomimia), routed through a Claude integration layer that generates a clinical-style report.

**Positioning:** a supporting-evidence / screening decision-aid, explicitly **not a diagnosis**.

**Hackathon:** Built with Claude: Life Sciences (Anthropic × Gladstone × Cerebral Valley), July 7–13 2026. Development Track.

---

## Changelog: CogniScreen (dementia) → ParkScreen (PD)

This project was pivoted from a dementia/cognitive-decline design. The changes are scientific, not cosmetic:

| Area | Was (dementia) | Now (PD) | Why |
|------|----------------|----------|-----|
| Primary dataset | ADReSSo / DementiaBank (membership approval, faculty-sponsor blocker) | NeuroVoz (Spanish) — Zenodo DUA cleared 2026-07-08, open access | Removes the access blocker; 53 PD + 51 age-matched HC (analysis cohort ≈ 49 × 46 after PATAKA-intersection), ~2× IPVS size, richer clinical labels (UPDRS, H-Y stage, medication ON/OFF, GRBAS ratings) |
| Fused modalities | acoustic × linguistic (TTR, fillers, RoBERTa — semantic/lexical) | phonation × articulation (DDK), both motor-speech | PD is a motor speech disorder, not a language disorder; semantic features are off-target |
| Core acoustic tool | librosa (preferred) | praat-parselmouth (preferred) | jitter/shimmer/HNR are Praat's native, reproducible, clinically interpretable indices |
| Phonation task | Cookie Theft spontaneous description | sustained vowels | jitter/shimmer/HNR are only valid on quasi-periodic, stable-pitch signals |
| Cross-corpus data | pooled backup corpora | single clean corpus (NeuroVoz); no external corpora merged into training | mixing tasks/corpora at small N → dataset-bias shortcut, inflated AUC |

### Sub-changelog: IPVS → NeuroVoz (2026-07-08)

Initial PD pivot targeted IPVS (Italian). NeuroVoz access cleared mid-week, and the dataset is a strict upgrade:

| Dimension | IPVS (was) | NeuroVoz (now) |
|-----------|------------|----------------|
| Sample size (usable) | 28 PD vs 22 elderly HC (50) | 53 PD vs 51 age-matched HC (≈ 49 × 46 after PATAKA-intersection) |
| Vowel reps | 2 per vowel (VA1/VA2 etc.) | up to 3 per vowel (A1/A2/A3 etc.); A3/E3/I3/O3/U3 are sparse (only a handful of subjects), A1/A2/I1/O2 near-complete |
| DDK task | separate /pa/ (D1) and /ta/ (D2) | **PATAKA** (/pa-ta-ka/) — canonical clinical DDK task; present for 49 PD + 46 HC after age filter |
| Reading | one phonetically-balanced Italian passage | 20+ Spanish word tasks + FREE (spontaneous) |
| Clinical severity labels | none in released files | UPDRS, Hoehn-Yahr stage, disease duration, medication ON/OFF |
| Perceptual labels | none | per-file GRBAS (Grade/Roughness/Breathiness/Asthenia/Strain) |
| Age matching | enforced by data (young HCs have no vowel/DDK) | active filter: HC age ≥ 50 drops only 2 subjects |

Language changes Italian → Spanish; vowels + DDK are language-neutral, so the demo remains cross-language compatible.

---

## Two-Layer Architecture (Core Design Decision)

### Layer 1 — Scientific Validation Spine (offline, on NeuroVoz)

**Score-level (late) fusion** of two speech channels from the same subjects but different tasks. Purpose: produce a rigorous ablation table showing fusion > single-channel. This answers the hardest evaluator question: "does fusion actually add signal?"

- **Phonation channel** — sustained vowels (3 reps per vowel, /a/e/i/o/u/) → jitter, shimmer, HNR, F0 stats (Parselmouth). The core PD phonatory markers.
- **Articulation channel (DDK)** — rapid /pa-ta-ka/ (PATAKA task) → DDK rate + timing/amplitude regularity. A strong, PD-specific motor marker (articulatory bradykinesia/irregularity), extracted from the intensity envelope — **no ASR needed**, and **language-neutral** (good for the demo).
- Tasks are **NOT concatenated** into one feature vector. Each channel trains its own small classifier; the probabilities are fused at score level. This respects the fact that voice-quality and articulatory-timing features live on different scales and come from different tasks.
- **Models trained:** phonation-only, DDK-only, phonation+DDK late fusion. Literature supports this: combining sustained phonation with a second task outperforms any single task.
- **Evaluation:** subject-level LOSO (Leave-One-Subject-Out); metrics: AUC, F1, sensitivity, specificity.
- Results saved to `eval/results/ablation_table.csv`.

**Model class:** small + interpretable (logistic regression / linear SVM), **NOT deep learning**. N is a few dozen subjects — this is exactly the regime PD-voice literature operates in. A small LogReg also gives (a) a LOSO AUC — the single most persuasive number for evaluators, and (b) a clean 0–1 probability to feed the fusion layer and the Claude report.

### Layer 2 — Product Layer (real-time, on user-uploaded video)

User uploads a single video (**task-matched** — see Demo Protocol) → pipeline extracts phonation features (from the vowel segment) + articulation/DDK features (from the /pa/–/ta/ segment) + facial hypomimia features → all passed as structured context into a Claude prompt → Claude generates a clinical-style report with confidence flags.

At inference time the pipeline extracts features from the uploaded input, and scores them with the **already-fitted Layer-1 classifiers**. No training happens at demo time.

**Fusion mechanism: weighted voting (late fusion)**
- **DDK/articulation** score weighted highest (Day 3 empirical result — see below). *Prior CogniScreen-era assumption "phonation is the most direct evidence" did not survive contact with NeuroVoz LOSO.*
- **Phonation** score weighted next.
- Weights follow the **AUC-excess-over-chance heuristic**: `w_c ∝ max(0, AUC_c − 0.5)`, derived from the Day 3 subject-level LOSO table. Current speech weights: phonation 0.35, DDK 0.65 (see `configs/model.yaml`).
- **Facial** score from a smile-task classifier trained on the ROC-HCI UFNet released feature dataset (1361 subjects). Feature extraction uses **OpenFace 2.0** (Docker `algebr/openface`) — the same extractor UFNet trained on, so the AU domain gap is zero. AU-only feature set (7 AUs × mean+var = 14 features, active-frame-only aggregation per Islam et al. 2023); MediaPipe geometric signals dropped because landmark indices are unpublished. A hypomimia JSON summary (AU12 amplitude, expression variance, blink rate, head-movement std) is also passed to Claude as narrative colour, **computed directly from the OpenFace CSV the classifier already produced** — one Docker run per demo, no second facial pipeline (py-feat dropped 2026-07-09; OpenFace already covers AU_r/AU_c + head pose, and its extra emotion/gaze outputs are not core to hypomimia framing).
- If channels agree → higher confidence; if they disagree → report flags for clinical review (never silently overridden)

---

## Data

### Primary Dataset: NeuroVoz (Spanish)

- **Source:** Zenodo (DUA cleared 2026-07-08). Spanish PD voice corpus, ~2900 audio files across 53 PD + 55 HC subjects. Verify citation terms before use.
- **Three task families** (present for most, but not all, subjects — coverage matters, see Cohort below):
  1. Sustained vowels /a/e/i/o/u/, up to 3 reps each (files named `A1..A3, E1..E3, I1..I3, O1..O3, U1..U3`) → phonation channel. Only the "1"/"2" reps are near-universal; A3/E3/I3/O3/U3 exist for only a handful of subjects and should be treated as bonus, not required inputs.
  2. **PATAKA** — rapid /pa-ta-ka/ repetition → articulation channel. This is the canonical clinical DDK task (more standard than IPVS's separate /pa/ + /ta/ files). Present for 46 HC + 49 PD subjects after age filtering — not universal, so subjects without PATAKA are excluded from the analysis cohort.
  3. 20+ Spanish word/sentence tasks + **FREE** (spontaneous speech) — present in the dataset but **not used** by this project (prosody channel dropped 2026-07-09 due to Spanish→English cross-language transfer concerns; see Scope Rules).
- **Language note:** vowels and PATAKA are language-neutral, so a non-Spanish self-recorded demo aligns cleanly with training.
- **Size:** 53 PD (ages 41–88) + 55 HC (ages 31–86). After strict age-matching (HC age ≥ 50, dropping 4 HC subjects — 2 with missing age and 2 aged 31/38) → **53 PD vs 51 HC**.
- **Analysis cohort (used for LOSO):** age-matched subjects who additionally have BOTH PATAKA AND ≥ 1 vowel file → **≈ 49 PD vs 46 HC**. Subjects failing this filter cannot participate in the DDK channel or the fusion model. Feature extraction per subject averages whatever vowels ARE present (does not require all vowels).
- **Bonus clinical labels** (used by the Claude report layer, and optionally as auxiliary supervised targets):
  - PD subjects: UPDRS scale, Hoehn-Yahr stage, disease duration (years), medication ON/OFF status, hypophonic-voice / tremor / dysphagia flags
  - Both groups: per-file **GRBAS** perceptual ratings (Grade, Roughness, Breathiness, Asthenia, Strain) at `data/raw/neurovoz/data/grbas/<task>.csv`
  - Both groups: pre-extracted eGeMAPS-style features at `data/raw/neurovoz/data/audio_features/audio_features.csv` (useful as sanity check vs our Parselmouth outputs; NOT used as training features)
  - Both groups: text transcriptions at `data/raw/neurovoz/data/transcriptions/` (skip Whisper on the word/reading tasks — ground truth is provided)
- **Access:** downloaded, DUA satisfied.
- Path: `data/raw/neurovoz/data/`
  - Audios: `data/raw/neurovoz/data/audios/` (flat directory; filename encodes group_task_subject: e.g. `PD_PATAKA_0004.wav`)
  - Metadata: `data/raw/neurovoz/data/metadata/metadata_pd.csv`, `metadata_hc.csv`

### Training-distribution bias: PD subjects were recorded ON medication

The NeuroVoz paper (Nature Sci Data 2024, §Data records) states: *"Participants with PD were recorded in ON state, having taken the prescribed medication — when applicable — 2 to 5 hours before the recording session."* This is a load-bearing caveat for every downstream statement of "PD" from our system:

- Our per-channel classifiers learn "medicated PD vs age-matched HC" — **not** "any PD vs HC".
- A hypothetical unmedicated (OFF-state) PD subject would present more perturbed features than the training distribution → the model would still tend to flag them, but the model's calibrated probability is not valid for OFF-state input.
- The Claude report layer MUST surface this in every generated report — via a mandatory disclaimer sentence and, when the score suggests elevated risk, an explicit note that OFF-state features can look more severe than ON-state training data.
- Scientific implication for the ablation table: our AUC / sensitivity numbers apply to the ON-state cohort. Do not extrapolate to OFF-state PD from these results.

### Sanity check on NeuroVoz's shipped `audio_features.csv` (2026-07-08)

Cross-checked our Parselmouth features against NeuroVoz's shipped CSV on 466 vowel rows. Findings:
- `shimmer_local` ↔ `rShimmer`: **r = 0.824** (compatible; theirs is percent, ours is ratio → ~100× scale)
- Three jitter variants ↔ `rJitter/RAP/rPPQ`: **r = 0.46–0.50** (same feature family per paper Table 5; residual ~3× magnitude gap after unit conversion attributed to AVCA-ByO's Praat parameters ≠ our `To Pitch (cc)` defaults — this is acceptable at hackathon quality)
- `hnr_mean_db` ↔ NeuroVoz `HNR`: **r = −0.18** — NeuroVoz's shipped `HNR` column is NOT standard Praat HNR (their mean −10.8 vs ours +23.1; the paper labels it "dB" but does not specify sign convention; AVCA-ByO's algorithm undisclosed at repo level). **Do not use their HNR column for anything downstream.** Ours is the canonical Praat `harmonicity_cc` output and is correct.

Practical rule: for our extracted jitter/shimmer, the values live in **ratio units**. Any downstream text/report that quotes clinical thresholds must first convert `× 100` to percent (the clinical convention) before comparison to published normal ranges.

### Under review — bonus only, NOT on the critical path

- **ParkCeleb** — access request still pending.
- If it clears in time, use it only as a **second phonation corpus for a cross-lingual robustness check** — never merged into NeuroVoz training (different language/device/task → dataset bias at small N).
- Treat as pure upside. The project ships complete on NeuroVoz alone; approval progress must not gate any deliverable.
- Path (if it arrives): `data/raw/external/`

### Demo Data

- Single self-recorded or volunteer video, **task-matched to training** (see Demo Protocol).
- Path: `data/samples/`

---

## Demo Protocol (task-matching is mandatory)

The demo tasks must match what the classifiers were trained on, or the extracted features are out-of-distribution and the score is meaningless. **Every corpus we use — NeuroVoz, IPVS, UFNet/PARK@Home — stores tasks as separate files**, so ParkScreen collects three separate uploads at demo time and skips runtime segmentation entirely. Task alignment is guaranteed by construction; no heuristic split, no ASR-driven boundary detection (decision date 2026-07-09; `src/audio/segment.py` was never written and is out of scope).

Three uploads (any of `.wav`, `.m4a`, `.mp4`, `.mov`), each a **directory** of task-matched clips:

1. **Sustained vowels /i/, /o/, /u/** — up to 3 reps per vowel, ~3–5 s each (steady pitch, comfortable loudness). Audio-only is fine → phonation channel. Filenames must follow `<group>_<VOWEL><REP>.<ext>` (e.g. `PD_I1.m4a`, `HC_O2.wav`) so the pipeline can group reps and apply the paper AUC-excess vowel weights (Day 4 finding #10 — [a] and [e] are dropped as barely above chance per Li et al. 2606.19125 Table IV; the pipeline filters silently). At least one of {I, O, U} must be present; weights renormalize over what's actually recorded. Language-neutral.
2. **Rapid /pa-ta-ka/ repetition** — 1+ reps of ~5 s each, as fast and steady as possible. Audio-only is fine → articulation/DDK channel (mirrors the PATAKA task in NeuroVoz). Features are averaged across reps. Language-neutral.
3. **8–12 seconds of smile ×3 alternating with a neutral face** (each smile phase ~2–3s + neutral ~1–2s between), face clearly visible, per Islam 2023's protocol — must be video → smile classifier + hypomimia narrative (AU12 amplitude, expression variance, blink rate, head-movement std). Multiple clips allowed; the pipeline max-pools the classifier score over clips and takes the hypomimia summary from the selected clip.

Any channel whose upload dir is missing / empty → the pipeline returns `N/A` on that channel and the Claude report is generated with the remaining channels (per `llm_fusion.py` N/A handling and the frozen system prompt in `claude_client.py`).

**Data-leakage guard:** if a held-out NeuroVoz PD sample is used as a second demo case, it MUST be the subject left out in its LOSO fold — never a subject the classifier was fitted on. The self-recorded healthy sample is out-of-training by construction.

**Primary demo pair (cleanest):** self-recorded healthy video (task-matched) + one held-out NeuroVoz PD sample (real label, task-matched, no privacy issue).

**Optional in-the-wild sample** (e.g. a YouTube PD clip) — bonus only, with guards:
- **Task mismatch:** most such clips are interviews/spontaneous speech, not sustained vowels or DDK. Do not feed connected speech to the phonation or DDK classifier — the pipeline should return those channels as **N/A** and score only the facial channel (task-compatible). This also showcases the system's modularity.
- **Signal quality:** compression, background music, multiple speakers inflate jitter/shimmer artefactually — pick single-speaker, quiet, close-mic segments and note quality in the report.
- **Label & ethics:** the condition is self-reported/inferred, not a clinical label → illustrative only, **never counted toward AUC**. Mark clearly as "public video, self-reported condition, illustrative." Prefer the NeuroVoz held-out sample as the labelled PD case; use the in-the-wild clip only as a secondary "runs on real videos" demonstration.

---

## Tech Stack

| Component | Library | Notes |
|-----------|---------|-------|
| Audio extraction | ffmpeg via subprocess | 16kHz mono WAV |
| ~~ASR / segmentation~~ | ~~`mlx_whisper`~~ | **Dropped from critical path 2026-07-09** — demo uses three separate task uploads (see Demo Protocol), no runtime segmentation. `src/audio/transcribe.py` stays in-tree for future use / debug but nothing on the demo path calls it. |
| Phonation features | `praat-parselmouth` | jitter, shimmer, HNR, F0 — the interpretable core |
| Articulation / DDK features | `praat-parselmouth` + `scipy.signal` | intensity-envelope peak picking on the PATAKA task → DDK rate + timing/amplitude regularity (no ASR) |
| (Optional) extra acoustic | `opensmile` (eGeMAPS, 88 feats) | prediction booster if time permits; includes jitter/shimmer/HNR too |
| Facial AU extraction (classifier + narrative) | OpenFace 2.0 via Docker `algebr/openface` | matches UFNet training extractor; 7 AUs × mean+var = 14 features, active-frame-only. Runs with `-aus -pose` flags so the same CSV feeds both `predict_smile_pd` (AU columns) and `summarize.py` (AU + `pose_Tx/Ty/Tz` for `head_movement_std`) — one Docker run per demo clip (Day 5 fix; earlier `-aus`-only calls dropped pose columns, forcing `head_movement_std=None`). |
| ~~Facial narrative (separate pipeline)~~ | ~~`py-feat`~~ | **Dropped 2026-07-09** — OpenFace already outputs per-frame AU_r/AU_c + head-pose columns, so the hypomimia JSON is derived from the same CSV as the classifier (zero AU domain gap). Emotion/gaze dropped as non-core to hypomimia framing. |
| Statistical fusion | `scikit-learn` | LogReg / linear SVM per channel + score-level fusion |
| LLM / report | `anthropic` SDK | Claude report layer |
| UI | `gradio` | fastest for hackathon demo |

---

## Repository Structure

```
parkscreen/
├── data/                          # gitignored
│   ├── raw/
│   │   ├── neurovoz/data/          # NeuroVoz: audios/ metadata/ transcriptions/ grbas/ audio_features/
│   │   └── external/               # (bonus) ParkCeleb if approved
│   ├── processed/
│   │   ├── transcripts/            # Whisper .json output (task segmentation)
│   │   ├── phonation_features/     # .npy arrays (jitter/shimmer/HNR/F0)
│   │   ├── ddk_features/           # .npy arrays (DDK rate + regularity)
│   │   └── facial_features/        # .json AU timeseries
│   └── samples/                    # demo video
│
├── src/
│   ├── audio/
│   │   ├── transcribe.py           # Whisper wrapper — NOT on the demo critical path (kept for future / debug)
│   │   ├── phonation.py            # Parselmouth jitter/shimmer/HNR/F0 stats on vowels
│   │   └── ddk.py                  # intensity-envelope peak picking → DDK rate + regularity
│   │   # segment.py NOT in scope — demo uses three separate uploads (see Demo Protocol)
│   ├── vision/
│   │   ├── train_smile_pd.py       # trains 14-feature smile classifier on UFNet CSV (active-frame-only, AU only)
│   │   ├── facial_features.py      # OpenFace Docker → per-frame AU_r + AU_c for 7 AUs (video → arrays + detection meta)
│   │   ├── aggregate.py            # per-frame AU_r/AU_c → 14-dim session vector via active-frame mean+var
│   │   ├── predict_smile_pd.py     # video → PD score + detection-rate gate
│   │   └── summarize.py            # OpenFace CSV → hypomimia JSON (AU12 amplitude, expression variance, blink rate, head-movement std) for Claude narrative — reuses the CSV facial_features.py already produced
│   ├── fusion/
│   │   ├── statistical.py          # Layer 1: per-channel LogReg + score-level late fusion
│   │   └── llm_fusion.py           # Layer 2: builds Claude context string (weighted voting)
│   ├── report/
│   │   ├── claude_client.py        # Anthropic SDK calls + prompt templates
│   │   └── formatter.py            # Claude output → display-ready report
│   └── pipeline.py                 # Single entry point: video path → report
│
├── eval/
│   ├── ablation.py                 # trains 3 models (phonation, DDK, fusion), subject-level LOSO
│   ├── metrics.py                  # AUC, F1, sensitivity, specificity
│   └── results/
│       ├── ablation_table.csv      # THE table shown to evaluators
│       └── figures/                # ROC curves, feature importance
│
├── demo/
│   ├── app.py                      # Gradio UI (pipeline.py → report display)
│   └── assets/
│       └── example_report.html     # pre-rendered demo output
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_ablation_results.ipynb
│
├── configs/
│   ├── model.yaml                  # hyperparams, which channels to include, fusion weights
│   └── paths.yaml                  # data paths
│
├── tests/
│   └── test_pipeline.py            # smoke test: mock audio → report runs
│
├── .env.example
├── requirements.txt
├── TODO.md
└── CLAUDE.md
```

---

## Existing Portfolio Code (Reusable)

Source: `KiwiHu0923/Multimodal-Video-Content-Segmentation-System` → `backend/ollama_audio.py`

### Directly reusable functions

**`extract_audio_sync(video_path, audio_path)`**
Uses ffmpeg subprocess to extract 16kHz mono WAV from video. Already copied into `src/audio/transcribe.py`.

**`load_audio_to_memory(wav_filename)`**
Loads WAV as float32 numpy array. Already copied.

**`run_whisper_extraction(video_path, audio_path)`**
Uses `mlx_whisper.transcribe(..., word_timestamps=True)`. Still needed for task-region segmentation on the demo video.

**`classify_non_speech_fast(audio_array, framerate, start_time, end_time)`**
RMS energy silence detection. Repurpose in `src/audio/segment.py` and `src/audio/ddk.py` (intensity envelope peak picking).

### What to replace
- `ollama` → `anthropic` SDK
- `MODEL_NAME / ollama.chat()` → `claude_client.py` with Anthropic SDK calls
- Content segmentation (intro/ads/outro/content) → **task segmentation dropped 2026-07-09** — demo uses three separate task uploads. PD feature extraction (phonation, DDK, facial) still consumes the audio-loading helpers.

---

## Phonation Features (sustained vowels, Parselmouth)

Computed on the vowel segment only. Valid only on quasi-periodic, stable-pitch signals.

**Steady-window preprocessing (Day 4, 2026-07-10):** every sustained-vowel WAV is cropped to its middle 60% before Parselmouth analysis (`configs/model.yaml → phonation.apply_steady_window: true`). Onset/offset carry glottal transients and F0 drift that inflate jitter/shimmer/F0-std in both PD and HC files; cropping keeps analysis on the stable middle. Short-clip guard: sounds below `steady_window_min_duration_s` (0.5s) are returned uncropped so the pitch tracker still has enough voiced material. See Day 4 findings below for the empirical justification (no impact on NeuroVoz LOSO because those files are already pre-trimmed; decisive impact on raw demo audio where it rescued the PD demo from misclassification).

**Vowel filter + weighted aggregation (Day 4, 2026-07-10, replaces Day-2 pipeline):** phonation training now uses only `[I, O, U]` per Li et al. arxiv 2606.19125 Table IV — the paper's Person AUC on NeuroVoz shows [a]=0.58 and [e]=0.63 (barely above chance) versus [i]=0.77, [o]=0.77, [u]=0.83. Aggregation is two-stage:
1. **Per-subject × per-vowel:** average across reps of the same vowel (`I1/I2/I3` → one /i/ vector per subject). Fixes a Day-2 bug where `BALANCED_VOWEL_SET = (A1, A2, I1, O2, U1)` treated A1 and A2 (two reps of the same /a/) as separate tasks and double-counted /a/.
2. **Cross-vowel weighted average:** using AUC-excess-over-chance weights from paper Table IV, renormalized after dropping [a]/[e]: `I=0.310, O=0.310, U=0.379`. Weights renormalize per-subject over the vowels actually present, so a subject missing /u/ still gets scored on {I, O} with weights 0.5 / 0.5.

Cohort intact: 49 PD + 46 HC after this pipeline. Coverage: [I] 49/46 (mean 2.12 reps/subject), [O] 49/46 (mean 2.11), [U] **45/46** (mean 1.26 — 4 PD subjects have no /u/ file and are scored on IO only via weight renormalization). Empirical impact: LOSO deployment-fusion row +0.006 (0.740 → 0.746, within CI); demo PD phonation +0.083 (0.530 → 0.613), demo PD fusion +0.025 (0.679 → 0.704). See Day 4 findings #10 below.

| Feature | Method |
|---------|--------|
| Jitter (local, ppq5, rap) | Parselmouth PointProcess |
| Shimmer (local, apq5, apq11, dda) | Parselmouth PointProcess |
| HNR (Harmonics-to-Noise Ratio, dB) | Parselmouth `harmonicity_cc` |
| F0 mean / std / range | Parselmouth pitch (crosscorrelation or autocorrelation) |
| Voicing fraction | fraction of frames with detected F0 |

---

## Articulation / DDK Features (PATAKA, no ASR)

Computed on the PATAKA (/pa-ta-ka/) segment via intensity-envelope peak picking. Language-neutral.

| Feature | Method |
|---------|--------|
| DDK rate (syllables/sec) | intensity peaks / duration |
| Inter-syllable-interval mean | mean of peak-to-peak times |
| Inter-syllable-interval CV | std / mean (timing regularity — lower = more regular) |
| Peak amplitude mean / CV | amplitude regularity |
| Amplitude decrement (linear slope) | early vs late peak amplitudes (fatigue / decrescendo) |

---

## Facial Features — Smile Classifier + Hypomimia Summary

The facial channel has two outputs, both fed to the Claude report:

### 1. Smile classifier score (quantitative)

Trained on the ROC-HCI UFNet released feature dataset (1361 subjects, MIT-licensed) via `src/vision/train_smile_pd.py`. UFNet's ShallowANN is mathematically a logistic regression over the session-level feature vector, so we use sklearn `LogisticRegression` + `StandardScaler` + SMOTE — same model class, no torch dependency at inference.

**Feature set: 14 = 7 AUs × {mean, var}, active-frame-only aggregation.**

The active-frame-only rule comes from Islam et al. 2023 ("Unmasking Parkinson's Disease with Smile", arxiv 2308.02588 — the precursor to UFNet), which specifies that mean/variance are computed **only on frames where the AU's binary presence flag (`AU_c == 1`) is set**, not over all frames. Physical meaning: "how intense the AU is *when it fires*" rather than "average intensity across the whole clip". Full-frame aggregation systematically dilutes toward 0 and mimics the py-feat scale-gap failure mode.

We dropped the 7 MediaPipe geometric signals (eye-open, mouth-width, jaw-open, etc.) because Islam 2023 does not publish the landmark indices used to compute them, so we cannot reproduce their extraction faithfully. Ablation cost: -0.026 test AUROC (0.839 → 0.812). We also dropped the entropy statistic across all signals because Islam 2023 does not publish the histogram binning, log base, or range — non-reproducible. Ablation cost: -0.028 test AUROC (0.837 → 0.812).

**Reference AUROC:** paper's smile-only 0.830 (Islam 2023, 10-fold CV, SVM ensemble); our test AUROC on UFNet's held-out participant split = **0.812** (LogReg on 14 active-frame features). The gap is within noise given the model-class simplification and the aggressive feature reduction for reproducibility.

**External validation on YouTubePD (in-the-wild, informational only — do NOT quote as our headline number):** The UFNet paper (AAAI 2025, Table 8) publishes **only the Smile+Speech fusion** number on YouTubePD (AUROC 0.838); it does **not** publish smile-only YouTubePD, so there is no paper baseline to compare to. Our smile-only numbers on YouTubePD, run in `eval/eval_smile_yt_subset.py`:
- Full released CSV (251 clips, 58 PD / 193 HC): **AUROC 0.602**
- UFNet's designated `splits/test_yt_pd.txt` subset (178 of the 184 listed IDs match the CSV; 21 PD / 157 HC): **AUROC 0.708**

The 73 clips in the full CSV but excluded from UFNet's subset are 51% PD (37/73) vs 12% PD in the retained subset — the excluded pool is not a random draw, so the AUROC gap (+0.106) is not a prevalence artefact. Working interpretation: UFNet's team dropped low-quality / off-task clips their own model also misclassified. The 0.708 subset number is the fairer measure of smile-only wild-transfer; the 0.602 full-CSV number is a superset that includes samples UFNet themselves excluded from evaluation. Neither number is demo-representative — the demo is task-matched (Islam 2023 smile ×3 protocol) and should sit near the in-distribution 0.812.

Three gotchas we had to hit precisely to get comparable numbers, all from reading the source paper + UFNet code carefully:
1. **Splits key off the `ID` column, not `Participant_ID`.** UFNet's `unimodal_smile_baal.py:136` uses `IDs = df['ID']`; the split files list `ID` values. Using `Participant_ID` matches only ~43% of test IDs and silently leaks the rest into train.
2. **NaN in the `pd` column is treated as PD=1**, per their `lambda x: 0 if str(x) in ['no','0'] else 1` rule. A `--nan-as drop` mode is exposed for comparison — dropping the 20 NaN rows changes test AUROC by 0.001 in this dataset, since all 20 happen to sit in train.
3. **AU aggregation is active-frame-only** (see above).

7 AUs used: AU01, AU06, **AU12** (lip corner puller — the dominant predictor, Islam 2023 Table 2 rank 1 with coefficient 203.99), AU14, AU25, AU26, AU45 (blink). AU45 could be dropped with negligible cost (< 0.005 AUROC) but is retained for column-alignment with UFNet's CSV schema.

**Task protocol at demo time:** 8–12 seconds total, **smile ×3 alternating with a neutral face** (each smile phase ~2–3s + neutral ~1–2s between), per Islam 2023's exact protocol. Language-neutral.

Saved artifacts:
- `eval/models/smile_pd_lr.joblib` — the classifier (14 features)
- `eval/models/smile_pd_scaler.joblib` — matching StandardScaler
- `eval/models/smile_pd_columns.json` — canonical 14-column order (7 AUs × mean/var)
- `eval/models/smile_pd_metrics.json` — dev/test/YoutubePD-external metrics

Data assets (all gitignored under `data/raw/ufnet_smile/`):
- `facial_dataset.csv` (training features, 1684 rows / 1361 subjects)
- `youtube_PD_features.csv` (external validation, 251 clips, 58 PD)
- `splits/{dev,test,calib}.txt` (participant IDs — reproduces the paper's split); `splits/test_yt_pd.txt` (184 IDs — UFNet's designated YouTubePD evaluation subset, of which 178 match the released CSV)
- `pretrained/` (their fitted `.pth` + scaler kept for reference / possible Path-A fallback)

### 2. Hypomimia summary (qualitative narrative)

Computed by `src/vision/summarize.py` **directly from the OpenFace CSV that the classifier already produced** — one Docker run per demo, no second facial pipeline. py-feat is dropped (2026-07-09): OpenFace covers AU_r/AU_c + head pose columns natively, and its extra emotion/gaze outputs are not core to hypomimia framing.

Output JSON is narrative colour for the Claude report, not a classifier:
```json
{
  "mean_AU12": 0.15,
  "AU12_amplitude_on_smile_cue": 0.28,
  "expression_variance": 0.12,
  "blink_rate_per_min": 8.4,
  "head_movement_std": 0.04,
  "detection_rate": 0.93,
  "warnings": []
}
```

Field definitions (see `src/vision/summarize.py` docstring for full rationale):
- `AU12_amplitude_on_smile_cue` = **max** of AU12_r on active-smile frames (`AU12_c == 1`). Peak, not mean — matches the clinical concept of "smile amplitude" and stays non-redundant with the classifier's active-frame mean feature.
- `expression_variance` = mean of per-AU temporal std across kept frames. Higher = more expressive face; low values consistent with mask-like (hypomimic) presentation.
- `blink_rate_per_min` = AU45_c 0→1 rising edges divided by kept-duration in minutes.
- `head_movement_std` = mean of std(`pose_Tx`), std(`pose_Ty`), std(`pose_Tz`) on kept frames.
- `hypomimia_score` **intentionally omitted** (design review 2026-07-10): a composite requires an arbitrary normalization anchor and duplicates work Claude does when synthesising the report. Raw markers are more defensible and give Claude flexibility.
- Insufficient-detection short-circuit: `< 10` kept frames → all fields None with reason in `warnings`. Low-detection warning: `detection_rate < 0.80` → surfaced in `warnings`.

---

## Claude Integration (Layer 2)

Fully built and end-to-end validated 2026-07-10 (Claude layer) and 2026-07-11 (full pipeline). Four files with clear separation of concerns:

- `src/vision/summarize.py` — hypomimia narrative JSON (companion to the smile classifier score; reuses the OpenFace CSV, no second Docker run). See Facial Features → "Hypomimia summary" above for the schema.
- `src/fusion/llm_fusion.py` — canonical `fuse_scores` (also imported by `quick_score.py` and `pipeline.py`, so late fusion lives in one place) + `build_claude_context` (composes the XML block for Claude; ratio→percent unit conversion isolated here).
- `src/report/claude_client.py` — Anthropic SDK call to `claude-opus-4-7`. Non-streaming, `max_tokens=2048`, `thinking={"type": "disabled"}`, no sampling params (Opus 4.7 400s on `temperature`/`top_p`/`top_k`). Frozen ~1143-token `SYSTEM_PROMPT` carries the clinical rules; `cache_control: {"type": "ephemeral"}` marker is set on the system block but a no-op today (Opus 4.7's cacheable-prefix floor is 4096 tokens — the marker self-activates if the prompt grows past 4K).
- `src/pipeline.py` — end-to-end orchestrator. `run_pipeline(vowel_dir, pataka_dir, smile_dir, call_claude=True) -> dict` collects three task-matched upload dirs, scores each channel using the same helpers as `quick_score.py` (`_score_phonation`, `_score_ddk`) for the audio channels and a local `_score_facial` that uses `predict_and_summarize` for the facial channel (single OpenFace Docker run per clip → classifier vector + hypomimia summary from the same CSV, preserving the "one OpenFace run per demo" invariant). Fuses via `fuse_scores`, composes context via `build_claude_context`, calls `generate_report`. CLI mirrors `quick_score`: `python -m src.pipeline --vowel-dir ... --pataka-dir ... --smile-dir ... [--no-claude] [--out-dir ...] [--label PD|HC]`. Gradio (Day 5 UI) is a thin shim over `run_pipeline`.

The system prompt enforces every non-negotiable at generation time:
- Report structure (Risk Level → Phonation → DDK → Facial → Cross-Channel Consistency → Recommended Next Steps → Disclaimers)
- Omitted channels are skipped narratively and called out in Consistency
- `any_flag_for_review=true` → explicit "Per-channel disagreement — flagged for clinical review", never silently reconciled
- Both mandatory disclaimers appear verbatim (screening-decision-aid + ON-medication caveat)
- Feature units labelled on Claude's side (jitter/shimmer percent, HNR dB, DDK syl/sec, F0 Hz) with clinical anchor ranges so hedged language has a reference point
- Hedged-tone whitelist, diagnostic-language blacklist, 2–4 sentences per channel section

Runtime flow for one report:
1. `pipeline.py` (Day 5) collects per-channel scores + features + `summarize_facial_features(...)` output
2. `fuse_scores(scores, weights, threshold)` → `{fused_score, weights_normalized, agreement}`
3. `build_claude_context(channels, facial_summary, fusion_result)` → XML block
4. `generate_report(context_str)` → markdown report

Canonical example artifacts at `demo/assets/example_context.xml` (Claude input) and `demo/assets/example_report.md` (Claude output) — useful for demo screenshots and regression-checking future prompt changes.

---

## Evaluation Protocol

### Ablation table (the deliverable for evaluators) — Day 3 results

Subject-level LOSO on NeuroVoz analysis cohort (49 PD × 46 HC = 95 subjects). LogReg with l2, C=1.0, StandardScaler-in-pipeline (re-fit per fold — no scaler leakage). Bootstrap 95% CI on AUC (1000 resamples over subjects).

| Model | AUC [95% CI] | F1 | Sens | Spec |
|-------|--------------|----|------|------|
| Phonation-only (per-subject mean over vowels) | 0.567 [0.455, 0.678] | 0.545 | 0.490 | 0.674 |
| DDK-only (PATAKA) | **0.740** [0.632, 0.845] | 0.701 | 0.694 | 0.696 |
| Phonation + DDK (unweighted avg) | 0.722 [0.617, 0.822] | 0.638 | 0.612 | 0.674 |
| Phonation + DDK (AUC-excess weighted, p=0.35 d=0.65) | 0.736 [0.634, 0.842] | 0.660 | 0.633 | 0.696 |
| Phonation per-file (per-file eval, n=466) | 0.603 [0.553, 0.653] | 0.581 | 0.560 | 0.604 |
| Phonation per-file (per-subject eval) | 0.630 [0.518, 0.735] | 0.571 | 0.531 | 0.652 |
| Phonation(per-file) + DDK (unweighted avg) | 0.756 [0.658, 0.854] | 0.701 | 0.694 | 0.696 |
| **Phonation(per-file) + DDK (AUC-excess weighted)** | **0.758** [0.662, 0.859] | 0.667 | 0.633 | 0.717 |

Persisted to `eval/results/ablation_table.csv` + `eval/results/ablation_summary.json` (config snapshot). Per-subject OOF probabilities at `eval/results/loso_oof_probs.csv`. Per-channel coefficient dump at `eval/results/coefficients.csv`.

Cross-validation: **subject-level LOSO** on NeuroVoz. Raw age-matched cohort is 53 PD vs 51 HC; the analysis cohort further requires each subject to have PATAKA + ≥ 1 vowel → **≈ 49 PD vs 46 HC** (actual: 95 = 49 + 46).

### Day 3 findings — what we learned building the ablation table (2026-07-09)

**1. DDK is the dominant motor-speech channel on NeuroVoz, not phonation.**
DDK-only AUC 0.740 vs phonation-only AUC 0.567 (per-subject baseline). This contradicts the CogniScreen-era prior "phonation is the most direct evidence" and matches the PD literature more accurately: articulatory bradykinesia on a canonical DDK task (PATAKA) is a stronger acoustic marker than sustained-vowel voice quality at this cohort size. The fusion weights were updated to reflect the empirical direction (see below).

**2. Mean-over-vowels aggregation was silently diluting phonation signal.**
The Day 2 decision to average the 5 balanced vowels (A1, A2, I1, O2, U1) into one 12-dim feature vector per subject lost signal. Switching to per-file training (466 vowel rows, `LeaveOneGroupOut` by `subject_id`) lifted phonation-only per-subject AUC from 0.567 → 0.630 (+0.063). The model gains from seeing individual vowel files as independent training data points, even without a vowel-identity feature.

**3. Fusion > best single channel required BOTH per-file phonation training AND AUC-excess-weighted fusion.**
- Baseline (per-subject phonation) + unweighted fusion: 0.722 < 0.740 DDK-only (**fusion loses**)
- Baseline + old fixed weights (p=0.50, d=0.35): 0.705 (**fusion loses harder**)
- Per-file phonation + unweighted: 0.756 > 0.740 (+0.016, CI overlaps)
- **Per-file phonation + AUC-excess weights: 0.758 (+0.018, best row)**

Both changes are needed. Per-file training alone helps but isn't enough; the wrong weight direction actively hurts because it upweights the weaker channel. The Day 3 default (per-file training + AUC-excess weights) is now the deployed configuration.

**4. AUC-excess weighting is the principled way to set fusion weights.**
`w_c ∝ max(0, AUC_c − 0.5)`, normalized over channels present. Concretely for Day 3: phonation excess 0.130, DDK excess 0.240 → phonation 0.35, DDK 0.65. This replaces the CogniScreen-era phonation=0.50/DDK=0.35 (which was the wrong direction). Weights live in `configs/model.yaml` with the derivation comment.

**5. Failed diagnostic (informative): outliers are NOT the phonation bottleneck.**
Hypothesized that right-skewed jitter/shimmer outliers were inflating StandardScaler's std and squashing the healthy distribution flat. Tried `log(x + 1e-6)` on jitter/shimmer columns + RobustScaler → phonation-only AUC went 0.567 → 0.534 (worse). Ruled out this hypothesis and pointed diagnosis at aggregation (finding #2). The log-transform code was reverted to keep the codebase clean; the finding is recorded here.

**6. Unexpected finding: per-file eval AUC (0.603) < per-subject aggregated AUC (0.630).**
Usually per-file evaluation is optimistic because within-subject correlation makes correct/incorrect predictions repeat 5× per subject. Here the opposite: individual vowel-file phonation features are noise-dominated (not signal-dominated), so averaging over a subject's vowel files reduces noise and gives a cleaner per-subject prediction than trusting any single file. Corollary: at demo time, if the user records only one vowel, phonation confidence should be labelled as lower than the LOSO-cohort AUC suggests. Multi-vowel demo recordings are preferable when possible.

**7. Per-subject LOSO is the honest deployment metric; per-file with subject-level LOSO is the literature-comparable metric.**
- **Per-subject LOSO** (95 datapoints, one prediction per patient) answers "will this work on a new patient?" — the actual deployment question. This is what we headline.
- **Per-file with `LeaveOneGroupOut(subject_id)`** (466 datapoints, no leakage) answers "how do we compare to per-file baselines in the literature (e.g. NeuroVoz paper)?" — reported as an auxiliary row, not the headline.
- Both are in the table. Neither has data leakage (both split by subject).

**Caveats that still apply** (unchanged from prior sections, restated for the Day 3 record):
- All PD training subjects recorded ON medication (2–5h post-dose per NeuroVoz paper §Data records). Numbers apply to ON-state PD vs age-matched HC; OFF-state PD would present more perturbed features than training distribution.
- Screening decision-aid, NOT a clinical diagnosis. Reports must include both disclaimers.

### Day 4 findings — steady-window preprocessing (2026-07-10)

**8. NeuroVoz vowel files are silently pre-trimmed; demo audio is not — steady window closes the gap.**

Cropping every vowel file to its middle 60% before Parselmouth analysis (`apply_steady_window: true`) was tested on both the NeuroVoz LOSO cohort and two real self-recorded demo cases (HC and PD, at `data/samples/{hc,pd}_demo/vowel/`, task-matched 3 reps × 5 vowels).

*On NeuroVoz LOSO:* essentially zero movement. Phonation-only per-subject AUC 0.567 → 0.569, per-file 0.603 → 0.601, best fused row 0.758 → 0.758. All deltas |ΔAUC| ≤ 0.004, well inside bootstrap CI widths (~0.12). Six vowel files fell below the 1s voicing floor after cropping (was 466, now 460), but all 95 analysis-cohort subjects retained ≥ 1 vowel — cohort unchanged.

*On demo audio:* decisive.

| | Jitter | Shimmer | HNR | F0 std | **PD prob** | Predicted |
|---|---|---|---|---|---|---|
| HC demo OFF | 0.351% | 3.35% | 23.05 dB | **10.68 Hz** | 0.163 | HC ✓ |
| HC demo ON | 0.284% | 3.19% | 23.46 dB | **5.78 Hz** | 0.466 | HC ✓ (borderline) |
| PD demo OFF | 0.425% | 3.49% | 23.10 dB | **8.93 Hz** | **0.240** | **HC ✗** |
| PD demo ON | 0.368% | 3.17% | 23.48 dB | **4.98 Hz** | **0.530** | **PD ✓** |

The load-bearing feature is F0 std. NeuroVoz training distribution has HC F0_std ≈ 3.3 Hz, PD F0_std ≈ 4.8 Hz. Un-trimmed demo audio sits at 9–11 Hz — completely outside training range, and the classifier is extrapolating garbage. Steady window brings demo F0_std into the 5–6 Hz range where training-time decision boundaries are calibrated. This rescued the PD demo from being misclassified as HC (0.240 → 0.530). The reason NeuroVoz LOSO didn't move is that NeuroVoz files were already pre-trimmed by the AVCA-ByO pipeline before Zenodo release — cropping the already-clean middle of already-clean audio changes nothing.

Practical rules from this:
- `apply_steady_window: true` is mandatory on the demo path. Turning it off produces wrong-distribution features and unreliable predictions.
- The HC demo's post-steady prob (0.466) is close to the decision threshold. This is not a bug: cross-recording-environment drift means demo speakers' F0 variability sits closer to NeuroVoz PD's than NeuroVoz HC's, even after preprocessing. Report layer should communicate confidence, not just a binary label.
- **Training-time hidden preprocessing is a first-class validity concern.** Any future feature-engineering change (e.g. Part 2 vowel-weighted training) must be re-benchmarked on both NeuroVoz LOSO AND demo audio, because the two evaluation surfaces disagree systematically.

Full demo multimodal scores (post steady-window, current deployed weights p=0.35 / d=0.65 / f=0.15, renormalized over channels present):

| Channel | Weight | HC demo | PD demo |
|---|---|---|---|
| Phonation | 0.30 | 0.466 | 0.530 |
| DDK | 0.57 | 0.028 | 0.881 |
| Facial (smile classifier, max-pool over clips) | 0.13 | 0.000 | 0.148 |
| **Fused** | — | **0.158** | **0.679** |
| **Predicted (thr=0.5)** | | **HC ✓** | **PD ✓** |

Both demos classified correctly; DDK carries the fusion (HC 0.028 vs PD 0.881 — 0.85 gap). Phonation channel is separating by only 0.064 on this pair even after steady window — motivated the Day 4 vowel-filter + weighted-aggregation refactor (finding #10).

### Day 4 findings — vowel-filter + paper-weighted phonation aggregation (2026-07-10)

**9. Day-2 bug: `BALANCED_VOWEL_SET = (A1, A2, I1, O2, U1)` double-counted /a/.**
A1 and A2 are reps 1 and 2 of the same /a/ vowel, not separate tasks. Treating them as independent training rows (Day 3 per-file finding #2) let the model see /a/ twice as often as any other vowel. Fixed by aggregating reps within a vowel BEFORE cross-vowel combination; see finding #10 for the new pipeline.

**10. Paper-informed vowel filter + weighted aggregation: LOSO neutral, demo improved.**
Per Li et al. arxiv 2606.19125 Table IV (NeuroVoz Person AUC), [a]=0.58 and [e]=0.63 are barely above chance while [i]=0.77, [o]=0.77, [u]=0.83. New pipeline: drop [a]/[e], average reps within each remaining vowel, then AUC-excess weighted average across vowels (I=0.310, O=0.310, U=0.379).

Effect on NeuroVoz LOSO (49 PD × 46 HC):

| Model | Day 4 baseline (all 5 tasks, mean-over-files) | Day 4 v2 ([i,o,u] weighted) | Δ |
|---|---|---|---|
| Phonation-only (per-subject) | 0.569 | 0.560 | −0.009 |
| DDK-only | 0.740 | 0.740 | 0 |
| Phon + DDK (weighted, deployed) | 0.740 | **0.746** | **+0.006** |
| Phon per-file (per-subject eval) | 0.629 | 0.596 | **−0.033** |
| Phon(per-file) + DDK (weighted) | 0.758 | 0.760 | +0.002 |

Effect on demos (raw un-trimmed audio, task-matched):

| | HC baseline | HC v2 | PD baseline | PD v2 |
|---|---|---|---|---|
| Phonation | 0.466 | 0.464 | 0.530 | **0.613 (+0.083)** |
| Fused | 0.158 | 0.157 | 0.679 | **0.704 (+0.025)** |
| Predicted | HC ✓ | HC ✓ | PD ✓ | PD ✓ |

LOSO barely moves (all deltas inside CI widths of ~0.12), matching the steady-window story: NeuroVoz training data is already sanitized and offers little headroom for preprocessing improvements. **Demo PD phonation improved noticeably (+0.083)**, pushing the deployment case further above the decision threshold. The demo HC case stayed almost identical (0.466 → 0.464) so we did not sacrifice HC discrimination for the PD gain.

**11. Per-file per-subject-eval dropped 0.033 with the vowel filter — expected side effect.**
Old baseline had 5 tasks × ~92 subjects = ~460 files; new pipeline has ~516 files but /u/ contributes only 1.26 reps/subject on average (vs /a/'s 2 reps in old baseline). Fewer averaging bandwidth on the highest-weighted vowel means per-file per-subject aggregation is noisier. This is a training-metric side effect and does not affect the deployment demo (which uses the per-subject weighted vector, not per-file predictions).

**12. Vowel-coverage asymmetry: 4 PD subjects have no /u/ file at all.**
NeuroVoz vowel-file coverage: [I] 49 PD / 46 HC (100%), [O] 49 PD / 46 HC (100%), [U] **45** PD / 46 HC (92%). Four PD subjects contribute IO-only via renormalized weights (0.5 / 0.5). The renormalization keeps them in cohort — none dropped — but their scores rely on the two lower-signal vowels. Corollary for the demo path: `apply_vowel_weights` handles missing-vowel cases gracefully, so demo users can skip /u/ if they can't produce it cleanly.

### Day 4 findings — Claude integration layer built and validated (2026-07-10)

**13. All three Claude-layer files landed and end-to-end validated.**
`summarize.py` (hypomimia narrative JSON), `llm_fusion.py` (canonical `fuse_scores` + `build_claude_context`), and `claude_client.py` (SDK call + frozen system prompt). Fusion consolidation: `_fuse` was extracted out of `quick_score.py` into `llm_fusion.fuse_scores`; `quick_score` now imports it, and `pipeline.py` (Day 5) will use the same function — one canonical late-fusion implementation across the codebase.

End-to-end validation ran on the synthetic PD demo values from finding #10 (phonation 0.613, DDK 0.881, facial 0.148 → fused 0.704, `any_flag_for_review=true`). Claude produced a well-formed Elevated report meeting every system-prompt rule:
- 7 sections in the specified order
- Both mandatory disclaimers appear verbatim (not-a-diagnosis + ON-med caveat)
- Cross-channel disagreement triggered the exact "Per-channel disagreement — flagged for clinical review" line; the never-silently-reconcile invariant held
- Feature units quoted correctly (jitter 0.425%, DDK rate 5.42 Hz, blink rate 8.4/min against the clinical 12–20/min anchor)
- Clinical hypothesis emerged unprompted for the speech/facial disagreement ("early bulbar signs before overt facial hypomimia") — a genuine clinical read the prompt didn't script

Two runs on identical input produced different prose but identical structural elements — the frozen system prompt is tight enough that Claude stochasticity affects wording only, not clinical claims. Canonical artifacts saved at `demo/assets/example_context.xml` (Claude input) and `demo/assets/example_report.md` (Claude output) for demo screenshots and regression-checking.

**Cache activity today:** system prompt is ~1143 tokens; Opus 4.7's cacheable-prefix floor is 4096 tokens, so the `cache_control` marker is a no-op. Kept in place for pattern hygiene — self-activates if the prompt grows past 4K (e.g. few-shot examples). Usage-log to stderr fires only when cache_read or cache_create is non-zero, so silence today is expected.

### Day 5 findings — end-to-end pipeline landed (2026-07-11)

**14. `src/pipeline.py` in place; HC demo runs cleanly from CLI through Claude.**
`run_pipeline(vowel_dir, pataka_dir, smile_dir, call_claude=True)` orchestrates the three per-channel scorers, `fuse_scores`, `build_claude_context`, and `generate_report` into one dict-returning function. Same audio-channel logic as `quick_score.py` (imported `_score_phonation` and `_score_ddk` — one canonical implementation, no drift); facial channel switched to a new `predict_and_summarize` (per-clip: one OpenFace Docker run → classifier score + hypomimia summary from the same DataFrame), so the "one OpenFace run per demo" invariant now actually holds on the code path (previously it was a spec statement; the actual demo path would have double-invoked Docker if you'd called `predict()` and then `summarize_facial_features(csv_path)` separately).

Smoke-test on `data/samples/hc_demo/` (3 vowels × [I,O,U], 3 PATAKA files, 2 smile clips):

| Channel | Score | Notes |
|---|---|---|
| Phonation | 0.464 | borderline; F0 std 1.2 Hz on very steady vowels |
| DDK | 0.028 | strong HC signal — rate 5.03 syl/s, regularity preserved |
| Facial | 0.000 | max-pool over 2 clips, both essentially 0 |
| **Fused** | **0.157** | HC ✓ (matches Day 4 finding #8 pre-Claude number of 0.158 exactly, the 0.001 delta is `-aus -pose` re-parse of pose columns) |

The fusion + agreement machinery worked as specified: `speech_channels_agree: False` (0.464 vs 0.028 gap of 0.436 > 0.30 threshold) → `any_flag_for_review: True` → Claude's report contains the exact "Per-channel disagreement — flagged for clinical review" line, never silently reconciled. Both mandatory disclaimers verbatim. Feature units correct throughout (jitter/shimmer percent, HNR dB, DDK syl/s, blink rate/min against the 12–20 anchor).

**15. Real bug caught by end-to-end integration: OpenFace `-aus` flag was silently dropping pose columns.**
`facial_features._run_openface` originally called `FeatureExtraction -aus` (AU-only output for speed). This was fine when only `predict_smile_pd` consumed the CSV — the classifier only touches AU columns. When the same CSV started feeding `summarize.py` on the demo path, `pose_Tx/Ty/Tz` were missing → `head_movement_std=None` on every demo run, with warnings *"missing column pose_Tx; missing column pose_Ty; missing column pose_Tz"* silently forwarded to Claude. Fixed by switching the Docker call to `-aus -pose` (fallback to no-flags on the rare build that rejects the combo, same as before). Post-fix: `head_movement_std=6.33` on the HC demo, no warnings. This is exactly the kind of surface-level regression that only appears once channels start sharing data — arguing for keeping the integration path exercised on real data, not just synthetic values.

**16. Pipeline vs. quick_score fusion parity confirmed.**
The HC demo through `src/pipeline.py` reproduces the fusion score `quick_score.py` produced on the same demo (0.157 vs 0.157 — same after the pose-column fix; the 0.158 in Day 4 finding #8 was the pre-vowel-filter number). This confirms `pipeline.py` is not silently drifting from the debug tool; both paths use `fuse_scores` from `llm_fusion.py`. If they ever disagree again, the divergence lives in either `_score_facial` (max-pool policy) or the OpenFace CSV — narrow places to look.

**17. PD demo through `pipeline.py` reproduces Day 4 finding #10's numbers to three decimal places.**

| Channel | Day 4 finding #10 (quick_score) | Day 5 pipeline.py | Δ |
|---|---|---|---|
| Phonation | 0.613 | 0.613 | 0.000 |
| DDK | 0.881 | 0.881 | 0.000 |
| Facial (max-pool) | 0.148 | 0.148 | 0.000 |
| **Fused** | **0.704** | **0.704** | **0.000** |
| Predicted | PD ✓ | PD ✓ | — |

Exact reproduction confirms pipeline.py is faithful to the deployed Day-4 configuration — no silent drift from the debug tool, no channel is being scored differently just because the wiring changed. Facial max-pool picked `PD_smile3.mp4` (score 0.148) over `PD_smile1.mp4` (score 0.032), matching the "pick the clip most suggestive of PD" rule.

The Claude report on the PD case: Elevated, driven by DDK's 0.881; speech channels agree with each other (|0.613 − 0.881| = 0.268 < 0.30 threshold), but facial disagrees with the speech mean (|0.148 − 0.747| = 0.599 > 0.30) → correctly flagged for clinical review. Claude produced a genuine clinical hypothesis unprompted: *"elevated speech-based risk with preserved facial expressivity is plausible in early or predominantly speech-affecting presentations, where bulbar/articulatory signs can emerge before overt hypomimia"* — matches the type of read Day 4 finding #13 also observed. Both disclaimers verbatim. Referral recommendation appropriate for Elevated. Full report at `out/pd_demo/report.md`.

**Caveats unchanged from Day 4** (restated to keep scope honest): ON-medication training bias, screening-only framing, etc. — all apply verbatim to `pipeline.py` output because the underlying classifiers and features are identical.

### Two demo deliverables (both honest)
1. `eval/results/ablation_table.csv` — scientific validation on NeuroVoz
2. End-to-end demo on one task-matched volunteer video (+ optional held-out NeuroVoz PD sample) — product running, single subject, one report

---

## Scope Rules (Hackathon Constraints)

**Never cut:**
- Day 3 ablation table (phonation-only, DDK-only, phonation+DDK late fusion, subject-level LOSO on NeuroVoz)
- Day 5 end-to-end demo on a task-matched video

**Can cut if behind:**
- eGeMAPS booster (Parselmouth jitter/shimmer/HNR alone is enough; NeuroVoz ships pre-computed eGeMAPS as a sanity check anyway)
- Fancy Gradio UI (plain output is fine)
- In-the-wild YouTube demo clip (held-out NeuroVoz + self-recording is enough)
- MediaPipe geometric signals for the smile classifier (already cut — Islam 2023 does not publish the landmark indices, so we cannot faithfully reproduce them; cost -0.026 AUROC accepted for reproducibility)

**Honest framing:**
- Facial channel = smile classifier score (quantitative, UFNet-derived, in-distribution test AUROC = 0.812 vs Islam 2023's smile-only 0.830 SVM ensemble) + hypomimia JSON (qualitative narrative). Feature extraction uses **OpenFace 2.0 via Docker** — same extractor as UFNet training, so AU domain gap is zero. AU-only, active-frame-only aggregation; MediaPipe geometric and entropy statistics dropped as non-reproducible.
- Channel inconsistency → flag for clinical review, never silently override
- Demo score is only meaningful when the video is **task-matched** to training (vowels + PATAKA for speech, smile ×3 for facial)
- Report must include explicit disclaimer: **screening decision-aid, not a clinical diagnosis**
- Report must include **ON-medication training caveat**: PD training subjects were all recorded 2–5h post-dose (NeuroVoz paper). The model is calibrated for ON-state PD vs HC; interpretation must note that OFF-state PD would present more perturbed features than the training distribution. This is separate from the not-a-diagnosis disclaimer and is required in every generated report.
- Do not merge external corpora (ParkCeleb) into NeuroVoz training — cross-lingual robustness check only. Same rule for the smile channel: do not merge our extraction into UFNet's training CSV; use their dev/test splits verbatim.

---

## Environment

- Platform: macOS (Apple Silicon)
- `mlx_whisper` requires Apple Silicon; swap to `openai-whisper` for CI/server
- API key: `ANTHROPIC_API_KEY` in `.env`
- Python: 3.12 (via `/opt/homebrew/opt/python@3.12/bin/python3.12`)
- Venv: `.venv/` at repo root — activate with `source .venv/bin/activate`
- Claude model: `claude-opus-4-7`

## What's Built (Day 1)

### `src/audio/transcribe.py`
Five functions: `extract_audio` (ffmpeg WAV extraction), `load_audio` (WAV → float32 numpy), `_run_whisper` (mlx_whisper wrapper, private so tests don't need MLX), `_extract_words_and_pauses` (flattens segments → word list + pause list from gaps ≥ 0.25s), `transcribe_file` (public entry point, handles video or audio input, cleans up temp WAV in `finally`).

Accepts `.wav` directly or any other format (`.m4a`, `.mp3`, `.mp4`, `.mov`, etc.) — non-WAV inputs are converted via ffmpeg to a temp WAV and deleted in `finally`. Condition is `suffix != '.wav'`, not an extension allowlist.

**Post-pivot role:** Whisper is no longer the primary feature extractor. It stays for task-region segmentation on the demo video. The vowel and DDK segments do not need ASR.

Output schema:
```json
{
  "source": "path/to/file",
  "duration_seconds": 120.3,
  "full_transcript": "the cookie...",
  "words": [{"word": "the", "start": 0.12, "end": 0.34, "probability": 0.98}],
  "pauses": [{"start": 5.2, "end": 6.8, "duration": 1.6}],
  "segments": [...]
}
```

### `tests/test_pipeline.py`
Smoke tests for `load_audio` and `_extract_words_and_pauses`. Uses stdlib only (no MLX/audio file needed). Run with `.venv/bin/python tests/test_pipeline.py`.
