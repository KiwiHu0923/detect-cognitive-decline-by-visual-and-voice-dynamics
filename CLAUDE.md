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
- **Facial** score from a smile-task classifier trained on the ROC-HCI UFNet released feature dataset (1361 subjects). Feature extraction uses **OpenFace 2.0** (Docker `algebr/openface`) — the same extractor UFNet trained on, so the AU domain gap is zero. AU-only feature set (7 AUs × mean+var = 14 features, active-frame-only aggregation per Islam et al. 2023); MediaPipe geometric signals dropped because landmark indices are unpublished. A hypomimia JSON summary (AU12 amplitude, expression variance, blink rate) is also passed to Claude as narrative colour, computed separately by py-feat.
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

The demo video must contain the same tasks the classifiers were trained on, or the extracted features are out-of-distribution and the score is meaningless.

Record a single video with these parts (order flexible, but keep tasks clearly separated by a brief pause):

1. **Sustained /a/ for ~5 s** (steady pitch, comfortable loudness) → phonation channel. Language-neutral.
2. **Rapid /pa-ta-ka/ repetition**, as fast and steady as possible for ~5 s → articulation/DDK channel (mirrors the PATAKA task in NeuroVoz). Language-neutral.
3. **Face clearly visible throughout**. For the smile classifier: **8–12 seconds** of smile ×3 alternating with a neutral face (each smile phase ~2–3s + neutral ~1–2s between), per Islam 2023's protocol → active-frame AU statistics + hypomimia narrative (AU12 amplitude, expression variance).

**Segmentation:** the sustained-vowel and DDK segments are separated by the instructed pauses (or detected: vowel = long continuously-voiced low-F0-variance region; DDK = regular high-rate intensity-peak train).

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
| ASR / segmentation | `mlx_whisper` (Apple Silicon) | `mlx-community/whisper-base-mlx`; swap to `openai-whisper` for non-Mac. Used for task-region segmentation on the demo video |
| Phonation features | `praat-parselmouth` | jitter, shimmer, HNR, F0 — the interpretable core |
| Articulation / DDK features | `praat-parselmouth` + `scipy.signal` | intensity-envelope peak picking on the PATAKA task → DDK rate + timing/amplitude regularity (no ASR) |
| (Optional) extra acoustic | `opensmile` (eGeMAPS, 88 feats) | prediction booster if time permits; includes jitter/shimmer/HNR too |
| Facial AU extraction (classifier) | OpenFace 2.0 via Docker `algebr/openface` | matches UFNet training extractor; 7 AUs × mean+var = 14 features, active-frame-only |
| Facial narrative (hypomimia summary) | `py-feat` | pip-installable; AUs, emotion, head pose, gaze — used for Claude narrative JSON, not the classifier |
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
│   │   ├── transcribe.py           # Whisper → word-timestamped .json (task segmentation)
│   │   ├── segment.py              # split uploaded audio into vowel / DDK regions
│   │   ├── phonation.py            # Parselmouth jitter/shimmer/HNR/F0 stats on vowels
│   │   └── ddk.py                  # intensity-envelope peak picking → DDK rate + regularity
│   ├── vision/
│   │   ├── train_smile_pd.py       # trains 14-feature smile classifier on UFNet CSV (active-frame-only, AU only)
│   │   ├── facial_features.py      # OpenFace Docker → per-frame AU_r + AU_c for 7 AUs (video → arrays + detection meta)
│   │   ├── aggregate.py            # per-frame AU_r/AU_c → 14-dim session vector via active-frame mean+var
│   │   ├── predict_smile_pd.py     # video → PD score + detection-rate gate
│   │   └── summarize.py            # py-feat AU + head + emotion → hypomimia JSON (Claude narrative, separate from classifier)
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
- Content segmentation (intro/ads/outro/content) → task segmentation (vowel / DDK) + PD feature extraction

---

## Phonation Features (sustained vowels, Parselmouth)

Computed on the vowel segment only. Valid only on quasi-periodic, stable-pitch signals.

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

Also computed by `src/vision/summarize.py` from py-feat outputs (AUs, emotion, head pose, gaze), as narrative colour for the Claude report — not a classifier:
```json
{
  "mean_AU12": 0.15,
  "AU12_amplitude_on_smile_cue": 0.28,
  "expression_variance": 0.12,
  "hypomimia_score": 0.72,
  "blink_rate_per_min": 8.4,
  "head_movement_std": 0.04,
  "dominant_emotion": "neutral",
  "emotion_variability": 0.18
}
```

---

## Claude Integration (Layer 2)

`src/report/claude_client.py` builds a structured prompt combining:
1. Phonation classifier score + confidence (Layer 1)
2. DDK classifier score + confidence (Layer 1)
3. Facial summary JSON (hypomimia markers)
4. Weighted-vote late-fusion score + per-channel agreement flag

Claude generates a report with:
- Risk level (Low / Moderate / Elevated — **NOT a diagnosis**)
- Key phonation observations (jitter/shimmer/HNR narrative)
- Key articulation observations (DDK rate/regularity narrative)
- Key facial observations (hypomimia narrative)
- Consistency flag (do speech and facial channels agree?)
- Recommended next steps
- **Explicit disclaimer** (screening decision-aid, not diagnosis)

Use `claude-opus-4-7`. Include prompt caching for repeated calls.

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
