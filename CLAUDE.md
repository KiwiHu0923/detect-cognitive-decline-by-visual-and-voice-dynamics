# ParkScreen — Built with Claude: Life Sciences Hackathon

## Working Protocol

Before every non-trivial code change:
1. **Brief the user** — explain each function/block in plain language (algorithm, non-obvious decisions)
2. **Wait for "go"** before touching any files
3. **After implementation** — mark TODO.md items `[x]` and update CLAUDE.md if architecture/schemas changed

## Project Overview

Multimodal screening tool that detects motor-speech and facial signs of Parkinson's disease (PD) from a single uploaded video. Fuses **phonation** signals (sustained-vowel voice quality) and **articulation** signals (DDK syllable rate/regularity) with **facial dynamics** (hypomimia), routed through a Claude integration layer that generates a clinical-style report. An optional **prosody** channel (reading-passage timing/melody) can be added as a third speech channel if time allows.

**Positioning:** a supporting-evidence / screening decision-aid, explicitly **not a diagnosis**.

**Hackathon:** Built with Claude: Life Sciences (Anthropic × Gladstone × Cerebral Valley), July 7–13 2026. Development Track.

---

## Changelog: CogniScreen (dementia) → ParkScreen (PD)

This project was pivoted from a dementia/cognitive-decline design. The changes are scientific, not cosmetic:

| Area | Was (dementia) | Now (PD) | Why |
|------|----------------|----------|-----|
| Primary dataset | ADReSSo / DementiaBank (membership approval, faculty-sponsor blocker) | NeuroVoz (Spanish) — Zenodo DUA cleared 2026-07-08, open access | Removes the access blocker; 53 PD + 51 age-matched HC (analysis cohort ≈ 49 × 46 after PATAKA-intersection), ~2× IPVS size, richer clinical labels (UPDRS, H-Y stage, medication ON/OFF, GRBAS ratings) |
| Fused modalities | acoustic × linguistic (TTR, fillers, RoBERTa — semantic/lexical) | phonation × articulation (DDK), both motor-speech (prosody = optional 3rd) | PD is a motor speech disorder, not a language disorder; semantic features are off-target |
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

Language changes Italian → Spanish; the demo caveat (vowels + DDK language-neutral, reading has a mild cross-language wrinkle) is unchanged.

---

## Two-Layer Architecture (Core Design Decision)

### Layer 1 — Scientific Validation Spine (offline, on NeuroVoz)

**Score-level (late) fusion** of two speech channels from the same subjects but different tasks. Purpose: produce a rigorous ablation table showing fusion > single-channel. This answers the hardest evaluator question: "does fusion actually add signal?"

- **Phonation channel** — sustained vowels (3 reps per vowel, /a/e/i/o/u/) → jitter, shimmer, HNR, F0 stats (Parselmouth). The core PD phonatory markers.
- **Articulation channel (DDK)** — rapid /pa-ta-ka/ (PATAKA task) → DDK rate + timing/amplitude regularity. A strong, PD-specific motor marker (articulatory bradykinesia/irregularity), extracted from the intensity envelope — **no ASR needed**, and **language-neutral** (good for the demo).
- **(Optional) Prosody channel** — Spanish word tasks + FREE spontaneous speech → speech rate, pause structure, F0 variability/monotonicity, intensity. Reuses the existing Whisper pause-gap code. Note: NeuroVoz is Spanish, so this channel has a mild cross-language wrinkle for a non-Spanish self-recorded demo (phonation and DDK do not).
- Tasks are **NOT concatenated** into one feature vector. Each channel trains its own small classifier; the probabilities are fused at score level. This respects the fact that voice-quality, articulatory-timing, and connected-speech features live on different scales and come from different tasks.
- **Models trained:** phonation-only, DDK-only, phonation+DDK late fusion (+ optional prosody as a further channel). Literature supports this: combining sustained phonation with a second task outperforms any single task.
- **Evaluation:** subject-level LOSO (Leave-One-Subject-Out); metrics: AUC, F1, sensitivity, specificity.
- Results saved to `eval/results/ablation_table.csv`.

**Model class:** small + interpretable (logistic regression / linear SVM), **NOT deep learning**. N is a few dozen subjects — this is exactly the regime PD-voice literature operates in. A small LogReg also gives (a) a LOSO AUC — the single most persuasive number for evaluators, and (b) a clean 0–1 probability to feed the fusion layer and the Claude report.

### Layer 2 — Product Layer (real-time, on user-uploaded video)

User uploads a single video (**task-matched** — see Demo Protocol) → pipeline extracts phonation features (from the vowel segment) + articulation/DDK features (from the /pa/–/ta/ segment) + facial hypomimia features → all passed as structured context into a Claude prompt → Claude generates a clinical-style report with confidence flags.

At inference time the pipeline extracts features from the uploaded input, and scores them with the **already-fitted Layer-1 classifiers**. No training happens at demo time.

**Fusion mechanism: weighted voting (late fusion)**
- **Phonation** score weighted highest (jitter/shimmer/HNR are the most established phonatory PD markers)
- **DDK/articulation** score weighted next
- **Facial** score from a smile-task classifier trained on the ROC-HCI UFNet released feature dataset (1361 subjects). Feature extraction uses **OpenFace 2.0** (Docker `algebr/openface`) — the same extractor UFNet trained on, so the AU domain gap is zero. AU-only feature set (7 AUs × mean+var = 14 features, active-frame-only aggregation per Islam et al. 2023); MediaPipe geometric signals dropped because landmark indices are unpublished. A hypomimia JSON summary (AU12 amplitude, expression variance, blink rate) is also passed to Claude as narrative colour, computed separately by py-feat.
- If channels agree → higher confidence; if they disagree → report flags for clinical review (never silently overridden)

---

## Data

### Primary Dataset: NeuroVoz (Spanish)

- **Source:** Zenodo (DUA cleared 2026-07-08). Spanish PD voice corpus, ~2900 audio files across 53 PD + 55 HC subjects. Verify citation terms before use.
- **Three task families** (present for most, but not all, subjects — coverage matters, see Cohort below):
  1. Sustained vowels /a/e/i/o/u/, up to 3 reps each (files named `A1..A3, E1..E3, I1..I3, O1..O3, U1..U3`) → phonation channel. Only the "1"/"2" reps are near-universal; A3/E3/I3/O3/U3 exist for only a handful of subjects and should be treated as bonus, not required inputs.
  2. **PATAKA** — rapid /pa-ta-ka/ repetition → articulation channel. This is the canonical clinical DDK task (more standard than IPVS's separate /pa/ + /ta/ files). Present for 46 HC + 49 PD subjects after age filtering — not universal, so subjects without PATAKA are excluded from the analysis cohort.
  3. 20+ Spanish word/sentence tasks (ABLANDADA, ACAMPADA, BARBAS, BURRO, CALLE, CARMEN, DIABLO, GANGA, MANGA, PAN_VINO, PATATA_BLANDA, PERRO, PETACA_BLANCA, PIDIO, SOMBRA, TOMAS, …) plus **FREE** (spontaneous speech) → optional prosody channel. Note: metadata references "ESPONTANEA" for the spontaneous-speech task, but the on-disk filename is `FREE_` — `src/data/build_labels.py` normalizes this at ingest.
- **Language note:** vowels and PATAKA are language-neutral, so a non-Spanish self-recorded demo aligns cleanly with training. The reading/free-speech tasks are Spanish, so the optional prosody channel carries a mild cross-language caveat at demo time.
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
3. **(Optional) Read a short passage aloud** → prosody channel. Mild language caveat if not Spanish.
4. **Face clearly visible throughout**. For the smile classifier: **8–12 seconds** of smile ×3 alternating with a neutral face (each smile phase ~2–3s + neutral ~1–2s between), per Islam 2023's protocol → active-frame AU statistics + hypomimia narrative (AU12 amplitude, expression variance).

**Segmentation:** the sustained-vowel and DDK segments are separated by the instructed pauses (or detected: vowel = long continuously-voiced low-F0-variance region; DDK = regular high-rate intensity-peak train). Whisper timestamps isolate the reading portion if used.

**Data-leakage guard:** if a held-out NeuroVoz PD sample is used as a second demo case, it MUST be the subject left out in its LOSO fold — never a subject the classifier was fitted on. The self-recorded healthy sample is out-of-training by construction.

**Primary demo pair (cleanest):** self-recorded healthy video (task-matched) + one held-out NeuroVoz PD sample (real label, task-matched, no privacy issue).

**Optional in-the-wild sample** (e.g. a YouTube PD clip) — bonus only, with guards:
- **Task mismatch:** most such clips are interviews/spontaneous speech, not sustained vowels or DDK. Do not feed connected speech to the phonation or DDK classifier — the pipeline should return those channels as **N/A** and score only the prosody + facial channels (task-compatible). This also showcases the system's modularity.
- **Signal quality:** compression, background music, multiple speakers inflate jitter/shimmer artefactually — pick single-speaker, quiet, close-mic segments and note quality in the report.
- **Label & ethics:** the condition is self-reported/inferred, not a clinical label → illustrative only, **never counted toward AUC**. Mark clearly as "public video, self-reported condition, illustrative." Prefer the NeuroVoz held-out sample as the labelled PD case; use the in-the-wild clip only as a secondary "runs on real videos" demonstration.

---

## Tech Stack

| Component | Library | Notes |
|-----------|---------|-------|
| Audio extraction | ffmpeg via subprocess | 16kHz mono WAV |
| ASR / segmentation | `mlx_whisper` (Apple Silicon) | `mlx-community/whisper-base-mlx`; swap to `openai-whisper` for non-Mac. Used for the optional reading task + task segmentation |
| Phonation features | `praat-parselmouth` | jitter, shimmer, HNR, F0 — the interpretable core |
| Articulation / DDK features | `praat-parselmouth` + `scipy.signal` | intensity-envelope peak picking on the PATAKA task → DDK rate + timing/amplitude regularity (no ASR) |
| (Optional) Prosody features | custom from Whisper timestamps + Parselmouth | speech rate, pause structure, F0 variability/monotonicity, intensity |
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
│   │   ├── transcripts/            # Whisper .json output (reading + segmentation)
│   │   ├── phonation_features/     # .npy arrays (jitter/shimmer/HNR/F0)
│   │   ├── ddk_features/           # .npy arrays (DDK rate + regularity)
│   │   ├── prosody_features/       # .npy arrays (optional)
│   │   └── facial_features/        # .json AU timeseries
│   └── samples/                    # demo video
│
├── src/
│   ├── audio/
│   │   ├── transcribe.py           # Whisper → word-timestamped .json (reading + segmentation)
│   │   ├── segment.py              # split uploaded audio into vowel / DDK / reading regions
│   │   ├── phonation.py            # Parselmouth jitter/shimmer/HNR/F0 stats on vowels
│   │   ├── ddk.py                  # intensity-envelope peak picking → DDK rate + regularity
│   │   └── prosody.py              # (optional) speech rate, pauses, F0 variability from reading
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
Uses `mlx_whisper.transcribe(..., word_timestamps=True)`. Already computes inter-segment gaps. Still needed for (a) task-region segmentation on the demo video and (b) the optional prosody channel.

**`classify_non_speech_fast(audio_array, framerate, start_time, end_time)`**
RMS energy silence detection. Repurpose in `src/audio/segment.py` and `src/audio/ddk.py` (intensity envelope peak picking).

### Critical insight from existing code
The `run_whisper_extraction` function already computes `gap = seg["start"] - current_time` for each silence between speech segments. This is the **pause-structure feature** for the optional prosody channel — it comes for free from the existing Whisper logic. Lift into `src/audio/prosody.py` if the prosody channel is added.

### What to replace
- `ollama` → `anthropic` SDK
- `MODEL_NAME / ollama.chat()` → `claude_client.py` with Anthropic SDK calls
- Content segmentation (intro/ads/outro/content) → task segmentation (vowel / DDK / reading) + PD feature extraction

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

## (Optional) Prosody Features (reading passage, from Whisper output)

Computed from word-timestamped Whisper JSON + Parselmouth pitch. **Skip if behind schedule.**

| Feature | Method |
|---------|--------|
| Speech rate (words/sec) | total words / voiced duration |
| Mean pause duration | mean of inter-word gaps > 0.25s |
| Pause rate | count of pauses > 0.25s per minute |
| F0 variability (SD, semitones) | Parselmouth pitch over voiced frames — monotonicity marker |
| F0 range (voiced) | max − min over voiced frames |
| Intensity mean / SD | Parselmouth intensity |

---

## Facial Features — Smile Classifier + Hypomimia Summary

The facial channel has two outputs, both fed to the Claude report:

### 1. Smile classifier score (quantitative)

Trained on the ROC-HCI UFNet released feature dataset (1361 subjects, MIT-licensed) via `src/vision/train_smile_pd.py`. UFNet's ShallowANN is mathematically a logistic regression over the session-level feature vector, so we use sklearn `LogisticRegression` + `StandardScaler` + SMOTE — same model class, no torch dependency at inference.

**Feature set: 14 = 7 AUs × {mean, var}, active-frame-only aggregation.**

The active-frame-only rule comes from Islam et al. 2023 ("Unmasking Parkinson's Disease with Smile", arxiv 2308.02588 — the precursor to UFNet), which specifies that mean/variance are computed **only on frames where the AU's binary presence flag (`AU_c == 1`) is set**, not over all frames. Physical meaning: "how intense the AU is *when it fires*" rather than "average intensity across the whole clip". Full-frame aggregation systematically dilutes toward 0 and mimics the py-feat scale-gap failure mode.

We dropped the 7 MediaPipe geometric signals (eye-open, mouth-width, jaw-open, etc.) because Islam 2023 does not publish the landmark indices used to compute them, so we cannot reproduce their extraction faithfully. Ablation cost: -0.026 test AUROC (0.839 → 0.812). We also dropped the entropy statistic across all signals because Islam 2023 does not publish the histogram binning, log base, or range — non-reproducible. Ablation cost: -0.028 test AUROC (0.837 → 0.812).

**Reference AUROC:** paper's smile-only 0.830 (Islam 2023, 10-fold CV, SVM ensemble); our test AUROC on UFNet's held-out participant split = **0.812** (LogReg on 14 active-frame features). The gap is within noise given the model-class simplification and the aggressive feature reduction for reproducibility.

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
- `splits/{dev,test,calib}.txt` (participant IDs — reproduces the paper's split)
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
3. (Optional) Prosody classifier score + confidence
4. Facial summary JSON (hypomimia markers)
5. Weighted-vote late-fusion score + per-channel agreement flag
6. Raw transcript excerpt (if reading task present)

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

### Ablation table (the deliverable for evaluators)
| Model | AUC | F1 | Sensitivity | Specificity |
|-------|-----|----|-------------|-------------|
| Phonation-only | ? | ? | ? | ? |
| DDK-only | ? | ? | ? | ? |
| Phonation + DDK (late fusion) | ? | ? | ? | ? |
| (+ optional) Phonation + DDK + Prosody | ? | ? | ? | ? |

Cross-validation: **subject-level LOSO** on NeuroVoz. Raw age-matched cohort is 53 PD vs 51 HC; the analysis cohort further requires each subject to have PATAKA + ≥ 1 vowel → **≈ 49 PD vs 46 HC**.

### Two demo deliverables (both honest)
1. `eval/results/ablation_table.csv` — scientific validation on NeuroVoz
2. End-to-end demo on one task-matched volunteer video (+ optional held-out NeuroVoz PD sample) — product running, single subject, one report

---

## Scope Rules (Hackathon Constraints)

**Never cut:**
- Day 3 ablation table (phonation-only, DDK-only, phonation+DDK late fusion, subject-level LOSO on NeuroVoz)
- Day 5 end-to-end demo on a task-matched video

**Can cut if behind:**
- Optional prosody channel (word/free-speech tasks) — phonation+DDK is the core deliverable
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

**Post-pivot role:** Whisper is no longer the primary feature extractor. It stays for (a) segmenting the demo video's reading-task region and (b) feeding the optional prosody channel (pauses, speech rate). The vowel and DDK segments do not need ASR.

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
