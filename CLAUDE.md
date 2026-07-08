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
| Primary dataset | ADReSSo / DementiaBank (membership approval, faculty-sponsor blocker) | Italian Parkinson's Voice and Speech (IPVS) — open access + HF mirror, no DUA | Removes the access blocker entirely; downloadable Day 1 |
| Fused modalities | acoustic × linguistic (TTR, fillers, RoBERTa — semantic/lexical) | phonation × articulation (DDK), both motor-speech (prosody = optional 3rd) | PD is a motor speech disorder, not a language disorder; semantic features are off-target |
| Core acoustic tool | librosa (preferred) | praat-parselmouth (preferred) | jitter/shimmer/HNR are Praat's native, reproducible, clinically interpretable indices |
| Phonation task | Cookie Theft spontaneous description | sustained vowels | jitter/shimmer/HNR are only valid on quasi-periodic, stable-pitch signals |
| Cross-corpus data | pooled backup corpora | single clean corpus (IPVS); no external corpora merged into training | mixing tasks/corpora at small N → dataset-bias shortcut, inflated AUC |

---

## Two-Layer Architecture (Core Design Decision)

### Layer 1 — Scientific Validation Spine (offline, on IPVS)

**Score-level (late) fusion** of two speech channels from the same subjects but different tasks. Purpose: produce a rigorous ablation table showing fusion > single-channel. This answers the hardest evaluator question: "does fusion actually add signal?"

- **Phonation channel** — sustained vowels → jitter, shimmer, HNR, F0 stats (Parselmouth). The core PD phonatory markers.
- **Articulation channel (DDK)** — rapid /pa/ and /ta/ repetition → DDK rate + timing/amplitude regularity. A strong, PD-specific motor marker (articulatory bradykinesia/irregularity), extracted from the intensity envelope — **no ASR needed**, and **language-neutral** (good for the demo).
- **(Optional) Prosody channel** — reading passage → speech rate, pause structure, F0 variability/monotonicity, intensity. Reuses the existing Whisper pause-gap code. Note: IPVS reading is Italian, so this channel has a mild cross-language wrinkle for a non-Italian self-recorded demo (phonation and DDK do not).
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
- **Facial** summary (py-feat AU timeseries → structured JSON) is a third measurement channel, lower weight, qualitative
- If channels agree → higher confidence; if they disagree → report flags for clinical review (never silently overridden)

---

## Data

### Primary Dataset: Italian Parkinson's Voice and Speech (IPVS)

- **Source:** Dimauro & Girardi, IEEE DataPort (doi:10.21227/aw6b-tg17). Open Access, CC BY 4.0 — files available to all users upon login. Hugging Face mirror also exists (`birgermoell/Italian_Parkinsons_Voice_and_Speech`) → near-zero-wait pull. Verify citation terms before use.
- **Three task families** (~831 audio files total; this is the key point — it is **NOT vowel-only**):
  1. Sustained vowels /a/e/i/o/u/ → phonation channel
  2. DDK syllables — rapid /pa/ and /ta/ repetition → articulation channel
  3. Reading a phonetically-balanced Italian text (with a pause) → optional prosody channel
- **Language note:** vowels and DDK are language-neutral, so a non-Italian self-recorded demo aligns cleanly with training. The reading task is Italian (Dimauro used a deliberately non-semantic passage), so the optional prosody channel carries a mild cross-language caveat at demo time.
- Many published studies use only the vowels (found more predictive, and it removes language/education confounds) — that is a researcher's subsetting choice, not a limitation of the raw data. DDK and reading are present.
- **Size:** 28 PD, 37 HC (15 young HC ages 19–29 + 22 elderly HC ages 60–77). (Counts vary across papers, 22–50 HC, depending on which tasks/subsets are used.)
- **Cohort:** the primary comparison is **28 PD vs 22 elderly HC** (age-matched) to avoid the classifier learning "young vs old" instead of "PD vs HC". The 15 young HC subjects have no vowel or DDK recordings — only reading files — so they cannot participate in the phonation or DDK channel regardless. Age-matching is enforced by the data itself.
- **Access:** download, no DUA / no membership approval.
- Path when downloaded: `data/raw/italian_pd/`

### Under review — bonus only, NOT on the critical path

- **NeuroVoz** (Zenodo DUA, Spanish — DDK + sustained vowels) and **ParkCeleb** — both access requests are pending approval.
- If either clears in time, use it only as a **second phonation corpus for a cross-lingual robustness check** — never merged into IPVS training (different language/device/task → dataset bias at small N).
- Treat as pure upside. The project ships complete on IPVS alone; approval progress must not gate any deliverable.
- Path (if they arrive): `data/raw/external/`

### Demo Data

- Single self-recorded or volunteer video, **task-matched to training** (see Demo Protocol).
- Path: `data/samples/`

---

## Demo Protocol (task-matching is mandatory)

The demo video must contain the same tasks the classifiers were trained on, or the extracted features are out-of-distribution and the score is meaningless.

Record a single video with these parts (order flexible, but keep tasks clearly separated by a brief pause):

1. **Sustained /a/ for ~5 s** (steady pitch, comfortable loudness) → phonation channel. Language-neutral.
2. **Rapid /pa/ then /ta/ repetition**, as fast and steady as possible for ~5 s each → articulation/DDK channel. Language-neutral.
3. **(Optional) Read a short passage aloud** → prosody channel. Mild language caveat if not Italian.
4. **Face clearly visible throughout**; optionally a neutral→smile→neutral sequence → hypomimia (AU12 amplitude, expression variance).

**Segmentation:** the sustained-vowel and DDK segments are separated by the instructed pauses (or detected: vowel = long continuously-voiced low-F0-variance region; DDK = regular high-rate intensity-peak train). Whisper timestamps isolate the reading portion if used.

**Data-leakage guard:** if a held-out IPVS PD sample is used as a second demo case, it MUST be the subject left out in its LOSO fold — never a subject the classifier was fitted on. The self-recorded healthy sample is out-of-training by construction.

**Primary demo pair (cleanest):** self-recorded healthy video (task-matched) + one held-out IPVS PD sample (real label, task-matched, no privacy issue).

**Optional in-the-wild sample** (e.g. a YouTube PD clip) — bonus only, with guards:
- **Task mismatch:** most such clips are interviews/spontaneous speech, not sustained vowels or DDK. Do not feed connected speech to the phonation or DDK classifier — the pipeline should return those channels as **N/A** and score only the prosody + facial channels (task-compatible). This also showcases the system's modularity.
- **Signal quality:** compression, background music, multiple speakers inflate jitter/shimmer artefactually — pick single-speaker, quiet, close-mic segments and note quality in the report.
- **Label & ethics:** the condition is self-reported/inferred, not a clinical label → illustrative only, **never counted toward AUC**. Mark clearly as "public video, self-reported condition, illustrative." Prefer the IPVS held-out sample as the labelled PD case; use the in-the-wild clip only as a secondary "runs on real videos" demonstration.

---

## Tech Stack

| Component | Library | Notes |
|-----------|---------|-------|
| Audio extraction | ffmpeg via subprocess | 16kHz mono WAV |
| ASR / segmentation | `mlx_whisper` (Apple Silicon) | `mlx-community/whisper-base-mlx`; swap to `openai-whisper` for non-Mac. Used for the optional reading task + task segmentation |
| Phonation features | `praat-parselmouth` | jitter, shimmer, HNR, F0 — the interpretable core |
| Articulation / DDK features | `praat-parselmouth` + `scipy.signal` | intensity-envelope peak picking → DDK rate + timing/amplitude regularity (no ASR) |
| (Optional) Prosody features | custom from Whisper timestamps + Parselmouth | speech rate, pause structure, F0 variability/monotonicity, intensity |
| (Optional) extra acoustic | `opensmile` (eGeMAPS, 88 feats) | prediction booster if time permits; includes jitter/shimmer/HNR too |
| Facial extraction | `py-feat` | pip-installable; AUs, emotion, head pose, gaze |
| Statistical fusion | `scikit-learn` | LogReg / linear SVM per channel + score-level fusion |
| LLM / report | `anthropic` SDK | Claude report layer |
| UI | `gradio` | fastest for hackathon demo |

---

## Repository Structure

```
parkscreen/
├── data/                          # gitignored
│   ├── raw/
│   │   ├── italian_pd/             # IPVS: vowels/ ddk/ reading/ + labels.csv
│   │   └── external/               # (bonus) NeuroVoz / ParkCeleb if approved
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
│   │   ├── extract.py              # py-feat AU + gaze + head timeseries
│   │   └── summarize.py            # timeseries → structured clinical JSON (hypomimia focus)
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

## Articulation / DDK Features (rapid /pa/–/ta/, no ASR)

Computed on the DDK segment via intensity-envelope peak picking. Language-neutral.

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

## Facial Features (py-feat) — Hypomimia Focus

`py-feat` outputs per-frame:
- Action Units (AUs): AU1, AU2, AU4, AU6, **AU12** (smile — key for hypomimia), AU15, AU17, AU20, AU25, etc.
- Emotion estimates (anger, disgust, fear, happy, sad, surprise, neutral)
- Head pose (yaw, pitch, roll)
- Gaze direction

Summarized in `src/vision/summarize.py` into clinical JSON (PD-relevant markers):
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

Facial channel is **qualitative supporting evidence** — not a validated classifier. See Scope Rules.

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

Cross-validation: **subject-level LOSO** on IPVS. Cohort: 28 PD vs 22 elderly HC (age-matched — enforced by the data, since young HCs have no vowel or DDK recordings).

### Two demo deliverables (both honest)
1. `eval/results/ablation_table.csv` — scientific validation on IPVS
2. End-to-end demo on one task-matched volunteer video (+ optional held-out IPVS PD sample) — product running, single subject, one report

---

## Scope Rules (Hackathon Constraints)

**Never cut:**
- Day 3 ablation table (phonation-only, DDK-only, phonation+DDK late fusion, subject-level LOSO on IPVS)
- Day 5 end-to-end demo on a task-matched video

**Can cut if behind:**
- Optional prosody channel (reading task) — phonation+DDK is the core deliverable
- eGeMAPS booster (Parselmouth jitter/shimmer/HNR alone is enough)
- Fancy Gradio UI (plain output is fine)
- In-the-wild YouTube demo clip (held-out IPVS + self-recording is enough)
- Facial validation on labeled dataset (keep facial path as qualitative only)

**Honest framing:**
- Facial hypomimia is a "third measurement channel" — not a validated classifier
- Channel inconsistency → flag for clinical review, never silently override
- Demo score is only meaningful when the video is **task-matched** to training
- Report must include explicit disclaimer: **screening decision-aid, not a clinical diagnosis**
- Do not merge external corpora (NeuroVoz / ParkCeleb) into IPVS training — cross-lingual robustness check only

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
