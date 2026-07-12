# ParkScreen — Limitations

**Read this before interpreting any ParkScreen output.**

ParkScreen is a hackathon-built research prototype. It is a **screening decision aid**, not a clinical diagnostic tool. This document is the evaluator- and user-facing catalog of every material limitation of the current release. For the underlying methodology see [`METHODOLOGY.md`](METHODOLOGY.md); for dataset citations see [`DATASETS.md`](DATASETS.md).

---

## 1. Not a diagnosis

Every report emitted by ParkScreen carries the disclaimer *"This is a screening decision aid, not a clinical diagnosis"* — verbatim, at generation time, enforced by the frozen system prompt. Any elevated or borderline score should be interpreted only as a suggestion to seek qualified clinical evaluation (movement disorder specialist, neurologist). No treatment decision should be made from ParkScreen output alone.

## 2. ON-medication training bias

Every PD subject in the NeuroVoz training corpus was recorded **on Parkinson's medication**, 2–5 hours after their scheduled dose (NeuroVoz paper §Data records). This has three consequences:

- The classifier is calibrated for **medicated (ON-state) PD vs age-matched healthy controls**, not for any-state PD.
- OFF-state PD (medication wearing off, or untreated PD) presents with more perturbed features than the training distribution. The model would tend to flag such subjects, but its calibrated probability is not valid for that distribution.
- The Claude report layer surfaces this caveat in every generated report — *"NeuroVoz training data was recorded on medication (2–5 h post-dose)"* — verbatim.

## 3. Age imbalance

Despite an age-matching filter (healthy controls age ≥ 50), the PD subjects in the analysis cohort are on average **5 years older** than the healthy controls:

| Group | Mean age | Median | Range |
|-------|----------|--------|-------|
| Healthy Control | 65.8 | 66 | 53–86 |
| Parkinson's | 70.8 | 71 | 41–88 |

(Welch t=2.53, p=0.013, Cohen's d = 0.52 — medium effect size.)

Voice quality changes with age (jitter and shimmer trend upward, F0 changes across adulthood). A fraction of our reported AUC is measuring age rather than PD. Two mitigating observations:

- Age filter (HC ≥ 50) means both groups sit in the vocal-aging plateau region, not comparing PD elderly vs young healthy.
- Fold-level failure analysis: misclassified subjects have essentially the same age distribution as correctly classified ones (67.3 vs 68.9, non-significant). If age were the dominant signal, we would expect the oldest subjects to be classified PD regardless of ground truth — but the age distributions of correct-vs-misclassified overlap heavily. Age is a real confounder but is not the primary driver.

We do not age-adjust the classifier because N=95 is too small for a stable regression-adjustment step.

## 4. Small cohort

The analysis cohort is 49 PD + 46 healthy controls after filtering (raw NeuroVoz has 53 PD + 55 HC; we drop HC subjects with age < 50 or missing age, and any subject without both PATAKA and ≥ 1 sustained-vowel file). This gives:

- Bootstrap 95% CI on AUC ≈ ±0.10 wide. Any AUC comparison narrower than that is inside noise.
- Per-fold prediction variance is high enough that fold-level analysis cannot cleanly attribute misclassifications to specific subject-level factors.

## 5. Mild-to-moderate PD only

The PD subjects in NeuroVoz span the mild-to-moderate range of the disease:

| Hoehn-Yahr stage | Count | Description |
|------------------|-------|-------------|
| 1 | 4 | unilateral |
| 2 | 32 | bilateral, no balance impairment |
| 3 | 11 | bilateral with postural instability |
| 5 | 1 | wheelchair-bound |

UPDRS median 14 (IQR 9.5–20), max 47. Two-thirds of subjects sit at H-Y stage 2. **Our AUC does not extrapolate to severe (H-Y stage 4–5) PD**. Conversely, our fold-level analysis shows early-stage (H-Y 1) PD is the hardest to detect — 25% of stage-1 subjects were misclassified.

## 6. Spanish training data; task-matching mandatory

NeuroVoz is a Spanish-language corpus. Two of our three input channels are language-neutral:

- Sustained vowels /i/, /o/, /u/ — no linguistic content
- PATAKA DDK task — nonsense-syllable rapid repetition, language-neutral by clinical convention

The third channel (Spanish reading tasks + spontaneous speech) was **dropped from ParkScreen** to avoid cross-language transfer concerns. We do not use prosody or linguistic features.

**However**, the demo input must still be task-matched: sustained-vowel recordings, PATAKA repetition, and a smile-task video, in three separate upload buckets. **Any off-task input** (spontaneous speech uploaded as a "vowel", conversation as "PATAKA", a still selfie as "smile") produces meaningless features. The pipeline returns `N/A` for channels whose input is missing or off-task; it does not — cannot — detect off-task uploads within a bucket.

## 7. Facial channel domain transfer

The facial (smile-task) classifier is trained on the ROC-HCI UFNet dataset (US clinical + YouTube samples, 1,361 subjects), NOT on NeuroVoz.

- **In-distribution AUROC on UFNet's held-out test split: 0.812** (compared to Islam 2023 paper's smile-only 0.830 SVM ensemble).
- Extraction pipeline uses OpenFace 2.0 via Docker — the same extractor UFNet trained on, so there is no AU domain gap between training and inference.
- **We do not have a Spanish facial validation set.** The AUROC 0.812 number applies to English-speaking (UFNet) subjects performing the same smile ×3 task. Cross-cultural expression differences are untested.
- External validation on YouTubePD: AUROC 0.708 on UFNet's designated subset, 0.602 on the full 251-clip release. Neither number is demo-representative — YouTubePD includes off-task, low-quality wild samples. These numbers give an upper and lower bound for wild-transfer.

## 8. UFNet training data provenance is opaque — the load-bearing reason we downweight facial

**This is one of the most material limitations of the facial channel and directly shapes how we weight it in the fusion.**

UFNet released **feature CSVs only, never the raw video** used to produce them. Consequently, none of the following can be verified from our side:

- **Camera hardware and resolution** used at recording time.
- **Framing, distance, and lighting** of participants.
- **Instruction quality** given to participants (how were they cued to smile? how many practice trials? were low-quality clips re-recorded or included as-is?).
- **OpenFace parameter choices** at extraction time — even though we run the same Docker image (`algebr/openface:latest`), UFNet's team did not publish the exact CLI flags, the detection-confidence threshold they applied, or the frame-drop policy for low-confidence frames. Our extraction settings (`-aus -pose`, confidence ≥ 0.75) are our best reproduction of their pipeline based on published methods, but we cannot audit whether they match bit-for-bit.
- **Participant demographics beyond age and PD status.** UFNet publishes basic labels; anything about ethnicity, gender expression, facial hair (which affects AU12 detection), or eyeglasses (which affects AU45 blink detection) is unavailable.

The classifier is therefore a **function of features from a partly-opaque upstream pipeline**. Even our in-distribution AUROC 0.812 sits on top of a training distribution whose recording protocol is not fully characterized.

**Concrete downstream consequence — this is why our facial fusion weight is 0.15, not 0.31:**

The AUC-excess-over-chance heuristic (see [`METHODOLOGY.md`](METHODOLOGY.md) §4.2) would give facial a fusion weight of `max(0, 0.812 − 0.5) ≈ 0.31`, ranking it as the strongest channel we have. **We deliberately downweighted facial to 0.15** (roughly half its formula weight) precisely because we cannot validate the upstream pipeline. The AUC number describes discrimination on UFNet's test split; our confidence in transferring that number to a live demo user's webcam-recorded smile is meaningfully lower than our confidence in transferring phonation and DDK numbers (which are extracted by our own Parselmouth pipeline from raw waveforms we control end-to-end).

**Practical implication for the demo user:** Even though the OpenFace Docker call at demo time uses the same image UFNet trained on, the *recording conditions* almost certainly differ:

- The user's laptop webcam is not the same camera UFNet participants used
- Home lighting is not UFNet's clinical lighting
- The user's smile ×3 execution may or may not match how UFNet participants were coached

Every one of these introduces a shift in the OpenFace feature distribution that we cannot quantify. The facial score at demo time should be read as **weak supporting evidence**, not as strong as the AUROC 0.812 headline number would naively suggest.

## 9. Facial extraction dependency

Facial features are extracted via OpenFace 2.0 running in Docker (`algebr/openface:latest`). This means:

- Docker Desktop must be installed on the user's machine to run the facial channel.
- On Apple Silicon, OpenFace runs via Rosetta 2 (`--platform linux/amd64`). Runtime is acceptable but not native.
- If Docker cannot be installed (e.g. institutional restrictions), the facial channel is unavailable and the pipeline gracefully returns audio-only fusion.

## 10. Facial-classifier feature set is a proper subset of UFNet's

We use 14 features (7 AUs × mean + variance, active-frame-only aggregation per Islam et al. 2023), a proper subset of UFNet's 42-feature original set. Dropped from the original set:

- **MediaPipe geometric signals** (eye-open, mouth-width, jaw-open, etc., 21 features across mean/var/entropy). Reason: Islam 2023 does not publish the landmark indices used to compute them; we cannot faithfully reproduce them.
- **Entropy statistic** across all base signals (14 features). Reason: histogram binning, log base, and range are unpublished.

The ablation cost of these reductions is small but real: full 42-feature test AUROC 0.837 → 14-feature 0.812 (Δ = −0.025). We accept this cost for reproducibility. Downstream implication: our facial score is slightly less discriminative than a hypothetically-full UFNet implementation, but is exactly reproducible from published methods.

## 11. Sex encoding in NeuroVoz is undocumented

NeuroVoz raw metadata encodes subject sex as 0/1 with **no legend published** in the paper or the Zenodo record. Sex is a known covariate for phonation (F0 differs ~1 octave between adult male and female speakers; jitter and shimmer differ subtly). We can report the raw balance in our cohort (HC: 24/22; PD: 20/29) but cannot audit the direction of any sex-related bias without a legend. This is a limitation of the source data, not our pipeline. We have flagged it for future work.

## 12. Not validated on any external audio corpus

The AUC numbers in [METHODOLOGY §4](METHODOLOGY.md#4-model-training) are LOSO on NeuroVoz. They have **not** been validated on any external PD audio corpus. Merging external corpora into training would create dataset-bias shortcuts at this small N (deliberate design decision, see METHODOLOGY §7 and §8). If a suitable cross-lingual corpus becomes available (e.g. ParkCeleb, application pending), we would use it as a robustness check only, never merged into training.

## 13. Report language is generated

Claude-generated reports use hedged, clinically-informed language and cite feature values with correct units. The prompt is engineered to avoid diagnostic phrasing ("indicates Parkinson's" is on the blacklist; "consistent with", "may suggest" are on the whitelist). Nonetheless, Claude output is generative — no LLM is perfectly reliable, and users should treat the language as advisory rather than authoritative. The structured scores in the report (per-channel probabilities, fused probability, agreement flags) are what the underlying models produced; the narrative is Claude's synthesis.

Reports are not persisted server-side. Uploaded audio and video are processed locally; no data leaves the user's machine except the Claude API call, which sends only the structured feature summary (numeric values), not the raw audio or video.

---

## What ParkScreen is validated to do, in one sentence

Estimate a screening probability of Parkinson's disease in **adults over 50** who record **task-matched** sustained vowels, PATAKA DDK, and a smile-task video, given a training distribution of **mild-to-moderate, medicated Spanish-speaking PD subjects** vs age-matched healthy controls, with a bootstrap 95% CI of ±0.10 on the underlying AUC of 0.758.

## What ParkScreen is NOT validated to do

- Diagnose Parkinson's disease
- Distinguish PD from other movement disorders (e.g. essential tremor, atypical parkinsonism)
- Screen unmedicated / OFF-state PD subjects
- Screen severe (H-Y stage 4–5) PD subjects
- Screen children or younger adults (< 50)
- Score inputs that are not task-matched (spontaneous speech, off-task recordings)
- Provide language-generation reports in languages other than English

Always consult a qualified movement disorder specialist for diagnosis.
