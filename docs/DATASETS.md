# ParkScreen — Datasets, Licenses, and Attribution

This document catalogs every dataset and third-party model used by ParkScreen, the terms under which they are used, and how ParkScreen complies with those terms.

---

## 1. NeuroVoz — Spanish PD Voice Corpus

**Role in this project:** primary training and evaluation dataset for the phonation and articulation (DDK) classifiers. All AUC numbers in `METHODOLOGY.md` §4 are computed on NeuroVoz.

### Citation

- **Paper:** Mendes-Laureano, J., Gómez-García, J.A., Guerrero-López, A. *et al*. NeuroVoz: a Castillian Spanish corpus of parkinsonian speech. *Sci Data* **11**, 1367 (2024). https://doi.org/10.1038/s41597-024-04186-zs41597-024-04186-z
- **Data:** Mendes-Laureano, J., Gómez-García, J. A., Guerrero-López, A., Luque-Buzo, E., Arias-Londoño, J. D., Grandas-Pérez, F. J., & Godino Llorente, J. I. (2024). NeuroVoz: a Castillian Spanish corpus of parkinsonian speech (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10777657

### License / Data Use Agreement

NeuroVoz is distributed under a Data Use Agreement (DUA) available on the Zenodo record page. Our compliance:

- **DUA cleared 2026-07-08** by the maintainer. Access permission on record.
- **We do not redistribute NeuroVoz data.** The raw audio, metadata, transcriptions, and pre-extracted feature CSVs are gitignored (`.gitignore` excludes `data/raw/neurovoz/`). Anyone reproducing our numbers must obtain NeuroVoz directly from Zenodo and accept the DUA independently.
- **We do not merge NeuroVoz with external corpora at training time.** Cross-corpus training would create dataset-bias shortcuts at small N (see `METHODOLOGY.md` §4 rationale). If a suitable cross-lingual corpus becomes available in the future (e.g. ParkCeleb), it would be used as a robustness check only.
- **We do not attempt subject re-identification** or link NeuroVoz recordings to external identifiers.
- **We report all extracted features and derived scores in aggregate.** Individual-subject scores are only computed at demo time for user-uploaded input, never for NeuroVoz subjects outside the aggregated evaluation.

### What we used from NeuroVoz

- Sustained-vowel WAV files (task codes A, E, I, O, U with reps 1–3) — used only /i/, /o/, /u/. Files at `data/raw/neurovoz/data/audios/`.
- PATAKA WAV files — the canonical clinical DDK task. Files at `data/raw/neurovoz/data/audios/`.
- Subject metadata (`metadata_pd.csv`, `metadata_hc.csv`) — for age filter, cohort construction, and downstream clinical labels (UPDRS, Hoehn-Yahr) surfaced in Claude reports.
- **Not used:** the 20+ Spanish word/sentence tasks + FREE spontaneous speech (dropped 2026-07-09 due to cross-language transfer concerns), the pre-extracted eGeMAPS-style features CSV (used only as a sanity cross-check, not for training), the per-file GRBAS perceptual ratings (candidate for future auxiliary supervised targets, not used in current models).

### Known dataset limitations we surface downstream

- **All PD subjects recorded ON medication** (2–5 h post-dose). Surfaced in every Claude report and in `LIMITATIONS.md` §2.
- **Age imbalance** (PD 5 y older than HC on average). Surfaced in `LIMITATIONS.md` §3.
- **Sex encoded 0/1 without published legend.** Surfaced in `LIMITATIONS.md` §11.

### Attribution in code

Files that consume NeuroVoz data cite the corpus in their module docstrings:
- `src/data/build_labels.py`
- `src/data/build_cohort.py`
- `src/audio/phonation.py`
- `src/audio/ddk.py`
- `eval/ablation.py`

---

## 2. UFNet — ROC-HCI Smile-Task Feature Dataset

**Role in this project:** training and internal validation of the facial (smile-task) PD classifier. The 14-feature classifier score is one of three channels fused by ParkScreen.

### Citation

- **Paper:** Accessible, At-Home Detection of Parkinson's Disease via Multi-task Video Analysis, **AAAI 2025**.
- **Code and data repository:** ROC-HCI, *UFNet*, GitHub: [`https://github.com/ROC-HCI/UFNet`](https://github.com/ROC-HCI/UFNet). MIT License.

**BibTeX placeholder:**

```bibtex
@article{islam2024accessible,
  title={Accessible, At-Home Detection of Parkinson's Disease via Multi-task Video Analysis},
  author={Islam, Md Saiful and Adnan, Tariq and Freyberg, Jan and Lee, Sangwu and Abdelkader, Abdelrahman and Pawlik, Meghan and Schwartz, Cathe and Jaffe, Karen and Schneider, Ruth B and Dorsey, E and others},
  journal={arXiv preprint arXiv:2406.14856},
  year={2024}
}
```

### License

MIT License, per the ROC-HCI GitHub repository.

### What we used from UFNet

- The released feature dataset `facial_dataset.csv` (1,684 rows / 1,361 subjects) — for training our facial classifier. At `data/raw/ufnet_smile/facial_dataset.csv` (gitignored).
- The released external-validation dataset `youtube_PD_features.csv` (251 clips, 58 PD) — for external validation. At `data/raw/ufnet_smile/youtube_PD_features.csv`.
- The participant-split files `splits/{dev,test,calib}.txt` — to reproduce UFNet's paper split.
- The YouTubePD evaluation subset `splits/test_yt_pd.txt` — UFNet's own designated subset of `youtube_PD_features.csv`, used for external validation reporting.
- Their pretrained model files (`.pth` + scaler in `pretrained/`) — retained for reference only, not deployed in ParkScreen.

**Feature subset:** We use 14 of UFNet's 42 features (7 AUs × mean + variance, active-frame-only aggregation per Islam 2023). Dropped subsets and rationale in `METHODOLOGY.md` §6.

### Reproducibility subtleties from UFNet source code

Documented in `METHODOLOGY.md` §6 and `train_smile_pd.py` docstring:
1. **Splits key off the `ID` column, not `Participant_ID`.** (`unimodal_smile_baal.py:136`.)
2. **NaN in the `pd` column is PD=1**, per UFNet's `lambda x: 0 if str(x) in ['no','0'] else 1` rule.
3. **AU aggregation is active-frame-only.** (Islam 2023 §2.)

### Attribution in code

- `src/vision/train_smile_pd.py` — trains our facial classifier on UFNet's CSV. Module docstring cites UFNet.
- `eval/eval_smile_yt_subset.py` — external validation on the UFNet-designated YouTubePD subset.

---

## 3. Islam et al. 2023 — Smile-Task Protocol Source

**Role in this project:** the demo-time smile task protocol (8–12 seconds of smile ×3 alternating with a neutral face) is taken from this paper. The active-frame-only AU aggregation rule (mean and variance computed only on frames where `AU_c == 1`) is also from this paper.

### Citation

- **Paper:** Islam, M. S. et al., *Unmasking Parkinson's Disease with Smile*, **arXiv:2308.02588**, 2023. [`https://arxiv.org/abs/2308.02588`](https://arxiv.org/abs/2308.02588).

**BibTeX:**

```bibtex
@article{islam2023smile,
  author  = {Islam, Md Saiful and et al.},
  title   = {Unmasking Parkinson's Disease with Smile: an AI-enabled Screening Framework},
  journal = {arXiv preprint},
  volume  = {arXiv:2308.02588},
  year    = {2023}
}
```

### Role in ParkScreen

Not a dataset (Islam 2023 released the protocol, not the data). We use:
- Their **smile ×3 task protocol** for the demo-time recording instructions (`docs/LIMITATIONS.md` §6, `demo/app.py` recording panel).
- Their **active-frame-only aggregation** rule for our facial classifier features.

Ablation cost of relying on this paper's reproducible-only methods (dropping MediaPipe landmark features and entropy statistics whose parameters they did not publish): −0.025 test AUROC (0.837 → 0.812). Accepted trade-off for reproducibility.

---

## 4. YouTubePD — External Validation, Informational Only

**Role in this project:** external validation of the facial classifier's wild-transfer AUROC. **Never used for training.** Numbers reported in `METHODOLOGY.md` §6 are informational, not headline metrics.

### Provenance

YouTubePD is a wild-collected dataset of publicly available YouTube clips, released as feature CSV by the UFNet authors as part of their released artifacts. We do not have direct access to the raw video, only the feature CSV that UFNet computed.

### Citation

- **Paper:** Accessible, At-Home Detection of Parkinson's Disease via Multi-task Video Analysis, **AAAI 2025**.
- **Code and data repository:** ROC-HCI, *UFNet*, GitHub: [`https://github.com/ROC-HCI/UFNet`](https://github.com/ROC-HCI/UFNet). MIT License.

**BibTeX placeholder:**

```bibtex
@article{islam2024accessible,
  title={Accessible, At-Home Detection of Parkinson's Disease via Multi-task Video Analysis},
  author={Islam, Md Saiful and Adnan, Tariq and Freyberg, Jan and Lee, Sangwu and Abdelkader, Abdelrahman and Pawlik, Meghan and Schwartz, Cathe and Jaffe, Karen and Schneider, Ruth B and Dorsey, E and others},
  journal={arXiv preprint arXiv:2406.14856},
  year={2024}
}
```

### Ethical notes

- Public-video subjects have **self-reported / inferred** PD status, not clinical labels.
- Numbers on YouTubePD are for cross-domain robustness reporting only. **Never counted toward headline AUC.**

### What we compute on YouTubePD

Two AUROC numbers, both reported side by side:
- Full released CSV (251 clips, 58 PD / 193 HC): AUROC 0.602
- UFNet's designated subset (178 IDs matched, 21 PD / 157 HC): AUROC 0.708

The gap is explained by non-random subset selection (excluded pool is 51% PD vs 12% PD in the retained subset — the UFNet team appears to have dropped low-quality / off-task clips their own model also failed on). Neither number is demo-representative; the demo is task-matched and should sit near the in-distribution 0.812.

---

## 5. Li et al. 2606.19125 — Per-Vowel AUC Table (Cited)

**Role in this project:** the paper's per-vowel Person AUC table (their Table IV, computed on NeuroVoz) is the empirical basis for our vowel-filter decision (keep /i/, /o/, /u/; drop /a/, /e/) and cross-vowel weight scheme (I=0.310, O=0.310, U=0.379 — AUC-excess-over-chance, renormalized).

### Citation

- **Paper:** Li et al., Continuous-Speech Parkinson’s Disease Detection Using Acoustic and Inharmonicity Features, **arXiv:2606.19125**, 2026. `https://arxiv.org/html/2606.19125v1`.

---

## 6. OpenFace 2.0 — Facial AU Extractor

**Role in this project:** the facial AU extraction tool used both for classifier features and for the hypomimia narrative JSON. Same tool as UFNet's training pipeline (zero AU domain gap).

### Citation

- Baltrusaitis, T., Zadeh, A., Lim, Y. C., & Morency, L. P. (2018). **OpenFace 2.0: Facial Behavior Analysis Toolkit.** *IEEE International Conference on Automatic Face and Gesture Recognition* (FG 2018). `https://github.com/TadasBaltrusaitis/OpenFace`.

### License

OpenFace 2.0 is released for **non-commercial research use only** by the original authors (see the OpenFace GitHub `LICENSE.md`). ParkScreen uses OpenFace as a research prototype; commercial deployment would require a separate license from the OpenFace authors.

We use the community Docker image `algebr/openface:latest` (linux/amd64 platform for compatibility with Apple Silicon via Rosetta 2).

---

## 7. Anthropic Claude API

**Role in this project:** the report-generation layer. Feature scores and hypomimia markers are passed as structured context; Claude generates a clinical-style report.

- Model: `claude-opus-4-7` (1M-context Opus 4.7).
- Non-streaming, `max_tokens=2048`, thinking disabled, no sampling parameters (Opus 4.7-specific requirements).
- Frozen system prompt (~1143 tokens) with both mandatory disclaimers baked in.

Terms of use per Anthropic's API terms of service.

No user-uploaded audio or video is sent to Claude — only the numeric feature summary is transmitted.

---

## 8. Attribution Summary

For a project reproducing our numbers or building on our work, the minimum citation footprint is:

1. **NeuroVoz** — for the phonation and DDK numbers. (§1)
2. **UFNet paper + repo (ROC-HCI, AAAI 2025)** — for the facial classifier. (§2)
3. **Islam et al. 2023** — for the smile-task protocol and active-frame-only aggregation methodology. (§3)
4. **OpenFace 2.0** — for the facial AU extraction. (§6)
5. **Anthropic Claude API** — for the report layer. (§7)

Third-party pointers (do not require citation but should be linked in derivative documentation):
- YouTubePD (via UFNet). (§4)
- Li et al. 2606.19125 (for the vowel-weighting decision). (§5)

---

## 9. Ethical Statement

- **Consent scope:** ParkScreen operates on user-uploaded content. The demo assumes the uploader is the subject of the recording, or has obtained the recording subject's consent. The project does not perform verification.
- **Not for children or younger adults.** Model calibration is for ≥ 50 years old.
- **Not a diagnostic tool.** See `LIMITATIONS.md` §1.
- **Data retention:** ParkScreen processes uploads locally. No user-uploaded audio or video is sent to any external service. The only external call is to the Claude API, and only the numeric feature summary is transmitted (see §7).
- **Bias and fairness:** ParkScreen is trained on a small, geographically-narrow (Spain) cohort with an age imbalance and undocumented sex encoding. Cross-cultural, cross-linguistic, and demographic-fairness validation is future work (see `LIMITATIONS.md` §3, §6, §11).
