# ParkScreen — Hackathon TODO

**Dates:** July 7–13, 2026 | **Track:** Development

Pivoted CogniScreen (dementia) → ParkScreen (PD) on 2026-07-07, then IPVS → NeuroVoz on 2026-07-08. Full pivot rationale in `CLAUDE.md`, decision journal in `docs/METHODOLOGY.md` §5.

---

## Weekly Overview

| Day | Theme | Must-ship | Status |
|-----|-------|-----------|--------|
| Day 1 (Mon 7/7) | Foundation + pivot | Repo + env + Whisper pipeline running; NeuroVoz downloaded; Parselmouth sanity-checked | ✅ Done |
| Day 2 (Tue 7/8) | Dataset swap + feature extraction | IPVS → NeuroVoz swap; Phonation + DDK features for all NeuroVoz samples (age-matched split) | ✅ Done |
| Day 3 (Wed 7/9) | **Ablation (never cut)** | Train 3 models, subject-level LOSO on NeuroVoz, produce `ablation_table.csv` | ✅ Done — headline row AUC 0.758 |
| Day 4 (Thu 7/10) | Facial + Claude layer | OpenFace-derived hypomimia JSON + Claude 3-channel report + `llm_fusion` context builder | ✅ Done — end-to-end validated |
| Day 5 (Fri 7/11) | **End-to-end demo (never cut)** | Gradio app: three task-matched dirs → full report; held-out NeuroVoz PD case runs | 🟡 Pipeline done, held-out NeuroVoz case pending |
| Day 6 (Sat 7/12) | Freeze + polish | README, LIMITATIONS, DATASETS, LICENSE, methodology audit committed | 🔴 In progress |
| Day 7 (Sun 7/13) | Submit | Demo video + pitch deck + submission | 🔴 |

---

## Day 1–3: Foundation, Data, Ablation — ✅ Done

Fully implemented and validated. See:
- `CLAUDE.md` — architecture, feature specs, tech stack
- `docs/METHODOLOGY.md` §2–4 — cohort construction, feature extraction, model training
- `docs/METHODOLOGY.md` §5.1–5.2 — Day 3 findings (per-file phonation, AUC-excess weighting, rejected hypotheses)
- `eval/results/ablation_table.csv` — headline table

Key artifacts:
- 12 phonation features + 8 DDK features extracted for 95-subject analysis cohort
- Subject-level LOSO on 8 model rows; best AUC 0.758 [0.662, 0.859]
- Deployment classifiers at `eval/models/{phonation,ddk}.joblib`
- Coefficients dumped to `eval/results/coefficients.csv`
- Notebook `notebooks/ablation_results.ipynb` renders the ablation table

---

## Day 4 — Facial Path + Claude Layer — ✅ Done

Fully implemented, validated with two demo cases (HC + PD). See:
- `docs/METHODOLOGY.md` §5.3 — steady-window preprocessing (mandatory on demo path)
- `docs/METHODOLOGY.md` §5.4 — vowel filter + paper-weighted aggregation
- `docs/METHODOLOGY.md` §6 — smile classifier external validation
- `CLAUDE.md` → Claude Integration — four-file architecture

Key artifacts:
- `src/vision/summarize.py` — hypomimia narrative JSON (5 markers + detection meta)
- `src/fusion/llm_fusion.py` — `fuse_scores` + `build_claude_context`
- `src/report/claude_client.py` — frozen ~1143-token system prompt with both mandatory disclaimers baked in
- `demo/assets/example_context.xml` + `example_report.md` — canonical Day 4 end-to-end artifacts

---

## Day 5 — End-to-End Demo (NEVER CUT)

### `src/pipeline.py` — ✅ Done

- [x] `run_pipeline(vowel_dir, pataka_dir, smile_dir, call_claude=True) -> dict`. Three task-matched upload dirs. Any dir empty → N/A on that channel + fusion renormalizes.
- [x] Uses `_score_phonation`/`_score_ddk` from `quick_score.py` (one canonical audio scorer); facial uses `predict_and_summarize` for single-Docker-run per clip.
- [x] `--no-claude` / `--out-dir` / `--label PD|HC` flags.
- [x] Bug fixed: `facial_features._run_openface` was calling `-aus` (AU-only); switched to `-aus -pose` so `head_movement_std` is not None. See METHODOLOGY finding #15.
- [x] HC demo smoke test: fused 0.157, Low-risk report with disagreement flag + both disclaimers verbatim.
- [x] PD demo smoke test: fused 0.704, Elevated report. Reproduces Day 4 numbers exactly.

### `demo/app.py` — Gradio UI — ✅ Wired and verified

- [x] Full UI shell + PDF export
- [x] `analyze()` calls real `src/pipeline.run_pipeline` (not MOCK_REPORT)
- [x] `RECORDING_PROTOCOL` panel rendered in an open accordion — runtime-verified via curl at `http://localhost:7860` on 2026-07-11
- [x] N/A channel rendering: `_channel_card` produces "N/A — task not detected" HTML when `status != "ok"`. Path exercised end-to-end by the NeuroVoz held-out demo below (empty smile bucket).
- [x] Launch command in README fixed: `python -m demo.app` (not `python demo/app.py` — the latter fails due to `demo.report_pdf` import path).

### Demo Materials

- [x] Self-recorded HC demo at `data/samples/hc_demo/` (vowel + pataka + smile). Validated.
- [x] Self-recorded PD demo at `data/samples/pd_demo/` — friend with clinically diagnosed PD (real label, not roleplay). Validated.
- [x] **Held-out NeuroVoz PD demo (audio-only)** at `data/samples/neurovoz_holdout_demo/`. Subject PD_109 (H-Y 2, UPDRS 15, disease 12 y, 7 vowel files across I/O/U + 1 PATAKA). Verified as own LOSO-held fold (fusion_per_file_weighted_prob = 0.829). Pipeline run: **phonation 0.955 + DDK 0.848 + facial N/A → fused 0.886 → PD ✓**. Speech channels agree. Full report + XML + JSON at `out/neurovoz_holdout_demo/`.
- [ ] Screenshot each report (HC, PD, NeuroVoz held-out) — deferred to Day 7 demo capture

---

## Day 6 — Freeze + Polish (Jul 12)

### Documentation to write
- [x] `docs/METHODOLOGY.md` — data lineage, features, training, sensitivity analyses, cohort audit (age, sex, severity, misclassification, duration confounder)
- [x] `README.md` at repo root — quick start, how to run demo + ablation, links to docs
- [x] `docs/LIMITATIONS.md` — evaluator-facing caveats (13 sections, plus "what ParkScreen is / isn't validated to do" summary)
- [x] `docs/DATASETS.md` — NeuroVoz, UFNet, Islam 2023, YouTubePD, OpenFace, Claude API citations + license terms + DUA compliance
- [x] `LICENSE` — MIT for code + third-party attribution notes
- [ ] **Verify citation placeholders in `docs/DATASETS.md`** — sections ⚠️-marked need actual author names, DOIs, BibTeX entries from the Zenodo record / AAAI proceedings / arXiv metadata
- [ ] Optional: `CITATION.cff` — how others should cite ParkScreen (nice-to-have, not required)

### Wrap-up
- [ ] Both deliverables verified working end-to-end: `eval/results/ablation_table.csv` + demo pipeline on task-matched clips
- [ ] Feature freeze at 6pm

---

## Day 7 — Submit (Jul 13)

- [ ] Record demo screen-capture: task-matched upload → pipeline → report; then held-out NeuroVoz PD sample → report
- [ ] Write 2–3 paragraph pitch summary — lead with the pivot rationale (motor-speech disorder → motor-speech features; open-access data → reproducible)
- [ ] Final submission

---

## Optional / Stretch (skip if behind)

- **Per-vowel modeling for phonation** — 5 separate LogRegs, per-subject soft vote. Est +45–60 min work; hypothesized +0.02–0.05 phonation AUC (METHODOLOGY §7.6).
- **XGBoost head-to-head with LogReg on per-file phonation** — literature-comparable to NeuroVoz's XGBoost baseline. Only if `configs/model.yaml` can accommodate a per-channel `model_class` field cleanly.
- **ParkCeleb cross-lingual robustness check** — access request pending. If it clears, use as second phonation corpus for cross-lingual sensitivity analysis (never merged into training).
- **In-the-wild YouTube demo clip** — bonus modularity demonstration (facial only; speech N/A due to task mismatch). See CLAUDE.md → Demo Protocol → Optional.
- **PATAKA duration ablation** — refit DDK model without `duration_s` and `n_peaks`, verify AUC holds. Coefficient audit (METHODOLOGY §7.4) argues it will but doesn't prove it.
- **Age-stratified LOSO** — hold out only subjects in a matched age window per fold. Would quantify age-contribution to headline AUC.
