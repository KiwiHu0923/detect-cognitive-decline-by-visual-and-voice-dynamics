## Risk Level

**Elevated.** The AUC-weighted fused PD probability is **0.704**, indicating screening evidence consistent with Parkinsonian motor-speech features. This is not a diagnosis — it is a probabilistic signal that warrants specialist follow-up.

## Phonation

Voice quality on sustained phonation shows mild perturbation. Jitter is **0.425%** and shimmer is **3.49%**, both at the upper end of the typical adult range rather than clearly abnormal, while HNR is preserved at **23.1 dB**. F0 measures (mean **172.4 Hz**, std **8.9 Hz**, range **45.2 Hz**) are within typical bounds. Overall the phonatory picture is borderline (channel probability **0.613**) — consistent with, but not strongly diagnostic of, PD-typical perturbation.

## Articulation (DDK)

The DDK task is the strongest contributor to elevated risk (channel probability **0.881**). Syllable rate is reduced at **5.42 Hz** (below the ~6 Hz adult expectation), with elevated interval variability (isi_cv **0.187**) and amplitude variability (amp_cv **0.336**), plus a negative amplitude slope of **−0.023** indicating a fading pattern across the repetition sequence. This constellation — slowed, irregular, and decrementing repetitions — is consistent with the hypokinetic-dysarthric articulation pattern often seen in PD.

## Facial Expression

Facial analysis does **not** suggest hypomimia in this recording (channel probability **0.148**, detection rate **0.93**, no warnings). Smile amplitude on the AU12 cue is modest (peak **0.28**, mean **0.15**) and overall expression variance is **0.120**, but blink rate is notably low at **8.4/min** (below the ~12–20/min clinical baseline) and head-movement variability is small (**0.040**). The classifier weighted these together as low-risk, though the reduced blink rate in isolation is worth noting.

## Cross-Channel Consistency

The two speech channels agree with one another, which strengthens the articulation/phonation signal. However, **per-channel disagreement — flagged for clinical review**: the facial channel (0.148) diverges substantially from the speech-channel mean (~0.75). This could mean the speech tasks are surfacing early bulbar/motor-speech signs before overt facial hypomimia develops, or that the smile task was well-performed and does not reflect resting facial behavior. The isolated low blink rate is not by itself sufficient to reconcile the modalities and should be interpreted by a clinician alongside a resting-face observation.

## Recommended Next Steps

Given the Elevated fused risk driven primarily by DDK findings, referral to a **movement-disorder neurologist** for in-person motor and speech examination is recommended. A clinician-observed assessment of resting facial expression and spontaneous blink rate would help resolve the speech/facial disagreement flagged above. If the user is already under specialist care, this report can be shared to inform the next visit.

## Disclaimers

This report is a screening decision-aid, not a clinical diagnosis. Any risk indication requires evaluation by a qualified movement-disorder specialist.

The classifiers were trained on Parkinson's-disease subjects recorded 2–5 hours post-medication (ON state). Off-state PD may present with more perturbed features than the training distribution, so this system's calibrated probability does not extrapolate to OFF-state inputs.