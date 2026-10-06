# Failure / Error Analysis — Task 3 — CycleGAN Style Transfer

**Member:** Nikhil (Member B)

## Evidence pack (facts only)
- Sample translations over training: `outputs/cyclegan_nikhil_r9c32_B_20261001-0022/samples/epoch001.png` … `epoch050.png`.
- Loss curves and per-epoch losses: `outputs/cyclegan_nikhil_r9c32_B_20261001-0022/loss_curves.png`, `train_history.csv`. Final-epoch training losses: cycle A 0.0922, cycle B 0.0898, identity A 0.0614, identity B 0.0763, generator adversarial A→B 0.5116 / B→A 0.9311, discriminators D_A 0.0228, D_B 0.1479.
- Stability: 0 non-finite steps; gradient norms are logged in `train_history.csv`.
- Cycle checks: cycle reconstruction L1 0.0477 (A→B) and 0.0551 (B→A); LPIPS between input and reconstruction 0.257 / 0.156.
- Direction asymmetry in the table of `results.md`: A→B has precision 0.753 and recall 0.356; B→A has precision 0.571 and recall 0.623.
- Human audit sheets (30 blinded samples, two raters): `outputs/cyclegan_nikhil_r9c32_B_20261001-0022/human_audit/`. Ratings and agreement: not yet collected.

## Assessment of visual quality
**[TODO – Nikhil]** Describe what the translations look like (style, content preservation, artifacts), with examples from the sample images.

## Cycle-consistency and training stability
**[TODO – Nikhil]** What do the cycle and identity values and the loss curves show? Any oscillation, mode collapse or other instability?

## Failure cases and shortcomings
**[TODO – Nikhil]** Two or three concrete failure examples (image names) and your explanation.

## Human audit results
**[TODO – Nikhil]** Mean scores and inter-rater agreement once the two raters have finished (`audit_agreement.py`).

**[TODO – Nikhil]** What would you change next?
