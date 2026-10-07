# Failure / Error Analysis — Task 3 — CycleGAN Style Transfer

**Member:** Nikhil (Member B)

## Evidence pack (facts only)
- Sample translations over training: `outputs/cyclegan_nikhil_r9c32_B_20261001-0022/samples/epoch001.png` … `epoch050.png`.
- Loss curves and per-epoch losses: `outputs/cyclegan_nikhil_r9c32_B_20261001-0022/loss_curves.png`, `train_history.csv`. Final-epoch training losses: cycle A 0.0922, cycle B 0.0898, identity A 0.0614, identity B 0.0763, generator adversarial A→B 0.5116 / B→A 0.9311, discriminators D_A 0.0228, D_B 0.1479.
- Stability: 0 non-finite steps; gradient norms are logged in `train_history.csv`.
- Cycle checks: cycle reconstruction L1 0.0477 (A→B) and 0.0551 (B→A); LPIPS between input and reconstruction 0.257 / 0.156.
- Direction asymmetry in the table of `results.md`: A→B has precision 0.753 and recall 0.356; B→A has precision 0.571 and recall 0.623.

## Assessment of visual quality
From `samples/epoch050.png` (rows: Monet input, A→B, reconstruction; photo input, B→A, reconstruction):
- **Photo → Monet (B→A):** colours move to a painterly palette with soft brush-like texture on coastlines, hills and a rock arch; composition is kept. Flat regions such as sky pick up a fine grid or streak pattern.
- **Monet → photo (A→B):** colours and lighting become more photographic and the layout is kept, but brush texture survives in places, so some outputs look like recoloured paintings. In one scene (a building against a hazy sky) the output shows a fire-coloured orange sky that is not in the painting.
- **Reconstructions** match the originals closely in layout and colour.

## Cycle-consistency and training stability
- Cycle L1 (test) is 0.0477 (A→B) and 0.0551 (B→A). Training cycle loss fell from 0.231 / 0.242 in epoch 1 to 0.092 / 0.090 in epoch 50, and identity loss from 0.217 / 0.220 to 0.061 / 0.076, both decreasing steadily with no divergence.
- Mean generator gradient norm fell from 54 to about 29 and then stayed flat; the discriminator's fell from 22 to 5.5. There were 0 non-finite steps, no oscillation and no sign of mode collapse in the sample sheets.
- The adversarial curves are imbalanced: D_A (the Monet discriminator) loss falls to 0.023 and the photo → Monet generator's adversarial loss rises from 0.43 to 0.93. With only 300 paintings the Monet discriminator keeps getting stronger, and Photo → Monet precision (0.571) is the lowest of my four precision/recall values. Anushka's run shows the same pattern.

## Failure cases and shortcomings
I have not named individual failure images; the cases below are read from the epoch-50 sample sheet and the metrics.
1. **Grid/streak texture on flat regions (photo → Monet).** Sky areas get a repeating fine pattern instead of brush strokes. Likely cause: transposed-convolution upsampling plus a 70×70 PatchGAN that rewards any high-frequency texture.
2. **Invented content (Monet → photo).** The fire-coloured sky appears in the translation but not the input. Likely cause: photos have more contrast than hazy paintings, so the generator adds it. Content cosine is 0.698 for this direction, below Anushka's 0.792.
3. **Surviving brush texture (Monet → photo).** Cycle and identity losses penalise removing strokes, so the generator keeps them.
4. **Limited detail overall.** FID improves from about 126 (untranslated) to 88.7 / 89.9, but the 128-px training resolution and 32-channel generators limit sharpness; A→B KID (0.0272) is higher than Anushka's (0.0163), and A→B recall is low (0.356).

## What I would change next
- Train at 256 px to separate resolution from generator width.
- Augment the discriminator inputs so the Monet discriminator cannot memorise 300 paintings.
- Replace transposed convolutions with upsample + convolution.
