# Joint analysis — fact sheet (numbers only, no interpretation)

Use this to write the three "joint analysis" sections of the team report (strengths, weaknesses, limitations, next steps). It lists measured differences between the two members' runs. What they mean is for Nikhil and Anushka to decide.

## Task 1 — character language model
| | Nikhil (2 blocks × 8 heads) | Anushka (6 blocks × 6 heads) |
|---|---|---|
| Parameters | 1.65M | 10.93M (6.6× larger) |
| Context / epochs | 128 chars / 12 | 512 chars / 20 |
| Validation perplexity | 2.326 | 1.760 |
| Validation top-1 accuracy | 73.20% | 82.01% |
| Generalization gap (val − train CE) | 0.0198 | 0.0399 |
| Distinct-3 | 0.921 (temperature 0.9) | 0.778 |
| Repeated 4-gram rate | 0.000 (temperature 0.9), 0.260 (greedy) | 0.102 |
| Max gradient norm | 6.71 | 2.26 |
| Training time / tokens per second | 211 s / 759K | 2,573 s / 399K |
| Peak memory | 541 MB | 14,805 MB |

## Task 2 — sentiment classification (38K test reviews)
| Model | Accuracy | MCC | Brier | ECE | Accuracy 95% CI |
|---|---|---|---|---|---|
| B3 BiLSTM + attention (Nikhil) | 0.9538 | 0.9077 | 0.0354 | 0.0163 | [.9518, .9557] |
| A3 BiGRU (Anushka) | 0.9476 | 0.8953 | 0.0461 | 0.0409 | [.9452, .9497] |
| A2 CNN k=3,4,5 (Anushka) | 0.9426 | 0.8851 | 0.0517 | 0.0472 | [.9404, .9449] |
| B2 CNN k=7 (Nikhil) | 0.9424 | 0.8850 | 0.0430 | 0.0030 | [.9400, .9448] |
| A1 mean-pool baseline (Anushka) | 0.9350 | 0.8701 | 0.0497 | 0.0142 | [.9327, .9374] |
| B1 last-token baseline (Nikhil) | 0.6539 | 0.3078 | 0.2112 | 0.0087 | [.6488, .6588] |

- The B3 and A3 confidence intervals do not overlap; the B2 and A2 intervals overlap almost completely.
- Training time: B3 755 s on an RTX 5090, A3 843 s on a Tesla T4 (different hardware).
- Different settings: Nikhil max length 256, 8 epochs, batch 128/128/64; Anushka max length 200, 10 epochs, batch 64.
- McNemar against each member's own baseline: all experimental models differ significantly (p < 1e-300 for Nikhil's models).

## Task 3 — CycleGAN
| | Nikhil (9 blocks, 32 ch, 128 px) | Anushka (9 blocks, 64 ch, 256 px) |
|---|---|---|
| Total parameters | 11.2M | 28.3M |
| FID A→B / B→A (1,000 held-out photos) | 88.71 / 89.87 | 81.16 / 88.58 |
| FID of untranslated inputs | about 126 | 122.7 |
| KID A→B / B→A | 0.0272 / 0.0175 | 0.0163 / 0.0143 |
| Precision / recall A→B | 0.753 / 0.356 | 0.740 / 0.469 |
| Precision / recall B→A | 0.571 / 0.623 | 0.504 / 0.643 |
| Cycle L1 A→B / B→A | 0.0477 / 0.0551 | 0.0345 / 0.0467 |
| Content cosine A→B / B→A | 0.698 / 0.689 | 0.792 / 0.741 |
| Course-script FID / MiFID (average) | 108.03 / 0.4129 | 96.89 / 0.4097 |
| Kaggle public score | −54.2222 | −48.6479 (team rank 21) |
| Training time / throughput | 27,580 s / 22.7 images/s | 22,507 s / 26.8 images/s |
| Peak training memory | 464 MB | 9,974 MB |
| Non-finite steps | 0 | 0 |

- Leaderboard context (6 Oct 2026, 50 teams): top score −39.78, rank 10 at −46.99, rank 20 at −48.50.
- Human audit: not yet rated by either member.
