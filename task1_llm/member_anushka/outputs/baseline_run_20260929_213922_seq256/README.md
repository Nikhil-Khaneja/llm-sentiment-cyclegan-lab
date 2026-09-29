# Baseline run 20260929_213922 (sequence length 256, 15 epochs)

This is the first full run, kept so the final seq-512 run can be compared with it. The files here were copied unchanged when that run finished.

- Checkpoint: `../../checkpoints/baseline_seq256_best_model.pt`. When this run was committed (commit `e7e3568`) the file was named `checkpoints/best_model.pt`, which is the name that `metrics_report.csv`, `metrics.json` and `manifest.json` in this folder refer to.
- Raw log: `reproducibility/raw_logs/task1_llm_anushka/full_training_log_20260929_213922.txt`
- Headline results: validation loss 0.6419, validation perplexity 1.900, validation bits per character 0.926, validation top-1 accuracy 79.67%
