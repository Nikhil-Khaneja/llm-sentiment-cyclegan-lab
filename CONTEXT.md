# Team workflow — llm-sentiment-cyclegan-lab

Read this once at the start, both of you. It covers branching, folder ownership, and the day-to-day commit flow.

## 1. Branches

```
main         ← protected. Always the submittable state. Task branches merge in here via PR.
├── nikhil   ← Nikhil's working branch — commit every run here directly
└── anushka  ← Anushka's working branch — commit every run here directly
```

- You each work on your own branch. Your folders (`member_nikhil/` vs `member_anushka/`) never overlap, so conflicts should be rare — if you hit one outside your own folder, stop and message the other person before resolving it.
- Open a PR from your branch into `main` **once per task** as you finish it (so ~3 PRs each over the lab, not one giant PR at the end). This keeps commit history as a clean evidence trail, which the rubric explicitly grades.
- The final `report/` PDF and any last cross-task fixes also go through a PR into `main`.
- Do not commit directly to `main`.

## 2. First-time setup (each of you, once)

```bash
git clone https://github.com/Nikhil-Khaneja/llm-sentiment-cyclegan-lab.git
cd llm-sentiment-cyclegan-lab
git checkout nikhil     # or: git checkout anushka
git lfs install         # once per machine — needed before committing model weights
```

## 3. Day-to-day flow

```bash
git checkout <your-branch>
git pull origin <your-branch>

# ... do your work in your own member_<name>/ folder ...

git add task1_llm/member_<name>/...
git commit -m "task1: add smoke-test run, 1 epoch"
git push origin <your-branch>
```

Commit **raw logs** after every run, unedited — that's your evidence trail (Section 5 of the spec, Reproducibility & Professional Practice, 5 team marks). Don't wait until a task is "done" to commit.

## 4. Opening a PR into `main`

Once a task folder is complete for you (code + checkpoints + `metrics_report.csv` + `failure_analysis.md` + `results.md`):

```bash
gh pr create --base main --head <your-branch> --title "Task 1: <name>'s LLM" --body "Smoke test + full run, metrics and results.md included."
```

The other person reviews/merges (or self-merge if you've agreed that's fine for this lab) — either way, keep it visible so you both know what landed.

## 5. Who owns what

| Path | Owner |
|---|---|
| `task*/member_nikhil/**` | Nikhil only |
| `task*/member_anushka/**` | Anushka only |
| `task*/data/**` | Shared — whoever adds the download script first, don't duplicate |
| `reproducibility/**` | Both — your own manifest/log subfolder, don't touch the other's |
| `report/**` | Joint — combined report, written together near the end |

Both of you build **all three tasks independently** — task ownership by folder is not task division. (Anushka is starting with Task 1 first; that's just sequencing, not "her task.")

## 6. Hard constraints (from the lab spec — don't break these)

- No `nn.Transformer` / `MultiheadAttention` / prebuilt attention modules in Task 1.
- No pretrained embeddings or pretrained LMs in Task 2.
- No pretrained/foundation models touching Task 3's submitted images; Kaggle submission must be your own CycleGAN's direct output.
- Your model and your teammate's model **cannot be near-identical** in architecture + hyperparameters.
- No hard-coded personal paths or secrets anywhere in the repo — config-driven runs only.
- Core architecture decisions and analysis must be your own understanding — you defend them individually in the viva.

## 7. Canvas submission (individual, per person)

Separate from this repo. Each of you builds your own zip:

```
YourName_DATA266_Lab1.zip
├── Part1/   ← contents of task1_llm/member_<you>/
├── Part2/   ← contents of task2_sentiment/member_<you>/
├── Part3/   ← contents of task3_gan/member_<you>/
└── Report.pdf   ← the same combined team report, with the GitHub repo link in it
```

If the zip is too big for Canvas, upload to Google Drive and share with all three ISAs:
`savitha.vijayarangan@sjsu.edu`, `rishivisweswar.boppana@sjsu.edu`, `shriansh.chari@sjsu.edu`.
