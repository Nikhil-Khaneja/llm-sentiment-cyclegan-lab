"""Human-audit agreement: mean scores and inter-rater agreement for the 30 blinded samples.

Usage (from repo root, after both raters filled their CSVs):
    python task3_gan/member_nikhil/src/audit_agreement.py task3_gan/member_nikhil/outputs/<run_id>/human_audit
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

audit = Path(sys.argv[1])
r1 = pd.read_csv(audit / "ratings_rater1.csv").set_index("sample_id")
r2 = pd.read_csv(audit / "ratings_rater2.csv").set_index("sample_id")
key = pd.read_csv(audit / "answer_key_do_not_show_raters.csv").set_index("sample_id")
rows = []
for crit in ["style_1to5", "content_1to5", "artifacts_1to5"]:
    a, b = r1[crit].astype(int), r2[crit].astype(int)
    rows.append({"criterion": crit,
                 "mean_rater1": a.mean(), "mean_rater2": b.mean(), "mean_both": (a.mean() + b.mean()) / 2,
                 "mean_B2A": pd.concat([a, b])[key["direction"].reindex(pd.concat([a, b]).index) == "B2A"].mean(),
                 "mean_A2B": pd.concat([a, b])[key["direction"].reindex(pd.concat([a, b]).index) == "A2B"].mean(),
                 "exact_agreement_pct": 100 * (a == b).mean(),
                 "within_1_agreement_pct": 100 * ((a - b).abs() <= 1).mean(),
                 "cohen_kappa_quadratic": cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3, 4, 5])})
out = pd.DataFrame(rows).round(4)
out.to_csv(audit / "audit_agreement.csv", index=False)
print(out.to_string(index=False))
