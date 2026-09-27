"""Step 5 (France push). Experiment/validation script e_lab.py for this step.

Step context: Step 5 (France push). A French-adapted cross-encoder trained on synthetic French name/address perturbations, used to propose France removals.
"""
import sys
import duckdb
from sklearn.metrics import roc_auc_score
c = duckdb.connect()
for m in sys.argv[1:]:
    r = c.execute(f"select lab, ce from 'labels60.parquet' l join '{m}' s using (rid, s1) where lab in ('same', 'diff')").fetchall()
    y = [int(a == 'same') for a, _ in r]; s = [b for _, b in r]
    acc = sum(int((b > 0) == (a == 'same')) for a, b in r)
    sa = sum(int(b > 0) for a, b in r if a == 'same'); ns = sum(y)
    da = sum(int(b <= 0) for a, b in r if a == 'diff'); nd = len(y) - ns
    print(f"{m}: n={len(r)} acc={acc}/{len(r)}={acc/len(r):.3f} same_ok={sa}/{ns} diff_ok={da}/{nd} auc={roc_auc_score(y, s):.3f}")
