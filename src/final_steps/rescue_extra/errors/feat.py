"""Step 6 (name-key rescue). Experiment/validation script feat.py for this step.

Step context: Step 6 (name-key rescue). Error analysis and rescue families (nm1*, hc, adr); gen_* build candidates, eval_* validate on the held-out split, apply_test writes test additions.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/top50/errors

Outputs:
    {E}/fa_feat.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb, re, numpy as np, pandas as pd
E = f'{WORK_DIR}/top50/errors'
c = duckdb.connect(f'{E}/err.duckdb')
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{E}/tmp'")
d = c.execute("""SELECT f.rid, f.s1, f.ts, f.p, f.src, f.fne, CASE WHEN f.ts=f.s1 THEN 'tp' WHEN f.ts IS NULL THEN 'decoy' ELSE 'wrong' END typ, pr.n_true, pr.tp,
  r.name_full rn, coalesce(r.address,'') ra, r.address_missing ram, s.name_full sn, coalesce(s.address,'') sa, s.country
  FROM fa f JOIN per pr USING (s1) JOIN rtxt r ON r.rid=f.rid JOIN s1txt s ON s.s1=f.s1""").df()
num = re.compile(r'\d+')
ORD = {'first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth','one','two','three','four','five','six','seven','eight','nine','ten',
       'i','ii','iii','iv','v','vi','vii','viii','ix','x','xi','xii','xiii','xiv','xv','xvi','xx','1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th'}
def nums(s):
    """Return the numbers found in a string."""
    return num.findall(s)
rn_ = d.rn.values; sn_ = d.sn.values; ra_ = d.ra.values; sa_ = d.sa.values
num_sub = []; num_miss = []; first_diff = []; ord_diff = []; name_numdiff = []
for rn, sn, ra, sa in zip(rn_, sn_, ra_, sa_):
    rnum, snum = nums(ra), nums(sa)
    ss = set(snum)
    miss = [x for x in rnum if x not in ss]
    num_miss.append(len(miss))
    num_sub.append(int(bool(miss) and all(any(y.startswith(x) or x.startswith(y) for y in snum) for x in miss)))
    rt, st = set(rn.split()), set(sn.split())
    a, b = rt - st, st - rt
    ord_diff.append(int(bool(a & ORD) and bool(b & ORD)))
    name_numdiff.append(int(set(nums(rn)) != set(nums(sn)) and bool(nums(rn)) and bool(nums(sn))))
d['num_miss'] = num_miss; d['num_sub'] = num_sub; d['ord_diff'] = ord_diff; d['name_numdiff'] = name_numdiff
d.drop(columns=[]).to_parquet(f'{E}/fa_feat.parquet')
for col in ['num_miss', 'ord_diff', 'name_numdiff']:
    print(pd.crosstab(np.minimum(d[col], 3), d.typ))
print(pd.crosstab([np.minimum(d.num_miss, 2), d.p >= 0.99], d.typ))
