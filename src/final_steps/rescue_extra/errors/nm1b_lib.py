"""Step 6 (name-key rescue). Experiment/validation script nm1b_lib.py for this step.

Step context: Step 6 (name-key rescue). Error analysis and rescue families (nm1*, hc, adr); gen_* build candidates, eval_* validate on the held-out split, apply_test writes test additions.
"""
import numpy as np
from rapidfuzz import fuzz
def feats(d):
    """Build the gate features for a frame of candidate pairs."""
    qn = [x.split(' | ', 1)[0] for x in d['q_text']]; sn = [x.split(' | ', 1)[0] for x in d['s_text']]
    rid = d['rid']; ce = (d['ce1'] + d['ce2']) / 2
    import pandas as pd
    df = pd.DataFrame({'rid': rid, 'ce': ce})
    g = df.groupby('rid')['ce']
    mx = g.transform('max').values; cnt = g.transform('count').values
    second = df.assign(r=g.rank(ascending=False, method='first')).query('r == 2').set_index('rid')['ce']
    sec = pd.Series(rid).map(second).fillna(-10).values
    other = np.where(ce >= mx, ce - sec, ce - mx)
    return np.column_stack([d['ce1'], d['ce2'], d['jw'], [fuzz.token_set_ratio(a, b) for a, b in zip(qn, sn)],
        [fuzz.ratio(a.replace(' ', ''), b.replace(' ', '')) for a, b in zip(qn, sn)], cnt, other]).astype(np.float32)
