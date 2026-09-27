"""Runs the France audit classifiers over candidate pairs and streams the results to parquet.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.
"""
import sys, pyarrow as pa, pyarrow.parquet as pq
from q import *
from jcls import name_class, addr_class
REG = r"\b(hauts de france|nouvelle aquitaine|pays de la loire|loire atlantique|pas de calais|gironde|nord|france)\b"
def AK(col, fr):
    """Return the SQL expression for the normalized address key of a column."""
    base = f"coalesce({col},'')"
    if fr:
        base = f"regexp_replace({base}, '{REG}', ' ', 'g')"
    base = f"regexp_replace({base}, '\\b([0-9]+)(bis|ter|[a-df-z])\\b', '\\1 \\2', 'g')"
    return f"array_to_string(list_sort(list_filter(string_split({base}, ' '), x -> x <> '' and x <> 'ndeg')), ' ')"
def run(c, sql, out):
    """Stream the result of a query to a parquet file."""
    rd = c.execute(sql).fetch_record_batch(200000)
    w = None; n = 0
    for b in rd:
        d = b.to_pydict()
        nc, na, nr, ac, dl = [], [], [], [], []
        for qn, sn, qk, sk in zip(d["qf"], d["sf"], d["qk"], d["sk"]):
            x = name_class(qn, sn); nc.append(x[0]); na.append(x[1]); nr.append(x[2])
            y = addr_class(qk, sk); ac.append(y[0]); dl.append(y[1])
        keep = {k: d[k] for k in d if k not in ("qf", "sf")}
        t = pa.table(keep | {"nc": nc, "nadd": na, "nrm": nr, "ac": ac, "delta": pa.array(dl, pa.int64())})
        if w is None: w = pq.ParquetWriter(out, t.schema)
        w.write_table(t); n += t.num_rows
    w.close(); print(out, n, flush=True)
