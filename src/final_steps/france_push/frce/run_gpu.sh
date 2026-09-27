# Step 5: trains the French cross-encoder (frce) from ce1 and scores the France pair sets a, h and b.
HERE="$(cd "$(dirname "$0")" && pwd)"
FS="$(cd "$HERE/../.." && pwd)"
WORK="$(cd "${WORK_DIR:-work}" && pwd)"
REPO="$(cd "${REPO_DIR:-.}" && pwd)"
export WORK_DIR="$WORK" REPO_DIR="$REPO"
cd $WORK/push/frce
PY="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
until grep -q ALLDONE ../../ce/test_high_ce1.log; do sleep 20; done
echo "gpu free $(date)" > gpu.log
$PY "$HERE/ce_train_fr.py" 148960 64 2e-5 128 frce 0 ../../ce/ce1.pt > train.log 2>&1
echo "trained $(date)" >> gpu.log
$PY "$HERE/ce_infer.py" score_a.parquet frce.pt sc_a.parquet 256 > inf_a.log 2>&1
echo "a $(date)" >> gpu.log
$PY "$HERE/ce_infer.py" score_h.parquet frce.pt sc_h.parquet 256 > inf_h.log 2>&1
echo "h $(date)" >> gpu.log
$PY "$HERE/ce_infer.py" score_b.parquet frce.pt sc_b.parquet 256 > inf_b.log 2>&1
echo "ALLDONE $(date)" >> gpu.log
