# Step 5: second French cross-encoder run (100k pairs) scoring the France pair sets a2, h and b2.
HERE="$(cd "$(dirname "$0")" && pwd)"
FS="$(cd "$HERE/../.." && pwd)"
WORK="$(cd "${WORK_DIR:-work}" && pwd)"
REPO="$(cd "${REPO_DIR:-.}" && pwd)"
export WORK_DIR="$WORK" REPO_DIR="$REPO"
cd $WORK/push/frce
PY="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
echo "start $(date)" > gpu2.log
$PY "$HERE/ce_train_fr.py" 100000 64 2e-5 128 frce 0 ../../ce/ce1.pt > train2.log 2>&1
echo "trained $(date)" >> gpu2.log
$PY "$HERE/ce_infer.py" score_a2.parquet frce.pt sc_a.parquet 256 > inf_a.log 2>&1
echo "a $(date)" >> gpu2.log
$PY "$HERE/ce_infer.py" score_h.parquet frce.pt sc_h.parquet 256 > inf_h.log 2>&1
echo "h $(date)" >> gpu2.log
$PY "$HERE/ce_infer.py" score_b2.parquet frce.pt sc_b.parquet 256 > inf_b.log 2>&1
echo "ALLDONE $(date)" >> gpu2.log
