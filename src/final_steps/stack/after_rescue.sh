# Step 4 input: scores the rescue candidates (validation and test) with ce1 and ce2.
# WORK_DIR must hold ce/ checkpoints and rescue/ candidate files.
HERE="$(cd "$(dirname "$0")" && pwd)"
FS="$(cd "$HERE/.." && pwd)"
WORK="$(cd "${WORK_DIR:-work}" && pwd)"
REPO="$(cd "${REPO_DIR:-.}" && pwd)"
export WORK_DIR="$WORK" REPO_DIR="$REPO"
cd $WORK/ce
PY="${PYTHON:-python}"
R=../rescue
until grep -q "ALLDONE" test_ce2.log 2>/dev/null; do sleep 20; done
$PY "$HERE/ce_infer.py" $R/valid_rescue.parquet ce1.pt $R/valid_rescue_ce1.parquet 256 > rescue_score.log 2>&1
$PY "$HERE/ce_infer.py" $R/valid_rescue.parquet ce2.pt $R/valid_rescue_ce2.parquet 256 >> rescue_score.log 2>&1
$PY "$HERE/ce_infer.py" $R/test_rescue.parquet ce2.pt $R/test_rescue_ce2.parquet 256 >> rescue_score.log 2>&1
$PY "$HERE/ce_infer.py" $R/test_rescue.parquet ce1.pt $R/test_rescue_ce1.parquet 256 >> rescue_score.log 2>&1
echo ALLDONE >> rescue_score.log
