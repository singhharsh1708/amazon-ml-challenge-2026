# Step 2: waits for ce2 training to finish, then scores the validation and test bands with ce2.
# Run from anywhere; WORK_DIR must hold ce/ (ce2.pt, valid_band.parquet, test_band.parquet).
HERE="$(cd "$(dirname "$0")" && pwd)"
FS="$(cd "$HERE/.." && pwd)"
WORK="$(cd "${WORK_DIR:-work}" && pwd)"
REPO="$(cd "${REPO_DIR:-.}" && pwd)"
export WORK_DIR="$WORK" REPO_DIR="$REPO"
cd $WORK/ce
PY="${PYTHON:-python}"
until grep -q "^done" ce2.log; do sleep 20; done
$PY "$HERE/ce_infer.py" valid_band.parquet ce2.pt valid_ce2.parquet 256 > valid_ce2.log 2>&1
$PY "$HERE/ce_infer.py" test_band.parquet ce2.pt test_ce2.parquet 256 > test_ce2.log 2>&1
echo ALLDONE >> test_ce2.log
