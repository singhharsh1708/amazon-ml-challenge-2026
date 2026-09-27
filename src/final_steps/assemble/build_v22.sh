set -e
# Step 8: final chain from cross-encoder scores to the submission files.
# Stacks the 5 cross-encoders (drop kbge from V and T for the 4-model v21 build), writes predictions, applies the high-confidence recheck,
# runs build_from_pred.sh, then applies the France push, nm1d, reverse extras and France fill sets and validates the result.
# Environment: REPO_DIR (repository root), WORK_DIR (experiment workspace), PYTHON (interpreter).
HERE="$(cd "$(dirname "$0")" && pwd)"
FS="$(cd "$HERE/.." && pwd)"
WORK="$(cd "${WORK_DIR:-work}" && pwd)"
REPO="$(cd "${REPO_DIR:-.}" && pwd)"
export WORK_DIR="$WORK" REPO_DIR="$REPO"
P="$WORK"
PY="${PYTHON:-python}"
V="valid_ce1.parquet,valid_ce2.parquet,valid_kbase.parquet,valid_klarge.parquet,valid_kbge.parquet"
T="test_ce1.parquet,test_ce2.parquet,test_kbase.parquet,test_klarge.parquet,test_kbge.parquet"
cd $P/ce && $PY "$FS/stack/stack_test2.py" $V $T $P/ce/test_stacked_v22.parquet
cd $P/v11 && $PY "$FS/stack/mkpred2.py" $P/v11/pred_v22.parquet $P/ce/test_stacked_v22.parquet && rm -rf $REPO/data/temp/mkpred
$PY "$FS/recheck/high_apply.py" $P/v11/pred_v22.parquet $P/v11/pred_v22h.parquet
bash "$FS/assemble/build_from_pred.sh" $P/v11/pred_v22h.parquet v22b
cd $REPO && mkdir -p output/v22
$PY "$FS/assemble/apply_changes.py" output/v22b/matching_results_v22b.tsv output/v22b/candidate_pairs_v22b.tsv output/v22/matching_results_v22.tsv output/v22/candidate_pairs_v22.tsv $P/push/gate/removals_trim.parquet $P/push/gate/additions_trim.parquet,$P/top50/gate/nm1d_adds.parquet,$P/top50/extras_adds.parquet,$P/fill/gate/additions_strict.parquet
rm -rf output/v22b
$PY student_resource/utils/validate_submission.py --matching output/v22/matching_results_v22.tsv --candidate output/v22/candidate_pairs_v22.tsv --test-dir student_resource/dataset/test --check-ids 2>&1 | tail -1
echo BUILD_V22_DONE
