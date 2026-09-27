set -e
# Step 4: decision stage and France assembly.
# Usage: build_from_pred.sh <pred.parquet> <tag>
# Runs predict_submission.write_outputs(0.85, pred, (sibling, additions, expected)), applies the France audit sets and rescue additions,
# then swaps in the France re-run rows. Writes output/<tag>/matching_results_<tag>.tsv and candidate_pairs_<tag>.tsv under REPO_DIR and validates them.
HERE="$(cd "$(dirname "$0")" && pwd)"
FS="$(cd "$HERE/.." && pwd)"
WORK="$(cd "${WORK_DIR:-work}" && pwd)"
REPO="$(cd "${REPO_DIR:-.}" && pwd)"
export WORK_DIR="$WORK" REPO_DIR="$REPO"
P="$WORK"
PY="${PYTHON:-python}"
PRED=$1; TAG=$2
J=$P/france_audit/judge
cd $REPO/src && $PY -c "
from pathlib import Path
import predict_submission as ps
ps.write_outputs(0.85, Path('$PRED'), ('sibling', 'additions', 'expected'))
"
cd $REPO && mkdir -p output/$TAG
mv output/matching_results.tsv output/$TAG/b_m.tsv && mv output/candidate_pairs.tsv output/$TAG/b_c.tsv
$PY "$FS/assemble/apply_changes.py" output/$TAG/b_m.tsv output/$TAG/b_c.tsv output/$TAG/s_m.tsv output/$TAG/s_c.tsv $J/judge_keep_removals.parquet $J/judge_keep_adds.parquet,$J/judge_n2_adds.parquet,$P/rescue/test_rescue_adds.parquet
$PY "$FS/assemble/fr_swap_sets.py" output/$TAG/s_m.tsv $P/fr_rerun/out/france_assign_partial.parquet $P/${TAG}_rem.parquet $P/${TAG}_add.parquet
$PY "$FS/assemble/apply_changes.py" output/$TAG/s_m.tsv output/$TAG/s_c.tsv output/$TAG/matching_results_$TAG.tsv output/$TAG/candidate_pairs_$TAG.tsv $P/${TAG}_rem.parquet $P/${TAG}_add.parquet
rm -f output/$TAG/b_m.tsv output/$TAG/b_c.tsv output/$TAG/s_m.tsv output/$TAG/s_c.tsv
$PY student_resource/utils/validate_submission.py --matching output/$TAG/matching_results_$TAG.tsv --candidate output/$TAG/candidate_pairs_$TAG.tsv --test-dir student_resource/dataset/test --check-ids 2>&1 | tail -1
echo BUILD_DONE $TAG
