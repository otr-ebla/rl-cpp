#!/bin/bash
# Single-agent policy of the original repository replicated on N robots
# (experiments/sb3_init, see scripts/multi_agent/init_from_sb3.py) against
# MAPPO and IPPO, on the three map sets of common.sh. Results in
# results/compare_sa: dashboard.pdf, the one page comparison
# dashboard_compare.pdf and the paired tests stats.txt.
#
# The single-agent policy cannot get out of head-on deadlocks with other
# robots, so it is also evaluated with flip_when_stuck (as
# rlm.mower_env.MowerEnv), and MAPPO and IPPO too, for the same rules.
# Optional methods, when their checkpoints exist:
#   * the single-agent policy fine-tuned alone (N = 1) in the MARL simulator
#     with the same budget as MAPPO (experiments/sa_finetune_n1_resume2):
#     separates the effect of the extra training from coordination
#   * IPPO trained exactly as MAPPO (pipelines/train_from_mappo.sh ippo_sb3_v2):
#     isolates the effect of the centralized critic
#
# ONLY selects the methods to collect (rows of the other methods are kept):
# ONLY=all (default), ONLY=finetuned or ONLY=ippo.
#
#   systemd-run --user --unit=marl-compare-sa --working-directory=$PWD pipelines/compare_sa.sh
#   ONLY=ippo pipelines/compare_sa.sh test_small      # some map sets only
source "$(dirname "$0")/common.sh"
ROOT=${ROOT:-results/compare_sa}
ONLY=${ONLY:-all}

SA=experiments/sb3_init/checkpoint
MAPPO=experiments/mappo_sb3_v2
IPPO=experiments/ippo_sb3_v2
SA_FT=experiments/sa_finetune_n1_resume2/checkpoint
RUNS=(
  --run "Single-agent xN=$SA"
  --run "Single-agent xN + flip=$SA,flip_when_stuck=true"
  --run "MAPPO=$MAPPO/checkpoint_best"
  --run "MAPPO + flip=$MAPPO/checkpoint_best,flip_when_stuck=true"
  --run "MAPPO (last)=$MAPPO/checkpoint"
)
FT_RUN=()
[ -d "$SA_FT" ] && FT_RUN=(--run "Single-agent fine-tuned xN + flip=$SA_FT,flip_when_stuck=true")
IPPO_RUN=()
[ -d "$IPPO/checkpoint_best" ] && IPPO_RUN=(
  --run "IPPO=$IPPO/checkpoint_best"
  --run "IPPO + flip=$IPPO/checkpoint_best,flip_when_stuck=true"
  --run "IPPO (last)=$IPPO/checkpoint"
)
case $ONLY in
  all) RUNS+=("${FT_RUN[@]}" "${IPPO_RUN[@]}") ;;
  finetuned) RUNS=("${FT_RUN[@]}") ;;
  ippo) RUNS=("${IPPO_RUN[@]}") ;;
  *) echo "unknown ONLY=$ONLY (all, finetuned or ippo)"; exit 1 ;;
esac
[ ${#RUNS[@]} -gt 0 ] || { echo "no checkpoints for ONLY=$ONLY"; exit 1; }

collect_sets "$@"
$PY -m scripts.analysis.dashboard --data "$ROOT"
$PY -m scripts.analysis.dashboard_compare --data "$ROOT"
$PY -m scripts.analysis.paired_stats --data "$ROOT" \
  --pair 'Single-agent fine-tuned xN + flip:MAPPO + flip' --pair 'MAPPO + flip:IPPO + flip' \
  --pair 'Single-agent fine-tuned xN + flip:IPPO + flip' > "$ROOT/stats.txt" || true
echo "$(date '+%F %T') done: $ROOT/dashboard.pdf, $ROOT/dashboard_compare.pdf, $ROOT/stats.txt"
