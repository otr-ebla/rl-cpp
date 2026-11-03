#!/bin/bash
# Single-agent policy replicated on N robots against MAPPO and IPPO, as
# compare_sa.sh, under stricter evaluation rules for every method:
#   * robot_lidar_obstacles: the lidar detections of the other robots are in
#     the obstacle map of the observation, as rlm.mower_env.MowerEnv adds all
#     lidar detections to the obstacle map (the single-agent policy otherwise
#     only sees the other robots in the lidar)
#   * collision_ends_episode: the first collision (wall, obstacle or robot) of
#     any robot ends the episode, so every episode ends in exactly one of
#     success, collision or timeout (the three rates sum to 1)
#   * no flip_when_stuck (a collision ends the episode before)
# Optional methods, when their checkpoints exist (pipelines/train_from_mappo.sh):
#   * 'Single-robot independent xN' (experiments/sa_indep): fine-tuned among
#     other robots without communication
#   * 'MAPPO retrained', 'IPPO retrained' (experiments/mappo_obst, ippo_obst):
#     retrained with robot_lidar_obstacles, the fair comparison with the
#     independent single robots
# Results in results/compare_obst: dashboard_compare.pdf (time to the goal over
# the successful episodes only) and the paired tests stats.txt.
# ONLY=new collects only the optional methods (rows of the others are kept).
#
#   systemd-run --user --unit=marl-compare-obst --working-directory=$PWD pipelines/compare_obst.sh
#   pipelines/compare_obst.sh test_small        # some map sets only
source "$(dirname "$0")/common.sh"
ROOT=${ROOT:-results/compare_obst}

RULES=robot_lidar_obstacles=true,collision_ends_episode=true,flip_when_stuck=false
RUNS=(
  --run "Single-agent xN=experiments/sb3_init/checkpoint,$RULES"
  --run "MAPPO=experiments/mappo_sb3_v2/checkpoint_best,$RULES"
  --run "IPPO=experiments/ippo_sb3_v2/checkpoint_best,$RULES"
)
NEW_RUNS=()
[ -d experiments/sa_indep/checkpoint_best ] && \
  NEW_RUNS+=(--run "Single-robot independent xN=experiments/sa_indep/checkpoint_best,$RULES")
for m in MAPPO IPPO; do
  ckpt=experiments/${m,,}_obst/checkpoint_best
  [ -d "$ckpt" ] && NEW_RUNS+=(--run "$m retrained=$ckpt,$RULES")
done
case ${ONLY:-all} in
  all) RUNS+=("${NEW_RUNS[@]}") ;;
  new) RUNS=("${NEW_RUNS[@]}")
       [ ${#RUNS[@]} -gt 0 ] || { echo "no independent or retrained checkpoints"; exit 1; } ;;
  *) echo "unknown ONLY=$ONLY (all or new)"; exit 1 ;;
esac

collect_sets "$@"
$PY -m scripts.analysis.dashboard_compare --data "$ROOT" --time_failed exclude \
  --methods 'Single-agent xN' 'Single-robot independent xN' 'IPPO' 'MAPPO' 'IPPO retrained' 'MAPPO retrained' \
  --title 'Single agent xN vs MAPPO vs IPPO: robots in the obstacle map, collisions end the episode' \
  --note 'Every method: lidar detections of robots in the obstacle map, the first collision ends the episode.'
PAIRS=(--pair 'Single-agent xN:MAPPO' --pair 'MAPPO:IPPO' --pair 'Single-agent xN:IPPO'
       --pair 'Single-robot independent xN:IPPO' --pair 'Single-robot independent xN:MAPPO'
       --pair 'Single-robot independent xN:IPPO retrained' --pair 'Single-robot independent xN:MAPPO retrained'
       --pair 'IPPO retrained:MAPPO retrained')
$PY -m scripts.analysis.paired_stats --data "$ROOT" --time_failed exclude "${PAIRS[@]}" > "$ROOT/stats.txt" || true
echo "$(date '+%F %T') done: $ROOT/dashboard_compare.pdf, $ROOT/stats.txt"
