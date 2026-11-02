#!/bin/bash
# Single-agent policy replicated on N robots (experiments/sb3_init, see
# init_from_sb3.py) against MAPPO and IPPO, as compare_sa.sh, under new
# evaluation rules for every method:
#   * robot_lidar_obstacles: the lidar detections of the other robots are in
#     the obstacle map of the observation, as rlm.mower_env.MowerEnv adds all
#     lidar detections to the obstacle map (the single-agent policy otherwise
#     only sees the other robots in the lidar)
#   * collision_ends_episode: the first collision (wall, obstacle or robot) of
#     any robot ends the episode, so every episode ends in exactly one of
#     success, collision or timeout (the three rates sum to 1)
#   * no flip_when_stuck (a collision ends the episode before)
# With experiments/sa_indep/checkpoint_best (train_sa_indep.sh) also the
# independent single robots fine-tuned among other robots without communication.
# With experiments/mappo_obst and ippo_obst (train_queue_obst.sh) also MAPPO
# and IPPO retrained with robot_lidar_obstacles ('MAPPO retrained', 'IPPO
# retrained'), the fair comparison with the independent single robots.
# ONLY=indep collects only the independent single robots, ONLY=new the
# independent single robots and the retrained methods (the other rows of the
# data are kept).
# Same three map sets, episodes and seed as compare_sa.sh, results in
# results/compare_obst: dashboard_compare.pdf (time to the goal over the
# successful episodes only) and the paired tests in stats.txt.
#
#   systemd-run --user --unit=marl-compare-obst --working-directory=$PWD ./compare_obst.sh
#   ./compare_obst.sh test_small        # one map set only
set -e
cd "$(dirname "$0")"
ROOT=${ROOT:-results/compare_obst}
PY=.venv-marl/bin/python
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$PWD/.jax_cache}
RULES=robot_lidar_obstacles=true,collision_ends_episode=true,flip_when_stuck=false
RUNS=(
  --run "Single-agent xN=experiments/sb3_init/checkpoint,$RULES"
  --run "MAPPO=experiments/mappo_sb3_v2/checkpoint_best,$RULES"
  --run "IPPO=experiments/ippo_sb3_v2/checkpoint_best,$RULES"
)
INDEP=experiments/sa_indep/checkpoint_best
INDEP_RUN=()
[ -d "$INDEP" ] && INDEP_RUN=(--run "Single-robot independent xN=$INDEP,$RULES")
RETRAINED_RUN=()
for m in MAPPO IPPO; do
  ckpt=experiments/${m,,}_obst/checkpoint_best
  [ -d "$ckpt" ] && RETRAINED_RUN+=(--run "$m retrained=$ckpt,$RULES")
done
case ${ONLY:-all} in
  all) RUNS+=("${INDEP_RUN[@]}" "${RETRAINED_RUN[@]}") ;;
  indep) [ -d "$INDEP" ] || { echo "no $INDEP (train it with ./train_sa_indep.sh)"; exit 1; }
         RUNS=("${INDEP_RUN[@]}") ;;
  new) RUNS=("${INDEP_RUN[@]}" "${RETRAINED_RUN[@]}")
       [ ${#RUNS[@]} -gt 0 ] || { echo "no independent or retrained checkpoints"; exit 1; } ;;
  *) echo "unknown ONLY=$ONLY (all, indep or new)"; exit 1 ;;
esac
COMMON=(--no-dashboard --num_agents 1 2 4 8 --episodes 3)

run_set() {
  local out="$ROOT/$1"
  mkdir -p "$out"
  rm -f "$out/DONE" "$out/FAILED"
  echo "$(date '+%F %T') $1 started"
  case $1 in
    test_small) $PY collect_marl.py --out "$out" "${COMMON[@]}" --maps 'maps/test_small_*.png' \
                  --goal_coverage 0.99 --max_steps 8000 --env_batch 300 "${RUNS[@]}" ;;
    test_large) $PY collect_marl.py --out "$out" "${COMMON[@]}" --maps 'maps/test_large_*.png' \
                  --goal_coverage 0.95 --max_steps 30000 --env_batch 150 "${RUNS[@]}" ;;
    paper)      $PY collect_marl.py --out "$out" "${COMMON[@]}" --map_set paper \
                  --maps 'maps/eval_mowing_*.png' 'maps/eval_exploration_*.png' \
                  --goal_coverage 0.95 --max_steps 40000 --env_batch 114 "${RUNS[@]}" ;;
    *) echo "unknown map set $1"; false ;;
  esac || { date > "$out/FAILED"; exit 1; }
  date > "$out/DONE"
  echo "$(date '+%F %T') $1 done"
}

for s in ${@:-test_small test_large paper}; do run_set "$s"; done
$PY dashboard_compare.py --data "$ROOT" --time_failed exclude \
  --methods 'Single-agent xN' 'Single-robot independent xN' 'IPPO' 'MAPPO' 'IPPO retrained' 'MAPPO retrained' \
  --title 'Single agent xN vs MAPPO vs IPPO: robots in the obstacle map, collisions end the episode' \
  --note 'Every method: lidar detections of robots in the obstacle map, the first collision ends the episode.'
echo "$(date '+%F %T') one page comparison: $ROOT/dashboard_compare.pdf"
PAIRS=(--pair 'Single-agent xN:MAPPO' --pair 'MAPPO:IPPO' --pair 'Single-agent xN:IPPO'
       --pair 'Single-robot independent xN:IPPO' --pair 'Single-robot independent xN:MAPPO'
       --pair 'Single-robot independent xN:IPPO retrained' --pair 'Single-robot independent xN:MAPPO retrained'
       --pair 'IPPO retrained:MAPPO retrained')
$PY compare_sa_stats.py --data "$ROOT" --time_failed exclude "${PAIRS[@]}" > "$ROOT/stats.txt" || true
echo "$(date '+%F %T') paired tests: $ROOT/stats.txt"
