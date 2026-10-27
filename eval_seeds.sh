#!/bin/bash
# Evaluate the methods trained by train_seeds.py on one map set (or all, one
# after the other) and draw results/eval/dashboard.pdf (see collect_marl.py and
# dashboard_marl.py).
#   test_small  100 held-out random maps of the training sizes (2.4-7.5 m)
#   test_large  100 held-out random maps, 12-24 m (make_test_maps.py)
#   paper       the 15 mowing and 23 exploration evaluation maps of the paper (4.8-36 m)
# Large maps take long to cover, so they use a 95 % goal and more steps.
# Each map set is written to results/eval/<map set>/ (so map sets can run in
# parallel on different GPUs, CUDA_VISIBLE_DEVICES), with a DONE (or FAILED) file at the end.
#
#   ./eval_seeds.sh                 # all map sets, then the dashboard
#   ./eval_seeds.sh test_large      # one map set, no dashboard
set -e
cd "$(dirname "$0")"
ROOT=${ROOT:-results/eval}
RUNS=(
  --run 'MAPPO=experiments/seeds/mappo_s*/checkpoint'
  --run 'IPPO=experiments/seeds/ippo_s*/checkpoint'
  --run 'MAPPO 5 m=experiments/seeds/mappo_comm5_s*/checkpoint'
  --run 'MAPPO tested at 5 m=experiments/seeds/mappo_s*/checkpoint,comm_range=5'
)
PY=.venv-marl/bin/python

run_set() {
  local out="$ROOT/$1"
  mkdir -p "$out"
  rm -f "$out/DONE" "$out/FAILED"
  case $1 in
    test_small) $PY collect_marl.py --out "$out" --no-dashboard --maps 'maps/test_small_*.png' \
                  --num_agents 1 2 4 8 --goal_coverage 0.99 --max_steps 8000 --env_batch 100 "${RUNS[@]}" ;;
    test_large) $PY collect_marl.py --out "$out" --no-dashboard --maps 'maps/test_large_*.png' \
                  --num_agents 2 4 8 --goal_coverage 0.95 --max_steps 30000 --env_batch 100 "${RUNS[@]}" ;;
    paper)      $PY collect_marl.py --out "$out" --no-dashboard --map_set paper \
                  --maps 'maps/eval_mowing_*.png' 'maps/eval_exploration_*.png' \
                  --num_agents 2 4 8 --goal_coverage 0.95 --max_steps 40000 --env_batch 38 "${RUNS[@]}" ;;
    *) echo "unknown map set $1"; false ;;
  esac || { date > "$out/FAILED"; exit 1; }
  date > "$out/DONE"
}

if [ $# -gt 0 ]; then
  for s in "$@"; do run_set "$s"; done
else
  for s in test_small test_large paper; do run_set "$s"; done
  $PY dashboard_marl.py --data "$ROOT"
fi
