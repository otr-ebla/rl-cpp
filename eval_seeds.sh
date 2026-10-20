#!/bin/bash
# Evaluate the methods trained by train_seeds.py on the three map sets and
# draw results/eval/dashboard.pdf (see collect_marl.py and dashboard_marl.py).
#   test_small  100 held-out random maps of the training sizes (2.4-7.5 m)
#   test_large  100 held-out random maps, 12-24 m (make_test_maps.py)
#   paper       the 15 mowing and 23 exploration evaluation maps of the paper (4.8-36 m)
# Large maps take long to cover, so they use a 95 % goal and more steps.
set -e
cd "$(dirname "$0")"
OUT=${OUT:-results/eval}
RUNS=(
  --run 'MAPPO=experiments/seeds/mappo_s*/checkpoint'
  --run 'IPPO=experiments/seeds/ippo_s*/checkpoint'
  --run 'MAPPO 5 m=experiments/seeds/mappo_comm5_s*/checkpoint'
  --run 'MAPPO tested at 5 m=experiments/seeds/mappo_s*/checkpoint,comm_range=5'
)
python collect_marl.py --out "$OUT" --no-dashboard --maps 'maps/test_small_*.png' \
  --num_agents 1 2 4 8 --goal_coverage 0.99 --max_steps 8000 --env_batch 100 "${RUNS[@]}"
python collect_marl.py --out "$OUT" --no-dashboard --maps 'maps/test_large_*.png' \
  --num_agents 2 4 8 --goal_coverage 0.95 --max_steps 30000 --env_batch 100 "${RUNS[@]}"
python collect_marl.py --out "$OUT" --no-dashboard --map_set paper \
  --maps 'maps/eval_mowing_*.png' 'maps/eval_exploration_*.png' \
  --num_agents 2 4 8 --goal_coverage 0.95 --max_steps 40000 --env_batch 38 "${RUNS[@]}"
python dashboard_marl.py --data "$OUT"
