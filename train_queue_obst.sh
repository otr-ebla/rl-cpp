#!/bin/bash
# Queue after train_sa_indep.sh (unit marl-sa-indep): MAPPO and IPPO retrained
# exactly as experiments/mappo_sb3_v2 and ippo_sb3_v2 (same settings, budget
# and initialization from experiments/sb3_init), only with
# robot_lidar_obstacles as the independent single robots, into
# experiments/mappo_obst and ippo_obst, one after the other on the GPU. Then
# the comparison of the new methods (ONLY=new ./compare_obst.sh).
#
#   systemd-run --user --unit=marl-queue-obst --working-directory=$PWD ./train_queue_obst.sh
set -eo pipefail
cd "$(dirname "$0")"
PY=.venv-marl/bin/python
MAPPO_RUN=experiments/mappo_sb3_v2
INIT=experiments/sb3_init/checkpoint
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$PWD/.jax_cache}

echo "$(date '+%F %T') waiting for marl-sa-indep"
while systemctl --user is-active --quiet marl-sa-indep; do sleep 300; done
STOP_UPDATE=$($PY -c "import json; print(json.load(open('$MAPPO_RUN/checkpoint/config.json')).get('update', 0))")

OUT=experiments/mappo_obst
mkdir -p "$OUT"
echo "$(date '+%F %T') MAPPO with robot_lidar_obstacles into $OUT, stop at update $STOP_UPDATE"
$PY train_marl.py --config_from "$MAPPO_RUN" --algo mappo --stop_update "$STOP_UPDATE" \
  --robot_lidar_obstacles --checkpoint "$INIT" --logdir "$OUT" 2>&1 | tee "$OUT/train.log"

echo "$(date '+%F %T') IPPO with robot_lidar_obstacles into experiments/ippo_obst"
OUT=experiments/ippo_obst ./train_ippo_sb3.sh --robot_lidar_obstacles

echo "$(date '+%F %T') comparison of the new methods"
ONLY=new ./compare_obst.sh
