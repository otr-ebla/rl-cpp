#!/bin/bash
# Independent single-robot baseline: the single-agent policy (experiments/sb3_init,
# see init_from_sb3.py) fine-tuned among other moving robots, without any
# communication (no_communication): each robot only knows its own coverage and
# lidar detections, gets no poses of the others and is rewarded for its own
# coverage only (team_reward_weight 0). The other robots are obstacles, seen by
# the lidar and in the obstacle map (robot_lidar_obstacles), with the robot
# collision penalty. All robots run the policy being trained (shared parameters,
# own experience each), with the local critic of IPPO, and the other settings
# and the budget of MAPPO (experiments/mappo_sb3_v2), as train_ippo_sb3.sh.
#
#   systemd-run --user --unit=marl-sa-indep --working-directory=$PWD ./train_sa_indep.sh
set -eo pipefail
cd "$(dirname "$0")"
PY=.venv-marl/bin/python
MAPPO_RUN=${MAPPO_RUN:-experiments/mappo_sb3_v2}
INIT=${INIT:-experiments/sb3_init/checkpoint}
OUT=${OUT:-experiments/sa_indep}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$PWD/.jax_cache}

[ -f "$MAPPO_RUN/config.json" ] || { echo "no $MAPPO_RUN/config.json"; exit 1; }
[ -f "$INIT/params.msgpack" ] || { echo "no initial checkpoint $INIT (see init_from_sb3.py)"; exit 1; }
STOP_UPDATE=${STOP_UPDATE:-$($PY -c "import json; print(json.load(open('$MAPPO_RUN/checkpoint/config.json')).get('update', 0))")}
mkdir -p "$OUT"
echo "$(date '+%F %T') independent single robots with the settings of $MAPPO_RUN, from $INIT, into $OUT, stop at update $STOP_UPDATE"
$PY train_marl.py --config_from "$MAPPO_RUN" --algo ippo --stop_update "$STOP_UPDATE" \
  --no_communication --robot_lidar_obstacles --team_reward_weight 0 \
  --checkpoint "$INIT" --logdir "$OUT" "$@" \
  2>&1 | tee "$OUT/train.log"
echo "$(date '+%F %T') done: $OUT/checkpoint_best"
