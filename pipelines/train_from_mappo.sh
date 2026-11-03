#!/bin/bash
# Baselines trained exactly as the MAPPO run experiments/mappo_sb3_v2 (all
# settings read from its config.json, seed included, same initialization from
# the single-agent policy converted by scripts/multi_agent/init_from_sb3.py,
# same budget: training stops at the update of MAPPO's last checkpoint, with
# the same learning rate schedule), changing only what the recipe names:
#
#   ippo_sb3_v2  IPPO: local critic V(o_i) instead of the centralized critic
#                (isolates the effect of the centralized critic)
#   sa_indep     independent single robots: no communication (each robot only
#                knows its own coverage and lidar detections, no poses of the
#                others), rewarded for its own coverage only, other robots in
#                the obstacle map; local critic
#   mappo_obst   MAPPO with robot_lidar_obstacles (lidar detections of other
#                robots in the obstacle map), the fair comparison with sa_indep
#   ippo_obst    IPPO with robot_lidar_obstacles
#
# A checkpoint whose critic does not match the configuration (the centralized
# critic of the initial checkpoint for IPPO) only initializes the actor.
# Output in experiments/<recipe>; further arguments go to
# scripts/multi_agent/train.py and override the copied settings.
#
#   systemd-run --user --unit=marl-ippo --working-directory=$PWD pipelines/train_from_mappo.sh ippo_sb3_v2
source "$(dirname "$0")/common.sh"
MAPPO_RUN=${MAPPO_RUN:-experiments/mappo_sb3_v2}
INIT=${INIT:-experiments/sb3_init/checkpoint}

RECIPE=${1:?usage: $0 ippo_sb3_v2|sa_indep|mappo_obst|ippo_obst [train options]}
shift
case $RECIPE in
  ippo_sb3_v2) ARGS=(--algo ippo) ;;
  sa_indep)    ARGS=(--algo ippo --no_communication --robot_lidar_obstacles --team_reward_weight 0) ;;
  mappo_obst)  ARGS=(--algo mappo --robot_lidar_obstacles) ;;
  ippo_obst)   ARGS=(--algo ippo --robot_lidar_obstacles) ;;
  *) echo "unknown recipe $RECIPE (ippo_sb3_v2, sa_indep, mappo_obst or ippo_obst)"; exit 1 ;;
esac
OUT=${OUT:-experiments/$RECIPE}

[ -f "$MAPPO_RUN/config.json" ] || { echo "no $MAPPO_RUN/config.json"; exit 1; }
[ -f "$INIT/params.msgpack" ] || { echo "no initial checkpoint $INIT (see scripts/multi_agent/init_from_sb3.py)"; exit 1; }
STOP_UPDATE=${STOP_UPDATE:-$(last_update "$MAPPO_RUN")}
mkdir -p "$OUT"
echo "$(date '+%F %T') $RECIPE with the settings of $MAPPO_RUN, from $INIT, into $OUT, stop at update $STOP_UPDATE"
$PY -m scripts.multi_agent.train --config_from "$MAPPO_RUN" --stop_update "$STOP_UPDATE" \
  --checkpoint "$INIT" --logdir "$OUT" "${ARGS[@]}" "$@" \
  2>&1 | tee "$OUT/train.log"
echo "$(date '+%F %T') done: $OUT/checkpoint_best"
