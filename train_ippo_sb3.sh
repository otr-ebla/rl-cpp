#!/bin/bash
# IPPO baseline of the MAPPO policy of compare_sa.sh: the same training as
# experiments/mappo_sb3_v2 (all settings read from its config.json, seed
# included) and the same initialization (the single-agent policy converted by
# init_from_sb3.py), only with the local critic V(o_i) of IPPO instead of the
# centralized critic. The critic of the initial checkpoint does not match the
# local critic, it is newly initialized (it is new in init_from_sb3.py anyway).
# MAPPO was stopped before total_steps: IPPO stops at the update of its last
# checkpoint (STOP_UPDATE, read from $MAPPO_RUN/checkpoint), with the same
# learning rate schedule, so both have the same budget and "(last)" checkpoint.
# Training, rollouts and updates run entirely on the GPU (JAX).
#
#   systemd-run --user --unit=marl-ippo --working-directory=$PWD ./train_ippo_sb3.sh
#   MAPPO_RUN=experiments/mappo_sb3 OUT=experiments/ippo_sb3 ./train_ippo_sb3.sh
#
# Any further arguments are passed to train_marl.py (they override the copied
# settings), then compare with: ONLY=ippo ./compare_sa.sh
set -eo pipefail
cd "$(dirname "$0")"
PY=.venv-marl/bin/python
MAPPO_RUN=${MAPPO_RUN:-experiments/mappo_sb3_v2}
INIT=${INIT:-experiments/sb3_init/checkpoint}
OUT=${OUT:-experiments/ippo_sb3_v2}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$PWD/.jax_cache}

[ -f "$MAPPO_RUN/config.json" ] || { echo "no $MAPPO_RUN/config.json"; exit 1; }
[ -f "$INIT/params.msgpack" ] || { echo "no initial checkpoint $INIT (see init_from_sb3.py)"; exit 1; }
STOP_UPDATE=${STOP_UPDATE:-$($PY -c "import json; print(json.load(open('$MAPPO_RUN/checkpoint/config.json')).get('update', 0))")}
mkdir -p "$OUT"
echo "$(date '+%F %T') IPPO with the settings of $MAPPO_RUN, from $INIT, into $OUT, stop at update $STOP_UPDATE"
$PY train_marl.py --config_from "$MAPPO_RUN" --algo ippo --stop_update "$STOP_UPDATE" \
  --checkpoint "$INIT" --logdir "$OUT" "$@" \
  2>&1 | tee "$OUT/train.log"
echo "$(date '+%F %T') done: $OUT/checkpoint_best, compare with: ONLY=ippo ./compare_sa.sh"
