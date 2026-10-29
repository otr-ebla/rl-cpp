#!/bin/bash
# IPPO policy of compare_sa.sh (experiments/ippo_sb3_v2/checkpoint_best,
# trained with scale_factor 4) evaluated with the multi-scale maps at
# scale_factor 2, 3 and 4: same network and weights, the scales only cover
# another area (input_size * scale_factor ** n * meters_per_pixel per side:
# 1.2 / 2.4 / 4.8 / 9.6 m with 2, 1.2 / 3.6 / 10.8 / 32.4 m with 3,
# 1.2 / 4.8 / 19.2 / 76.8 m with 4). Same episodes (maps, start poses, seed,
# deterministic actions) and map sets as compare_sa.sh, without
# flip_when_stuck, then draws results/compare_scale/dashboard_compare.pdf and
# the paired tests against scale_factor 4 (stats.txt).
# With scale_factor 3 the canvas must be a multiple of 27: global_size is
# set to 27 (the global maps are only seen by the critic, not evaluated here).
#
#   systemd-run --user --unit=marl-compare-scale --working-directory=$PWD ./compare_scale.sh
#   ./compare_scale.sh test_small     # only some map sets
set -e
cd "$(dirname "$0")"
ROOT=${ROOT:-results/compare_scale}
PY=.venv-marl/bin/python
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$PWD/.jax_cache}
IPPO=${IPPO:-experiments/ippo_sb3_v2/checkpoint_best}
[ -d "$IPPO" ] || { echo "no IPPO checkpoint $IPPO (train it with ./train_ippo_sb3.sh)"; exit 1; }
RUNS=(
  --run "IPPO scale 2=$IPPO,scale_factor=2"
  --run "IPPO scale 3=$IPPO,scale_factor=3,global_size=27"
  --run "IPPO scale 4=$IPPO,scale_factor=4"
)
METHODS=('IPPO scale 2' 'IPPO scale 3' 'IPPO scale 4')
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
$PY dashboard_compare.py --data "$ROOT" --methods "${METHODS[@]}" --reference 'IPPO scale 4' \
  --title 'IPPO: scale factor of the multi-scale maps (2, 3, 4)'
echo "$(date '+%F %T') one page comparison: $ROOT/dashboard_compare.pdf"
$PY compare_sa_stats.py --data "$ROOT" --pair 'IPPO scale 4:IPPO scale 2' --pair 'IPPO scale 4:IPPO scale 3' \
  > "$ROOT/stats.txt" || true
echo "$(date '+%F %T') paired tests: $ROOT/stats.txt"
