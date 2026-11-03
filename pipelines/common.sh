# Shared setup of the pipelines, sourced by the other scripts (not run directly).
# Runs from the repository root with the MARL environment, keeps compiled JAX
# programs on disk (.jax_cache) so later runs with the same configuration are
# not compiled again.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PY:-.venv-marl/bin/python}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$PWD/.jax_cache}

# Settings of the last update of a run's checkpoint (0 if unknown)
last_update() {
  $PY -c "import json; print(json.load(open('$1/checkpoint/config.json')).get('update', 0))"
}

# The three map sets of every comparison, same episodes (maps, start poses,
# seed, deterministic actions) for every method:
#   test_small  100 held-out random maps of the training sizes (2.4-7.5 m)
#   test_large  100 held-out random maps, 12-24 m (scripts/multi_agent/make_test_maps.py)
#   paper       the 15 mowing and 23 exploration evaluation maps of the paper (4.8-36 m)
# Large maps take long to cover, so they use a 95 % goal and more steps.
# Each map set goes to $ROOT/<map set>/ (runs listed in the RUNS array).
COLLECT_COMMON=(--no-dashboard --num_agents 1 2 4 8 --episodes 3)

collect_set() {
  local out="$ROOT/$1"
  mkdir -p "$out"
  echo "$(date '+%F %T') $1 started"
  case $1 in
    test_small) $PY -m scripts.multi_agent.collect --out "$out" "${COLLECT_COMMON[@]}" --maps 'maps/test_small_*.png' \
                  --goal_coverage 0.99 --max_steps 8000 --env_batch 300 "${RUNS[@]}" ;;
    test_large) $PY -m scripts.multi_agent.collect --out "$out" "${COLLECT_COMMON[@]}" --maps 'maps/test_large_*.png' \
                  --goal_coverage 0.95 --max_steps 30000 --env_batch 150 "${RUNS[@]}" ;;
    paper)      $PY -m scripts.multi_agent.collect --out "$out" "${COLLECT_COMMON[@]}" --map_set paper \
                  --maps 'maps/eval_mowing_*.png' 'maps/eval_exploration_*.png' \
                  --goal_coverage 0.95 --max_steps 40000 --env_batch 114 "${RUNS[@]}" ;;
    *) echo "unknown map set $1 (test_small, test_large or paper)"; return 1 ;;
  esac
  echo "$(date '+%F %T') $1 done"
}

# collect_sets [map set ...]: the given map sets, or all three
collect_sets() {
  local s
  for s in ${@:-test_small test_large paper}; do collect_set "$s"; done
}
