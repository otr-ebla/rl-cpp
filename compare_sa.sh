#!/bin/bash
# Single-agent policy of the original repository replicated on N robots
# (experiments/sb3_init, see init_from_sb3.py) against the MAPPO policy, on the
# same episodes (same maps, start poses and seed, deterministic actions) of the
# three map sets of eval_seeds.sh, then draws results/compare_sa/dashboard.pdf.
# The single-agent policy cannot get out of head-on deadlocks with other robots,
# so it is also evaluated with flip_when_stuck (as rlm.mower_env.MowerEnv), and
# MAPPO too, for a comparison under the same rules.
#
# The single-agent policy fine-tuned alone (N = 1) in the MARL simulator with
# the same budget as MAPPO (experiments/sa_finetune_n1, sa_finetune_n1_resume, sa_finetune_n1_resume2) separates the effect of
# the extra training from the effect of coordination.
#
# IPPO trained exactly as MAPPO (same settings and initialization, local
# critic V(o_i) instead of the centralized one, see train_ippo_sb3.sh) in
# experiments/ippo_sb3_v2 isolates the effect of the centralized critic.
#
# ONLY selects the methods to collect (the other rows of the data are kept):
# ONLY=finetuned (single-agent fine-tuned), ONLY=ippo, or ONLY=all (default).
# ONLY_FINETUNED=1 is the same as ONLY=finetuned.
# At the end the one page comparison dashboard_compare.pdf is drawn too
# (dashboard_compare.py), and the paired tests (compare_sa_stats.py).
#
#   systemd-run --user --unit=marl-compare-sa --working-directory=$PWD ./compare_sa.sh
#   ONLY=ippo ./compare_sa.sh
set -e
cd "$(dirname "$0")"
ROOT=${ROOT:-results/compare_sa}
PY=.venv-marl/bin/python
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
# compiled rollouts are kept on disk: map sets, team sizes and methods with the
# same configuration are not compiled again in later runs
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$PWD/.jax_cache}
ONLY=${ONLY:-all}
[ -n "$ONLY_FINETUNED" ] && ONLY=finetuned
SA=experiments/sb3_init/checkpoint
MAPPO=experiments/mappo_sb3_v2/checkpoint_best
RUNS=(
  --run "Single-agent xN=$SA"
  --run "Single-agent xN + flip=$SA,flip_when_stuck=true"
  --run "MAPPO=$MAPPO"
  --run "MAPPO + flip=$MAPPO,flip_when_stuck=true"
  --run "MAPPO (last)=experiments/mappo_sb3_v2/checkpoint"
)
# resumed at level 4 with max_episode_steps 8000 (1 robot cannot cover the 12 m maps in 2000 steps),
# then from level 8 with 512 environments (out of GPU memory with 1024 when the random maps are regenerated)
# final checkpoint: checkpoint_best stops at the first 100 % success (update 50 of 3700)
SA_FT=experiments/sa_finetune_n1_resume2/checkpoint
FT_RUN=()
[ -d "$SA_FT" ] && FT_RUN=(--run "Single-agent fine-tuned xN + flip=$SA_FT,flip_when_stuck=true")
# IPPO: best and last checkpoint, as for MAPPO
IPPO=${IPPO:-experiments/ippo_sb3_v2}
IPPO_RUN=()
if [ -d "$IPPO/checkpoint_best" ]; then
  IPPO_RUN=(
    --run "IPPO=$IPPO/checkpoint_best"
    --run "IPPO + flip=$IPPO/checkpoint_best,flip_when_stuck=true"
    --run "IPPO (last)=$IPPO/checkpoint"
  )
elif [ "$ONLY" = ippo ]; then
  echo "no IPPO checkpoint in $IPPO (train it with ./train_ippo_sb3.sh)"; exit 1
fi
case $ONLY in
  all) RUNS+=("${FT_RUN[@]}" "${IPPO_RUN[@]}") ;;
  finetuned) RUNS=("${FT_RUN[@]}") ;;
  ippo) RUNS=("${IPPO_RUN[@]}") ;;
  *) echo "unknown ONLY=$ONLY (all, finetuned or ippo)"; exit 1 ;;
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
$PY dashboard_marl.py --data "$ROOT"
echo "$(date '+%F %T') dashboard: $ROOT/dashboard.pdf"
$PY dashboard_compare.py --data "$ROOT"
echo "$(date '+%F %T') one page comparison: $ROOT/dashboard_compare.pdf"
PAIRS=(--pair 'Single-agent fine-tuned xN + flip:MAPPO + flip' --pair 'MAPPO + flip:IPPO + flip'
       --pair 'Single-agent fine-tuned xN + flip:IPPO + flip')
$PY compare_sa_stats.py --data "$ROOT" "${PAIRS[@]}" > "$ROOT/stats.txt" || true
echo "$(date '+%F %T') paired tests: $ROOT/stats.txt"
