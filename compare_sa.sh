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
# the extra training from the effect of coordination. ONLY_FINETUNED=1 only
# collects that method (the other rows of the data are kept).
#
#   systemd-run --user --unit=marl-compare-sa --working-directory=$PWD ./compare_sa.sh
#   ONLY_FINETUNED=1 ./compare_sa.sh
set -e
cd "$(dirname "$0")"
ROOT=${ROOT:-results/compare_sa}
PY=.venv-marl/bin/python
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
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
if [ -d "$SA_FT" ]; then
  FT_RUN=(--run "Single-agent fine-tuned xN + flip=$SA_FT,flip_when_stuck=true")
  if [ -n "$ONLY_FINETUNED" ]; then RUNS=("${FT_RUN[@]}"); else RUNS+=("${FT_RUN[@]}"); fi
fi
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
