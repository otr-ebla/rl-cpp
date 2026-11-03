#!/bin/bash
# Ablation of the scale factor of the multi-scale maps: the IPPO policy of
# compare_sa.sh (experiments/ippo_sb3_v2/checkpoint_best, trained with
# scale_factor 4) evaluated at scale_factor 2, 3 and 4. Same network and
# weights, the scales only cover another area (input_size * scale_factor ** n
# * meters_per_pixel per side: 1.2 / 2.4 / 4.8 / 9.6 m with 2,
# 1.2 / 3.6 / 10.8 / 32.4 m with 3, 1.2 / 4.8 / 19.2 / 76.8 m with 4).
# Without flip_when_stuck. With scale_factor 3 the canvas must be a multiple
# of 27: global_size is set to 27 (the global maps are only seen by the
# critic, not evaluated here).
# Results in results/compare_scale: dashboard_compare.pdf and the paired tests
# against scale_factor 4 (stats.txt).
#
#   systemd-run --user --unit=marl-compare-scale --working-directory=$PWD pipelines/compare_scale.sh
#   pipelines/compare_scale.sh test_small     # some map sets only
source "$(dirname "$0")/common.sh"
ROOT=${ROOT:-results/compare_scale}
IPPO=${IPPO:-experiments/ippo_sb3_v2/checkpoint_best}
[ -d "$IPPO" ] || { echo "no IPPO checkpoint $IPPO (pipelines/train_from_mappo.sh ippo_sb3_v2)"; exit 1; }

RUNS=(
  --run "IPPO scale 2=$IPPO,scale_factor=2"
  --run "IPPO scale 3=$IPPO,scale_factor=3,global_size=27"
  --run "IPPO scale 4=$IPPO,scale_factor=4"
)

collect_sets "$@"
$PY -m scripts.analysis.dashboard_compare --data "$ROOT" --reference 'IPPO scale 4' \
  --methods 'IPPO scale 2' 'IPPO scale 3' 'IPPO scale 4' \
  --title 'IPPO: scale factor of the multi-scale maps (2, 3, 4)'
$PY -m scripts.analysis.paired_stats --data "$ROOT" --pair 'IPPO scale 4:IPPO scale 2' --pair 'IPPO scale 4:IPPO scale 3' \
  > "$ROOT/stats.txt" || true
echo "$(date '+%F %T') done: $ROOT/dashboard_compare.pdf, $ROOT/stats.txt"
