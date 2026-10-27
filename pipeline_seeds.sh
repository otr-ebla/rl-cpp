#!/bin/bash
# Waits until train_seeds.py has trained every run (on any machine), then
# evaluates the three map sets and draws results/eval/dashboard.pdf.
# By default all on this machine, one map set after the other. With
# REMOTE=<host> (same file system, two GPUs) the map sets run in parallel:
# test_small here, test_large and paper on the two GPUs of the remote host.
# Log: results/eval/pipeline.log
#
#   nohup setsid ./pipeline_seeds.sh > results/eval/pipeline.log 2>&1 &
cd "$(dirname "$0")"
REMOTE=${REMOTE:-}
ROOT=results/eval
DIR=$(pwd)
PY=.venv-marl/bin/python

until $PY train_seeds.py --check > /dev/null; do sleep 600; done
echo "$(date '+%F %T') all runs trained"
$PY train_seeds.py --check | tail -1

mkdir -p $ROOT
if [ -z "$REMOTE" ]; then
  echo "$(date '+%F %T') evaluation started on this machine"
  ./eval_seeds.sh > $ROOT/eval.log 2>&1 && echo "$(date '+%F %T') dashboard: $ROOT/dashboard.pdf" \
    || echo "$(date '+%F %T') evaluation failed, see $ROOT/eval.log"
  exit
fi
for s in test_small test_large paper; do rm -f $ROOT/$s/DONE $ROOT/$s/FAILED; done
./eval_seeds.sh test_small > $ROOT/eval_test_small.log 2>&1 &
ssh -o BatchMode=yes $REMOTE "cd $DIR && \
  CUDA_VISIBLE_DEVICES=0 nohup setsid ./eval_seeds.sh test_large > $ROOT/eval_test_large.log 2>&1 < /dev/null & \
  CUDA_VISIBLE_DEVICES=1 nohup setsid ./eval_seeds.sh paper > $ROOT/eval_paper.log 2>&1 < /dev/null &"
echo "$(date '+%F %T') evaluation started: test_small here, test_large and paper on $REMOTE"

while true; do
  done_sets=0
  for s in test_small test_large paper; do
    if [ -f $ROOT/$s/FAILED ]; then echo "$(date '+%F %T') evaluation of $s failed, see $ROOT/eval_$s.log"; exit 1; fi
    [ -f $ROOT/$s/DONE ] && done_sets=$((done_sets + 1))
  done
  [ $done_sets -eq 3 ] && break
  sleep 300
done
echo "$(date '+%F %T') evaluation done"
$PY dashboard_marl.py --data $ROOT && echo "$(date '+%F %T') dashboard: $ROOT/dashboard.pdf"
