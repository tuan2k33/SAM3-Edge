#!/bin/bash
cd "$(dirname "$0")/../.."
until grep -q FRAME_DONE SAM3_test/results_pairs/run_frame.log; do sleep 10; done
source .venv/bin/activate; set -a; . ./.env; set +a
P="python -u SAM3_test/collect_box_pairs.py"
for sz in 336 420 504 560 672 700 840 1260; do $P --sizes $sz --variants sineq; done
$P --sizes 420  --variants sineq0.5,sineq1.5,sineq2
$P --sizes 700  --variants sineq0.5,sineq1.5
$P --sizes 1260 --variants sineq0.5,sineq1.5
echo SINEQ_DONE
