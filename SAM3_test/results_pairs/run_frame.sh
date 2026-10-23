#!/bin/bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate; set -a; . ./.env; set +a
P="python -u SAM3_test/collect_box_pairs.py"
$P --sizes 420 --variants sinekey,sineq,frame72
$P --sizes 336 --variants sinekey,sineq,frame72
$P --sizes 1008 --variants sinekey,sineq,frame72
echo FRAME_DONE
