#!/bin/bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate; set -a; . ./.env; set +a
P="python -u SAM3_test/collect_box_pairs.py"
$P --sizes 420in1008 --variants encpad
$P --sizes 700in1008 --variants encpad
$P --sizes 336in1008 --variants orig,encpad
echo BISECT_DONE
