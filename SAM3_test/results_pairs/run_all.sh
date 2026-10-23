#!/bin/bash
# order = importance; every (size, variant) is saved as soon as it finishes, so this can be stopped and resumed
cd "$(dirname "$0")/../.."
source .venv/bin/activate; set -a; . ./.env; set +a
P="python -u SAM3_test/collect_box_pairs.py"
$P --sizes 420        --variants orig,71/72,1.0,cell
$P --sizes 420in1008  --variants orig
$P --sizes 700        --variants orig,cell,71/72,1.0
$P --sizes 1008       --variants orig,cell,71/72,1.0
$P --sizes 700in1008  --variants orig
$P --sizes 504        --variants orig,cell,71/72
$P --sizes 336        --variants orig,cell
$P --sizes 560        --variants orig
$P --sizes 840        --variants orig,cell,71/72
$P --sizes 1260       --variants orig
echo ALL_DONE
