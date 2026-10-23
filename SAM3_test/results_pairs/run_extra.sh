#!/bin/bash
# runs after run_all.sh finished (waits for ALL_DONE in its log)
cd "$(dirname "$0")/../.."
until grep -q ALL_DONE SAM3_test/results_pairs/run_all.log; do sleep 10; done
source .venv/bin/activate; set -a; . ./.env; set +a
P="python -u SAM3_test/collect_box_pairs.py"
$P --sizes 672  --variants orig
$P --sizes 420  --variants ropecell,peinterp
$P --sizes 700  --variants ropecell,peinterp
$P --sizes 1008 --variants ropecell,peinterp
echo EXTRA_DONE
