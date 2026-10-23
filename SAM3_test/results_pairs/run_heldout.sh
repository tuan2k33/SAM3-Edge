#!/bin/bash
# held-out validation: images 65-128 (never used for exploration). One process.
cd "$(dirname "$0")/../.."
source .venv/bin/activate; set -a; . ./.env; set +a
python -u SAM3_test/collect_box_pairs.py --skip 64 --nimg 64 --sizes 336,420,560,700,840,1008,1260 \
  --variants "orig,71/72,sineq+71/72,sineq1.2+71/72"
echo HELDOUT_DONE
