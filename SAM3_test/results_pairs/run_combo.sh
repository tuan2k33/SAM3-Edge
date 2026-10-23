#!/bin/bash
# one process: decoder patch (sineq<a>) x grid (orig / 71/72), first 64 images. The held-out half is run afterwards.
cd "$(dirname "$0")/../.."
source .venv/bin/activate; set -a; . ./.env; set +a
python -u SAM3_test/collect_box_pairs.py --sizes 336,420,560,700,840,1008,1260 \
  --variants "sineq+71/72,sineq1.2,sineq1.2+71/72,sineq1.4,sineq1.4+71/72"
echo COMBO_DONE
