#!/bin/sh
# Regenerates every number and figure of the MethodsX article from the CORA deposit placed in data/.
# The extensive run (simulate_extensive_run.py and the two figures that read it) took about two hours on one core;
# pass --skip-extensive to omit them.
set -eu
cd "$(dirname "$0")/scripts"
python build_conditional_tables.py
python simulate_matched_run.py
python compute_validation_table.py
python compute_first_round_table.py
python make_conditional_prob_heatmaps.py
if [ "${1:-}" != "--skip-extensive" ]; then
  python simulate_extensive_run.py
  python make_validation_overlays.py
  python make_EHE_characterisation.py
fi
python ../tools/verify_reproduction.py "${1:-}"
