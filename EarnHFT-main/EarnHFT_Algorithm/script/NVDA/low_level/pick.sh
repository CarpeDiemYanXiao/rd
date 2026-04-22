#!/bin/bash
# Pick best agents for NVDA strategy pool (Stage II)

STOCK="NVDA"

python analysis/pick_agent/pick_agent_position.py \
    --root_path result_risk/${STOCK} \
    --save_path result_risk/${STOCK}/potential_model

echo "Agent selection complete for ${STOCK}"
