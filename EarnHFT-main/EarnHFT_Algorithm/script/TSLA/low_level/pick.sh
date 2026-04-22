#!/bin/bash
STOCK="TSLA"
python analysis/pick_agent/pick_agent_position.py \
    --root_path result_risk/${STOCK} \
    --save_path result_risk/${STOCK}/potential_model
echo "Agent selection complete for ${STOCK}"
