#!/bin/bash
STOCK="TSLA"
MAX_HOLD=1.5
FEE=0.0001
for epoch in $(seq 1 100); do
    MODEL_PATH="result_risk/${STOCK}/high_level/seed_12345/epoch_${epoch}"
    if [ -d "${MODEL_PATH}" ]; then
        python RL/agent/high_level/test_dqn_position.py \
            --test_path ${MODEL_PATH} \
            --dataset_name ${STOCK} \
            --valid_data_path data/${STOCK}/valid.feather \
            --test_data_path data/${STOCK}/test.feather \
            --max_holding_number ${MAX_HOLD} \
            --transcation_cost ${FEE}
    fi
done
echo "High-level testing complete for ${STOCK}"
