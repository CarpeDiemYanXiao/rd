#!/bin/bash
STOCK="WMT"
MAX_HOLD=2.0
FEE=0.0001
for beta in "-90.0" "-10.0" "30.0" "100.0"; do
    for epoch in $(seq 1 50); do
        MODEL_PATH="result_risk/${STOCK}/beta_${beta}_risk_bond_0.1/seed_12345/epoch_${epoch}"
        if [ -d "${MODEL_PATH}" ]; then
            for init_action in 0 1 2 3 4; do
                python RL/agent/low_level/test_ddqn.py \
                    --test_path ${MODEL_PATH} \
                    --test_df_path data/${STOCK}/valid \
                    --dataset_name ${STOCK} \
                    --max_holding_number ${MAX_HOLD} \
                    --transcation_cost ${FEE} \
                    --initial_action ${init_action}
            done
        fi
    done
done
echo "Low-level testing complete for ${STOCK}"
