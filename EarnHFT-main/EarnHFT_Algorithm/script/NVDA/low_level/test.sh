#!/bin/bash
# Low-level agent testing for NVDA
# Test all trained models on validation set with different initial actions

STOCK="NVDA"
MAX_HOLD=0.5
FEE=0.0001

mkdir -p log/low_level_test/${STOCK}

for beta in "-90.0" "-10.0" "30.0" "100.0"; do
    for epoch in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50; do
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
