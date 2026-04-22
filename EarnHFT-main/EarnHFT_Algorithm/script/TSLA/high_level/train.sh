#!/bin/bash
STOCK="TSLA"
MAX_HOLD=1.5
FEE=0.0001
mkdir -p log/high_level/${STOCK}
nohup python RL/agent/high_level/dqn_position.py \
    --train_data_path data/${STOCK}/train.feather \
    --dataset_name ${STOCK} \
    --max_holding_number ${MAX_HOLD} \
    --transcation_cost ${FEE} \
    --num_sample 100 \
    --result_path result_risk \
    >log/high_level/${STOCK}/train.log 2>&1 &
wait
echo "High-level training complete for ${STOCK}"
