#!/bin/bash
# Low-level agent training for TSLA (bear market)
# TSLA ~$200/share, max_holding=1.5, transcation_cost=0.0001

STOCK="TSLA"
MAX_HOLD=1.5
FEE=0.0001

mkdir -p log/low_level/${STOCK}

for beta in -90 -10 30 100; do
    nohup python RL/agent/low_level/ddqn_pes_risk_aware.py \
        --beta ${beta} \
        --train_data_path data/${STOCK}/train \
        --dataset_name ${STOCK} \
        --max_holding_number ${MAX_HOLD} \
        --transcation_cost ${FEE} \
        --num_sample 200 \
        --result_path result_risk \
        >log/low_level/${STOCK}/beta_${beta}.log 2>&1 &
done

wait
echo "Low-level training complete for ${STOCK}"
