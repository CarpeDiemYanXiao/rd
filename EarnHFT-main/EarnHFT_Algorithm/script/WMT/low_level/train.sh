#!/bin/bash
# Low-level agent training for WMT (sideways market)
# WMT ~$160/share, max_holding=2.0, transcation_cost=0.0001

STOCK="WMT"
MAX_HOLD=2.0
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
