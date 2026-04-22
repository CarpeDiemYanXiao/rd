#!/bin/bash
# Low-level agent training for NVDA (bull market)
# NVDA ~$600/share, max_holding=0.5, transcation_cost=0.0001
# Train with 4 different beta values for diverse agent pool

STOCK="NVDA"
MAX_HOLD=0.5
FEE=0.0001

mkdir -p log/low_level/${STOCK}

# Beta = -90 (risk-averse, prefer bear segments)
nohup python RL/agent/low_level/ddqn_pes_risk_aware.py \
    --beta -90 \
    --train_data_path data/${STOCK}/train \
    --dataset_name ${STOCK} \
    --max_holding_number ${MAX_HOLD} \
    --transcation_cost ${FEE} \
    --num_sample 200 \
    --result_path result_risk \
    >log/low_level/${STOCK}/beta_-90.log 2>&1 &

# Beta = -10 (slightly risk-averse)
nohup python RL/agent/low_level/ddqn_pes_risk_aware.py \
    --beta -10 \
    --train_data_path data/${STOCK}/train \
    --dataset_name ${STOCK} \
    --max_holding_number ${MAX_HOLD} \
    --transcation_cost ${FEE} \
    --num_sample 200 \
    --result_path result_risk \
    >log/low_level/${STOCK}/beta_-10.log 2>&1 &

# Beta = 30 (slightly risk-seeking)
nohup python RL/agent/low_level/ddqn_pes_risk_aware.py \
    --beta 30 \
    --train_data_path data/${STOCK}/train \
    --dataset_name ${STOCK} \
    --max_holding_number ${MAX_HOLD} \
    --transcation_cost ${FEE} \
    --num_sample 200 \
    --result_path result_risk \
    >log/low_level/${STOCK}/beta_30.log 2>&1 &

# Beta = 100 (risk-seeking, prefer bull segments)
nohup python RL/agent/low_level/ddqn_pes_risk_aware.py \
    --beta 100 \
    --train_data_path data/${STOCK}/train \
    --dataset_name ${STOCK} \
    --max_holding_number ${MAX_HOLD} \
    --transcation_cost ${FEE} \
    --num_sample 200 \
    --result_path result_risk \
    >log/low_level/${STOCK}/beta_100.log 2>&1 &

wait
echo "Low-level training complete for ${STOCK}"
