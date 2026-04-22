#!/bin/bash
# Split data for all US stock datasets: NVDA, TSLA, WMT
# Usage: bash data/split.sh

mkdir -p log/data/split

# NVDA (bull market)
nohup python data/split_data.py --data_path data/NVDA --chunk_length 14400 --future_sight 3600 \
    >log/data/split/NVDA.log 2>&1 &

# TSLA (bear market)
nohup python data/split_data.py --data_path data/TSLA --chunk_length 14400 --future_sight 3600 \
    >log/data/split/TSLA.log 2>&1 &

# WMT (sideways market)
nohup python data/split_data.py --data_path data/WMT --chunk_length 14400 --future_sight 3600 \
    >log/data/split/WMT.log 2>&1 &

wait
echo "Data split complete for NVDA, TSLA, WMT"
