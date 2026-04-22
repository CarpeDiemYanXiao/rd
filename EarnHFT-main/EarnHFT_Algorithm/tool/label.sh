#!/bin/bash
# Market dynamics labeling for US stock datasets
# Usage: bash tool/label.sh

mkdir -p log/data/split_valid

# NVDA (bull market)
nohup python tool/slice_model.py --data_path data/NVDA/valid.feather \
    >log/data/split_valid/NVDA.log 2>&1 &

# TSLA (bear market)
nohup python tool/slice_model.py --data_path data/TSLA/valid.feather \
    >log/data/split_valid/TSLA.log 2>&1 &

# WMT (sideways market)
nohup python tool/slice_model.py --data_path data/WMT/valid.feather \
    >log/data/split_valid/WMT.log 2>&1 &

wait
echo "Market dynamics labeling complete for NVDA, TSLA, WMT"