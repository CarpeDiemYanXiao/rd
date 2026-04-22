#!/bin/bash
# ============================================================
# EarnHFT US Stock Complete Pipeline
# Run from the EarnHFT_Algorithm directory
# Usage: bash run_all.sh
# ============================================================

set -e

echo "========================================="
echo "EarnHFT US Stock Training Pipeline"
echo "========================================="
echo "Stocks: NVDA (bull), TSLA (bear), WMT (sideways)"
echo ""

# Create log directories
mkdir -p log/data/split log/data/split_valid
mkdir -p log/low_level/{NVDA,TSLA,WMT}
mkdir -p log/low_level_test/{NVDA,TSLA,WMT}
mkdir -p log/high_level/{NVDA,TSLA,WMT}

# ============================================================
# Step 1: Data Split (60/20/20 + chunking)
# ============================================================
echo "[Step 1/7] Splitting data..."
bash data/split.sh
wait
echo "[Step 1/7] Data split complete."

# ============================================================
# Step 2: Market Dynamics Labeling (valid set)
# ============================================================
echo "[Step 2/7] Market dynamics labeling..."
bash tool/label.sh
wait
echo "[Step 2/7] Labeling complete."

# ============================================================
# Step 3: Low-Level Agent Training (Stage I + II)
# Train sequentially per stock to manage GPU memory
# ============================================================
echo "[Step 3/7] Low-level agent training..."

for STOCK in NVDA TSLA WMT; do
    echo "  Training ${STOCK} low-level agents..."
    bash script/${STOCK}/low_level/train.sh
    wait
done

echo "[Step 3/7] Low-level training complete."

# ============================================================
# Step 4: Low-Level Agent Testing on Validation Set
# ============================================================
echo "[Step 4/7] Low-level agent testing..."

for STOCK in NVDA TSLA WMT; do
    echo "  Testing ${STOCK} low-level agents..."
    bash script/${STOCK}/low_level/test.sh
done

echo "[Step 4/7] Low-level testing complete."

# ============================================================
# Step 5: Agent Selection (Build 5x5 Strategy Pool)
# ============================================================
echo "[Step 5/7] Agent selection (building strategy pool)..."

for STOCK in NVDA TSLA WMT; do
    echo "  Picking best agents for ${STOCK}..."
    bash script/${STOCK}/low_level/pick.sh
done

echo "[Step 5/7] Agent selection complete."

# ============================================================
# Step 6: High-Level Router Training (Stage III)
# ============================================================
echo "[Step 6/7] High-level router training..."

for STOCK in NVDA TSLA WMT; do
    echo "  Training ${STOCK} high-level router..."
    bash script/${STOCK}/high_level/train.sh
    wait
done

echo "[Step 6/7] High-level training complete."

# ============================================================
# Step 7: High-Level Router Testing
# ============================================================
echo "[Step 7/7] High-level router testing..."

for STOCK in NVDA TSLA WMT; do
    echo "  Testing ${STOCK} high-level router..."
    bash script/${STOCK}/high_level/test.sh
done

echo "[Step 7/7] High-level testing complete."

# ============================================================
echo ""
echo "========================================="
echo "EarnHFT Pipeline Complete!"
echo "Results saved in result_risk/{NVDA,TSLA,WMT}/"
echo "========================================="
