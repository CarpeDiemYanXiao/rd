#!/bin/bash
# ============================================================
# Setup script for EarnHFT on cloud server
# Run this FIRST after uploading the project to /data/
# ============================================================

set -e

echo "========================================="
echo "EarnHFT Server Setup"
echo "========================================="

PROJECT_DIR="/data/EarnHFT-main/EarnHFT_Algorithm"
DATA_SRC="/data/data/EarnHFT_Algorithm/data"

# ============================================================
# Step 1: Copy data to the expected directory structure
# ============================================================
echo "[Step 1] Setting up data directories..."

# Create data directories inside the algorithm folder
mkdir -p ${PROJECT_DIR}/data/NVDA
mkdir -p ${PROJECT_DIR}/data/TSLA
mkdir -p ${PROJECT_DIR}/data/WMT
mkdir -p ${PROJECT_DIR}/data/feature/NVDA
mkdir -p ${PROJECT_DIR}/data/feature/TSLA
mkdir -p ${PROJECT_DIR}/data/feature/WMT

# Copy feather data files
echo "  Copying df.feather files..."
cp ${DATA_SRC}/NVDA/df.feather ${PROJECT_DIR}/data/NVDA/df.feather
cp ${DATA_SRC}/TSLA/df.feather ${PROJECT_DIR}/data/TSLA/df.feather
cp ${DATA_SRC}/WMT/df.feather ${PROJECT_DIR}/data/WMT/df.feather

# Copy feature files
echo "  Copying feature files..."
cp ${DATA_SRC}/feature/NVDA/second_feature.npy ${PROJECT_DIR}/data/feature/NVDA/second_feature.npy
cp ${DATA_SRC}/feature/NVDA/minitue_feature.npy ${PROJECT_DIR}/data/feature/NVDA/minitue_feature.npy
cp ${DATA_SRC}/feature/TSLA/second_feature.npy ${PROJECT_DIR}/data/feature/TSLA/second_feature.npy
cp ${DATA_SRC}/feature/TSLA/minitue_feature.npy ${PROJECT_DIR}/data/feature/TSLA/minitue_feature.npy
cp ${DATA_SRC}/feature/WMT/second_feature.npy ${PROJECT_DIR}/data/feature/WMT/second_feature.npy
cp ${DATA_SRC}/feature/WMT/minitue_feature.npy ${PROJECT_DIR}/data/feature/WMT/minitue_feature.npy

echo "[Step 1] Data setup complete."

# ============================================================
# Step 2: Install Python dependencies
# ============================================================
echo "[Step 2] Installing dependencies..."

cd ${PROJECT_DIR}

pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || \
pip install torch==2.0.1

pip install \
    numpy==1.24.3 \
    pandas==1.3.5 \
    gym==0.21.0 \
    scipy==1.11.4 \
    scikit-learn==1.2.2 \
    statsmodels==0.14.0 \
    dtaidistance==2.3.10 \
    fastdtw==0.3.4 \
    tslearn==0.5.3.2 \
    matplotlib==3.7.1 \
    prettytable==3.9.0 \
    PyYAML==6.0 \
    tqdm==4.65.0 \
    tensorboard \
    pyarrow

echo "[Step 2] Dependencies installed."

# ============================================================
# Step 3: Verify GPU
# ============================================================
echo "[Step 3] Verifying GPU..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected! Training will be slow.')
"

# ============================================================
# Step 4: Verify data
# ============================================================
echo "[Step 4] Verifying data..."
python -c "
import numpy as np
import pandas as pd

for stock in ['NVDA', 'TSLA', 'WMT']:
    df = pd.read_feather(f'data/{stock}/df.feather')
    sec = np.load(f'data/feature/{stock}/second_feature.npy')
    minute = np.load(f'data/feature/{stock}/minitue_feature.npy')
    print(f'{stock}: {len(df)} rows, {len(sec)} second features, {len(minute)} minute features')
    
    # Verify all features exist in the dataframe
    missing_sec = [f for f in sec if f not in df.columns]
    missing_min = [f for f in minute if f not in df.columns]
    if missing_sec:
        print(f'  WARNING: Missing second features: {missing_sec[:5]}...')
    if missing_min:
        print(f'  WARNING: Missing minute features: {missing_min[:5]}...')
    else:
        print(f'  All features verified OK')
"

echo ""
echo "========================================="
echo "Setup complete! Ready to train."
echo "Run: cd ${PROJECT_DIR} && bash run_all.sh"
echo "========================================="
