#!/bin/bash
cd /root/EarnHFT-main/EarnHFT_Algorithm
# Add more K values for robustness; OMP_NUM_THREADS=1 to reduce contention
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for s in NVDA TSLA WMT; do
  for k in 3 5 20; do
    nohup /root/miniconda3/envs/earnhft/bin/python -u /tmp/_remote_ensemble_one.py "$s" "$k" > /tmp/ens_${s}_K${k}.log 2>&1 < /dev/null &
    disown
  done
done
sleep 2
echo "running:"; pgrep -af ensemble_one | wc -l
