#!/bin/bash
cd /root/EarnHFT-main/EarnHFT_Algorithm
rm -rf /tmp/ensemble_v2
for s in NVDA TSLA WMT; do
  for k in 1 10; do
    nohup /root/miniconda3/envs/earnhft/bin/python -u /tmp/_remote_ensemble_one.py "$s" "$k" > /tmp/ens_${s}_K${k}.log 2>&1 < /dev/null &
    disown
  done
done
sleep 2
echo "running:"; pgrep -af ensemble_one | wc -l
