#!/bin/bash
date
echo "summaries: $(ls /tmp/ensemble_v2/*/K*/summary.json 2>/dev/null | wc -l)"
echo "running: $(pgrep -af ensemble_one | wc -l)"
for f in /tmp/ens_*.log; do
  last=$(grep -oE 'step [0-9]+ t=[0-9.]+s' "$f" | tail -1)
  echo "$f: $last"
done
