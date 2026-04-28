"""
Remote: dump action.npy + reward.npy for TOP-K (K=10) epochs by valid_sum,
for each stock's high_level (earnhft).  Also dump test action.npy of best-valid baseline.
This lets us try test-time action ensembling locally (majority vote / median).
"""
import os
import json
import numpy as np

ROOT = "/root/EarnHFT-main/EarnHFT_Algorithm/result_risk"
OUT = "/tmp/ensemble_data"
os.makedirs(OUT, exist_ok=True)

STOCKS = ["NVDA", "TSLA", "WMT"]
K = 10

with open("/tmp/full_metrics.json") as f:
    full = json.load(f)

for s in STOCKS:
    info = full[s]["earnhft"]
    eps = []
    for k, v in info["epochs"].items():
        if v["valid_sum"] is not None:
            eps.append((int(k), v["valid_sum"]))
    eps.sort(key=lambda x: -x[1])
    topk = [e for e, _ in eps[:K]]
    meta = {"top_epochs": topk, "rm_test": info.get("rm_median_test")}
    with open(os.path.join(OUT, f"{s}_earnhft_topk_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    base = os.path.join(ROOT, s, "high_level", "seed_12345")
    for ep in topk:
        for phase in ("test", "valid"):
            for kind in ("action", "reward", "model_history", "pure_balance", "require_money"):
                src = os.path.join(base, f"epoch_{ep}", phase, f"{kind}.npy")
                if not os.path.exists(src):
                    continue
                a = np.load(src, allow_pickle=False).astype(np.float64)
                np.save(os.path.join(
                    OUT, f"{s}_earnhft_ep{ep}_{phase}_{kind}.npy"), a)

print("done; files:", len(os.listdir(OUT)))
