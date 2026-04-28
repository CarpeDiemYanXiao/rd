"""Per-stock per-K ensemble inference, ONE stock+K per process invocation.
Args: STOCK K
Output: /tmp/ensemble_v2/STOCK/K{K}/...
"""
import pandas as pd
import torch
import numpy as np
from env.high_level_env import (
    high_level_testing_env, build_model_path_list_dict,
    load_minute_features,
)
from model.net import Qnet_high_level_position
import os
import sys
import json
import time
sys.path.insert(0, "/root/EarnHFT-main/EarnHFT_Algorithm")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

DEVICE = "cpu"
ROOT = "/root/EarnHFT-main/EarnHFT_Algorithm"
RR = os.path.join(ROOT, "result_risk")
DD = os.path.join(ROOT, "data")
OUT_BASE = "/tmp/ensemble_v2"

with open("/tmp/full_metrics.json") as f:
    full = json.load(f)

CFG = {
    "NVDA": {"max_hold": 10, "tc": 0.0},
    "TSLA": {"max_hold": 20, "tc": 0.0},
    "WMT":  {"max_hold": 40, "tc": 0.0},
}
HIDDEN = 128
ACT = 5
BTL = 1


def main():
    stock = sys.argv[1]
    K = int(sys.argv[2])
    out_dir = os.path.join(OUT_BASE, stock, f"K{K}")
    os.makedirs(out_dir, exist_ok=True)

    info = full[stock]["earnhft"]
    rows = [(int(k), v["valid_sum"])
            for k, v in info["epochs"].items() if v["valid_sum"] is not None]
    rows.sort(key=lambda x: -x[1])
    epochs = [e for e, _ in rows[:K]]
    print(f"=== {stock} K={K} epochs={epochs} ===", flush=True)

    n_state = len(load_minute_features(stock))
    mp_dict = build_model_path_list_dict(stock)
    num_model = len(mp_dict[0])
    nets = []
    for ep in epochs:
        net = Qnet_high_level_position(n_state, num_model, HIDDEN).to(DEVICE)
        net.load_state_dict(torch.load(os.path.join(RR, stock, "high_level", "seed_12345",
                                                    f"epoch_{ep}", "trained_model.pkl"),
                                       map_location=DEVICE))
        net.eval()
        nets.append(net)

    test_df = pd.read_feather(os.path.join(DD, stock, "test.feather"))
    cfg = CFG[stock]
    env = high_level_testing_env(
        df=test_df, dataset_name=stock, transcation_cost=cfg["tc"],
        back_time_length=BTL, max_holding_number=cfg["max_hold"],
        action_dim=ACT, early_stop=0, initial_action=0,
        model_path_list_dict=mp_dict,
    )

    s, ino = env.reset()
    actions = []
    rewards = []
    done = False
    n = 0
    t0 = time.time()
    while not done:
        raw = ino["high_level_state"]
        t = torch.tensor(raw, dtype=torch.float32, device=DEVICE)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        pos = torch.tensor([ino["previous_action"]],
                           dtype=torch.float32, device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            q = sum(net(t, pos) for net in nets)
        a = int(torch.argmax(q, dim=1).cpu().numpy()[0])
        s, r, done, ino = env.step(a)
        actions.append(a)
        rewards.append(r)
        n += 1
        if n % 500 == 0:
            print(f"  step {n} t={time.time()-t0:.1f}s", flush=True)

    actions = np.array(actions)
    rewards = np.array(rewards)
    np.save(os.path.join(out_dir, "test_action.npy"), actions)
    np.save(os.path.join(out_dir, "test_reward.npy"), rewards)
    np.save(os.path.join(out_dir, "test_require_money.npy"), env.required_money)
    rm = float(np.mean(np.array(env.required_money).ravel()))
    ret = float(rewards.sum()) / max(abs(rm), 1e-9) * 100.0
    summary = {"K": K, "epochs": epochs, "test_T": int(len(rewards)),
               "test_sum": float(rewards.sum()), "test_rm": rm, "test_ret_pct": ret,
               "elapsed_sec": time.time() - t0}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"  DONE T={len(rewards)} sum={rewards.sum():.4f} rm={rm:.2f} ret={ret:+.4f}%", flush=True)


if __name__ == "__main__":
    main()
