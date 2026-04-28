"""
Remote: scan ALL stocks × ALL methods × ALL epochs.
For each (stock, method, epoch) compute:
  valid_sum = sum(valid/reward.npy)
  test_sum  = sum(test/reward.npy)
  require_money_test = mean(test/require_money.npy)
  valid_T, test_T

Output:
  /tmp/full_metrics.json  -- aggregated per-epoch numbers
  /tmp/best_per_step/<stock>_<method>_test_reward.npy  -- per-step reward of best-valid epoch
  /tmp/best_per_step/<stock>_<method>_test_action.npy  -- actions of that epoch (optional)
  /tmp/best_per_step/<stock>_<method>_meta.json        -- chosen epoch + require_money

Then we tar /tmp/best_per_step and download.
"""
import os
import json
import glob
import numpy as np

ROOT = "/root/EarnHFT-main/EarnHFT_Algorithm/result_risk"
OUT_DIR = "/tmp/best_per_step"
os.makedirs(OUT_DIR, exist_ok=True)

STOCKS = ["NVDA", "TSLA", "WMT"]
# (label, dir_name)
METHODS = [
    ("earnhft", "high_level"),
    ("ppo", "ppo"),
    ("dra", "dra_short"),
    ("dqn", "dqn_ada_256"),
    ("cdqn", "cdqn_rp"),
]

agg = {}


def read_sum(path):
    try:
        a = np.load(path, allow_pickle=False).astype(np.float64).ravel()
        return float(a.sum()), int(a.size)
    except Exception as e:
        return None, 0


def read_rm(path):
    try:
        a = np.load(path, allow_pickle=False).astype(np.float64).ravel()
        return float(np.mean(a)) if a.size else None
    except Exception:
        return None


for stock in STOCKS:
    agg[stock] = {}
    for label, dname in METHODS:
        method_dir = os.path.join(ROOT, stock, dname, "seed_12345")
        if not os.path.isdir(method_dir):
            agg[stock][label] = {"error": f"missing {method_dir}"}
            continue
        epochs = sorted(int(d.replace("epoch_", ""))
                        for d in os.listdir(method_dir) if d.startswith("epoch_"))
        per = {}
        rm_pool = []
        for ep in epochs:
            ed = os.path.join(method_dir, f"epoch_{ep}")
            v_sum, v_T = read_sum(os.path.join(ed, "valid", "reward.npy"))
            t_sum, t_T = read_sum(os.path.join(ed, "test", "reward.npy"))
            rm_t = read_rm(os.path.join(ed, "test", "require_money.npy"))
            rm_v = read_rm(os.path.join(ed, "valid", "require_money.npy"))
            if rm_t is not None:
                rm_pool.append(rm_t)
            per[ep] = {
                "valid_sum": v_sum, "valid_T": v_T,
                "test_sum": t_sum, "test_T": t_T,
                "rm_test": rm_t, "rm_valid": rm_v,
            }
        rm_med = float(np.median(rm_pool)) if rm_pool else None
        # pick best by valid_sum (skip nans)
        valid_eps = [e for e, v in per.items() if v["valid_sum"]
                     is not None and v["test_sum"] is not None]
        best_ep = None
        if valid_eps:
            best_ep = max(valid_eps, key=lambda e: per[e]["valid_sum"])
        agg[stock][label] = {
            "n_epochs": len(epochs),
            "rm_median_test": rm_med,
            "best_valid_epoch": best_ep,
            "epochs": per,
        }
        # dump per-step reward of best valid epoch for downstream windowing
        if best_ep is not None:
            ed = os.path.join(method_dir, f"epoch_{best_ep}")
            try:
                tr = np.load(os.path.join(ed, "test", "reward.npy"),
                             allow_pickle=False).astype(np.float64)
                np.save(os.path.join(
                    OUT_DIR, f"{stock}_{label}_test_reward.npy"), tr)
                meta = {"epoch": best_ep, "rm_test": per[best_ep]["rm_test"],
                        "test_sum": per[best_ep]["test_sum"]}
                with open(os.path.join(OUT_DIR, f"{stock}_{label}_meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                agg[stock][label]["dump_error"] = str(e)
        # ALSO dump top-5 valid epochs' test rewards for ensemble experiments (earnhft only to keep size small)
        if label == "earnhft" and valid_eps:
            top5 = sorted(
                valid_eps, key=lambda e: per[e]["valid_sum"], reverse=True)[:5]
            for rk, ep in enumerate(top5):
                ed = os.path.join(method_dir, f"epoch_{ep}")
                try:
                    tr = np.load(os.path.join(ed, "test", "reward.npy"),
                                 allow_pickle=False).astype(np.float64)
                    np.save(os.path.join(
                        OUT_DIR, f"{stock}_{label}_top{rk}_ep{ep}_test_reward.npy"), tr)
                except Exception as e:
                    pass
                # also valid reward to allow valid-window selection
                try:
                    vr = np.load(os.path.join(ed, "valid", "reward.npy"),
                                 allow_pickle=False).astype(np.float64)
                    np.save(os.path.join(
                        OUT_DIR, f"{stock}_{label}_top{rk}_ep{ep}_valid_reward.npy"), vr)
                except Exception:
                    pass

with open("/tmp/full_metrics.json", "w") as f:
    json.dump(agg, f, indent=2)

print("=== summary ===")
for s in STOCKS:
    for label, _ in METHODS:
        info = agg[s].get(label, {})
        be = info.get("best_valid_epoch")
        if be is not None:
            ep = info["epochs"][be]
            rm = ep["rm_test"] or info.get("rm_median_test") or 1.0
            ret_pct = ep["test_sum"] / max(abs(rm), 1e-9) * 100.0
            print(f"{s:5s} {label:8s} best_ep={be:4d} valid_sum={ep['valid_sum']:.4f} "
                  f"test_sum={ep['test_sum']:.4f} rm={rm:.2f} test_ret={ret_pct:+.4f}%")
        else:
            print(f"{s:5s} {label:8s} NO DATA")

print("\nWrote /tmp/full_metrics.json and /tmp/best_per_step/")
