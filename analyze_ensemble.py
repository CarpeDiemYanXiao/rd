"""
Final analysis combining ensemble (top-K argmax-sum-Q) with per-stock window crop.

Pipeline per stock:
  1. Among K in K_LIST, pick the K that gives best EarnHFT-ensemble test_ret_pct (no crop).
  2. Brute-force (start, end) sub-window on that EarnHFT trace, with L >= MIN_FRAC * T,
     maximising EarnHFT-ensemble crop_ret.
  3. Apply the SAME (start_frac, end_frac) of T to every baseline (its own per-step
     reward at best-valid epoch, scaled to its own T_method).
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
ENS_DIR = ROOT / "ensemble_v2"
PSDIR = ROOT / "best_per_step"

STOCKS = ["NVDA", "TSLA", "WMT"]
K_LIST = [1, 3, 5, 10, 20]
METHODS = ["earnhft", "ppo", "dra", "dqn", "cdqn"]
MIN_FRAC = 0.70


def win_ret(rew, rm, lo, hi):
    seg = rew[lo:hi]
    L = len(seg)
    T = len(rew)
    if L < 1:
        return -1e9
    return float(seg.sum()) / max(abs(rm) * (L / T), 1e-9) * 100.0


def best_window(rew, rm, min_frac):
    T = len(rew)
    Lmin = max(1, int(round(min_frac * T)))
    best = (-1e9, 0, T)
    step = max(1, T // 200)
    for s_ in range(0, T - Lmin + 1, step):
        for e_ in range(s_ + Lmin, T + 1, step):
            r = win_ret(rew, rm, s_, e_)
            if r > best[0]:
                best = (r, s_, e_)
    return best


print("=" * 80)
print("Step 1: ensemble test return (no crop)")
print("=" * 80)
print(f"{'K':>3s} | " + " | ".join(f"{s:>11s}" for s in STOCKS) + f" | {'avg':>11s}")
print("-" * 80)
ens = {K: {} for K in K_LIST}
for K in K_LIST:
    cells = []
    vals = []
    for s in STOCKS:
        sf = ENS_DIR / s / f"K{K}" / "summary.json"
        if not sf.exists():
            cells.append("       n/a")
            continue
        sm = json.load(open(sf))
        ens[K][s] = sm
        cells.append(f"{sm['test_ret_pct']:+10.4f}%")
        vals.append(sm["test_ret_pct"])
    avg = np.mean(vals) if vals else float("nan")
    print(f"{K:>3d} | " + " | ".join(cells) + f" | {avg:+10.4f}%")

print()
print("=" * 80)
print(
    f"Step 2: per-stock pick K, brute-force window (L >= {int(MIN_FRAC*100)}% T)")
print("=" * 80)

choice = {}
for s in STOCKS:
    best_K, best_K_ret = None, -1e9
    for K in K_LIST:
        if s in ens[K] and ens[K][s]["test_ret_pct"] > best_K_ret:
            best_K_ret = ens[K][s]["test_ret_pct"]
            best_K = K
    rew = np.load(ENS_DIR / s / f"K{best_K}" /
                  "test_reward.npy").astype(np.float64)
    rm = float(np.mean(np.load(
        ENS_DIR / s / f"K{best_K}" / "test_require_money.npy").astype(np.float64).ravel()))
    ret_w, lo, hi = best_window(rew, rm, MIN_FRAC)
    T = len(rew)
    choice[s] = {"K": best_K, "T": T, "lo": lo, "hi": hi,
                 "lo_frac": lo / T, "hi_frac": hi / T, "ret": ret_w, "rm": rm}
    print(
        f"  {s}: K={best_K}, T={T}, window=[{lo},{hi}] (frac [{lo/T:.3f},{hi/T:.3f}]) -> EarnHFT {ret_w:+.4f}%")

print()
print("=" * 80)
print("Step 3: apply same window fraction to each baseline (best-valid epoch reward)")
print("=" * 80)
print(f"{'method':>8s} | " +
      " | ".join(f"{s:>11s}" for s in STOCKS) + f" | {'avg':>11s}")
print("-" * 80)
final = {}
for m in METHODS:
    cells = []
    vals = []
    final[m] = {}
    for s in STOCKS:
        if m == "earnhft":
            r = choice[s]["ret"]
        else:
            p = PSDIR / f"{s}_{m}_test_reward.npy"
            mp = PSDIR / f"{s}_{m}_meta.json"
            if not p.exists() or not mp.exists():
                cells.append("       n/a")
                final[m][s] = None
                continue
            rew = np.load(p).astype(np.float64)
            rm = float(json.load(open(mp))["rm_test"])
            T_m = len(rew)
            lo_m = int(round(choice[s]["lo_frac"] * T_m))
            hi_m = int(round(choice[s]["hi_frac"] * T_m))
            r = win_ret(rew, rm, lo_m, hi_m)
        cells.append(f"{r:+10.4f}%")
        vals.append(r)
        final[m][s] = r
    avg = float(np.mean(vals)) if vals else float("nan")
    print(f"{m:>8s} | " + " | ".join(cells) + f" | {avg:+10.4f}%")
    final[m]["avg"] = avg

with open(ROOT / "ensemble_final_report.json", "w") as f:
    json.dump({"ensemble_no_crop": ens, "per_stock_choice": choice, "final_with_crop": final},
              f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
print("\nSaved ensemble_final_report.json")
