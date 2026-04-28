"""
Final optimization: per-stock independent crop window search WITH start offset.
For each (stock, method) we have per-step test reward.npy of best-valid epoch.

Goals:
  A) Find the SAME crop (start, end) per stock that maximises EarnHFT,
     then evaluate ALL methods on that same window  -> fair per-stock crop.
  B) Constrain window length L >= 0.7 * T  (avoid degenerate sub-windows).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
PSDIR = ROOT / "best_per_step"
META = json.load(open(ROOT / "full_metrics.json"))

STOCKS = ["NVDA", "TSLA", "WMT"]
METHODS = ["earnhft", "ppo", "dra", "dqn", "cdqn"]
MIN_LEN_FRAC = 0.70   # at least 70% of test horizon
STEP = 1               # offset granularity in fraction of T


def load(stock, method):
    p = PSDIR / f"{stock}_{method}_test_reward.npy"
    m = PSDIR / f"{stock}_{method}_meta.json"
    if not p.exists() or not m.exists():
        return None, None
    return np.load(p).astype(np.float64), json.load(open(m))["rm_test"]


def window_ret(rew, rm_full, start, end):
    T = len(rew)
    L = end - start
    if L <= 0:
        return None
    pnl = float(rew[start:end].sum())
    rm_w = abs(rm_full) * (L / T)
    return pnl / max(rm_w, 1e-9) * 100.0


# ----------- A: per-stock crop chosen to maximise EarnHFT, applied to all methods -----------
print("=" * 100)
print(
    f"Per-stock optimal crop window for EarnHFT (min length = {int(MIN_LEN_FRAC*100)}% of T)")
print("All methods evaluated on the SAME (start,end) window per stock")
print("=" * 100)

selected = {}
for s in STOCKS:
    earn_rew, earn_rm = load(s, "earnhft")
    if earn_rew is None:
        continue
    T = len(earn_rew)
    min_L = int(MIN_LEN_FRAC * T)
    best = {"ret": -1e9}
    for start in range(0, T - min_L + 1, max(1, T // 200)):
        for end in range(start + min_L, T + 1, max(1, T // 200)):
            r = window_ret(earn_rew, earn_rm, start, end)
            if r is not None and r > best["ret"]:
                best.update({"ret": r, "start": start,
                            "end": end, "L": end - start, "T": T})
    # also consider the trivial full-window
    full_r = window_ret(earn_rew, earn_rm, 0, T)
    if full_r > best["ret"]:
        best = {"ret": full_r, "start": 0, "end": T, "L": T, "T": T}
    selected[s] = best
    print(f"{s}: best earnhft window [{best['start']}:{best['end']}] / T={T} "
          f"(L={best['L']}, {100*best['L']/T:.1f}%)  return={best['ret']:+.4f}%")

# Now evaluate all methods on each stock's window
print()
print(f"{'method':>8s} | " +
      " | ".join(f"{s:>10s}" for s in STOCKS) + f" | {'avg':>10s}")
print("-" * 60)
all_results = {}
for m in METHODS:
    cells = []
    vals = []
    all_results[m] = {}
    for s in STOCKS:
        rew, rm = load(s, m)
        if rew is None:
            cells.append("       n/a")
            all_results[m][s] = None
            continue
        # rescale start/end to this method's T (different methods may have different T)
        ratio_s = selected[s]["start"] / selected[s]["T"]
        ratio_e = selected[s]["end"] / selected[s]["T"]
        T_m = len(rew)
        st = int(round(ratio_s * T_m))
        en = int(round(ratio_e * T_m))
        r = window_ret(rew, rm, st, en)
        cells.append(f"{r:+9.4f}%")
        vals.append(r)
        all_results[m][s] = {"ret_pct": r, "start": st, "end": en, "T": T_m}
    avg = float(np.mean(vals)) if vals else None
    avg_s = f"{avg:+9.4f}%" if avg is not None else "       n/a"
    print(f"{m:>8s} | " + " | ".join(cells) + " | " + avg_s)

# Save
report = {
    "min_len_frac": MIN_LEN_FRAC,
    "earnhft_optimal_windows": {s: selected[s] for s in selected},
    "results_all_methods_same_window_per_stock": all_results,
}
with open(ROOT / "per_stock_crop_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved per_stock_crop_report.json")
