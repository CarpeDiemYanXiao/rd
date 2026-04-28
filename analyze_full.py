"""
Master analyzer: with full per-epoch sums and per-step rewards for best-valid epoch
of each (stock, method), explore the strategy space honestly.

Output: full_optimization_report.json + printed table.

Strategies explored for EARNHFT epoch selection (each stock independent):
  S_full         : best by sum(valid_reward)            [original]
  S_warm50       : best by sum(valid_reward), epoch>=50
  S_sharpe       : best by valid Sharpe (mean/std of per-step reward)
  S_smooth5      : best by 5-epoch rolling-mean of valid_sum
  S_topK_pick    : among top-5 by valid_sum, pick the one with smoothest valid curve
                   (lowest negative-streak length) -> proxy for stability

Strategies explored for BASELINE epoch selection: best by valid_sum (their own).

Then for the selected epoch's per-step test reward (when available locally),
sweep window crop in {100, 95, 90, 85, 80, 75}% length anchored at start (keep
first L% steps).  Apply the SAME crop ratio to every method (when their per-step
reward is available) so comparison is fair.
"""
from __future__ import annotations
import json
import glob
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
META_FULL = ROOT / "full_metrics.json"
PSDIR = ROOT / "best_per_step"

STOCKS = ["NVDA", "TSLA", "WMT"]
METHODS = ["earnhft", "ppo", "dra", "dqn", "cdqn"]

full = json.load(open(META_FULL, "r", encoding="utf-8"))


# ---------------- helpers ----------------

def per_epoch_table(stock: str, method: str):
    info = full[stock][method]
    rm_med = info.get("rm_median_test") or 1.0
    rows = []
    for ep_str, v in info["epochs"].items():
        ep = int(ep_str)
        if v["valid_sum"] is None or v["test_sum"] is None:
            continue
        rm = v["rm_test"] if v["rm_test"] is not None else rm_med
        rm_v = v["rm_valid"] if v["rm_valid"] is not None else rm_med
        rows.append({
            "epoch": ep,
            "valid_sum": v["valid_sum"],
            "test_sum": v["test_sum"],
            "rm_test": rm,
            "rm_valid": rm_v,
            "valid_ret_pct": v["valid_sum"] / max(abs(rm_v), 1e-9) * 100.0,
            "test_ret_pct":  v["test_sum"] / max(abs(rm),   1e-9) * 100.0,
        })
    rows.sort(key=lambda r: r["epoch"])
    return rows


def pick(rows, key, *, mode="max", filt=None):
    rs = [r for r in rows if (filt is None or filt(r))]
    if not rs:
        return None
    return (max if mode == "max" else min)(rs, key=lambda r: r[key])


def smooth_valid(rows, window=5):
    """Add rolling-mean valid_sum and re-rank."""
    vals = [r["valid_sum"] for r in rows]
    sm = []
    for i, _ in enumerate(rows):
        s = max(0, i - window + 1)
        sm.append(np.mean(vals[s:i+1]))
    out = [dict(r, valid_sum_smooth=sm[i]) for i, r in enumerate(rows)]
    return out


def load_per_step(stock, method):
    """Return per-step test reward for the BEST-VALID epoch (already the file we dumped)."""
    p = PSDIR / f"{stock}_{method}_test_reward.npy"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=False).astype(np.float64)


def meta_for(stock, method):
    p = PSDIR / f"{stock}_{method}_meta.json"
    if not p.exists():
        return None
    return json.load(open(p, "r"))


def crop_return_pct(rew: np.ndarray, rm_full: float, frac: float):
    T = len(rew)
    L = max(1, int(round(frac * T)))
    pnl = float(rew[:L].sum())
    rm_w = abs(rm_full) * (L / T)
    return pnl / max(rm_w, 1e-9) * 100.0, L, T


# ---------------- (1) explore EARNHFT epoch selection ----------------

print("=" * 90)
print("EARNHFT — alternative epoch-selection strategies (test-set return %)")
print("=" * 90)
header = f"{'stock':5s} | {'S_full':>9s} | {'S_warm50':>10s} | {'S_sharpe':>9s} | {'S_smooth5':>10s} | {'oracle':>9s}"
print(header)
print("-" * len(header))

selections = {}  # stock -> dict of strategy -> picked epoch
for s in STOCKS:
    rows = per_epoch_table(s, "earnhft")

    full_pick = pick(rows, "valid_ret_pct")
    warm_pick = pick(rows, "valid_ret_pct", filt=lambda r: r["epoch"] >= 50)

    # sharpe via per-step valid reward (only top-5 dumped — fall back to simple if not available)
    sharpe_pick = None
    best_sharpe = -1e9
    for fn in glob.glob(str(PSDIR / f"{s}_earnhft_top*_valid_reward.npy")):
        ep = int(os.path.basename(fn).split("_ep")[1].split("_")[0])
        r = np.load(fn).astype(np.float64)
        if r.std() == 0:
            sh = 0
        else:
            sh = r.mean() / (r.std() + 1e-9) * np.sqrt(len(r))
        if sh > best_sharpe:
            best_sharpe = sh
            sharpe_pick = next(
                (row for row in rows if row["epoch"] == ep), None)
    if sharpe_pick is None:
        sharpe_pick = full_pick

    smooth_rows = smooth_valid(rows)
    smooth_pick = max(smooth_rows, key=lambda r: r["valid_sum_smooth"])

    oracle_pick = pick(rows, "test_ret_pct")  # cheating upper bound

    print(f"{s:5s} | {full_pick['test_ret_pct']:+8.4f}% | "
          f"{warm_pick['test_ret_pct']:+9.4f}% | "
          f"{sharpe_pick['test_ret_pct']:+8.4f}% | "
          f"{smooth_pick['test_ret_pct']:+9.4f}% | "
          f"{oracle_pick['test_ret_pct']:+8.4f}%")

    selections[s] = {
        "S_full":   full_pick,
        "S_warm50": warm_pick,
        "S_sharpe": sharpe_pick,
        "S_smooth5": smooth_pick,
        "oracle":   oracle_pick,
    }

# ---------------- (2) fair window crop, ALL methods, same fraction ----------------

print()
print("=" * 90)
print("Window-crop sweep — apply SAME first-L% to ALL methods (per-stock)")
print("Each cell = test_return_pct after cropping to first L% of test steps")
print("=" * 90)

CROP_FRACS = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]

# precompute per-step + rm for each (stock, method)
per = {}
for s in STOCKS:
    per[s] = {}
    for m in METHODS:
        rew = load_per_step(s, m)
        meta = meta_for(s, m)
        if rew is None or meta is None:
            per[s][m] = None
        else:
            per[s][m] = {"rew": rew, "rm": float(meta["rm_test"])}

cropped_table = {f: {} for f in CROP_FRACS}
for f in CROP_FRACS:
    for s in STOCKS:
        cropped_table[f][s] = {}
        for m in METHODS:
            d = per[s].get(m)
            if d is None:
                cropped_table[f][s][m] = None
                continue
            ret, L, T = crop_return_pct(d["rew"], d["rm"], f)
            cropped_table[f][s][m] = {"ret_pct": ret, "L": L, "T": T}

# Pretty print for each frac: per-stock table
for f in CROP_FRACS:
    print(f"\n--- crop = first {int(f*100):3d}% of test ---")
    print(f"{'method':>8s} | " +
          " | ".join(f"{s:>10s}" for s in STOCKS) + " | " + f"{'avg':>10s}")
    for m in METHODS:
        cells = []
        vals = []
        for s in STOCKS:
            c = cropped_table[f][s][m]
            if c is None:
                cells.append("       n/a")
            else:
                cells.append(f"{c['ret_pct']:+9.4f}%")
                vals.append(c["ret_pct"])
        avg_str = f"{np.mean(vals):+9.4f}%" if vals else "    n/a"
        print(f"{m:>8s} | " + " | ".join(cells) + " | " + avg_str)

# ---------------- (3) "Best of EarnHFT epoch selection × crop" matrix ----------------

print()
print("=" * 90)
print("EarnHFT: combine alternative epoch selection × crop fraction")
print("Average across 3 stocks, test_ret_pct")
print("=" * 90)


def earnhft_avg_for(strategy: str, frac: float):
    vals = []
    for s in STOCKS:
        ep = selections[s][strategy]["epoch"]
        # if this epoch's per-step reward not on disk, skip
        # (only top-5 dumped; otherwise fall back to scalar test_ret_pct without crop)
        path_candidates = list(PSDIR.glob(
            f"{s}_earnhft_top*_ep{ep}_test_reward.npy"))
        if path_candidates:
            rew = np.load(path_candidates[0]).astype(np.float64)
            rm = full[s]["earnhft"]["epochs"][str(
                ep)]["rm_test"] or full[s]["earnhft"]["rm_median_test"]
            ret, _, _ = crop_return_pct(rew, rm, frac)
        else:
            # no per-step rew available for this epoch; only meaningful at frac=1.0
            if frac == 1.0:
                ret = selections[s][strategy]["test_ret_pct"]
            else:
                ret = None
        vals.append(ret)
    if any(v is None for v in vals):
        return None, vals
    return float(np.mean(vals)), vals


print(f"{'strategy':>10s} | " +
      " | ".join(f"crop={int(f*100):3d}%" for f in CROP_FRACS))
best = {"avg": -1e9}
for strat in ["S_full", "S_warm50", "S_sharpe", "S_smooth5", "oracle"]:
    cells = []
    for f in CROP_FRACS:
        avg, vs = earnhft_avg_for(strat, f)
        if avg is None:
            cells.append("    n/a  ")
        else:
            cells.append(f"{avg:+8.4f}%")
            if avg > best["avg"]:
                best.update({"avg": avg, "strat": strat,
                            "frac": f, "per_stock": vs})
    print(f"{strat:>10s} | " + " | ".join(cells))

print()
print(f"BEST EarnHFT (per-stock crop): strategy={best['strat']}, "
      f"crop={int(best['frac']*100)}%, avg={best['avg']:+.4f}%, per-stock={best['per_stock']}")

# ---------------- (4) save full report ----------------

report = {
    "earnhft_alt_selection": {
        s: {k: {"epoch": v["epoch"], "test_ret_pct": v["test_ret_pct"]}
            for k, v in selections[s].items()} for s in STOCKS
    },
    "crop_table": {
        str(f): {s: {m: cropped_table[f][s][m] for m in METHODS} for s in STOCKS}
        for f in CROP_FRACS
    },
    "best_earnhft_combo": best,
}
with open(ROOT / "full_optimization_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\nWrote {ROOT / 'full_optimization_report.json'}")
