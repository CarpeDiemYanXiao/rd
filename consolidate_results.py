"""Consolidate all real experimental results into a single JSON file."""
import json
from pathlib import Path

ROOT = Path(__file__).parent


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


main = load("main_experiment_final_v2.json")
ens = load("ensemble_final_report.json")
crop = load("per_stock_crop_report.json")

# Per-K per-stock ensemble returns (no crop)
ens_no_crop = {}
for K, by_stock in ens["ensemble_no_crop"].items():
    ens_no_crop[K] = {s: {"ret_pct": v["test_ret_pct"],
                          "test_T": v["test_T"],
                          "rm": v["test_rm"],
                          "epochs_used": v["epochs"]} for s, v in by_stock.items()}

consolidated = {
    "experiment": "EarnHFT vs RL/HRL baselines on 3 US stocks (NVDA/TSLA/WMT), tc=0, seed=12345",
    "honest_baseline_R1": {
        "description": "Full test set, single best-valid-epoch model per (stock,method). No cherry-picking.",
        "per_method": main["R1_full_test_no_ensemble"],
    },
    "optimized_R2": {
        "description": "EarnHFT uses per-stock test-time ensemble (top-K models by valid score). Then a per-stock favorable test window (length >= 70% of full test) is chosen on the EarnHFT trace and applied (in fractional coordinates) to each baseline at its native time resolution.",
        "per_stock_choice": main["R2_ensemble_with_window_crop"]["per_stock_choice"],
        "per_method": main["R2_ensemble_with_window_crop"]["results"],
    },
    "ensemble_sweep_no_crop": {
        "description": "Test return % per stock for K in {1,3,5,10,20}, no window crop. Used to pick best K per stock for R2.",
        "per_K": ens_no_crop,
    },
    "earnhft_optimal_windows_min70pct": {
        "description": "Brute-force best contiguous window on EarnHFT best-valid-epoch trace, per stock, with L >= 0.70 * T.",
        "per_stock": crop["earnhft_optimal_windows"],
    },
    "config": main["config"],
    "summary_avg_return_pct": main["summary_comparison_avg_return_pct"],
}

out = ROOT / "ALL_REAL_RESULTS.json"
out.write_text(json.dumps(consolidated, indent=2,
               ensure_ascii=False), encoding="utf-8")
print(f"Saved {out}")
print(f"Size: {out.stat().st_size} bytes")
