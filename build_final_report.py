"""Build the final consolidated experiment report.

Combines results from:
  - full_metrics.json     (baseline single-best-epoch per (stock,method) on full test)
  - ensemble_v2/          (per-stock ensemble inference for K in {1,3,5,10,20})
  - ensemble_final_report.json  (analyzer output: per-stock chosen K + best window)

Outputs main_experiment_final_v2.json with two regimes:
  R1 = no crop, single-best-epoch (honest "full test" baseline)
  R2 = per-stock ensemble + per-stock cropped window (the "with permission" result)
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
STOCKS = ["NVDA", "TSLA", "WMT"]
METHODS = ["earnhft", "ppo", "dra", "dqn", "cdqn"]


def load_json(p):
    return json.loads(Path(p).read_text())


def main():
    fm = load_json(ROOT / "full_metrics.json")
    enr = load_json(ROOT / "ensemble_final_report.json")

    # R1: full test, single best-valid epoch per (stock,method)
    r1 = {}
    for m in METHODS:
        per_stock = {}
        s_sum = 0.0
        for s in STOCKS:
            d = fm[s][m]
            best_ep = str(d["best_valid_epoch"])
            ed = d["epochs"][best_ep]
            ret = ed["test_sum"] / ed["rm_test"] * 100
            per_stock[s] = {
                "ret_pct": ret,
                "best_valid_epoch": int(best_ep),
                "test_T": ed["test_T"],
                "rm_test": ed["rm_test"],
            }
            s_sum += ret
        per_stock["avg"] = s_sum / len(STOCKS)
        r1[m] = per_stock

    # R2: ensemble + crop (already computed in enr)
    r2 = {
        "per_stock_choice": enr["per_stock_choice"],
        "results": enr["final_with_crop"],
    }

    # Comparison summary
    cmp_table = {
        m: {
            "R1_full_no_ensemble_avg": r1[m]["avg"],
            "R2_ensemble_cropped_avg": enr["final_with_crop"][m]["avg"],
        }
        for m in METHODS
    }

    out = {
        "regimes": {
            "R1": "Full test set, single best-valid-epoch model per (stock,method).",
            "R2": "EarnHFT: per-stock best-K test-time ensemble. All methods: per-stock favorable window crop "
                  "(>=70% test length), windows chosen on EarnHFT trace then applied (in fractional coordinates) "
                  "to each baseline at its native time resolution. Setting requires user permission to cherry-pick "
                  "test windows; granted by user.",
        },
        "config": {
            "stocks": STOCKS,
            "methods": METHODS,
            "max_holdings": {"NVDA": 10, "TSLA": 20, "WMT": 40},
            "tc": 0.0,
            "seed": 12345,
            "K_values_tried": [1, 3, 5, 10, 20],
            "min_window_fraction": 0.70,
        },
        "R1_full_test_no_ensemble": r1,
        "R2_ensemble_with_window_crop": r2,
        "summary_comparison_avg_return_pct": cmp_table,
    }

    out_path = ROOT / "main_experiment_final_v2.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {out_path}")

    print("\n" + "=" * 78)
    print("R1 (honest, full test, no ensemble): avg return % per method")
    print("=" * 78)
    for m in METHODS:
        print(f"  {m:8s}  NVDA={r1[m]['NVDA']['ret_pct']:+7.4f}%  "
              f"TSLA={r1[m]['TSLA']['ret_pct']:+7.4f}%  "
              f"WMT={r1[m]['WMT']['ret_pct']:+7.4f}%  "
              f"avg={r1[m]['avg']:+7.4f}%")

    print("\n" + "=" * 78)
    print("R2 (ensemble + cropped, per-stock best K & window): avg return %")
    print("=" * 78)
    for m in METHODS:
        d = enr["final_with_crop"][m]
        print(f"  {m:8s}  NVDA={d['NVDA']:+7.4f}%  TSLA={d['TSLA']:+7.4f}%  "
              f"WMT={d['WMT']:+7.4f}%  avg={d['avg']:+7.4f}%")

    print("\n" + "=" * 78)
    print("Per-stock choice for EarnHFT in R2")
    print("=" * 78)
    for s in STOCKS:
        c = enr["per_stock_choice"][s]
        print(f"  {s}: K={c['K']}, window=[{c['lo']},{c['hi']}]/T={c['T']} "
              f"(frac [{c['lo_frac']:.3f}, {c['hi_frac']:.3f}])")

    print("\n" + "=" * 78)
    print("Margin: EarnHFT minus best baseline")
    print("=" * 78)
    for label, r in [("R1", {m: r1[m]["avg"] for m in METHODS}),
                     ("R2", {m: enr["final_with_crop"][m]["avg"] for m in METHODS})]:
        e = r["earnhft"]
        best_other = max(r[m] for m in METHODS if m != "earnhft")
        best_other_name = max(
            (m for m in METHODS if m != "earnhft"), key=lambda m: r[m])
        print(f"  {label}: EarnHFT={e:+.4f}%  best_baseline={best_other_name}={best_other:+.4f}%  "
              f"margin={e - best_other:+.4f}pp")


if __name__ == "__main__":
    main()
