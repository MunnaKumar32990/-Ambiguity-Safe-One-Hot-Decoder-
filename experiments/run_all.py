"""
run_all.py
==========
Master experiment runner.

Runs all experiments in sequence, collects all metrics, and generates:
  - results/experiment_summary.csv
  - results/plots/status_distribution.png
  - results/plots/baseline_vs_proposed.png
  - results/plots/detection_rate.png

Usage:
    python experiments/run_all.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import subprocess
import traceback
from src.safe_decoder import AmbiguitySafeOneHotDecoder
from src.evaluator import ExperimentEvaluator

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

evaluator = ExperimentEvaluator()
all_metrics = []


def section(title: str):
    print("\n" + "#" * 65)
    print(f"  {title}")
    print("#" * 65)


# -----------------------------------------------------------------------------
# Helper to run one inline experiment and collect metrics
# -----------------------------------------------------------------------------
def run_experiment(name, X_train, X_test, ground_truth, drop, handle_unknown, notes=""):
    decoder = AmbiguitySafeOneHotDecoder(drop=drop, handle_unknown=handle_unknown)
    decoder.fit(X_train)
    X_enc = decoder.transform(X_test)
    results = decoder.safe_inverse_transform(X_enc)
    metrics = evaluator.compute_metrics(
        experiment_name=name,
        results=results,
        ground_truth=ground_truth,
        notes=notes,
    )
    all_metrics.append(metrics)
    return metrics


# -----------------------------------------------------------------------------
# 1. Baseline canonical case
# -----------------------------------------------------------------------------
section("1. Baseline — Female/Male/Unknown")
m = run_experiment(
    "Baseline_F_M_Unknown",
    X_train=[["Female"], ["Male"]],
    X_test=[["Female"], ["Male"], ["Unknown"]],
    ground_truth=[["Female"], ["Male"], ["Unknown"]],
    drop="if_binary",
    handle_unknown="ignore",
    notes="Canonical research problem baseline",
)
print(f"  SAFE={m.safe_count}  AMBIGUOUS={m.ambiguous_count}  "
      f"UNKNOWN={m.unknown_count}  detection={m.detection_rate*100:.0f}%")

# -----------------------------------------------------------------------------
# 2. Binary variants
# -----------------------------------------------------------------------------
section("2. Binary Feature Variants")

binary_cases = [
    ("Binary_drop_if_binary_ignore",  [["Active"],["Inactive"]], [["Active"],["Inactive"],["Terminated"]], [["Active"],["Inactive"],["Terminated"]], "if_binary", "ignore"),
    ("Binary_drop_none_ignore",       [["Active"],["Inactive"]], [["Active"],["Inactive"],["Terminated"]], [["Active"],["Inactive"],["Terminated"]], None,       "ignore"),
    ("Binary_drop_first_ignore",      [["Active"],["Inactive"]], [["Active"],["Inactive"],["Terminated"]], [["Active"],["Inactive"],["Terminated"]], "first",    "ignore"),
]
for name, tr, te, gt, dr, hu in binary_cases:
    m = run_experiment(name, tr, te, gt, dr, hu)
    print(f"  {name:<40} SAFE={m.safe_count} AMBIG={m.ambiguous_count} UNKN={m.unknown_count}")

# -----------------------------------------------------------------------------
# 3. Multi-class variants
# -----------------------------------------------------------------------------
section("3. Multi-class Feature Variants")

multi_cases = [
    ("Multiclass_no_drop",   [["Red"],["Green"],["Blue"]], [["Red"],["Green"],["Blue"],["Yellow"]], [["Red"],["Green"],["Blue"],["Yellow"]], None,    "ignore"),
    ("Multiclass_drop_first",[["Red"],["Green"],["Blue"]], [["Red"],["Green"],["Blue"],["Yellow"]], [["Red"],["Green"],["Blue"],["Yellow"]], "first", "ignore"),
    ("Cities_drop_first",    [["Delhi"],["Mumbai"],["Chennai"]], [["Delhi"],["Mumbai"],["Chennai"],["Kolkata"]], [["Delhi"],["Mumbai"],["Chennai"],["Kolkata"]], "first", "ignore"),
    ("Seasons_if_binary",    [["Spring"],["Summer"],["Autumn"],["Winter"]], [["Spring"],["Summer"],["Autumn"],["Winter"],["Monsoon"]], [["Spring"],["Summer"],["Autumn"],["Winter"],["Monsoon"]], "if_binary", "ignore"),
]
for name, tr, te, gt, dr, hu in multi_cases:
    m = run_experiment(name, tr, te, gt, dr, hu)
    print(f"  {name:<40} SAFE={m.safe_count} AMBIG={m.ambiguous_count} UNKN={m.unknown_count}")

# -----------------------------------------------------------------------------
# 4. Multi-column variants
# -----------------------------------------------------------------------------
section("4. Multi-column Feature Variants")

X_train_mc = [["Female","Delhi"],["Female","Hyderabad"],["Male","Delhi"],["Male","Chennai"]]
X_test_mc  = [["Female","Delhi"],["Male","Hyderabad"],["Unknown","Delhi"],["Female","Bangalore"],["Unknown","Bangalore"]]
gt_mc      = [["Female","Delhi"],["Male","Hyderabad"],["Unknown","Delhi"],["Female","Bangalore"],["Unknown","Bangalore"]]

mc_cases = [
    ("MultiCol_gender_city_drop_if_binary", X_train_mc, X_test_mc, gt_mc, "if_binary", "ignore"),
    ("MultiCol_gender_city_no_drop",        X_train_mc, X_test_mc, gt_mc, None,        "ignore"),
    ("MultiCol_gender_city_drop_first",     X_train_mc, X_test_mc, gt_mc, "first",     "ignore"),
]
for name, tr, te, gt, dr, hu in mc_cases:
    m = run_experiment(name, tr, te, gt, dr, hu)
    print(f"  {name:<45} SAFE={m.safe_count} AMBIG={m.ambiguous_count} UNKN={m.unknown_count}")

# -----------------------------------------------------------------------------
# 5. Summary table
# -----------------------------------------------------------------------------
section("FULL EXPERIMENT SUMMARY")
print(f"\n{'Experiment':<45} {'Tot':>4} {'SAFE':>5} {'AMBG':>5} {'UNKN':>5} {'BsErr':>6} {'Det%':>6} {'SH%':>5}")
print("-" * 90)
for m in all_metrics:
    print(f"{m.experiment_name:<45} {m.total_samples:>4} {m.safe_count:>5} "
          f"{m.ambiguous_count:>5} {m.unknown_count:>5} {m.baseline_incorrect:>6} "
          f"{m.detection_rate*100:>5.0f}% {m.safe_handling_rate*100:>4.0f}%")

# -----------------------------------------------------------------------------
# 6. Save results and plots
# -----------------------------------------------------------------------------
section("SAVING RESULTS & PLOTS")

evaluator.save_csv(all_metrics, os.path.join(RESULTS_DIR, "experiment_summary.csv"))
print(f"  [OK] Summary CSV saved -> {RESULTS_DIR}/experiment_summary.csv")

# Use a subset for clearer plots (too many bars gets cluttered)
plot_subset = all_metrics[:8]
evaluator.plot_all(plot_subset, PLOTS_DIR)
print(f"  [OK] Plots saved -> {PLOTS_DIR}/")

print("\n" + "=" * 65)
print("All experiments complete.")
print(f"Total experiments run: {len(all_metrics)}")
print("=" * 65)
