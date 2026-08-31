"""
experiment_configs.py
=====================
Experiment C — Comparing different OneHotEncoder configurations.

Systematically compares three configurations on identical data:

  Config 1: drop=None,       handle_unknown="ignore"
  Config 2: drop="first",    handle_unknown="ignore"
  Config 3: drop="if_binary",handle_unknown="ignore"

For each config, applies the same training and test data and reports
ambiguity detection behavior.

Key Research Question:
  Is the ambiguity problem configuration-dependent, or universal?
  Answer: It is configuration-dependent but predictable from metadata.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.safe_decoder import AmbiguitySafeOneHotDecoder
from src.evaluator import ExperimentEvaluator

evaluator = ExperimentEvaluator()
all_metrics = []

# Shared data
X_train_binary = [["Female"], ["Male"]]
X_test_binary  = [["Female"], ["Male"], ["Unknown"]]
gt_binary      = [["Female"], ["Male"], ["Unknown"]]

X_train_multi  = [["Red"], ["Green"], ["Blue"]]
X_test_multi   = [["Red"], ["Green"], ["Blue"], ["Yellow"]]
gt_multi       = [["Red"], ["Green"], ["Blue"], ["Yellow"]]

CONFIGS = [
    ("drop=None",       None,        "ignore"),
    ("drop='first'",    "first",     "ignore"),
    ("drop='if_binary'","if_binary", "ignore"),
]


def run_config_experiment(name, X_train, X_test, ground_truth, drop, handle_unknown):
    print(f"\n  -- {name}  drop={drop!r}, handle_unknown={handle_unknown!r}")

    decoder = AmbiguitySafeOneHotDecoder(drop=drop, handle_unknown=handle_unknown)
    decoder.fit(X_train)

    feat = decoder.metadata.features[0]
    dropped = feat.dropped_category
    can_ambig = feat.can_produce_all_zeros()

    print(f"     dropped_category={dropped!r}, can_produce_all_zeros={can_ambig}")

    X_encoded = decoder.transform(X_test)
    results = decoder.safe_inverse_transform(X_encoded)

    for inp, r in zip(X_test, results):
        enc_str = str([int(v) for v in r.encoded])
        print(f"     {inp[0]:<12} {enc_str:<20} -> {r.row_status.value:<12}  safe={r.safe_output[0]}")

    metrics = evaluator.compute_metrics(
        experiment_name=f"C_{name}_{X_train[0][0]}",
        results=results,
        ground_truth=ground_truth,
        notes=f"Config comparison, {name}",
    )
    all_metrics.append(metrics)
    return metrics


# -----------------------------------------------------------------------------
# Binary feature — config comparison
# -----------------------------------------------------------------------------
print("=" * 65)
print("Experiment C — Binary Feature (Female/Male) — Config Comparison")
print("=" * 65)

for label, drop, hu in CONFIGS:
    run_config_experiment(label, X_train_binary, X_test_binary, gt_binary, drop, hu)

# -----------------------------------------------------------------------------
# Multi-class feature — config comparison
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("Experiment C — Multi-class Feature (Red/Green/Blue) — Config Comparison")
print("=" * 65)

for label, drop, hu in CONFIGS:
    run_config_experiment(label, X_train_multi, X_test_multi, gt_multi, drop, hu)

# -----------------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------------
print("\n\n-- Configuration Comparison Summary --")
print(f"{'Experiment':<35} {'SAFE':>6} {'AMBIG':>7} {'UNKN':>6} {'Detect%':>8}")
print("-" * 65)
for m in all_metrics:
    print(f"{m.experiment_name:<35} {m.safe_count:>6} {m.ambiguous_count:>7} "
          f"{m.unknown_count:>6} {m.detection_rate*100:>7.0f}%")

print("\nKey Findings:")
print("  drop=None:        Unknown -> UNKNOWN  (no conflict with known category)")
print("  drop='first':     Unknown -> AMBIGUOUS (conflicts with dropped category)")
print("  drop='if_binary': Binary only -> Female-like category AMBIGUOUS with unknown")

os.makedirs("results", exist_ok=True)
evaluator.save_csv(all_metrics, "results/experiment_configs_metrics.csv")
print("\n[OK] Config comparison metrics saved -> results/experiment_configs_metrics.csv")
