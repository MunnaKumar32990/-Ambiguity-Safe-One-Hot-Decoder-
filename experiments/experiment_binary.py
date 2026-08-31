"""
experiment_binary.py
====================
Experiment A — Binary categorical feature with drop and unknown.

Tests the full ambiguity pipeline on different binary feature datasets
beyond the Female/Male example:

  A1 — Gender: Female/Male  + Unknown, drop="if_binary", handle_unknown="ignore"
  A2 — Status: Active/Inactive + NULL_VALUE, same config
  A3 — Binary + handle_unknown="error" (no ambiguity possible)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.safe_decoder import AmbiguitySafeOneHotDecoder
from src.evaluator import ExperimentEvaluator

evaluator = ExperimentEvaluator()
all_metrics = []

# -----------------------------------------------------------------------------
def run_binary_experiment(name, X_train, X_test, ground_truth, drop, handle_unknown):
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"  drop={drop!r}, handle_unknown={handle_unknown!r}")
    print(f"  Train: {[x[0] for x in X_train]}")
    print(f"  Test:  {[x[0] for x in X_test]}")
    print(f"{'='*60}")

    decoder = AmbiguitySafeOneHotDecoder(drop=drop, handle_unknown=handle_unknown)
    decoder.fit(X_train)
    X_encoded = decoder.transform(X_test)
    results = decoder.safe_inverse_transform(X_encoded)

    print(f"\n{'Input':<15} {'Encoded':<12} {'Status':<12} {'Safe Output':<20} {'sklearn Decoded'}")
    print("-" * 75)
    for inp, r in zip(X_test, results):
        enc_str = str([int(v) for v in r.encoded])
        print(f"{inp[0]:<15} {enc_str:<12} {r.row_status.value:<12} "
              f"{str(r.safe_output[0]):<20} {r.sklearn_decoded[0]}")

    metrics = evaluator.compute_metrics(
        experiment_name=name,
        results=results,
        ground_truth=ground_truth,
        notes=f"Binary, drop={drop}, handle_unknown={handle_unknown}",
    )
    print(f"\nSAFE={metrics.safe_count}, AMBIGUOUS={metrics.ambiguous_count}, "
          f"UNKNOWN={metrics.unknown_count}, detection_rate={metrics.detection_rate*100:.0f}%")
    all_metrics.append(metrics)
    return results, metrics

# -----------------------------------------------------------------------------
# A1: Gender — canonical baseline variant
# -----------------------------------------------------------------------------
run_binary_experiment(
    name="A1_gender_drop_if_binary",
    X_train=[["Female"], ["Male"]],
    X_test=[["Female"], ["Male"], ["Unknown"], ["Female"]],
    ground_truth=[["Female"], ["Male"], ["Unknown"], ["Female"]],
    drop="if_binary",
    handle_unknown="ignore",
)

# -----------------------------------------------------------------------------
# A2: Employment Status — different binary domain
# -----------------------------------------------------------------------------
run_binary_experiment(
    name="A2_status_drop_if_binary",
    X_train=[["Active"], ["Inactive"]],
    X_test=[["Active"], ["Inactive"], ["Terminated"], ["Active"]],
    ground_truth=[["Active"], ["Inactive"], ["Terminated"], ["Active"]],
    drop="if_binary",
    handle_unknown="ignore",
)

# -----------------------------------------------------------------------------
# A3: Binary + handle_unknown="error" -> no ambiguity expected
# (sklearn would raise error for unknown, but we test known values only)
# -----------------------------------------------------------------------------
run_binary_experiment(
    name="A3_gender_drop_error",
    X_train=[["Female"], ["Male"]],
    X_test=[["Female"], ["Male"]],           # No unknowns
    ground_truth=[["Female"], ["Male"]],
    drop="if_binary",
    handle_unknown="error",
)

# -----------------------------------------------------------------------------
# A4: Binary + drop=None (no category dropped)
# Unknown still maps to all-zeros but is UNKNOWN not AMBIGUOUS
# -----------------------------------------------------------------------------
run_binary_experiment(
    name="A4_gender_no_drop",
    X_train=[["Female"], ["Male"]],
    X_test=[["Female"], ["Male"], ["Unknown"]],
    ground_truth=[["Female"], ["Male"], ["Unknown"]],
    drop=None,
    handle_unknown="ignore",
)

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------
os.makedirs("results", exist_ok=True)
evaluator.save_csv(all_metrics, "results/experiment_binary_metrics.csv")
print("\n\n[OK] Binary experiment metrics saved -> results/experiment_binary_metrics.csv")
print(f"  Total experiments: {len(all_metrics)}")
