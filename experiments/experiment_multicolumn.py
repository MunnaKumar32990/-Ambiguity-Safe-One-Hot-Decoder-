"""
experiment_multicolumn.py
=========================
Experiment D — Multiple categorical feature columns.

Tests ambiguity detection across datasets with more than one
categorical feature, including cases where:

  - Only one column has an unknown value
  - Multiple columns have unknown values
  - Different drop configurations per column (via global encoder settings)

Dataset:
  Column 0 — Gender: Female / Male
  Column 1 — City:   Delhi / Hyderabad / Chennai

Known ambiguity:
  With drop="if_binary":
    Gender (binary) -> Female gets dropped -> all-zeros sub-vector is AMBIGUOUS
    City (3-class)  -> No drop if not binary -> unknown -> UNKNOWN
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.safe_decoder import AmbiguitySafeOneHotDecoder
from src.evaluator import ExperimentEvaluator

evaluator = ExperimentEvaluator()
all_metrics = []


def print_multicolumn_table(X_test, results, col_names=None):
    if col_names is None:
        col_names = [f"col{i}" for i in range(len(results[0].sklearn_decoded))]

    header = f"{'Input':<35} {'Row Status':<12} {'Safe Output':<35} {'sklearn Decoded'}"
    print(header)
    print("-" * 110)
    for inp, r in zip(X_test, results):
        inp_str = str(inp)
        row_status = r.row_status.value
        safe_str = str(r.safe_output)
        sk_str = str(r.sklearn_decoded)
        print(f"{inp_str:<35} {row_status:<12} {safe_str:<35} {sk_str}")


def run_mc_experiment(name, X_train, X_test, ground_truth, drop, handle_unknown, col_names=None):
    print(f"\n{'='*70}")
    print(f"Experiment: {name}")
    print(f"  drop={drop!r}, handle_unknown={handle_unknown!r}")
    print(f"  Training rows: {X_train[:3]}")
    print(f"  Test rows:     {X_test}")
    print(f"{'='*70}")

    decoder = AmbiguitySafeOneHotDecoder(drop=drop, handle_unknown=handle_unknown)
    decoder.fit(X_train)

    print(f"\n  Encoder metadata per feature:")
    for fm in decoder.metadata.features:
        print(f"    Feature {fm.feature_index}: categories={fm.categories}, "
              f"dropped='{fm.dropped_category}', "
              f"can_produce_all_zeros={fm.can_produce_all_zeros()}")

    X_encoded = decoder.transform(X_test)
    results = decoder.safe_inverse_transform(X_encoded)

    print()
    print_multicolumn_table(X_test, results, col_names)

    metrics = evaluator.compute_metrics(
        experiment_name=name,
        results=results,
        ground_truth=ground_truth,
        notes=f"Multi-column, drop={drop}, handle_unknown={handle_unknown}",
    )
    print(f"\n  SAFE={metrics.safe_count}, AMBIGUOUS={metrics.ambiguous_count}, "
          f"UNKNOWN={metrics.unknown_count}")
    all_metrics.append(metrics)
    return results, metrics


# -----------------------------------------------------------------------------
# D1: Gender + City, drop="if_binary" -> Gender binary drop creates ambiguity
# -----------------------------------------------------------------------------
X_train_D1 = [
    ["Female", "Delhi"],
    ["Female", "Hyderabad"],
    ["Male",   "Delhi"],
    ["Male",   "Chennai"],
]
X_test_D1 = [
    ["Female",  "Delhi"],       # Both known -> SAFE
    ["Male",    "Hyderabad"],   # Both known -> SAFE
    ["Unknown", "Delhi"],       # Gender unknown -> AMBIGUOUS for gender feature
    ["Female",  "Bangalore"],   # City unknown -> UNKNOWN for city feature
    ["Unknown", "Bangalore"],   # Both unknown -> AMBIGUOUS + UNKNOWN
]
gt_D1 = [
    ["Female",  "Delhi"],
    ["Male",    "Hyderabad"],
    ["Unknown", "Delhi"],
    ["Female",  "Bangalore"],
    ["Unknown", "Bangalore"],
]

run_mc_experiment(
    name="D1_gender_city_drop_if_binary",
    X_train=X_train_D1,
    X_test=X_test_D1,
    ground_truth=gt_D1,
    drop="if_binary",
    handle_unknown="ignore",
    col_names=["Gender", "City"],
)

# -----------------------------------------------------------------------------
# D2: Gender + Status (binary+binary), drop="if_binary"
# Both columns are binary -> both can produce ambiguity
# -----------------------------------------------------------------------------
X_train_D2 = [
    ["Female", "Active"],
    ["Female", "Inactive"],
    ["Male",   "Active"],
    ["Male",   "Inactive"],
]
X_test_D2 = [
    ["Female", "Active"],     # Both known -> SAFE
    ["Male",   "Inactive"],   # Both known -> SAFE
    ["Other",  "Active"],     # Gender unknown -> AMBIGUOUS
    ["Male",   "Unknown"],    # Status unknown -> AMBIGUOUS
    ["Other",  "Unknown"],    # Both unknown -> AMBIGUOUS + AMBIGUOUS
]
gt_D2 = [
    ["Female", "Active"],
    ["Male",   "Inactive"],
    ["Other",  "Active"],
    ["Male",   "Unknown"],
    ["Other",  "Unknown"],
]

run_mc_experiment(
    name="D2_gender_status_both_binary",
    X_train=X_train_D2,
    X_test=X_test_D2,
    ground_truth=gt_D2,
    drop="if_binary",
    handle_unknown="ignore",
    col_names=["Gender", "Status"],
)

# -----------------------------------------------------------------------------
# D3: Same data but drop=None -> no dropped categories -> all unknowns are UNKNOWN
# -----------------------------------------------------------------------------
run_mc_experiment(
    name="D3_gender_city_no_drop",
    X_train=X_train_D1,
    X_test=X_test_D1,
    ground_truth=gt_D1,
    drop=None,
    handle_unknown="ignore",
    col_names=["Gender", "City"],
)

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------
os.makedirs("results", exist_ok=True)
evaluator.save_csv(all_metrics, "results/experiment_multicolumn_metrics.csv")
print("\n\n[OK] Multi-column metrics saved -> results/experiment_multicolumn_metrics.csv")
