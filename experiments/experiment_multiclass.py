"""
experiment_multiclass.py
========================
Experiment B — Three or more categories (non-binary features).

Tests whether ambiguity exists when a feature has 3+ categories:

  B1 — Colors: Red/Green/Blue + unknown "Yellow", drop=None
  B2 — Colors: Red/Green/Blue + "Yellow", drop="first"
  B3 — Cities: Delhi/Mumbai/Chennai + unknown "Kolkata", drop="first"

Key insight:
  With 3+ categories and drop=None:
    All-zeros sub-vector -> UNKNOWN (not AMBIGUOUS with known category)
  With 3+ categories and drop="first":
    All-zeros sub-vector -> AMBIGUOUS (could be dropped "first" or unseen)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.safe_decoder import AmbiguitySafeOneHotDecoder
from src.evaluator import ExperimentEvaluator

evaluator = ExperimentEvaluator()
all_metrics = []

def run_multiclass_experiment(name, X_train, X_test, ground_truth, drop, handle_unknown):
    print(f"\n{'='*65}")
    print(f"Experiment: {name}")
    print(f"  drop={drop!r}, handle_unknown={handle_unknown!r}")
    cats = list({x[0] for x in X_train})
    print(f"  Training categories: {cats}")
    print(f"  Test values:         {[x[0] for x in X_test]}")
    print(f"{'='*65}")

    decoder = AmbiguitySafeOneHotDecoder(drop=drop, handle_unknown=handle_unknown)
    decoder.fit(X_train)
    X_encoded = decoder.transform(X_test)
    results = decoder.safe_inverse_transform(X_encoded)

    feat = decoder.metadata.features[0]
    print(f"\n  Encoder metadata:")
    print(f"    categories:      {feat.categories}")
    print(f"    dropped_category:{feat.dropped_category!r}")
    print(f"    can_produce_all_zeros: {feat.can_produce_all_zeros()}")

    print(f"\n{'Input':<12} {'Encoded':<25} {'Status':<12} {'Safe Output':<15} {'sklearn'}")
    print("-" * 80)
    for inp, r in zip(X_test, results):
        enc_str = str([int(v) for v in r.encoded])
        print(f"{inp[0]:<12} {enc_str:<25} {r.row_status.value:<12} "
              f"{str(r.safe_output[0]):<15} {r.sklearn_decoded[0]}")

    metrics = evaluator.compute_metrics(
        experiment_name=name,
        results=results,
        ground_truth=ground_truth,
        notes=f"Multi-class, drop={drop}, handle_unknown={handle_unknown}",
    )
    print(f"\nSAFE={metrics.safe_count}, AMBIGUOUS={metrics.ambiguous_count}, "
          f"UNKNOWN={metrics.unknown_count}, detection_rate={metrics.detection_rate*100:.0f}%")
    all_metrics.append(metrics)
    return results, metrics

# -----------------------------------------------------------------------------
# B1: 3-class colors, drop=None — unknown -> UNKNOWN (no ambiguity with known)
# -----------------------------------------------------------------------------
run_multiclass_experiment(
    name="B1_colors_no_drop",
    X_train=[["Red"], ["Green"], ["Blue"]],
    X_test=[["Red"], ["Green"], ["Blue"], ["Yellow"]],
    ground_truth=[["Red"], ["Green"], ["Blue"], ["Yellow"]],
    drop=None,
    handle_unknown="ignore",
)

# -----------------------------------------------------------------------------
# B2: 3-class colors, drop="first" — unknown -> AMBIGUOUS (collides with "Blue")
# -----------------------------------------------------------------------------
run_multiclass_experiment(
    name="B2_colors_drop_first",
    X_train=[["Red"], ["Green"], ["Blue"]],
    X_test=[["Red"], ["Green"], ["Blue"], ["Yellow"]],
    ground_truth=[["Red"], ["Green"], ["Blue"], ["Yellow"]],
    drop="first",
    handle_unknown="ignore",
)

# -----------------------------------------------------------------------------
# B3: Cities, drop="first"
# -----------------------------------------------------------------------------
run_multiclass_experiment(
    name="B3_cities_drop_first",
    X_train=[["Delhi"], ["Mumbai"], ["Chennai"]],
    X_test=[["Delhi"], ["Mumbai"], ["Chennai"], ["Kolkata"]],
    ground_truth=[["Delhi"], ["Mumbai"], ["Chennai"], ["Kolkata"]],
    drop="first",
    handle_unknown="ignore",
)

# -----------------------------------------------------------------------------
# B4: 4-class + drop="if_binary" (no drop expected for 4-class)
# -----------------------------------------------------------------------------
run_multiclass_experiment(
    name="B4_seasons_drop_if_binary",
    X_train=[["Spring"], ["Summer"], ["Autumn"], ["Winter"]],
    X_test=[["Spring"], ["Summer"], ["Autumn"], ["Winter"], ["Monsoon"]],
    ground_truth=[["Spring"], ["Summer"], ["Autumn"], ["Winter"], ["Monsoon"]],
    drop="if_binary",   # No effect: not binary, so nothing dropped
    handle_unknown="ignore",
)

os.makedirs("results", exist_ok=True)
evaluator.save_csv(all_metrics, "results/experiment_multiclass_metrics.csv")
print("\n\n[OK] Multi-class metrics saved -> results/experiment_multiclass_metrics.csv")
