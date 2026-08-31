"""
demo.py
=======
Capstone Review-1 Demonstration Script

Demonstrates the complete research prototype in one clean, readable run.
Designed for live execution during college review / viva.

Run: python demo.py

Sections:
  1. Original training data & encoder configuration
  2. Encoded representations (baseline)
  3. Baseline inverse_transform result (with the problem)
  4. Ambiguity detection analysis
  5. Proposed safe decoding result
  6. Comparison: Baseline vs Proposed
  7. Evaluation metrics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.safe_decoder import AmbiguitySafeOneHotDecoder
from src.evaluator import ExperimentEvaluator

# -----------------------------------------------------------------------------
# Console Utilities
# -----------------------------------------------------------------------------
WIDTH = 68

def banner(text):
    print("\n" + "=" * WIDTH)
    print(f"  {text}")
    print("=" * WIDTH)

def section(text):
    print(f"\n-- {text} {'-' * max(0, WIDTH - len(text) - 5)}")

def ok(msg):
    print(f"  [OK] {msg}")

def warn(msg):
    print(f"  [!] {msg}")


# -----------------------------------------------------------------------------
banner("Ambiguity-Safe Inverse Decoding for One-Hot Encoded Categorical Data")
banner("Capstone Project — Review 1 Demonstration")
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# SECTION 1: Training Data & Configuration
# -----------------------------------------------------------------------------
section("1. Training Data & Encoder Configuration")

X_train = [["Female"], ["Male"]]
X_test  = [["Female"], ["Male"], ["Unknown"]]
ground_truth = [["Female"], ["Male"], ["Unknown"]]

print(f"\n  Training data:   {[x[0] for x in X_train]}")
print(f"  Test data:       {[x[0] for x in X_test]}")
print(f"\n  Encoder config:")
print(f"    drop           = 'if_binary'")
print(f"    handle_unknown = 'ignore'")
print(f"    sparse_output  = False")

# -----------------------------------------------------------------------------
# SECTION 2: Encoded Representations (Baseline)
# -----------------------------------------------------------------------------
section("2. Encoded Representations")

baseline_enc = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
baseline_enc.fit(X_train)
X_encoded_baseline = baseline_enc.transform(X_test)

cats = list(baseline_enc.categories_[0])
drop_idx = int(baseline_enc.drop_idx_[0])
dropped_cat = cats[drop_idx]

print(f"\n  Training categories : {cats}")
print(f"  Dropped category    : '{dropped_cat}' (index {drop_idx})")
print(f"  Encoded output width: 1 column (binary after drop)")

print(f"\n  {'Input':<12} {'Encoded Vector'}")
print(f"  {'-'*30}")
for inp, enc in zip(X_test, X_encoded_baseline):
    enc_str = str(list(enc.astype(int)))
    note = " <- dropped category maps here" if inp[0] == dropped_cat else (
           " <- UNSEEN: also maps here! (collision)" if inp[0] == "Unknown" else ""
    )
    print(f"  {inp[0]:<12} {enc_str}{note}")

print(f"\n  COLLISION DETECTED:")
print(f"    'Female' -> [0]")
print(f"    'Unknown' -> [0]   <- same encoded vector!")

# -----------------------------------------------------------------------------
# SECTION 3: Baseline inverse_transform Result
# -----------------------------------------------------------------------------
section("3. Baseline sklearn inverse_transform (THE PROBLEM)")

X_decoded_baseline = baseline_enc.inverse_transform(X_encoded_baseline)

print(f"\n  {'Input':<12} {'Encoded':<10} {'sklearn Decoded':<18} {'Correct?'}")
print(f"  {'-'*60}")
for inp, enc, dec, gt in zip(X_test, X_encoded_baseline, X_decoded_baseline, ground_truth):
    enc_str = str(list(enc.astype(int)))
    dec_str = dec[0] if dec[0] is not None else "None"
    correct = "[OK] Correct" if dec_str == gt[0] else "[FAIL] INCORRECT — silent error!"
    print(f"  {inp[0]:<12} {enc_str:<10} {dec_str:<18} {correct}")

print(f"\n  PROBLEM: sklearn returns 'Female' for 'Unknown'.")
print(f"  This is a silent incorrect reconstruction — no warning, no error.")

# -----------------------------------------------------------------------------
# SECTION 4: Ambiguity Detection Analysis (Proposed)
# -----------------------------------------------------------------------------
section("4. Ambiguity Detection — Metadata Analysis")

safe_dec = AmbiguitySafeOneHotDecoder(drop="if_binary", handle_unknown="ignore")
safe_dec.fit(X_train)

feat = safe_dec.metadata.features[0]
print(f"\n  Encoder Metadata (Feature 0 — Gender):")
print(f"    categories:            {feat.categories}")
print(f"    dropped_category:      '{feat.dropped_category}'")
print(f"    drop_config:           '{feat.drop_config}'")
print(f"    handle_unknown:        '{feat.handle_unknown}'")
print(f"    is_binary:             {feat.is_binary}")
print(f"    can_produce_all_zeros: {feat.can_produce_all_zeros()}")

print(f"\n  AMBIGUITY CONDITION (from metadata):")
print(f"    dropped_category != None  -> True  ('{feat.dropped_category}' was dropped)")
print(f"    handle_unknown == 'ignore' -> True  (unseen -> all-zeros)")
print(f"    BOTH TRUE -> all-zeros vector is AMBIGUOUS")
print(f"    It can be produced by '{feat.dropped_category}' OR any unseen category.")

# -----------------------------------------------------------------------------
# SECTION 5: Proposed Safe Decoding Result
# -----------------------------------------------------------------------------
section("5. Proposed System — Safe Inverse Transform")

X_enc = safe_dec.transform(X_test)
safe_results = safe_dec.safe_inverse_transform(X_enc)

print(f"\n  {'Input':<12} {'Encoded':<10} {'Status':<12} {'Safe Output':<18} {'Possible Values'}")
print(f"  {'-'*72}")
for inp, r in zip(X_test, safe_results):
    enc_str = str([int(v) for v in r.encoded])
    status = r.row_status.value
    safe_out = str(r.safe_output[0]) if r.safe_output[0] is not None else "[!] None (withheld)"
    poss = str(r.possible_values[0])
    print(f"  {inp[0]:<12} {enc_str:<10} {status:<12} {safe_out:<18} {poss}")

print(f"\n  KEY RESULT:")
print(f"    'Female':  AMBIGUOUS -> safe_output=None  (cannot safely decode)")
print(f"    'Male':    SAFE      -> safe_output='Male' (HIGH confidence)")
print(f"    'Unknown': AMBIGUOUS -> safe_output=None  (cannot safely decode)")

# -----------------------------------------------------------------------------
# SECTION 6: Comparison Table
# -----------------------------------------------------------------------------
section("6. Comparison: Baseline vs Proposed")

print(f"\n  {'Input':<12} {'sklearn Result':<18} {'Correct?':<12} {'Proposed Status':<16} {'Safe?'}")
print(f"  {'-'*72}")
for inp, bd, r, gt in zip(X_test, X_decoded_baseline, safe_results, ground_truth):
    sk_dec = bd[0] if bd[0] else "None"
    correct = "Yes" if sk_dec == gt[0] else "No (silent)"
    proposed = r.row_status.value
    safe = "Yes" if r.safe_output[0] is not None else "Withheld"
    print(f"  {inp[0]:<12} {sk_dec:<18} {correct:<12} {proposed:<16} {safe}")

print(f"\n  SUMMARY:")
print(f"    Baseline : 2 correct, 1 silently incorrect (no indication of error)")
print(f"    Proposed : 1 safe correct, 2 ambiguous (explicitly withheld, no silent error)")

# -----------------------------------------------------------------------------
# SECTION 7: Evaluation Metrics
# -----------------------------------------------------------------------------
section("7. Evaluation Metrics")

evaluator = ExperimentEvaluator()
metrics = evaluator.compute_metrics(
    experiment_name="demo_canonical",
    results=safe_results,
    ground_truth=ground_truth,
    notes="Review-1 demonstration run",
)

print(f"\n  Total test samples:       {metrics.total_samples}")
print(f"  Baseline correct:         {metrics.baseline_correct}  ({metrics.baseline_correct/metrics.total_samples*100:.0f}%)")
print(f"  Baseline incorrect:       {metrics.baseline_incorrect} (silent wrong reconstruction)")
print(f"  Proposed SAFE:            {metrics.safe_count}")
print(f"  Proposed AMBIGUOUS:       {metrics.ambiguous_count}")
print(f"  Proposed UNKNOWN:         {metrics.unknown_count}")
print(f"  Ambiguity detection rate: {metrics.detection_rate*100:.0f}%")
print(f"  Safe handling rate:       {metrics.safe_handling_rate*100:.0f}%")

print(f"  INTERPRETATION:")
print(f"    sklearn silently misidentified {metrics.baseline_incorrect} sample(s).")
print(f"    The proposed system flagged {metrics.ambiguous_count+metrics.unknown_count} sample(s) as AMBIGUOUS/UNKNOWN,")
print(f"    withholding the reconstruction instead of silently returning a wrong value.")
print(f"    Detection rate: {metrics.detection_rate*100:.0f}%  |  Safe handling rate: {metrics.safe_handling_rate*100:.0f}%")

# -----------------------------------------------------------------------------
# SECTION 8: Research Gap Statement
# -----------------------------------------------------------------------------
section("8. Research Gap (Formal Statement)")

print("""
  "Existing categorical encoding workflows using scikit-learn's
  OneHotEncoder generally focus on transformation compatibility
  and integration with ML pipelines. However, the ambiguity
  introduced during inverse reconstruction — specifically, the
  collision between dropped categories and unseen/unknown
  categories under the combination of drop='if_binary' and
  handle_unknown='ignore' — is not explicitly surfaced to
  downstream users or data scientists. This project proposes
  a metadata-driven safety layer that detects and explicitly
  reports such ambiguities rather than silently returning
  potentially incorrect decoded values."
""")

banner("Demonstration Complete — Run 'pytest' and 'python experiments/run_all.py' for full evaluation")
