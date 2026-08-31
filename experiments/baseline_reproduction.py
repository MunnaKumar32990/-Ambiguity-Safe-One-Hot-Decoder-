"""
baseline_reproduction.py
========================
Reproduces the canonical baseline from the research problem statement.

Configuration:
    OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)

Training data:
    ["Female"], ["Male"]

Test data:
    ["Female"], ["Male"], ["Unknown"]

Expected encoding:
    Female  -> [0]
    Male    -> [1]
    Unknown -> [0]   <- same as Female!

Expected inverse_transform:
    [0] -> Female     (correct for Female)
    [1] -> Male       (correct for Male)
    [0] -> Female     (WRONG for Unknown — this is the research problem)

This script prints a clear comparison table and saves results to CSV.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.safe_decoder import AmbiguitySafeOneHotDecoder
from src.evaluator import ExperimentEvaluator

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
X_train = [["Female"], ["Male"]]
X_test  = [["Female"], ["Male"], ["Unknown"]]

ground_truth = [["Female"], ["Male"], ["Unknown"]]   # True labels

# -----------------------------------------------------------------------------
# PART 1: Exact sklearn baseline (no modifications)
# -----------------------------------------------------------------------------
print("=" * 65)
print("BASELINE REPRODUCTION — sklearn OneHotEncoder")
print("=" * 65)
print(f"\nConfiguration: drop='if_binary', handle_unknown='ignore'\n")

baseline_enc = OneHotEncoder(
    drop="if_binary",
    handle_unknown="ignore",
    sparse_output=False,
)
baseline_enc.fit(X_train)

print("Training categories:", list(baseline_enc.categories_[0]))
drop_idx = int(baseline_enc.drop_idx_[0])
print(f"Dropped category index: {drop_idx}  -> '{baseline_enc.categories_[0][drop_idx]}'")

X_test_encoded = baseline_enc.transform(X_test)
X_test_decoded = baseline_enc.inverse_transform(X_test_encoded)

print("\n-- Baseline Encoding Table --")
print(f"{'Input':<12} {'Encoded':<12} {'sklearn inverse_transform':<28} {'Correct?'}")
print("-" * 65)

baseline_rows = []
for i, (inp, enc, dec, gt) in enumerate(
    zip(X_test, X_test_encoded, X_test_decoded, ground_truth)
):
    inp_str = inp[0]
    enc_str = str(list(enc.astype(int)))
    dec_str = str(dec[0]) if dec[0] is not None else "None"
    correct = "[OK] Correct" if dec[0] == gt[0] else "[FAIL] INCORRECT (Ambiguous)"
    print(f"{inp_str:<12} {enc_str:<12} {dec_str:<28} {correct}")
    baseline_rows.append({
        "input": inp_str,
        "encoded": enc_str,
        "sklearn_decoded": dec_str,
        "ground_truth": gt[0],
        "correct": dec[0] == gt[0],
    })

print()
print("KEY OBSERVATION:")
print("  'Female' and 'Unknown' both encode to [0].")
print("  sklearn's inverse_transform returns 'Female' for both.")
print("  For the 'Unknown' input, this is SILENT INCORRECT RECONSTRUCTION.")
print("  This is the core research problem this project addresses.")

# -----------------------------------------------------------------------------
# PART 2: Proposed ambiguity-safe system
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("PROPOSED SYSTEM — AmbiguitySafeOneHotDecoder")
print("=" * 65)

decoder = AmbiguitySafeOneHotDecoder(drop="if_binary", handle_unknown="ignore")
decoder.fit(X_train)
X_encoded = decoder.transform(X_test)
safe_results = decoder.safe_inverse_transform(X_encoded)

print(f"\n{'Input':<12} {'Encoded':<12} {'Status':<12} {'Safe Output':<15} {'Possible Values'}")
print("-" * 72)
proposed_rows = []
for i, (inp, r) in enumerate(zip(X_test, safe_results)):
    inp_str = inp[0]
    enc_str = str([int(v) for v in r.encoded])
    status = r.row_status.value
    safe_out = str(r.safe_output[0]) if r.safe_output[0] is not None else "[!] None (withheld)"
    poss = str(r.possible_values[0])
    print(f"{inp_str:<12} {enc_str:<12} {status:<12} {safe_out:<15} {poss}")
    proposed_rows.append({
        "input": inp_str,
        "encoded": enc_str,
        "proposed_status": status,
        "safe_output": r.safe_output[0],
        "possible_values": r.possible_values[0],
        "confidence": r.confidence,
    })

print()
print("KEY RESULT:")
print("  For 'Female':  AMBIGUOUS -> returns None (withheld).")
print("    (Its encoding [0] is identical to an unseen category's encoding.)")
print("  For 'Male':    SAFE -> returns 'Male' with HIGH confidence.")
print("  For 'Unknown': AMBIGUOUS -> returns None (withheld), flagging")
print("  the ambiguity between ['Female', '<UNSEEN>'].")
print("  No silent incorrect reconstruction in either ambiguous case.")

# -----------------------------------------------------------------------------
# PART 3: Encoder metadata inspection
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("ENCODER METADATA USED BY AMBIGUITY DETECTOR")
print("=" * 65)
meta = decoder.metadata
feat = meta.features[0]
print(f"  Feature 0 — categories:         {feat.categories}")
print(f"  Feature 0 — dropped category:   '{feat.dropped_category}'")
print(f"  Feature 0 — drop_config:        '{feat.drop_config}'")
print(f"  Feature 0 — handle_unknown:     '{feat.handle_unknown}'")
print(f"  Feature 0 — is_binary:          {feat.is_binary}")
print(f"  Feature 0 — can_produce_all_zeros (ambiguous): {feat.can_produce_all_zeros()}")
print()
print("REASONING: all-zeros sub-vector is produced by BOTH")
print(f"  (1) Dropped category '{feat.dropped_category}' (expected)")
print(f"  (2) Any unseen category (handle_unknown='ignore')")
print("-> Collision confirmed. Ambiguity is structural, not example-specific.")

# -----------------------------------------------------------------------------
# PART 4: Evaluation metrics
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("EVALUATION METRICS")
print("=" * 65)

evaluator = ExperimentEvaluator()
metrics = evaluator.compute_metrics(
    experiment_name="baseline_reproduction",
    results=safe_results,
    ground_truth=ground_truth,
    notes="Canonical Female/Male/Unknown baseline",
)

print(f"  Total samples:          {metrics.total_samples}")
print(f"  SAFE (proposed):        {metrics.safe_count}")
print(f"  AMBIGUOUS (proposed):   {metrics.ambiguous_count}")
print(f"  UNKNOWN (proposed):     {metrics.unknown_count}")
print(f"  Baseline correct:       {metrics.baseline_correct}")
print(f"  Baseline incorrect:     {metrics.baseline_incorrect}")
print(f"  Detection rate:         {metrics.detection_rate * 100:.1f}%")
print(f"  Safe handling rate:     {metrics.safe_handling_rate * 100:.1f}%")

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------
os.makedirs("results", exist_ok=True)

baseline_df = pd.DataFrame(baseline_rows)
baseline_df.to_csv("results/baseline_results.csv", index=False)
print("\n[OK] Baseline results saved -> results/baseline_results.csv")

proposed_df = pd.DataFrame(proposed_rows)
proposed_df.to_csv("results/proposed_results.csv", index=False)
print("[OK] Proposed results saved -> results/proposed_results.csv")

evaluator.save_csv([metrics], "results/baseline_experiment_metrics.csv")
print("[OK] Metrics saved         -> results/baseline_experiment_metrics.csv")

print("\n" + "=" * 65)
print("Baseline reproduction complete.")
print("=" * 65)
