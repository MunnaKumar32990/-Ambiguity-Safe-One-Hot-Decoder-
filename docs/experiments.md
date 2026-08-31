# Experiments Documentation
## Ambiguity-Safe Inverse Decoding for One-Hot Encoded Categorical Data

---

## Experiment Design Overview

All experiments use a controlled setup:
- Training data is known and fixed
- Test data contains a mix of known, unknown/unseen, and (where applicable) dropped categories
- Ground truth labels are provided for correctness evaluation
- Results are automatically saved to `results/`

---

## Experiment A — Binary Category + Drop + Unknown

**Script:** `experiments/experiment_binary.py`

**Hypothesis:** Under `drop="if_binary"` + `handle_unknown="ignore"`, any unseen
category collides with the dropped category in the encoded representation.

| Sub-experiment | Training | Test Values | Expected Status for Unknown |
|---|---|---|---|
| A1 — Gender | Female, Male | Female, Male, Unknown | AMBIGUOUS |
| A2 — Employment | Active, Inactive | Active, Inactive, Terminated | AMBIGUOUS |
| A3 — Gender (error) | Female, Male | Female, Male (no unknown) | N/A (no unknown) |
| A4 — Gender (no drop) | Female, Male | Female, Male, Unknown | UNKNOWN |

**Key finding:** The ambiguity is not specific to Gender — it appears in any
binary feature with `drop="if_binary"` + `handle_unknown="ignore"`.

---

## Experiment B — Multi-class Categories

**Script:** `experiments/experiment_multiclass.py`

**Hypothesis:** For 3+ class features, ambiguity depends on whether a category is dropped.

| Sub-experiment | Training | Unknown Test Value | Drop Config | Expected Unknown Status |
|---|---|---|---|---|
| B1 — Colors no drop | Red, Green, Blue | Yellow | None | UNKNOWN |
| B2 — Colors drop first | Red, Green, Blue | Yellow | first | AMBIGUOUS |
| B3 — Cities drop first | Delhi, Mumbai, Chennai | Kolkata | first | AMBIGUOUS |
| B4 — Seasons if_binary | Spring, Summer, Autumn, Winter | Monsoon | if_binary (no effect) | UNKNOWN |

**Key finding:** `drop="if_binary"` does NOT drop categories from 4-class features.
Ambiguity is absent in B4, confirming the condition is configuration-dependent.

---

## Experiment C — Configuration Comparison

**Script:** `experiments/experiment_configs.py`

**Hypothesis:** The ambiguity condition is predictable from encoder configuration.

| Config | Can Produce All-Zeros? | Unknown Status |
|---|---|---|
| drop=None | No | UNKNOWN (no known cat collision) |
| drop="first" | Yes | AMBIGUOUS |
| drop="if_binary" (binary feature) | Yes | AMBIGUOUS |
| drop="if_binary" (4-class feature) | No | UNKNOWN |

**Key finding:** `can_produce_all_zeros()` correctly predicts the ambiguity
condition before any test data is seen.

---

## Experiment D — Multiple Categorical Columns

**Script:** `experiments/experiment_multicolumn.py`

**Hypothesis:** Ambiguity in one feature column does not corrupt detection
in other feature columns.

| Input | Gender Status | City Status | Row Status |
|---|---|---|---|
| Female, Delhi | AMBIGUOUS | SAFE | AMBIGUOUS |
| Male, Hyderabad | SAFE | SAFE | SAFE |
| Unknown, Delhi | AMBIGUOUS | SAFE | AMBIGUOUS |
| Female, Bangalore | AMBIGUOUS | UNKNOWN | AMBIGUOUS |
| Unknown, Bangalore | AMBIGUOUS | UNKNOWN | AMBIGUOUS |

**Key finding:** Per-feature analysis is independent. A SAFE city feature
retains a valid `safe_output` even when the gender feature is AMBIGUOUS.

---

## Results Files

| File | Contents |
|---|---|
| `results/baseline_results.csv` | Canonical baseline row-level results |
| `results/proposed_results.csv` | Proposed system results for canonical case |
| `results/experiment_binary_metrics.csv` | Binary experiment aggregate metrics |
| `results/experiment_multiclass_metrics.csv` | Multi-class aggregate metrics |
| `results/experiment_multicolumn_metrics.csv` | Multi-column aggregate metrics |
| `results/experiment_configs_metrics.csv` | Config comparison aggregate metrics |
| `results/experiment_summary.csv` | Full cross-experiment summary |
| `results/plots/status_distribution.png` | Bar chart: SAFE/AMBIGUOUS/UNKNOWN per experiment |
| `results/plots/baseline_vs_proposed.png` | Baseline wrong vs proposed flagged |
| `results/plots/detection_rate.png` | Detection rate per experiment |
