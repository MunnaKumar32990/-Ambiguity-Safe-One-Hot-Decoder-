# System Architecture
## Ambiguity-Safe Inverse Decoding for One-Hot Encoded Categorical Data

---

## 1. Overview

The system is a Python research prototype that wraps scikit-learn's
`OneHotEncoder` with a metadata-driven safety layer. The architecture is
linear with a branching output stage.

```
Input Dataset
      │
      ▼
┌─────────────────────────┐
│  MetadataAwareEncoder   │  ← src/encoder.py
│  (wraps OneHotEncoder)  │
└────────────┬────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
 Encoded Data   EncoderMetadata    ← src/metadata.py
                     │
                     ▼
         ┌───────────────────────┐
         │   AmbiguityDetector   │  ← src/ambiguity_detector.py
         │  (metadata-driven)    │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ AmbiguitySafeOneHot   │  ← src/safe_decoder.py
         │      Decoder          │
         └────────┬──────────────┘
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
   SAFE       AMBIGUOUS    UNKNOWN
  output      → None       → None
  returned   (withheld)  (withheld)
      │           │           │
      └───────────┼───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │  ExperimentEvaluator │  ← src/evaluator.py
       │  (metrics + plots)   │
       └──────────────────────┘
                  │
                  ▼
         results/ + plots/
```

---

## 2. Component Descriptions

### 2.1 MetadataAwareEncoder (`src/encoder.py`)

**Responsibility:** Fit, transform, and expose encoder metadata.

**Key design decision:** Instead of subclassing `OneHotEncoder`, we wrap it.
This ensures full sklearn compatibility while allowing us to extract metadata
in a structured way after fitting.

**Outputs after fit:**
- `EncoderMetadata` — structured metadata for all feature columns
- Standard sklearn encode/decode interface

**Why not subclass?**
Subclassing sklearn estimators requires careful handling of `get_params()`
and `__sklearn_tags__`. A composition-over-inheritance approach is safer
for this research prototype.

---

### 2.2 EncoderMetadata + FeatureMetadata (`src/metadata.py`)

**Responsibility:** Store per-feature structured information after fitting.

**Key fields per feature:**
| Field | Source | Purpose |
|---|---|---|
| `categories` | `encoder.categories_[i]` | All training categories |
| `dropped_category` | `encoder.drop_idx_[i]` → lookup | Actual dropped category |
| `handle_unknown` | Config parameter | Determines unknown encoding behavior |
| `can_produce_all_zeros()` | Derived | Core ambiguity condition flag |

**The `can_produce_all_zeros()` method** is the central invariant:
it returns `True` if and only if the all-zeros sub-vector for this feature
can be produced by *both* a known category (dropped) and an unseen category.
This is a pure metadata-derived judgment — no data needed.

---

### 2.3 AmbiguityDetector (`src/ambiguity_detector.py`)

**Responsibility:** Classify encoded sub-vectors as SAFE, AMBIGUOUS, or UNKNOWN.

**Algorithm (per feature column):**

```
Input: sub_vector v, FeatureMetadata m, sklearn_decoded_value

if is_all_zeros(v):
    if m.has_dropped_category AND m.handle_unknown == "ignore":
        → AMBIGUOUS
          possible_values = [dropped_category, "<UNSEEN>"]
          reason = "Dropped known cat and unseen cat collide at all-zeros"
    elif m.has_dropped_category AND m.handle_unknown != "ignore":
        → SAFE
          reason = "All-zeros uniquely identifies dropped cat; unknown would error"
    else:
        → UNKNOWN
          reason = "All-zeros without drop: indicates unseen category"
else:
    match = find_matching_known_category(v, m)
    if match found:
        → SAFE
          possible_values = [match]
    else:
        → UNKNOWN
          reason = "Vector doesn't match any known category"
```

**Key property:** This algorithm uses ONLY the encoded vector + metadata.
It does NOT look up the original input value. This makes it applicable
to any encoded vector, including those produced externally.

---

### 2.4 AmbiguitySafeOneHotDecoder (`src/safe_decoder.py`)

**Responsibility:** Public API. Orchestrates encode → detect → report.

**Output per sample (`DecodingResult`):**
```python
{
  "sample_index": int,
  "encoded": [float, ...],
  "sklearn_decoded": [category, ...],      # Raw sklearn result
  "row_status": "SAFE" | "AMBIGUOUS" | "UNKNOWN",
  "safe_output": [category | None, ...],  # None when not safe
  "possible_values": [[...], ...],         # Per-feature candidates
  "reasons": [str, ...],                  # Per-feature explanation
  "confidence": "HIGH" | "MEDIUM" | "LOW"
}
```

**Design decision — Isolation of ambiguity:**
Each feature is analyzed independently. A SAFE feature in an AMBIGUOUS row
still returns a valid `safe_output`. This allows partial reconstruction
(e.g., City is known even when Gender is ambiguous in a two-column dataset).

---

### 2.5 ExperimentEvaluator (`src/evaluator.py`)

**Responsibility:** Metrics computation, CSV export, matplotlib visualization.

**Metrics:**
| Metric | Formula |
|---|---|
| `detection_rate` | flagged_cases / baseline_incorrect |
| `safe_handling_rate` | (safe_count + flagged) / total |

**Plots generated:**
1. `status_distribution.png` — SAFE/AMBIGUOUS/UNKNOWN per experiment
2. `baseline_vs_proposed.png` — Baseline incorrect vs proposed flagged
3. `detection_rate.png` — Detection rate by experiment (horizontal bar)

---

## 3. Data Flow (Concrete Example)

**Input:** `[["Unknown"]]`  
**Training:** `[["Female"], ["Male"]]`  
**Config:** `drop="if_binary"`, `handle_unknown="ignore"`

```
Step 1: MetadataAwareEncoder.transform([["Unknown"]])
        → X_encoded = [[0.0]]
           (handle_unknown='ignore' → all-zeros for unseen)

Step 2: AmbiguityDetector.analyze_vector([0.0], ["Female"])
        → sub_vec = [0.0]
        → is_all_zeros = True
        → feat.dropped_category = "Female"   (has_dropped = True)
        → feat.handle_unknown = "ignore"      (ignores_unknown = True)
        → BOTH TRUE → Status = AMBIGUOUS
        → possible_values = ["Female", "<UNSEEN>"]

Step 3: AmbiguitySafeOneHotDecoder builds DecodingResult:
        → row_status   = AMBIGUOUS
        → safe_output  = [None]          (withheld)
        → sklearn_decoded = ["Female"]   (recorded for comparison)
        → confidence   = "LOW"
        → reason       = "Dropped category 'Female' and unseen share [0]"
```

---

## 4. Extension Points

| Extension | Where to add |
|---|---|
| Custom ambiguity rules | `AmbiguityDetector._analyze_feature()` |
| New output formats | `AmbiguitySafeOneHotDecoder.results_to_dataframe()` |
| New metrics | `ExperimentEvaluator.compute_metrics()` |
| Additional plots | `ExperimentEvaluator.plot_*()` |
| Strategy selection | Add a `strategy` parameter to `AmbiguitySafeOneHotDecoder` |
