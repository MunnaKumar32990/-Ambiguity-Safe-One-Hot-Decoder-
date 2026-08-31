# Methodology
## Ambiguity-Safe Inverse Decoding for One-Hot Encoded Categorical Data

---

## 1. Problem Formulation

Let `C` be a set of training categories for a feature, and let `E : C → ℝⁿ`
be the one-hot encoding function. The inverse decoding function is `D : ℝⁿ → C`.

A **round-trip fidelity violation** occurs when there exists an input `x` such that:

```
D(E(x)) ≠ x
```

This project studies a specific class of such violations where `x ∉ C` (unseen
category) but `E(x) = E(c*)` for some known `c* ∈ C` (the dropped category).

Under `drop="if_binary"` and `handle_unknown="ignore"`:
- `E("Female") = [0]`  — Female is dropped, mapped to all-zeros
- `E("Unknown") = [0]` — Unknown is unseen, mapped to all-zeros by ignore
- Therefore: `D([0]) → "Female"` for both inputs — a silent error for "Unknown".

---

## 2. Strategy Investigation

We investigated three strategies for safe inverse decoding:

### Strategy 1 — Sentinel / Explicit Unknown Representation

**Concept:** Reserve a special encoded vector (e.g., all-ones, or an extra column)
to represent unseen categories.

**Pros:**
- Removes the collision: unseen categories have a unique representation
- Encoder output is still numeric, compatible with ML models

**Cons:**
- Requires modifying the encoder itself (breaks sklearn API compatibility)
- The reserved sentinel may conflict with legitimate combinations in multi-class
- Requires consumers of the encoded data to be aware of the sentinel

**Applicability:** Requires custom encoder, not applicable as a safety layer.

---

### Strategy 2 — Provenance / Side-Channel Metadata

**Concept:** Record the original input alongside the encoded output. During
inverse transform, look up the original value.

**Pros:**
- Perfect reconstruction fidelity
- No ambiguity possible

**Cons:**
- Requires storing original data (privacy, memory overhead)
- Not applicable when encoded data arrives from external sources
- Defeats the purpose of encoding in many ML scenarios

**Applicability:** Only useful for batch pipelines where full data is available.

---

### Strategy 3 — Strict Ambiguity Detection / Rejection (SELECTED)

**Concept:** Reason from encoder metadata to detect when a given encoded
vector could have been produced by multiple sources. When ambiguity is
detected, explicitly withhold the reconstruction and report the ambiguity.

**Pros:**
- No modification to the encoder required
- Fully sklearn-compatible
- Metadata-driven: works without original data
- Explainable: provides reasons and possible values
- Conservative: prefers "I don't know" over "wrong answer"

**Cons:**
- Cannot resolve ambiguity — only detects it
- Reduces throughput of safe reconstructions (some SAFE cases are withheld
  unnecessarily if they happen to produce all-zeros encoded vectors)

**Applicability:** Immediately applicable as a drop-in safety layer.

---

## 3. Selected Strategy — Justification

**Strategy 3 (Strict Ambiguity Detection)** was selected because:

1. **No encoder modification needed** — Works with the existing sklearn API.
2. **Generalizable** — Applies to any encoded vector from any source.
3. **Conservative principle** — It is safer to report ambiguity than to silently
   return a potentially wrong value in safety-critical applications.
4. **Explainable** — Every decision is accompanied by a reason string derived
   from metadata, suitable for debugging and audit.
5. **Metadata-sufficient** — The detection algorithm requires only the
   `EncoderMetadata` object, not the original training data.

**Limitation acknowledged:**
The dropped category (`Female`) also produces all-zeros and will be flagged
as AMBIGUOUS even when the input truly was `Female`. This is a known
conservative trade-off: we prioritize avoiding silent errors over
maximizing safe reconstruction throughput.

---

## 4. Ambiguity Detection Algorithm

```
FUNCTION detect_ambiguity(sub_vector v, FeatureMetadata m):

    IF all_zeros(v):
        IF m.dropped_category IS NOT None AND m.handle_unknown == "ignore":
            RETURN Status=AMBIGUOUS,
                   possible_values=[m.dropped_category, "<UNSEEN>"],
                   reason="Dropped category and unseen category share all-zeros"

        ELIF m.dropped_category IS NOT None:
            RETURN Status=SAFE,
                   possible_values=[m.dropped_category],
                   reason="All-zeros uniquely identifies dropped category"

        ELSE:
            RETURN Status=UNKNOWN,
                   possible_values=["<UNSEEN>"],
                   reason="All-zeros without drop: indicates unseen category"
    ELSE:
        match = find_category_by_vector(v, m)
        IF match IS NOT None:
            RETURN Status=SAFE,
                   possible_values=[match]
        ELSE:
            RETURN Status=UNKNOWN,
                   reason="Vector does not match any known category"
```

---

## 5. Functional Requirements

| ID | Requirement | Implemented |
|---|---|---|
| FR1 | Accept categorical training data | ✓ |
| FR2 | Fit a one-hot encoder | ✓ |
| FR3 | Encode test data | ✓ |
| FR4 | Track encoder metadata | ✓ |
| FR5 | Perform baseline inverse decoding | ✓ |
| FR6 | Detect ambiguous representations | ✓ |
| FR7 | Detect unseen categories | ✓ (UNKNOWN status) |
| FR8 | Produce safe decoding status | ✓ |
| FR9 | Evaluate reconstruction correctness | ✓ |
| FR10 | Generate experiment results | ✓ |

---

## 6. Non-Functional Requirements

| Requirement | Approach |
|---|---|
| Correctness | Algorithm verified by unit tests (pytest) |
| Explainability | Every result includes a human-readable `reason` string |
| Reproducibility | Fixed data, no randomness, deterministic algorithm |
| Modularity | Five independent modules with clear interfaces |
| Maintainability | PEP8 style, docstrings, typed parameters |
| sklearn compatibility | Wraps, not subclasses, OneHotEncoder |
| Computational overhead | O(n × k) per sample where k = number of features |
| Extensibility | New strategies can be added to AmbiguityDetector |
