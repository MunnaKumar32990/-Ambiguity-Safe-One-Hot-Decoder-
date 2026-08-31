# Ambiguity-Safe Inverse Decoding for One-Hot Encoded Categorical Data

**B.Tech Final-Year Capstone Project — Review 1**

---

## 1. Problem

When scikit-learn's `OneHotEncoder` is configured with:
```python
OneHotEncoder(drop="if_binary", handle_unknown="ignore")
```
the `inverse_transform` method silently returns **incorrect values** for unseen
categories, without any warning or error.

**Concrete example:**
```
Training: Female, Male
Test:     Female, Male, Unknown

Encoding:
  Female  → [0]   ← dropped category
  Male    → [1]
  Unknown → [0]   ← unseen, mapped to zeros by handle_unknown="ignore"

inverse_transform result:
  [0] → "Female"   ✓ (correct for Female)
  [1] → "Male"     ✓ (correct for Male)
  [0] → "Female"   ✗ (WRONG for Unknown — should be flagged!)
```

`Female` and `Unknown` collide at the same encoded representation `[0]`.
sklearn silently returns `"Female"` for `Unknown`. **This is the core research problem.**

---

## 2. Motivation

This problem is relevant whenever:
- Categorical data is encoded, transformed through a pipeline, and then
  needs to be recovered (e.g., recommendation systems, explainability tools)
- Unseen categories appear at inference time (which is common in production)
- Silent errors are more dangerous than explicit "I don't know" responses

---

## 3. Research Gap

> *"Existing categorical encoding workflows generally focus on transformation
> compatibility and integration with ML pipelines. However, the ambiguity
> introduced during inverse reconstruction — specifically, the collision
> between dropped categories and unseen/unknown categories under
> `drop='if_binary'` + `handle_unknown='ignore'` — is not explicitly
> surfaced to downstream users or data scientists."*

---

## 4. Research Objectives

1. Analyze the ambiguity in OHE inverse decoding under specific configurations.
2. Reproduce and characterize the baseline sklearn behavior.
3. Design a metadata-driven ambiguity detection mechanism.
4. Develop a safety layer that explicitly reports ambiguous reconstructions.
5. Evaluate the proposed system against the baseline across multiple scenarios.

---

## 5. Proposed Methodology

A **metadata-driven safety layer** wraps the standard `inverse_transform`:

1. After fitting, extract structured metadata from the encoder.
2. For each encoded vector, check whether the sub-vector for each feature
   is ambiguous (i.e., can it be produced by both a dropped category AND
   an unseen category?).
3. Return an explicit status: `SAFE`, `AMBIGUOUS`, or `UNKNOWN`.
4. Withhold the decoded value when ambiguity is detected.

**No encoder modification required.** The system works as a wrapper.

---

## 6. Architecture

```
Input Dataset → MetadataAwareEncoder → Encoded Data
                      │
                 EncoderMetadata
                      │
               AmbiguityDetector
                      │
          AmbiguitySafeOneHotDecoder
                      │
            ┌─────────┼─────────┐
          SAFE    AMBIGUOUS  UNKNOWN
            │         │         │
          output    None      None
                      │
               ExperimentEvaluator → results/ + plots/
```

---

## 7. Installation

```bash
cd CapStoneProject
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, scikit-learn ≥ 1.3, numpy, pandas, matplotlib, pytest

---

## 8. How to Run

```bash
# Main demonstration (for review/viva)
python demo.py

# Canonical baseline (shows the problem)
python experiments/baseline_reproduction.py

# All experiments + plots
python experiments/run_all.py

# Individual experiments
python experiments/experiment_binary.py
python experiments/experiment_multiclass.py
python experiments/experiment_multicolumn.py
python experiments/experiment_configs.py

# All tests
pytest --tb=short -v
```

---

## 9. Example Output (`python demo.py`)

```
════════════════════════════════════════════════════════════════════
  3. Baseline sklearn inverse_transform (THE PROBLEM)
════════════════════════════════════════════════════════════════════

  Input        Encoded    sklearn Decoded    Correct?
  ────────────────────────────────────────────────────────────
  Female       [0]        Female             ✓ Correct
  Male         [1]        Male               ✓ Correct
  Unknown      [0]        Female             ✗ INCORRECT — silent error!

  PROBLEM: sklearn returns 'Female' for 'Unknown'.
  This is a silent incorrect reconstruction — no warning, no error.

────────────────────────────────────────────────────────────────────

  5. Proposed System — Safe Inverse Transform
════════════════════════════════════════════════════════════════════

  Input        Encoded    Status       Safe Output        Possible Values
  ────────────────────────────────────────────────────────────────────────
  Female       [0]        AMBIGUOUS    ⚠ None (withheld)  ['Female', '<UNSEEN>']
  Male         [1]        SAFE         Male               ['Male']
  Unknown      [0]        AMBIGUOUS    ⚠ None (withheld)  ['Female', '<UNSEEN>']
```

---

## 10. Experiments

| Experiment | Focus | Key Finding |
|---|---|---|
| A — Binary variants | Gender, Employment | Ambiguity appears in any binary feature with drop+ignore |
| B — Multi-class | Colors, Cities | Ambiguity requires active drop; `if_binary` doesn't drop 4-class |
| C — Config comparison | drop=None/first/if_binary | Ambiguity is predictable from `can_produce_all_zeros()` |
| D — Multi-column | Gender + City | Per-feature isolation prevents cross-contamination |

---

## 11. Results

Results are generated by running the experiments. Key metrics from the canonical baseline:

| Metric | Baseline | Proposed |
|---|---|---|
| Correct decodes | 2/3 (66.7%) | 1/3 SAFE (Male correct) |
| Silent wrong decodes | 1 (Unknown→Female) | 0 |
| Ambiguity detected | 0 | 2 (Female+Unknown both flagged) |
| Detection rate | 0% | 100% |

> **Note:** All result values in this README are derived from actual experimental runs,
> not manually entered. Run `python experiments/run_all.py` to reproduce.

---

## 12. Limitations

1. **Conservative flagging:** The dropped known category (e.g., `Female`) is also
   flagged as AMBIGUOUS because its encoding is indistinguishable from an unknown.
   True positive rate for the dropped category is 0% in SAFE outcomes.

2. **No resolution:** The system detects but cannot resolve ambiguity. If the
   application requires a decoded value, a fallback strategy must be provided
   separately.

3. **handle_unknown='error' interaction:** When `handle_unknown='error'`, sklearn
   raises an exception for unseen categories before our safety layer is invoked.
   We handle this by testing only with `handle_unknown='ignore'` configurations.

---

## 13. Future Work

1. **Probabilistic disambiguation:** Use feature value priors to estimate which
   candidate (dropped or unseen) is more likely.
2. **Partial resolution:** If additional context columns are unambiguous, use
   them to refine the estimate for ambiguous columns.
3. **Sentinel encoding extension:** Integrate Strategy 1 (sentinel vectors) as
   an optional encoder modification for cases where ambiguity must be prevented
   rather than detected.
4. **Integration with pandas pipelines:** Extend to work with `ColumnTransformer`.
5. **Review-2:** Evaluate on real-world datasets where unseen categories appear
   naturally at inference time.

---

## 14. References

1. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
   JMLR, 12, 2825–2830. https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html

2. Zheng, A. & Casari, A. (2018). Feature Engineering for Machine Learning.
   O'Reilly Media. ISBN 978-1-491-95324-2.

3. Potdar, K., et al. (2017). A Comparative Study of Categorical Variable
   Encoding Techniques for Neural Network Classifiers.
   IJCA, 175(4). DOI: 10.5120/ijca2017915495.

4. García, S., Luengo, J., Herrera, F. (2015). Data Preprocessing in Data Mining.
   Springer. ISBN 978-3-319-10247-4.

---

*Project maintained for academic research purposes. B.Tech Capstone, 2025–26.*
