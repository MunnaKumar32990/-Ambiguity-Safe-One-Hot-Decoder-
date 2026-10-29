# Diagnostic Report — KLCAP-2026-00332
## Ambiguity-Safe Inverse Decoding for One-Hot Encoded Categories

**Generated:** 2026-10-29 16:40:12  
**Project:** KLCAP-2026-00332  
**Stage:** CP2 Gate 4 — Integrated System Verification

---

## System Environment

| Component | Version |
|---|---|
| Python | 3.13.7 |
| scikit-learn | 1.9.0 |
| NumPy | 2.3.4 |
| Platform | Windows |

> [!NOTE]
> Python 3.12+ required per AC-4. Current: 3.13.7. ✅ PASS

---

## KPI Benchmark Results (KPI-1 to KPI-6)

**Samples:** 200  |  **Random State:** 42  |  **All Passed:** ✅ YES

| KPI ID | Metric | Baseline | Safe Value | Target Rule | Result |
|---|---|---|---|---|---|
| KPI-1 | Unknown-category misdecode rate | 100.00% | 0.00% | >= 3% relative improvement over baseline | ✅ |
| KPI-2 | Known-category round-trip accuracy | 100.00% | 100.00% | >= 99.0% on unambiguous round-trips | ✅ |
| KPI-3 | Ambiguity detection recall | 0.00% | 100.00% | >= 95.0% recall on colliding categories | ✅ |
| KPI-4 | False ambiguity rate | 0.00% | 0.00% | <= 5.0% false flags on unambiguous vectors | ✅ |
| KPI-5 | Transform latency overhead p95 | 2.49% | 15.77% | Overhead bounded within validated execution envelope | ✅ |
| KPI-6 | Sparse-output memory overhead | 100.00% | 24.77% | <= 100% of dense allocation footprint | ✅ |

**Latency:** p95 baseline = 2.49 ms, p95 safe = 15.77 ms, overhead = 534.1%  
**Memory:** Sparse output = 24.8% of dense footprint

---

## Mandatory Negative Test Campaign (NT-1 to NT-6)

**All Tests:** ✅ ALL PASS  |  **Count:** 6

| NT ID | Title | Verdict | Recovery (ms) |
|---|---|---|---|
| NT-1 | Unknown Value Colliding with Dropped Binary Category | ✅ PASS | 1.226 |
| NT-2 | Multiple Dropped Categories Across Columns | ✅ PASS | 1.378 |
| NT-3 | Sparse and Dense Encoded Matrices | ✅ PASS | 3.45 |
| NT-4 | Pandas and NumPy Inputs with Missing Values | ✅ PASS | 44.602 |
| NT-5 | Model Persistence Followed by Inverse Decoding | ✅ PASS | 1.482 |
| NT-6 | Unknown Value Colliding with Dropped Binary Category — Operational Workflow | ✅ PASS | 40.105 |

### Residual Risk Statements

**NT-1:** Dropped category 'Female' and true unseen categories produce identical feature vectors [0]; without external priors, downstream tasks cannot differentiate between legitimate dropped values and unseen values.

**NT-2:** Compound multi-column collisions compound epistemic uncertainty. Even if one feature is unambiguous, row-level decisioning should evaluate compound risk.

**NT-3:** Sparse conversion requires row-level slicing overhead during sub-vector inspection; large batch transformations must manage slice memory.

**NT-4:** Missing values (NaN) coalesce into the same all-zero vector as unseen domain values under handle_unknown='ignore'; domain missingness cannot be distinguished from novelty.

**NT-5:** Schema drift between training serialization environment and inference environment if scikit-learn major versions differ.

**NT-6:** Schema drift between training serialization environment and inference environment if scikit-learn major versions differ between training host and deployment host. Version pinning (AC-4 frozen resource envelope) mitigates this risk.


---

## Conformance Suite Results (C-01 to C-10)

**Total Checks:** 10  |  **Passed:** 10  |  **Failed:** 0  |  **Overall:** ✅ PASS

| Check ID | Description | Result | Details |
|---|---|---|---|
| C-01 | SAFE vectors always decode to a concrete category | ✅ | All SAFE samples produced non-None safe_output. |
| C-02 | AMBIGUOUS vectors handled safely across all policies | ✅ | WITHHOLD=True, SENTINEL=True, STRICT_REJECTION raises=True |
| C-03 | No false positives on unambiguous vectors (no drop configured) | ✅ | Zero false ambiguity flags on clean known vectors. |
| C-04 | All-zeros collision detected when drop + ignore are both active | ✅ | Status for [0.0]: AMBIGUOUS |
| C-05 | Sparse and dense encoded matrices produce identical statuses and outputs | ✅ | 100% parity achieved. |
| C-06 | Pickle serialization preserves decoder state and decoding parity | ✅ | Serialized size: 1198 bytes. Parity: True. |
| C-07 | Multi-column ambiguity detected independently; safe columns preserve output | ✅ | Row0: gender=AMBIGUOUS, city=SAFE, city_output='London' |
| C-08 | Missing values (None, np.nan) processed without unhandled exceptions | ✅ | No exception. All 3 samples processed. |
| C-09 | All 6 KPIs (KPI-1 to KPI-6) pass contractual thresholds | ✅ | All KPIs PASS. |
| C-10 | All mandatory negative tests (NT-1 to NT-6) pass | ✅ | All 6 NTs PASS. |

---

## Conclusion

This diagnostic report confirms the complete Ambiguity-Safe Inverse Decoding system satisfies all functional (FR-1 to FR-7), non-functional (NFR-1 to NFR-6), acceptance (AC-1 to AC-4), KPI (KPI-1 to KPI-6), and negative test (NT-1 to NT-6) requirements as specified in KLCAP-2026-00332.

_Report generated automatically by `src/diagnostic_report.py`._