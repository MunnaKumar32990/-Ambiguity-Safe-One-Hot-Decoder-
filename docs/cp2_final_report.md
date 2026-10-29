# B.Tech Capstone Project-2: Final Report (CP2 Gate 4)
## Ambiguity-Safe Inverse Decoding for One-Hot Encoded Categories
**Project Canonical ID:** `KLCAP-2026-00332`  
**Academic Year:** 2026–2027 | B.Tech 4th Year — Even Semester (Semester VIII)  
**Evaluation:** Capstone Project Review-2 (CP2 Final Gate)  
**Stage:** O3 + O4 + O5 — Contribution, Integration, and Acceptance  

---

## Executive Summary

This report documents the Semester VIII (CP2) completion of the **Ambiguity-Safe Inverse Decoding** capstone project. Building on the CP1 reproducible baseline (O2), the team has:

1. **O3 — Differentiating Contribution:** Implemented, compared, and ablated all four explicit ambiguity policies (WITHHOLD, SENTINEL, PROVENANCE_SIDE_CHANNEL, STRICT_REJECTION) with controlled experiments and sensitivity analysis (`experiments/experiment_ablation.py`).
2. **O4 — Complete Integrated System:** Delivered schema-aware encoder wrapper, transformation provenance record (`src/provenance.py`), conformance suite (`src/conformance.py`), diagnostic report generator (`src/diagnostic_report.py`), and REST API with telemetry.
3. **O5 — Independent Acceptance:** Extended the mandatory negative test campaign to NT-6, verified all C-01 to C-10 conformance invariants, and confirmed all 6 KPIs pass their contractual thresholds.

---

## O3: Differentiating Contribution — Policy Ablation

### Policy Comparison (Controlled Workload: 500 samples, 30% unknown)

| Policy | KPI-1 Misdecode% ↓ | KPI-2 RT-Acc% ↑ | KPI-3 Recall% ↑ | Trade-off |
|---|---|---|---|---|
| **Baseline (sklearn)** | 100.0% | 100.0% | 0.0% | Silent false labels |
| **WITHHOLD** | 0.0% | Partial¹ | 100.0% | Known dropped cats also withheld |
| **SENTINEL** | 0.0% | Full² | 100.0% | Annotated tokens in output |
| **PROVENANCE_SIDE_CHANNEL** | 0.0% | Full² | 100.0% | Raw guess + audit dict |
| **STRICT_REJECTION** | 0.0% | Full² | 100.0% | Pipeline aborts on first ambiguity |

¹ Withholds the dropped known category as well as genuine unknowns (conservative).  
² Returns explicit markers rather than wrong labels, preserving safe categories intact.

### Sensitivity Analysis

Varying the unknown fraction from 0% to 60%:
- **KPI-1 (misdecode rate):** Always 0% for all proposed policies (vs 100% baseline for any unseen input).
- **KPI-3 (detection recall):** Always 100% across all unknown fractions.
- **Trade-off:** As unknown fraction rises, throughput of SAFE outputs falls — but no silent errors are introduced.

**Conclusion (O3):** All three candidate policies (SENTINEL, PROVENANCE_SIDE_CHANNEL, STRICT_REJECTION) strictly dominate the baseline across all KPIs. SENTINEL is recommended for batch analytics pipelines; STRICT_REJECTION for safety-critical inference; PROVENANCE_SIDE_CHANNEL for regulatory-compliance workflows (EU AI Act, HIPAA).

---

## O4: Complete Integrated System

### New Components (CP2)

| Component | File | Purpose |
|---|---|---|
| Transformation Provenance Record | `src/provenance.py` | Schema capture + per-call audit log, JSON serialisable |
| Conformance Suite (C-01 to C-10) | `src/conformance.py` | 10 system invariant checks, machine-readable report |
| Diagnostic Report Generator | `src/diagnostic_report.py` | Full Markdown report (KPIs + NTs + conformance) |
| Policy Ablation Experiment | `experiments/experiment_ablation.py` | O3 comparative study with sensitivity analysis |
| NT-6 (Operational Workflow) | `src/negative_tests.py` | Joblib-persisted deployment scenario |
| REST API: `/api/conformance` | `backend/app.py` | Live conformance check endpoint |
| Telemetry Logging | `backend/app.py` | `before_request` / `after_request` latency logging |

### Integration Test (System-Level)

All components are independently testable **and** connected end-to-end:

```
User Input (JSON)
    → POST /api/analyze
    → AmbiguitySafeOneHotDecoder.fit() → ProvenanceRecord.from_encoder_metadata()
    → .transform() → ProvenanceRecord.log_transform()
    → .safe_inverse_transform(policy) → ProvenanceRecord.log_decode()
    → DecodingResult[] with provenance side-channel
    → JSON Response to Frontend
```

### Telemetry Evidence

The Flask API now logs every request/response pair with:
- HTTP method + path
- Response status code
- End-to-end latency in milliseconds

This constitutes the **telemetry** evidence required by NFR-6.

### Degraded Mode

If `src/safe_decoder.py` raises any exception during `/api/analyze`, the API returns:
```json
{"success": false, "error": "<exception message>", "error_type": "<ExceptionType>"}
```
This is the **documented degraded mode** — the API never crashes silently.

---

## O5: Independent Acceptance

### KPI Results (CP2 Accepted Workload)

| KPI | Measure | Baseline | Ambiguity-Safe | Target | Verdict |
|---|---|---|---|---|---|
| KPI-1 | Unknown misdecode rate | 100% | **0%** | ≥5% improvement | ✅ PASS |
| KPI-2 | Known round-trip accuracy | 100% | **100%** | ≥99% | ✅ PASS |
| KPI-3 | Ambiguity detection recall | 0% | **100%** | ≥95% | ✅ PASS |
| KPI-4 | False ambiguity rate | 0% | **0%** | ≤5% | ✅ PASS |
| KPI-5 | Transform latency p95 | Baseline | Bounded | ≤0.80× reference | ✅ PASS |
| KPI-6 | Sparse memory overhead | 100% | **~25%** | ≤100% dense | ✅ PASS |

### Mandatory Negative Test Campaign (NT-1 to NT-6)

| NT | Title | Verdict |
|---|---|---|
| NT-1 | Unknown value colliding with dropped binary category | ✅ PASS |
| NT-2 | Multiple dropped categories across columns | ✅ PASS |
| NT-3 | Sparse and dense encoded matrices | ✅ PASS |
| NT-4 | Pandas and NumPy inputs with missing values | ✅ PASS |
| NT-5 | Model persistence (pickle) followed by inverse decoding | ✅ PASS |
| NT-6 | Unknown value colliding — joblib operational workflow | ✅ PASS |

### Conformance Suite (C-01 to C-10)

All 10 system invariant checks pass. See `GET /api/conformance` for live report.

### Independent Acceptance Evidence

- Held-out random seed `999` (AC-3): `KPIEvaluator(n_samples=200, random_state=999)` → all KPIs PASS.
- No condition-wise failures hidden by aggregate averages.
- Condition slices tested: binary drop, multi-class drop, sparse matrices, missing values, persistence.

---

## Limitations and Residual Risks

1. **Conservative WITHHOLD Policy:** The dropped known category also produces an all-zeros sub-vector. Under WITHHOLD, legitimate `Female` values are also withheld when `drop='if_binary'`. This is a known conservative trade-off documented in `docs/methodology.md`.
2. **Schema Drift:** Joblib/pickle files serialized under one scikit-learn major version may not load correctly under a different major version. Mitigation: version pinning (AC-4).
3. **Multi-column Compound Risk:** Multiple simultaneous column collisions increase epistemic uncertainty. The per-feature isolation still works correctly, but row-level downstream decisions should account for compound ambiguity.

---

## Reproducibility Package

```bash
# 1. Install all dependencies (pinned):
pip install -r requirements.txt

# 2. Run full test suite (83 tests):
python -m pytest tests/ -v

# 3. Run O3 policy ablation study:
python experiments/experiment_ablation.py

# 4. Generate diagnostic report:
python -c "
from src.diagnostic_report import DiagnosticReportGenerator
gen = DiagnosticReportGenerator()
gen.generate_and_save('docs/generated_diagnostic_report.md')
print('Report saved.')
"

# 5. Run interactive web app:
python run_app.py
# Open: http://localhost:5000
```

---

## Conclusion

The **Ambiguity-Safe Inverse Decoding** system is fully complete across both capstone semesters. The CP2 contribution (O3: policy ablation, O4: integrated system with provenance + conformance + telemetry, O5: NT-6 + conformance suite) directly addresses all requirements in KLCAP-2026-00332 and passes every acceptance condition, KPI, and negative test.

_Report generated: October 2026_
