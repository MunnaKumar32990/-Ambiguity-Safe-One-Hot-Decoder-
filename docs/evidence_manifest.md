# Evidence Manifest: KLCAP-2026-00332
## Ambiguity-Safe Inverse Decoding for One-Hot Encoded Categories
**B.Tech Capstone Project I (Semester VII / CP1 Gate 2)**

---

### 1. Primary Evidence & Cited Defect Record

| Field | Detail |
|---|---|
| **Canonical ID** | `KLCAP-2026-00332` |
| **Domain** | Computational Intelligence and Optimization |
| **Cited Source** | `scikit-learn` GitHub Issue `#34549` / PR reference `#14549` |
| **Issue Title** | `OneHotEncoder.inverse_transform decodes unknown categories as the dropped category when drop and handle_unknown='ignore'` |
| **Access Date** | Pinned to current stable release (`scikit-learn 1.9.0`) |
| **Licence** | BSD-3-Clause (scikit-learn open source) |
| **Vulnerability Root Cause** | Conflation of all-zero sub-vectors between dropped known categories and unseen categories handled via `ignore`. |
| **Consequence** | Silent label attribution without error or warning. |

---

### 2. Protocol and Quality Assurance (AC-1 to AC-4)

1. **Protocol Pinning (AC-4):**
   - Toolchain: Python 3.12+ (tested on Python 3.13.7)
   - Libraries: `scikit-learn==1.9.0`, `numpy`, `scipy`, `pandas`, `flask`, `pytest`
   - Deterministic synthetic fixtures with frozen random seeds (`random_state=42`) to guarantee reproducible baseline and candidate evaluations.

2. **Boundary and Failure Operation (AC-2):**
   - Verified across binary dropped features, multi-class dropped features, compound multi-column drops, sparse/dense matrices, and inputs with missing values (`NaN`/`None`).

3. **Independent Partition & Blinding (AC-3):**
   - Separate 400-sample Monte-Carlo held-out partition reserved exclusively for acceptance benchmark evaluation (`src/kpi_benchmarking.py`), preventing training data leakage.

---

### 3. Deliverables Matrix (D1 to D7)

| Deliverable | Description | Location / Evidence | Verification Status |
|---|---|---|---|
| **D1** | Problem Charter, Requirements & Architecture | `docs/architecture.md`, `docs/review2_report.md` | ✅ Complete & Approved |
| **D2** | Reproducible Baseline with Oracle & Failure Log | `tests/test_baseline.py`, `experiments/baseline_reproduction.py` | ✅ 100% Reproducible |
| **D3** | Explicit Ambiguity Policies (Sentinel, Side-Channel, Rejection) | `src/policies.py`, `tests/test_policies.py` | ✅ Verified across 4 policies |
| **D4** | Ambiguity-Safe Inverse Decoder Prototype System | `src/safe_decoder.py`, `backend/app.py`, `frontend/` | ✅ Interactive UI & REST API |
| **D5** | Automated Acceptance Harness (NT-1..5, AC-1..4, KPI-1..6) | `src/negative_tests.py`, `src/kpi_benchmarking.py`, `tests/` | ✅ 63/63 Pytest Suite Passing |
| **D6** | Versioned Repository, Operating Guide & API Docs | `README.md`, `PROJECT_STATUS.md`, `run_app.py` | ✅ Complete Runbook |
| **D7** | Demonstration Package & Viva Defense Guide | `notebooks/demonstration.ipynb`, `docs/presentation_and_viva_guide.md` | ✅ Ready for Panel Viva |

---

### 4. Mandatory Negative-Test Campaign Summary

- **NT-1 (Binary Collision):** PASS (0% silent errors; AmbiguityStatus.AMBIGUOUS detected).
- **NT-2 (Multi-Column Drop):** PASS (Column-isolated detection; no cross-contamination).
- **NT-3 (Sparse/Dense Parity):** PASS (100% semantic parity between `csr_matrix` and dense arrays).
- **NT-4 (Missing Values):** PASS (Proper handling of `None`/`np.nan` with ambiguity detection).
- **NT-5 (Model Persistence):** PASS (Serialization via `pickle` maintains 100% post-reload fidelity).
