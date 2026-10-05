# Capstone Project-1: Live Demonstration & Viva Defense Guide
## Ambiguity-Safe Inverse Decoding for One-Hot Encoded Categories (`KLCAP-2026-00332`)

This guide prepares you for the live presentation, code walkthrough, and technical Q&A before the Capstone Project Review-2 panel.

---

## 1. 3-Minute Live Presentation Script

### Slide / Opening: Problem Statement & Motivation
> "Good morning, respected panel members. Our Capstone Project is **Ambiguity-Safe Inverse Decoding for One-Hot Encoded Categories** (`KLCAP-2026-00332`).
>
> In machine learning, one-hot encoding is the standard method for categorical features. When engineers configure scikit-learn's `OneHotEncoder` with category dropping (`drop='first'` or `drop='if_binary'`) to prevent multicollinearity, and `handle_unknown='ignore'` to handle unseen data in production, a critical silent defect occurs ([scikit-learn Issue #34549](https://github.com/scikit-learn/scikit-learn/issues/34549)).
>
> Both the dropped known category and any unseen test category are encoded as an **all-zeros vector** `[0]`. Consequently, standard `inverse_transform` **silently misidentifies unseen categories as the dropped category** with zero error warning. In healthcare or financial ML, a novel patient symptom or unseen credit profile is silently misclassified as a valid baseline category."

### Middle: Our Innovation & Architecture
> "To solve this without modifying upstream scikit-learn internals, we developed the **AmbiguitySafeOneHotDecoder**.
>
> 1. **Schema-Aware Metadata Envelope:** During `fit()`, we extract structural metadata: which categories exist, which index was dropped, and the exact sub-vector column ranges.
> 2. **Ambiguity Analysis Engine:** During decoding, we inspect each sub-vector. If an all-zero sub-vector is encountered where a drop was active and unknowns are ignored, the system classifies the feature as `AMBIGUOUS`. If no drop was active, it is classified as `UNKNOWN`. If non-zero, it is verified as `SAFE`.
> 3. **Deliverable D3 Policies:** We introduced 4 explicit ambiguity resolution policies:
>    - `WITHHOLD`: Replaces ambiguous features with `None` (data cleaning).
>    - `SENTINEL`: Injects informative tokens like `<AMBIGUOUS:Female|UNSEEN>`.
>    - `PROVENANCE_SIDE_CHANNEL`: Emits candidate predictions with an audit dictionary for regulatory compliance.
>    - `STRICT_REJECTION`: Halts execution by raising `AmbiguityRejectionError` for safety-critical pipelines."

### Closing: Verification & KPI Achievements
> "For Review-2, we implemented the complete end-to-end system:
> - **63 automated pytest tests** covering **AC-1 to AC-4** and **NT-1 to NT-5** with 100% pass rate.
> - **KPI-1:** Unknown category misdecode rate dropped from **100% in scikit-learn to 0% in our system**.
> - **KPI-2 & KPI-3:** 100% known round-trip accuracy and 100% ambiguity detection recall.
> - **Live Interactive Demonstrator:** Full-stack Flask REST API + modern glassmorphic web UI."

---

## 2. Step-by-Step Live Demonstration Walkthrough

### Step 1: Launch the System
In the terminal, run:
```bash
python run_app.py
```
This automatically launches the Flask API and opens `http://localhost:5000` in your web browser.

### Step 2: Tab 1 — Interactive Sandbox
1. **Show Vanilla scikit-learn failure:**
   - Select the preset **"Gender (Binary — Canonical Example)"**.
   - Set Policy to **WITHHOLD**.
   - Click **Run Safe Decoding Analysis**.
   - Point out the comparison table:
     - Sample 3: Input was `Unknown`.
     - Baseline (sklearn) output: `Female` ⚠️ *(Point out that sklearn silently corrupted the label!)*
     - Status: `AMBIGUOUS`.
     - Safe Output: `None` *(Our system withheld the false label)*.
2. **Demonstrate Policy D3 (Sentinel):**
   - Switch Policy to **SENTINEL**.
   - Click **Run Safe Decoding Analysis**.
   - Point out that Sample 3 now outputs `<AMBIGUOUS:Female|<UNSEEN>>`.
3. **Demonstrate Policy D3 (Provenance Side-Channel):**
   - Switch Policy to **PROVENANCE_SIDE_CHANNEL**.
   - Click **Run Safe Decoding Analysis**.
   - Click the **Audit Trace** button in the table. Show the modal popup with the JSON audit log containing confidence `LOW`, candidate collision set `['Female', '<UNSEEN>']`, and ambiguity flag `true`.
4. **Demonstrate Policy D3 (Strict Rejection):**
   - Switch Policy to **STRICT_REJECTION**.
   - Click **Run Safe Decoding Analysis**.
   - Show the red alert banner: `AmbiguityRejectionError` was raised and caught, preventing tainted data from propagating downstream.
5. **Show Real-World Demographics:**
   - Select the preset **"Real-World: Adult Census Demographics"**.
   - Point out per-column isolation: unseen categories in `Occupation` and `Sex` are isolated without corrupting known categories in `Education` or `Workclass`.

### Step 3: Tab 2 — Mandatory Negative Tests (NT-1 to NT-5)
1. Click the **Mandatory Negative Tests (NT-1..5)** tab.
2. Click **Run All 5 Negative Tests**.
3. Walk through each card:
   - **NT-1:** Binary Dropped Collision.
   - **NT-2:** Multi-Column Compound Dropped Collision.
   - **NT-3:** Sparse and Dense Matrix Parity (`scipy.sparse.csr_matrix`).
   - **NT-4:** Missing Values (`NaN`, `None`).
   - **NT-5:** Model Persistence (`pickle` serialization and reload).
4. Highlight that all 5 tests show `PASS` with sub-millisecond recovery times and formal residual risk statements.

### Step 4: Tab 3 — KPI Benchmarks Dashboard (KPI-1 to KPI-6)
1. Click the **KPI Benchmarks (KPI-1..6)** tab.
2. Click **Run KPI Benchmark Suite**.
3. Point out the summary banner: **ALL 6 KPIS PASSED (100% Contract Compliance)**.
4. Show the cards:
   - **KPI-1:** Unknown-category misdecode rate (100% baseline -> 0% safe layer).
   - **KPI-2:** Known-category round-trip accuracy (100%).
   - **KPI-3:** Ambiguity detection recall (100%).
   - **KPI-4:** False ambiguity rate (0%).
   - **KPI-5:** Transform latency overhead p95 (bounded under 17ms).
   - **KPI-6:** Sparse-output memory overhead (75% memory reduction).

### Step 5: Tab 4 — Contract & Deliverables
1. Click the **Contract & Deliverables (D1..D7)** tab.
2. Show the Deliverables checklist: D1 through D7 are all `COMPLETE` or `APPROVED`.
3. Reference the scikit-learn issue citation `#34549`.

---

## 3. Potential Review Panel Questions & Model Answers

### Q1: Why not submit a PR to scikit-learn instead of writing a new wrapper?
> **Answer:** "A PR to scikit-learn is an ultimate upstream goal, but core library maintainers must preserve backwards compatibility with millions of existing pipelines that expect `inverse_transform` to return a 2D array matching the input shape and type without `None` or sentinel strings.
> 
> Our approach builds a **non-invasive, estimator-compliant safety layer** that gives engineers immediate protection, flexible resolution policies, and structured audit trails without breaking their existing scikit-learn workflows."

### Q2: What is the computational overhead of your ambiguity detection?
> **Answer:** "As measured in our contractual benchmark (**KPI-5**), our sub-vector analysis runs in $O(N \cdot M)$ where $N$ is the number of samples and $M$ is the number of categorical features. By optimizing vector matching to $O(1)$ argmax indexing and avoiding redundant array allocations, the p95 latency overhead is under 17ms for 1,000 samples. Furthermore, using sparse matrix representation (**KPI-6**) reduces memory allocation footprint by **74.98%**."

### Q3: Why does `handle_unknown='error'` not have this ambiguity problem?
> **Answer:** "When `handle_unknown='error'`, scikit-learn raises a `ValueError` during `transform()` whenever an unseen category is encountered. Therefore, no all-zeros vector produced by an unknown value ever reaches the encoded representation.
> 
> However, in modern web services and real-time production pipelines, `handle_unknown='ignore'` is widely configured because engineers cannot allow unexpected user inputs to crash the server. That is precisely where this silent misclassification defect manifests."

### Q4: How do you handle multi-column datasets where only some features are ambiguous?
> **Answer:** "Our detector isolates features at the **sub-vector level**. As demonstrated in our multi-column experiments and **NT-2**, if a row has 5 features where feature 1 is ambiguous and features 2 through 5 are unambiguous, the safe output retains features 2 through 5 and only withholds or flags feature 1. This prevents false ambiguity propagation."

### Q5: How do you handle missing values like `None` or `np.nan`?
> **Answer:** "Under **NT-4**, we explicitly test missing values in pandas DataFrames and NumPy arrays. In scikit-learn with `handle_unknown='ignore'`, `None` and `np.nan` are treated as unseen categories and mapped to all-zeros. Our system detects this collision and flags it as `AMBIGUOUS`, preventing silent conversion of missing values into valid category labels."
