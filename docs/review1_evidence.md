# Review 1 Evidence Mapping
## Ambiguity-Safe Inverse Decoding for One-Hot Encoded Categorical Data

This document explicitly maps implemented evidence to each Review-1 evaluation criterion.

---

## Criterion 1: Topic Selection & Problem Identification

**What we can claim:**
The project addresses a concrete, reproducible problem in the most widely
used categorical preprocessing library (scikit-learn). The problem is
demonstrated with a 3-line reproducible example and verified experimentally.

**Evidence files:**
- `experiments/baseline_reproduction.py` — runs and prints the exact problem
- `tests/test_baseline.py::TestBaselineInverseTransform::test_unknown_decoded_as_female`
- `demo.py` Section 3

**What to show during review:**
Run `python experiments/baseline_reproduction.py` and point to the line:
```
Unknown | [0] | Female | ✗ INCORRECT (Ambiguous)
```

**Verbal explanation:**
> "We identified that when sklearn's OneHotEncoder is configured with
> drop='if_binary' and handle_unknown='ignore', the inverse_transform
> silently returns an incorrect category for unseen inputs. Our project
> detects and prevents this silent error."

---

## Criterion 2: Literature Review

**What we can claim:**
6 real publications reviewed covering feature encoding, preprocessing,
and unknown value handling. Research gap clearly identified.

**Evidence files:**
- `docs/literature_review.md` — complete review with gap derivation

**What to show:**
- The table comparing what literature covers vs. what is missing.
- The formal research gap statement (Section 3.3).

**Verbal explanation:**
> "We reviewed both foundational sklearn documentation and categorical
> encoding literature. None of the reviewed works explicitly addresses
> the ambiguity in inverse reconstruction under the drop+ignore combination."

---

## Criterion 3: Research Gap & Objectives

**What we can claim:**
A specific, defensible research gap:
> "The ambiguity introduced during inverse reconstruction under
> drop='if_binary' + handle_unknown='ignore' is not explicitly surfaced
> to users in existing sklearn workflows."

5 clear objectives, directly traceable to the gap.

**Evidence files:**
- `docs/literature_review.md` — Section 3
- `docs/methodology.md` — Section 6 (requirements table)
- `demo.py` Section 8

**What to show:**
Read the gap statement and the 5 objectives aloud from `demo.py` output.

---

## Criterion 4: Research Specification

**What we can claim:**
- 10 Functional Requirements (FR1–FR10)
- 8 Non-Functional Requirements
- Formal ambiguity condition: `can_produce_all_zeros()`
- 3 alternative strategies documented and compared

**Evidence files:**
- `docs/methodology.md` — Sections 2, 5, 6
- `src/metadata.py::FeatureMetadata.can_produce_all_zeros()`

**What to show:**
The requirements table in `docs/methodology.md`.

---

## Criterion 5: System Architecture & Design

**What we can claim:**
5-module architecture with clear separation of concerns. Architecture
diagram maps directly to actual source files.

**Evidence files:**
- `docs/architecture.md` — complete architecture with ASCII diagram and data flow
- `src/` — 5 modules: `metadata.py`, `encoder.py`, `ambiguity_detector.py`,
  `safe_decoder.py`, `evaluator.py`

**What to show:**
The ASCII architecture diagram from `docs/architecture.md`.

**Verbal explanation:**
> "The architecture separates encoding, metadata extraction, ambiguity
> analysis, safe decoding, and evaluation into five independent modules.
> Each module has a single responsibility."

---

## Criterion 6: Proposed Methodology

**What we can claim:**
- Investigated 3 strategies with pros/cons comparison
- Selected Strategy 3 (Strict Ambiguity Detection) with justification
- Implemented formal algorithm with pseudocode
- Algorithm is metadata-driven (not hard-coded to specific values)

**Evidence files:**
- `docs/methodology.md` — Sections 2–4
- `src/ambiguity_detector.py::AmbiguityDetector._analyze_feature()`

**What to show:**
The strategy comparison table and the pseudocode in `docs/methodology.md`.

**Verbal explanation:**
> "We compared three strategies: sentinel encoding, provenance tracking,
> and strict ambiguity detection. We selected Strategy 3 because it requires
> no encoder modification, is sklearn-compatible, and generalizes to any
> encoded vector."

---

## Criterion 7: Initial Implementation & Progress

**What we can claim:**
- Fully working prototype with 5 source modules
- 5 experiments covering binary, multi-class, multi-column, and config variants
- 4 test files with 25+ individual tests, all passing
- CSV result outputs from actual runs
- 3 matplotlib plots from real data

**Evidence files:**
```
src/
    metadata.py, encoder.py, ambiguity_detector.py,
    safe_decoder.py, evaluator.py

experiments/
    baseline_reproduction.py  → results/baseline_results.csv
    experiment_binary.py      → results/experiment_binary_metrics.csv
    experiment_multiclass.py  → results/experiment_multiclass_metrics.csv
    experiment_multicolumn.py → results/experiment_multicolumn_metrics.csv
    experiment_configs.py     → results/experiment_configs_metrics.csv
    run_all.py                → results/experiment_summary.csv
                              → results/plots/

tests/
    test_baseline.py
    test_ambiguity_detector.py
    test_safe_decoder.py
    test_multicolumn.py
```

**What to show:**
1. Run `python demo.py` — live output
2. Run `pytest --tb=short -q` — tests pass
3. Show `results/experiment_summary.csv`
4. Show `results/plots/baseline_vs_proposed.png`

---

## Criterion 8: Presentation & Technical Q&A

**Likely questions and answers (15 covered in README):**

**Q: Why does sklearn return Female for Unknown?**
A: Because `handle_unknown='ignore'` maps any unseen category to an
all-zeros vector, and `drop='if_binary'` also maps the dropped category
(Female) to all-zeros. Both share the same representation, so
inverse_transform cannot distinguish them.

**Q: Why is this a problem?**
A: Because sklearn silently returns the dropped category's name without
warning that the input was actually unseen. Applications relying on
inverse_transform for data recovery will silently get wrong values.

**Q: How does your detector know about the ambiguity without seeing the input?**
A: It reads `encoder.drop_idx_` and `encoder.categories_` after fitting
to determine whether any dropped category exists and whether
`handle_unknown='ignore'` is set. If both are true, all-zeros vectors
are structurally ambiguous regardless of what the original input was.

**Q: What is the output of your system for Female?**
A: Female also produces an AMBIGUOUS result because its encoding (all-zeros)
is indistinguishable from an unseen category's encoding. This is a
conservative but correct response — we cannot know from the encoded vector
alone whether the original input was the dropped known category or an
unseen category.

**Q: Is this problem configuration-dependent?**
A: Yes. With `drop=None`, unknown categories produce UNKNOWN (not AMBIGUOUS)
because there is no known dropped category sharing the all-zeros representation.
The ambiguity specifically requires both `drop` and `handle_unknown='ignore'`.

---

## Demonstration Commands for Review Day

```bash
# Install dependencies
pip install -r requirements.txt

# Show the problem and proposed solution (main demo)
python demo.py

# Show the canonical baseline
python experiments/baseline_reproduction.py

# Run all experiments and generate CSVs + plots
python experiments/run_all.py

# Run all tests
pytest --tb=short -v
```
