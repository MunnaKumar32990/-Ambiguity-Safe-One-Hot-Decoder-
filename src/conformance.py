"""
conformance.py
==============
Conformance Suite for KLCAP-2026-00332.

Satisfies O4 requirement: "conformance suite and diagnostic report" as a
complete, separately testable system component.

This module defines a ConformanceSuite that checks the complete
AmbiguitySafeOneHotDecoder system against a set of invariants derived
from the capstone acceptance conditions (AC-1 to AC-4) and produces
a machine-readable ConformanceReport.

Conformance Invariants
----------------------
C-01  SAFE vectors always decode to a concrete category (non-None safe_output).
C-02  AMBIGUOUS vectors always produce None (WITHHOLD), sentinel (SENTINEL),
      or raise AmbiguityRejectionError (STRICT_REJECTION).
C-03  Known categories that produce non-zero encoded sub-vectors are never
      classified as AMBIGUOUS (no false positives on unambiguous vectors).
C-04  The all-zeros collision condition is correctly detected when
      drop is active and handle_unknown='ignore'.
C-05  Sparse and dense encoded matrices produce identical status and output.
C-06  Fitted decoder survives pickle serialization with full parity.
C-07  Multi-column ambiguity is detected independently per feature.
C-08  Missing values (None, np.nan) are handled without unhandled exceptions.
C-09  All 6 KPI metrics pass their contractual thresholds.
C-10  All NT-1 to NT-6 negative tests pass.
"""

import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from scipy import sparse

from .safe_decoder import AmbiguitySafeOneHotDecoder
from .policies import AmbiguityPolicy, AmbiguityRejectionError
from .ambiguity_detector import AmbiguityStatus
from .kpi_benchmarking import KPIEvaluator
from .negative_tests import NegativeTestCampaign


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConformanceCheck:
    """Result of a single conformance invariant check."""
    check_id: str
    description: str
    passed: bool
    details: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "description": self.description,
            "passed": self.passed,
            "details": self.details,
            "evidence": self.evidence,
        }


@dataclass
class ConformanceReport:
    """Aggregated conformance report for the full system."""
    checks: List[ConformanceCheck] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_checks": len(self.checks),
            "passed": self.pass_count,
            "failed": self.fail_count,
            "all_passed": self.all_passed,
            "check_results": [c.to_dict() for c in self.checks],
        }

    def to_markdown(self) -> str:
        """Render the report as a Markdown table."""
        lines = [
            "# Conformance Report — KLCAP-2026-00332",
            "",
            f"**Total Checks:** {len(self.checks)}  |  "
            f"**Passed:** {self.pass_count}  |  "
            f"**Failed:** {self.fail_count}  |  "
            f"**Overall:** {'✅ PASS' if self.all_passed else '❌ FAIL'}",
            "",
            "| ID | Description | Result | Details |",
            "|---|---|---|---|",
        ]
        for c in self.checks:
            icon = "✅" if c.passed else "❌"
            detail = c.details.replace("|", "\\|")[:100]
            lines.append(f"| {c.check_id} | {c.description} | {icon} | {detail} |")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conformance Suite
# ---------------------------------------------------------------------------

class ConformanceSuite:
    """
    Executes all conformance checks and returns a ConformanceReport.

    Usage
    -----
    >>> suite = ConformanceSuite()
    >>> report = suite.run_all()
    >>> print(report.to_markdown())
    """

    def run_all(self) -> ConformanceReport:
        """Execute all conformance checks and return the aggregated report."""
        report = ConformanceReport()
        for method in [
            self.check_c01_safe_vectors_decode_correctly,
            self.check_c02_ambiguous_vectors_handled,
            self.check_c03_no_false_positives_on_clean_vectors,
            self.check_c04_all_zeros_collision_detected,
            self.check_c05_sparse_dense_parity,
            self.check_c06_pickle_serialization_parity,
            self.check_c07_multi_column_independent_detection,
            self.check_c08_missing_values_handled,
            self.check_c09_kpi_all_pass,
            self.check_c10_negative_tests_all_pass,
        ]:
            try:
                result = method()
                report.checks.append(result)
            except Exception as exc:
                check_id = method.__name__.split("_")[1].upper()
                report.checks.append(ConformanceCheck(
                    check_id=check_id,
                    description=method.__doc__ or "",
                    passed=False,
                    details=f"Exception during check: {exc}",
                ))
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_c01_safe_vectors_decode_correctly(self) -> ConformanceCheck:
        """C-01: SAFE vectors always decode to a concrete (non-None) category."""
        decoder = AmbiguitySafeOneHotDecoder(drop=None, handle_unknown="ignore")
        decoder.fit([["Red"], ["Green"], ["Blue"]])
        X_enc = decoder.transform([["Red"], ["Green"], ["Blue"]])
        results = decoder.safe_inverse_transform(X_enc, policy=AmbiguityPolicy.WITHHOLD)

        violations = [
            i for i, r in enumerate(results)
            if r.row_status == AmbiguityStatus.SAFE and any(v is None for v in r.safe_output)
        ]
        passed = len(violations) == 0
        return ConformanceCheck(
            check_id="C-01",
            description="SAFE vectors always decode to a concrete category",
            passed=passed,
            details="All SAFE samples produced non-None safe_output." if passed
                    else f"Violations at sample indices: {violations}",
        )

    def check_c02_ambiguous_vectors_handled(self) -> ConformanceCheck:
        """C-02: AMBIGUOUS vectors produce None/sentinel/exception, never a silent wrong label."""
        decoder = AmbiguitySafeOneHotDecoder(drop="if_binary", handle_unknown="ignore")
        decoder.fit([["Female"], ["Male"]])
        X_enc = decoder.transform([["Unknown_Unseen"]])

        result_w = decoder.safe_inverse_transform(X_enc, policy=AmbiguityPolicy.WITHHOLD)[0]
        result_s = decoder.safe_inverse_transform(X_enc, policy=AmbiguityPolicy.SENTINEL)[0]

        withhold_ok = result_w.row_status == AmbiguityStatus.AMBIGUOUS and result_w.safe_output[0] is None
        sentinel_ok = result_s.row_status == AmbiguityStatus.AMBIGUOUS and "<AMBIGUOUS" in str(result_s.safe_output[0])

        # STRICT_REJECTION must raise
        strict_ok = False
        try:
            decoder.safe_inverse_transform(X_enc, policy=AmbiguityPolicy.STRICT_REJECTION)
        except AmbiguityRejectionError:
            strict_ok = True

        passed = withhold_ok and sentinel_ok and strict_ok
        return ConformanceCheck(
            check_id="C-02",
            description="AMBIGUOUS vectors handled safely across all policies",
            passed=passed,
            details=f"WITHHOLD={withhold_ok}, SENTINEL={sentinel_ok}, STRICT_REJECTION raises={strict_ok}",
            evidence={"withhold_output": result_w.safe_output, "sentinel_output": result_s.safe_output},
        )

    def check_c03_no_false_positives_on_clean_vectors(self) -> ConformanceCheck:
        """C-03: Clean unambiguous vectors are never flagged as AMBIGUOUS."""
        # No drop → no all-zeros collision possible → Male/Female/Other all unambiguous
        decoder = AmbiguitySafeOneHotDecoder(drop=None, handle_unknown="ignore")
        decoder.fit([["Red"], ["Green"], ["Blue"]])
        X_enc = decoder.transform([["Red"], ["Green"], ["Blue"]])
        results = decoder.safe_inverse_transform(X_enc, policy=AmbiguityPolicy.WITHHOLD)

        false_positives = [
            i for i, r in enumerate(results)
            if r.row_status == AmbiguityStatus.AMBIGUOUS
        ]
        passed = len(false_positives) == 0
        return ConformanceCheck(
            check_id="C-03",
            description="No false positives on unambiguous vectors (no drop configured)",
            passed=passed,
            details="Zero false ambiguity flags on clean known vectors." if passed
                    else f"False positive at indices: {false_positives}",
        )

    def check_c04_all_zeros_collision_detected(self) -> ConformanceCheck:
        """C-04: All-zeros collision is detected when drop is active and handle_unknown=ignore."""
        decoder = AmbiguitySafeOneHotDecoder(drop="if_binary", handle_unknown="ignore")
        decoder.fit([["Female"], ["Male"]])
        # Feed an all-zeros vector directly
        X_zeros = np.array([[0.0]])
        results = decoder.safe_inverse_transform(X_zeros, policy=AmbiguityPolicy.WITHHOLD)
        detected = results[0].row_status == AmbiguityStatus.AMBIGUOUS
        return ConformanceCheck(
            check_id="C-04",
            description="All-zeros collision detected when drop + ignore are both active",
            passed=detected,
            details=f"Status for [0.0]: {results[0].row_status.value}",
            evidence={"encoded": [0.0], "status": results[0].row_status.value},
        )

    def check_c05_sparse_dense_parity(self) -> ConformanceCheck:
        """C-05: Sparse and dense encoded matrices produce identical results."""
        train = [["Alpha"], ["Beta"], ["Gamma"]]
        test = [["Alpha"], ["Delta_Unseen"]]

        dec_dense = AmbiguitySafeOneHotDecoder(drop="first", handle_unknown="ignore", sparse_output=False)
        dec_sparse = AmbiguitySafeOneHotDecoder(drop="first", handle_unknown="ignore", sparse_output=True)
        dec_dense.fit(train)
        dec_sparse.fit(train)

        X_dense = dec_dense.transform(test)
        X_sparse = dec_sparse.transform(test)

        r_dense = dec_dense.safe_inverse_transform(X_dense, policy=AmbiguityPolicy.SENTINEL)
        r_sparse = dec_sparse.safe_inverse_transform(X_sparse, policy=AmbiguityPolicy.SENTINEL)

        parity = all(
            r_dense[i].row_status == r_sparse[i].row_status
            and r_dense[i].safe_output == r_sparse[i].safe_output
            for i in range(len(test))
        )
        return ConformanceCheck(
            check_id="C-05",
            description="Sparse and dense encoded matrices produce identical statuses and outputs",
            passed=parity,
            details="100% parity achieved." if parity
                    else "Discrepancy between sparse and dense outputs.",
            evidence={
                "dense_statuses": [r.row_status.value for r in r_dense],
                "sparse_statuses": [r.row_status.value for r in r_sparse],
            },
        )

    def check_c06_pickle_serialization_parity(self) -> ConformanceCheck:
        """C-06: Fitted decoder survives pickle serialization with full decoding parity."""
        train = [["Cat"], ["Dog"], ["Bird"]]
        test = [["Cat"], ["Elephant_Unseen"]]

        decoder = AmbiguitySafeOneHotDecoder(drop="first", handle_unknown="ignore")
        decoder.fit(train)
        X_enc = decoder.transform(test)

        serialized = pickle.dumps(decoder)
        reloaded: AmbiguitySafeOneHotDecoder = pickle.loads(serialized)

        r_orig = decoder.safe_inverse_transform(X_enc, policy=AmbiguityPolicy.SENTINEL)
        r_rel = reloaded.safe_inverse_transform(X_enc, policy=AmbiguityPolicy.SENTINEL)

        parity = all(
            r_orig[i].row_status == r_rel[i].row_status
            and r_orig[i].safe_output == r_rel[i].safe_output
            for i in range(len(test))
        )
        return ConformanceCheck(
            check_id="C-06",
            description="Pickle serialization preserves decoder state and decoding parity",
            passed=parity,
            details=f"Serialized size: {len(serialized)} bytes. Parity: {parity}.",
            evidence={"pickle_bytes": len(serialized), "reloaded_is_fitted": reloaded.is_fitted},
        )

    def check_c07_multi_column_independent_detection(self) -> ConformanceCheck:
        """C-07: Multi-column ambiguity is detected independently per feature column."""
        train = [["Female", "London"], ["Male", "Paris"], ["Female", "Tokyo"]]
        test = [["Unknown_A", "London"], ["Female", "Unknown_B"]]

        decoder = AmbiguitySafeOneHotDecoder(drop="if_binary", handle_unknown="ignore")
        decoder.fit(train)
        X_enc = decoder.transform(test)
        results = decoder.safe_inverse_transform(X_enc, policy=AmbiguityPolicy.WITHHOLD)

        # Row 0: Gender is ambiguous, City is safe
        r0 = results[0]
        col0_ambiguous = r0.feature_reports[0].status == AmbiguityStatus.AMBIGUOUS
        col1_safe = r0.feature_reports[1].status == AmbiguityStatus.SAFE
        col1_has_value = r0.safe_output[1] is not None

        # Row 1: Gender is ambiguous (Female encodes same as dropped), City is unknown
        r1 = results[1]
        col0_ambiguous_r1 = r1.feature_reports[0].status == AmbiguityStatus.AMBIGUOUS

        passed = col0_ambiguous and col1_safe and col1_has_value and col0_ambiguous_r1
        return ConformanceCheck(
            check_id="C-07",
            description="Multi-column ambiguity detected independently; safe columns preserve output",
            passed=passed,
            details=(
                f"Row0: gender={r0.feature_reports[0].status.value}, "
                f"city={r0.feature_reports[1].status.value}, "
                f"city_output='{r0.safe_output[1]}'"
            ),
            evidence={
                "row0_statuses": [rep.status.value for rep in r0.feature_reports],
                "row0_safe_output": r0.safe_output,
            },
        )

    def check_c08_missing_values_handled(self) -> ConformanceCheck:
        """C-08: Missing values (None, np.nan) are handled without exceptions."""
        train_df = pd.DataFrame({"Cat": ["A", "B", "C"]})
        test_df = pd.DataFrame({"Cat": ["A", None, np.nan]})

        decoder = AmbiguitySafeOneHotDecoder(drop="first", handle_unknown="ignore")
        decoder.fit(train_df)

        try:
            X_enc = decoder.transform(test_df)
            results = decoder.safe_inverse_transform(X_enc, policy=AmbiguityPolicy.SENTINEL)
            passed = len(results) == 3
            details = "No exception. All 3 samples processed."
        except Exception as exc:
            passed = False
            details = f"Exception: {exc}"

        return ConformanceCheck(
            check_id="C-08",
            description="Missing values (None, np.nan) processed without unhandled exceptions",
            passed=passed,
            details=details,
        )

    def check_c09_kpi_all_pass(self) -> ConformanceCheck:
        """C-09: All 6 KPI metrics pass their contractual thresholds."""
        evaluator = KPIEvaluator(n_samples=200, random_state=42)
        result = evaluator.run_all_benchmarks()
        all_passed = result["all_passed"]
        measurements = result["measurements"]

        failing = [m["kpi_id"] for m in measurements if not m["pass_verdict"]]
        return ConformanceCheck(
            check_id="C-09",
            description="All 6 KPIs (KPI-1 to KPI-6) pass contractual thresholds",
            passed=all_passed,
            details="All KPIs PASS." if all_passed else f"Failing KPIs: {failing}",
            evidence={m["kpi_id"]: m["pass_verdict"] for m in measurements},
        )

    def check_c10_negative_tests_all_pass(self) -> ConformanceCheck:
        """C-10: All NT-1 to NT-6 negative tests pass."""
        reports = NegativeTestCampaign.run_all()
        all_passed = all(r.verdict == "PASS" for r in reports)
        failing = [r.nt_id for r in reports if r.verdict != "PASS"]
        return ConformanceCheck(
            check_id="C-10",
            description="All mandatory negative tests (NT-1 to NT-6) pass",
            passed=all_passed,
            details=f"All {len(reports)} NTs PASS." if all_passed
                    else f"Failing NTs: {failing}",
            evidence={r.nt_id: r.verdict for r in reports},
        )
