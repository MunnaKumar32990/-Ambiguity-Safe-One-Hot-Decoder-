"""
test_conformance.py
===================
Tests for the Conformance Suite (C-01 to C-10).

Each test maps to one conformance invariant defined in src/conformance.py.
These tests independently verify the complete system's invariants as
required by the O4 "conformance suite" deliverable.
"""

import pytest
from src.conformance import ConformanceSuite, ConformanceReport


class TestConformanceSuite:
    """Verify each conformance check individually."""

    @pytest.fixture(scope="class")
    def suite(self):
        return ConformanceSuite()

    def test_c01_safe_vectors_decode_correctly(self, suite):
        result = suite.check_c01_safe_vectors_decode_correctly()
        assert result.passed, f"C-01 FAILED: {result.details}"

    def test_c02_ambiguous_vectors_handled(self, suite):
        result = suite.check_c02_ambiguous_vectors_handled()
        assert result.passed, f"C-02 FAILED: {result.details}"

    def test_c03_no_false_positives_on_clean_vectors(self, suite):
        result = suite.check_c03_no_false_positives_on_clean_vectors()
        assert result.passed, f"C-03 FAILED: {result.details}"

    def test_c04_all_zeros_collision_detected(self, suite):
        result = suite.check_c04_all_zeros_collision_detected()
        assert result.passed, f"C-04 FAILED: {result.details}"

    def test_c05_sparse_dense_parity(self, suite):
        result = suite.check_c05_sparse_dense_parity()
        assert result.passed, f"C-05 FAILED: {result.details}"

    def test_c06_pickle_serialization_parity(self, suite):
        result = suite.check_c06_pickle_serialization_parity()
        assert result.passed, f"C-06 FAILED: {result.details}"

    def test_c07_multi_column_independent_detection(self, suite):
        result = suite.check_c07_multi_column_independent_detection()
        assert result.passed, f"C-07 FAILED: {result.details}"

    def test_c08_missing_values_handled(self, suite):
        result = suite.check_c08_missing_values_handled()
        assert result.passed, f"C-08 FAILED: {result.details}"

    def test_c09_kpi_all_pass(self, suite):
        result = suite.check_c09_kpi_all_pass()
        assert result.passed, f"C-09 FAILED: {result.details}"

    def test_c10_negative_tests_all_pass(self, suite):
        result = suite.check_c10_negative_tests_all_pass()
        assert result.passed, f"C-10 FAILED: {result.details}"


class TestConformanceReport:
    """Verify the full run_all() campaign and report format."""

    @pytest.fixture(scope="class")
    def report(self):
        return ConformanceSuite().run_all()

    def test_all_checks_executed(self, report):
        assert len(report.checks) == 10

    def test_all_checks_pass(self, report):
        failing = [c.check_id for c in report.checks if not c.passed]
        assert report.all_passed, f"Failing checks: {failing}"

    def test_report_summary_structure(self, report):
        summary = report.summary()
        assert "total_checks" in summary
        assert "passed" in summary
        assert "failed" in summary
        assert "all_passed" in summary
        assert summary["total_checks"] == 10

    def test_report_to_markdown(self, report):
        md = report.to_markdown()
        assert "# Conformance Report" in md
        assert "C-01" in md
        assert "C-10" in md
        assert "✅" in md
