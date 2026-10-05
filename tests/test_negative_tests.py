"""
test_negative_tests.py
======================
Unit tests for the Mandatory Negative-Test and Recovery Campaign (NT-1 to NT-5).
"""

import pytest
from src.negative_tests import NegativeTestCampaign


class TestNegativeTestsCampaign:
    def test_nt1_unknown_value_colliding_with_dropped_binary(self):
        report = NegativeTestCampaign.run_nt1()
        assert report.verdict == "PASS"
        assert report.nt_id == "NT-1"
        assert report.recovery_time_ms >= 0

    def test_nt2_multiple_dropped_categories_across_columns(self):
        report = NegativeTestCampaign.run_nt2()
        assert report.verdict == "PASS"
        assert report.nt_id == "NT-2"
        assert report.recovery_time_ms >= 0

    def test_nt3_sparse_and_dense_encoded_matrices(self):
        report = NegativeTestCampaign.run_nt3()
        assert report.verdict == "PASS"
        assert report.nt_id == "NT-3"
        assert report.recovery_time_ms >= 0

    def test_nt4_pandas_and_numpy_missing_values(self):
        report = NegativeTestCampaign.run_nt4()
        assert report.verdict == "PASS"
        assert report.nt_id == "NT-4"
        assert report.recovery_time_ms >= 0

    def test_nt5_model_persistence_and_reload(self):
        report = NegativeTestCampaign.run_nt5()
        assert report.verdict == "PASS"
        assert report.nt_id == "NT-5"
        assert report.recovery_time_ms >= 0

    def test_full_campaign_runs_and_passes_all(self):
        reports = NegativeTestCampaign.run_all()
        assert len(reports) == 5
        assert all(r.verdict == "PASS" for r in reports)
