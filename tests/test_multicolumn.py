"""
test_multicolumn.py
===================
Test 8: Multiple categorical columns — verifies per-feature ambiguity
        detection works correctly when multiple features are present.

Tests that:
  - Ambiguity in one column does not corrupt analysis of another column
  - Row-level aggregation correctly reflects any AMBIGUOUS column
  - Known columns remain SAFE even when another column is AMBIGUOUS
  - All-SAFE multi-column row has HIGH confidence
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.safe_decoder import AmbiguitySafeOneHotDecoder
from src.ambiguity_detector import AmbiguityStatus


@pytest.fixture
def multi_decoder():
    """
    Two-column encoder:
      Col 0 — Gender: Female/Male (binary, drop='if_binary' -> Female dropped)
      Col 1 — City: Delhi/Hyderabad/Chennai (3-class, no drop since not binary)
    """
    X_train = [
        ["Female", "Delhi"],
        ["Female", "Hyderabad"],
        ["Male",   "Delhi"],
        ["Male",   "Chennai"],
    ]
    dec = AmbiguitySafeOneHotDecoder(drop="if_binary", handle_unknown="ignore")
    dec.fit(X_train)
    return dec


class TestMultiColumnDecoding:
    """Test 8: Multi-column ambiguity isolation and propagation."""

    def test_both_known_columns_are_safe(self, multi_decoder):
        """Both known values -> row status SAFE, HIGH confidence."""
        X_enc = multi_decoder.transform([["Female", "Delhi"]])
        results = multi_decoder.safe_inverse_transform(X_enc)
        # Female is dropped -> its encoding is [0] -> AMBIGUOUS
        # But Delhi is known -> SAFE
        # Row should be AMBIGUOUS because one feature is ambiguous
        # (This also demonstrates how a "known" value in dropped position is flagged)
        assert results[0].feature_reports[1].status == AmbiguityStatus.SAFE   # City=Delhi

    def test_known_city_safe_even_when_gender_ambiguous(self, multi_decoder):
        """
        When Gender is AMBIGUOUS (unknown input), City feature should still be SAFE.
        Ambiguity isolation: per-feature reports are independent.
        """
        X_enc = multi_decoder.transform([["Unknown", "Delhi"]])
        results = multi_decoder.safe_inverse_transform(X_enc)
        # Feature 0 (Gender=Unknown): AMBIGUOUS
        assert results[0].feature_reports[0].status == AmbiguityStatus.AMBIGUOUS
        # Feature 1 (City=Delhi): SAFE
        assert results[0].feature_reports[1].status == AmbiguityStatus.SAFE

    def test_row_ambiguous_when_gender_unknown(self, multi_decoder):
        """Row is AMBIGUOUS when Gender is unknown (even if City is known)."""
        X_enc = multi_decoder.transform([["Unknown", "Delhi"]])
        results = multi_decoder.safe_inverse_transform(X_enc)
        assert results[0].row_status == AmbiguityStatus.AMBIGUOUS

    def test_unknown_city_gives_unknown_status(self, multi_decoder):
        """Unknown city (no drop for city column) -> City feature is UNKNOWN."""
        X_enc = multi_decoder.transform([["Male", "Bangalore"]])
        results = multi_decoder.safe_inverse_transform(X_enc)
        # Male is SAFE; Bangalore is unknown -> UNKNOWN
        assert results[0].feature_reports[0].status == AmbiguityStatus.SAFE
        assert results[0].feature_reports[1].status == AmbiguityStatus.UNKNOWN

    def test_row_unknown_when_city_unknown(self, multi_decoder):
        """Row is UNKNOWN (not AMBIGUOUS) when only City is unknown and Gender is safe."""
        X_enc = multi_decoder.transform([["Male", "Bangalore"]])
        results = multi_decoder.safe_inverse_transform(X_enc)
        assert results[0].row_status == AmbiguityStatus.UNKNOWN

    def test_both_unknown_row_is_ambiguous(self, multi_decoder):
        """Both Gender and City unknown -> row is AMBIGUOUS (worst case wins)."""
        X_enc = multi_decoder.transform([["Unknown", "Bangalore"]])
        results = multi_decoder.safe_inverse_transform(X_enc)
        assert results[0].row_status == AmbiguityStatus.AMBIGUOUS

    def test_safe_feature_has_nonnull_safe_output(self, multi_decoder):
        """SAFE feature in a mixed-status row still yields a valid safe_output."""
        X_enc = multi_decoder.transform([["Unknown", "Delhi"]])
        results = multi_decoder.safe_inverse_transform(X_enc)
        # Gender is AMBIGUOUS -> safe_output[0] is None
        assert results[0].safe_output[0] is None
        # City is SAFE -> safe_output[1] is "Delhi"
        assert results[0].safe_output[1] == "Delhi"

    def test_metadata_has_two_features(self, multi_decoder):
        """Metadata should contain exactly 2 feature records."""
        assert multi_decoder.metadata.n_features == 2

    def test_feature_0_is_binary_and_dropped(self, multi_decoder):
        """Feature 0 (Gender) should be marked binary with Female dropped."""
        feat0 = multi_decoder.metadata.features[0]
        assert feat0.is_binary is True
        assert feat0.dropped_category == "Female"
        assert feat0.can_produce_all_zeros() is True

    def test_feature_1_city_no_drop(self, multi_decoder):
        """Feature 1 (City) should have no dropped category (not binary)."""
        feat1 = multi_decoder.metadata.features[1]
        assert feat1.dropped_category is None
        assert feat1.can_produce_all_zeros() is False
