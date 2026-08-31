"""
test_ambiguity_detector.py
==========================
Tests for AmbiguityDetector — the core metadata-driven analysis engine.

Verifies:
  - Correct SAFE classification for known round-trip categories
  - Correct AMBIGUOUS classification for all-zeros with drop+ignore_unknown
  - Correct UNKNOWN classification for all-zeros without drop
  - Non-zero vectors correctly identified as SAFE
  - Multi-class behavior
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from src.encoder import MetadataAwareEncoder
from src.ambiguity_detector import AmbiguityDetector, AmbiguityStatus


def make_detector(X_train, drop, handle_unknown):
    """Convenience: fit encoder and return (encoder, detector)."""
    enc = MetadataAwareEncoder(drop=drop, handle_unknown=handle_unknown)
    enc.fit(X_train)
    detector = AmbiguityDetector(enc.metadata)
    return enc, detector


class TestBinaryAmbiguity:
    """Tests on binary features with drop='if_binary'."""

    @pytest.fixture
    def binary_setup(self):
        enc, det = make_detector(
            [["Female"], ["Male"]],
            drop="if_binary",
            handle_unknown="ignore",
        )
        return enc, det

    def test_female_known_is_safe(self, binary_setup):
        """Female (the dropped category, all-zeros vector) with drop+ignore -> AMBIGUOUS."""
        enc, det = binary_setup
        vec = enc.transform([["Female"]])[0]      # Should be [0.]
        decoded_row = list(enc.inverse_transform([vec])[0])
        reports = det.analyze_vector(vec, decoded_row)
        # Female encodes to all-zeros -> AMBIGUOUS (same as unknown)
        assert reports[0].status == AmbiguityStatus.AMBIGUOUS

    def test_male_is_safe(self, binary_setup):
        """Male (non-dropped, vector=[1]) -> SAFE."""
        enc, det = binary_setup
        vec = enc.transform([["Male"]])[0]        # Should be [1.]
        decoded_row = list(enc.inverse_transform([vec])[0])
        reports = det.analyze_vector(vec, decoded_row)
        assert reports[0].status == AmbiguityStatus.SAFE

    def test_unknown_is_ambiguous(self, binary_setup):
        """Unknown -> encodes to [0] -> AMBIGUOUS (collision with dropped Female)."""
        enc, det = binary_setup
        vec = enc.transform([["Unknown"]])[0]     # Should be [0.]
        decoded_row = list(enc.inverse_transform([vec])[0])
        reports = det.analyze_vector(vec, decoded_row)
        assert reports[0].status == AmbiguityStatus.AMBIGUOUS

    def test_possible_values_include_dropped_and_unseen(self, binary_setup):
        """When AMBIGUOUS, possible_values must include the dropped category and <UNSEEN>."""
        enc, det = binary_setup
        vec = enc.transform([["Unknown"]])[0]
        decoded_row = list(enc.inverse_transform([vec])[0])
        reports = det.analyze_vector(vec, decoded_row)
        assert "Female" in reports[0].possible_values
        assert "<UNSEEN>" in reports[0].possible_values

    def test_aggregate_safe_when_all_safe(self, binary_setup):
        """Aggregation of all-SAFE features -> row status SAFE."""
        enc, det = binary_setup
        vec = enc.transform([["Male"]])[0]
        decoded_row = list(enc.inverse_transform([vec])[0])
        reports = det.analyze_vector(vec, decoded_row)
        agg = det.aggregate_row_status(reports)
        assert agg == AmbiguityStatus.SAFE

    def test_aggregate_ambiguous_when_any_ambiguous(self, binary_setup):
        """Aggregation with any AMBIGUOUS feature -> row status AMBIGUOUS."""
        enc, det = binary_setup
        vec = enc.transform([["Unknown"]])[0]
        decoded_row = list(enc.inverse_transform([vec])[0])
        reports = det.analyze_vector(vec, decoded_row)
        agg = det.aggregate_row_status(reports)
        assert agg == AmbiguityStatus.AMBIGUOUS


class TestNoDropAmbiguity:
    """Binary feature with drop=None — unknowns are UNKNOWN not AMBIGUOUS."""

    @pytest.fixture
    def no_drop_setup(self):
        enc, det = make_detector(
            [["Female"], ["Male"]],
            drop=None,
            handle_unknown="ignore",
        )
        return enc, det

    def test_known_female_safe_no_drop(self, no_drop_setup):
        """Female -> encodes to [1,0] -> SAFE (no dropped category)."""
        enc, det = no_drop_setup
        vec = enc.transform([["Female"]])[0]
        decoded_row = list(enc.inverse_transform([vec])[0])
        reports = det.analyze_vector(vec, decoded_row)
        assert reports[0].status == AmbiguityStatus.SAFE

    def test_unknown_is_unknown_not_ambiguous_no_drop(self, no_drop_setup):
        """Unknown -> all-zeros sub-vector but no dropped category -> UNKNOWN."""
        enc, det = no_drop_setup
        vec = enc.transform([["Unknown"]])[0]
        decoded_row = list(enc.inverse_transform([vec])[0])
        reports = det.analyze_vector(vec, decoded_row)
        assert reports[0].status == AmbiguityStatus.UNKNOWN


class TestMulticlassAmbiguity:
    """3-class features with different drop settings."""

    def test_multiclass_no_drop_unknown_is_unknown(self):
        """3-class + drop=None: unknown -> UNKNOWN (not ambiguous with known)."""
        enc, det = make_detector(
            [["Red"], ["Green"], ["Blue"]],
            drop=None,
            handle_unknown="ignore",
        )
        vec = enc.transform([["Yellow"]])[0]    # All-zeros, no drop
        decoded_row = list(enc.inverse_transform([vec])[0])
        reports = det.analyze_vector(vec, decoded_row)
        assert reports[0].status == AmbiguityStatus.UNKNOWN

    def test_multiclass_drop_first_unknown_is_ambiguous(self):
        """3-class + drop='first': unknown -> AMBIGUOUS (collides with dropped cat)."""
        enc, det = make_detector(
            [["Red"], ["Green"], ["Blue"]],
            drop="first",
            handle_unknown="ignore",
        )
        vec = enc.transform([["Yellow"]])[0]    # All-zeros, drop active
        decoded_row = list(enc.inverse_transform([vec])[0])
        reports = det.analyze_vector(vec, decoded_row)
        assert reports[0].status == AmbiguityStatus.AMBIGUOUS

    def test_known_multiclass_categories_are_safe(self):
        """
        Non-dropped known 3-class categories produce non-zero vectors -> SAFE.

        sklearn sorts categories alphabetically: [Blue, Green, Red].
        drop='first' drops 'Blue' (index 0).
        So 'Green' and 'Red' are non-dropped and should be SAFE.
        We dynamically find the dropped category from metadata to avoid
        hardcoding assumptions about sort order.
        """
        enc, det = make_detector(
            [["Red"], ["Green"], ["Blue"]],
            drop="first",
            handle_unknown="ignore",
        )
        # Find the actual dropped category from encoder metadata
        dropped_cat = enc.metadata.features[0].dropped_category  # 'Blue' (sorted first)
        all_training_cats = ["Red", "Green", "Blue"]
        non_dropped_cats = [c for c in all_training_cats if c != dropped_cat]

        for cat in non_dropped_cats:
            vec = enc.transform([[cat]])[0]
            decoded_row = list(enc.inverse_transform([vec])[0])
            reports = det.analyze_vector(vec, decoded_row)
            assert reports[0].status == AmbiguityStatus.SAFE, \
                f"Expected SAFE for non-dropped '{cat}', got {reports[0].status}. " \
                f"(Dropped category is '{dropped_cat}')"

    def test_feature_can_produce_all_zeros_method(self):
        """FeatureMetadata.can_produce_all_zeros() correctly identifies ambiguity potential."""
        enc, _ = make_detector(
            [["Female"], ["Male"]],
            drop="if_binary",
            handle_unknown="ignore",
        )
        feat = enc.metadata.features[0]
        assert feat.can_produce_all_zeros() is True

    def test_no_ambiguity_potential_no_drop(self):
        """can_produce_all_zeros() returns False when no drop is configured."""
        enc, _ = make_detector(
            [["Female"], ["Male"]],
            drop=None,
            handle_unknown="ignore",
        )
        feat = enc.metadata.features[0]
        assert feat.can_produce_all_zeros() is False
