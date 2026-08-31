"""
test_safe_decoder.py
====================
Tests for AmbiguitySafeOneHotDecoder — the main public API.

Covers:
  Test 1: Known Female -> correct round-trip (Female)
  Test 2: Known Male -> correct round-trip (Male)
  Test 3: Unknown category -> ambiguity detected, safe_output=None
  Test 4: Dropped category correctly identified in possible_values
  Test 5: Known categories never falsely marked ambiguous (male)
  Test 6: Multi-class known values all SAFE
  Test 7: Multi-class unknown value properly classified
  Test 9: Different encoder configurations behave correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from src.safe_decoder import AmbiguitySafeOneHotDecoder
from src.ambiguity_detector import AmbiguityStatus


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def gender_decoder():
    """Fitted decoder for binary Gender feature."""
    dec = AmbiguitySafeOneHotDecoder(drop="if_binary", handle_unknown="ignore")
    dec.fit([["Female"], ["Male"]])
    return dec


@pytest.fixture
def color_decoder():
    """Fitted decoder for 3-class Color feature, no drop."""
    dec = AmbiguitySafeOneHotDecoder(drop=None, handle_unknown="ignore")
    dec.fit([["Red"], ["Green"], ["Blue"]])
    return dec


@pytest.fixture
def color_decoder_drop_first():
    """Fitted decoder for 3-class Color feature, drop='first'."""
    dec = AmbiguitySafeOneHotDecoder(drop="first", handle_unknown="ignore")
    dec.fit([["Red"], ["Green"], ["Blue"]])
    return dec


# -----------------------------------------------------------------------------
# Test 1 & 2: Binary known categories
# -----------------------------------------------------------------------------

class TestKnownCategoryRoundTrip:
    """
    Test 1: Female -> encode -> safe_decode -> Female (round-trip)
    Test 2: Male   -> encode -> safe_decode -> Male   (round-trip)
    """

    def test_male_known_safe_output(self, gender_decoder):
        """Test 2: Male (non-dropped, unambiguous) -> safe_output = 'Male'."""
        X_enc = gender_decoder.transform([["Male"]])
        results = gender_decoder.safe_inverse_transform(X_enc)
        assert results[0].row_status == AmbiguityStatus.SAFE
        assert results[0].safe_output[0] == "Male"
        assert results[0].confidence == "HIGH"

    def test_male_not_falsely_ambiguous(self, gender_decoder):
        """Test 5: Male must not be marked AMBIGUOUS (no false positives)."""
        X_enc = gender_decoder.transform([["Male"]])
        results = gender_decoder.safe_inverse_transform(X_enc)
        assert results[0].row_status != AmbiguityStatus.AMBIGUOUS


# -----------------------------------------------------------------------------
# Test 3: Unknown category -> ambiguity detected
# -----------------------------------------------------------------------------

class TestUnknownCategoryDetection:
    """Test 3: Unknown/unseen category -> ambiguity detected, safe_output withheld."""

    def test_unknown_gender_status_is_ambiguous(self, gender_decoder):
        """
        Test 3: Unknown gender -> encodes to [0] -> AMBIGUOUS with dropped Female.
        """
        X_enc = gender_decoder.transform([["Unknown"]])
        results = gender_decoder.safe_inverse_transform(X_enc)
        assert results[0].row_status == AmbiguityStatus.AMBIGUOUS

    def test_unknown_gender_safe_output_is_none(self, gender_decoder):
        """When AMBIGUOUS, safe_output must be None (not silently Female)."""
        X_enc = gender_decoder.transform([["Unknown"]])
        results = gender_decoder.safe_inverse_transform(X_enc)
        assert results[0].safe_output[0] is None

    def test_unknown_gender_confidence_is_low(self, gender_decoder):
        """Ambiguous result has LOW confidence."""
        X_enc = gender_decoder.transform([["Unknown"]])
        results = gender_decoder.safe_inverse_transform(X_enc)
        assert results[0].confidence == "LOW"


# -----------------------------------------------------------------------------
# Test 4: Dropped category identified in possible values
# -----------------------------------------------------------------------------

class TestDroppedCategoryTracking:
    """Test 4: Dropped category must appear in possible_values for ambiguous cases."""

    def test_dropped_female_in_possible_values(self, gender_decoder):
        """
        Test 4: When Unknown encodes to [0], possible_values must include 'Female'
        (the dropped category) to make the ambiguity explicit.
        """
        X_enc = gender_decoder.transform([["Unknown"]])
        results = gender_decoder.safe_inverse_transform(X_enc)
        assert "Female" in results[0].possible_values[0]

    def test_unseen_sentinel_in_possible_values(self, gender_decoder):
        """Test 4: possible_values must also include '<UNSEEN>' as a candidate."""
        X_enc = gender_decoder.transform([["Unknown"]])
        results = gender_decoder.safe_inverse_transform(X_enc)
        assert "<UNSEEN>" in results[0].possible_values[0]

    def test_metadata_records_dropped_category(self, gender_decoder):
        """Test 4: Decoder metadata correctly records Female as dropped."""
        feat = gender_decoder.metadata.features[0]
        assert feat.dropped_category == "Female"


# -----------------------------------------------------------------------------
# Test 6 & 7: Multi-class categories
# -----------------------------------------------------------------------------

class TestMulticlassDecoding:
    """
    Test 6: Multi-class known values all SAFE.
    Test 7: Multi-class unknown value properly classified.
    """

    def test_all_known_colors_safe(self, color_decoder):
        """Test 6: All training colors (Red, Green, Blue) -> SAFE status."""
        for color in ["Red", "Green", "Blue"]:
            X_enc = color_decoder.transform([[color]])
            results = color_decoder.safe_inverse_transform(X_enc)
            assert results[0].row_status == AmbiguityStatus.SAFE, \
                f"Known color '{color}' should be SAFE, got {results[0].row_status}"

    def test_unknown_color_is_unknown(self, color_decoder):
        """Test 7: Unknown color (Yellow) -> UNKNOWN (no ambiguity with known cat since drop=None)."""
        X_enc = color_decoder.transform([["Yellow"]])
        results = color_decoder.safe_inverse_transform(X_enc)
        assert results[0].row_status == AmbiguityStatus.UNKNOWN

    def test_unknown_color_drop_first_is_ambiguous(self, color_decoder_drop_first):
        """Test 7: Unknown color with drop='first' -> AMBIGUOUS."""
        X_enc = color_decoder_drop_first.transform([["Yellow"]])
        results = color_decoder_drop_first.safe_inverse_transform(X_enc)
        assert results[0].row_status == AmbiguityStatus.AMBIGUOUS


# -----------------------------------------------------------------------------
# Test 9: Different encoder configurations
# -----------------------------------------------------------------------------

class TestEncoderConfigurations:
    """Test 9: Behavior differs predictably across drop configurations."""

    def test_drop_none_unknown_is_unknown_not_ambiguous(self):
        """drop=None: unknown -> UNKNOWN (no collision with known dropped category)."""
        dec = AmbiguitySafeOneHotDecoder(drop=None, handle_unknown="ignore")
        dec.fit([["Female"], ["Male"]])
        X_enc = dec.transform([["Unknown"]])
        results = dec.safe_inverse_transform(X_enc)
        assert results[0].row_status == AmbiguityStatus.UNKNOWN

    def test_drop_if_binary_unknown_is_ambiguous(self):
        """drop='if_binary': binary unknown -> AMBIGUOUS."""
        dec = AmbiguitySafeOneHotDecoder(drop="if_binary", handle_unknown="ignore")
        dec.fit([["Female"], ["Male"]])
        X_enc = dec.transform([["Unknown"]])
        results = dec.safe_inverse_transform(X_enc)
        assert results[0].row_status == AmbiguityStatus.AMBIGUOUS

    def test_drop_first_known_retained_is_safe(self):
        """drop='first': second known category -> SAFE."""
        dec = AmbiguitySafeOneHotDecoder(drop="first", handle_unknown="ignore")
        dec.fit([["Female"], ["Male"]])
        X_enc = dec.transform([["Male"]])
        results = dec.safe_inverse_transform(X_enc)
        assert results[0].row_status == AmbiguityStatus.SAFE

    def test_decoder_not_fitted_raises(self):
        """Calling transform before fit raises RuntimeError."""
        dec = AmbiguitySafeOneHotDecoder()
        with pytest.raises(RuntimeError):
            dec.transform([["Female"]])

    def test_results_to_dataframe(self):
        """results_to_dataframe produces a non-empty DataFrame."""
        import pandas as pd
        dec = AmbiguitySafeOneHotDecoder(drop="if_binary", handle_unknown="ignore")
        dec.fit([["Female"], ["Male"]])
        X_enc = dec.transform([["Female"], ["Male"], ["Unknown"]])
        results = dec.safe_inverse_transform(X_enc)
        df = dec.results_to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3    # 3 samples × 1 feature = 3 rows


# -----------------------------------------------------------------------------
# Integration test
# -----------------------------------------------------------------------------

class TestFullPipelineIntegration:
    """End-to-end pipeline from raw data to safe decoding results."""

    def test_canonical_pipeline(self):
        """
        Full pipeline test using the canonical Female/Male/Unknown example.
        Expected:
          Female  -> AMBIGUOUS (dropped category shares vector with unknown)
          Male    -> SAFE
          Unknown -> AMBIGUOUS
        """
        X_train = [["Female"], ["Male"]]
        X_test  = [["Female"], ["Male"], ["Unknown"]]

        dec = AmbiguitySafeOneHotDecoder(drop="if_binary", handle_unknown="ignore")
        dec.fit(X_train)
        X_enc = dec.transform(X_test)
        results = dec.safe_inverse_transform(X_enc)

        assert len(results) == 3
        assert results[0].row_status == AmbiguityStatus.AMBIGUOUS   # Female (dropped)
        assert results[1].row_status == AmbiguityStatus.SAFE         # Male
        assert results[2].row_status == AmbiguityStatus.AMBIGUOUS    # Unknown
        assert results[1].safe_output[0] == "Male"
        assert results[0].safe_output[0] is None
        assert results[2].safe_output[0] is None
