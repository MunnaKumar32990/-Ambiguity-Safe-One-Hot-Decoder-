"""
test_baseline.py
================
Tests that reproduce the canonical Female/Male/Unknown baseline and verify
that sklearn's default behavior exhibits the research problem.

These tests document the PROBLEM, not the proposed solution.
They should PASS to confirm we correctly understand the baseline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from sklearn.preprocessing import OneHotEncoder


@pytest.fixture
def baseline_encoder():
    """Return a fitted baseline sklearn encoder."""
    enc = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
    enc.fit([["Female"], ["Male"]])
    return enc


class TestBaselineEncoding:
    """Verify that the baseline encoding matches the documented behavior."""

    def test_female_encodes_to_zero(self, baseline_encoder):
        """Female (dropped category) must encode to [0]."""
        encoded = baseline_encoder.transform([["Female"]])
        assert encoded.shape == (1, 1)
        assert encoded[0, 0] == 0.0

    def test_male_encodes_to_one(self, baseline_encoder):
        """Male (retained category) must encode to [1]."""
        encoded = baseline_encoder.transform([["Male"]])
        assert encoded.shape == (1, 1)
        assert encoded[0, 0] == 1.0

    def test_unknown_encodes_to_zero(self, baseline_encoder):
        """
        Unknown category must encode to [0] under handle_unknown='ignore'.
        This is the COLLISION — same vector as Female.
        """
        encoded = baseline_encoder.transform([["Unknown"]])
        assert encoded.shape == (1, 1)
        assert encoded[0, 0] == 0.0

    def test_female_and_unknown_share_encoded_vector(self, baseline_encoder):
        """Core research problem: Female and Unknown produce identical encoded vectors."""
        enc_female  = baseline_encoder.transform([["Female"]])[0]
        enc_unknown = baseline_encoder.transform([["Unknown"]])[0]
        np.testing.assert_array_equal(
            enc_female, enc_unknown,
            err_msg="Female and Unknown must share the same encoded vector [0]"
        )

    def test_drop_idx_is_female(self, baseline_encoder):
        """
        Verify that the encoder dropped 'Female' (index 0 in sorted categories).
        sklearn sorts categories: ['Female', 'Male'] -> Female at index 0.
        """
        drop_idx = int(baseline_encoder.drop_idx_[0])
        dropped_cat = baseline_encoder.categories_[0][drop_idx]
        assert dropped_cat == "Female"


class TestBaselineInverseTransform:
    """
    Document that sklearn's inverse_transform silently returns 'Female'
    for both Female and Unknown encoded vectors.
    """

    def test_female_roundtrip(self, baseline_encoder):
        """Female encodes and decodes correctly (round-trip)."""
        enc = baseline_encoder.transform([["Female"]])
        dec = baseline_encoder.inverse_transform(enc)
        assert dec[0][0] == "Female"

    def test_male_roundtrip(self, baseline_encoder):
        """Male encodes and decodes correctly (round-trip)."""
        enc = baseline_encoder.transform([["Male"]])
        dec = baseline_encoder.inverse_transform(enc)
        assert dec[0][0] == "Male"

    def test_unknown_decoded_as_female(self, baseline_encoder):
        """
        THE RESEARCH PROBLEM:
        sklearn decodes 'Unknown' as 'Female' — silently incorrect.
        This test CONFIRMS the bug we are solving.
        """
        enc = baseline_encoder.transform([["Unknown"]])
        dec = baseline_encoder.inverse_transform(enc)
        # sklearn returns "Female" for Unknown — this is the incorrect behavior
        assert dec[0][0] == "Female", (
            "sklearn should return 'Female' for Unknown (demonstrating the problem)"
        )
        # Ground truth is "Unknown", so this reconstruction is WRONG
        assert dec[0][0] != "Unknown"

    def test_batch_decoding_correctness(self, baseline_encoder):
        """
        Run the full baseline on all three test inputs and confirm counts.
        Expected: 2 correct, 1 incorrect.
        """
        X_test = [["Female"], ["Male"], ["Unknown"]]
        ground_truth = ["Female", "Male", "Unknown"]

        X_encoded = baseline_encoder.transform(X_test)
        X_decoded = baseline_encoder.inverse_transform(X_encoded)

        decoded_flat = [row[0] for row in X_decoded]
        correct_count = sum(d == g for d, g in zip(decoded_flat, ground_truth))
        incorrect_count = sum(d != g for d, g in zip(decoded_flat, ground_truth))

        assert correct_count == 2, f"Expected 2 correct, got {correct_count}"
        assert incorrect_count == 1, f"Expected 1 incorrect, got {incorrect_count}"
