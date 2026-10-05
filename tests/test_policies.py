"""
test_policies.py
================
Tests for Deliverable D3: Ambiguity Resolution Policies.
Covers:
  1. WITHHOLD (None replacement)
  2. SENTINEL (informative string injection)
  3. PROVENANCE_SIDE_CHANNEL (structured metadata channel)
  4. STRICT_REJECTION (AmbiguityRejectionError)
"""

import pytest
import numpy as np
from src.safe_decoder import AmbiguitySafeOneHotDecoder
from src.policies import AmbiguityPolicy, AmbiguityRejectionError
from src.ambiguity_detector import AmbiguityStatus


class TestAmbiguityPolicies:
    @pytest.fixture
    def fitted_binary_decoder(self):
        train = [["Female"], ["Male"]]
        decoder = AmbiguitySafeOneHotDecoder(drop="if_binary", handle_unknown="ignore")
        decoder.fit(train)
        return decoder

    def test_withhold_policy_outputs_none_on_ambiguous(self, fitted_binary_decoder):
        decoder = fitted_binary_decoder
        test = [["Female"], ["Male"], ["Unknown"]]
        X_enc = decoder.transform(test)
        results = decoder.safe_inverse_transform(X_enc, policy=AmbiguityPolicy.WITHHOLD)

        assert results[0].safe_output == [None]  # Female collides with unknown -> AMBIGUOUS
        assert results[1].safe_output == ["Male"]  # Male is non-zero -> SAFE
        assert results[2].safe_output == [None]  # Unknown maps to zeros -> AMBIGUOUS

    def test_sentinel_policy_injects_sentinel_string(self, fitted_binary_decoder):
        decoder = fitted_binary_decoder
        test = [["Female"], ["Male"], ["Unknown"]]
        X_enc = decoder.transform(test)
        results = decoder.safe_inverse_transform(X_enc, policy=AmbiguityPolicy.SENTINEL)

        # Ambiguous positions should have sentinel formatting
        assert "<AMBIGUOUS" in results[0].safe_output[0]
        assert results[1].safe_output == ["Male"]
        assert "<AMBIGUOUS" in results[2].safe_output[0]

    def test_provenance_side_channel_retains_prediction_with_audit_channel(self, fitted_binary_decoder):
        decoder = fitted_binary_decoder
        test = [["Female"], ["Male"], ["Unknown"]]
        X_enc = decoder.transform(test)
        results = decoder.safe_inverse_transform(X_enc, policy=AmbiguityPolicy.PROVENANCE_SIDE_CHANNEL)

        # Safe output returns the raw prediction but attaches structured audit metadata
        assert results[0].safe_output == ["Female"]
        assert results[0].provenance["has_ambiguity"] is True
        assert results[0].provenance["confidence"] == "LOW"
        assert results[0].provenance["feature_provenance"][0]["is_ambiguous"] is True

        assert results[1].safe_output == ["Male"]
        assert results[1].provenance["has_ambiguity"] is False
        assert results[1].provenance["confidence"] == "HIGH"

        assert results[2].safe_output == ["Female"]  # sklearn's raw silent guess
        assert results[2].provenance["has_ambiguity"] is True

    def test_strict_rejection_raises_exception_on_ambiguity(self, fitted_binary_decoder):
        decoder = fitted_binary_decoder
        test_safe = [["Male"]]
        X_safe = decoder.transform(test_safe)
        res = decoder.safe_inverse_transform(X_safe, policy=AmbiguityPolicy.STRICT_REJECTION)
        assert res[0].safe_output == ["Male"]

        test_ambiguous = [["Unknown"]]
        X_amb = decoder.transform(test_ambiguous)
        with pytest.raises(AmbiguityRejectionError) as exc_info:
            decoder.safe_inverse_transform(X_amb, policy=AmbiguityPolicy.STRICT_REJECTION)

        assert exc_info.value.sample_index == 0
        assert exc_info.value.feature_index == 0
        assert "Female" in exc_info.value.possible_values
