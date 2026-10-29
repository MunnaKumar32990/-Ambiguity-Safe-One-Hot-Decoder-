"""
test_provenance.py
==================
Tests for the ProvenanceRecord module (src/provenance.py).

Verifies that transformation provenance is correctly captured, structured,
and serialisable — fulfilling FR-4 (O4) evidence requirements.
"""

import json
import pytest
from src.safe_decoder import AmbiguitySafeOneHotDecoder
from src.policies import AmbiguityPolicy
from src.provenance import ProvenanceRecord, build_provenance, TransformEvent


class TestProvenanceBuild:
    """Tests for building provenance from a fitted decoder."""

    @pytest.fixture
    def fitted_decoder(self):
        decoder = AmbiguitySafeOneHotDecoder(drop="if_binary", handle_unknown="ignore")
        decoder.fit([["Female"], ["Male"]])
        return decoder

    def test_build_provenance_from_fitted_decoder(self, fitted_decoder):
        prov = build_provenance(fitted_decoder)
        assert isinstance(prov, ProvenanceRecord)

    def test_record_id_is_set(self, fitted_decoder):
        prov = build_provenance(fitted_decoder)
        assert prov.record_id.startswith("PROV-")

    def test_schema_captures_n_features(self, fitted_decoder):
        prov = build_provenance(fitted_decoder)
        assert prov.schema.n_features == 1

    def test_schema_captures_drop_config(self, fitted_decoder):
        prov = build_provenance(fitted_decoder)
        assert prov.schema.drop_config == "if_binary"

    def test_schema_captures_handle_unknown(self, fitted_decoder):
        prov = build_provenance(fitted_decoder)
        assert prov.schema.handle_unknown == "ignore"

    def test_schema_captures_sklearn_version(self, fitted_decoder):
        prov = build_provenance(fitted_decoder)
        assert "." in prov.schema.sklearn_version

    def test_schema_feature_schemas_present(self, fitted_decoder):
        prov = build_provenance(fitted_decoder)
        assert len(prov.schema.feature_schemas) == 1
        fs = prov.schema.feature_schemas[0]
        assert "categories" in fs
        assert "dropped_category" in fs
        assert "can_produce_all_zeros" in fs
        assert fs["can_produce_all_zeros"] is True  # binary drop + ignore = True

    def test_build_provenance_raises_if_not_fitted(self):
        decoder = AmbiguitySafeOneHotDecoder()
        with pytest.raises(RuntimeError, match="fitted"):
            build_provenance(decoder)


class TestProvenanceLogging:
    """Tests for logging transform and decode events."""

    @pytest.fixture
    def prov_with_events(self):
        decoder = AmbiguitySafeOneHotDecoder(drop="if_binary", handle_unknown="ignore")
        decoder.fit([["Female"], ["Male"]])
        prov = build_provenance(decoder)

        train_data = [["Female"], ["Male"], ["Unknown"]]
        X_enc = decoder.transform(train_data)
        prov.log_transform(input_shape=(3, 1), output_shape=(3, 1))

        results = decoder.safe_inverse_transform(X_enc, policy=AmbiguityPolicy.SENTINEL)
        n_safe = sum(1 for r in results if r.is_safe())
        n_ambiguous = sum(1 for r in results if r.is_ambiguous())
        n_unknown = sum(1 for r in results if r.is_unknown())
        prov.log_decode(
            input_shape=(3, 1),
            output_shape=(3, 1),
            n_safe=n_safe,
            n_ambiguous=n_ambiguous,
            n_unknown=n_unknown,
            policy_applied=AmbiguityPolicy.SENTINEL.value,
        )
        return prov

    def test_events_count(self, prov_with_events):
        assert len(prov_with_events.events) == 2

    def test_transform_event_recorded(self, prov_with_events):
        transform_events = [e for e in prov_with_events.events if e.operation == "transform"]
        assert len(transform_events) == 1

    def test_decode_event_recorded(self, prov_with_events):
        decode_events = [e for e in prov_with_events.events if e.operation == "safe_inverse_transform"]
        assert len(decode_events) == 1

    def test_summary_structure(self, prov_with_events):
        s = prov_with_events.summary()
        assert "record_id" in s
        assert "total_transform_calls" in s
        assert "total_decode_calls" in s
        assert s["total_transform_calls"] == 1
        assert s["total_decode_calls"] == 1


class TestProvenanceSerialization:
    """Tests for provenance JSON serialisation."""

    @pytest.fixture
    def prov(self):
        decoder = AmbiguitySafeOneHotDecoder(drop="first", handle_unknown="ignore")
        decoder.fit([["Red"], ["Green"], ["Blue"]])
        return build_provenance(decoder)

    def test_to_dict_has_required_keys(self, prov):
        d = prov.to_dict()
        assert "record_id" in d
        assert "schema" in d
        assert "events" in d

    def test_to_json_is_valid(self, prov):
        j = prov.to_json()
        parsed = json.loads(j)
        assert parsed["record_id"] == prov.record_id

    def test_schema_json_structure(self, prov):
        d = prov.to_dict()
        schema = d["schema"]
        assert schema["n_features"] == 1
        assert schema["drop_config"] == "first"
        assert isinstance(schema["feature_schemas"], list)
