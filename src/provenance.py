"""
provenance.py
=============
Transformation Provenance Record — a separately testable subsystem.

Satisfies FR-4 (O4): "Implement and verify Schema-aware encoder wrapper,
transformation provenance record, ambiguity detector, inverse-decoding
and its input, output, error, configuration and telemetry contracts."

A ProvenanceRecord captures:
  - The encoder schema (drop config, handle_unknown, categories per feature)
  - Each transform invocation (timestamp, input shape, output shape)
  - Each safe_inverse_transform invocation with per-sample status counts
  - Versioning of the scikit-learn toolchain

Records can be serialised to dict / JSON for audit and regulatory reporting.
"""

from __future__ import annotations

import json
import datetime
import platform
import sys
import sklearn
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SchemaRecord:
    """
    Captures the encoder configuration at fit-time.

    Attributes
    ----------
    n_features : int
    feature_schemas : list[dict]
        Per-feature: categories, dropped_category, handle_unknown.
    drop_config : str or None
    handle_unknown : str
    sklearn_version : str
    python_version : str
    fit_timestamp : str   ISO-8601 UTC
    """
    n_features: int
    feature_schemas: List[Dict[str, Any]]
    drop_config: Optional[str]
    handle_unknown: str
    sklearn_version: str
    python_version: str
    fit_timestamp: str


@dataclass
class TransformEvent:
    """
    Records one call to transform() or safe_inverse_transform().

    Attributes
    ----------
    operation : str          "transform" | "safe_inverse_transform"
    timestamp : str          ISO-8601 UTC
    input_shape : tuple
    output_shape : tuple
    n_ambiguous : int        Count of AMBIGUOUS samples (0 for transform)
    n_unknown : int          Count of UNKNOWN samples
    n_safe : int             Count of SAFE samples
    policy_applied : str     Active ambiguity policy name
    notes : str              Free-text annotation
    """
    operation: str
    timestamp: str
    input_shape: tuple
    output_shape: tuple
    n_ambiguous: int = 0
    n_unknown: int = 0
    n_safe: int = 0
    policy_applied: str = "N/A"
    notes: str = ""


@dataclass
class ProvenanceRecord:
    """
    Complete provenance record for one fitted AmbiguitySafeOneHotDecoder session.

    Build one instance per fitted decoder, then call log_transform() and
    log_decode() for each operation.

    Attributes
    ----------
    record_id : str          UUID-like identifier
    schema : SchemaRecord    Captured at fit-time
    events : list[TransformEvent]
    """
    record_id: str
    schema: SchemaRecord
    events: List[TransformEvent] = field(default_factory=list)

    def log_transform(
        self,
        input_shape: tuple,
        output_shape: tuple,
        notes: str = "",
    ) -> None:
        """Record a transform() invocation."""
        self.events.append(
            TransformEvent(
                operation="transform",
                timestamp=_utc_now(),
                input_shape=input_shape,
                output_shape=output_shape,
                notes=notes,
            )
        )

    def log_decode(
        self,
        input_shape: tuple,
        output_shape: tuple,
        n_safe: int,
        n_ambiguous: int,
        n_unknown: int,
        policy_applied: str,
        notes: str = "",
    ) -> None:
        """Record a safe_inverse_transform() invocation."""
        self.events.append(
            TransformEvent(
                operation="safe_inverse_transform",
                timestamp=_utc_now(),
                input_shape=input_shape,
                output_shape=output_shape,
                n_safe=n_safe,
                n_ambiguous=n_ambiguous,
                n_unknown=n_unknown,
                policy_applied=policy_applied,
                notes=notes,
            )
        )

    def summary(self) -> Dict[str, Any]:
        """Return a dict summarising all decode events."""
        decode_events = [e for e in self.events if e.operation == "safe_inverse_transform"]
        return {
            "record_id": self.record_id,
            "n_features": self.schema.n_features,
            "drop_config": self.schema.drop_config,
            "handle_unknown": self.schema.handle_unknown,
            "total_transform_calls": sum(
                1 for e in self.events if e.operation == "transform"
            ),
            "total_decode_calls": len(decode_events),
            "cumulative_safe": sum(e.n_safe for e in decode_events),
            "cumulative_ambiguous": sum(e.n_ambiguous for e in decode_events),
            "cumulative_unknown": sum(e.n_unknown for e in decode_events),
            "sklearn_version": self.schema.sklearn_version,
            "python_version": self.schema.python_version,
            "fit_timestamp": self.schema.fit_timestamp,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full serialisation to a nested dict (JSON-compatible)."""
        return {
            "record_id": self.record_id,
            "schema": asdict(self.schema),
            "events": [asdict(e) for e in self.events],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_encoder_metadata(cls, metadata: Any) -> "ProvenanceRecord":
        """
        Construct a ProvenanceRecord from an EncoderMetadata object
        (as produced by MetadataAwareEncoder after fitting).

        Parameters
        ----------
        metadata : EncoderMetadata
        """
        feature_schemas = []
        for fm in metadata.features:
            feature_schemas.append({
                "feature_index": fm.feature_index,
                "categories": [str(c) for c in fm.categories],
                "dropped_category": str(fm.dropped_category)
                    if fm.dropped_category is not None else None,
                "handle_unknown": fm.handle_unknown,
                "is_binary": fm.is_binary,
                "can_produce_all_zeros": fm.can_produce_all_zeros(),
                "n_encoded_columns": fm.n_encoded_columns,
            })

        schema = SchemaRecord(
            n_features=metadata.n_features,
            feature_schemas=feature_schemas,
            drop_config=metadata.drop_config,
            handle_unknown=metadata.handle_unknown,
            sklearn_version=sklearn.__version__,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            fit_timestamp=_utc_now(),
        )
        return cls(
            record_id=_make_record_id(),
            schema=schema,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _make_record_id() -> str:
    """Generate a short deterministic-ish record ID."""
    import hashlib
    ts = _utc_now()
    h = hashlib.md5(ts.encode()).hexdigest()[:12]
    return f"PROV-{h.upper()}"


# ---------------------------------------------------------------------------
# Convenience: build provenance from a fitted AmbiguitySafeOneHotDecoder
# ---------------------------------------------------------------------------

def build_provenance(decoder: Any) -> ProvenanceRecord:
    """
    Build a fresh ProvenanceRecord from a fitted AmbiguitySafeOneHotDecoder.

    Parameters
    ----------
    decoder : AmbiguitySafeOneHotDecoder
        Must have been fitted (decoder.is_fitted == True).

    Returns
    -------
    ProvenanceRecord
    """
    if not decoder.is_fitted:
        raise RuntimeError("Decoder must be fitted before building provenance.")
    return ProvenanceRecord.from_encoder_metadata(decoder.metadata)
