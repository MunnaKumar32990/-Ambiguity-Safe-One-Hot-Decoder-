"""
policies.py
===========
Ambiguity resolution and handling policies for the AmbiguitySafeOneHotDecoder.

As specified in Capstone Deliverable D3:
"Implement and compare explicit ambiguity policies:
  1. Sentinel output: Replace ambiguous/unknown reconstructions with informative sentinel tokens.
  2. Provenance side-channel: Return candidate decoding with structured metadata (confidence, collision set).
  3. Strict rejection: Raise an exception / abort on ambiguity to protect downstream safety-critical tasks.
  4. Withhold (default): Return None for ambiguous or unknown features."
"""

from enum import Enum
from typing import Any, List, Optional, Dict
from dataclasses import dataclass


class AmbiguityPolicy(str, Enum):
    """
    Explicit ambiguity handling policies.

    WITHHOLD
        Replaces ambiguous/unknown values with None. Safe default for data cleaning.
    SENTINEL
        Replaces ambiguous/unknown values with a structured sentinel string
        (e.g., '<AMBIGUOUS:Female|UNKNOWN>').
    PROVENANCE_SIDE_CHANNEL
        Returns sklearn's raw decode but attaches side-channel provenance metadata
        (collision candidates, confidence score, ambiguity flag) for downstream auditability.
    STRICT_REJECTION
        Raises an AmbiguityRejectionError immediately if any feature is ambiguous or unknown.
        Designed for safety-critical ML pipelines (e.g., healthcare, credit scoring).
    """
    WITHHOLD = "WITHHOLD"
    SENTINEL = "SENTINEL"
    PROVENANCE_SIDE_CHANNEL = "PROVENANCE_SIDE_CHANNEL"
    STRICT_REJECTION = "STRICT_REJECTION"


class AmbiguityRejectionError(ValueError):
    """
    Raised when STRICT_REJECTION policy is active and an ambiguous or unknown
    feature is encountered during inverse transformation.
    """
    def __init__(self, message: str, sample_index: int, feature_index: int, possible_values: List[Any]):
        super().__init__(message)
        self.sample_index = sample_index
        self.feature_index = feature_index
        self.possible_values = possible_values


def format_sentinel(
    status: str,
    dropped_category: Optional[Any] = None,
    possible_values: Optional[List[Any]] = None,
) -> str:
    """
    Generate an informative sentinel string for ambiguous or unknown decodings.

    Examples:
      - '<AMBIGUOUS:Female|UNSEEN>'
      - '<UNKNOWN:UNSEEN>'
    """
    if status == "AMBIGUOUS":
        if possible_values:
            candidates = "|".join(str(v) for v in possible_values)
            return f"<AMBIGUOUS:{candidates}>"
        elif dropped_category is not None:
            return f"<AMBIGUOUS:{dropped_category}|UNSEEN>"
        return "<AMBIGUOUS>"
    elif status == "UNKNOWN":
        return "<UNKNOWN:UNSEEN>"
    return "<UNKNOWN>"
