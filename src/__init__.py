"""
Ambiguity-Safe Inverse Decoding for One-Hot Encoded Categorical Data
=====================================================================

Package: src

Modules:
    metadata          - EncoderMetadata dataclass
    encoder           - MetadataAwareEncoder (wraps sklearn OneHotEncoder)
    ambiguity_detector - AmbiguityDetector (core analysis engine)
    safe_decoder      - AmbiguitySafeOneHotDecoder (main public API)
    evaluator         - ExperimentEvaluator (metrics and reporting)
"""

from .metadata import EncoderMetadata, FeatureMetadata
from .encoder import MetadataAwareEncoder
from .ambiguity_detector import AmbiguityDetector, AmbiguityStatus
from .safe_decoder import AmbiguitySafeOneHotDecoder, DecodingResult
from .evaluator import ExperimentEvaluator

__version__ = "0.1.0"
__all__ = [
    "EncoderMetadata",
    "FeatureMetadata",
    "MetadataAwareEncoder",
    "AmbiguityDetector",
    "AmbiguityStatus",
    "AmbiguitySafeOneHotDecoder",
    "DecodingResult",
    "ExperimentEvaluator",
]
