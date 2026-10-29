"""
Ambiguity-Safe Inverse Decoding for One-Hot Encoded Categorical Data
=====================================================================

Package: src

Modules:
    metadata           - EncoderMetadata dataclass
    encoder            - MetadataAwareEncoder (wraps sklearn OneHotEncoder)
    ambiguity_detector - AmbiguityDetector (core analysis engine)
    safe_decoder       - AmbiguitySafeOneHotDecoder (main public API)
    policies           - AmbiguityPolicy, AmbiguityRejectionError, format_sentinel
    evaluator          - ExperimentEvaluator (metrics and reporting)
    provenance         - ProvenanceRecord (transformation provenance side-channel)
    conformance        - ConformanceSuite (system invariant checker)
    diagnostic_report  - DiagnosticReportGenerator (full system diagnostic output)
    negative_tests     - NegativeTestCampaign (NT-1 to NT-6)
    kpi_benchmarking   - KPIEvaluator (KPI-1 to KPI-6 benchmarks)
"""

from .metadata import EncoderMetadata, FeatureMetadata
from .encoder import MetadataAwareEncoder
from .ambiguity_detector import AmbiguityDetector, AmbiguityStatus
from .safe_decoder import AmbiguitySafeOneHotDecoder, DecodingResult
from .evaluator import ExperimentEvaluator
from .policies import AmbiguityPolicy, AmbiguityRejectionError, format_sentinel
from .provenance import ProvenanceRecord, build_provenance
from .conformance import ConformanceSuite, ConformanceReport
from .diagnostic_report import DiagnosticReportGenerator

__version__ = "0.3.0"
__all__ = [
    "EncoderMetadata",
    "FeatureMetadata",
    "MetadataAwareEncoder",
    "AmbiguityDetector",
    "AmbiguityStatus",
    "AmbiguitySafeOneHotDecoder",
    "DecodingResult",
    "ExperimentEvaluator",
    "AmbiguityPolicy",
    "AmbiguityRejectionError",
    "format_sentinel",
    "ProvenanceRecord",
    "build_provenance",
    "ConformanceSuite",
    "ConformanceReport",
    "DiagnosticReportGenerator",
]
