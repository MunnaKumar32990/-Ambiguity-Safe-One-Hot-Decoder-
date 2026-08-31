"""
safe_decoder.py
===============
AmbiguitySafeOneHotDecoder — the main public API of this research prototype.

This class provides an ambiguity-safe wrapper around OneHotEncoder's
inverse_transform. It orchestrates:

  1. Fitting via MetadataAwareEncoder
  2. Encoding (transform)
  3. Safe decoding (safe_inverse_transform)
  4. Reporting per-sample decoding results with ambiguity status

Design Principle
----------------
The decoder does NOT modify what sklearn encodes. It adds a safety layer
on top of the decode step: instead of blindly trusting inverse_transform,
it uses AmbiguityDetector to classify each decoding as SAFE, AMBIGUOUS,
or UNKNOWN, and withholds the reconstructed value when ambiguous.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field

from .encoder import MetadataAwareEncoder
from .ambiguity_detector import AmbiguityDetector, AmbiguityStatus, FeatureAmbiguityReport
from .metadata import EncoderMetadata


@dataclass
class DecodingResult:
    """
    Complete decoding result for a single input sample.

    Attributes
    ----------
    sample_index : int
        Position of this sample in the input array.
    encoded : list
        The encoded vector (as a flat list).
    sklearn_decoded : list
        Raw sklearn inverse_transform result (one value per feature).
    feature_reports : list[FeatureAmbiguityReport]
        Per-feature ambiguity analysis.
    row_status : AmbiguityStatus
        Aggregated status for the entire row.
    safe_output : list or None
        The safe decoded output. Each element is either the reconstructed
        category (if SAFE for that feature) or None (if AMBIGUOUS/UNKNOWN).
    possible_values : list[list]
        For each feature, the list of possible original categories.
    reasons : list[str]
        Per-feature reason strings.
    confidence : str
        "HIGH"   -> all features SAFE
        "MEDIUM" -> some features SAFE, some UNKNOWN
        "LOW"    -> at least one feature AMBIGUOUS
    """
    sample_index: int
    encoded: List[float]
    sklearn_decoded: List[Any]
    feature_reports: List[FeatureAmbiguityReport]
    row_status: AmbiguityStatus
    safe_output: List[Optional[Any]]
    possible_values: List[List[Any]]
    reasons: List[str]
    confidence: str

    def to_dict(self) -> Dict:
        """Serialize to a flat dict suitable for DataFrame/CSV export."""
        return {
            "sample_index": self.sample_index,
            "encoded": self.encoded,
            "sklearn_decoded": self.sklearn_decoded,
            "row_status": self.row_status.value,
            "safe_output": self.safe_output,
            "possible_values": self.possible_values,
            "reasons": self.reasons,
            "confidence": self.confidence,
        }

    def is_safe(self) -> bool:
        return self.row_status == AmbiguityStatus.SAFE

    def is_ambiguous(self) -> bool:
        return self.row_status == AmbiguityStatus.AMBIGUOUS

    def is_unknown(self) -> bool:
        return self.row_status == AmbiguityStatus.UNKNOWN


class AmbiguitySafeOneHotDecoder:
    """
    Ambiguity-safe wrapper around OneHotEncoder's inverse_transform.

    Usage
    -----
    >>> decoder = AmbiguitySafeOneHotDecoder(drop="if_binary",
    ...                                      handle_unknown="ignore")
    >>> decoder.fit(X_train)
    >>> X_encoded = decoder.transform(X_test)
    >>> results = decoder.safe_inverse_transform(X_encoded)
    >>> for r in results:
    ...     print(r.row_status, r.safe_output)

    Parameters
    ----------
    drop : str or None
        Passed to OneHotEncoder.
    handle_unknown : str
        Passed to OneHotEncoder.
    sparse_output : bool
        Whether to use sparse matrices (default False).
    """

    def __init__(
        self,
        drop: Optional[str] = "if_binary",
        handle_unknown: str = "ignore",
        sparse_output: bool = False,
    ):
        self.drop = drop
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output

        self._enc = MetadataAwareEncoder(
            drop=drop,
            handle_unknown=handle_unknown,
            sparse_output=sparse_output,
        )
        self._detector: Optional[AmbiguityDetector] = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(self, X: List[List[Any]]) -> "AmbiguitySafeOneHotDecoder":
        """Fit the encoder and build the ambiguity detector."""
        self._enc.fit(X)
        self._detector = AmbiguityDetector(self._enc.metadata)
        self.is_fitted = True
        return self

    def transform(self, X: List[List[Any]]) -> np.ndarray:
        """Encode input data. Returns np.ndarray."""
        self._check_fitted()
        return self._enc.transform(X)

    def fit_transform(self, X: List[List[Any]]) -> np.ndarray:
        """Fit then transform."""
        return self.fit(X).transform(X)

    def safe_inverse_transform(
        self, X_encoded: np.ndarray
    ) -> List[DecodingResult]:
        """
        Perform ambiguity-safe inverse transformation.

        For each encoded row:
          1. Run sklearn's inverse_transform (baseline result).
          2. Run AmbiguityDetector on the row.
          3. Build a DecodingResult with full ambiguity metadata.
          4. Set safe_output to None for ambiguous/unknown features.

        Parameters
        ----------
        X_encoded : np.ndarray of shape (n_samples, n_encoded_cols)

        Returns
        -------
        list[DecodingResult]
            One result per input row.
        """
        self._check_fitted()
        X_encoded = np.array(X_encoded)

        # Baseline sklearn decode (used for comparison, not as ground truth)
        sklearn_decoded_all = self._enc.inverse_transform(X_encoded)

        results = []
        for i, row in enumerate(X_encoded):
            sklearn_row = list(sklearn_decoded_all[i])
            feature_reports = self._detector.analyze_vector(row, sklearn_row)
            row_status = self._detector.aggregate_row_status(feature_reports)

            # Build safe_output: None where ambiguous/unknown
            safe_output = []
            possible_values = []
            reasons = []
            for rep in feature_reports:
                if rep.status == AmbiguityStatus.SAFE:
                    safe_output.append(rep.possible_values[0]
                                       if rep.possible_values else rep.sklearn_decoded)
                else:
                    safe_output.append(None)
                possible_values.append(rep.possible_values)
                reasons.append(rep.reason)

            confidence = self._compute_confidence(feature_reports)

            result = DecodingResult(
                sample_index=i,
                encoded=list(row),
                sklearn_decoded=sklearn_row,
                feature_reports=feature_reports,
                row_status=row_status,
                safe_output=safe_output,
                possible_values=possible_values,
                reasons=reasons,
                confidence=confidence,
            )
            results.append(result)

        return results

    def baseline_inverse_transform(
        self, X_encoded: np.ndarray
    ) -> np.ndarray:
        """
        Raw sklearn inverse_transform without safety checking.
        Used for baseline comparison.
        """
        self._check_fitted()
        return self._enc.inverse_transform(X_encoded)

    @property
    def metadata(self) -> Optional[EncoderMetadata]:
        return self._enc.metadata if self.is_fitted else None

    @property
    def encoder(self):
        """Access the underlying sklearn encoder directly."""
        return self._enc.encoder

    def results_to_dataframe(
        self, results: List[DecodingResult]
    ) -> pd.DataFrame:
        """
        Convert a list of DecodingResult objects to a pandas DataFrame.
        Useful for experiment reporting.
        """
        rows = []
        for res in results:
            n_features = len(res.feature_reports)
            for feat_idx, rep in enumerate(res.feature_reports):
                rows.append({
                    "sample_index": res.sample_index,
                    "feature_index": feat_idx,
                    "encoded_sub_vector": rep.encoded_sub_vector,
                    "sklearn_decoded": rep.sklearn_decoded,
                    "feature_status": rep.status.value,
                    "safe_output": res.safe_output[feat_idx],
                    "possible_values": rep.possible_values,
                    "reason": rep.reason,
                    "row_status": res.row_status.value,
                    "confidence": res.confidence,
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "Decoder is not fitted. Call fit() before transform/decode."
            )

    @staticmethod
    def _compute_confidence(
        feature_reports: List[FeatureAmbiguityReport],
    ) -> str:
        """Compute overall confidence from per-feature reports."""
        statuses = {r.status for r in feature_reports}
        if AmbiguityStatus.AMBIGUOUS in statuses:
            return "LOW"
        if AmbiguityStatus.UNKNOWN in statuses:
            return "MEDIUM"
        return "HIGH"
