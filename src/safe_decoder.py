"""
safe_decoder.py
===============
AmbiguitySafeOneHotDecoder — the main public API of this research prototype.

This class provides an ambiguity-safe wrapper around OneHotEncoder's
inverse_transform. It orchestrates:

  1. Fitting via MetadataAwareEncoder
  2. Encoding (transform) with sparse or dense outputs
  3. Safe decoding (safe_inverse_transform) with selectable Ambiguity Policies
  4. Reporting per-sample decoding results with ambiguity status and provenance side-channels

Design Principle
----------------
The decoder does NOT modify what sklearn encodes. It adds a safety layer
on top of the decode step: instead of blindly trusting inverse_transform,
it uses AmbiguityDetector to classify each decoding as SAFE, AMBIGUOUS,
or UNKNOWN, and enforces explicit ambiguity resolution policies (Deliverable D3).
"""

import numpy as np
import pandas as pd
from scipy import sparse
from typing import List, Optional, Any, Dict, Union
from dataclasses import dataclass, field

from .encoder import MetadataAwareEncoder
from .ambiguity_detector import AmbiguityDetector, AmbiguityStatus, FeatureAmbiguityReport
from .metadata import EncoderMetadata
from .policies import AmbiguityPolicy, AmbiguityRejectionError, format_sentinel


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
        The safe decoded output governed by the active AmbiguityPolicy.
    possible_values : list[list]
        For each feature, the list of possible original categories.
    reasons : list[str]
        Per-feature reason strings.
    confidence : str
        "HIGH"   -> all features SAFE
        "MEDIUM" -> some features SAFE, some UNKNOWN
        "LOW"    -> at least one feature AMBIGUOUS
    policy_applied : str
        The AmbiguityPolicy used during decoding.
    provenance : dict, optional
        Structured side-channel audit metadata (Deliverable D3).
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
    policy_applied: str = AmbiguityPolicy.WITHHOLD.value
    provenance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
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
            "policy_applied": self.policy_applied,
            "provenance": self.provenance,
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
    ...                                      handle_unknown="ignore",
    ...                                      default_policy=AmbiguityPolicy.SENTINEL)
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
        Passed to OneHotEncoder ("ignore" or "error").
    sparse_output : bool
        Whether to use sparse matrices (default False).
    default_policy : AmbiguityPolicy
        Default policy for handling ambiguous features (Deliverable D3).
    """

    def __init__(
        self,
        drop: Optional[str] = "if_binary",
        handle_unknown: str = "ignore",
        sparse_output: bool = False,
        default_policy: Union[AmbiguityPolicy, str] = AmbiguityPolicy.WITHHOLD,
    ):
        self.drop = drop
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output
        self.default_policy = (
            AmbiguityPolicy(default_policy)
            if isinstance(default_policy, str)
            else default_policy
        )

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

    def fit(self, X: Any) -> "AmbiguitySafeOneHotDecoder":
        """Fit the encoder and build the ambiguity detector."""
        self._enc.fit(X)
        self._detector = AmbiguityDetector(self._enc.metadata)
        self.is_fitted = True
        return self

    def transform(self, X: Any) -> Union[np.ndarray, sparse.spmatrix]:
        """Encode input data. Returns np.ndarray or scipy.sparse matrix."""
        self._check_fitted()
        return self._enc.transform(X)

    def fit_transform(self, X: Any) -> Union[np.ndarray, sparse.spmatrix]:
        """Fit then transform."""
        return self.fit(X).transform(X)

    def safe_inverse_transform(
        self,
        X_encoded: Union[np.ndarray, sparse.spmatrix, List[List[float]]],
        policy: Optional[Union[AmbiguityPolicy, str]] = None,
    ) -> List[DecodingResult]:
        """
        Perform ambiguity-safe inverse transformation across drop modes and policies.

        Supports both dense arrays and scipy sparse matrices.

        Parameters
        ----------
        X_encoded : np.ndarray or scipy.sparse matrix or list
            Encoded representation.
        policy : AmbiguityPolicy or str, optional
            Resolution policy (WITHHOLD, SENTINEL, PROVENANCE_SIDE_CHANNEL, STRICT_REJECTION).

        Returns
        -------
        list[DecodingResult]
            One result per input row.
        """
        self._check_fitted()

        active_policy = (
            AmbiguityPolicy(policy)
            if isinstance(policy, str)
            else (policy or self.default_policy)
        )

        # Baseline sklearn decode (supports sparse and dense directly)
        sklearn_decoded_all = self._enc.inverse_transform(X_encoded)

        # Ensure dense array representation for per-row sub-vector inspection
        if sparse.issparse(X_encoded):
            X_dense = X_encoded.toarray()
        else:
            X_dense = np.asarray(X_encoded)

        results: List[DecodingResult] = []
        for i, row in enumerate(X_dense):
            sklearn_row = list(sklearn_decoded_all[i])
            feature_reports = self._detector.analyze_vector(row, sklearn_row)
            row_status = self._detector.aggregate_row_status(feature_reports)

            # Check strict rejection policy
            if active_policy == AmbiguityPolicy.STRICT_REJECTION:
                for rep in feature_reports:
                    if rep.status in (AmbiguityStatus.AMBIGUOUS, AmbiguityStatus.UNKNOWN):
                        raise AmbiguityRejectionError(
                            f"Sample {i} rejected by safety policy: Feature {rep.feature_index} "
                            f"is {rep.status.value} (possible candidates: {rep.possible_values}).",
                            sample_index=i,
                            feature_index=rep.feature_index,
                            possible_values=rep.possible_values,
                        )

            # Build safe_output and provenance based on policy
            safe_output = []
            possible_values = []
            reasons = []

            for rep in feature_reports:
                feat_meta = self.metadata.features[rep.feature_index]
                if rep.status == AmbiguityStatus.SAFE:
                    val = rep.possible_values[0] if rep.possible_values else rep.sklearn_decoded
                    safe_output.append(val)
                elif active_policy == AmbiguityPolicy.SENTINEL:
                    sentinel_val = format_sentinel(
                        rep.status.value,
                        dropped_category=feat_meta.dropped_category,
                        possible_values=rep.possible_values,
                    )
                    safe_output.append(sentinel_val)
                elif active_policy == AmbiguityPolicy.PROVENANCE_SIDE_CHANNEL:
                    # Returns candidate prediction while flagging side-channel
                    safe_output.append(rep.sklearn_decoded)
                else:  # WITHHOLD
                    safe_output.append(None)

                possible_values.append(rep.possible_values)
                reasons.append(rep.reason)

            confidence = self._compute_confidence(feature_reports)

            # Provenance side-channel metadata
            provenance = {
                "sample_index": i,
                "has_ambiguity": any(rep.status == AmbiguityStatus.AMBIGUOUS for rep in feature_reports),
                "has_unknown": any(rep.status == AmbiguityStatus.UNKNOWN for rep in feature_reports),
                "confidence": confidence,
                "policy": active_policy.value,
                "feature_provenance": [
                    {
                        "feature_index": rep.feature_index,
                        "status": rep.status.value,
                        "candidates": [str(c) for c in rep.possible_values],
                        "sklearn_prediction": str(rep.sklearn_decoded) if rep.sklearn_decoded is not None else None,
                        "is_ambiguous": rep.status == AmbiguityStatus.AMBIGUOUS,
                        "reason": rep.reason,
                    }
                    for rep in feature_reports
                ],
            }

            result = DecodingResult(
                sample_index=i,
                encoded=[float(v) for v in row],
                sklearn_decoded=sklearn_row,
                feature_reports=feature_reports,
                row_status=row_status,
                safe_output=safe_output,
                possible_values=possible_values,
                reasons=reasons,
                confidence=confidence,
                policy_applied=active_policy.value,
                provenance=provenance,
            )
            results.append(result)

        return results

    def baseline_inverse_transform(
        self, X_encoded: Union[np.ndarray, sparse.spmatrix, List[List[float]]]
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
                    "policy_applied": res.policy_applied,
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
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
