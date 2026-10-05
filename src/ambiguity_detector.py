"""
ambiguity_detector.py
=====================
AmbiguityDetector — the core research component.

This module implements metadata-driven ambiguity analysis for
one-hot encoded vectors. It does NOT inspect the original input
values; it reasons purely from:

  1. Encoder configuration (drop, handle_unknown)
  2. Per-feature metadata (categories, dropped index, column ranges)
  3. The encoded vector itself

Ambiguity Conditions
--------------------
A decoded value is AMBIGUOUS when a single encoded sub-vector
can be validly produced by more than one original category:

  Case A — Binary drop + ignore unknown:
    Encoded all-zeros sub-vector -> could be:
      (i)  The dropped known category (e.g., "Female" when drop="if_binary")
      (ii) An unseen category (which is also mapped to zeros by handle_unknown="ignore")

  Case B — Multi-class + ignore unknown:
    If ALL feature bits for a feature are zero but no category was
    dropped, the zero-vector is produced only by unknown -> UNKNOWN (not
    ambiguous between two known categories, but still unrecoverable).

  Case C — Safe:
    Exactly one known category maps to this encoded sub-vector -> SAFE.
"""

import numpy as np
from enum import Enum
from typing import List, Optional, Any
from dataclasses import dataclass, field

from .metadata import EncoderMetadata, FeatureMetadata


class AmbiguityStatus(str, Enum):
    """
    Status codes for a decoded value.

    SAFE      — Encoded vector uniquely identifies a known training category.
    AMBIGUOUS — Encoded vector can correspond to multiple possible sources.
                Cannot determine the correct original value.
    UNKNOWN   — Encoded vector indicates a definitely unseen category with
                no known category confusion (e.g., all-zeros but no drop
                was configured -> no known category shares this vector).
    """
    SAFE = "SAFE"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"


@dataclass
class FeatureAmbiguityReport:
    """
    Ambiguity analysis result for a single feature column.

    Attributes
    ----------
    feature_index : int
        Which input feature column this report is for.
    encoded_sub_vector : list
        The encoded sub-vector for this feature.
    status : AmbiguityStatus
    possible_values : list
        Candidate original categories (may have >1 element when AMBIGUOUS).
    sklearn_decoded : Any
        What sklearn's inverse_transform returned for this feature.
    reason : str
        Human-readable explanation of the status.
    """
    feature_index: int
    encoded_sub_vector: List[float]
    status: AmbiguityStatus
    possible_values: List[Any] = field(default_factory=list)
    sklearn_decoded: Any = None
    reason: str = ""


class AmbiguityDetector:
    """
    Analyzes one-hot encoded vectors for ambiguity using encoder metadata.

    This is a metadata-driven approach: it does not need access to the
    original training data — only to the EncoderMetadata produced after
    fitting.

    Parameters
    ----------
    metadata : EncoderMetadata
        The metadata extracted by MetadataAwareEncoder.
    """

    def __init__(self, metadata: EncoderMetadata):
        self.metadata = metadata

    def analyze_vector(
        self,
        full_encoded_vector: np.ndarray,
        sklearn_decoded_row: List[Any],
    ) -> List[FeatureAmbiguityReport]:
        """
        Analyze a single encoded row and return a per-feature ambiguity report.

        Parameters
        ----------
        full_encoded_vector : np.ndarray
            A 1-D array: one complete encoded sample row.
        sklearn_decoded_row : list
            The raw sklearn inverse_transform result for this row
            (one element per feature).

        Returns
        -------
        list[FeatureAmbiguityReport]
            One report per feature.
        """
        reports = []
        for feat_meta in self.metadata.features:
            sub_vec = feat_meta.get_encoded_slice(full_encoded_vector)
            sklearn_val = sklearn_decoded_row[feat_meta.feature_index]
            report = self._analyze_feature(feat_meta, sub_vec, sklearn_val)
            reports.append(report)
        return reports

    def _analyze_feature(
        self,
        feat: FeatureMetadata,
        sub_vec: np.ndarray,
        sklearn_val: Any,
    ) -> FeatureAmbiguityReport:
        """
        Core algorithm: classify a feature's encoded sub-vector.

        Decision logic:
          1. Is the sub-vector all-zeros?
             a. If a category was dropped AND handle_unknown="ignore":
                -> AMBIGUOUS (both dropped category and unseen map here)
             b. If a category was dropped but handle_unknown != "ignore":
                -> SAFE (only the dropped category maps to zeros; unseen
                  would have raised an error at encode time)
             c. If no category was dropped:
                -> UNKNOWN (only unseen/unknown categories map to all-zeros)
          2. If not all-zeros:
             -> identify which known category produced this vector -> SAFE
        """
        is_all_zeros = np.all(sub_vec == 0)

        if is_all_zeros:
            return self._handle_all_zeros(feat, sub_vec, sklearn_val)
        else:
            return self._handle_nonzero(feat, sub_vec, sklearn_val)

    def _handle_all_zeros(
        self,
        feat: FeatureMetadata,
        sub_vec: np.ndarray,
        sklearn_val: Any,
    ) -> FeatureAmbiguityReport:
        """Handle the all-zeros case — the central ambiguity scenario."""

        has_dropped = feat.dropped_category is not None
        ignores_unknown = feat.handle_unknown == "ignore"

        if has_dropped and ignores_unknown:
            # ==============================================================
            # CORE AMBIGUITY: All-zeros can mean EITHER:
            #   (1) The dropped known category (e.g., "Female")
            #   (2) Any unseen/unknown category (e.g., "Unknown")
            # sklearn silently returns (1) — this is the research problem.
            # ==============================================================
            possible = [feat.dropped_category, "<UNSEEN>"]
            reason = (
                f"All-zeros sub-vector is produced by BOTH the dropped "
                f"category '{feat.dropped_category}' (due to drop='{feat.drop_config}') "
                f"AND any unseen category (due to handle_unknown='ignore'). "
                f"sklearn silently returns '{feat.dropped_category}' "
                f"but the true source is unrecoverable."
            )
            return FeatureAmbiguityReport(
                feature_index=feat.feature_index,
                encoded_sub_vector=list(sub_vec),
                status=AmbiguityStatus.AMBIGUOUS,
                possible_values=possible,
                sklearn_decoded=sklearn_val,
                reason=reason,
            )

        elif has_dropped and not ignores_unknown:
            # Drop is active but unknown would cause an error at encode time.
            # So all-zeros uniquely identifies the dropped category.
            reason = (
                f"All-zeros uniquely identifies the dropped category "
                f"'{feat.dropped_category}'. handle_unknown='{feat.handle_unknown}' "
                f"means unseen values would have raised an error, so "
                f"all-zeros cannot come from an unseen category."
            )
            return FeatureAmbiguityReport(
                feature_index=feat.feature_index,
                encoded_sub_vector=list(sub_vec),
                status=AmbiguityStatus.SAFE,
                possible_values=[feat.dropped_category],
                sklearn_decoded=sklearn_val,
                reason=reason,
            )

        else:
            # No category was dropped. All-zeros means an unseen category.
            reason = (
                f"All-zeros sub-vector with no dropped category. "
                f"This indicates an unseen/unknown category was encoded "
                f"with handle_unknown='{feat.handle_unknown}'. "
                f"No known training category produces all-zeros without "
                f"an active drop configuration."
            )
            return FeatureAmbiguityReport(
                feature_index=feat.feature_index,
                encoded_sub_vector=list(sub_vec),
                status=AmbiguityStatus.UNKNOWN,
                possible_values=["<UNSEEN>"],
                sklearn_decoded=sklearn_val,
                reason=reason,
            )

    def _handle_nonzero(
        self,
        feat: FeatureMetadata,
        sub_vec: np.ndarray,
        sklearn_val: Any,
    ) -> FeatureAmbiguityReport:
        """
        Handle a non-zero sub-vector.

        A non-zero sub-vector should identify exactly one non-dropped
        category. We verify this by reconstructing what each category
        would produce and matching.
        """
        # Find which non-dropped category this sub-vector corresponds to
        matched_cat = self._match_category(feat, sub_vec)

        if matched_cat is not None:
            reason = (
                f"Sub-vector uniquely matches known training category "
                f"'{matched_cat}'. Reconstruction is unambiguous."
            )
            return FeatureAmbiguityReport(
                feature_index=feat.feature_index,
                encoded_sub_vector=list(sub_vec),
                status=AmbiguityStatus.SAFE,
                possible_values=[matched_cat],
                sklearn_decoded=sklearn_val,
                reason=reason,
            )
        else:
            # Non-zero but doesn't match any known category — malformed vector
            reason = (
                f"Sub-vector {list(sub_vec)} does not match any known "
                f"training category. The encoded vector may be corrupted "
                f"or come from a different encoder."
            )
            return FeatureAmbiguityReport(
                feature_index=feat.feature_index,
                encoded_sub_vector=list(sub_vec),
                status=AmbiguityStatus.UNKNOWN,
                possible_values=[],
                sklearn_decoded=sklearn_val,
                reason=reason,
            )

    def _match_category(
        self,
        feat: FeatureMetadata,
        sub_vec: np.ndarray,
    ) -> Optional[Any]:
        """
        Reconstruct which category produced `sub_vec` by simulating
        what each non-dropped category's one-hot encoding looks like.

        For a feature with n categories and 1 dropped:
          - The remaining (n-1) categories each produce a 1-hot vector
            of length (n-1), with a 1 at their respective position.
          - We match the observed sub-vector against each.
        """
        cats = feat.categories
        dropped_idx = feat.dropped_category_index

        # Direct O(1) matching if vector is standard one-hot (single 1.0)
        if np.sum(sub_vec == 1.0) == 1 and np.sum(sub_vec) == 1.0:
            enc_pos = int(np.argmax(sub_vec))
            curr_pos = 0
            for i, cat in enumerate(cats):
                if dropped_idx is not None and i == dropped_idx:
                    continue
                if curr_pos == enc_pos:
                    return cat
                curr_pos += 1

        return None

    def aggregate_row_status(
        self,
        feature_reports: List[FeatureAmbiguityReport],
    ) -> AmbiguityStatus:
        """
        Aggregate per-feature reports into a single row-level status.

        Rule:
          - If ANY feature is AMBIGUOUS -> row is AMBIGUOUS
          - Elif ANY feature is UNKNOWN  -> row is UNKNOWN
          - Else                         -> row is SAFE
        """
        statuses = {r.status for r in feature_reports}
        if AmbiguityStatus.AMBIGUOUS in statuses:
            return AmbiguityStatus.AMBIGUOUS
        if AmbiguityStatus.UNKNOWN in statuses:
            return AmbiguityStatus.UNKNOWN
        return AmbiguityStatus.SAFE
