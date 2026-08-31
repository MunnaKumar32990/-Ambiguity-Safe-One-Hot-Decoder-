"""
metadata.py
===========
Dataclasses that capture encoder configuration and per-feature metadata.

These structures are populated after fitting and are used by the
AmbiguityDetector to reason about ambiguity without requiring access
to original training data.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
import numpy as np


@dataclass
class FeatureMetadata:
    """
    Metadata for a single categorical feature column.

    Attributes
    ----------
    feature_index : int
        Column index in the original input array (0-based).
    categories : list
        All categories seen during training (as returned by sklearn).
    drop_config : str or None
        The value of the `drop` parameter for this encoder
        ("first", "if_binary", or None).
    dropped_category : Any or None
        The actual category that was dropped (if any).
        None if no category was dropped for this feature.
    dropped_category_index : int or None
        The position of the dropped category in `categories`.
    n_encoded_columns : int
        Number of columns produced in the encoded output for this feature.
    encoded_col_start : int
        Start column index in the full encoded array.
    encoded_col_end : int
        End column index (exclusive) in the full encoded array.
    handle_unknown : str
        Value of handle_unknown passed to OneHotEncoder
        ("ignore" or "error").
    is_binary : bool
        True if the feature has exactly 2 training categories.
    """

    feature_index: int
    categories: List[Any]
    drop_config: Optional[str]
    dropped_category: Optional[Any]
    dropped_category_index: Optional[int]
    n_encoded_columns: int
    encoded_col_start: int
    encoded_col_end: int
    handle_unknown: str
    is_binary: bool

    def can_produce_all_zeros(self) -> bool:
        """
        Return True if an all-zeros sub-vector for this feature is
        ambiguous — i.e., it can be produced by BOTH a dropped known
        category AND an unseen/unknown category.

        This is the core ambiguity condition detected by this project.
        """
        dropped_maps_to_zeros = self.dropped_category is not None
        unknown_maps_to_zeros = self.handle_unknown == "ignore"
        return dropped_maps_to_zeros and unknown_maps_to_zeros

    def get_encoded_slice(self, full_encoded_vector: np.ndarray) -> np.ndarray:
        """Extract this feature's sub-vector from a full encoded vector."""
        return full_encoded_vector[self.encoded_col_start: self.encoded_col_end]


@dataclass
class EncoderMetadata:
    """
    Complete metadata for a fitted encoder.

    Attributes
    ----------
    n_features : int
        Number of input feature columns.
    features : list[FeatureMetadata]
        Per-feature metadata, one entry per input column.
    drop_config : str or None
        Global drop parameter passed to OneHotEncoder.
    handle_unknown : str
        Global handle_unknown parameter.
    total_encoded_columns : int
        Total width of the encoded output array.
    """

    n_features: int
    features: List[FeatureMetadata] = field(default_factory=list)
    drop_config: Optional[str] = None
    handle_unknown: str = "error"
    total_encoded_columns: int = 0
