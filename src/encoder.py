"""
encoder.py
==========
MetadataAwareEncoder — wraps sklearn's OneHotEncoder and extracts
rich per-feature metadata after fitting.

This is the entry point of the pipeline. After fit(), the .metadata
attribute contains an EncoderMetadata instance that drives the
ambiguity analysis engine without needing the original training data.
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Optional, List, Any

from .metadata import EncoderMetadata, FeatureMetadata


class MetadataAwareEncoder:
    """
    A thin wrapper around sklearn's OneHotEncoder that:
      1. Fits the encoder normally.
      2. Extracts and stores structured metadata about each feature.
      3. Exposes encode() and decode() proxies for downstream use.

    Parameters
    ----------
    drop : str or None
        Passed directly to OneHotEncoder. Supports "first", "if_binary", None.
    handle_unknown : str
        Passed directly to OneHotEncoder. "ignore" or "error".
    sparse_output : bool
        Whether to return a sparse matrix (default False for research clarity).

    Attributes
    ----------
    encoder : OneHotEncoder
        The underlying sklearn encoder.
    metadata : EncoderMetadata
        Populated after fit(). Contains per-feature metadata.
    is_fitted : bool
        True after fit() has been called.
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
        self.encoder: Optional[OneHotEncoder] = None
        self.metadata: Optional[EncoderMetadata] = None
        self.is_fitted: bool = False

    def fit(self, X: List[List[Any]]) -> "MetadataAwareEncoder":
        """
        Fit the encoder on training data and extract metadata.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training categorical data.

        Returns
        -------
        self
        """
        self.encoder = OneHotEncoder(
            drop=self.drop,
            handle_unknown=self.handle_unknown,
            sparse_output=self.sparse_output,
        )
        self.encoder.fit(X)
        self.metadata = self._extract_metadata()
        self.is_fitted = True
        return self

    def _extract_metadata(self) -> EncoderMetadata:
        """
        Build EncoderMetadata from the fitted sklearn encoder.

        Key attributes consulted:
          encoder.categories_     — list of arrays, one per feature
          encoder.drop_idx_        — indices of dropped categories (or None)
        """
        enc = self.encoder
        n_features = len(enc.categories_)
        feature_metas = []
        col_cursor = 0

        for feat_idx in range(n_features):
            cats = list(enc.categories_[feat_idx])
            is_binary = len(cats) == 2

            # Determine the dropped category for this feature
            dropped_cat = None
            dropped_cat_idx = None

            if enc.drop_idx_ is not None:
                drop_idx_val = enc.drop_idx_[feat_idx]
                if drop_idx_val is not None:
                    # np.intp / integer -> Python int
                    dropped_cat_idx = int(drop_idx_val)
                    dropped_cat = cats[dropped_cat_idx]

            # Number of columns in encoded output for this feature
            n_encoded_cols = len(cats)
            if dropped_cat is not None:
                n_encoded_cols -= 1  # One column removed for dropped category

            fm = FeatureMetadata(
                feature_index=feat_idx,
                categories=cats,
                drop_config=self.drop,
                dropped_category=dropped_cat,
                dropped_category_index=dropped_cat_idx,
                n_encoded_columns=n_encoded_cols,
                encoded_col_start=col_cursor,
                encoded_col_end=col_cursor + n_encoded_cols,
                handle_unknown=self.handle_unknown,
                is_binary=is_binary,
            )
            feature_metas.append(fm)
            col_cursor += n_encoded_cols

        return EncoderMetadata(
            n_features=n_features,
            features=feature_metas,
            drop_config=self.drop,
            handle_unknown=self.handle_unknown,
            total_encoded_columns=col_cursor,
        )

    def transform(self, X: List[List[Any]]) -> np.ndarray:
        """
        Encode input data using the fitted encoder.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_encoded_columns)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform().")
        return self.encoder.transform(X)

    def fit_transform(self, X: List[List[Any]]) -> np.ndarray:
        """Convenience: fit then transform."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_encoded: np.ndarray):
        """
        Proxy for sklearn's inverse_transform (baseline behavior).
        Returns the raw sklearn decoded result WITHOUT safety checking.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before inverse_transform().")
        return self.encoder.inverse_transform(X_encoded)

    def get_feature_names(self) -> List[str]:
        """Return feature names from the encoder."""
        return list(self.encoder.get_feature_names_out())
