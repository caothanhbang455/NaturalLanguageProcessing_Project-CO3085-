"""Các cách biểu diễn văn bản: sparse (túi từ) và static embedding."""

from absa.features.registry import (
    FEATURE_REGISTRY,
    FeatureSpec,
    build_feature,
    feature_catalog,
    get_feature_spec,
    list_features,
)
from absa.features.static_embeddings import (
    Doc2VecVectorizer,
    GensimWordEmbeddingVectorizer,
    PretrainedFastTextVectorizer,
)

__all__ = [
    "FEATURE_REGISTRY",
    "Doc2VecVectorizer",
    "FeatureSpec",
    "GensimWordEmbeddingVectorizer",
    "PretrainedFastTextVectorizer",
    "build_feature",
    "feature_catalog",
    "get_feature_spec",
    "list_features",
]
