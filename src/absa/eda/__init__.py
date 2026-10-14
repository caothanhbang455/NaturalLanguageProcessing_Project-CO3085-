"""Thống kê phục vụ phân tích khám phá dữ liệu (EDA)."""

from absa.eda.analysis import (
    aspect_cooccurrence,
    aspect_polarity_counts,
    dataset_overview,
    emoji_frequency,
    is_vietnamese_syllable,
    label_distribution_shift,
    labels_per_comment,
    mixed_sentiment_rate,
    nonstandard_token_rate,
    nonstandard_token_report,
    preprocessing_impact,
    star_polarity_crosstab,
    text_length_frame,
    text_pattern_report,
    top_terms_by_aspect,
)

__all__ = [
    "aspect_cooccurrence",
    "aspect_polarity_counts",
    "dataset_overview",
    "emoji_frequency",
    "is_vietnamese_syllable",
    "label_distribution_shift",
    "labels_per_comment",
    "mixed_sentiment_rate",
    "nonstandard_token_rate",
    "nonstandard_token_report",
    "preprocessing_impact",
    "star_polarity_crosstab",
    "text_length_frame",
    "text_pattern_report",
    "top_terms_by_aspect",
]
