"""Metric ABSA, phân tích lỗi và tổng hợp kết quả thí nghiệm."""

from absa.evaluation.comparison import summarize_by
from absa.evaluation.error_analysis import (
    build_error_table,
    error_type_summary,
    polarity_confusions,
    sample_errors,
)
from absa.evaluation.metrics import (
    all_confusion_matrices,
    aspect_confusion_matrix,
    evaluate_absa,
    evaluate_pair_bits,
    pair_report,
    pair_report_from_bits,
    per_aspect_report,
)

__all__ = [
    "all_confusion_matrices",
    "aspect_confusion_matrix",
    "build_error_table",
    "error_type_summary",
    "evaluate_absa",
    "evaluate_pair_bits",
    "pair_report",
    "pair_report_from_bits",
    "per_aspect_report",
    "polarity_confusions",
    "sample_errors",
    "summarize_by",
]
