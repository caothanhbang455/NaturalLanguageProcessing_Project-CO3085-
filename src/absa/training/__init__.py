"""Huấn luyện: lưới thí nghiệm M x N, tinh chỉnh, cross-validation, ablation, mô hình cuối."""

from absa.training.ablation import (
    cumulative_variants,
    leave_one_out_variants,
    run_preprocessing_ablation,
)
from absa.training.builders import build_ml_pipeline
from absa.training.cross_validation import cross_val_predict_absa, stratification_key
from absa.training.experiment import ExperimentRunner
from absa.training.final import fit_final_pipeline
from absa.training.tuning import candidate_params, tune_model, tune_top_k

__all__ = [
    "ExperimentRunner",
    "build_ml_pipeline",
    "candidate_params",
    "cross_val_predict_absa",
    "cumulative_variants",
    "fit_final_pipeline",
    "leave_one_out_variants",
    "run_preprocessing_ablation",
    "stratification_key",
    "tune_model",
    "tune_top_k",
]
