"""Chạy một thí nghiệm học sâu từ tên trong ``DL_REGISTRY``.

Kết quả trả về cùng định dạng dòng với nhánh ML (``feature``, ``model``, metric...),
nên có thể ghép chung một bảng xếp hạng ML + DL.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from absa.evaluation.metrics import evaluate_absa
from absa.features.static_embeddings import GensimWordEmbeddingVectorizer
from absa.models.dl.registry import get_dl_spec
from absa.preprocessing.pipeline import PreparedCorpus
from absa.utils.io import ensure_dir
from absa.utils.logging import get_logger
from absa.utils.timing import Timer

logger = get_logger(__name__)


@dataclass
class DLRunResult:
    """Kết quả một thí nghiệm DL."""

    row: dict
    history: pd.DataFrame
    predictions: np.ndarray
    trainer: object


def build_dl_trainer(name: str, train_texts, overrides: Mapping | None = None):
    """Dựng mô hình + bộ mã hóa + :class:`ABSATrainer` cho cấu hình ``name``.

    Với kiến trúc mức từ, từ vựng và embedding (Word2Vec/FastText) **chỉ** học trên
    ``train_texts``.
    """
    from absa.models.dl.architectures import BiLSTMAttentionABSA, TextCNNABSA, TransformerABSA
    from absa.models.dl.data import (
        HFTokenEncoder,
        VocabEncoder,
        Vocabulary,
        build_embedding_matrix,
    )
    from absa.models.dl.trainer import ABSATrainer, TrainConfig

    spec = get_dl_spec(name)
    config = replace(TrainConfig(), **{**spec.train, **(overrides or {})})
    if spec.arch == "transformer":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(spec.model_name)
        model = TransformerABSA(spec.model_name, **spec.arch_kwargs)
        encoder = HFTokenEncoder(tokenizer, spec.max_length)
        return ABSATrainer(model, config, encoder, pad_id=tokenizer.pad_token_id or 0)

    vocab = Vocabulary(min_freq=2).fit(train_texts)
    dim = 300
    keyed_vectors = None
    if spec.embedding in {"word2vec", "fasttext"}:
        embedder = GensimWordEmbeddingVectorizer(spec.embedding, vector_size=dim, pooling="mean")
        keyed_vectors = embedder.fit(train_texts).keyed_vectors_
    weights = build_embedding_matrix(vocab, keyed_vectors, dim)
    if spec.arch == "textcnn":
        model = TextCNNABSA(len(vocab), dim, weights, **spec.arch_kwargs)
    elif spec.arch == "bilstm_attention":
        model = BiLSTMAttentionABSA(len(vocab), dim, weights, **spec.arch_kwargs)
    else:
        raise ValueError(f"Kiến trúc không hỗ trợ: {spec.arch}")
    return ABSATrainer(model, config, VocabEncoder(vocab, spec.max_length), pad_id=0)


def run_dl_experiment(
    name: str,
    corpus: PreparedCorpus,
    labels: Mapping[str, np.ndarray],
    train_split: str = "train",
    eval_split: str = "dev",
    overrides: Mapping | None = None,
    output_dir: str | Path | None = None,
) -> DLRunResult:
    """Huấn luyện mô hình DL ``name`` trên train, early stopping và đánh giá trên dev.

    Args:
        name: Tên trong ``DL_REGISTRY``.
        corpus: Kho văn bản.
        labels: Ma trận nhãn theo tập.
        train_split: Tập huấn luyện.
        eval_split: Tập đánh giá (early stopping và metric).
        overrides: Ghi đè ``TrainConfig`` (vd ``{"epochs": 2}`` để chạy thử).
        output_dir: Nếu có, lưu lịch sử huấn luyện ra CSV.
    """
    spec = get_dl_spec(name)
    train_texts = corpus.get(spec.profile, train_split)
    eval_texts = corpus.get(spec.profile, eval_split)
    trainer = build_dl_trainer(name, train_texts, overrides)
    with Timer() as t:
        history = trainer.fit(train_texts, labels[train_split], eval_texts, labels[eval_split])
    Y_pred = trainer.predict(eval_texts)
    metrics = evaluate_absa(labels[eval_split], Y_pred)
    n_params = sum(p.numel() for p in trainer.model.parameters())
    row = {
        "run_id": f"dl|{name}",
        "tag": "dl",
        "feature": spec.model_name or f"{spec.embedding}_embedding",
        "feature_family": "finetuned" if spec.arch == "transformer" else "learned",
        "model": name,
        "model_family": f"dl_{spec.arch}",
        "strategy": "joint",
        "params": json.dumps(trainer.config_dict(), default=str),
        "status": "ok",
        "n_features": n_params,
        "fit_sec": round(t.seconds, 1),
        "epochs_run": len(history),
        **{k: round(v, 5) for k, v in metrics.items()},
    }
    if output_dir:
        history.to_csv(ensure_dir(output_dir) / f"history_{name}.csv", index=False)
    return DLRunResult(row, history, Y_pred, trainer)
