"""Danh mục mô hình học sâu (DL registry).

Giống nhánh ML, nhánh DL cũng là lưới *kiến trúc x biểu diễn*:

* Kiến trúc mức từ (TextCNN, BiLSTM-Attention) x khởi tạo embedding
  (ngẫu nhiên, Word2Vec, FastText học trên train).
* Encoder Transformer fine-tune: PhoBERT-v2, ViSoBERT, XLM-R, mDeBERTa-v3, PhoBERT-large.

Fine-tune Transformer trên CPU rất chậm (hàng giờ); nên chạy trên GPU (Colab/Kaggle).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class DLModelSpec:
    """Mô tả một cấu hình DL.

    Attributes:
        name: Khóa duy nhất.
        arch: ``"textcnn" | "bilstm_attention" | "transformer"``.
        profile: Profile tiền xử lý đầu vào.
        embedding: ``"random" | "word2vec" | "fasttext"`` (kiến trúc mức từ).
        model_name: Tên mô hình Hugging Face (kiến trúc transformer).
        train: Ghi đè :class:`~absa.models.dl.trainer.TrainConfig`.
        arch_kwargs: Tham số thêm cho kiến trúc.
        max_length: Độ dài chuỗi tối đa (token).
        heavy: Cần GPU để chạy trong thời gian hợp lý.
        summary: Mô tả ngắn.
        reference: Công bố gốc.
    """

    name: str
    arch: str
    profile: str
    embedding: str = "random"
    model_name: str = ""
    train: dict = field(default_factory=dict)
    arch_kwargs: dict = field(default_factory=dict)
    max_length: int = 128
    heavy: bool = False
    summary: str = ""
    reference: str = ""


_WORD_TRAIN = {"epochs": 20, "batch_size": 64, "lr": 1e-3, "patience": 4}
_TRANSFORMER_TRAIN = {
    "epochs": 8,
    "batch_size": 16,
    "lr": 1e-3,
    "encoder_lr": 2e-5,
    "patience": 2,
    "warmup_ratio": 0.1,
}

_SPECS: list[DLModelSpec] = [
    DLModelSpec(
        "textcnn_random",
        "textcnn",
        "classic",
        "random",
        train=_WORD_TRAIN,
        summary="TextCNN, embedding khởi tạo ngẫu nhiên – mốc để thấy lợi ích embedding.",
        reference="Kim (2014), EMNLP",
    ),
    DLModelSpec(
        "textcnn_word2vec",
        "textcnn",
        "classic",
        "word2vec",
        train=_WORD_TRAIN,
        summary="TextCNN khởi tạo bằng Word2Vec học trên train, tiếp tục fine-tune.",
        reference="Kim (2014); Mikolov et al. (2013)",
    ),
    DLModelSpec(
        "textcnn_fasttext",
        "textcnn",
        "classic",
        "fasttext",
        train=_WORD_TRAIN,
        summary="TextCNN khởi tạo bằng FastText (có subword) học trên train.",
        reference="Kim (2014); Bojanowski et al. (2017)",
    ),
    DLModelSpec(
        "bilstm_attention_fasttext",
        "bilstm_attention",
        "classic",
        "fasttext",
        train=_WORD_TRAIN,
        summary="BiLSTM 2 lớp + attention riêng cho từng khía cạnh, embedding FastText.",
        reference="Wang et al. (2016), EMNLP (ATAE-LSTM)",
    ),
    DLModelSpec(
        "phobert_base_v2",
        "transformer",
        "transformer_seg",
        model_name="vinai/phobert-base-v2",
        train=_TRANSFORMER_TRAIN,
        max_length=160,
        heavy=True,
        arch_kwargs={"pooling": "aspect_attention"},
        summary="Fine-tune PhoBERT-base-v2 + aspect attention – SOTA phổ biến cho tiếng Việt.",
        reference="Nguyen & Nguyen (2020), Findings of EMNLP",
    ),
    DLModelSpec(
        "visobert",
        "transformer",
        "transformer_raw",
        model_name="uitnlp/visobert",
        train=_TRANSFORMER_TRAIN,
        max_length=160,
        heavy=True,
        arch_kwargs={"pooling": "aspect_attention"},
        summary="Fine-tune ViSoBERT (pretrain trên mạng xã hội tiếng Việt) – hợp teencode/emoji.",
        reference="Nguyen et al. (2023), EMNLP",
    ),
    DLModelSpec(
        "xlm_roberta_base",
        "transformer",
        "transformer_raw",
        model_name="FacebookAI/xlm-roberta-base",
        train=_TRANSFORMER_TRAIN,
        max_length=160,
        heavy=True,
        arch_kwargs={"pooling": "aspect_attention"},
        summary="Fine-tune XLM-RoBERTa-base (đa ngữ 100 ngôn ngữ).",
        reference="Conneau et al. (2020), ACL",
    ),
    DLModelSpec(
        "mdeberta_v3_base",
        "transformer",
        "transformer_raw",
        model_name="microsoft/mdeberta-v3-base",
        train={**_TRANSFORMER_TRAIN, "encoder_lr": 3e-5},
        max_length=160,
        heavy=True,
        arch_kwargs={"pooling": "aspect_attention"},
        summary="Fine-tune mDeBERTa-v3-base (disentangled attention, pretrain kiểu ELECTRA).",
        reference="He et al. (2023), ICLR",
    ),
    DLModelSpec(
        "phobert_large",
        "transformer",
        "transformer_seg",
        model_name="vinai/phobert-large",
        train={**_TRANSFORMER_TRAIN, "encoder_lr": 1e-5, "batch_size": 8},
        max_length=160,
        heavy=True,
        arch_kwargs={"pooling": "aspect_attention"},
        summary="Fine-tune PhoBERT-large (370M) – mạnh nhất nhưng cần GPU >= 16GB.",
        reference="Nguyen & Nguyen (2020), Findings of EMNLP",
    ),
]

DL_REGISTRY: dict[str, DLModelSpec] = {spec.name: spec for spec in _SPECS}


def get_dl_spec(name: str) -> DLModelSpec:
    """Lấy :class:`DLModelSpec` theo tên."""
    if name not in DL_REGISTRY:
        raise KeyError(f"Mô hình DL '{name}' không tồn tại. Có: {list(DL_REGISTRY)}")
    return DL_REGISTRY[name]


def list_dl_models(include_heavy: bool = False) -> list[str]:
    """Liệt kê mô hình DL (mặc định bỏ mô hình cần GPU)."""
    return [s.name for s in DL_REGISTRY.values() if include_heavy or not s.heavy]


def dl_catalog() -> pd.DataFrame:
    """Bảng mô tả các mô hình DL."""
    return pd.DataFrame(
        [
            {
                "name": s.name,
                "arch": s.arch,
                "embedding / backbone": s.model_name or s.embedding,
                "input_profile": s.profile,
                "needs_gpu": s.heavy,
                "idea": s.summary,
                "reference": s.reference,
            }
            for s in DL_REGISTRY.values()
        ]
    ).set_index("name")
