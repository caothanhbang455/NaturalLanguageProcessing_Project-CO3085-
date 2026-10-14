"""Dữ liệu cho mô hình học sâu: từ vựng, Dataset và hàm gộp batch.

Hai loại đầu vào:

* **Mức từ** (TextCNN, BiLSTM): :class:`Vocabulary` ánh xạ token -> chỉ số; embedding
  có thể khởi tạo từ Word2Vec/FastText đã học trên train.
* **Subword** (PhoBERT, ViSoBERT, XLM-R): dùng tokenizer của mô hình pretrained.

Mọi batch có cùng định dạng ``{"input_ids", "attention_mask", "labels"}`` nên một
:class:`~absa.models.dl.trainer.ABSATrainer` dùng chung cho mọi kiến trúc.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"


class Vocabulary:
    """Từ điển token -> chỉ số, học từ văn bản train.

    Args:
        min_freq: Token xuất hiện ít hơn ngưỡng này được coi là ``<unk>``.
        max_size: Giới hạn số token (giữ token phổ biến nhất).
    """

    def __init__(self, min_freq: int = 2, max_size: int | None = None) -> None:
        self.min_freq = min_freq
        self.max_size = max_size
        self.itos: list[str] = [PAD_TOKEN, UNK_TOKEN]
        self.stoi: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    def fit(self, texts: Sequence[str]) -> Vocabulary:
        """Xây từ vựng từ danh sách văn bản đã tiền xử lý."""
        counter = Counter(tok for t in texts for tok in t.split())
        for token, count in counter.most_common(self.max_size):
            if count < self.min_freq:
                break
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
        return self

    def encode(self, text: str, max_length: int) -> list[int]:
        """Chuyển văn bản thành dãy chỉ số (cắt ở ``max_length``, tối thiểu 1 token)."""
        ids = [self.stoi.get(tok, 1) for tok in text.split()[:max_length]]
        return ids or [1]

    def __len__(self) -> int:
        return len(self.itos)


def build_embedding_matrix(
    vocab: Vocabulary, keyed_vectors, dim: int, seed: int = 42
) -> np.ndarray:
    """Ma trận embedding khởi tạo từ vector pretrained; token thiếu được khởi tạo ngẫu nhiên.

    Args:
        vocab: Từ vựng.
        keyed_vectors: ``gensim`` KeyedVectors (``None`` = khởi tạo ngẫu nhiên hoàn toàn).
        dim: Số chiều embedding.
        seed: Seed cho phần khởi tạo ngẫu nhiên.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.1, size=(len(vocab), dim)).astype(np.float32)
    matrix[0] = 0.0  # <pad>
    if keyed_vectors is None:
        return matrix
    # FastText sinh được vector cho mọi token nhờ n-gram ký tự; Word2Vec thì không.
    fasttext_like = type(keyed_vectors).__name__ == "FastTextKeyedVectors"
    for token, idx in vocab.stoi.items():
        if idx < 2:
            continue
        if token in keyed_vectors.key_to_index or fasttext_like:
            matrix[idx] = keyed_vectors[token]
    return matrix


class ABSADataset(Dataset):
    """Dataset trả về ``(chỉ số token, nhãn)`` cho từng bình luận.

    Args:
        texts: Văn bản đã tiền xử lý.
        labels: Ma trận ``Y`` (n, 11) hoặc ``None`` khi chỉ dự đoán.
        encode: Hàm ``str -> list[int]`` (từ vựng hoặc tokenizer pretrained).
    """

    def __init__(
        self, texts: Sequence[str], labels: np.ndarray | None, encode: Callable[[str], list[int]]
    ) -> None:
        self.ids = [encode(t) for t in texts]
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        label = None if self.labels is None else self.labels[idx]
        return self.ids[idx], label


def make_collate(pad_id: int = 0) -> Callable:
    """Tạo hàm gộp batch: đệm về cùng độ dài và sinh ``attention_mask``."""

    def collate(batch):
        ids, labels = zip(*batch)
        max_len = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(ids), max_len), dtype=torch.long)
        for i, seq in enumerate(ids):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, : len(seq)] = 1
        out = {"input_ids": input_ids, "attention_mask": mask}
        if labels[0] is not None:
            out["labels"] = torch.tensor(np.stack(labels), dtype=torch.long)
        return out

    return collate


class VocabEncoder:
    """Mã hóa văn bản bằng :class:`Vocabulary` (dạng lớp để pickle được khi lưu mô hình)."""

    def __init__(self, vocab: Vocabulary, max_length: int = 128) -> None:
        self.vocab = vocab
        self.max_length = max_length

    def __call__(self, text: str) -> list[int]:
        """Mã hóa một văn bản thành dãy chỉ số token."""
        return self.vocab.encode(text, self.max_length)


class HFTokenEncoder:
    """Mã hóa văn bản bằng tokenizer Hugging Face (có token đặc biệt, cắt độ dài)."""

    def __init__(self, tokenizer, max_length: int = 160) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, text: str) -> list[int]:
        """Mã hóa một văn bản thành dãy chỉ số token."""
        return self.tokenizer(text, truncation=True, max_length=self.max_length)["input_ids"]
