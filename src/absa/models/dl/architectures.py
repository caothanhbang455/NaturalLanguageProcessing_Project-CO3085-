"""Kiến trúc học sâu cho ABSA đa khía cạnh.

Mọi kiến trúc gồm hai phần:

* **Encoder** biến dãy token thành biểu diễn: TextCNN, BiLSTM hoặc Transformer pretrained.
* **Aspect heads**: mỗi khía cạnh một lớp tuyến tính riêng dự đoán
  ``None/Neg/Neu/Pos`` (OTHERS: 2 lớp). Đây là phiên bản "joint" của ABSA trong một
  mạng duy nhất – encoder dùng chung, đầu ra tách theo khía cạnh.

Tùy chọn **aspect attention pooling**: mỗi khía cạnh có một vector truy vấn học được,
tự "chú ý" vào các token liên quan tới nó (``pin``, ``sạc`` cho BATTERY; ``nhân viên``
cho SER&ACC) thay vì dùng chung một vector câu. Ý tưởng theo ATAE/aspect-specific
attention (Wang et al., 2016, EMNLP).
"""

from __future__ import annotations

import torch
from torch import nn

from absa.config import ALL_ASPECTS, ASPECT_CLASS_NAMES, OTHERS_ASPECT, OTHERS_CLASS_NAMES

CLASSES_PER_ASPECT: tuple[int, ...] = tuple(
    len(OTHERS_CLASS_NAMES) if a == OTHERS_ASPECT else len(ASPECT_CLASS_NAMES) for a in ALL_ASPECTS
)
POOLINGS = ("cls", "mean", "max", "aspect_attention")


class AspectAttentionPooling(nn.Module):
    """Gộp chuỗi trạng thái ẩn thành **một vector riêng cho từng khía cạnh**.

    Args:
        hidden_size: Số chiều trạng thái ẩn.
        n_aspects: Số khía cạnh.

    Shape:
        - hidden: ``(B, L, H)``, mask: ``(B, L)``
        - output: ``(B, A, H)``
    """

    def __init__(self, hidden_size: int, n_aspects: int = len(ALL_ASPECTS)) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_aspects, hidden_size) * 0.02)
        self.key = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Trả về vector theo khía cạnh ``(B, A, H)``."""
        keys = torch.tanh(self.key(hidden))  # (B, L, H)
        scores = torch.einsum("blh,ah->bal", keys, self.queries)  # (B, A, L)
        scores = scores.masked_fill(mask[:, None, :] == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bal,blh->bah", weights, hidden)


def pool_sequence(hidden: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Gộp ``(B, L, H)`` thành ``(B, H)`` theo ``cls``, ``mean`` hoặc ``max``."""
    if mode == "cls":
        return hidden[:, 0]
    m = mask.unsqueeze(-1).to(hidden.dtype)
    if mode == "mean":
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
    if mode == "max":
        return hidden.masked_fill(m == 0, torch.finfo(hidden.dtype).min).max(1).values
    raise ValueError(f"pooling không hợp lệ: {mode}")


class AspectHeads(nn.Module):
    """Một bộ phân loại tuyến tính cho mỗi khía cạnh.

    Nhận ``(B, H)`` (vector câu dùng chung) hoặc ``(B, A, H)`` (vector theo khía cạnh) và
    trả về list ``A`` tensor logits, phần tử thứ ``a`` có dạng ``(B, C_a)``.
    """

    def __init__(self, in_features: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList(nn.Linear(in_features, c) for c in CLASSES_PER_ASPECT)

    def forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        """Trả về list logits, mỗi khía cạnh một tensor ``(B, C_a)``."""
        features = self.dropout(features)
        if features.dim() == 2:
            return [head(features) for head in self.heads]
        return [head(features[:, a]) for a, head in enumerate(self.heads)]


def _embedding(vocab_size: int, dim: int, weights=None, freeze: bool = False) -> nn.Embedding:
    if weights is not None:
        return nn.Embedding.from_pretrained(torch.as_tensor(weights), freeze=freeze, padding_idx=0)
    return nn.Embedding(vocab_size, dim, padding_idx=0)


class TextCNNABSA(nn.Module):
    """TextCNN (Kim, 2014): tích chập 1D nhiều kích thước cửa sổ + max-pooling theo thời gian.

    Mỗi kích thước kernel ``k`` đóng vai trò bộ phát hiện k-gram (``pin trâu``,
    ``không mượt lắm``). Nhanh, mạnh với câu ngắn như bình luận sản phẩm.

    Args:
        vocab_size: Kích thước từ vựng.
        embed_dim: Số chiều embedding.
        embedding_weights: Ma trận khởi tạo (Word2Vec/FastText) hoặc ``None``.
        kernel_sizes: Các kích thước cửa sổ.
        n_filters: Số bộ lọc mỗi kích thước.
        dropout: Tỉ lệ dropout.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        embedding_weights=None,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
        n_filters: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = _embedding(vocab_size, embed_dim, embedding_weights)
        self.convs = nn.ModuleList(
            nn.Conv1d(embed_dim, n_filters, k, padding=k // 2) for k in kernel_sizes
        )
        self.heads = AspectHeads(n_filters * len(kernel_sizes), dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
        """Trả về list logits theo khía cạnh."""
        x = self.embedding(input_ids).transpose(1, 2)  # (B, E, L)
        mask = attention_mask[:, None, :]
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))[:, :, : input_ids.size(1)]
            h = h.masked_fill(mask == 0, torch.finfo(h.dtype).min)
            pooled.append(h.max(dim=2).values)
        return self.heads(torch.cat(pooled, dim=1))


class BiLSTMAttentionABSA(nn.Module):
    """BiLSTM 2 chiều + aspect attention pooling.

    BiLSTM đọc câu theo cả hai chiều nên mỗi token mang ngữ cảnh hai phía (hiểu được
    ``không ... lắm``); attention theo khía cạnh chọn đúng đoạn nói về khía cạnh đó trong
    bình luận dài nhiều ý.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        embedding_weights=None,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = _embedding(vocab_size, embed_dim, embedding_weights)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pool = AspectAttentionPooling(2 * hidden_size)
        self.heads = AspectHeads(2 * hidden_size, dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
        """Trả về list logits theo khía cạnh."""
        lengths = attention_mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embedding(input_ids), lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, total_length=input_ids.size(1)
        )
        return self.heads(self.pool(hidden, attention_mask))


class TransformerABSA(nn.Module):
    """Encoder Transformer pretrained (PhoBERT, ViSoBERT, XLM-R...) + aspect heads, fine-tune.

    Args:
        model_name: Tên mô hình trên Hugging Face Hub.
        pooling: ``"cls" | "mean" | "max" | "aspect_attention"``.
        dropout: Dropout trước các head.
    """

    def __init__(self, model_name: str, pooling: str = "aspect_attention", dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel

        if pooling not in POOLINGS:
            raise ValueError(f"pooling phải thuộc {POOLINGS}")
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.pooling = pooling
        self.pool = AspectAttentionPooling(hidden) if pooling == "aspect_attention" else None
        self.heads = AspectHeads(hidden, dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
        """Trả về list logits theo khía cạnh."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pool is not None:
            return self.heads(self.pool(hidden, attention_mask))
        return self.heads(pool_sequence(hidden, attention_mask, self.pooling))
