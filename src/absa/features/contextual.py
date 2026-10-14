"""Embedding ngữ cảnh (contextual) từ mô hình Transformer pretrained, dùng ở dạng *đóng băng*.

Mô hình không được fine-tune: ta chỉ lấy vector câu làm đặc trưng cho các bộ phân loại
ML truyền thống. Cách này rẻ (chạy được trên CPU) và cho phép so sánh công bằng
"chất lượng biểu diễn" giữa TF-IDF và embedding hiện đại trên cùng một bộ phân loại.

Hai backend:

* ``"sentence_transformers"`` – cho mô hình đã huấn luyện để tạo embedding câu
  (SBERT, multilingual-E5, BGE-M3, Qwen3-Embedding, vietnamese-bi-encoder).
* ``"hf_mean"`` – cho encoder thuần (PhoBERT, ViSoBERT, XLM-R): mean pooling trên
  hidden state lớp cuối, có tính attention mask.

Embedding được cache **theo từng câu** ra đĩa, nên cross-validation hay chạy lại
notebook không phải tính lại.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from absa.config import PATHS
from absa.utils.io import ensure_dir, hash_text
from absa.utils.logging import get_logger

logger = get_logger(__name__)
BACKENDS = ("sentence_transformers", "hf_mean")


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


class EmbeddingCache:
    """Cache embedding theo câu: ``{md5(câu): vector}`` lưu ở một file ``.npz``.

    Args:
        path: File cache; ``None`` = chỉ cache trong bộ nhớ.
    """

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self.store: dict[str, np.ndarray] = {}
        if self.path and self.path.exists():
            with np.load(self.path, allow_pickle=False) as data:
                keys, vectors = data["keys"], data["vectors"]
            self.store = {str(k): v for k, v in zip(keys, vectors)}

    def missing(self, keys: Sequence[str]) -> list[str]:
        """Các khóa chưa có trong cache."""
        return [k for k in dict.fromkeys(keys) if k not in self.store]

    def add(self, keys: Sequence[str], vectors: np.ndarray) -> None:
        """Thêm vector mới."""
        self.store.update(zip(keys, vectors.astype(np.float32)))

    def get(self, keys: Sequence[str]) -> np.ndarray:
        """Lấy ma trận vector theo đúng thứ tự ``keys``."""
        return np.vstack([self.store[k] for k in keys])

    def save(self) -> None:
        """Ghi cache ra đĩa (nếu có ``path``)."""
        if not self.path or not self.store:
            return
        ensure_dir(self.path.parent)
        keys = np.array(list(self.store))
        vectors = np.vstack([self.store[k] for k in keys])
        np.savez(self.path, keys=keys, vectors=vectors)


class TransformerEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Trích vector câu từ mô hình Transformer đóng băng.

    Args:
        model_name: Tên mô hình trên Hugging Face Hub.
        backend: ``"sentence_transformers"`` hoặc ``"hf_mean"``.
        prefix: Tiền tố thêm vào mỗi câu (E5 yêu cầu ``"query: "``).
        max_length: Số token tối đa (PhoBERT tối đa 256).
        batch_size: Kích thước batch khi mã hóa.
        normalize: Chuẩn hóa L2 vector đầu ra (hợp với SVM/LR và độ đo cosine).
        device: ``"cpu"``, ``"cuda"``... ``None`` = tự chọn.
        trust_remote_code: Cho phép code tùy biến của mô hình (gte-multilingual cần).
        cache_dir: Thư mục cache embedding; ``None`` = ``outputs/cache/embeddings``.
        use_cache: Bật/tắt cache đĩa.
    """

    def __init__(
        self,
        model_name: str,
        backend: str = "sentence_transformers",
        prefix: str = "",
        max_length: int = 256,
        batch_size: int = 32,
        normalize: bool = True,
        device: str | None = None,
        trust_remote_code: bool = False,
        cache_dir: str | None = None,
        use_cache: bool = True,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self.prefix = prefix
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        self.use_cache = use_cache

    # ------------------------------------------------------------------ model --
    def _load_model(self) -> None:
        device = _resolve_device(self.device)
        logger.info("Nạp %s (%s) trên %s", self.model_name, self.backend, device)
        if self.backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                self.model_name, device=device, trust_remote_code=self.trust_remote_code
            )
            model.max_seq_length = min(self.max_length, model.max_seq_length or self.max_length)
            self._model, self._tokenizer = model, None
        elif self.backend == "hf_mean":
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
            self._model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            ).to(device)
            self._model.eval()
        else:
            raise ValueError(f"backend phải thuộc {BACKENDS}")
        self._device = device

    def _encode(self, texts: list[str]) -> np.ndarray:
        if getattr(self, "_model", None) is None:
            self._load_model()
        texts = [self.prefix + t for t in texts]
        if self.backend == "sentence_transformers":
            return self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > self.batch_size * 4,
            )
        import torch

        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch = self._tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                hidden = self._model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
            outputs.append(pooled.float().cpu().numpy())
        return np.vstack(outputs)

    # ---------------------------------------------------------------- caching --
    def _cache(self) -> EmbeddingCache:
        if getattr(self, "_cache_obj", None) is None:
            path = None
            if self.use_cache:
                root = Path(self.cache_dir or PATHS.cache_dir / "embeddings")
                tag = hash_text(
                    f"{self.model_name}|{self.backend}|{self.prefix}|{self.max_length}|{self.normalize}"
                )[:12]
                path = root / f"{self.model_name.replace('/', '__')}_{tag}.npz"
            self._cache_obj = EmbeddingCache(path)
        return self._cache_obj

    # ------------------------------------------------------------ sklearn API --
    def fit(self, X: Sequence[str], y=None) -> TransformerEmbeddingVectorizer:
        """Không học gì (mô hình đóng băng)."""
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        """Trả về ma trận embedding ``(n, d)``; chỉ mã hóa những câu chưa có trong cache."""
        texts = [t if t.strip() else "." for t in X]
        keys = [hash_text(t) for t in texts]
        cache = self._cache()
        missing = cache.missing(keys)
        if missing:
            key_to_text = dict(zip(keys, texts))
            logger.info("Mã hóa %d câu mới bằng %s", len(missing), self.model_name)
            cache.add(missing, self._encode([key_to_text[k] for k in missing]))
            cache.save()
        return cache.get(keys)

    def __getstate__(self) -> dict:
        # Không pickle trọng số mô hình/cache: tải lại từ Hugging Face khi cần.
        state = self.__dict__.copy()
        for attr in ("_model", "_tokenizer", "_cache_obj"):
            state.pop(attr, None)
        return state
