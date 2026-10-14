"""Vòng huấn luyện dùng chung cho mọi kiến trúc học sâu ABSA.

Điểm chính:

* **Loss** = tổng cross-entropy của 11 head; có thể đánh trọng số lớp (nghịch đảo căn
  tần suất) để lớp hiếm như ``SCREEN#Neutral`` không bị bỏ qua.
* **Tối ưu**: AdamW + warmup tuyến tính rồi giảm tuyến tính; learning rate riêng
  (nhỏ hơn) cho encoder pretrained.
* **Early stopping** theo metric chính trên dev (không theo loss), giữ trọng số tốt nhất.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from absa.config import PRIMARY_METRIC, SEED
from absa.evaluation.metrics import evaluate_absa
from absa.models.dl.architectures import CLASSES_PER_ASPECT
from absa.models.dl.data import ABSADataset, make_collate
from absa.utils.logging import get_logger
from absa.utils.reproducibility import set_seed
from absa.utils.timing import Timer

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    """Siêu tham số huấn luyện.

    Attributes:
        epochs: Số epoch tối đa.
        batch_size: Kích thước batch.
        lr: Learning rate cho phần không pretrained (heads, CNN, LSTM).
        encoder_lr: Learning rate cho encoder pretrained (``None`` = dùng ``lr``).
        weight_decay: Hệ số L2 của AdamW.
        warmup_ratio: Tỉ lệ bước warmup.
        max_grad_norm: Ngưỡng cắt gradient.
        patience: Số epoch không cải thiện trước khi dừng sớm.
        class_weighting: Bật trọng số lớp trong loss.
        metric: Metric dev dùng để chọn checkpoint.
        device: ``None`` = tự chọn GPU nếu có.
        seed: Seed.
        num_workers: Số worker của DataLoader.
    """

    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    encoder_lr: float | None = None
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    patience: int = 3
    class_weighting: bool = True
    metric: str = PRIMARY_METRIC
    device: str | None = None
    seed: int = SEED
    num_workers: int = 0


def compute_class_weights(Y: np.ndarray, power: float = 0.5) -> list[torch.Tensor]:
    """Trọng số lớp cho từng khía cạnh: ``(tần_suất)^-power``, chuẩn hóa trung bình 1."""
    weights = []
    for j, n_classes in enumerate(CLASSES_PER_ASPECT):
        counts = np.bincount(np.asarray(Y)[:, j], minlength=n_classes).astype(float)
        w = (counts.clip(min=1) / counts.sum()) ** (-power)
        weights.append(torch.tensor(w / w.mean(), dtype=torch.float32))
    return weights


class ABSATrainer:
    """Huấn luyện, đánh giá và dự đoán cho một mô hình PyTorch ABSA.

    Args:
        model: ``nn.Module`` có ``forward(input_ids, attention_mask) -> list[logits]``.
        config: :class:`TrainConfig`.
        encode: Hàm ``str -> list[int]`` dùng để mã hóa văn bản khi dự đoán.
        pad_id: Chỉ số token đệm.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        encode: Callable[[str], list[int]],
        pad_id: int = 0,
    ) -> None:
        self.model = model
        self.config = config
        self.encode = encode
        self.pad_id = pad_id
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.history: list[dict] = []

    # ------------------------------------------------------------- helpers --
    def _loader(self, texts: Sequence[str], Y: np.ndarray | None, shuffle: bool) -> DataLoader:
        dataset = ABSADataset(texts, Y, self.encode)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=make_collate(self.pad_id),
            num_workers=self.config.num_workers,
        )

    def _optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config
        encoder_params, other_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (encoder_params if name.startswith("encoder.") else other_params).append(param)
        groups = [{"params": other_params, "lr": cfg.lr}]
        if encoder_params:
            groups.append({"params": encoder_params, "lr": cfg.encoder_lr or cfg.lr})
        return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    def _loss(self, logits: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        return sum(
            nn.functional.cross_entropy(
                out, labels[:, a], weight=None if self._weights is None else self._weights[a]
            )
            for a, out in enumerate(logits)
        )

    # ----------------------------------------------------------------- fit --
    def fit(
        self,
        train_texts: Sequence[str],
        Y_train: np.ndarray,
        dev_texts: Sequence[str],
        Y_dev: np.ndarray,
    ) -> pd.DataFrame:
        """Huấn luyện với early stopping theo ``config.metric`` trên dev.

        Returns:
            Lịch sử theo epoch: loss train và các metric dev.
        """
        cfg = self.config
        set_seed(cfg.seed)
        self.model.to(self.device)
        self._weights = (
            [w.to(self.device) for w in compute_class_weights(Y_train)]
            if cfg.class_weighting
            else None
        )
        train_loader = self._loader(train_texts, Y_train, shuffle=True)
        optimizer = self._optimizer()
        total_steps = max(1, cfg.epochs * len(train_loader))
        warmup = int(cfg.warmup_ratio * total_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: (
                min(1.0, (step + 1) / max(1, warmup))
                * max(0.0, (total_steps - step) / max(1, total_steps - warmup))
            ),
        )
        best_score, best_state, bad_epochs = -np.inf, None, 0
        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            running = 0.0
            with Timer() as t:
                for batch in train_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    logits = self.model(batch["input_ids"], batch["attention_mask"])
                    loss = self._loss(logits, batch["labels"])
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    running += loss.item() * len(batch["labels"])
            metrics = evaluate_absa(Y_dev, self.predict(dev_texts))
            record = {
                "epoch": epoch,
                "train_loss": running / len(Y_train),
                "seconds": t.seconds,
                **metrics,
            }
            self.history.append(record)
            logger.info(
                "epoch %2d loss=%.4f %s=%.4f (%.0fs)",
                epoch,
                record["train_loss"],
                cfg.metric,
                metrics[cfg.metric],
                t.seconds,
            )
            if metrics[cfg.metric] > best_score:
                best_score, bad_epochs = metrics[cfg.metric], 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    logger.info("Dừng sớm ở epoch %d (tốt nhất %.4f)", epoch, best_score)
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return pd.DataFrame(self.history)

    # ------------------------------------------------------------- predict --
    @torch.no_grad()
    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Dự đoán ma trận nhãn ``(n, 11)`` cho văn bản đã tiền xử lý."""
        self.model.to(self.device).eval()
        outputs = []
        for batch in self._loader(texts, None, shuffle=False):
            logits = self.model(
                batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)
            )
            outputs.append(torch.stack([out.argmax(-1) for out in logits], dim=1).cpu().numpy())
        return np.vstack(outputs).astype(np.int8)

    def config_dict(self) -> dict:
        """Cấu hình huấn luyện dạng dict (lưu kèm kết quả)."""
        return asdict(self.config)

    def __getstate__(self) -> dict:
        # Đưa mô hình về CPU trước khi pickle để nạp được trên máy không có GPU.
        state = self.__dict__.copy()
        state["model"] = self.model.to("cpu")
        state["device"] = torch.device("cpu")
        state.pop("_weights", None)
        return state
