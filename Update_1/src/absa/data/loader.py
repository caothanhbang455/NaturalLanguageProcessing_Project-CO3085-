"""Đọc các tập dữ liệu UIT-ViSFD và kiểm tra schema.

Chính sách sử dụng 3 tập (quy định cho toàn project):

* **Train** – dùng cho EDA, thiết kế tiền xử lý, fit vectorizer/mô hình.
* **Dev**   – đóng vai trò *validation*: chọn mô hình, tinh chỉnh siêu tham số,
  early stopping. Chỉ được truy cập qua code để tính metric; **không đọc nội dung
  bình luận để đặt luật tiền xử lý** (tránh rò rỉ thông tin khiến điểm dev bị ảo).
* **Test**  – *hidden*: chỉ đánh giá **một lần duy nhất** ở cuối với cấu hình đã
  chốt. Nếu file không tồn tại, các hàm trả về ``None`` thay vì báo lỗi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from absa.config import CLEAN_TEXT_COL, LABEL_COL, PATHS, SPLIT_FILES, STAR_COL, TEXT_COL
from absa.data.labels import encode_labels
from absa.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ABSASplit:
    """Một tập dữ liệu đã đọc và mã hóa nhãn.

    Attributes:
        name: ``"train"``, ``"dev"`` hoặc ``"test"``.
        frame: DataFrame gốc (giữ lại ``n_star``, ``date_time`` cho EDA).
        texts: Danh sách bình luận thô (chưa tiền xử lý).
        labels: Ma trận nhãn ``Y`` (n, 11) – xem :mod:`absa.data.labels`.
    """

    name: str
    frame: pd.DataFrame
    texts: list[str] = field(repr=False)
    labels: np.ndarray = field(repr=False)

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def stars(self) -> np.ndarray | None:
        """Số sao người dùng chấm (nếu có cột ``n_star``)."""
        return self.frame[STAR_COL].to_numpy() if STAR_COL in self.frame else None


def get_text_column(frame: pd.DataFrame) -> str:
    """Xác định cột văn bản (ưu tiên ``clean_comment`` nếu có, sau đó ``comment``)."""
    if CLEAN_TEXT_COL in frame.columns:
        return CLEAN_TEXT_COL
    if TEXT_COL in frame.columns:
        return TEXT_COL
    raise ValueError(
        f"Không tìm thấy cột văn bản ('{TEXT_COL}' hoặc '{CLEAN_TEXT_COL}') trong: {list(frame.columns)}"
    )


def validate_frame(frame: pd.DataFrame, split: str) -> None:
    """Kiểm tra các cột bắt buộc; báo lỗi rõ ràng nếu file sai định dạng."""
    if LABEL_COL not in frame.columns:
        raise ValueError(f"Tập '{split}' thiếu cột nhãn bắt buộc: '{LABEL_COL}'")
    if TEXT_COL not in frame.columns and CLEAN_TEXT_COL not in frame.columns:
        raise ValueError(
            f"Tập '{split}' thiếu cột văn bản ('{TEXT_COL}' hoặc '{CLEAN_TEXT_COL}')"
        )


def load_split(
    split: str, data_dir: str | Path | None = None, required: bool = True
) -> ABSASplit | None:
    """Đọc một tập dữ liệu.

    Args:
        split: ``"train"``, ``"dev"`` hoặc ``"test"``.
        data_dir: Thư mục chứa CSV, mặc định ``PATHS.data_dir``.
        required: Nếu ``False`` và file không tồn tại thì trả về ``None``
            (dùng cho tập test bị ẩn).

    Returns:
        :class:`ABSASplit` hoặc ``None``.
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"split phải thuộc {list(SPLIT_FILES)}, nhận '{split}'")
    path = Path(data_dir or PATHS.data_dir) / SPLIT_FILES[split]
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        logger.info("Không tìm thấy %s – coi như tập '%s' đang bị ẩn.", path.name, split)
        return None

    frame = pd.read_csv(path)
    frame = frame.loc[:, ~frame.columns.str.match(r"^Unnamed")]
    validate_frame(frame, split)
    text_col = get_text_column(frame)
    frame[text_col] = frame[text_col].fillna("").astype(str)
    labels = encode_labels(frame[LABEL_COL])
    return ABSASplit(split, frame, frame[text_col].tolist(), labels)


def load_splits(
    data_dir: str | Path | None = None, include_test: bool = True
) -> dict[str, ABSASplit | None]:
    """Đọc train, dev và (tùy chọn) test. Test vắng mặt sẽ có giá trị ``None``."""
    splits: dict[str, ABSASplit | None] = {
        "train": load_split("train", data_dir),
        "dev": load_split("dev", data_dir),
    }
    if include_test:
        splits["test"] = load_split("test", data_dir, required=False)
    return splits
