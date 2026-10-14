"""Cấu hình trung tâm: đường dẫn, hằng số nhãn và các metric dùng chung toàn project.

Mọi module khác đều import hằng số từ đây, nhờ vậy khi đổi vị trí dữ liệu hoặc
thêm/bớt khía cạnh thì chỉ cần sửa đúng một nơi.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 42


@dataclass(frozen=True)
class Paths:
    """Tập hợp các thư mục của project.

    Attributes:
        root: Thư mục gốc của project.
        data_dir: Nơi chứa ``Train.csv``, ``Dev.csv`` (và ``Test.csv`` nếu có).
        resources_dir: Tài nguyên tĩnh (từ điển teencode, emoji, stopword...).
        outputs_dir: Mọi thứ sinh ra khi chạy (cache, mô hình, báo cáo, hình).
    """

    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "UIT-ViSFD"
    resources_dir: Path = PROJECT_ROOT / "resources"
    outputs_dir: Path = PROJECT_ROOT / "outputs"

    @property
    def lexicon_dir(self) -> Path:
        """Thư mục chứa các từ điển dùng cho tiền xử lý."""
        return self.resources_dir / "lexicons"

    @property
    def cache_dir(self) -> Path:
        """Cache văn bản đã tiền xử lý và embedding đã tính (xóa được an toàn)."""
        return self.outputs_dir / "cache"

    @property
    def models_dir(self) -> Path:
        """Nơi lưu pipeline/mô hình đã huấn luyện."""
        return self.outputs_dir / "models"

    @property
    def reports_dir(self) -> Path:
        """Nơi lưu bảng kết quả thí nghiệm (CSV/JSON)."""
        return self.outputs_dir / "reports"

    @property
    def figures_dir(self) -> Path:
        """Nơi lưu hình vẽ phục vụ báo cáo."""
        return self.outputs_dir / "figures"


PATHS = Paths()

# --------------------------------------------------------------------------- #
# Dữ liệu
# --------------------------------------------------------------------------- #
SPLIT_FILES: dict[str, str] = {"train": "Train.csv", "dev": "Dev.csv", "test": "Test.csv"}
TEXT_COL = "comment"
LABEL_COL = "label"
STAR_COL = "n_star"

# --------------------------------------------------------------------------- #
# Nhãn ABSA
# --------------------------------------------------------------------------- #
SENTIMENT_ASPECTS: tuple[str, ...] = (
    "SCREEN",
    "CAMERA",
    "FEATURES",
    "BATTERY",
    "PERFORMANCE",
    "STORAGE",
    "DESIGN",
    "PRICE",
    "GENERAL",
    "SER&ACC",
)
OTHERS_ASPECT = "OTHERS"  # Khía cạnh đặc biệt: chỉ có mặt/không, không mang cảm xúc.
ALL_ASPECTS: tuple[str, ...] = SENTIMENT_ASPECTS + (OTHERS_ASPECT,)

POLARITIES: tuple[str, ...] = ("Negative", "Neutral", "Positive")
NONE_ID = 0  # Khía cạnh không được nhắc tới.
POLARITY_TO_ID: dict[str, int] = {"Negative": 1, "Neutral": 2, "Positive": 3}
ID_TO_POLARITY: dict[int, str] = {v: k for k, v in POLARITY_TO_ID.items()}
OTHERS_PRESENT_ID = 1  # Cột OTHERS chỉ nhận 0 (không có) hoặc 1 (có).

# Tên lớp cho từng cột của ma trận nhãn (dùng khi vẽ confusion matrix).
ASPECT_CLASS_NAMES: tuple[str, ...] = ("None", "Negative", "Neutral", "Positive")
OTHERS_CLASS_NAMES: tuple[str, ...] = ("None", "Present")

# --------------------------------------------------------------------------- #
# Đánh giá
# --------------------------------------------------------------------------- #
# Metric dùng để chọn mô hình: macro-F1 trên các cặp (khía cạnh, cảm xúc).
# Macro được chọn vì dữ liệu lệch mạnh (STORAGE, SCREEN#Neutral rất hiếm); micro-F1
# vẫn được báo cáo song song để so sánh với các công bố khác.
PRIMARY_METRIC = "pair_f1_macro"
