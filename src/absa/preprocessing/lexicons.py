"""Nạp các từ điển (lexicon) phục vụ tiền xử lý.

Tất cả từ điển nằm ở ``resources/lexicons`` dưới dạng text thuần để người đọc
xem/sửa trực tiếp mà không cần đụng vào code:

* ``teencode.tsv``         – viết tắt/teencode -> dạng chuẩn.
* ``emoji_sentiment.tsv``  – ghi đè nhóm cảm xúc cho emoji phổ biến.
* ``stopwords_vi.txt``     – stopword tiếng Việt (đã tách từ).
* ``protected_words.txt``  – từ không bao giờ bị xóa dù nằm trong stopword.
* ``negation_words.txt``   – từ phủ định.

Dòng bắt đầu bằng ``#`` là chú thích và bị bỏ qua.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from absa.config import PATHS
from absa.utils.io import hash_text


def _iter_lines(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


def load_mapping(path: str | Path) -> dict[str, str]:
    """Đọc file TSV ``khóa<TAB>giá_trị``; khóa được đưa về chữ thường.

    Nếu một khóa xuất hiện nhiều lần, lần xuất hiện **đầu tiên** được giữ.
    """
    mapping: dict[str, str] = {}
    for line in _iter_lines(Path(path)):
        if "\t" not in line:
            continue
        key, value = (part.strip() for part in line.split("\t", 1))
        mapping.setdefault(key.lower(), value)
    return mapping


def load_word_list(path: str | Path) -> frozenset[str]:
    """Đọc danh sách từ (mỗi dòng một từ), trả về ``frozenset`` chữ thường."""
    return frozenset(line.lower() for line in _iter_lines(Path(path)))


@dataclass(frozen=True)
class Lexicons:
    """Gói tất cả từ điển đã nạp.

    Attributes:
        teencode: Ánh xạ teencode -> từ chuẩn.
        emoji_sentiment: Ánh xạ emoji -> ``"pos" | "neg" | "neu"``.
        stopwords: Stopword **đã trừ** đi ``protected``.
        protected: Từ được bảo vệ khỏi bước xóa stopword.
        negations: Từ phủ định.
        fingerprint: Mã băm nội dung các file, dùng làm một phần khóa cache –
            sửa từ điển thì cache tiền xử lý tự động bị vô hiệu.
    """

    teencode: dict[str, str]
    emoji_sentiment: dict[str, str]
    stopwords: frozenset[str]
    protected: frozenset[str]
    negations: frozenset[str]
    fingerprint: str


@lru_cache(maxsize=4)
def load_lexicons(lexicon_dir: str | Path | None = None) -> Lexicons:
    """Nạp toàn bộ từ điển (có cache trong bộ nhớ, gọi nhiều lần không tốn thêm)."""
    directory = Path(lexicon_dir or PATHS.lexicon_dir)
    files = {
        "teencode": directory / "teencode.tsv",
        "emoji": directory / "emoji_sentiment.tsv",
        "stopwords": directory / "stopwords_vi.txt",
        "protected": directory / "protected_words.txt",
        "negations": directory / "negation_words.txt",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu file từ điển: {missing}")

    protected = load_word_list(files["protected"])
    fingerprint = hash_text("".join(p.read_text(encoding="utf-8") for p in files.values()))
    return Lexicons(
        teencode=load_mapping(files["teencode"]),
        emoji_sentiment=load_mapping(files["emoji"]),
        stopwords=load_word_list(files["stopwords"]) - protected,
        protected=protected,
        negations=load_word_list(files["negations"]),
        fingerprint=fingerprint,
    )
