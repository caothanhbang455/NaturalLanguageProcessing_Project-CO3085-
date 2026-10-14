"""Đọc/ghi file và băm (hash) dữ liệu để làm khóa cache."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib


def ensure_dir(path: str | Path) -> Path:
    """Tạo thư mục (kể cả thư mục cha) nếu chưa tồn tại và trả về ``Path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> Path:
    """Ghi ``obj`` ra file JSON (UTF-8, giữ nguyên tiếng Việt có dấu)."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return path


def load_json(path: str | Path) -> Any:
    """Đọc file JSON UTF-8."""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def save_joblib(obj: Any, path: str | Path, compress: int = 3) -> Path:
    """Lưu đối tượng Python (mô hình, pipeline...) bằng joblib."""
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path, compress=compress)
    return path


def load_joblib(path: str | Path) -> Any:
    """Nạp đối tượng đã lưu bằng :func:`save_joblib`."""
    return joblib.load(Path(path))


def hash_text(text: str) -> str:
    """Băm một chuỗi thành mã hex 32 ký tự (MD5, chỉ dùng làm khóa cache)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def hash_texts(texts: Iterable[str]) -> str:
    """Băm cả một danh sách chuỗi; thứ tự phần tử ảnh hưởng tới kết quả."""
    digest = hashlib.md5()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x1e")  # Ký tự phân cách để ["ab","c"] khác ["a","bc"].
    return digest.hexdigest()


def hash_object(obj: Any) -> str:
    """Băm một đối tượng JSON-hóa được (dict cấu hình, list tham số...)."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hash_text(payload)
