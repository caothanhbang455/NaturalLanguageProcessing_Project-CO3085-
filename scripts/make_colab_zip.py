"""Đóng gói project thành file zip sạch để đưa lên Google Drive và chạy trên Colab.

File zip chứa **một thư mục gốc duy nhất** tên là tên project (mặc định ``absa-visfd``),
bên trong có đủ ``src``, ``notebooks``, ``resources``, ``UIT-ViSFD``... Các thư mục nặng hoặc
sinh ra khi chạy (``outputs``, ``refs``, ``graphify-out``, cache) bị loại.

Cách dùng::

    python scripts/make_colab_zip.py                    # -> dist/absa-visfd.zip
    python scripts/make_colab_zip.py --name MyProject   # đổi tên thư mục gốc trong zip
    python scripts/make_colab_zip.py --include-test     # kèm Test.csv (mặc định KHÔNG kèm)
"""

from __future__ import annotations

import argparse
import fnmatch
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME = "absa-visfd"

# Thư mục/tệp (so khớp theo từng thành phần của đường dẫn) không đưa vào zip.
EXCLUDE_DIRS = {
    "outputs",
    "refs",
    "dist",
    "graphify-out",
    ".claude",
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "node_modules",
}
EXCLUDE_FILES = [
    "*.pyc",
    "*.pyo",
    "*.zip",
    "note.txt",
    "CLAUDE.md",
    ".DS_Store",
    "_*.ipynb",
    "*.egg-info",
]
TEST_FILE = "Test.csv"


def should_skip(relative: Path, include_test: bool) -> bool:
    """Quyết định bỏ qua một đường dẫn (tương đối so với gốc project)."""
    parts = relative.parts
    if any(part in EXCLUDE_DIRS or part.endswith(".egg-info") for part in parts):
        return True
    if relative.name == TEST_FILE and not include_test:
        return True
    return any(fnmatch.fnmatch(relative.name, pattern) for pattern in EXCLUDE_FILES)


def build_zip(
    name: str = DEFAULT_NAME, include_test: bool = False, out_dir: Path | None = None
) -> Path:
    """Tạo ``dist/<name>.zip`` và trả về đường dẫn."""
    out_dir = out_dir or ROOT / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.zip"
    files = sorted(
        p
        for p in ROOT.rglob("*")
        if p.is_file() and not should_skip(p.relative_to(ROOT), include_test)
    )
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, Path(name) / path.relative_to(ROOT))
    return target


def main() -> None:
    """Điểm vào dòng lệnh."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--name", default=DEFAULT_NAME, help="tên thư mục gốc bên trong zip")
    parser.add_argument("--include-test", action="store_true", help="kèm UIT-ViSFD/Test.csv")
    args = parser.parse_args()

    target = build_zip(args.name, args.include_test)
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
    size_mb = target.stat().st_size / 1024**2
    print(f"Đã tạo {target}  ({size_mb:.2f} MB, {len(names)} file)")
    print("Thư mục gốc trong zip:", sorted({n.split("/")[0] for n in names}))
    print("Nhóm cấp 2:", sorted({"/".join(n.split("/")[:2]) for n in names if "/" in n})[:30])


if __name__ == "__main__":
    main()
