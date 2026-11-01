"""Script làm sạch dữ liệu UIT-ViSFD theo quy trình chuẩn:

1. Bỏ cột n_star.
2. Tiền xử lý 9 bước trên cột comment.
3. Đổi tên cột comment thành clean_comment.
4. Giữ nguyên các cột index, date_time, label.

Cách dùng:
    python scripts/clean_data.py
    python scripts/clean_data.py --splits train dev
    python scripts/clean_data.py --input UIT-ViSFD/Train.csv --output UIT-ViSFD/Train_clean.csv
"""

import argparse
import sys
from pathlib import Path

# Thêm src vào sys.path để chạy trực tiếp không cần pip install -e .
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from absa.config import PATHS, SPLIT_FILES
from absa.preprocessing import clean_csv
from absa.utils.logging import get_logger

logger = get_logger("clean_data")


def main():
    parser = argparse.ArgumentParser(description="Làm sạch tập dữ liệu UIT-ViSFD")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev"],
        choices=["train", "dev", "test"],
        help="Các tập cần làm sạch trong UIT-ViSFD (mặc định: train dev)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Đường dẫn file CSV đầu vào (nếu chỉ xử lý 1 file tùy chỉnh)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Đường dẫn file CSV đầu ra (nếu chỉ xử lý 1 file tùy chỉnh)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_clean",
        help="Hậu tố thêm vào tên file khi làm sạch theo splits (vd: Train_clean.csv)",
    )
    args = parser.parse_args()

    if args.input and args.output:
        logger.info("Làm sạch file đơn: %s -> %s", args.input, args.output)
        clean_csv(args.input, args.output, show_progress=True)
        return

    data_dir = PATHS.data_dir
    for split in args.splits:
        filename = SPLIT_FILES.get(split)
        if not filename:
            continue
        in_path = data_dir / filename
        if not in_path.exists():
            logger.warning("Không tìm thấy file %s, bỏ qua.", in_path)
            continue

        out_name = in_path.stem + args.suffix + in_path.suffix
        out_path = data_dir / out_name
        logger.info("=== Đang xử lý tập [%s]: %s -> %s ===", split.upper(), in_path.name, out_name)
        clean_csv(in_path, out_path, show_progress=True)

    logger.info("Hoàn tất quy trình làm sạch dữ liệu.")


if __name__ == "__main__":
    main()
