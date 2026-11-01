UPDATE_1
│
├── clean_data_all_in_one.py             # File code All-in-One độc lập chạy full 11 bước
├── final1.md                            # Báo cáo kỹ thuật chi tiết toàn diện
├── cleaning-pipeline-steps-rules.md     # Tài liệu định nghĩa 11 bước và 5 RULE clean data
├── so-sanh-pipeline-tfidf.md            # Báo cáo đối chiếu bảng 9 bước, Gap analysis & TF-IDF
│
├── docs\
│   └── preprocessing.md                 # Tài liệu thiết kế tiền xử lý của package
│
├── scripts\
│   └── clean_data.py                    # Script CLI chạy tự động trên tập dữ liệu
│
├── tests\
│   └── test_preprocessing.py            # Unit test kiểm thử pipeline làm sạch
│
└── src\absa\
    ├── config.py                        # Cấu hình hằng số (thêm CLEAN_TEXT_COL)
    ├── data\
    │   └── loader.py                    # Loader nạp dữ liệu hỗ trợ cả comment và clean_comment
    └── preprocessing\
        ├── pipeline.py                  # Pipeline Scikit-learn (thêm clean_dataframe, clean_csv)
        ├── __init__.py                  # Export các hàm làm sạch
        ├── normalizers.py               # Hệ thống regex chuẩn hóa
        ├── tokenization.py              # Xử lý gắn phủ định và ranh giới câu
        └── lexicons.py                  # Bộ nạp từ điển teencode, emoji, phủ định

Mọi trình bày report ghi trong final1.md