# ABSA cho đánh giá điện thoại tiếng Việt (UIT-ViSFD)

Đồ án môn Xử lý Ngôn ngữ Tự nhiên (HCMUT, HK261). Hệ thống nhận bình luận của khách hàng về điện thoại, xác định **khía cạnh** được nhắc tới (màn hình, pin, camera, giá, ...) và **cảm xúc** với từng khía cạnh, sau đó tổng hợp điểm mạnh và điểm yếu của sản phẩm.

- **Dữ liệu**: UIT-ViSFD, 11.122 bình luận (train 7.786 / dev 1.112 / test 2.224 – test đang ẩn), 10 khía cạnh + `OTHERS`, 3 cảm xúc.
- **Hai nhánh mô hình**: ML truyền thống (lưới **M mô hình x N biểu diễn**) và Deep Learning (TextCNN, BiLSTM-Attention, fine-tune PhoBERT/ViSoBERT/XLM-R).
- **Một notebook duy nhất** (`notebooks/absa_pipeline.ipynb`) điều phối toàn bộ quy trình; mọi logic nằm trong gói `src/absa`.

## Cấu trúc thư mục

```
.
├── notebooks/absa_pipeline.ipynb   # Notebook duy nhất: chạy full flow từ dữ liệu đến triển khai
├── src/absa/                       # Gói Python chứa toàn bộ logic
│   ├── config.py                   #   Đường dẫn, hằng số nhãn, metric chính
│   ├── data/                       #   Đọc CSV, kiểm tra schema, mã hóa nhãn (labels.py, loader.py)
│   ├── eda/                        #   Thống kê EDA, chỉ trả về bảng số liệu
│   ├── preprocessing/              #   Chuẩn hóa tiếng Việt theo luật tổng quát
│   │   ├── normalizers.py          #     Unicode, emoji, emoticon, số+đơn vị, kéo dài, -ng, teencode
│   │   ├── tokenization.py         #     Tách từ, gắn phủ định, stopword
│   │   ├── lexicons.py             #     Nạp từ điển trong resources/lexicons
│   │   └── pipeline.py             #     PreprocessConfig, TextPreprocessor, PreparedCorpus (có cache)
│   ├── features/                   #   N cách biểu diễn văn bản
│   │   ├── sparse.py               #     BoW, TF-IDF từ/ký tự (8 biến thể sparse trong registry)
│   │   ├── static_embeddings.py    #     Word2Vec, FastText, Doc2Vec, FastText cc.vi
│   │   ├── contextual.py           #     PhoBERT, ViSoBERT, SBERT, E5, GTE, BGE-M3, Qwen3 (đóng băng, có cache)
│   │   └── registry.py             #     Danh mục biểu diễn + tóm tắt ý tưởng
│   ├── models/
│   │   ├── ml/                     #   ABSAClassifier (multilabel / joint / two_stage) + registry M mô hình
│   │   └── dl/                     #   Kiến trúc, dữ liệu, trainer, registry mô hình DL
│   ├── training/                   #   Lưới M x N, tinh chỉnh, cross-validation, ablation, mô hình cuối
│   ├── evaluation/                 #   Metric ABSA, phân tích lỗi
│   ├── visualization/              #   Toàn bộ hàm vẽ (EDA + kết quả)
│   ├── inference/                  #   ABSAPipeline: đóng gói để dự đoán và triển khai
│   └── utils/                      #   I/O, logging, seed, timer
├── resources/lexicons/             # Từ điển (teencode, emoji, stopword, từ bảo vệ, phủ định)
├── app/streamlit_app.py            # Demo web
├── scripts/make_colab_zip.py       # Đóng gói project thành zip sạch để đưa lên Google Drive
├── requirements.txt                # Cài đặt đầy đủ trên máy cá nhân
├── requirements-colab.txt          # Chỉ các gói Colab còn thiếu (notebook tự cài)
├── tests/                          # Kiểm thử đơn vị (pytest)
├── docs/
│   ├── mlAssignments_v1.1_english.pdf
│   ├── preprocessing.md            # Thiết kế và động cơ từng bước tiền xử lý
│   └── feature_representations.md  # Tóm tắt ý tưởng các cách biểu diễn
├── refs/                           # Code tham khảo từ các project cũ (không dùng trực tiếp)
├── UIT-ViSFD/                      # Dữ liệu (Train.csv, Dev.csv; Test.csv khi được công bố)
└── outputs/                        # Sinh ra khi chạy: cache/, reports/, figures/, models/
```

## Quy trình (ML lifecycle)

Notebook đi theo các bước chuẩn của một dự án ML (tương tự CRISP-DM): định nghĩa bài toán, hiểu dữ liệu, chuẩn bị dữ liệu, mô hình hóa, đánh giá, triển khai.

| # | Giai đoạn | Đầu ra chính |
|---|---|---|
| 0 | Thiết lập | Tải zip về Colab (nếu cần), cấu hình chạy (`FEATURES`, `MODELS`, `STRATEGIES`), seed |
| 1 | Bài toán & dữ liệu | Định nghĩa nhãn, chính sách train/dev/test |
| 2 | EDA (train) | Phân bố nhãn, mất cân bằng, đồng xuất hiện, xu hướng viết, kiểm tra dev ở mức tổng hợp |
| 3 | Tiền xử lý | Minh họa từng bước, tác động lên từ vựng, **ablation** |
| 4 | Biểu diễn | Danh mục N biểu diễn |
| 5 | Mô hình ML | Baseline, lưới M x N x {multilabel, joint, two_stage}, heatmap, thống kê theo chiến lược/biểu diễn/mô hình, tinh chỉnh top-k |
| 6 | Mô hình DL | TextCNN / BiLSTM / fine-tune Transformer (bật khi có GPU) |
| 7 | Chọn mô hình | Bảng xếp hạng chung ML + DL trên dev |
| 8 | Đánh giá chi tiết | Theo khía cạnh, confusion matrix, từng cặp nhãn |
| 9 | Phân tích lỗi | Trên dự đoán out-of-fold của train |
| 10 | Mô hình cuối & test | Huấn luyện lại trên train+dev, đánh giá test một lần |
| 11 | Triển khai | `ABSAPipeline` (.joblib), tổng hợp theo sản phẩm, Streamlit |

### Chính sách dữ liệu (chống rò rỉ thông tin)

| Tập | Vai trò | Quy tắc |
|---|---|---|
| Train | Học | EDA đầy đủ; thiết kế tiền xử lý; fit vectorizer và mô hình; cross-validation; phân tích lỗi (out-of-fold) |
| Dev | Validation | Chọn mô hình, tinh chỉnh, early stopping, ablation. Chỉ code đọc để tính metric; không đọc văn bản để đặt luật. EDA chỉ ở mức tỉ lệ nhãn tổng hợp |
| Test | Hidden test | Chỉ đánh giá một lần với cấu hình đã chốt; không chỉnh sửa gì sau khi thấy điểm |

Vectorizer và embedding nằm *trong* pipeline nên chỉ học từ dữ liệu huấn luyện của từng lần fit (kể cả trong từng fold).

## Bài toán và đánh giá

- **Nhãn**: ma trận `Y (n, 11)`, giá trị `0 = không nhắc, 1 = Negative, 2 = Neutral, 3 = Positive` (OTHERS: 0/1).
- **Loại bài toán**: về bản chất là multi-output multiclass; cách đơn giản nhất là trải thành **31 nhãn nhị phân** (10 khía cạnh × 3 cảm xúc + `OTHERS`) và dùng multilabel classification.
- **Chiến lược** (`ABSAClassifier(strategy=...)`):
  - `multilabel` (**mặc định**): 31 bộ phân loại nhị phân độc lập (Binary Relevance). Một khía cạnh bị bật nhiều cảm xúc thì chọn cảm xúc có điểm tin cậy cao nhất. `predict_pairs` trả về 31 bit thô.
  - `joint`: mỗi khía cạnh một bộ phân loại 4 lớp.
  - `two_stage`: phát hiện khía cạnh rồi phân cực trên các bình luận có nhắc khía cạnh đó.
- **Metric** (P/R/F1 cho từng nhiệm vụ, cả micro lẫn macro):
  - `aspect_*` – phát hiện khía cạnh (11 nhãn).
  - `pair_*` – khía cạnh + cảm xúc (31 cặp). Đây là metric chính: `pair_f1_macro`.
  - `polarity_acc_on_detected` – độ chính xác cảm xúc trên các khía cạnh phát hiện đúng.

## Cài đặt và chạy

```bash
pip install -r requirements.txt
pip install -e .                 # cài gói absa ở chế độ phát triển (khuyến nghị)
python -m pytest                 # kiểm thử
jupyter lab notebooks/absa_pipeline.ipynb
streamlit run app/streamlit_app.py   # sau khi notebook đã lưu outputs/models/best_ml_pipeline.joblib
```

- **Cấu hình chạy** (ô cấu hình ở đầu notebook, chỉ có một): 6 biểu diễn (`bow`, `tfidf_word`, `tfidf_word_char`, `word2vec`, `fasttext`, `doc2vec`) × 3 mô hình cơ bản (Logistic Regression, LinearSVM, kNN) × 3 chiến lược = 54 cấu hình. Muốn đổi thì sửa ba danh sách `FEATURES`, `MODELS`, `STRATEGIES`. Registry vẫn có đủ 8 biến thể sparse.
- **Mô hình ML cơ bản** (`CORE_MODELS` trong `models/ml/registry.py`): không có cây/ensemble (Random Forest, boosting, XGBoost, AdaBoost) hay mạng nơ-ron.
- **Thời gian**: chưa đo trọn vẹn với bộ mặc định (ablation, tinh chỉnh và việc học embedding là phần tốn nhất). Colab miễn phí (2 CPU) chậm hơn máy nhiều nhân khoảng 4–6 lần (ước tính); đặt `RUN_ABLATION = False`, `RUN_TUNING = False` để nhanh hơn.
- Kết quả lưới được ghi từng dòng vào `outputs/reports/ml_grid_<mode>.csv`. Chạy lại sẽ tiếp tục từ chỗ dừng.
- Embedding contextual được cache theo câu trong `outputs/cache/embeddings`.
- Nhánh DL: đặt `RUN_DL = True` và thêm `phobert_base_v2`, `visobert` vào `DL_MODELS` khi chạy trên GPU (Colab/Kaggle).

## Chạy trên Google Colab

1. Trên máy: `python scripts/make_colab_zip.py` tạo `dist/absa-visfd.zip` (một thư mục gốc `absa-visfd/`, đã bỏ `outputs/`, `refs/`, cache và `Test.csv`). Dùng `--name <tên project>` để đổi tên thư mục gốc.
2. Upload zip lên Google Drive, bật chia sẻ **Anyone with the link**, copy link (hoặc file-id).
3. Mở `notebooks/absa_pipeline.ipynb` trên Colab (upload riêng file notebook, hoặc mở từ Drive).
4. Dán link/file-id vào biến `GDRIVE_ZIP` ở ô "Thiết lập (Colab / máy cá nhân)", rồi **Runtime → Run all**. Ô đó tự `gdown` zip về `/content`, giải nén, cài các gói còn thiếu (`requirements-colab.txt`) và chuyển vào thư mục project.
5. Cuối notebook (mục 13) gói `outputs/` thành `absa_outputs.zip` và tải về máy. `/content` là bộ nhớ tạm nên hãy tải trước khi runtime bị ngắt.

## Mở rộng

- **Thêm biểu diễn**: thêm một `FeatureSpec` vào `src/absa/features/registry.py`.
- **Thêm mô hình ML**: thêm một `ModelSpec` vào `src/absa/models/ml/registry.py`.
- **Thêm mô hình DL**: thêm một `DLModelSpec` vào `src/absa/models/dl/registry.py`.
- **Sửa từ điển**: chỉnh trực tiếp file trong `resources/lexicons/`; cache tiền xử lý tự động bị vô hiệu.

Lưới thí nghiệm tự nhận các mục mới.

## Quy ước code

PEP 8 (độ dài dòng 99, cấu hình `ruff`/`black` trong `pyproject.toml`), type hints, docstring tiếng Việt kiểu Google. Notebook chỉ gọi hàm, không định nghĩa logic.
