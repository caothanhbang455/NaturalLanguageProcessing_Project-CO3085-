# Graph Report - Project - Copy  (2026-10-11)

## Corpus Check
- 69 files · ~100,559 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 868 nodes · 1525 edges · 53 communities (44 shown, 9 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 41 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- text_classification.py
- experiment.py
- What You Must Do When Invoked
- result_plots.py
- graphify reference: extra exports and benchmark
- TextPreprocessor
- graphify reference: query, path, explain
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- CLAUDE.md
- .claude/CLAUDE.md
- extraction-spec.md
- analysis.py
- normalizers.py
- ExperimentRunner
- architectures.py
- eda_plots.py
- 3. Họ contextual embedding (Transformer pretrained, đóng băng)
- evaluate_absa
- GensimWordEmbeddingVectorizer
- dl_experiment.py
- Vocabulary
- ml/registry.py
- ABSATrainer
- error_analysis.py
- ABSAClassifier
- data/__init__.py
- features/registry.py
- features/__init__.py
- sparse.py
- ablation.py
- EmbeddingCache
- Timer
- config.py
- Paths
- dl/__init__.py
- trainer.py
- ABSA cho đánh giá điện thoại tiếng Việt (UIT-ViSFD)
- PretrainedFastTextVectorizer
- training/__init__.py
- tuning.py
- Thiết kế tiền xử lý
- decode_labels
- build_ml_pipeline
- VocabEncoder
- test_metrics_and_models.py
- .stars
- absa/__init__.py
- models/__init__.py
- absa-visfd

## God Nodes (most connected - your core abstractions)
1. `TextPreprocessor` - 27 edges
2. `ExperimentRunner` - 21 edges
3. `evaluate_absa()` - 20 edges
4. `Timer` - 19 edges
5. `ABSAClassifier` - 17 edges
6. `PreprocessConfig` - 16 edges
7. `build_ml_pipeline()` - 16 edges
8. `ensure_dir()` - 16 edges
9. `to_aspect_matrix()` - 15 edges
10. `TransformerEmbeddingVectorizer` - 15 edges

## Surprising Connections (you probably didn't know these)
- `test_classifier_rejects_wrong_label_shape()` --calls--> `ABSAClassifier`  [INFERRED]
  tests/test_metrics_and_models.py → src/absa/models/ml/absa_classifier.py
- `test_perfect_prediction_scores_one()` --calls--> `evaluate_absa()`  [INFERRED]
  tests/test_metrics_and_models.py → src/absa/evaluation/metrics.py
- `test_wrong_polarity_hurts_pairs_not_aspects()` --calls--> `evaluate_absa()`  [INFERRED]
  tests/test_metrics_and_models.py → src/absa/evaluation/metrics.py
- `test_compatibility_rules()` --calls--> `get_feature_spec()`  [INFERRED]
  tests/test_metrics_and_models.py → src/absa/features/registry.py
- `test_absa_classifier_learns_toy_data()` --calls--> `ABSAClassifier`  [INFERRED]
  tests/test_metrics_and_models.py → src/absa/models/ml/absa_classifier.py

## Import Cycles
- None detected.

## Communities (53 total, 9 thin omitted)

### Community 0 - "text_classification.py"
Cohesion: 0.05
Nodes (22): DataLoader, collate_fn(), get_w2v_model(), prepare_dataloaders(), prepare_w2v_dataloaders(), preprocess_text(), Dataset, Mô hình TextCNN sử dụng Word2Vec Embedding cho phân loại văn bản. Luồng dữ liệu… (+14 more)

### Community 1 - "experiment.py"
Cohesion: 0.06
Nodes (53): Any, Tiền xử lý văn bản tiếng Việt theo luật tổng quát., _iter_lines(), load_lexicons(), load_mapping(), load_word_list(), Path, Nạp các từ điển (lexicon) phục vụ tiền xử lý. Tất cả từ điển nằm ở… (+45 more)

### Community 2 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 3 - "result_plots.py"
Cohesion: 0.06
Nodes (42): load_pipeline(), main(), Demo Streamlit: phân tích đánh giá sản phẩm theo khía cạnh. Chạy: streamlit run…, Nạp pipeline một lần cho mọi phiên làm việc., cache_resource, Đóng gói mô hình để dự đoán và triển khai., ABSAPipeline, DataFrame (+34 more)

### Community 4 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 5 - "TextPreprocessor"
Cohesion: 0.05
Nodes (30): Collection, BaseEstimator, DataFrame, TransformerMixin, Pipeline tiền xử lý tiếng Việt, tương thích scikit-learn. Bộ tiền xử lý **không…, Danh sách ``(tên_bước, hàm)`` theo đúng thứ tự áp dụng (tạo lười biếng)., Không học gì; có mặt để tương thích ``Pipeline`` của scikit-learn., Tiền xử lý một bình luận. (+22 more)

### Community 6 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 7 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 8 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 9 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 15 - "analysis.py"
Cohesion: 0.09
Nodes (42): Ma trận nhị phân (n, 11): khía cạnh có được nhắc tới hay không., to_aspect_matrix(), aspect_cooccurrence(), aspect_polarity_counts(), dataset_overview(), emoji_frequency(), is_vietnamese_syllable(), label_distribution_shift() (+34 more)

### Community 16 - "normalizers.py"
Cohesion: 0.06
Nodes (34): collapse_repeated_tags(), mark_line_breaks(), normalize_numbers(), normalize_punctuation(), normalize_unicode(), normalize_whitespace(), Các bước chuẩn hóa văn bản ở mức ký tự/chuỗi. Mỗi hàm ở đây là **một luật tổng…, Gộp nhãn cảm xúc lặp liên tiếp (``emopos emopos`` -> ``emopos``). (+26 more)

### Community 17 - "ExperimentRunner"
Cohesion: 0.09
Nodes (17): ExperimentRunner, FittedFeature, params_key(), DataFrame, ndarray, Path, Fit biểu diễn ``name`` trên tập huấn luyện (có cache trong bộ nhớ)., Giải phóng bộ nhớ đặc trưng (trừ các biểu diễn trong ``keep``). (+9 more)

### Community 18 - "architectures.py"
Cohesion: 0.12
Nodes (17): AspectAttentionPooling, AspectHeads, BiLSTMAttentionABSA, _embedding(), pool_sequence(), Tensor, Kiến trúc học sâu cho ABSA đa khía cạnh. Mọi kiến trúc gồm hai phần: *…, BiLSTM 2 chiều + aspect attention pooling. BiLSTM đọc câu theo cả hai chiều nên… (+9 more)

### Community 19 - "eda_plots.py"
Cohesion: 0.13
Nodes (26): plot_aspect_polarity(), plot_cooccurrence(), plot_labels_per_comment(), plot_length_distribution(), plot_polarity_share(), plot_preprocessing_impact(), plot_split_shift(), plot_star_polarity() (+18 more)

### Community 20 - "3. Họ contextual embedding (Transformer pretrained, đóng băng)"
Cohesion: 0.08
Nodes (23): 1.1 Bag-of-Words (`bow`), 1.2 TF-IDF mức từ (`tfidf_word`), 1.3 TF-IDF n-gram ký tự (`tfidf_char`), 1.4 TF-IDF từ + ký tự (`tfidf_word_char`), 1. Họ sparse (túi từ), 2.1 Word2Vec (`word2vec`), 2.2 FastText (`fasttext`), 2.3 Doc2Vec (`doc2vec`) (+15 more)

### Community 21 - "evaluate_absa"
Cohesion: 0.17
Nodes (21): pair_label_names(), Tên 31 cột của pair matrix: ``ASPECT#Polarity`` cho 10 khía cạnh + ``OTHERS``., Ma trận nhị phân (n, 31) cho từng cặp (khía cạnh, cảm xúc). Một dự đoán chỉ…, to_pair_matrix(), Metric ABSA và phân tích lỗi., all_confusion_matrices(), aspect_confusion_matrix(), evaluate_absa() (+13 more)

### Community 22 - "GensimWordEmbeddingVectorizer"
Cohesion: 0.13
Nodes (13): _default_workers(), Doc2VecVectorizer, GensimWordEmbeddingVectorizer, BaseEstimator, ndarray, Embedding tĩnh (static): Word2Vec, FastText, Doc2Vec huấn luyện trên tập train.…, Vector câu cho từng văn bản, kích thước ``(n, vector_size)`` (hoặc gấp đôi)., Doc2Vec PV-DBOW (Le & Mikolov, 2014): học trực tiếp vector cho cả văn bản. Khi… (+5 more)

### Community 23 - "dl_experiment.py"
Cohesion: 0.11
Nodes (16): Trả về list logits theo khía cạnh., TextCNN (Kim, 2014): tích chập 1D nhiều kích thước cửa sổ + max-pooling theo…, TextCNNABSA, HFTokenEncoder, Mã hóa văn bản bằng tokenizer Hugging Face (có token đặc biệt, cắt độ dài)., Mã hóa một văn bản thành dãy chỉ số token., Tạo bản sao với một số trường thay đổi (cấu hình là bất biến)., build_dl_trainer() (+8 more)

### Community 24 - "Vocabulary"
Cohesion: 0.11
Nodes (13): ABSADataset, build_embedding_matrix(), make_collate(), Dataset, ndarray, Dữ liệu cho mô hình học sâu: từ vựng, Dataset và hàm gộp batch. Hai loại đầu…, Tạo hàm gộp batch: đệm về cùng độ dài và sinh ``attention_mask``., Từ điển token -> chỉ số, học từ văn bản train. Args: min_freq: Token xuất hiện… (+5 more)

### Community 25 - "ml/registry.py"
Cohesion: 0.14
Nodes (15): Mô hình học máy truyền thống cho ABSA., check_compatibility(), get_model_spec(), list_models(), model_catalog(), ModelSpec, DataFrame, Danh mục M mô hình học máy truyền thống (model registry). Mỗi mô hình được mô… (+7 more)

### Community 26 - "ABSATrainer"
Cohesion: 0.16
Nodes (12): no_grad, Optimizer, ABSATrainer, compute_class_weights(), DataFrame, ndarray, Tensor, Huấn luyện với early stopping theo ``config.metric`` trên dev. Returns: Lịch sử… (+4 more)

### Community 27 - "error_analysis.py"
Cohesion: 0.16
Nodes (17): decode_row(), format_label_string(), Định dạng ngược về chuỗi nhãn gốc (thứ tự khía cạnh theo ``ALL_ASPECTS``)., Chuyển một hàng của ``Y`` về dict ``{khía_cạnh: cảm_xúc}`` (bỏ khía cạnh vắng…, build_error_table(), error_type_summary(), polarity_confusions(), DataFrame (+9 more)

### Community 28 - "ABSAClassifier"
Cohesion: 0.15
Nodes (11): ClassifierMixin, ABSAClassifier, _fit_one(), BaseEstimator, ndarray, Bộ phân loại ABSA bọc quanh một estimator scikit-learn bất kỳ. Bài toán ABSA…, Dự đoán ma trận nhãn ``(n, 11)``., Macro-F1 trên cặp (khía cạnh, cảm xúc) – metric chính của project. (+3 more)

### Community 29 - "data/__init__.py"
Cohesion: 0.19
Nodes (14): Đọc dữ liệu và mã hóa nhãn ABSA., encode_labels(), Chuyển danh sách chuỗi nhãn thành ma trận ``Y`` (n, 11) kiểu ``int8``., ABSASplit, load_split(), load_splits(), DataFrame, Path (+6 more)

### Community 30 - "features/registry.py"
Cohesion: 0.15
Nodes (10): BaseEstimator, TransformerMixin, Embedding ngữ cảnh (contextual) từ mô hình Transformer pretrained, dùng ở dạng…, Không học gì (mô hình đóng băng)., Trích vector câu từ mô hình Transformer đóng băng. Args: model_name: Tên mô…, _resolve_device(), TransformerEmbeddingVectorizer, _hf() (+2 more)

### Community 31 - "features/__init__.py"
Cohesion: 0.14
Nodes (14): Các cách biểu diễn văn bản: sparse, static embedding, contextual embedding,…, build_feature(), feature_catalog(), FeatureSpec, get_feature_spec(), list_features(), DataFrame, TransformerMixin (+6 more)

### Community 32 - "sparse.py"
Cohesion: 0.16
Nodes (14): CountVectorizer, FeatureUnion, fixture, make_bow(), make_tfidf_char(), make_tfidf_word(), make_tfidf_word_char(), r"""Biểu diễn thưa (sparse): Bag-of-Words và TF-IDF ở mức từ và ký tự. Đầu vào… (+6 more)

### Community 33 - "ablation.py"
Cohesion: 0.14
Nodes (13): Logger, cumulative_variants(), leave_one_out_variants(), DataFrame, ndarray, Ablation tiền xử lý: đo đóng góp của từng bước bằng số liệu, không bằng cảm…, Thêm dần từng bước từ cấu hình tối thiểu tới ``classic``, rồi thử hai bước tùy…, ``classic`` và các bản *đảo* đúng một bước (bỏ bước đang bật / thêm bước đang… (+5 more)

### Community 34 - "EmbeddingCache"
Cohesion: 0.16
Nodes (8): EmbeddingCache, ndarray, Path, Trả về ma trận embedding ``(n, d)``; chỉ mã hóa những câu chưa có trong cache., Cache embedding theo câu: ``{md5(câu): vector}`` lưu ở một file ``.npz``. Args:…, Các khóa chưa có trong cache., Lấy ma trận vector theo đúng thứ tự ``keys``., Ghi cache ra đĩa (nếu có ``path``).

### Community 35 - "Timer"
Cohesion: 0.16
Nodes (9): BaseException, fit_final_pipeline(), ndarray, Huấn luyện mô hình cuối cùng và đánh giá một lần trên tập test ẩn. Quy trình…, Huấn luyện cấu hình đã chọn trên ``fit_splits`` và đóng gói để triển khai.…, Đo thời gian chạy bằng context manager., Đồng hồ bấm giờ dùng với ``with``. Example: >>> with Timer() as t: ... _ =…, Timer (+1 more)

### Community 36 - "config.py"
Cohesion: 0.20
Nodes (10): Cấu hình trung tâm: đường dẫn, hằng số nhãn và các metric dùng chung toàn…, parse_label_string(), Mã hóa / giải mã nhãn ABSA của UIT-ViSFD. Định dạng gốc trong cột ``label``::…, Tách chuỗi nhãn gốc thành ``{khía_cạnh: cảm_xúc}``. Args: label: Chuỗi dạng…, Pipeline suy luận hoàn chỉnh: bình luận thô -> nhãn khía cạnh -> tổng hợp theo…, Kiểm thử mã hóa/giải mã nhãn ABSA., test_aspect_and_pair_matrices(), test_parse_empty_and_nan() (+2 more)

### Community 37 - "Paths"
Cohesion: 0.21
Nodes (8): Paths, Path, Tập hợp các thư mục của project. Attributes: root: Thư mục gốc của project.…, Thư mục chứa các từ điển dùng cho tiền xử lý., Cache văn bản đã tiền xử lý và embedding đã tính (xóa được an toàn)., Nơi lưu pipeline/mô hình đã huấn luyện., Nơi lưu bảng kết quả thí nghiệm (CSV/JSON)., Nơi lưu hình vẽ phục vụ báo cáo.

### Community 38 - "dl/__init__.py"
Cohesion: 0.22
Nodes (11): Mô hình học sâu cho ABSA (yêu cầu PyTorch)., dl_catalog(), DLModelSpec, get_dl_spec(), list_dl_models(), DataFrame, Danh mục mô hình học sâu (DL registry). Giống nhánh ML, nhánh DL cũng là lưới…, Lấy :class:`DLModelSpec` theo tên. (+3 more)

### Community 39 - "trainer.py"
Cohesion: 0.22
Nodes (7): Module, Vòng huấn luyện dùng chung cho mọi kiến trúc học sâu ABSA. Điểm chính: *…, Siêu tham số huấn luyện. Attributes: epochs: Số epoch tối đa. batch_size: Kích…, TrainConfig, Cố định seed để kết quả thí nghiệm tái lập được., Cố định seed cho ``random``, ``numpy`` và ``torch`` (nếu đã cài). Args: seed:…, set_seed()

### Community 40 - "ABSA cho đánh giá điện thoại tiếng Việt (UIT-ViSFD)"
Cohesion: 0.22
Nodes (8): ABSA cho đánh giá điện thoại tiếng Việt (UIT-ViSFD), Bài toán và đánh giá, Chính sách dữ liệu (chống rò rỉ thông tin), Cài đặt và chạy, Cấu trúc thư mục, Mở rộng, Quy trình (ML lifecycle), Quy ước code

### Community 41 - "PretrainedFastTextVectorizer"
Cohesion: 0.25
Nodes (5): PretrainedFastTextVectorizer, TransformerMixin, FastText pretrained của Facebook cho tiếng Việt (``cc.vi.300.bin``, Grave et…, Nạp vector pretrained (không học gì thêm)., Trung bình vector các âm tiết của từng văn bản.

### Community 42 - "training/__init__.py"
Cohesion: 0.33
Nodes (7): cross_val_predict_absa(), ndarray, Cross-validation trên tập train: dự đoán out-of-fold phục vụ phân tích lỗi. Dự…, Nhãn đại diện để chia fold cân bằng cho dữ liệu đa nhãn. Mỗi bình luận được đại…, Sinh dự đoán out-of-fold cho toàn bộ ``texts``. Args: texts: Văn bản **đã tiền…, stratification_key(), Huấn luyện: lưới thí nghiệm M x N, tinh chỉnh, cross-validation, ablation, mô…

### Community 43 - "tuning.py"
Cohesion: 0.31
Nodes (8): candidate_params(), DataFrame, Tinh chỉnh siêu tham số trên tập validation (dev). Chiến lược hai tầng để tiết…, Sinh danh sách bộ tham số: cả lưới nếu đủ nhỏ, ngược lại lấy mẫu ngẫu nhiên., Dò siêu tham số cho một cặp (biểu diễn, mô hình). Returns: Bảng các bộ tham số…, Tinh chỉnh ``k`` cấu hình tốt nhất của vòng sàng lọc. Returns: Mỗi cấu hình một…, tune_model(), tune_top_k()

### Community 44 - "Thiết kế tiền xử lý"
Cohesion: 0.25
Nodes (7): Các bước, Hạn chế đã biết, Nguyên tắc, Profile cho từng họ biểu diễn, Thay đổi so với tài nguyên trong `refs/`, Thiết kế tiền xử lý, Vì sao `classic` không tách từ? (bằng chứng ablation)

### Community 45 - "decode_labels"
Cohesion: 0.25
Nodes (7): decode_labels(), label_frame(), DataFrame, ndarray, Giải mã toàn bộ ma trận ``Y``., Bảng dễ đọc: mỗi cột là một khía cạnh, giá trị là cảm xúc hoặc ``"-"``., Bảng kết quả: mỗi bình luận một dòng, mỗi khía cạnh một cột.

### Community 46 - "build_ml_pipeline"
Cohesion: 0.29
Nodes (7): build_estimator(), Pipeline, Tạo estimator đơn nhãn đã ghép bộ chuẩn hóa (nếu cần). Luôn trả về…, build_ml_pipeline(), Pipeline, Ghép biểu diễn + mô hình thành pipeline scikit-learn hoàn chỉnh., Tạo pipeline ``văn bản đã tiền xử lý -> đặc trưng -> ABSAClassifier`` (chưa…

### Community 47 - "VocabEncoder"
Cohesion: 0.40
Nodes (3): Mã hóa văn bản bằng :class:`Vocabulary` (dạng lớp để pickle được khi lưu mô…, Mã hóa một văn bản thành dãy chỉ số token., VocabEncoder

### Community 48 - "test_metrics_and_models.py"
Cohesion: 0.50
Nodes (3): Kiểm thử metric ABSA và bộ phân loại bọc ``ABSAClassifier``., test_classifier_rejects_wrong_label_shape(), test_perfect_prediction_scores_one()

## Knowledge Gaps
- **74 isolated node(s):** `absa-visfd`, `graphify`, `Usage`, `What graphify is for`, `Step 0 - GitHub repos and multi-path merge (only if a URL or several paths)` (+69 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 455 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ABSATrainer` connect `ABSATrainer` to `Timer`, `dl/__init__.py`, `trainer.py`, `dl_experiment.py`, `Vocabulary`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Why does `Timer` connect `Timer` to `ablation.py`, `experiment.py`, `trainer.py`, `ExperimentRunner`, `dl_experiment.py`, `ABSATrainer`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Why does `TextPreprocessor` connect `TextPreprocessor` to `experiment.py`, `result_plots.py`, `config.py`, `Timer`, `analysis.py`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `TextPreprocessor` (e.g. with `ABSAPipeline` and `Lexicons`) actually correct?**
  _`TextPreprocessor` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `ExperimentRunner` (e.g. with `ABSAClassifier` and `PreparedCorpus`) actually correct?**
  _`ExperimentRunner` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `evaluate_absa()` (e.g. with `test_absa_classifier_learns_toy_data()` and `test_perfect_prediction_scores_one()`) actually correct?**
  _`evaluate_absa()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Timer` (e.g. with `ABSATrainer` and `ExperimentRunner`) actually correct?**
  _`Timer` has 2 INFERRED edges - model-reasoned connections that need verification._