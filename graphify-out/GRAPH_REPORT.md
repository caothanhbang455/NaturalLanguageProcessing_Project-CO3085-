# Graph Report - Project - Copy  (2026-10-14)

## Corpus Check
- 63 files · ~93,125 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 24 file(s) not represented in the graph (top: .ipynb 10, .csv 8, (none) 3)

## Summary
- 932 nodes · 1935 edges · 62 communities (40 shown, 22 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 64 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- io.py
- pandas
- ABSAPipeline
- TextPreprocessor
- test_preprocessing.py
- data.py
- nonstandard_token_report
- attach_negation
- collapse_repeated_tags
- get_w2v_model
- TextDatasetW2V
- CLAUDE.md
- .claude/CLAUDE.md
- WordSegmenter
- analysis.py
- normalizers.py
- ExperimentRunner
- architectures.py
- matplotlib_pyplot
- 3. Họ contextual embedding (Transformer pretrained, đóng băng)
- metrics.py
- TextDataset
- PreprocessConfig
- build_dl_trainer
- ml/registry.py
- ABSATrainer
- error_analysis.py
- ABSAClassifier
- data/__init__.py
- features/registry.py
- features/__init__.py
- sparse.py
- hash_text
- experiment.py
- test_labels.py
- Paths
- dl/__init__.py
- trainer.py
- ABSA cho đánh giá điện thoại tiếng Việt (UIT-ViSFD)
- PretrainedFastTextVectorizer
- cross_validation.py
- ablation.py
- Thiết kế tiền xử lý
- numpy
- builders.py
- TeencodeNormalizer
- test_multilabel.py
- pipeline.py
- absa-visfd
- lexicons.py
- static_embeddings.py
- TextRNN_W2V
- TextBiLSTM_W2V
- make_colab_zip.py
- summarize_by
- train_model
- Vocab
- preprocess_text

## God Nodes (most connected - your core abstractions)
1. `TextPreprocessor` - 28 edges
2. `ABSAClassifier` - 23 edges
3. `evaluate_absa()` - 22 edges
4. `ExperimentRunner` - 22 edges
5. `Timer` - 19 edges
6. `preprocess_corpus()` - 17 edges
7. `to_aspect_matrix()` - 16 edges
8. `ABSAPipeline` - 16 edges
9. `PreprocessConfig` - 16 edges
10. `build_ml_pipeline()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `Mở rộng` --references--> `FeatureSpec`  [INFERRED]
  README.md → src/absa/features/registry.py
- `Quy trình (ML lifecycle)` --references--> `ABSAPipeline`  [INFERRED]
  README.md → src/absa/inference/predictor.py
- `Mở rộng` --references--> `ModelSpec`  [INFERRED]
  README.md → src/absa/models/ml/registry.py
- `Các bước` --references--> `text_pattern_report()`  [INFERRED]
  docs/preprocessing.md → src/absa/eda/analysis.py
- `Các bước` --references--> `nonstandard_token_report()`  [INFERRED]
  docs/preprocessing.md → src/absa/eda/analysis.py

## Import Cycles
- None detected.

## Communities (62 total, 22 thin omitted)

### Community 1 - "io.py"
Cohesion: 0.18
Nodes (7): preprocess_corpus(), ensure_dir(), hash_texts(), load_joblib(), load_json(), save_joblib(), save_json()

### Community 2 - "pandas"
Cohesion: 0.19
Nodes (3): load_pipeline(), main(), dl_catalog()

### Community 3 - "ABSAPipeline"
Cohesion: 0.11
Nodes (3): decode_labels(), ABSAPipeline, summarize_predictions()

### Community 4 - "TextPreprocessor"
Cohesion: 0.17
Nodes (3): TextPreprocessor, test_classic_seg_profile_segments_words(), test_transformer_raw_keeps_numbers_and_k_unit()

### Community 5 - "test_preprocessing.py"
Cohesion: 0.18
Nodes (6): test_classic_negation_does_not_cross_lines(), test_classic_profile_end_to_end(), test_normalize_numbers(), test_reduce_elongation(), test_restore_final_ng(), test_trace_lists_every_step()

### Community 7 - "nonstandard_token_report"
Cohesion: 0.22
Nodes (7): Các bước, Vì sao `classic` không tách từ? (bằng chứng ablation), is_vietnamese_syllable(), nonstandard_token_rate(), nonstandard_token_report(), text_pattern_report(), _words()

### Community 8 - "attach_negation"
Cohesion: 0.20
Nodes (6): attach_negation(), is_punctuation_token(), remove_stopwords(), test_attach_negation_with_filler(), test_double_negation_is_not_fused(), test_negation_stops_at_sentence_boundary()

### Community 9 - "collapse_repeated_tags"
Cohesion: 0.24
Nodes (6): collapse_repeated_tags(), replace_emojis(), _repl(), replace_emoticons(), replace_laughter(), _sentiment_token()

### Community 10 - "get_w2v_model"
Cohesion: 0.22
Nodes (3): get_w2v_model(), TextCNN_W2V, TextLSTM_W2V

### Community 11 - "TextDatasetW2V"
Cohesion: 0.29
Nodes (3): collate_fn(), prepare_w2v_dataloaders(), TextDatasetW2V

### Community 15 - "analysis.py"
Cohesion: 0.12
Nodes (10): aspect_cooccurrence(), aspect_polarity_counts(), dataset_overview(), label_distribution_shift(), labels_per_comment(), mixed_sentiment_rate(), preprocessing_impact(), star_polarity_crosstab() (+2 more)

### Community 16 - "normalizers.py"
Cohesion: 0.08
Nodes (13): emoji_frequency(), emoji_category(), mark_line_breaks(), normalize_numbers(), normalize_punctuation(), normalize_unicode(), normalize_whitespace(), reduce_elongation() (+5 more)

### Community 17 - "ExperimentRunner"
Cohesion: 0.11
Nodes (3): get_feature_spec(), ExperimentRunner, hash_object()

### Community 18 - "architectures.py"
Cohesion: 0.09
Nodes (7): AspectAttentionPooling, AspectHeads, BiLSTMAttentionABSA, _embedding(), pool_sequence(), TextCNNABSA, TransformerABSA

### Community 19 - "matplotlib_pyplot"
Cohesion: 0.07
Nodes (24): plot_aspect_polarity(), plot_cooccurrence(), plot_labels_per_comment(), plot_length_distribution(), plot_polarity_share(), plot_preprocessing_impact(), plot_split_shift(), plot_star_polarity() (+16 more)

### Community 20 - "3. Họ contextual embedding (Transformer pretrained, đóng băng)"
Cohesion: 0.08
Nodes (24): 1.1 Bag-of-Words (`bow`), 1.2 TF-IDF mức từ (`tfidf_word`), 1.3 TF-IDF n-gram ký tự (`tfidf_char`), 1.4 TF-IDF từ + ký tự (`tfidf_word_char`), 1.5 Bốn biến thể sparse để đo từng yếu tố, 1. Họ sparse (túi từ), 2.1 Word2Vec (`word2vec`), 2.2 FastText (`fasttext`) (+16 more)

### Community 21 - "metrics.py"
Cohesion: 0.16
Nodes (10): pair_label_names(), all_confusion_matrices(), aspect_confusion_matrix(), evaluate_absa(), evaluate_pair_bits(), pair_report(), pair_report_from_bits(), per_aspect_report() (+2 more)

### Community 23 - "PreprocessConfig"
Cohesion: 0.10
Nodes (4): PreparedCorpus, PreprocessConfig, DLRunResult, run_dl_experiment()

### Community 24 - "build_dl_trainer"
Cohesion: 0.12
Nodes (4): build_embedding_matrix(), VocabEncoder, Vocabulary, build_dl_trainer()

### Community 25 - "ml/registry.py"
Cohesion: 0.11
Nodes (5): check_compatibility(), list_models(), model_catalog(), ModelSpec, test_compatibility_rules()

### Community 27 - "error_analysis.py"
Cohesion: 0.18
Nodes (7): decode_row(), format_label_string(), build_error_table(), error_type_summary(), polarity_confusions(), sample_errors(), test_encode_decode_roundtrip()

### Community 28 - "ABSAClassifier"
Cohesion: 0.12
Nodes (3): ABSAClassifier, _binary_scores(), _fit_one()

### Community 29 - "data/__init__.py"
Cohesion: 0.15
Nodes (5): encode_labels(), ABSASplit, load_split(), load_splits(), validate_frame()

### Community 30 - "features/registry.py"
Cohesion: 0.15
Nodes (4): _resolve_device(), TransformerEmbeddingVectorizer, _hf(), _st()

### Community 31 - "features/__init__.py"
Cohesion: 0.18
Nodes (3): feature_catalog(), FeatureSpec, list_features()

### Community 32 - "sparse.py"
Cohesion: 0.16
Nodes (4): make_bow(), make_tfidf_char(), make_tfidf_word(), make_tfidf_word_char()

### Community 35 - "experiment.py"
Cohesion: 0.09
Nodes (5): resolve_config(), FittedFeature, params_key(), fit_final_pipeline(), Timer

### Community 36 - "test_labels.py"
Cohesion: 0.38
Nodes (4): parse_label_string(), test_parse_empty_and_nan(), test_parse_label_string(), test_parse_rejects_unknown_aspect()

### Community 39 - "trainer.py"
Cohesion: 0.16
Nodes (3): TrainConfig, get_logger(), set_seed()

### Community 40 - "ABSA cho đánh giá điện thoại tiếng Việt (UIT-ViSFD)"
Cohesion: 0.17
Nodes (10): ABSA cho đánh giá điện thoại tiếng Việt (UIT-ViSFD), Bài toán và đánh giá, Chính sách dữ liệu (chống rò rỉ thông tin), Chạy trên Google Colab, Cài đặt và chạy, Cấu trúc thư mục, Mở rộng, Quy trình (ML lifecycle) (+2 more)

### Community 43 - "ablation.py"
Cohesion: 0.13
Nodes (7): get_model_spec(), cumulative_variants(), leave_one_out_variants(), run_preprocessing_ablation(), candidate_params(), tune_model(), tune_top_k()

### Community 44 - "Thiết kế tiền xử lý"
Cohesion: 0.29
Nodes (6): Hạn chế đã biết, Lịch sử thay đổi logic (`PREPROCESS_VERSION`), Nguyên tắc, Profile cho từng họ biểu diễn, Thay đổi so với tài nguyên trong `refs/`, Thiết kế tiền xử lý

### Community 45 - "numpy"
Cohesion: 0.20
Nodes (9): count_pair_conflicts(), label_frame(), pairs_to_labels(), to_aspect_matrix(), to_pair_matrix(), test_aspect_and_pair_matrices(), test_conflict_resolution_uses_scores(), test_no_bit_means_aspect_absent() (+1 more)

### Community 46 - "builders.py"
Cohesion: 0.22
Nodes (3): build_feature(), build_estimator(), build_ml_pipeline()

### Community 48 - "test_multilabel.py"
Cohesion: 0.13
Nodes (8): test_classifier_rejects_wrong_label_shape(), test_perfect_prediction_scores_one(), test_wrong_polarity_hurts_pairs_not_aspects(), toy_data(), test_multilabel_classifier_learns_toy_data(), test_multilabel_is_a_registered_strategy_and_default(), test_predict_pairs_only_for_multilabel(), toy()

### Community 53 - "lexicons.py"
Cohesion: 0.22
Nodes (5): _iter_lines(), Lexicons, load_lexicons(), load_mapping(), load_word_list()

### Community 54 - "static_embeddings.py"
Cohesion: 0.20
Nodes (3): _default_workers(), GensimWordEmbeddingVectorizer, _tokenize()

### Community 57 - "make_colab_zip.py"
Cohesion: 0.31
Nodes (3): build_zip(), main(), should_skip()

### Community 58 - "summarize_by"
Cohesion: 0.40
Nodes (3): _grid_rows(), summarize_by(), test_summarize_by_uses_only_complete_pairs()

### Community 59 - "train_model"
Cohesion: 0.67
Nodes (3): train_model(), train_one_epoch(), validate()

## Knowledge Gaps
- **1 isolated node(s):** `absa-visfd`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 465 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TextPreprocessor` connect `TextPreprocessor` to `io.py`, `._build_steps`, `ABSAPipeline`, `experiment.py`, `test_preprocessing.py`, `WordSegmenter`, `pipeline.py`, `lexicons.py`, `PreprocessConfig`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `TextPreprocessor` (e.g. with `ABSAPipeline` and `Lexicons`) actually correct?**
  _`TextPreprocessor` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `absa-visfd` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `text_classification.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05555555555555555 - nodes in this community are weakly interconnected._
- **Why does `Timer` connect `experiment.py` to `io.py`, `trainer.py`, `ablation.py`, `pipeline.py`, `ExperimentRunner`, `PreprocessConfig`, `ABSATrainer`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `ABSAClassifier` (e.g. with `ExperimentRunner` and `test_absa_classifier_learns_toy_data()`) actually correct?**
  _`ABSAClassifier` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Should `ABSAPipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.11333333333333333 - nodes in this community are weakly interconnected._