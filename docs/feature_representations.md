# Các cách biểu diễn văn bản: tóm tắt ý tưởng

Tài liệu đi kèm `src/absa/features/registry.py`. Mỗi mục gồm: ý tưởng cốt lõi, cách dùng trong project, ưu/nhược điểm với bài toán ABSA trên bình luận điện thoại, và công bố gốc.

Sơ đồ chung: **văn bản → vector cố định số chiều → bộ phân loại ML** (LR, SVM, NB, RF, MLP, ...). Mọi biểu diễn chỉ được fit trên train.

| Họ | Biểu diễn | Số chiều | Học từ dữ liệu của ta? | Hiểu ngữ cảnh? |
|---|---|---|---|---|
| Sparse | BoW, TF-IDF từ/ký tự | 10⁴–10⁵ (thưa) | Có (từ vựng, IDF) | Không (chỉ n-gram cục bộ) |
| Static | Word2Vec, FastText, Doc2Vec | 300 | Có (huấn luyện trên train) | Không (1 từ = 1 vector) |
| Contextual | PhoBERT, ViSoBERT, SBERT, E5, GTE, BGE-M3, Qwen3 | 768–1024 | Không (pretrained, đóng băng) | Có |
| Hybrid | TF-IDF + E5 | thưa + 768 | Một phần | Một phần |

---

## 1. Họ sparse (túi từ)

### 1.1 Bag-of-Words (`bow`)
- **Ý tưởng**: mỗi chiều là một n-gram trong từ vựng, giá trị là số lần xuất hiện. Bỏ qua thứ tự từ ngoài phạm vi n-gram.
- **Trong project**: uni + bi-gram trên văn bản đã tách từ (`màn_hình`, `pin trâu`, `không_mượt`), bỏ n-gram xuất hiện ở ít hơn 2 bình luận.
- **Ưu**: đơn giản, nhanh, dễ giải thích (xem được trọng số từng từ). **Nhược**: từ phổ biến lấn át; `đẹp` và `xinh` là hai chiều không liên quan.
- **Tham khảo**: Harris (1954); Manning, Raghavan & Schütze (2008), *Introduction to Information Retrieval*.

### 1.2 TF-IDF mức từ (`tfidf_word`)
- **Ý tưởng**: trọng số = tần suất trong câu × `log(N / số câu chứa từ)`. Từ xuất hiện khắp nơi (`máy`, `mua`) bị giảm trọng số, từ đặc trưng cho khía cạnh (`pin`, `camera`, `nhân_viên`) được nhấn mạnh. Dùng `sublinear_tf` (`1 + log tf`) để từ lặp nhiều lần không lấn át.
- **Ưu**: baseline mạnh nhất trong nhóm cổ điển cho văn bản ngắn. Với ABSA, từ khóa khía cạnh là tín hiệu rất mạnh nên TF-IDF thường đã tốt.
- **Tham khảo**: Sparck Jones (1972); Salton & Buckley (1988).

### 1.3 TF-IDF n-gram ký tự (`tfidf_char`)
- **Ý tưởng**: thay vì từ, dùng các chuỗi 2–5 ký tự bên trong mỗi từ (`char_wb`). `mượtt`, `mượt`, `muot` chia sẻ nhiều n-gram ký tự nên vẫn gần nhau.
- **Ưu**: bền với lỗi gõ, teencode chưa có trong từ điển, thiếu dấu. Đây là "lưới an toàn" cho những biến thể mà luật tiền xử lý chưa bắt được, rất phù hợp với bình luận tự do. **Nhược**: số chiều lớn, khó giải thích hơn.
- **Tham khảo**: Cavnar & Trenkle (1994); Kanaris et al. (2007).

### 1.4 TF-IDF từ + ký tự (`tfidf_word_char`)
- **Ý tưởng**: nối hai ma trận trên. Mức từ cho ngữ nghĩa, mức ký tự cho độ bền chính tả.
- **Tham khảo**: Wang & Manning (2012), *Baselines and Bigrams* – n-gram + mô hình tuyến tính là baseline rất khó vượt cho phân loại cảm xúc.

### 1.5 Bốn biến thể sparse để đo từng yếu tố

Mỗi biến thể chỉ thay **một** yếu tố so với biểu diễn lõi, để biết yếu tố nào đáng giá:

| Biểu diễn | Thay đổi so với lõi | Câu hỏi trả lời |
|---|---|---|
| `bow_binary` | `bow` nhưng chỉ ghi có/không (không đếm số lần) | Số lần lặp có mang thông tin không, hay chỉ cần biết từ có mặt? |
| `tfidf_word_uni` | `tfidf_word` bỏ bigram (chỉ unigram) | Cụm hai từ (`pin trâu`, `không_mượt`) đóng góp bao nhiêu? |
| `tfidf_word_13` | `tfidf_word` thêm trigram | Cụm ba từ (`không tốt lắm`) có giúp không, hay chỉ làm từ vựng thưa thêm? |
| `tfidf_char_cross` | n-gram ký tự 3–6 **vắt qua dấu cách** (`analyzer="char"`) thay vì chỉ trong từ (`char_wb`) | Bắt cụm hai từ ở mức ký tự có tốt hơn n-gram trong từ không? |

---

## 2. Họ static embedding (học trên train)

Ý tưởng chung là **giả thuyết phân bố** (*distributional hypothesis*): từ xuất hiện trong ngữ cảnh giống nhau thì có nghĩa giống nhau. Mỗi từ được ánh xạ vào một vector đặc 300 chiều; vector câu là trung bình (có trọng số IDF) các vector từ.

### 2.1 Word2Vec (`word2vec`)
- **Ý tưởng**: mạng nông học dự đoán từ ngữ cảnh từ từ trung tâm (Skip-gram) hoặc ngược lại (CBOW), với negative sampling. Sau khi học, `trâu` (pin trâu) sẽ gần `khỏe`, `lâu`.
- **Trong project**: Skip-gram 300d, cửa sổ 5, 30 epoch (corpus nhỏ nên cần nhiều vòng). Gộp câu bằng trung bình có trọng số IDF.
- **Nhược**: từ không có trong train sẽ không có vector; corpus 7.800 câu là nhỏ nên vector còn nhiễu.
- **Tham khảo**: Mikolov et al. (2013), *Distributed Representations of Words and Phrases*, NeurIPS.

### 2.2 FastText (`fasttext`)
- **Ý tưởng**: như Word2Vec nhưng vector của một từ = tổng vector các **n-gram ký tự** của nó. Nhờ vậy tạo được vector cho từ chưa gặp (teencode mới, lỗi gõ).
- **Ưu**: hợp với văn bản mạng xã hội tiếng Việt nhiều biến thể.
- **Tham khảo**: Bojanowski et al. (2017), *Enriching Word Vectors with Subword Information*, TACL.

### 2.3 Doc2Vec (`doc2vec`)
- **Ý tưởng**: PV-DBOW học **trực tiếp** một vector cho cả văn bản bằng cách dùng nó để dự đoán các từ trong văn bản. Văn bản mới được suy luận vector bằng vài vòng gradient.
- **Tham khảo**: Le & Mikolov (2014), *Distributed Representations of Sentences and Documents*, ICML.

### 2.4 FastText pretrained `cc.vi.300` (`fasttext_cc_vi`, tùy chọn)
- **Ý tưởng**: FastText huấn luyện sẵn trên Common Crawl + Wikipedia tiếng Việt (hàng tỉ token), mức âm tiết. Cần tải file ~7GB và đặt biến môi trường `ABSA_FASTTEXT_VI_PATH`.
- **Tham khảo**: Grave et al. (2018), *Learning Word Vectors for 157 Languages*, LREC.

---

## 3. Họ contextual embedding (Transformer pretrained, đóng băng)

Ý tưởng chung: Transformer dùng **self-attention** để mỗi token "nhìn" toàn câu, nên vector của `trâu` trong `pin trâu` khác với trong `con trâu`. Mô hình được pretrain trên lượng văn bản rất lớn. Ở đây ta **không fine-tune**: chỉ lấy vector câu (mean pooling hoặc vector câu chuyên dụng) làm đặc trưng cho bộ phân loại ML. Cách này cho phép so sánh công bằng *chất lượng biểu diễn* với TF-IDF trên cùng bộ phân loại, và chạy được trên CPU. Fine-tune đầy đủ nằm ở nhánh DL.

Có hai loại:
- **Encoder thuần** (PhoBERT, ViSoBERT): pretrain bằng Masked Language Modeling, vector câu = trung bình hidden state lớp cuối. Mạnh về ngôn ngữ nhưng vector câu chưa được tối ưu cho độ tương đồng.
- **Sentence embedding** (SBERT, E5, GTE, BGE-M3, Qwen3): fine-tune thêm bằng **contrastive learning** (kéo câu cùng nghĩa lại gần, đẩy câu khác nghĩa ra xa), nên vector câu dùng trực tiếp được.

### 3.1 PhoBERT-base-v2 (`phobert`)
- **Ý tưởng**: RoBERTa huấn luyện trên ~20GB văn bản tiếng Việt **đã tách từ** (v2 thêm dữ liệu). Là mô hình ngôn ngữ đơn ngữ tiếng Việt phổ biến nhất.
- **Trong project**: dùng profile `transformer_seg` (văn bản đã tách từ, giữ dấu câu, emoji → từ), tối đa 256 token.
- **Tham khảo**: Nguyen & Nguyen (2020), *PhoBERT: Pre-trained language models for Vietnamese*, Findings of EMNLP.

### 3.2 ViSoBERT (`visobert`)
- **Ý tưởng**: pretrain trên văn bản **mạng xã hội** tiếng Việt (Facebook, TikTok, YouTube): đúng văn phong có teencode, emoji, viết tắt như bình luận UIT-ViSFD. Tokenizer giữ được emoji.
- **Trong project**: profile `transformer_raw` (không tách từ, giữ emoji).
- **Tham khảo**: Nguyen et al. (2023), *ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing*, EMNLP.

### 3.3 Sentence-BERT đa ngữ (`sbert_multilingual`)
- **Ý tưởng**: kiến trúc siamese tạo vector câu so sánh được bằng cosine. Bản đa ngữ được tạo bằng **knowledge distillation**: mô hình học sinh đa ngữ bắt chước vector của mô hình tiếng Anh trên các cặp câu dịch.
- **Mô hình**: `paraphrase-multilingual-mpnet-base-v2` (50+ ngôn ngữ, có tiếng Việt).
- **Tham khảo**: Reimers & Gurevych (2019), EMNLP; Reimers & Gurevych (2020), *Making Monolingual Sentence Embeddings Multilingual*, EMNLP.

### 3.4 Vietnamese bi-encoder (`vietnamese_sbert`)
- **Ý tưởng**: PhoBERT fine-tune theo kiểu SBERT (contrastive) trên dữ liệu tiếng Việt. Kết hợp hiểu biết tiếng Việt của PhoBERT với vector câu chất lượng của SBERT. Cần văn bản đã tách từ.
- **Mô hình**: `bkai-foundation-models/vietnamese-bi-encoder` (BKAI, ĐH Bách khoa Hà Nội, 2023).

### 3.5 multilingual-E5 (`multilingual_e5`)
- **Ý tưởng**: pretrain contrastive yếu giám sát trên ~1 tỉ cặp văn bản đa ngữ, sau đó fine-tune có giám sát. Đầu vào cần tiền tố `"query: "`. Là một trong các embedding đa ngữ mạnh và được dùng rộng rãi nhất.
- **Tham khảo**: Wang et al. (2024), *Multilingual E5 Text Embeddings: A Technical Report*, arXiv:2402.05672.

### 3.6 GTE-multilingual (`gte_multilingual`)
- **Ý tưởng**: encoder 305M tham số, 70+ ngôn ngữ, ngữ cảnh 8192 token, huấn luyện contrastive nhiều giai đoạn. Hiệu năng gần các mô hình lớn hơn nhiều.
- **Tham khảo**: Zhang et al. (2024), *mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval*, EMNLP Industry.

### 3.7 BGE-M3 (`bge_m3`, nặng)
- **Ý tưởng**: nền XLM-RoBERTa-large (568M). "M3" = Multi-lingual (100+ ngôn ngữ), Multi-granularity (tới 8192 token), Multi-functionality (dense, sparse và multi-vector). Huấn luyện bằng **self-knowledge distillation** giữa ba kiểu biểu diễn. Thuộc nhóm đầu bảng MMTEB của các mô hình mã nguồn mở.
- **Tham khảo**: Chen et al. (2024), *BGE M3-Embedding*, Findings of ACL.

### 3.8 Vietnamese_Embedding (`vietnamese_embedding`, nặng)
- **Ý tưởng**: BGE-M3 fine-tune thêm trên khoảng 300k cặp câu tiếng Việt. Là một embedding tiếng Việt mới (2025).
- **Mô hình**: `AITeamVN/Vietnamese_Embedding`.

### 3.9 Qwen3-Embedding-0.6B (`qwen3_embedding`, nặng)
- **Ý tưởng**: embedding dựa trên **LLM** (Qwen3, decoder-only) thay vì encoder kiểu BERT. Vector câu lấy từ token cuối; huấn luyện nhiều giai đoạn với dữ liệu tổng hợp do LLM sinh ra. Đứng đầu MMTEB trong nhóm cỡ nhỏ (6/2025), đại diện cho xu hướng "LLM làm embedding" hiện nay.
- **Tham khảo**: Zhang et al. (2025), *Qwen3 Embedding*, arXiv:2506.05176.

---

## 4. Hybrid (`hybrid_tfidf_e5`)
- **Ý tưởng**: nối TF-IDF (từ khóa khía cạnh chính xác) với E5 (ngữ nghĩa câu, đồng nghĩa). Cả hai đều chuẩn hóa L2 nên cùng thang đo. Tương tự *hybrid retrieval* kết hợp sparse và dense, vốn bù được điểm yếu của nhau.
- **Tham khảo**: Luan et al. (2021), *Sparse, Dense, and Attentional Representations for Text Retrieval*, TACL.

---

## 5. Câu hỏi nghiên cứu mà lưới M x N trả lời

1. **Sparse hay contextual?** Đề yêu cầu so sánh TF-IDF với contextual embedding. Đọc theo hàng của heatmap (cố định mô hình, đổi biểu diễn).
2. **Embedding tiếng Việt hay đa ngữ?** PhoBERT / ViSoBERT / vietnamese-bi-encoder so với E5 / SBERT / GTE.
3. **Domain mạng xã hội có giúp không?** ViSoBERT so với PhoBERT.
4. **Static embedding học trên corpus nhỏ có đủ không?** Word2Vec/FastText so với TF-IDF.
5. **Mô hình nào hợp với biểu diễn nào?** Tuyến tính thường thắng trên sparse; SVM-RBF/MLP thường hợp hơn với embedding đặc.
