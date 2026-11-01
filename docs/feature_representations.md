# Các cách biểu diễn văn bản: tóm tắt ý tưởng

Tài liệu đi kèm `src/absa/features/registry.py`. Mỗi mục gồm: ý tưởng cốt lõi, cách dùng trong project, ưu/nhược điểm với bài toán ABSA trên bình luận điện thoại, và công bố gốc.

Sơ đồ chung: **văn bản → vector cố định số chiều → bộ phân loại ML** (LR, SVM, NB, RF, MLP, ...). Mọi biểu diễn chỉ được fit trên train.

| Họ | Biểu diễn | Số chiều | Học từ dữ liệu của ta? | Hiểu ngữ cảnh? |
|---|---|---|---|---|
| Sparse | BoW, TF-IDF từ/ký tự | 10⁴–10⁵ (thưa) | Có (từ vựng, IDF) | Không (chỉ n-gram cục bộ) |
| Static | Word2Vec, FastText, Doc2Vec | 300 | Có (huấn luyện trên train) | Không (1 từ = 1 vector) |

---

## 1. Họ sparse (túi từ)

### 1.1 Bag-of-Words (`bow`)
- **Ý tưởng**: mỗi chiều là một n-gram trong từ vựng, giá trị là số lần xuất hiện. Bỏ qua thứ tự từ ngoài phạm vi n-gram.
- **Trong project**: uni + bi-gram trên văn bản đã tiền xử lý (`màn hình`, `pin trâu`, `không_mượt`), bỏ n-gram xuất hiện ở ít hơn 2 bình luận.
- **Ưu**: đơn giản, nhanh, dễ giải thích (xem được trọng số từng từ). **Nhược**: từ phổ biến lấn át; `đẹp` và `xinh` là hai chiều không liên quan.
- **Tham khảo**: Harris (1954); Manning, Raghavan & Schütze (2008), *Introduction to Information Retrieval*.

### 1.2 TF-IDF mức từ (`tfidf_word`)
- **Ý tưởng**: trọng số = tần suất trong câu × `log(N / số câu chứa từ)`. Từ xuất hiện khắp nơi (`máy`, `mua`) bị giảm trọng số, từ đặc trưng cho khía cạnh (`pin`, `camera`, `nhân viên`) được nhấn mạnh. Dùng `sublinear_tf` (`1 + log tf`) để từ lặp nhiều lần không lấn át.
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

## 3. Câu hỏi nghiên cứu mà lưới M x N trả lời

1. **Sparse hay static?** Đọc theo hàng của heatmap (cố định mô hình, đổi biểu diễn) để so TF-IDF với Word2Vec/FastText/Doc2Vec.
2. **Static embedding học trên corpus nhỏ có đủ không?** Word2Vec/FastText so với TF-IDF.
3. **Chi tiết nào của sparse đáng giá?** Các biến thể ở mục 1.5 tách riêng ảnh hưởng của bigram, trigram, nhị phân hóa và n-gram ký tự.
4. **Mô hình nào hợp với biểu diễn nào?** Tuyến tính thường thắng trên sparse; SVM-RBF/MLP thường hợp hơn với embedding đặc.
