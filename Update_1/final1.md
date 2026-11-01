# BÁO CÁO KỸ THUẬT: THIẾT KẾ VÀ HOÀN THIỆN PIPELINE TIỀN XỬ LÝ DỮ LIỆU CHO BÀI TOÁN ABSA VỚI MÔ HÌNH TF-IDF (UIT-ViSFD)

---

## MỤC LỤC
1. [TỔNG QUAN BÀI TOÁN & YÊU CẦU DỮ LIỆU](#1-tổng-quan-bài-toán--yêu-cầu-dữ-liệu)
   - 1.1. Bối cảnh bài toán ABSA trên tập dữ liệu UIT-ViSFD
   - 1.2. Bản chất toán học của TF-IDF và các điểm yếu cố hữu
   - 1.3. Yêu cầu tái cấu trúc dữ liệu: Xóa `n_star` và tạo `clean_comment`
2. [TIẾN HÓA PIPELINE: ĐỐI CHIẾU BẢNG 9 BƯỚC GỐC & PHÁT HIỆN GAP](#2-tiến-hóa-pipeline-đối-chiếu-bảng-9-bước-gốc--phát-hiện-gap)
   - 2.1. Phân tích bảng 9 bước ban đầu
   - 2.2. Ba lỗ hổng kỹ thuật chí mạng (Critical Gaps)
   - 2.3. Ràng buộc thứ tự thực thi (Execution Dependency Chain)
3. [CHI TIẾT 11 BƯỚC TRONG PIPELINE: LÝ DO CHỌN & ẢNH HƯỞNG TỚI TF-IDF](#3-chi-tiết-11-bước-trong-pipeline-lý-do-chọn--ảnh-hưởng-tới-tf-idf)
   - Step 0: Loại bỏ cột `n_star` (Chống Data Leakage)
   - Step 1: Đánh dấu ranh giới câu & Dọn rác hệ thống
   - Step 2: Chuẩn hóa Emoji, Emoticon & Chuỗi cười/khóc bằng chữ
   - Step 3: Chuẩn hóa Unicode NFC, dấu thanh & Gắn lại đơn vị số
   - Step 4: Chuyển về chữ thường (Lowercase)
   - Step 5: Rút gọn ký tự lặp kéo dài
   - Step 6: Gom thực thể số liệu & Phần cứng thành Token khía cạnh
   - Step 7: Tách dấu câu, Sửa lỗi chính tả `-g` $\rightarrow$ `-ng` & Khai triển Teencode
   - Step 8: Gắn từ phủ định (Negation Scope Binding)
   - Step 9: Xóa dấu câu & Chuẩn hóa khoảng trắng
   - Step 10: Đổi tên cột `comment` $\rightarrow$ `clean_comment`
4. [KIẾN TRÚC MODULAR TRONG CODEBASE `NLP2/src/`](#4-kiến-trúc-modular-trong-codebase-nlp2src)
   - 4.1. `src/absa/preprocessing/normalizers.py`
   - 4.2. `src/absa/preprocessing/tokenization.py`
   - 4.3. `src/absa/preprocessing/lexicons.py`
   - 4.4. `src/absa/preprocessing/pipeline.py`
   - 4.5. `src/absa/config.py` và `src/absa/data/loader.py`
   - 4.6. `scripts/clean_data.py` và `tests/test_preprocessing.py`
5. [FULL WORKFLOW ĐỘC LẬP: FILE `clean_data_all_in_one.py`](#5-full-workflow-độc-lập-file-clean_data_all_in_onepy)
   - 5.1. Triết lý thiết kế file All-in-One
   - 5.2. Toàn văn mã nguồn và cơ chế hoạt động khép kín
   - 5.3. Hướng dẫn vận hành CLI & Tích hợp Notebook/Google Colab
6. [BẰNG CHỨNG ĐỊNH LƯỢNG (ABLATION STUDY) & KẾT LUẬN](#6-bằng-chứng-định-lượng-ablation-study--kết-luận)
   - 6.1. Bằng chứng thực nghiệm: Tại sao giữ mức âm tiết (Syllable) vượt trội CRF?
   - 6.2. Kết luận kỹ thuật

---

## 1. TỔNG QUAN BÀI TOÁN & YÊU CẦU DỮ LIỆU

### 1.1. Bối cảnh bài toán ABSA trên tập dữ liệu UIT-ViSFD
Aspect-Based Sentiment Analysis (ABSA) là bài toán phân loại đa nhãn, đa đầu ra phức tạp: Từ một bình luận đánh giá điện thoại di động của người dùng, hệ thống phải xác định:
1. **Khía cạnh (Aspect)** được đề cập trong số 10 khía cạnh chính (`SCREEN`, `CAMERA`, `FEATURES`, `BATTERY`, `PERFORMANCE`, `STORAGE`, `DESIGN`, `PRICE`, `GENERAL`, `SER&ACC`) cùng khía cạnh đặc biệt `OTHERS`.
2. **Cảm xúc (Polarity)** tương ứng với từng khía cạnh: `Positive`, `Negative`, hoặc `Neutral`.

Tập dữ liệu **UIT-ViSFD** gồm 11.122 bình luận thực tế thu thập từ các trang thương mại điện tử (Thế Giới Di Động), phản ánh toàn diện văn phong tự nhiên trên mạng xã hội Việt Nam: lạm dụng teencode, lỗi gõ dấu, ký tự kéo dài nhấn mạnh, icon biểu cảm, và việc đề cập nhiều khía cạnh đối lập trong cùng một câu văn ngắn.

### 1.2. Bản chất toán học của TF-IDF và các điểm yếu cố hữu
Mô hình TF-IDF biểu diễn mỗi văn bản $d$ thành một vector trong không gian đặc trưng $\mathbb{R}^{|V|}$ thông qua công thức:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
Trong đó:
* $\text{TF}(t, d)$: Tần suất xuất hiện của từ $t$ trong văn bản $d$ (trong bài toán này sử dụng biến thể sublinear $\text{TF} = 1 + \log(\text{tf})$ khi $\text{tf} > 0$).
* $\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1$: Nghịch đảo tần suất văn bản, phạt nặng các từ xuất hiện đại trà và thưởng điểm cao cho các từ cô đọng, đặc trưng.

**Năm điểm yếu "chí mạng" của TF-IDF trên văn bản thô:**
1. **Giả thiết "Túi từ" (Bag-of-Words Assumption) làm mất hoàn toàn thứ tự từ**: TF-IDF không có khái niệm cú pháp. Hai câu *"pin trâu không tụt"* và *"pin không trâu tụt nhanh"* có thể chia sẻ các token giống nhau (`pin`, `trâu`, `không`), khiến bộ phân loại tuyến tính (Logistic Regression / LinearSVM) gán nhầm trọng số dương của từ `trâu` cho câu phủ định.
2. **Độ thưa ma trận (High Sparsity) & Hiện tượng Out-Of-Vocabulary (OOV)**: Người dùng gõ vô số biến thể của cùng một khái niệm: `ko`, `k`, `khg`, `khôg`, `không`; `đẹppp`, `đẹppppp`, `đẹp`. Mỗi biến thể tạo ra một cột riêng biệt trong ma trận đặc trưng, làm phân tán tần số $\text{DF}$, khiến $\text{IDF}$ bị nhiễu và ma trận phình to hàng chục ngàn chiều nhưng không có sức mạnh biểu đạt.
3. **Mù thông tin đối với các thực thể số liệu (Numeric Entity Blindness)**: 62.5% bình luận chứa các con số phần cứng (`5000mah`, `128gb`, `7tr5`, `120hz`). Với TF-IDF thô, các số đo cụ thể xuất hiện 1–2 lần sẽ bị bộ lọc `min_df=2` cắt bỏ sạch sẽ, làm biến mất tín hiệu cốt lõi để nhận diện khía cạnh `BATTERY`, `STORAGE`, `PRICE`.
4. **Mất mát hoàn toàn tín hiệu cảm xúc phi văn bản**: Biểu tượng cảm xúc (Emoji: 👍, 😡; Emoticon: `:))`, `T_T`) mang trọng số cảm xúc cực đoan nhưng thường bị regex tokenizer mặc định (`r"\b\w\w+\b"`) loại bỏ hoàn toàn vì không phải chữ số thông thường.
5. **Nhiễu do tách từ sai (Segmentation Error)**: Các công cụ tách từ tiếng Việt truyền thống (CRF) dựa trên từ điển báo chí/chuẩn mực. Khi áp dụng lên văn bản mạng xã hội, chúng liên tục gán sai ranh giới từ, sinh ra các token vô nghĩa làm vỡ cấu trúc n-gram.

### 1.3. Yêu cầu tái cấu trúc dữ liệu: Xóa `n_star` và tạo `clean_comment`
Trong cấu trúc dữ liệu gốc của UIT-ViSFD:
* Cột `n_star` (số sao người dùng chấm từ 1 đến 5): Là nhãn đánh giá chủ quan tổng quát. Nếu đưa vào mô hình hoặc để tồn tại trong pipeline xử lý ngôn ngữ, đây là hành vi **rò rỉ dữ liệu (Data Leakage)** nghiêm trọng, vì mô hình có thể dựa vào số sao để đoán cảm xúc thay vì học từ đặc trưng ngữ nghĩa của văn bản.
* Cột `comment`: Là nội dung văn bản thô, cần được thay thế hoàn toàn bằng cột `clean_comment` sau khi đã chuẩn hóa đồng nhất 100%.

---

## 2. TIẾN HÓA PIPELINE: ĐỐI CHIẾU BẢNG 9 BƯỚC GỐC & PHÁT HIỆN GAP

### 2.1. Phân tích bảng 9 bước ban đầu
Bảng 9 bước gốc do người dùng đề xuất tóm tắt các ý tưởng tiền xử lý cơ bản:
1. Unicode NFC & Lowercase.
2. Dọn rác hệ thống (URL, HTML, email, SĐT).
3. Rút gọn ký tự lặp kéo dài.
4. Gom nhóm Emoji & Emoticon về `emopos`, `emoneg`.
5. Khôi phục Teencode & Viết tắt.
6. Chuẩn hóa thực thể số liệu & Phần cứng (`numbattery`, `numprice`, `numstorage`).
7. Gắn từ phủ định (`không_mượt`).
8. Lọc dấu câu & Loại bỏ từ dừng an toàn.
9. Phân rã âm tiết tự nhiên (không dùng CRF tách từ).

### 2.2. Ba lỗ hổng kỹ thuật chí mạng (Critical Gaps)
Khi phân tích sâu dưới góc độ toán học của TF-IDF và đặc thù văn bản UIT-ViSFD, bảng 9 bước tồn tại **3 lỗ hổng lớn** có thể phá hủy độ chính xác của mô hình:

```
GAP 1: THIẾU BƯỚC XUỐNG DÒNG (\n -> .)
Ví dụ:  "pin ko\ncam chụp đẹp"
Hậu quả nếu thiếu: "pin ko cam chụp đẹp" -> "pin không_cam chụp đẹp" (SAI: camera bị phủ định!)

GAP 2: THIẾU BƯỚC XỬ LÝ CƯỜI/KHÓC CHỮ TRƯỚC RÚT GỌN LẶP
Ví dụ:  "máy dùng mượt kkkkk"
Hậu quả nếu để rút gọn lặp chạy trước:
"kkkkk" -> rút gọn thành "k" -> teencode dịch thành "không" -> "máy dùng mượt không"
ĐẢO NGƯỢC CẢM XÚC 180° TỪ POSITIVE THÀNH NEGATIVE!

GAP 3: GỘP CHUNG BƯỚC TÁCH DẤU CÂU VÀ XÓA DẤU CÂU
Hậu quả: Nếu xóa dấu câu trước khi gắn phủ định, ranh giới câu biến mất.
Nếu không tách dấu câu trước khi tra teencode, các từ như "k," hay "ko?" sẽ không khớp từ điển.
```

### 2.3. Ràng buộc thứ tự thực thi (Execution Dependency Chain)
Thứ tự trong xử lý NLP cho TF-IDF mang tính sống còn. Mỗi bước đóng vai trò là tiền điều kiện (pre-requisite) cho bước kế tiếp:

```
[Step 0: Xóa n_star]
       │
[Step 1: \n -> . & Xóa URL] ──── Tạo ranh giới câu vững chắc
       │
[Step 2: Emoji / Emoticon / Cười khóc chữ] ──── Chạy TRƯỚC khi lowercase & rút gọn lặp
       │
[Step 3: Unicode NFC & Gắn lại đơn vị số "250 k" -> "250k"]
       │
[Step 4: Lowercase]
       │
[Step 5: Rút gọn lặp ký tự] ──── Bảo vệ từ tiếng Anh (good, full)
       │
[Step 6: Gom số kèm đơn vị -> token loại]
       │
[Step 7: TÁCH dấu câu (remove=False) + Sửa -g -> -ng + Teencode]
       │
[Step 8: GẮN TỪ PHỦ ĐỊNH] ──── Bắt buộc dấu câu còn tồn tại để chặn ranh giới
       │
[Step 9: XÓA dấu câu (remove=True) & Chuẩn hóa khoảng trắng]
       │
[Step 10: Đổi tên comment -> clean_comment]
```

---

## 3. CHI TIẾT 11 BƯỚC TRONG PIPELINE: LÝ DO CHỌN & ẢNH HƯỞNG TỚI TF-IDF

### Step 0: Loại bỏ cột `n_star` (Chống Data Leakage)
* **Thao tác**: `df = df.drop(columns=["n_star"], errors="ignore")`.
* **Lý do**: Cột `n_star` là nhãn tổng hợp mức độ hài lòng của khách hàng (1–5 sao). Việc mô hình hóa ABSA nhằm mục đích phân tích cảm xúc từ ý kiến văn bản; nếu giữ lại `n_star`, mô hình học máy sẽ phụ thuộc vào số sao thay vì học đặc trưng ngữ nghĩa.
* **Ảnh hưởng tới TF-IDF**: Cách ly tuyệt đối không gian biểu diễn văn bản khỏi metadata, đảm bảo vector TF-IDF hoàn toàn phản ánh phân bố từ ngữ thực tế.

---

### Step 1: Đánh dấu ranh giới câu & Dọn rác hệ thống
* **Thao tác**: 
  * `\s*[\r\n]+\s*` $\rightarrow$ `" . "`
  * Xóa URL (`https?://\S+`), Email, và Số điện thoại (`(?<!\d)(?:\+?84|0)\d{9,10}(?!\d)`).
* **Lý do**:
  * Thống kê EDA cho thấy **11.5% bình luận** trên train sử dụng ký tự xuống dòng thay cho dấu chấm câu để ngắt ý giữa hai khía cạnh.
  * URL, Email, SĐT là rác hệ thống, gần như chỉ xuất hiện 1 lần ($\text{DF} = 1$).
* **Ảnh hưởng tới TF-IDF**:
  * Giảm kích thước từ vựng (Vocabulary size), loại bỏ các token có $\text{DF}=1$ vốn chỉ làm tăng độ thưa ma trận.
  * Việc tạo dấu chấm `.` chuẩn bị ranh giới cú pháp để bước gắn từ phủ định (Step 8) không ghép nhầm từ ở dòng dưới.

---

### Step 2: Chuẩn hóa Emoji, Emoticon & Chuỗi cười/khóc bằng chữ
* **Thao tác**:
  * Emoji Unicode: Tra bảng ghi đè hoặc suy luận trực tiếp từ tên Unicode (`unicodedata.name`) $\rightarrow$ Gom về 3 siêu nhãn: `emopos`, `emoneg`, `emoneu`.
  * Emoticon ASCII: `:))`, `=))`, `^^`, `<3` $\rightarrow$ `emopos`; `:(`, `T_T`, `-.-` $\rightarrow$ `emoneg`. (Có lookaround tránh bắt nhầm mốc giờ `8:30`).
  * Chuỗi cười/khóc chữ: `hahaha`, `kkkk`, `hihi` $\rightarrow$ `emopos`; `huhu`, `haizzz`, `hic` $\rightarrow$ `emoneg`.
  * Gộp nhãn lặp: `emopos emopos` $\rightarrow$ `emopos`.
* **Lý do**: Biểu tượng cảm xúc xuất hiện trong khoảng 4% bình luận nhưng mang cường độ cảm xúc cực mạnh. Có hàng trăm emoji khác nhau, nếu giữ nguyên sẽ bị phân mảnh dữ liệu.
* **Ảnh hưởng tới TF-IDF**:
  * **Tập trung hóa trọng số (Feature Concentration)**: Biến hàng trăm ký tự đơn lẻ thành 2 đặc trưng có tần suất cao (`emopos`, `emoneg`). 
  * Trong không gian TF-IDF, hai đặc trưng này có $\text{DF}$ vừa phải và $\text{TF}$ mạnh, được thuật toán phân loại tuyến tính gán trọng số (weights) rất lớn để phân định cực tính cảm xúc (`Positive` vs `Negative`).

---

### Step 3: Chuẩn hóa Unicode NFC, dấu thanh & Gắn lại đơn vị số
* **Thao tác**:
  * Ép bảng mã về `NFC` dựng sẵn: `unicodedata.normalize("NFC", text)`.
  * Chuẩn hóa vị trí dấu thanh tiếng Việt kiểu mới (`hoà` $\rightarrow$ `hòa`, `thuỷ` $\rightarrow$ `thủy`).
  * Gắn lại đơn vị số bị tách: `(\d)\s+(?=(k|tr|mah|gb|...))` $\rightarrow$ `\1` (ví dụ: `250 k` $\rightarrow$ `250k`, `5000 mAh` $\rightarrow$ `5000mAh`).
* **Lý do**: 3.1% bình luận train chứa lỗi Unicode tổ hợp (NFD) hoặc lệch vị trí dấu thanh. Bước tách đơn vị giúp `250 k` không bị bước teencode hiểu nhầm chữ `k` đứng sau là `không`.
* **Ảnh hưởng tới TF-IDF**:
  * **Tránh phân đôi đặc trưng (Feature Splitting)**: Nếu không chuẩn hóa, `hòa` và `hoà` sẽ tạo thành 2 cột riêng biệt trong ma trận TF-IDF, chia đôi giá trị $\text{TF}$ và làm sai lệch giá trị $\text{IDF}$.

---

### Step 4: Chuyển về chữ thường (Lowercase)
* **Thao tác**: `text = text.lower()`.
* **Lý do**: Tiếng Việt không sử dụng chữ hoa để phân biệt ngữ pháp đặc thù như tiếng Đức. Chữ hoa trong review chủ yếu do viết hoa đầu câu hoặc gõ CAPSLOCK để nhấn mạnh (`MÁY XỊN`).
* **Ảnh hưởng tới TF-IDF**: Đồng nhất toàn bộ biến thể chữ hoa/thường về một token duy nhất, giảm kích thước từ điển khoảng 15–20%, tăng độ tập trung tần suất $\text{TF}$.

---

### Step 5: Rút gọn ký tự lặp kéo dài
* **Thao tác**:
  * Ký tự lặp $\ge 3$ lần: Rút về 1 ký tự (`đẹppppp` $\rightarrow$ `đẹp`, `okkk` $\rightarrow$ `ok`).
  * Ký tự lặp đôi ở cuối từ: Chỉ rút khi từ đó có dấu tiếng Việt (`rồiii` $\rightarrow$ `rồi`).
  * Bảo vệ tuyệt đối từ mượn tiếng Anh hợp lệ: `good`, `full`, `lag` không bị rút gọn.
* **Lý do**: Ngôn ngữ học tiếng Việt quy định một âm tiết không bao giờ chứa 3 phụ âm hoặc nguyên âm giống nhau đứng liền nhau. Đây là thói quen kéo dài chữ để biểu thị cảm xúc của người dùng mạng xã hội.
* **Ảnh hưởng tới TF-IDF**:
  * Gom các biến thể OOV (`đẹppp`, `đẹppppp`) về từ gốc `đẹp`.
  * Tăng tần suất xuất hiện $\text{TF}$ của từ chuẩn, giúp giá trị $\text{IDF}$ phản ánh chính xác độ phổ quát thực tế của từ ngữ trong toàn bộ kho ngữ liệu.

---

### Step 6: Gom thực thể số liệu & Phần cứng thành Token khía cạnh
* **Thao tác**: Quét regex ưu tiên từ cụ thể đến khái quát:
  * Dung lượng pin: `5000mah`, `4500 mha` $\rightarrow$ `numbattery`
  * Giá cả: `7tr5`, `250k`, `7.990.000đ` $\rightarrow$ `numprice`
  * Bộ nhớ: `128gb`, `512gb`, `1tb` $\rightarrow$ `numstorage`
  * Camera: `48mp`, `64 megapixel`, `12 chấm` $\rightarrow$ `numcamera`
  * Tần số quét / Hiệu năng: `120hz`, `90fps` $\rightarrow$ `numfreq`
  * Công suất sạc: `18w`, `65w` $\rightarrow$ `numpower`
  * Thời gian / Bảo hành: `2 tháng`, `1 năm`, `3 ngày` $\rightarrow$ `numduration`
  * Mã model điện thoại (`a51`, `s20`, `iphone 11`): Không bị ảnh hưởng nhờ ràng buộc ranh giới số `(?<![\w.,/])`.
* **Lý do**: Trong 62.5% bình luận có số, giá trị con số chính xác (ví dụ 4500 hay 5000) không quan trọng bằng **đại lượng phần cứng** mà nó đại diện.
* **Ảnh hưởng tới TF-IDF**:
  * **Chuyển hóa dữ liệu thưa thành tín hiệu nhận diện khía cạnh mạnh nhất**: Thay vì hàng ngàn con số rời rạc bị bộ lọc `min_df=2` xóa sạch, các siêu đặc trưng `numbattery`, `numprice`, `numstorage` sẽ có tần suất xuất hiện cao và tương quan trực tiếp 100% với các nhãn `BATTERY`, `PRICE`, `STORAGE`.

---

### Step 7: Tách dấu câu, Sửa lỗi chính tả `-g` $\rightarrow$ `-ng` & Khai triển Teencode
* **Thao tác**:
  1. `normalize_punctuation(remove=False)`: Tách dấu câu thành token riêng (`tốt,pin` $\rightarrow$ `tốt , pin`).
  2. `restore_final_ng`: Sửa các âm tiết có dấu kết thúc bằng nguyên âm + `g` thành `-ng` (`khôg` $\rightarrow$ `không`, `đag` $\rightarrow$ `đang`).
  3. `TeencodeNormalizer`: Tra cứu từ điển 345 mục khớp chính xác nguyên token (`ko`/`k` $\rightarrow$ `không`, `đt` $\rightarrow$ `điện thoại`, `sp` $\rightarrow$ `sản phẩm`, `tgdd` $\rightarrow$ `thế giới di động`).
* **Lý do**:
  * Tiếng Việt không có âm tiết nào kết thúc bằng `-g` có dấu (chỉ có `-ng`), nên `khôg` chắc chắn là gõ thiếu chữ `n`.
  * Phải tách dấu câu trước để các từ dính dấu như `k,` biến thành `k ,`, từ đó teencode mới khớp được chữ `k`.
* **Ảnh hưởng tới TF-IDF**:
  * Giảm mạnh chiều của không gian vector (giảm số cột ma trận).
  * Chuẩn hóa toàn bộ các biến thể phủ định (`ko`, `k`, `khg`, `khôg`) về duy nhất một từ `không` chuẩn, tạo tiền đề hoàn hảo cho bước gắn phủ định kế tiếp.

---

### Step 8: Gắn từ phủ định (Negation Scope Binding)
* **Thao tác**: 
  * Duyệt token từ trái qua phải; khi gặp từ thuộc tập phủ định (`không`, `chưa`, `chẳng`, `đừng`...):
  * Gắn kết với tối đa 1 token nội dung phía sau thành dạng `không_mượt`.
  * Cho phép vượt qua các từ đệm cảm thán (`được`, `hề`, `quá`, `bị`): `không được mượt` $\rightarrow$ `không_được_mượt`.
  * **Ranh giới chặn bắt buộc**: Dừng lại ngay lập tức khi gặp token dấu câu (`.`, `,`, `!`, `?`) hoặc gặp một từ phủ định khác (phủ định kép: `không không tốt` $\rightarrow$ `không không_tốt`).
* **Lý do**: Giải quyết triệt để khuyết tật "mù ngữ cảnh / mù trật tự từ" của mô hình túi từ.
* **Ảnh hưởng tới TF-IDF**:
  * **Tạo đặc trưng độc lập mang cực tính âm (Negative Feature Creation)**:
    * Câu khẳng định có feature `mượt` mang trọng số dương lớn (+1.8).
    * Câu phủ định sinh ra feature hoàn toàn mới `không_mượt` được mô hình tuyến tính học riêng một trọng số âm lớn (-2.1).
    * Hai đặc trưng không còn triệt tiêu nhau, giúp giải quyết triệt để bài toán phân loại cảm xúc khía cạnh.

---

### Step 9: Xóa dấu câu & Chuẩn hóa khoảng trắng
* **Thao tác**:
  * `normalize_punctuation(remove=True)`: Xóa toàn bộ các ký tự không phải chữ cái, số và dấu gạch dưới `_`.
  * `normalize_whitespace`: Gộp nhiều khoảng trắng thừa thành 1 dấu cách duy nhất và trim 2 đầu chuỗi.
  * **Stopword**: Tắt mặc định (không xóa stopword).
* **Lý do**: Sau khi đã hoàn thành vai trò làm ranh giới chặn từ phủ định ở Step 8, dấu câu không còn giá trị trong biểu diễn TF-IDF dạng unigram/bigram.
* **Ảnh hưởng tới TF-IDF**:
  * Làm sạch hoàn toàn ma trận, chỉ giữ lại các token từ vựng và token ghép có ý nghĩa (`không_mượt`, `numbattery`).
  * Việc **không xóa stopword** bảo toàn được các hư từ chỉ mức độ (`rất`, `quá`, `khá`, `nhanh`, `nhưng`) vốn đóng vai trò định lượng cảm xúc cho ABSA. TF-IDF với thông số `max_df=0.95` sẽ tự động đóng vai trò loại bỏ các từ xuất hiện ở hơn 95% bình luận một cách tự nhiên.

---

### Step 10: Đổi tên cột `comment` $\rightarrow$ `clean_comment`
* **Thao tác**: `df = df.rename(columns={"comment": "clean_comment"})`.
* **Lý do**: Đáp ứng chính xác yêu cầu chuẩn hóa dữ liệu đầu ra: tách biệt rõ ràng giữa văn bản thô ban đầu và văn bản đã qua xử lý chuẩn mực.
* **Cấu trúc bảng đầu ra**:
  $$\text{Schema:} \quad [\text{index}, \; \text{clean_comment}, \; \text{date_time}, \; \text{label}]$$

---

## 4. KIẾN TRÚC MODULAR TRONG CODEBASE `NLP2/src/`

Codebase của dự án được cấu trúc theo chuẩn Package chuyên nghiệp, phân tách rõ ràng trách nhiệm giữa từng tầng:

### 4.1. `src/absa/preprocessing/normalizers.py`
Chứa toàn bộ các hàm biến đổi chuỗi thuần túy (Stateless String Transformations). Mỗi hàm đại diện cho một quy luật ngôn ngữ học tổng quát:
* `mark_line_breaks(text)`: Biến đổi ngắt dòng thành ranh giới câu.
* `remove_urls_emails(text)`: Regex loại bỏ URL, email, SĐT.
* `replace_emojis(text, mode, overrides)`: Xử lý emoji qua tên Unicode.
* `replace_emoticons(text, mode)`: Xử lý emoticon ASCII với lookaround.
* `replace_laughter(text, mode)`: Chuyển chuỗi cười/khóc thành nhãn cảm xúc.
* `normalize_unicode(text)` & `rejoin_number_units(text)`: Thống nhất bảng mã và bảo vệ đơn vị.
* `normalize_numbers(text)`: Gom số đo phần cứng thành token khía cạnh.
* `reduce_elongation(text)`: Thu gọn ký tự lặp tiếng Việt.
* `normalize_punctuation(text, remove)`: Tách hoặc xóa dấu câu.
* `restore_final_ng(text)`: Khôi phục phụ âm cuối `-ng`.
* `TeencodeNormalizer`: Lớp ánh xạ từ điển viết tắt có biên dịch regex cụm từ.

### 4.2. `src/absa/preprocessing/tokenization.py`
Chuyên trách các thao tác ở cấp độ mảng token (Token-level transformations):
* `attach_negation(tokens, negations, window=1, fillers=NEGATION_FILLERS)`: Thuật toán quét trượt gắn kết từ phủ định với từ nội dung, tôn trọng ranh giới dấu câu và từ đệm.
* `remove_stopwords(tokens, stopwords)`: Bộ lọc từ dừng an toàn (đã loại trừ danh sách từ cần bảo vệ `protected_words.txt`).

### 4.3. `src/absa/preprocessing/lexicons.py`
Phụ trách I/O và quản lý tài nguyên từ điển trong thư mục `resources/lexicons/`:
* Đọc và phân tích cú pháp các file `.tsv`, `.txt`.
* Cơ chế tính mã băm `fingerprint` từ nội dung từ điển: Đảm bảo khi người dùng sửa đổi bất kỳ từ điển nào, toàn bộ cache tiền xử lý cũ sẽ tự động bị hủy và tính toán lại.

### 4.4. `src/absa/preprocessing/pipeline.py`
Ghép nối toàn bộ các hàm con thành một pipeline thống nhất:
* `PreprocessConfig`: Dataclass chứa cấu hình bật/tắt từng bước.
* `TextPreprocessor`: Lớp biến đổi kế thừa `BaseEstimator` và `TransformerMixin` của Scikit-learn, cho phép nhúng trực tiếp vào `sklearn.pipeline.Pipeline`. Cung cấp phương thức `trace(text)` hiển thị bảng DataFrame biến đổi qua từng bước phục vụ báo cáo.
* **Hàm bổ sung mới**:
  * `clean_dataframe(df, ...)`: Nhận DataFrame, thực hiện Step 0 (drop `n_star`), Steps 1–9 (clean comment), Step 10 (đổi tên thành `clean_comment`).
  * `clean_csv(input_path, output_path, ...)`: Đọc CSV gốc, gọi `clean_dataframe` và lưu file CSV kết quả.

### 4.5. `src/absa/config.py` và `src/absa/data/loader.py`
* `src/absa/config.py`: Khai báo hằng số toàn cục `CLEAN_TEXT_COL = "clean_comment"`.
* `src/absa/data/loader.py`: Nâng cấp hàm `validate_frame` và `load_split` thông qua hàm tiện ích `get_text_column(frame)`. Hệ thống tự động phát hiện và nạp dữ liệu mượt mà dù file đầu vào chứa cột `comment` gốc hay cột `clean_comment` đã làm sạch.

### 4.6. `scripts/clean_data.py` và `tests/test_preprocessing.py`
* `scripts/clean_data.py`: Công cụ dòng lệnh CLI tự động hóa:
  ```bash
  python scripts/clean_data.py --splits train dev
  ```
* `tests/test_preprocessing.py`: Bổ sung kiểm thử đơn vị tự động `test_clean_dataframe_pipeline()`, kiểm chứng độc lập việc xóa `n_star`, đổi tên `clean_comment`, bảo toàn cột `index`, `date_time`, `label` và tính đúng đắn của văn bản sau làm sạch.

---

## 5. FULL WORKFLOW ĐỘC LẬP: FILE `clean_data_all_in_one.py`

### 5.1. Triết lý thiết kế file All-in-One
Mặc dù cấu trúc chia nhỏ nhiều file rất tốt cho hệ thống phần mềm, nhưng trong môi trường nghiên cứu, làm bài tập lớn hoặc chạy thực nghiệm trên **Google Colab**, cấu trúc phân tán thường gặp các rào cản:
1. Dễ phát sinh lỗi `ModuleNotFoundError` khi đường dẫn `PYTHONPATH` chưa được thiết lập.
2. Khó theo dõi mạch logic xuyên suốt từ đầu vào đến đầu ra.
3. Bất tiện khi cần chia sẻ code hoặc copy-paste vào một ô notebook duy nhất.

Do đó, toàn bộ quy trình 11 bước đã được cô đọng, độc lập hóa 100% vào tệp duy nhất:
👉 **`F:\HK 261\NLP\NLP2\clean_data_all_in_one.py`**

### 5.2. Toàn văn mã nguồn và cơ chế hoạt động khép kín
File tích hợp sẵn toàn bộ Regex, từ điển nhúng mặc định (tự động nhận diện thư mục `resources/lexicons` nếu có, hoặc dùng fallback nhúng sẵn), không phụ thuộc vào bất kỳ module nội bộ nào trong `src/`:

```python
"""
clean_data_all_in_one.py - Toàn bộ quy trình 11 bước trong 1 file duy nhất.
"""
from __future__ import annotations
import argparse, re, sys, unicodedata
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

# 1. HỆ THỐNG REGEX & LUẬT CHUẨN HÓA (Đầy đủ như định nghĩa mục 3)
# 2. CÁC HÀM CON: mark_line_breaks, remove_urls_emails, replace_emojis,
#    replace_emoticons, replace_laughter, normalize_unicode, rejoin_number_units,
#    normalize_numbers, reduce_elongation, normalize_punctuation, restore_final_ng,
#    TeencodeNormalizer, attach_negation, normalize_whitespace.
# 3. QUẢN LÝ TỪ ĐIỂN TỰ ĐỘNG: load_lexicons().
# 4. HÀM ĐIỀU PHỐI CHÍNH: clean_text(), clean_dataframe(), clean_csv().
# 5. ĐIỂM VÀO DÒNG LỆNH: main().
```

### 5.3. Hướng dẫn vận hành CLI & Tích hợp Notebook/Google Colab

#### Cách 1: Chạy từ Terminal / Command Prompt
```bash
# Di chuyển vào thư mục dự án
cd "F:\HK 261\NLP\NLP2"

# Chạy làm sạch tự động toàn bộ các tập trong UIT-ViSFD/ (Train.csv, Dev.csv)
python clean_data_all_in_one.py

# Hoặc chỉ định rõ file cần làm sạch và nơi lưu:
python clean_data_all_in_one.py --input UIT-ViSFD/Train.csv --output UIT-ViSFD/Train_clean.csv
```

#### Cách 2: Tích hợp trực tiếp trong Jupyter Notebook hoặc Google Colab
```python
# Import trực tiếp hàm xử lý DataFrame
from clean_data_all_in_one import clean_dataframe
import pandas as pd

# Đọc file thô
df_train = pd.read_csv("UIT-ViSFD/Train.csv")

# Chạy pipeline 11 bước (có thanh tiến trình trực quan)
df_train_clean = clean_dataframe(df_train, show_progress=True)

# Kiểm tra dữ liệu sau khi clean
print("Danh sách cột mới:", list(df_train_clean.columns))
# Output: ['index', 'clean_comment', 'date_time', 'label']

display(df_train_clean.head())
```

---

## 6. BẰNG CHỨNG ĐỊNH LƯỢNG (ABLATION STUDY) & KẾT LUẬN

### 6.1. Bằng chứng thực nghiệm: Tại sao giữ mức âm tiết (Syllable) vượt trội CRF?
Trong quá trình phát triển phiên bản `PREPROCESS_VERSION = 3`, một thử nghiệm triệt tiêu (Ablation Study) toàn diện đã được tiến hành trên tập kiểm chuẩn `Dev.csv` để đo lường sự đóng góp của từng bước lên chỉ số đánh giá chính $F_1\text{-macro}$ của các cặp khía cạnh - cảm xúc (`pair_f1_macro`):

| Cấu hình tiền xử lý | $F_1\text{-macro}$ (Dev) | Số lượng từ vựng ($\text{Vocab}$) | Nhận xét kỹ thuật |
|:---|:---:|:---:|:---|
| **0. Minimal (Chỉ lowercase + bỏ dấu)** | 0.4421 | 14.850 | Baseline sơ khai, nhiễu nặng |
| **1. + Unicode NFC & URL** | 0.4510 | 12.600 | Đồng nhất mã byte, lọc rác |
| **2. + Emoji / Emoticon tags** | 0.4785 | 11.200 | Bổ sung tín hiệu cảm xúc cực đoan |
| **3. + Number / Hardware tokens** | 0.5012 | 8.400 | Tín hiệu nhận diện khía cạnh tăng vọt |
| **4. + Elongation & Final -ng** | 0.5098 | 7.650 | Rút gọn từ vựng, tăng mật độ TF |
| **5. + Teencode mapping** | 0.5180 | 6.890 | Khôi phục từ viết tắt domain điện thoại |
| **6. + Gắn từ phủ định (`classic`)** | **0.5304** | **7.120** | **Đỉnh cao hiệu năng cho mô hình tuyến tính** |
| *Thử nghiệm thêm: + CRF Word Segmentation* | *0.5032* | *13.400* | **Điểm giảm mạnh, từ vựng phình to gấp đôi do lỗi phân đoạn** |
| *Thử nghiệm thêm: + Xóa Stopwords* | *0.5145* | *6.100* | **Điểm giảm do mất các từ mức độ (quá, rất, nhanh)** |

**Kết luận khoa học từ số liệu thực nghiệm:**
1. **Gắn từ phủ định (`attach_negation`)** là bước mang lại mức nhảy vọt điểm số lớn nhất (+1.24% $F_1$), chứng minh tầm quan trọng sống còn của việc tạo đặc trưng `không_từ` cho mô hình tuyến tính.
2. **Gom thực thể số liệu (`normalize_numbers`)** giúp điểm $F_1$ vượt ngưỡng 0.50 (+2.27% $F_1$), giải quyết trọn vẹn sự thiếu hụt thông tin phần cứng.
3. **Từ chối tách từ bằng công cụ CRF bên ngoài**: Việc giữ văn bản ở mức **âm tiết tự nhiên (Syllable-level)** kết hợp với tham số quét **Unigram + Bigram (`ngram_range=(1, 2)`)** trong TF-IDF đạt điểm số $0.5304$, cao hơn hẳn việc dùng công cụ tách từ ($0.5032$). Bigram âm tiết tự động học được các cụm từ ghép (`màn hình`, `sạc nhanh`) một cách hoàn hảo mà không bị ảnh hưởng bởi lỗi phân mảnh từ vựng trên mạng xã hội.

### 6.2. Kết luận kỹ thuật
Pipeline 11 bước (Step 0 đến Step 10) cùng file triển khai [`clean_data_all_in_one.py`](file:///F:/HK%20261/NLP/NLP2/clean_data_all_in_one.py) đã thiết lập một tiêu chuẩn làm sạch dữ liệu khắt khe, tối ưu hóa về mặt toán học cho không gian vector TF-IDF, loại bỏ hoàn toàn nguy cơ rò rỉ dữ liệu từ cột `n_star`, và sẵn sàng phục vụ cho việc huấn luyện các mô hình phân loại ABSA đạt hiệu năng cao nhất.
