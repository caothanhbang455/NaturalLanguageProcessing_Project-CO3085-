# Thiết kế tiền xử lý

Code: `src/absa/preprocessing/`. Từ điển: `resources/lexicons/`. Bằng chứng định lượng: mục 2.3 (xu hướng viết) và mục 3.4 (ablation) của notebook.

## Nguyên tắc

1. **Luật tổng quát, không hardcode.** Train chỉ được dùng để *nhận diện xu hướng viết* (teencode, emoji, kéo dài ký tự, số + đơn vị, ...). Mỗi xu hướng được xử lý bằng một luật dựa trên chính tả tiếng Việt hoặc một từ điển đọc được và sửa được. Không có luật nào nhắm vào một câu cụ thể.
2. **Không học từ dữ liệu.** Bộ tiền xử lý stateless nên áp dụng giống hệt cho train, dev, test và bình luận mới khi triển khai.
3. **Dev không được đọc.** Từ điển và luật chỉ được thiết kế dựa trên train. Dev chỉ dùng để *đo* tác dụng (ablation).
4. **Đo, không đoán.** Mỗi bước phải được chứng minh bằng ablation trên dev.

## Các bước

Bảng dưới liệt kê mọi bước đã cài đặt theo đúng thứ tự chạy. Chỉ có một profile `classic`, bật tất cả các bước, trừ stopword (14).

Tỉ lệ (%) là phần trăm bình luận train có xu hướng đó (`eda.text_pattern_report`).

| # | Bước | Luật | Động cơ (xu hướng trên train) |
|---|---|---|---|
| 0 | Xuống dòng → `.` | `\r\n` thành ranh giới câu | 11,5% bình luận dùng xuống dòng thay dấu câu; nếu không xử lý, bước gắn phủ định có thể vượt qua ranh giới câu. |
| 1 | Xóa URL/email/SĐT | regex | Hiếm, nhưng không mang thông tin khía cạnh. |
| 2 | Emoji → `emopos/emoneg/emoneu` | Bảng ghi đè + **suy luận từ tên Unicode** (`SMILING`, `CRYING`, `THUMBS DOWN`, ...) | 3,9% có emoji, mang cảm xúc mạnh. Gom về 3 nhãn để tổng quát hóa; luật theo tên Unicode xử lý được cả emoji chưa từng gặp. |
| 3 | Emoticon ASCII → nhãn cảm xúc | regex `:)) =)) :( T_T ^^ <3 :v` có lookaround tránh bắt nhầm giờ `8:30` | 2,1% bình luận. Phải chạy trước bước bỏ dấu câu. |
| 4 | Unicode NFC + chuẩn dấu thanh | `unicodedata` + `underthesea.text_normalize` (`hoà`→`hòa`) | 3,1% không ở dạng NFC; hai kiểu bỏ dấu khiến một từ thành nhiều token. Sau đó gắn lại đơn vị bị tách (`250 k`→`250k`). |
| 5 | Chữ thường | | |
| 6 | Tiếng cười/khóc viết bằng chữ | `hahaha hihi kkkk` → `emopos`; `huhu hic haizzz` → `emoneg` | Chạy **trước** bước rút gọn ký tự lặp, nếu không `kkkk` → `k` → `không`. |
| 7 | Số + đơn vị → token theo loại | `5000mah`→`numbattery`, `7tr5/250k/7.990.000đ`→`numprice`, `128gb`→`numstorage`, `48mp`→`numcamera`, `90hz`→`numfreq`, `18w`→`numpower`, `88%`→`numpercent`, `2 tháng`→`numduration`, `8/3/2019`→`numdate` | 62,5% bình luận có chữ số; *loại* đơn vị gắn chặt với khía cạnh, còn giá trị cụ thể thì không quan trọng. Mã model (`a51`, `s20`) không bị đụng tới. Pipeline của refs xóa hết số nên mất tín hiệu này. |
| 8 | Rút gọn ký tự kéo dài | lặp ≥ 3 → 1; lặp đôi ở cuối chỉ rút khi token có dấu tiếng Việt | 3,2% bình luận. Âm tiết tiếng Việt không bao giờ có 3 chữ giống nhau liên tiếp. Không rút `good`, `full`. |
| 9 | Tách dấu câu | `tốt,pin`→`tốt , pin`; gộp `!!!`; giữ `7.990.000` | 14,3% bình luận có dấu câu lặp; dấu câu dính liền từ làm teencode không khớp. |
| 10 | `-g` → `-ng` | âm tiết có dấu kết thúc bằng *nguyên âm + g* | Luật chính tả: tiếng Việt chỉ có `-ng`, nên `khôg`, `nhữg`, `đag` luôn là viết thiếu. Không áp dụng cho từ ASCII như `lag`. |
| 11 | Teencode/viết tắt | Từ điển 345 mục, khớp **nguyên token** | `ko/k/dc/đt/nv/sp/tgdd/bin/wf/...`. Token không phải âm tiết chuẩn mà chưa có trong từ điển được liệt kê bằng `eda.nonstandard_token_report` để rà soát. |
| 12 | Gắn phủ định | `không mượt`→`không_mượt`, bỏ qua từ đệm (`không được mượt`→`không_được_mượt`). **Dừng tại dấu câu** (ranh giới câu) và tại một từ phủ định khác | Phủ định đảo cảm xúc; với túi từ, `không_mượt` cần là đặc trưng riêng (Das & Chen 2001; Pang et al. 2002). Phải chạy khi dấu câu còn là token riêng: `"... em ko ? ) . - cam thì"` cho `ko` đứng một mình, không thành `không_cam`. |
| 13 | Xóa dấu câu | | Xóa sau bước gắn phủ định, để bước đó còn thấy ranh giới câu. |
| 14 | Stopword (tắt mặc định) | danh sách refs **trừ** `protected_words.txt` | Chỉ bật nếu ablation cho thấy có lợi. TF-IDF với `max_df` đã tự giảm trọng số từ phổ biến. |

## Profile tiền xử lý

| Profile | Dùng cho | Khác biệt |
|---|---|---|
| `classic` | Mọi biểu diễn (BoW, TF-IDF, Word2Vec, FastText, Doc2Vec) | Đầy đủ các bước trên, mức âm tiết (không tách từ); bi-gram bắt cụm từ ghép |

## Thay đổi so với tài nguyên trong `refs/`

| Tài nguyên refs | Vấn đề phát hiện | Xử lý |
|---|---|---|
| `wrong-word.txt` (11.851 từ) | Chứa `pin`, `ok`, `game`, `camera`, `wifi`, `nhanh`, `mua`, `loa`, `sim`, `chip`, ... Xóa các từ này sẽ phá hỏng việc phát hiện khía cạnh | **Không dùng** |
| `vietnamese-stopwords.txt` | Chứa `rất`, `quá`, `chưa`, `nhưng`, `được`, `khá`, `nhanh`, `mạnh`, `lâu`, `cao`, `thấp`, ... | Dùng sau khi trừ `protected_words.txt`; tắt mặc định |
| `teencode.txt` (398 mục) | Mục sai/mơ hồ trong domain: `note`→`chú ý` (Galaxy Note), `face`→`facebook` (face ID), `c`→`chị` (USB-C), `a`→`anh`; `hic`→`hích`, `plz`→`pờ ly`; khai triển tục tĩu | Giữ lại 230 mục của refs (bỏ các mục lỗi và các mục trùng với phần bổ sung), thêm 115 mục văn phong review điện thoại (đầu file `teencode.tsv`); tổng 345 |
| `emojicon.txt` (43 emoji → từ) | Chỉ phủ 43 emoji, gán từ không nhất quán (`😂`→`cười`, `😓`→`bùn`) | Thay bằng 3 nhãn cảm xúc, bảng ghi đè và luật suy luận theo tên Unicode |
| Pipeline refs | Xóa toàn bộ số và dấu câu trước khi gắn phủ định, lọc theo POS | Thay bằng token số theo loại; xóa dấu câu sau khi gắn phủ định; không lọc POS (mất danh từ khía cạnh) |

## Hạn chế đã biết

- Bình luận gần như không dấu (5/7.786 ở train) chưa được khôi phục dấu. TF-IDF ký tự và FastText phần nào bù lại.
- `4g` mơ hồ (mạng 4G hay RAM 4GB) nên được giữ thành token riêng `numg` để mô hình dựa vào ngữ cảnh.
- Đơn vị viết tắt không có trong luật số (vd `10n` = 10 ngày) thành `num n`.

## Lịch sử thay đổi logic (`PREPROCESS_VERSION`)

Số phiên bản nằm trong khóa cache và `run_id`, nên khi logic đổi thì cache tiền xử lý và kết quả cũ tự bị bỏ qua.

| Phiên bản | Thay đổi |
|---|---|
| 1 | Bản đầu |
| 2 | Gắn phủ định không còn vượt qua dấu câu/ranh giới câu (sửa lỗi `ko ? . cam` → `không_cam`) và không nối phủ định kép (`không không tốt` → `không`, `không_tốt`) |
| 3 | Bỏ bước tách từ và các profile `classic_seg`/`transformer_seg`/`transformer_raw`; chỉ còn `classic` |
