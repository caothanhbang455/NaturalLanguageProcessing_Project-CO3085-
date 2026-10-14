"""Gói ``absa``: hệ thống Aspect-Based Sentiment Analysis cho bình luận điện thoại (UIT-ViSFD).

Cấu trúc gói đi theo đúng vòng đời một dự án ML:

* ``absa.data``           – đọc dữ liệu, kiểm tra schema, mã hóa nhãn.
* ``absa.eda``            – các hàm thống kê phục vụ EDA (chỉ trả về số liệu/bảng).
* ``absa.preprocessing``  – chuẩn hóa văn bản tiếng Việt theo luật tổng quát.
* ``absa.features``       – N cách biểu diễn văn bản (sparse, static, contextual).
* ``absa.models``         – M mô hình ML truyền thống và các kiến trúc DL.
* ``absa.training``       – chạy lưới thí nghiệm M x N, tinh chỉnh, cross-validation.
* ``absa.evaluation``     – metric ABSA và phân tích lỗi.
* ``absa.visualization``  – toàn bộ hàm vẽ biểu đồ.
* ``absa.inference``      – đóng gói pipeline cuối cùng để dự đoán và triển khai.
"""

__version__ = "0.1.0"
