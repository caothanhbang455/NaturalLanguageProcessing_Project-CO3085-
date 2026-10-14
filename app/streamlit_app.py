"""Demo Streamlit: phân tích đánh giá sản phẩm theo khía cạnh.

Chạy:
    streamlit run app/streamlit_app.py

Yêu cầu đã có mô hình tại ``outputs/models/best_ml_pipeline.joblib`` (tạo ở mục
"Đóng gói & triển khai" của notebook ``notebooks/absa_pipeline.ipynb``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from absa.config import PATHS  # noqa: E402
from absa.inference import ABSAPipeline  # noqa: E402
from absa.visualization.result_plots import plot_product_summary  # noqa: E402

DEFAULT_MODEL = PATHS.models_dir / "best_ml_pipeline.joblib"
EXAMPLES = (
    "Pin trâu, chơi game mượt nhưng camera chụp đêm hơi mờ. Nhân viên tư vấn nhiệt tình.\n"
    "Máy nóng quá, sạc lâu, giá thì hơi cao so với cấu hình :(\n"
    "Màn hình đẹp, thiết kế sang trọng, rất hài lòng 👍"
)


@st.cache_resource
def load_pipeline(path: str) -> ABSAPipeline:
    """Nạp pipeline một lần cho mọi phiên làm việc."""
    return ABSAPipeline.load(path)


def main() -> None:
    """Giao diện chính."""
    st.set_page_config(page_title="ABSA – Đánh giá điện thoại", layout="wide")
    st.title("Phân tích cảm xúc theo khía cạnh cho đánh giá điện thoại")

    model_path = st.sidebar.text_input("Đường dẫn mô hình", str(DEFAULT_MODEL))
    if not Path(model_path).exists():
        st.error(f"Không tìm thấy mô hình: {model_path}. Hãy chạy notebook để tạo mô hình.")
        return
    pipeline = load_pipeline(model_path)
    with st.sidebar.expander("Thông tin mô hình"):
        st.json(pipeline.metadata)

    source = st.radio("Nguồn dữ liệu", ["Nhập bình luận", "Tải file CSV"], horizontal=True)
    if source == "Nhập bình luận":
        raw = st.text_area("Mỗi dòng một bình luận", EXAMPLES, height=160)
        comments = [line.strip() for line in raw.splitlines() if line.strip()]
    else:
        uploaded = st.file_uploader("CSV có cột 'comment'", type="csv")
        comments = (
            pd.read_csv(uploaded)["comment"].dropna().astype(str).tolist() if uploaded else []
        )

    if not comments or not st.button("Phân tích", type="primary"):
        return

    predictions = pipeline.predict_frame(comments)
    summary = pipeline.summarize(comments)

    st.subheader("Tổng hợp theo khía cạnh")
    left, right = st.columns([3, 2])
    with left:
        st.pyplot(plot_product_summary(summary))
    with right:
        strengths = summary[(summary["mentions"] > 0) & (summary["net_score"] > 0)]
        weaknesses = summary[(summary["mentions"] > 0) & (summary["net_score"] < 0)]
        st.markdown("**Điểm mạnh:** " + (", ".join(strengths.index) or "—"))
        st.markdown("**Điểm yếu:** " + (", ".join(weaknesses.index) or "—"))
        st.dataframe(summary, use_container_width=True)

    st.subheader("Chi tiết từng bình luận")
    st.dataframe(predictions, use_container_width=True)
    if pipeline.metadata.get("strategy") == "multilabel":
        with st.expander("31 nhãn nhị phân thô (chưa xử lý xung đột)"):
            st.dataframe(pipeline.predict_pairs(comments), use_container_width=True)
    st.download_button(
        "Tải kết quả CSV",
        predictions.to_csv(index=False).encode("utf-8-sig"),
        "absa_predictions.csv",
    )


if __name__ == "__main__":
    main()
