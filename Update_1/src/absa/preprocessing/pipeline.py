"""Ghép các bước tiền xử lý thành pipeline có cấu hình, có cache và giải thích được.

Ba thành phần chính:

* :class:`PreprocessConfig` – bật/tắt từng bước; mỗi cấu hình có ``fingerprint``
  dùng làm khóa cache.
* :class:`TextPreprocessor` – transformer tương thích scikit-learn: ``transform``
  một danh sách câu, ``trace`` một câu để xem tác động của từng bước.
* :class:`PreparedCorpus` – giữ văn bản thô của các tập và tính lười biếng (có cache
  ra đĩa) văn bản đã xử lý theo từng *profile*.

Chỉ còn **một** profile ``classic``: chuẩn hóa mạnh, gắn phủ định, gom số và emoji
thành token loại, **giữ mức âm tiết** (không tách từ). Bi-gram âm tiết đã bắt được từ
ghép (``màn hình``), còn tách từ trên văn bản mạng xã hội lại làm từ vựng phình gấp đôi,
nên không tách từ là lựa chọn mặc định.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from absa.config import PATHS
from absa.preprocessing import normalizers as nz
from absa.preprocessing.lexicons import Lexicons, load_lexicons
from absa.preprocessing.tokenization import attach_negation, remove_stopwords
from absa.utils.io import ensure_dir, hash_object, hash_texts, load_json, save_json
from absa.utils.logging import get_logger

logger = get_logger(__name__)
Step = tuple[str, Callable[[str], str]]

# Tăng số này mỗi khi đổi *logic* của một bước (không đổi cấu hình): nó nằm trong
# ``PreprocessConfig.fingerprint`` nên cache tiền xử lý và kết quả thí nghiệm cũ tự bị bỏ qua.
#   v2: gắn phủ định không vượt qua dấu câu/ranh giới câu (không_cam), không nối phủ định kép.
#   v3: bỏ tách từ và các profile classic_seg/transformer_seg/transformer_raw.
PREPROCESS_VERSION = 3


@dataclass(frozen=True)
class PreprocessConfig:
    """Cấu hình bật/tắt từng bước tiền xử lý.

    Attributes:
        normalize_unicode: NFC + chuẩn vị trí dấu thanh.
        remove_urls: Xóa URL/email/số điện thoại.
        emoji_mode: ``"tag" | "word" | "remove" | "keep"`` cho emoji Unicode.
        emoticon_mode: Như ``emoji_mode`` nhưng cho emoticon ASCII (``:))``, ``T_T``) và
            chuỗi cười/khóc viết bằng chữ (``hahaha``, ``huhu``).
        lowercase: Đưa về chữ thường.
        normalize_numbers: Gom số + đơn vị thành token loại (``numbattery``...).
        reduce_elongation: Rút gọn ký tự kéo dài.
        remove_punctuation: ``True`` bỏ dấu câu; ``False`` chỉ chuẩn hóa dấu câu.
        restore_final_ng: Sửa ``khôg`` -> ``không``.
        expand_teencode: Thay teencode/viết tắt theo từ điển.
        negation_window: Số token gộp sau từ phủ định (``0`` = tắt).
        remove_stopwords: Bỏ stopword (đã trừ danh sách bảo vệ).
    """

    normalize_unicode: bool = True
    remove_urls: bool = True
    emoji_mode: str = "tag"
    emoticon_mode: str = "tag"
    lowercase: bool = True
    normalize_numbers: bool = True
    reduce_elongation: bool = True
    remove_punctuation: bool = True
    restore_final_ng: bool = True
    expand_teencode: bool = True
    negation_window: int = 1
    remove_stopwords: bool = False

    def replace(self, **changes) -> PreprocessConfig:
        """Tạo bản sao với một số trường thay đổi (cấu hình là bất biến)."""
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        """Chuyển sang dict (để log hoặc lưu kèm mô hình)."""
        return dataclasses.asdict(self)

    @property
    def fingerprint(self) -> str:
        """Mã băm của cấu hình và phiên bản logic – dùng làm khóa cache."""
        return hash_object([self.to_dict(), PREPROCESS_VERSION])


PREPROCESS_PROFILES: dict[str, PreprocessConfig] = {
    "classic": PreprocessConfig(),
}

# Cấu hình "tối thiểu" dùng làm mốc so sánh trong ablation.
MINIMAL_CONFIG = PreprocessConfig(
    normalize_unicode=False,
    remove_urls=False,
    emoji_mode="keep",
    emoticon_mode="keep",
    normalize_numbers=False,
    reduce_elongation=False,
    remove_punctuation=True,
    restore_final_ng=False,
    expand_teencode=False,
    negation_window=0,
)


def resolve_config(config: str | PreprocessConfig | None) -> PreprocessConfig:
    """Nhận tên profile hoặc đối tượng cấu hình, trả về :class:`PreprocessConfig`."""
    if config is None:
        return PREPROCESS_PROFILES["classic"]
    if isinstance(config, PreprocessConfig):
        return config
    if config not in PREPROCESS_PROFILES:
        raise KeyError(f"Profile '{config}' không tồn tại. Có: {list(PREPROCESS_PROFILES)}")
    return PREPROCESS_PROFILES[config]


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Pipeline tiền xử lý tiếng Việt, tương thích scikit-learn.

    Bộ tiền xử lý **không học gì từ dữ liệu** (stateless): mọi luật và từ điển đều cố
    định, nên áp dụng giống hệt cho train, dev, test và bình luận mới khi triển khai.

    Args:
        config: Tên profile (``"classic"``...) hoặc :class:`PreprocessConfig`.
        lexicon_dir: Thư mục từ điển, mặc định ``resources/lexicons``.

    Example:
        >>> pre = TextPreprocessor("classic")
        >>> pre.process("Pin trâu 5000mAh 👍, ko lag")          # doctest: +SKIP
        'pin trâu numbattery emopos không_lag'
    """

    def __init__(
        self, config: str | PreprocessConfig | None = "classic", lexicon_dir: str | None = None
    ) -> None:
        self.config = config
        self.lexicon_dir = lexicon_dir

    # ----------------------------------------------------------------- build --
    @property
    def config_(self) -> PreprocessConfig:
        """Cấu hình đã được phân giải."""
        return resolve_config(self.config)

    @property
    def lexicons_(self) -> Lexicons:
        """Từ điển đang dùng."""
        return load_lexicons(self.lexicon_dir)

    def _build_steps(self) -> list[Step]:
        cfg, lex = self.config_, self.lexicons_
        emoji_mode, emoticon_mode = cfg.emoji_mode, cfg.emoticon_mode
        steps: list[Step] = [("line_breaks", nz.mark_line_breaks)]
        if cfg.remove_urls:
            steps.append(("remove_urls", nz.remove_urls_emails))
        if emoji_mode != "keep":
            steps.append(
                ("emoji", lambda t: nz.replace_emojis(t, emoji_mode, lex.emoji_sentiment))
            )
        if emoticon_mode != "keep":
            steps.append(("emoticon", lambda t: nz.replace_emoticons(t, emoticon_mode)))
        if cfg.normalize_unicode:
            steps.append(("unicode", lambda t: nz.rejoin_number_units(nz.normalize_unicode(t))))
        if cfg.lowercase:
            steps.append(("lowercase", str.lower))
        if emoticon_mode != "keep":
            steps.append(("laughter", lambda t: nz.replace_laughter(t, emoticon_mode)))
        if cfg.normalize_numbers:
            steps.append(("numbers", nz.normalize_numbers))
        if cfg.reduce_elongation:
            steps.append(("elongation", nz.reduce_elongation))
        # Dấu câu được *tách* ra trước (để teencode khớp "k,"), chỉ *xóa* sau khi gắn phủ
        # định – tránh ghép nhầm từ/phủ định của hai câu liền nhau.
        steps.append(("punctuation", lambda t: nz.normalize_punctuation(t, remove=False)))
        if cfg.restore_final_ng:
            steps.append(("final_ng", nz.restore_final_ng))
        if cfg.expand_teencode:
            steps.append(("teencode", nz.TeencodeNormalizer(lex.teencode)))
        steps.append(("whitespace", nz.normalize_whitespace))
        # Gắn phủ định phải chạy KHI DẤU CÂU CÒN LÀ TOKEN RIÊNG: dấu câu là ranh giới chặn phạm vi
        # phủ định ("... em ko ? ) . - cam thì" không được thành "ko_cam"). Chỉ sau đó mới xóa dấu.
        if cfg.negation_window > 0:
            steps.append(
                (
                    "negation",
                    lambda t: " ".join(
                        attach_negation(t.split(), lex.negations, cfg.negation_window)
                    ),
                )
            )
        if cfg.remove_punctuation:
            steps.append(
                (
                    "remove_punctuation",
                    lambda t: nz.normalize_whitespace(nz.normalize_punctuation(t, remove=True)),
                )
            )
        if cfg.remove_stopwords:
            steps.append(
                ("stopwords", lambda t: " ".join(remove_stopwords(t.split(), lex.stopwords)))
            )
        steps.append(("final_whitespace", nz.normalize_whitespace))
        return steps

    @property
    def steps_(self) -> list[Step]:
        """Danh sách ``(tên_bước, hàm)`` theo đúng thứ tự áp dụng (tạo lười biếng)."""
        if getattr(self, "_steps", None) is None:
            self._steps = self._build_steps()
        return self._steps

    # ------------------------------------------------------------- sklearn API --
    def fit(self, X: Iterable[str] | None = None, y=None) -> TextPreprocessor:
        """Không học gì; có mặt để tương thích ``Pipeline`` của scikit-learn."""
        return self

    def process(self, text: str) -> str:
        """Tiền xử lý một bình luận."""
        text = "" if text is None else str(text)
        for _, step in self.steps_:
            text = step(text)
        return text

    def transform(self, X: Iterable[str]) -> list[str]:
        """Tiền xử lý một danh sách bình luận."""
        return [self.process(text) for text in X]

    def trace(self, text: str) -> pd.DataFrame:
        """Hiển thị đầu ra sau **từng bước** – dùng để minh họa trong notebook/báo cáo."""
        rows = [{"step": "input", "output": text, "changed": False}]
        for name, step in self.steps_:
            new_text = step(text)
            rows.append({"step": name, "output": new_text, "changed": new_text != text})
            text = new_text
        return pd.DataFrame(rows)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_steps"] = None  # lambda không pickle được; sẽ dựng lại khi cần.
        return state


def preprocess_corpus(
    texts: Sequence[str],
    config: str | PreprocessConfig | None = "classic",
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    show_progress: bool = False,
) -> list[str]:
    """Tiền xử lý cả một tập văn bản, có cache ra đĩa.

    Kết quả được lưu theo khóa ``(cấu hình, nội dung từ điển, nội dung văn bản)``.
    Đổi bất kỳ thành phần nào thì cache tự động bị bỏ qua.
    """
    preprocessor = TextPreprocessor(config)
    cfg = preprocessor.config_
    key = hash_object(
        [cfg.fingerprint, preprocessor.lexicons_.fingerprint, hash_texts(texts), len(texts)]
    )
    cache_path = Path(cache_dir or PATHS.cache_dir / "preprocessed") / f"{key}.json"
    if use_cache and cache_path.exists():
        return load_json(cache_path)

    iterator: Iterable[str] = texts
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(texts, desc="preprocess", leave=False)
    processed = [preprocessor.process(t) for t in iterator]
    if use_cache:
        ensure_dir(cache_path.parent)
        save_json(processed, cache_path)
    return processed


class PreparedCorpus:
    """Kho văn bản của các tập, tính văn bản đã xử lý theo profile khi được yêu cầu.

    Args:
        raw_texts: ``{"train": [...], "dev": [...], "test": [...]}`` – văn bản thô.
        cache_dir: Thư mục cache cho :func:`preprocess_corpus`.

    Example:
        >>> corpus = PreparedCorpus({"train": train.texts, "dev": dev.texts})  # doctest: +SKIP
        >>> corpus.get("classic", "train")[:2]                                 # doctest: +SKIP
    """

    def __init__(
        self, raw_texts: Mapping[str, Sequence[str]], cache_dir: str | Path | None = None
    ) -> None:
        self.raw = {k: list(v) for k, v in raw_texts.items() if v is not None}
        self.cache_dir = cache_dir
        self._memory: dict[tuple[str, str], list[str]] = {}

    @property
    def splits(self) -> list[str]:
        """Tên các tập đang có."""
        return list(self.raw)

    def get(self, profile: str | PreprocessConfig, split: str) -> list[str]:
        """Văn bản của ``split`` sau khi xử lý bằng ``profile`` (hoặc thô nếu ``"raw"``)."""
        if split not in self.raw:
            raise KeyError(f"Không có tập '{split}'. Có: {self.splits}")
        if profile == "raw":
            return self.raw[split]
        cfg = resolve_config(profile)
        key = (cfg.fingerprint, split)
        if key not in self._memory:
            logger.info("Tiền xử lý tập '%s' (%d câu)...", split, len(self.raw[split]))
            self._memory[key] = preprocess_corpus(
                self.raw[split], cfg, cache_dir=self.cache_dir, show_progress=True
            )
        return self._memory[key]

    def concat(self, profile: str | PreprocessConfig, splits: Sequence[str]) -> list[str]:
        """Nối văn bản đã xử lý của nhiều tập (dùng khi huấn luyện lại trên train+dev)."""
        return [t for s in splits for t in self.get(profile, s)]


def clean_dataframe(
    df: pd.DataFrame,
    config: str | PreprocessConfig | None = "classic",
    text_col: str = "comment",
    clean_col: str = "clean_comment",
    drop_cols: Sequence[str] = ("n_star",),
    show_progress: bool = False,
) -> pd.DataFrame:
    """Áp dụng toàn bộ quy trình làm sạch dữ liệu theo chuẩn:

    * Step 0: Xóa cột không dùng (mặc định ``n_star``).
    * Steps 1–9: Chuẩn hóa toàn bộ nội dung văn bản trong cột ``comment``
      theo pipeline 9 bước (ranh giới câu, URL, emoji/emoticon, unicode,
      lowercase, rút gọn lặp, số kèm đơn vị, -g/-ng & teencode, phủ định,
      xóa dấu câu).
    * Step 10: Đổi tên cột ``comment`` thành ``clean_comment``, giữ nguyên
      các cột còn lại (``index``, ``date_time``, ``label``...).
    """
    out = df.copy()

    # Step 0: Xóa các cột không mong muốn (n_star và Unnamed)
    to_drop = [c for c in drop_cols if c in out.columns]
    unnamed = [c for c in out.columns if str(c).startswith("Unnamed")]
    all_drop = list(dict.fromkeys(to_drop + unnamed))
    if all_drop:
        out = out.drop(columns=all_drop)

    # Steps 1-9: Tiền xử lý giá trị text trong cột comment
    if text_col in out.columns:
        preprocessor = TextPreprocessor(config)
        raw_series = out[text_col].fillna("").astype(str)
        if show_progress:
            from tqdm.auto import tqdm

            cleaned_texts = [
                preprocessor.process(t) for t in tqdm(raw_series, desc=f"Cleaning {text_col}")
            ]
        else:
            cleaned_texts = preprocessor.transform(raw_series)

        out[text_col] = cleaned_texts

        # Step 10: Đổi tên cột comment thành clean_comment
        out = out.rename(columns={text_col: clean_col})

    return out


def clean_csv(
    input_path: str | Path,
    output_path: str | Path,
    config: str | PreprocessConfig | None = "classic",
    text_col: str = "comment",
    clean_col: str = "clean_comment",
    drop_cols: Sequence[str] = ("n_star",),
    show_progress: bool = True,
) -> pd.DataFrame:
    """Đọc CSV gốc, thực hiện làm sạch và lưu file kết quả."""
    in_p, out_p = Path(input_path), Path(output_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {in_p}")

    logger.info("Đang đọc %s...", in_p)
    df = pd.read_csv(in_p)
    df_clean = clean_dataframe(
        df,
        config=config,
        text_col=text_col,
        clean_col=clean_col,
        drop_cols=drop_cols,
        show_progress=show_progress,
    )

    ensure_dir(out_p.parent)
    df_clean.to_csv(out_p, index=False, encoding="utf-8")
    logger.info("Đã lưu dữ liệu làm sạch tại %s (%d dòng)", out_p, len(df_clean))
    return df_clean

