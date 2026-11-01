"""=============================================================================
PIPELINE LÀM SẠCH DỮ LIỆU UIT-ViSFD (TẤT CẢ TRONG 1 FILE DUY NHẤT)
=============================================================================
Thực hiện trọn vẹn quy trình 11 bước (Step 0 đến Step 10) và 5 RULE:
  - Step 0 : Xóa cột 'n_star' (và các cột 'Unnamed' rác).
  - Step 1 : Xuống dòng '\\n' -> ' . ' (tạo ranh giới câu) + Xóa URL, email, SĐT.
  - Step 2 : Gom Emoji, Emoticon, chuỗi cười/khóc -> nhãn 'emopos', 'emoneg', 'emoneu'.
  - Step 3 : Chuẩn hóa Unicode NFC, chuẩn dấu thanh tiếng Việt, gắn lại đơn vị ('250 k' -> '250k').
  - Step 4 : Chuyển thành chữ thường (lowercase).
  - Step 5 : Rút gọn ký tự lặp kéo dài ('đẹppppp' -> 'đẹp', 'rồiii' -> 'rồi', giữ 'good', 'full').
  - Step 6 : Gom số đo kèm đơn vị -> token đại lượng ('5000mah' -> 'numbattery', '7tr5' -> 'numprice'...).
  - Step 7 : Tách dấu câu + Sửa lỗi '-g' -> '-ng' ('khôg' -> 'không') + Khai triển Teencode.
  - Step 8 : Gắn từ phủ định ('không_mượt', 'không_được_mượt') dừng tại dấu câu và phủ định kép.
  - Step 9 : Xóa sạch dấu câu còn lại + chuẩn hóa khoảng trắng.
  - Step 10: Đổi tên cột 'comment' thành 'clean_comment', giữ nguyên 'index', 'date_time', 'label'.

Cách chạy độc lập từ dòng lệnh:
    python clean_data_all_in_one.py
    python clean_data_all_in_one.py --input UIT-ViSFD/Train.csv --output UIT-ViSFD/Train_clean.csv
=============================================================================
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import unicodedata
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# Thống nhất UTF-8 cho console Windows tránh UnicodeEncodeError
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd

# =============================================================================
# 1. HẰNG SỐ VÀ REGEX CHUẨN HÓA
# =============================================================================

# Ký tự vô hình
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff\ufe0e\ufe0f]")

# URL, Email, Số điện thoại
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?84|0)\d{9,10}(?!\d)")
_LINE_BREAK_RE = re.compile(r"\s*[\r\n]+\s*")

# Nhãn cảm xúc
SENTIMENT_TAGS: dict[str, str] = {"pos": "emopos", "neg": "emoneg", "neu": "emoneu"}

# Emoji pattern
_EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"  # emoji, biểu tượng, cờ, skin tone
    "\u2600-\u27bf"          # biểu tượng linh tinh, dingbats (❤ ✌ ☺ ★)
    "\u2b00-\u2bff"          # ⭐ và mũi tên
    "\u2300-\u23ff"          # ⌚ ⏰
    "]"
)

_NEG_STRONG = ("BROKEN HEART", "THUMBS DOWN", "MIDDLE FINGER", "PILE OF POO")
_POS_KEYWORDS = (
    "SMIL", "GRIN", "LAUGH", "JOY", "HEART", "KISS", "THUMBS UP", "OK HAND",
    "CLAPPING", "STAR", "PARTY", "SPARKL", "HUGGING", "WINK", "SAVOURING",
    "SUNGLASSES", "HUNDRED", "TROPHY", "MEDAL", "RAISING BOTH HANDS", "VICTORY",
    "FLEXED", "BLOWING A KISS"
)
_NEG_KEYWORDS = (
    "CRY", "SOB", "ANGRY", "POUTING", "RAGE", "FROWN", "DISAPPOINT", "WEARY",
    "TIRED", "CONFOUNDED", "PERSEVER", "ANGUISH", "FEARFUL", "SCREAM", "NAUSEA",
    "VOMIT", "UNAMUSED", "EXPRESSIONLESS", "SKULL", "WORRIED", "CONFUSED",
    "PENSIVE", "DOWNCAST", "CROSS MARK", "NO ENTRY", "FACE PALM", "ROLLING EYES",
    "GRIMAC", "HOT FACE", "STEAM FROM NOSE", "SYMBOLS ON MOUTH"
)
_NEU_KEYWORDS = (
    "THINKING", "NEUTRAL FACE", "WITHOUT MOUTH", "HUSHED", "ASTONISHED",
    "FLUSHED", "OPEN MOUTH", "PLEADING", "FOLDED HANDS"
)

# Emoticon ASCII
_EMOTICON_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<![\w:/])[:;=]['\-^]?[)\]}>dDpP3]+(?![\w/])"), "pos"),
    (re.compile(r"(?<!\w)\^[_.\-o]?\^+(?!\w)"), "pos"),
    (re.compile(r"<3+"), "pos"),
    (re.compile(r"(?<!\w)[xX]D+(?!\w)"), "pos"),
    (re.compile(r"(?<![\w:/])[:;=]['\-]?[(\[{<]+(?![\w/])"), "neg"),
    (re.compile(r"(?<!\w)[tT][_.\-][tT](?!\w)"), "neg"),
    (re.compile(r"(?<!\w)-[_.]-(?!\w)"), "neg"),
    (re.compile(r"(?<!\w)>[_.]?<(?!\w)"), "neg"),
    (re.compile(r"(?<![\w:/])[:=]['\-]?[/\\|](?![\w/])"), "neg"),
    (re.compile(r"(?<![\w:/])[:=]['\-]?[vV]+(?!\w)"), "neu"),
    (re.compile(r"(?<![\w:/\d])[:=]['\-]?[oO0](?!\w)"), "neu"),
)

# Chuỗi cười / khóc bằng chữ
_LAUGH_RE = re.compile(
    r"(?<!\w)(?:(?:ha|he|hi|hj|hô|hì|hí|há|hà|hé|hè|hê|kha|khà){2,}[a-zàáèéìíêôj]*|k{3,})(?!\w)"
)
_SIGH_CRY_RE = re.compile(r"(?<!\w)(?:(?:hu){2,}u*|h[ij][cx]+|hai?z{1,}|haz{2,}i*)(?!\w)")
_REPEATED_TAG_RE = re.compile(r"\b(emopos|emoneg|emoneu)(?:\s+\1\b)+")

# Số kèm đơn vị phần cứng & giá
_NUM = r"\d+(?:[.,]\d+)*"
_START = r"(?<![\w.,/])"
_END = r"(?!\w)"
NUMBER_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"{_START}\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?{_END}"), "numdate"),
    (re.compile(rf"{_START}\d{{1,2}}\s*[:h]\s*\d{{2}}{_END}"), "numtime"),
    (re.compile(rf"{_START}{_NUM}\s*(?:tr\d*|triệu|trieu|củ|nghìn|ngàn|ngan|nghin|đồng|đ|vnđ|vnd){_END}"), "numprice"),
    (re.compile(rf"{_START}\d{{2,}}(?:[.,]\d+)*\s*k{_END}"), "numprice"),
    (re.compile(rf"{_START}\d{{1,3}}(?:[.,]\d{{3}}){{2,}}{_END}"), "numprice"),
    (re.compile(rf"{_START}{_NUM}\s*(?:mah|mha|mh){_END}"), "numbattery"),
    (re.compile(rf"{_START}{_NUM}\s*(?:gb|tb|mb|gig){_END}"), "numstorage"),
    (re.compile(rf"{_START}\d{{2,}}\s*g{_END}"), "numstorage"),
    (re.compile(rf"{_START}\d\s*g{_END}"), "numg"),
    (re.compile(rf"{_START}{_NUM}\s*(?:mp|mpx|megapixel|chấm){_END}"), "numcamera"),
    (re.compile(rf"{_START}{_NUM}\s*(?:hz|ghz|fps){_END}"), "numfreq"),
    (re.compile(rf"{_START}{_NUM}\s*w{_END}"), "numpower"),
    (re.compile(rf"{_START}{_NUM}\s*%"), "numpercent"),
    (re.compile(rf"{_START}{_NUM}\s*(?:inch|inches|\"|”)"), "numsize"),
    (re.compile(rf"{_START}{_NUM}\s*(?:giờ|tiếng|h|phút|p|giây|s|ngày|tuần|tháng|năm|th){_END}"), "numduration"),
    (re.compile(rf"{_START}{_NUM}{_END}"), "num"),
)
_SPLIT_UNIT_RE = re.compile(
    r"(\d)\s+(?=(?:k|tr\d*|triệu|củ|đ|vnđ|vnd|mah|mha|mh|gb|tb|mb|g|mp|mpx|hz|ghz|fps|w|%|inch|h|p|s)(?!\w))",
    re.IGNORECASE,
)

# Ký tự kéo dài
_REPEAT3_RE = re.compile(r"([^\W\d_])\1{2,}")
_FINAL_DOUBLE_RE = re.compile(r"([^\W\d_])\1$")

# Dấu câu
_PUNCT_RE = re.compile(r"[^\w\s]")
_PUNCT_SPLIT_RE = re.compile(r"(?<!\d)[^\w\s]|[^\w\s](?!\d)")
_REPEATED_PUNCT_RE = re.compile(r"([^\w\s])\1+")

# Lỗi thiếu 'n' trước 'g'
_FINAL_VOWEL_G_RE = re.compile(r"[aeiouy]g$")

# Từ đệm sau phủ định
NEGATION_FILLERS: frozenset[str] = frozenset(
    {"được", "hề", "có", "phải", "bị", "quá", "lắm", "mấy", "thể", "còn", "thấy"}
)


# =============================================================================
# 2. CÁC HÀM XỬ LÝ TEXT CON
# =============================================================================

def mark_line_breaks(text: str) -> str:
    """Đổi xuống dòng thành dấu chấm ranh giới câu."""
    return _LINE_BREAK_RE.sub(" . ", text.strip())


def remove_urls_emails(text: str) -> str:
    """Xóa URL, email và số điện thoại."""
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    return _PHONE_RE.sub(" ", text)


def emoji_category(char: str, overrides: Mapping[str, str] | None = None) -> str | None:
    """Xác định nhóm cảm xúc (pos/neg/neu) của emoji."""
    if overrides and char in overrides:
        return overrides[char]
    name = unicodedata.name(char, "")
    if not name:
        return None
    if any(k in name for k in _NEG_STRONG):
        return "neg"
    if any(k in name for k in _POS_KEYWORDS):
        return "pos"
    if any(k in name for k in _NEG_KEYWORDS):
        return "neg"
    if any(k in name for k in _NEU_KEYWORDS):
        return "neu"
    return None


def collapse_repeated_tags(text: str) -> str:
    """Gộp nhãn lặp 'emopos emopos' -> 'emopos'."""
    return _REPEATED_TAG_RE.sub(r"\1", text)


def replace_emojis(text: str, overrides: Mapping[str, str] | None = None) -> str:
    """Thay thế emoji bằng tag cảm xúc."""
    def _repl(match: re.Match[str]) -> str:
        cat = emoji_category(match.group(0), overrides)
        tag = SENTIMENT_TAGS.get(cat, "") if cat else ""
        return f" {tag} " if tag else " "
    return collapse_repeated_tags(_EMOJI_RE.sub(_repl, text))


def replace_emoticons(text: str) -> str:
    """Thay thế emoticon ASCII (:)) T_T <3)."""
    for pattern, category in _EMOTICON_RULES:
        tag = SENTIMENT_TAGS[category]
        text = pattern.sub(f" {tag} ", text)
    return collapse_repeated_tags(text)


def replace_laughter(text: str) -> str:
    """Thay thế chuỗi cười/khóc/thở dài bằng chữ ('hahaha', 'kkkk', 'huhu', 'haizzz')."""
    text = _LAUGH_RE.sub(f" {SENTIMENT_TAGS['pos']} ", text)
    text = _SIGH_CRY_RE.sub(f" {SENTIMENT_TAGS['neg']} ", text)
    return collapse_repeated_tags(text)


def normalize_unicode(text: str) -> str:
    """Chuẩn hóa NFC và thống nhất kiểu bỏ dấu thanh tiếng Việt."""
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE_RE.sub("", text)
    try:
        from underthesea import text_normalize
        text = text_normalize(text)
    except ImportError:
        pass
    return text


def rejoin_number_units(text: str) -> str:
    """Gắn lại đơn vị bị tách: '250 k' -> '250k'."""
    return _SPLIT_UNIT_RE.sub(r"\1", text)


def normalize_numbers(text: str) -> str:
    """Gom số đo kèm đơn vị thành token loại (numbattery, numprice...)."""
    for pattern, token in NUMBER_RULES:
        text = pattern.sub(f" {token} ", text)
    return text


def reduce_elongation(text: str) -> str:
    """Rút gọn ký tự lặp kéo dài (đẹppppp -> đẹp, rồiii -> rồi)."""
    text = _REPEAT3_RE.sub(r"\1", text)
    tokens = []
    for token in text.split():
        if not token.isascii():
            token = _FINAL_DOUBLE_RE.sub(r"\1", token)
        tokens.append(token)
    return " ".join(tokens)


def normalize_punctuation(text: str, remove: bool = True) -> str:
    """remove=False: tách dấu câu thành token riêng; remove=True: xóa dấu câu."""
    if remove:
        return _PUNCT_RE.sub(" ", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    return _PUNCT_SPLIT_RE.sub(lambda m: f" {m.group(0)} ", text)


def restore_final_ng(text: str) -> str:
    """Sửa lỗi gõ thiếu 'n' trước 'g' cuối âm tiết có dấu (khôg -> không, đag -> đang)."""
    tokens = []
    for token in text.split():
        if not token.isascii():
            decomposed = unicodedata.normalize("NFD", token.replace("đ", "d").replace("Đ", "D"))
            stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
            if _FINAL_VOWEL_G_RE.search(stripped):
                token = token[:-1] + "ng"
        tokens.append(token)
    return " ".join(tokens)


class TeencodeNormalizer:
    """Khai triển teencode / từ viết tắt theo từ điển."""
    def __init__(self, mapping: Mapping[str, str]) -> None:
        self.single = {k.lower(): v for k, v in mapping.items() if " " not in k}
        phrases = sorted((k.lower() for k in mapping if " " in k), key=len, reverse=True)
        self.phrases = {k: mapping[k] for k in phrases}
        self._phrase_re = (
            re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, phrases)) + r")(?!\w)")
            if phrases else None
        )

    def __call__(self, text: str) -> str:
        if self._phrase_re is not None:
            text = self._phrase_re.sub(lambda m: self.phrases[m.group(0)], text)
        return " ".join(self.single.get(tok, tok) for tok in text.split())


def attach_negation(
    tokens: Sequence[str],
    negations: Collection[str],
    window: int = 1,
    fillers: Collection[str] = NEGATION_FILLERS,
) -> list[str]:
    """Gắn từ phủ định với từ theo sau, dừng lại khi gặp dấu câu / ranh giới câu."""
    if window <= 0:
        return list(tokens)
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        token = tokens[i]
        if token not in negations:
            out.append(token)
            i += 1
            continue
        span = [token]
        j, taken = i + 1, 0
        while j < n and taken < window:
            following = tokens[j]
            # Dừng nếu là dấu câu hoặc một từ phủ định khác
            if not any(ch.isalnum() for ch in following) or following in negations:
                break
            span.append(following)
            if following not in fillers:
                taken += 1
            j += 1
        out.append("_".join(span))
        i = j
    return out


def normalize_whitespace(text: str) -> str:
    """Gộp mọi khoảng trắng thành một dấu cách và cắt 2 đầu."""
    return " ".join(text.split())


# =============================================================================
# 3. QUẢN LÝ TỪ ĐIỂN (LEXICONS)
# =============================================================================

@dataclass
class LexiconBundle:
    teencode: dict[str, str]
    emoji_sentiment: dict[str, str]
    negations: frozenset[str]


def load_lexicons(lexicon_dir: Path | str | None = None) -> LexiconBundle:
    """Tìm và đọc các file từ điển trong resources/lexicons."""
    candidates = [
        Path(lexicon_dir) if lexicon_dir else None,
        Path(__file__).resolve().parent / "resources" / "lexicons",
        Path.cwd() / "resources" / "lexicons",
    ]
    lex_path = next((p for p in candidates if p and p.exists()), None)

    teencode: dict[str, str] = {}
    emoji_sentiment: dict[str, str] = {}
    negations: set[str] = {"không", "chẳng", "chả", "chưa", "đừng", "chớ", "ko", "k", "khg", "khong"}

    if lex_path:
        # 1. Teencode
        teen_file = lex_path / "teencode.tsv"
        if teen_file.exists():
            for line in teen_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "\t" in line:
                    k, v = line.split("\t", 1)
                    teencode.setdefault(k.strip().lower(), v.strip())

        # 2. Emoji sentiment override
        emoji_file = lex_path / "emoji_sentiment.tsv"
        if emoji_file.exists():
            for line in emoji_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "\t" in line:
                    k, v = line.split("\t", 1)
                    emoji_sentiment.setdefault(k.strip(), v.strip().lower())

        # 3. Negations
        neg_file = lex_path / "negation_words.txt"
        if neg_file.exists():
            for line in neg_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    negations.add(line.lower())

    return LexiconBundle(
        teencode=teencode,
        emoji_sentiment=emoji_sentiment,
        negations=frozenset(negations),
    )


# =============================================================================
# 4. HÀM CLEAN CHÍNH (ÁP DỤNG TRỌN VẸN 11 BƯỚC)
# =============================================================================

def clean_text(text: str, lexicons: LexiconBundle) -> str:
    """Áp dụng tuần tự 9 bước chuẩn hóa trên một chuỗi văn bản."""
    if not text or not isinstance(text, str):
        return ""

    # Bước 1: Ranh giới câu + Xóa URL/email/SĐT
    t = mark_line_breaks(text)
    t = remove_urls_emails(t)

    # Bước 2: Emoji + Emoticon
    t = replace_emojis(t, overrides=lexicons.emoji_sentiment)
    t = replace_emoticons(t)

    # Bước 3: Unicode NFC + Chuẩn dấu thanh + Gắn lại đơn vị số
    t = rejoin_number_units(normalize_unicode(t))

    # Bước 4: Lowercase
    t = t.lower()

    # Bước 5: Cười/khóc bằng chữ (chạy TRƯỚC khi rút gọn lặp)
    t = replace_laughter(t)

    # Bước 6: Số + đơn vị -> token loại khía cạnh
    t = normalize_numbers(t)

    # Bước 7: Rút gọn ký tự lặp
    t = reduce_elongation(t)

    # Bước 8: Tách dấu câu (để teencode khớp k, và làm ranh giới phủ định)
    t = normalize_punctuation(t, remove=False)

    # Bước 9: Sửa -g -> -ng và khai triển teencode
    t = restore_final_ng(t)
    t = TeencodeNormalizer(lexicons.teencode)(t)
    t = normalize_whitespace(t)

    # Bước 10: Gắn từ phủ định (khi dấu câu còn là token riêng)
    tokens = attach_negation(t.split(), lexicons.negations, window=1)
    t = " ".join(tokens)

    # Bước 11: Xóa dấu câu sạch sẽ và gộp khoảng trắng cuối
    t = normalize_punctuation(t, remove=True)
    t = normalize_whitespace(t)

    return t


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str = "comment",
    clean_col: str = "clean_comment",
    drop_cols: Sequence[str] = ("n_star",),
    lexicon_dir: Path | str | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Thực hiện đầy đủ pipeline trên DataFrame:

    1. Step 0 : Xóa cột n_star và các cột Unnamed rác.
    2. Steps 1-9: Clean toàn bộ nội dung cột 'comment'.
    3. Step 10: Đổi tên cột 'comment' thành 'clean_comment'.
    4. Giữ nguyên toàn bộ các cột còn lại: 'index', 'date_time', 'label'.
    """
    out = df.copy()

    # Step 0: Xóa cột n_star và các cột rác Unnamed
    to_drop = [c for c in drop_cols if c in out.columns]
    unnamed = [c for c in out.columns if str(c).startswith("Unnamed")]
    all_drop = list(dict.fromkeys(to_drop + unnamed))
    if all_drop:
        out = out.drop(columns=all_drop)

    # Steps 1-9: Tiền xử lý text trong cột comment
    if text_col in out.columns:
        lexicons = load_lexicons(lexicon_dir)
        raw_series = out[text_col].fillna("").astype(str)

        if show_progress:
            try:
                from tqdm.auto import tqdm
                cleaned = [clean_text(t, lexicons) for t in tqdm(raw_series, desc=f"Cleaning {text_col}")]
            except ImportError:
                cleaned = [clean_text(t, lexicons) for t in raw_series]
        else:
            cleaned = [clean_text(t, lexicons) for t in raw_series]

        out[text_col] = cleaned

        # Step 10: Đổi tên cột comment thành clean_comment
        out = out.rename(columns={text_col: clean_col})

    return out


def clean_csv(
    input_path: str | Path,
    output_path: str | Path,
    text_col: str = "comment",
    clean_col: str = "clean_comment",
    drop_cols: Sequence[str] = ("n_star",),
    lexicon_dir: Path | str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Đọc file CSV, làm sạch và lưu ra file kết quả."""
    in_p, out_p = Path(input_path), Path(output_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {in_p}")

    print(f"Đang đọc {in_p}...")
    df = pd.read_csv(in_p)
    df_clean = clean_dataframe(
        df,
        text_col=text_col,
        clean_col=clean_col,
        drop_cols=drop_cols,
        lexicon_dir=lexicon_dir,
        show_progress=show_progress,
    )

    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_p, index=False, encoding="utf-8")
    print(f"Đã lưu file làm sạch tại: {out_p} ({len(df_clean)} dòng)")
    print(f"Các cột trong file mới: {list(df_clean.columns)}")
    return df_clean


# =============================================================================
# 5. CLI CHẠY TRỰC TIẾP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Clean Data Pipeline (All-in-one script)")
    parser.add_argument("--input", type=Path, default=None, help="Đường dẫn file CSV đầu vào")
    parser.add_argument("--output", type=Path, default=None, help="Đường dẫn file CSV đầu ra")
    parser.add_argument("--splits", nargs="+", default=["Train", "Dev"], help="Tên các file CSV trong UIT-ViSFD (mặc định: Train Dev)")
    args = parser.parse_args()

    # Nếu truyền file cụ thể
    if args.input and args.output:
        clean_csv(args.input, args.output, show_progress=True)
        return

    # Mặc định tìm trong thư mục UIT-ViSFD
    data_dir = Path(__file__).resolve().parent / "UIT-ViSFD"
    if not data_dir.exists():
        data_dir = Path.cwd() / "UIT-ViSFD"

    for split in args.splits:
        in_file = data_dir / f"{split}.csv"
        if not in_file.exists():
            in_file = data_dir / f"{split.lower()}.csv"
        if not in_file.exists():
            print(f"Không tìm thấy {in_file}, bỏ qua.")
            continue

        out_file = data_dir / f"{in_file.stem}_clean.csv"
        print(f"\n=== ĐANG LÀM SẠCH [{split.upper()}]: {in_file.name} -> {out_file.name} ===")
        clean_csv(in_file, out_file, show_progress=True)

    print("\nHoàn tất toàn bộ quy trình làm sạch dữ liệu!")


if __name__ == "__main__":
    main()
