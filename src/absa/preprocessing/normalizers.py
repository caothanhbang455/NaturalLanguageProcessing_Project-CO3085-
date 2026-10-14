"""Các bước chuẩn hóa văn bản ở mức ký tự/chuỗi.

Mỗi hàm ở đây là **một luật tổng quát** rút ra từ quan sát xu hướng viết trên
tập train (không sửa riêng cho câu nào), nhận vào ``str`` và trả về ``str`` nên có
thể ghép tùy ý trong :class:`absa.preprocessing.pipeline.TextPreprocessor`.

Thứ tự gợi ý (và cũng là thứ tự pipeline dùng):

0. :func:`mark_line_breaks`         – xuống dòng -> dấu chấm (ranh giới câu).
1. :func:`remove_urls_emails`       – URL, email, số điện thoại.
2. :func:`replace_emojis`           – emoji -> nhãn cảm xúc.
3. :func:`replace_emoticons`        – ``:)) =)) T_T`` -> nhãn cảm xúc.
4. :func:`normalize_unicode`        – NFC + chuẩn kiểu bỏ dấu (hoà -> hòa),
   rồi :func:`rejoin_number_units` (``250 k`` -> ``250k``).
5. ``lowercase``
6. :func:`replace_laughter`         – ``hahaha kkk huhu haizz`` -> nhãn cảm xúc.
7. :func:`normalize_numbers`        – ``5000mah 7tr5 128gb`` -> token theo loại.
8. :func:`reduce_elongation`        – ``đẹppppp okkk rồii`` -> ``đẹp ok rồi``.
9. :func:`normalize_punctuation`    – tách dấu câu khỏi từ (chỉ xóa sau bước tách từ).
10. :func:`restore_final_ng`        – ``khôg nhữg đag`` -> ``không những đang``.
11. :class:`TeencodeNormalizer`     – ``ko dc đt nv`` -> ``không được điện thoại nhân viên``.
12. :func:`split_joined_placeholders` – sau tách từ, tách token giữ chỗ bị ghép nhầm.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

# --------------------------------------------------------------------------- #
# 1. Unicode
# --------------------------------------------------------------------------- #
_INVISIBLE_RE = re.compile("[\u200b-\u200f\u2060\ufeff\ufe0e\ufe0f]")


def normalize_unicode(text: str, fix_tone_marks: bool = True) -> str:
    """Chuẩn hóa Unicode về NFC và thống nhất vị trí dấu thanh.

    Văn bản tiếng Việt có thể được gõ bằng Unicode tổ hợp (NFD) hoặc dựng sẵn
    (NFC) và theo hai kiểu bỏ dấu (``hoà``/``hòa``, ``thuỷ``/``thủy``). Nếu không
    chuẩn hóa, cùng một từ sẽ thành nhiều token khác nhau trong từ điển đặc trưng.

    Args:
        text: Chuỗi đầu vào.
        fix_tone_marks: Dùng ``underthesea.text_normalize`` để đưa dấu thanh về
            kiểu chuẩn. Nếu thư viện không có, chỉ áp dụng NFC.
    """
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE_RE.sub("", text)
    if fix_tone_marks:
        try:
            from underthesea import text_normalize
        except ImportError:  # pragma: no cover - phụ thuộc môi trường
            return text
        text = text_normalize(text)
    return text


# --------------------------------------------------------------------------- #
# 2. URL / email / số điện thoại
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?84|0)\d{9,10}(?!\d)")


_LINE_BREAK_RE = re.compile(r"\s*[\r\n]+\s*")


def mark_line_breaks(text: str) -> str:
    r"""Đổi xuống dòng thành dấu chấm (11% bình luận train dùng xuống dòng thay dấu câu).

    Nếu chỉ coi xuống dòng là khoảng trắng, từ cuối dòng trên và từ đầu dòng dưới có thể
    bị bộ tách từ ghép nhầm (``ngon \n Âm báo`` -> ``ngon_âm``).
    """
    return _LINE_BREAK_RE.sub(" . ", text.strip())


def remove_urls_emails(text: str) -> str:
    """Xóa URL, email và số điện thoại – không mang thông tin khía cạnh/cảm xúc."""
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    return _PHONE_RE.sub(" ", text)


# --------------------------------------------------------------------------- #
# 3–4. Emoji và emoticon
# --------------------------------------------------------------------------- #
SENTIMENT_TAGS: dict[str, str] = {"pos": "emopos", "neg": "emoneg", "neu": "emoneu"}
SENTIMENT_WORDS: dict[str, str] = {"pos": "vui", "neg": "buồn", "neu": ""}

_EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"  # emoji, biểu tượng, cờ, skin tone
    "\u2600-\u27bf"  # biểu tượng linh tinh, dingbats (❤ ✌ ☺ ★)
    "\u2b00-\u2bff"  # ⭐ và mũi tên
    "\u2300-\u23ff"  # ⌚ ⏰
    "]"
)
EMOJI_PATTERN = _EMOJI_RE  # Công khai để module EDA dùng chung đúng một định nghĩa emoji.
# Từ khóa trong tên Unicode -> nhóm cảm xúc. Áp dụng cho emoji chưa có trong bảng ghi đè,
# giúp xử lý được cả emoji chưa từng xuất hiện trong dữ liệu huấn luyện.
_NEG_STRONG = ("BROKEN HEART", "THUMBS DOWN", "MIDDLE FINGER", "PILE OF POO")
_POS_KEYWORDS = (
    "SMIL",
    "GRIN",
    "LAUGH",
    "JOY",
    "HEART",
    "KISS",
    "THUMBS UP",
    "OK HAND",
    "CLAPPING",
    "STAR",
    "PARTY",
    "SPARKL",
    "HUGGING",
    "WINK",
    "SAVOURING",
    "SUNGLASSES",
    "HUNDRED",
    "TROPHY",
    "MEDAL",
    "RAISING BOTH HANDS",
    "VICTORY",
    "FLEXED",
    "BLOWING A KISS",
)
_NEG_KEYWORDS = (
    "CRY",
    "SOB",
    "ANGRY",
    "POUTING",
    "RAGE",
    "FROWN",
    "DISAPPOINT",
    "WEARY",
    "TIRED",
    "CONFOUNDED",
    "PERSEVER",
    "ANGUISH",
    "FEARFUL",
    "SCREAM",
    "NAUSEA",
    "VOMIT",
    "UNAMUSED",
    "EXPRESSIONLESS",
    "SKULL",
    "WORRIED",
    "CONFUSED",
    "PENSIVE",
    "DOWNCAST",
    "CROSS MARK",
    "NO ENTRY",
    "FACE PALM",
    "ROLLING EYES",
    "GRIMAC",
    "HOT FACE",
    "STEAM FROM NOSE",
    "SYMBOLS ON MOUTH",
)
_NEU_KEYWORDS = (
    "THINKING",
    "NEUTRAL FACE",
    "WITHOUT MOUTH",
    "HUSHED",
    "ASTONISHED",
    "FLUSHED",
    "OPEN MOUTH",
    "PLEADING",
    "FOLDED HANDS",
)


def emoji_category(char: str, overrides: Mapping[str, str] | None = None) -> str | None:
    """Xác định nhóm cảm xúc của một emoji.

    Ưu tiên bảng ghi đè; nếu không có thì suy ra từ tên Unicode của ký tự.

    Returns:
        ``"pos"``, ``"neg"``, ``"neu"`` hoặc ``None`` (không rõ – sẽ bị bỏ).
    """
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


def _sentiment_token(category: str | None, mode: str) -> str:
    if category is None or mode == "remove":
        return " "
    token = SENTIMENT_TAGS[category] if mode == "tag" else SENTIMENT_WORDS[category]
    return f" {token} " if token else " "


_REPEATED_TAG_RE = re.compile(r"\b(emopos|emoneg|emoneu|vui|buồn)(?:\s+\1\b)+")


def collapse_repeated_tags(text: str) -> str:
    """Gộp nhãn cảm xúc lặp liên tiếp (``emopos emopos`` -> ``emopos``)."""
    return _REPEATED_TAG_RE.sub(r"\1", text)


def replace_emojis(
    text: str, mode: str = "tag", overrides: Mapping[str, str] | None = None
) -> str:
    """Thay emoji bằng nhãn cảm xúc.

    Emoji xuất hiện ở ~4% bình luận train và mang tín hiệu cảm xúc mạnh (👍 😍 😡),
    nhưng tập ký tự emoji rất lớn và thưa. Gom về 3 nhãn giúp mô hình học được tín
    hiệu chung thay vì ghi nhớ từng ký tự.

    Args:
        text: Chuỗi đầu vào.
        mode: ``"tag"`` (``emopos``/``emoneg``/``emoneu``), ``"word"`` (``vui``/``buồn``
            – hợp với mô hình pretrained), ``"remove"`` hoặc ``"keep"``.
        overrides: Bảng emoji -> nhóm cảm xúc ưu tiên.
    """
    if mode == "keep":
        return text

    def _repl(match: re.Match[str]) -> str:
        return _sentiment_token(emoji_category(match.group(0), overrides), mode)

    return collapse_repeated_tags(_EMOJI_RE.sub(_repl, text))


# (regex, nhóm cảm xúc). Lookaround tránh bắt nhầm giờ "8:30" hoặc tỉ số "2:1".
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


def replace_emoticons(text: str, mode: str = "tag") -> str:
    """Thay emoticon ASCII (``:)) =)) :( T_T ^^ <3 :v``) bằng nhãn cảm xúc.

    Phải chạy **trước** khi bỏ dấu câu và trước khi đưa về chữ thường (``:D``).
    """
    if mode == "keep":
        return text
    for pattern, category in _EMOTICON_RULES:
        text = pattern.sub(lambda _m, c=category: _sentiment_token(c, mode), text)
    return collapse_repeated_tags(text)


# --------------------------------------------------------------------------- #
# 6. Tiếng cười / khóc / thở dài viết bằng chữ
# --------------------------------------------------------------------------- #
_LAUGH_RE = re.compile(
    r"(?<!\w)(?:(?:ha|he|hi|hj|hô|hì|hí|há|hà|hé|hè|hê|kha|khà){2,}[a-zàáèéìíêôj]*|k{3,})(?!\w)"
)
_SIGH_CRY_RE = re.compile(r"(?<!\w)(?:(?:hu){2,}u*|h[ij][cx]+|hai?z{1,}|haz{2,}i*)(?!\w)")


def replace_laughter(text: str, mode: str = "tag") -> str:
    """Thay chuỗi cười/khóc/thở dài viết bằng chữ bằng nhãn cảm xúc.

    Cười: ``hahaha``, ``hihi``, ``kkkk``. Khóc/thở dài: ``huhu``, ``hic``, ``haizzz``.

    Cần chạy trên chữ thường và **trước** bước rút gọn ký tự lặp; nếu không
    ``kkkk`` sẽ thành ``k`` rồi bị hiểu nhầm thành ``không``.
    """
    if mode in {"keep"}:
        return text
    text = _LAUGH_RE.sub(lambda _m: _sentiment_token("pos", mode), text)
    text = _SIGH_CRY_RE.sub(lambda _m: _sentiment_token("neg", mode), text)
    return collapse_repeated_tags(text)


# --------------------------------------------------------------------------- #
# 7. Số kèm đơn vị
# --------------------------------------------------------------------------- #
_NUM = r"\d+(?:[.,]\d+)*"
_START = r"(?<![\w.,/])"
_END = r"(?!\w)"
# Thứ tự quan trọng: luật cụ thể chạy trước luật chung "num".
NUMBER_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"{_START}\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?{_END}"), "numdate"),
    (re.compile(rf"{_START}\d{{1,2}}\s*[:h]\s*\d{{2}}{_END}"), "numtime"),
    (
        re.compile(
            rf"{_START}{_NUM}\s*(?:tr\d*|triệu|trieu|củ|nghìn|ngàn|ngan|nghin|đồng|đ|vnđ|vnd){_END}"
        ),
        "numprice",
    ),
    (re.compile(rf"{_START}\d{{2,}}(?:[.,]\d+)*\s*k{_END}"), "numprice"),
    (re.compile(rf"{_START}\d{{1,3}}(?:[.,]\d{{3}}){{2,}}{_END}"), "numprice"),
    (re.compile(rf"{_START}{_NUM}\s*(?:mah|mha|mh){_END}"), "numbattery"),
    (re.compile(rf"{_START}{_NUM}\s*(?:gb|tb|mb|gig){_END}"), "numstorage"),
    (re.compile(rf"{_START}\d{{2,}}\s*g{_END}"), "numstorage"),
    # "4g" mơ hồ (mạng 4G hay RAM 4G): giữ một token riêng để mô hình dựa vào ngữ cảnh.
    (re.compile(rf"{_START}\d\s*g{_END}"), "numg"),
    (re.compile(rf"{_START}{_NUM}\s*(?:mp|mpx|megapixel|chấm){_END}"), "numcamera"),
    (re.compile(rf"{_START}{_NUM}\s*(?:hz|ghz|fps){_END}"), "numfreq"),
    (re.compile(rf"{_START}{_NUM}\s*w{_END}"), "numpower"),
    (re.compile(rf"{_START}{_NUM}\s*%"), "numpercent"),
    (re.compile(rf"{_START}{_NUM}\s*(?:inch|inches|\"|”)"), "numsize"),
    (
        re.compile(
            rf"{_START}{_NUM}\s*(?:giờ|tiếng|h|phút|p|giây|s|ngày|tuần|tháng|năm|th){_END}"
        ),
        "numduration",
    ),
    (re.compile(rf"{_START}{_NUM}{_END}"), "num"),
)


_SPLIT_UNIT_RE = re.compile(
    r"(\d)\s+(?=(?:k|tr\d*|triệu|củ|đ|vnđ|vnd|mah|mha|mh|gb|tb|mb|g|mp|mpx|hz|ghz|fps|w|%|"
    r"inch|h|p|s)(?!\w))",
    re.IGNORECASE,
)


def rejoin_number_units(text: str) -> str:
    """Gắn lại đơn vị bị tách khỏi số (``250 k`` -> ``250k``, ``5000 mAh`` -> ``5000mAh``).

    Bước chuẩn hóa Unicode của underthesea tách số và chữ dính nhau; nếu không gắn lại,
    ``250 k`` sẽ bị bước teencode hiểu nhầm ``k`` là ``không``.
    """
    return _SPLIT_UNIT_RE.sub(r"\1", text)


def normalize_numbers(text: str) -> str:
    """Thay số (kèm đơn vị) bằng token theo **loại đại lượng**.

    62% bình luận train có chữ số, phần lớn đi kèm đơn vị gắn chặt với một khía
    cạnh: ``mah`` -> BATTERY, ``tr/k/đ`` -> PRICE, ``gb`` -> STORAGE, ``mp`` ->
    CAMERA, ``hz/fps`` -> SCREEN/PERFORMANCE. Giá trị cụ thể (5000 hay 4000) hầu như
    không quan trọng nhưng *loại* đơn vị thì có, nên gom về token loại giúp giảm
    thưa và tổng quát hóa sang giá trị chưa gặp.

    Mã model như ``a51``, ``s20`` không bị ảnh hưởng (chữ số dính liền chữ cái).
    Phải chạy trên chữ thường và trước khi bỏ dấu câu (``7.990.000đ``, ``8/3/2019``).
    """
    for pattern, token in NUMBER_RULES:
        text = pattern.sub(f" {token} ", text)
    return text


# --------------------------------------------------------------------------- #
# 8. Ký tự kéo dài
# --------------------------------------------------------------------------- #
_REPEAT3_RE = re.compile(r"([^\W\d_])\1{2,}")
_FINAL_DOUBLE_RE = re.compile(r"([^\W\d_])\1$")


def reduce_elongation(text: str) -> str:
    """Rút gọn chữ cái bị kéo dài để nhấn mạnh.

    Luật dựa trên chính tả tiếng Việt: một âm tiết không bao giờ có cùng một chữ
    cái lặp 3 lần liên tiếp, nên mọi chuỗi lặp ``>= 3`` được rút về 1
    (``đẹppppp`` -> ``đẹp``, ``okkk`` -> ``ok``). Chữ cái lặp đôi ở **cuối** token
    chỉ được rút khi token có dấu tiếng Việt (``rồii`` -> ``rồi``), để không làm
    hỏng từ tiếng Anh như ``good``, ``full``, ``lagg`` vẫn giữ nguyên.
    """
    text = _REPEAT3_RE.sub(r"\1", text)
    tokens = []
    for token in text.split():
        if not token.isascii():
            token = _FINAL_DOUBLE_RE.sub(r"\1", token)
        tokens.append(token)
    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# 9. Dấu câu
# --------------------------------------------------------------------------- #
_PUNCT_RE = re.compile(r"[^\w\s]")
# Dấu câu cần tách khỏi từ, trừ dấu nằm giữa hai chữ số (7.990.000, 8/3/2019, 2,5).
_PUNCT_SPLIT_RE = re.compile(r"(?<!\d)[^\w\s]|[^\w\s](?!\d)")
_REPEATED_PUNCT_RE = re.compile(r"([^\w\s])\1+")


def normalize_punctuation(text: str, remove: bool = True) -> str:
    """Bỏ dấu câu (cho mô hình túi-từ) hoặc chuẩn hóa chúng (cho Transformer).

    Args:
        text: Chuỗi đầu vào.
        remove: ``True`` thay mọi ký tự không phải chữ/số/``_`` bằng khoảng trắng.
            ``False`` chỉ gộp dấu lặp (``!!!`` -> ``!``) và tách dấu khỏi từ
            (``tốt,pin`` -> ``tốt , pin``) – mô hình pretrained hiểu được dấu câu.
    """
    if remove:
        return _PUNCT_RE.sub(" ", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    return _PUNCT_SPLIT_RE.sub(lambda m: f" {m.group(0)} ", text)


# --------------------------------------------------------------------------- #
# 10. Thiếu "n" trước "g" cuối âm tiết
# --------------------------------------------------------------------------- #
_FINAL_VOWEL_G_RE = re.compile(r"[aeiouy]g$")


def _strip_diacritics(token: str) -> str:
    decomposed = unicodedata.normalize("NFD", token.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def restore_final_ng(text: str) -> str:
    """Khôi phục ``ng`` bị viết tắt thành ``g`` ở cuối âm tiết.

    Luật chính tả: âm tiết tiếng Việt không kết thúc bằng *nguyên âm + g*, chỉ có
    ``-ng``. Kiểu viết ``khôg``, ``nhữg``, ``cùg``, ``đag`` vì vậy luôn là thiếu
    ``n``. Chỉ áp dụng cho token có dấu tiếng Việt để không đụng tới từ tiếng Anh
    (``lag``, ``bag``, ``tag``).
    """
    tokens = []
    for token in text.split():
        if not token.isascii() and _FINAL_VOWEL_G_RE.search(_strip_diacritics(token)):
            token = token[:-1] + "ng"
        tokens.append(token)
    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# 11. Teencode / viết tắt
# --------------------------------------------------------------------------- #
class TeencodeNormalizer:
    """Thay teencode/viết tắt bằng dạng chuẩn dựa trên từ điển.

    Khớp **nguyên token** (không thay chuỗi con) để ``k`` không ảnh hưởng tới
    ``kém``. Khóa nhiều từ (``cũng z``) được thay trước bằng regex có ranh giới từ.

    Args:
        mapping: Từ điển ``teencode -> dạng chuẩn`` (khóa chữ thường).
    """

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self.single = {k: v for k, v in mapping.items() if " " not in k}
        phrases = sorted((k for k in mapping if " " in k), key=len, reverse=True)
        self.phrases = {k: mapping[k] for k in phrases}
        self._phrase_re = (
            re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, phrases)) + r")(?!\w)")
            if phrases
            else None
        )

    def __call__(self, text: str) -> str:
        """Áp dụng từ điển cho một văn bản."""
        if self._phrase_re is not None:
            text = self._phrase_re.sub(lambda m: self.phrases[m.group(0)], text)
        return " ".join(self.single.get(tok, tok) for tok in text.split())


PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    {tok for _, tok in NUMBER_RULES} | set(SENTIMENT_TAGS.values())
)


def split_joined_placeholders(text: str) -> str:
    """Tách token giữ chỗ bị bộ tách từ nối nhầm (``cao_numduration`` -> ``cao numduration``).

    Bộ tách từ coi token lạ như ``numduration`` là một âm tiết và đôi khi ghép nó với
    từ bên cạnh; token giữ chỗ phải luôn đứng riêng để mang đúng ý nghĩa.
    """
    tokens = []
    for token in text.split():
        parts = token.split("_")
        if len(parts) > 1 and any(p in PLACEHOLDER_TOKENS for p in parts):
            buffer: list[str] = []
            for part in parts:
                if part in PLACEHOLDER_TOKENS:
                    if buffer:
                        tokens.append("_".join(buffer))
                        buffer = []
                    tokens.append(part)
                else:
                    buffer.append(part)
            if buffer:
                tokens.append("_".join(buffer))
        else:
            tokens.append(token)
    return " ".join(tokens)


def normalize_whitespace(text: str) -> str:
    r"""Gộp mọi khoảng trắng (kể cả ``\r\n``) thành một dấu cách và cắt hai đầu."""
    return " ".join(text.split())
