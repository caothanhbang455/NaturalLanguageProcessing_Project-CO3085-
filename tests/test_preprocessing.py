"""Kiểm thử các luật tiền xử lý: mỗi test là một xu hướng viết tổng quát."""

import pytest

from absa.preprocessing import TextPreprocessor
from absa.preprocessing import normalizers as nz
from absa.preprocessing.tokenization import attach_negation


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("pin 5000mah", "pin numbattery"),
        ("giá 7.990.000đ", "giá numprice"),
        ("tặng 250k", "tặng numprice"),
        ("giá 7tr5", "giá numprice"),
        ("bộ nhớ 128gb", "bộ nhớ numstorage"),
        ("cam 48mp", "cam numcamera"),
        ("màn 120hz", "màn numfreq"),
        ("pin còn 88%", "pin còn numpercent"),
        ("mua 8/3/2019", "mua numdate"),
        ("dùng 2 tháng", "dùng numduration"),
        ("galaxy a51", "galaxy a51"),  # mã model không bị đụng tới
    ],
)
def test_normalize_numbers(text, expected):
    assert nz.normalize_whitespace(nz.normalize_numbers(text)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("đẹppppp quá", "đẹp quá"), ("okkkk", "ok"), ("rồii", "rồi"), ("good game", "good game")],
)
def test_reduce_elongation(text, expected):
    assert nz.reduce_elongation(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("khôg", "không"), ("nhữg", "những"), ("đag", "đang"), ("lag", "lag")],
)
def test_restore_final_ng(text, expected):
    assert nz.restore_final_ng(text) == expected


def test_emoji_and_unknown_emoji_fallback():
    assert nz.normalize_whitespace(nz.replace_emojis("tốt 👍👍")) == "tốt emopos"
    # 🫠 (không có trong bảng ghi đè) vẫn được suy ra nhóm từ tên Unicode hoặc bị bỏ.
    out = nz.normalize_whitespace(nz.replace_emojis("máy 😡"))
    assert out == "máy emoneg"


def test_emoticons_do_not_touch_times():
    assert "emopos" in nz.replace_emoticons("ngon :))")
    assert "emoneg" in nz.replace_emoticons("buồn T_T")
    assert nz.replace_emoticons("lúc 8:30") == "lúc 8:30"


def test_laughter_before_elongation():
    assert nz.normalize_whitespace(nz.replace_laughter("hay kkkk")) == "hay emopos"
    assert nz.normalize_whitespace(nz.replace_laughter("haizzz chán")) == "emoneg chán"


def test_teencode_whole_token_only():
    normalizer = nz.TeencodeNormalizer({"k": "không", "dc": "được", "cũng z": "cũng vậy"})
    assert normalizer("k dc") == "không được"
    assert normalizer("kém") == "kém"
    assert normalizer("cũng z thôi") == "cũng vậy thôi"


def test_attach_negation_with_filler():
    negations = {"không"}
    assert attach_negation(["không", "mượt"], negations) == ["không_mượt"]
    assert attach_negation(["không", "được", "mượt"], negations) == ["không_được_mượt"]
    assert attach_negation(["pin", "không"], negations) == ["pin", "không"]


def test_negation_stops_at_sentence_boundary():
    negations = {"không"}
    # "em ko ? ) . - cam thì": dấu câu là ranh giới, "không" không được nối sang "cam".
    tokens = ["em", "không", "?", ")", ".", "-", "cam", "thì"]
    assert attach_negation(tokens, negations) == tokens
    assert attach_negation(["không", ",", "tốt"], negations) == ["không", ",", "tốt"]
    # từ đệm đứng trước dấu câu vẫn được nối, nhưng không vượt qua dấu câu
    assert attach_negation(["không", "được", ".", "pin"], negations) == [
        "không_được",
        ".",
        "pin",
    ]


def test_double_negation_is_not_fused():
    assert attach_negation(["không", "không", "tốt"], {"không"}) == ["không", "không_tốt"]


def test_classic_negation_does_not_cross_lines():
    out = (
        TextPreprocessor("classic").process("pin ko ?\n- cam thì oppo vô đối, không mượt").split()
    )
    assert "không_cam" not in out
    assert "không" in out  # ko -> không đứng riêng
    assert "không_mượt" in out  # phủ định trong cùng câu vẫn được gắn


def test_classic_profile_end_to_end():
    out = TextPreprocessor("classic").process(
        "Pin trâu 5000mAh 👍, ko lag tí nào\nNhân viên nhiệt tình"
    )
    tokens = out.split()
    assert "numbattery" in tokens and "emopos" in tokens
    assert "không_lag" in tokens
    assert "nhân" in tokens and "viên" in tokens  # classic giữ mức âm tiết


def test_classic_seg_profile_segments_words():
    tokens = TextPreprocessor("classic_seg").process("Nhân viên nhiệt tình").split()
    assert "nhân_viên" in tokens


def test_transformer_raw_keeps_numbers_and_k_unit():
    out = TextPreprocessor("transformer_raw").process("tặng phiếu 250k, ok")
    assert "250k" in out and "không" not in out


def test_trace_lists_every_step():
    trace = TextPreprocessor("classic").trace("Máy đẹppp :))")
    assert trace["step"].iloc[0] == "input"
    assert trace["changed"].any()
