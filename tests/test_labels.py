"""Kiểm thử mã hóa/giải mã nhãn ABSA."""

import numpy as np
import pytest

from absa.config import ALL_ASPECTS
from absa.data.labels import (
    decode_row,
    encode_labels,
    format_label_string,
    parse_label_string,
    to_aspect_matrix,
    to_pair_matrix,
)

RAW = "{BATTERY#Negative};{GENERAL#Positive};{OTHERS};"


def test_parse_label_string():
    assert parse_label_string(RAW) == {
        "BATTERY": "Negative",
        "GENERAL": "Positive",
        "OTHERS": None,
    }


def test_parse_empty_and_nan():
    assert parse_label_string("") == {}
    assert parse_label_string(float("nan")) == {}


def test_parse_rejects_unknown_aspect():
    with pytest.raises(ValueError):
        parse_label_string("{WIFI#Positive};")


def test_encode_decode_roundtrip():
    Y = encode_labels([RAW, "{SER&ACC#Neutral};"])
    assert Y.shape == (2, len(ALL_ASPECTS))
    assert format_label_string(decode_row(Y[0])) == RAW
    assert decode_row(Y[1]) == {"SER&ACC": "Neutral"}


def test_aspect_and_pair_matrices():
    Y = encode_labels([RAW])
    assert to_aspect_matrix(Y).sum() == 3
    pairs = to_pair_matrix(Y)
    assert pairs.shape == (1, 31)
    assert pairs.sum() == 3  # BATTERY#Negative, GENERAL#Positive, OTHERS
    assert np.array_equal(to_pair_matrix(encode_labels([""])), np.zeros((1, 31)))
