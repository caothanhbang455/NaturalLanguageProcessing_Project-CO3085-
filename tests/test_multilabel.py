"""Kiểm thử chiến lược multilabel (31 nhãn nhị phân) và giải mã về ma trận nhãn."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from absa.data.labels import (
    count_pair_conflicts,
    encode_labels,
    pair_label_names,
    pairs_to_labels,
    to_pair_matrix,
)
from absa.evaluation import evaluate_absa, summarize_by
from absa.models.ml import STRATEGIES, ABSAClassifier


def test_pairs_roundtrip_is_lossless():
    Y = encode_labels(
        ["{BATTERY#Negative};{GENERAL#Positive};{OTHERS};", "{SER&ACC#Neutral};", ""]
    )
    assert np.array_equal(pairs_to_labels(to_pair_matrix(Y)), Y)


def test_conflict_resolution_uses_scores():
    names = pair_label_names()
    bits = np.zeros((1, 31), dtype=int)
    neg, pos = names.index("BATTERY#Negative"), names.index("BATTERY#Positive")
    bits[0, [neg, pos]] = 1
    # không có scores: chọn theo thứ tự Negative < Neutral < Positive -> Negative (id 1)
    assert pairs_to_labels(bits)[0, 3] == 1
    scores = np.zeros((1, 31))
    scores[0, pos], scores[0, neg] = 2.0, 0.5
    assert pairs_to_labels(bits, scores)[0, 3] == 3  # điểm Positive cao hơn
    assert count_pair_conflicts(bits)["n_conflicts"] == 1


def test_no_bit_means_aspect_absent():
    assert not pairs_to_labels(np.zeros((2, 31), dtype=int)).any()


@pytest.fixture
def toy():
    texts = ["pin trâu", "pin yếu", "camera đẹp", "camera mờ", "pin trâu camera đẹp", "giá rẻ"] * 5
    labels = [
        "{BATTERY#Positive};",
        "{BATTERY#Negative};",
        "{CAMERA#Positive};",
        "{CAMERA#Negative};",
        "{BATTERY#Positive};{CAMERA#Positive};",
        "{PRICE#Positive};",
    ] * 5
    return TfidfVectorizer(token_pattern=r"\S+").fit_transform(texts), encode_labels(labels)


@pytest.mark.parametrize(
    "estimator",
    [LogisticRegression(C=10, max_iter=500), LinearSVC(), MultinomialNB()],
    ids=["logreg", "linearsvc", "multinomial_nb"],
)
def test_multilabel_classifier_learns_toy_data(toy, estimator):
    X, Y = toy
    clf = ABSAClassifier(estimator, strategy="multilabel").fit(X, Y)
    assert len(clf.estimators_) == 31
    pred = clf.predict(X)
    assert pred.shape == Y.shape
    assert evaluate_absa(Y, pred)["pair_f1_micro"] > 0.9
    assert clf.predict_pairs(X).shape == (X.shape[0], 31)


def test_predict_pairs_only_for_multilabel(toy):
    X, Y = toy
    clf = ABSAClassifier(LogisticRegression(max_iter=200), strategy="joint").fit(X, Y)
    with pytest.raises(ValueError):
        clf.predict_pairs(X)


def test_multilabel_is_a_registered_strategy_and_default():
    assert "multilabel" in STRATEGIES
    assert ABSAClassifier(LogisticRegression()).strategy == "multilabel"


def test_summarize_by_uses_only_complete_pairs():
    import pandas as pd

    rows = [
        {
            "status": "ok",
            "tag": "grid",
            "feature": "f",
            "model": "m1",
            "strategy": s,
            "pair_f1_macro": v,
        }
        for s, v in [("a", 0.5), ("b", 0.6)]
    ] + [
        {
            "status": "ok",
            "tag": "grid",
            "feature": "f",
            "model": "m2",
            "strategy": "a",
            "pair_f1_macro": 0.9,
        }  # m2 thiếu chiến lược b -> phải bị bỏ khỏi so sánh
    ]
    frame = pd.DataFrame(rows).assign(
        pair_f1_micro=0.0,
        aspect_f1_macro=0.0,
        aspect_f1_micro=0.0,
        polarity_acc_on_detected=0.0,
        fit_sec=0.0,
    )
    summary = summarize_by(frame, "strategy")
    assert summary.loc["a", "n_pairs"] == 1
    assert summary.loc["a", "pair_f1_macro"] == pytest.approx(0.5)
