"""Kiểm thử metric ABSA và bộ phân loại bọc ``ABSAClassifier``."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from absa.data.labels import encode_labels
from absa.evaluation import error_type_summary, evaluate_absa, per_aspect_report
from absa.features import get_feature_spec
from absa.models.ml import ABSAClassifier, check_compatibility, get_model_spec


def test_perfect_prediction_scores_one():
    Y = encode_labels(["{BATTERY#Positive};", "{CAMERA#Negative};{OTHERS};"])
    metrics = evaluate_absa(Y, Y)
    assert metrics["pair_f1_micro"] == pytest.approx(1.0)
    assert metrics["aspect_f1_micro"] == pytest.approx(1.0)
    assert metrics["exact_match"] == 1.0


def test_wrong_polarity_hurts_pairs_not_aspects():
    Y_true = encode_labels(["{BATTERY#Positive};"])
    Y_pred = encode_labels(["{BATTERY#Negative};"])
    metrics = evaluate_absa(Y_true, Y_pred)
    assert metrics["aspect_f1_micro"] == pytest.approx(1.0)
    assert metrics["pair_f1_micro"] == pytest.approx(0.0)
    assert error_type_summary(Y_true, Y_pred).loc["BATTERY", "polarity"] == 1


@pytest.fixture
def toy_data():
    texts = ["pin trâu", "pin yếu", "camera đẹp", "camera mờ", "pin trâu camera đẹp", "giá rẻ"] * 5
    labels = [
        "{BATTERY#Positive};",
        "{BATTERY#Negative};",
        "{CAMERA#Positive};",
        "{CAMERA#Negative};",
        "{BATTERY#Positive};{CAMERA#Positive};",
        "{PRICE#Positive};",
    ] * 5
    X = TfidfVectorizer(token_pattern=r"\S+").fit_transform(texts)
    return X, encode_labels(labels)


@pytest.mark.parametrize("strategy", ["joint", "two_stage"])
def test_absa_classifier_learns_toy_data(toy_data, strategy):
    X, Y = toy_data
    clf = ABSAClassifier(LogisticRegression(C=10, max_iter=500), strategy=strategy).fit(X, Y)
    pred = clf.predict(X)
    assert pred.shape == Y.shape
    assert evaluate_absa(Y, pred)["pair_f1_micro"] > 0.9
    assert per_aspect_report(Y, pred).loc["BATTERY", "detect_f1"] > 0.9


def test_compatibility_rules():
    ok, _ = check_compatibility(get_model_spec("multinomial_nb"), get_feature_spec("tfidf_word"))
    assert ok
    ok, reason = check_compatibility(
        get_model_spec("multinomial_nb"), get_feature_spec("word2vec")
    )
    assert not ok and reason
    # GaussianNB chỉ nhận đặc trưng đặc.
    ok, _ = check_compatibility(get_model_spec("gaussian_nb"), get_feature_spec("tfidf_word"))
    assert not ok


def test_core_models_are_registered_and_basic():
    from absa.models.ml import CORE_MODELS, MODEL_REGISTRY

    assert set(CORE_MODELS) == {"logistic_regression", "linear_svm", "knn"}
    assert all(MODEL_REGISTRY[m].family not in {"tree_ensemble", "neural"} for m in CORE_MODELS)
    # Cả 3 mô hình dùng được với mọi biểu diễn trong notebook (sparse lẫn embedding đặc).
    for feature in ("bow", "tfidf_word", "tfidf_word_char", "word2vec", "fasttext", "doc2vec"):
        for model in CORE_MODELS:
            ok, _ = check_compatibility(get_model_spec(model), get_feature_spec(feature))
            assert ok, (feature, model)


def test_classifier_rejects_wrong_label_shape(toy_data):
    X, _ = toy_data
    with pytest.raises(ValueError):
        ABSAClassifier(LogisticRegression()).fit(X, np.zeros((X.shape[0], 3)))
