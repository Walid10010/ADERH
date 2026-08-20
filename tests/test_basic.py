import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from aderh import ADERH


def _easy():
    X, _ = make_blobs(n_samples=400, centers=2, n_features=6,
                      cluster_std=0.5, random_state=0)
    rng = np.random.RandomState(0)
    out = rng.uniform(X.min(0) - 4, X.max(0) + 4, size=(40, 6))
    y = np.r_[np.zeros(400), np.ones(40)]
    return MinMaxScaler().fit_transform(np.vstack([X, out])), y


def test_detects_obvious_outliers():
    X, y = _easy()
    det = ADERH(random_state=0).fit(X)
    assert roc_auc_score(y, det.decision_scores_) > 0.9


def test_deterministic_and_api():
    X, _ = _easy()
    a = ADERH(random_state=42).fit(X)
    b = ADERH(random_state=42).fit(X)
    assert np.allclose(a.decision_scores_, b.decision_scores_)
    assert set(np.unique(a.labels_)) <= {0, 1}
    assert a.predict(X[:10]).shape == (10,)
    assert np.allclose(a.score_samples(X[:10]),
                       -a.decision_function(X[:10]))
    assert abs(a.labels_.mean() - 0.1) < 0.05  # ~contamination
