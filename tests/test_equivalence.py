"""Prove the packaged ADERH is score-identical to the original ADERH.py."""
import sys, pathlib
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _legacy_original import ADERH as LegacyADERH  # noqa: E402
from aderh import ADERH  # noqa: E402


def _data(seed):
    X, _ = make_blobs(n_samples=300, centers=3, n_features=8,
                      cluster_std=1.0, random_state=seed)
    rng = np.random.RandomState(seed)
    outliers = rng.uniform(X.min(0) - 2, X.max(0) + 2, size=(20, 8))
    X = np.vstack([X, outliers])
    return MinMaxScaler().fit_transform(X)


def test_scores_identical_to_reference():
    for seed in (0, 7, 1000):
        X = _data(seed)
        legacy = LegacyADERH(random_state=seed).fit(X)
        new = ADERH(random_state=seed).fit(X)
        assert np.allclose(legacy.outlier_score, new.decision_scores_,
                           atol=1e-12), f"train scores differ (seed={seed})"
        assert np.array_equal(legacy.labels_, new.labels_)
        X_test = _data(seed + 1)[:50]
        assert np.allclose(legacy.decision_function(X_test),
                           new.decision_function(X_test), atol=1e-12), \
            f"test scores differ (seed={seed})"
