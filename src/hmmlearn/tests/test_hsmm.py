"""Tests for Hidden Semi-Markov Models (hmmlearn.hsmm)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hmmlearn.hsmm import (
    GaussianHSMM,
    CategoricalHSMM,
    PoissonHSMM,
    _PoissonDuration,
    _GaussianDuration,
    _NegativeBinomialDuration,
    _UniformDuration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gaussian_hsmm(n_components=2, covariance_type="diag",
                        duration_distribution="poisson",
                        max_duration=15, seed=42):
    rng = np.random.RandomState(seed)
    model = GaussianHSMM(n_components=n_components,
                          covariance_type=covariance_type,
                          duration_distribution=duration_distribution,
                          max_duration=max_duration,
                          random_state=seed,
                          n_iter=5)
    return model


# ---------------------------------------------------------------------------
# Duration distribution unit tests
# ---------------------------------------------------------------------------

class TestDurationDistributions:

    def _check_log_pmf(self, model, n_components, max_duration):
        model.initialize()
        lp = model.log_pmf()
        assert lp.shape == (max_duration, n_components)
        # Each column must be a valid log-probability (sums to ~1)
        sums = np.exp(lp).sum(axis=0)
        assert_allclose(sums, np.ones(n_components), atol=1e-6)

    def test_poisson_log_pmf(self):
        m = _PoissonDuration(3, 0, 20)
        self._check_log_pmf(m, 3, 20)

    def test_gaussian_log_pmf(self):
        m = _GaussianDuration(3, 0, 20)
        self._check_log_pmf(m, 3, 20)

    def test_negative_binomial_log_pmf(self):
        m = _NegativeBinomialDuration(3, 0, 20)
        self._check_log_pmf(m, 3, 20)

    def test_uniform_log_pmf(self):
        m = _UniformDuration(3, 0, 20)
        self._check_log_pmf(m, 3, 20)

    def test_poisson_sample(self):
        rng = np.random.RandomState(0)
        m = _PoissonDuration(2, 0, 50)
        m.initialize()
        for state in range(2):
            d = m.sample(state, rng)
            assert d >= 1

    def test_uniform_sample(self):
        rng = np.random.RandomState(0)
        m = _UniformDuration(2, 0, 20)
        m.initialize()
        for state in range(2):
            d = m.sample(state, rng)
            assert m.d_min_[state] <= d <= m.d_max_[state]

    def test_poisson_mstep(self):
        m = _PoissonDuration(2, 0, 10)
        m.initialize()
        # Fake expected durations: all weight on d=5 for state 0, d=3 for state 1
        eta = np.zeros((10, 2))
        eta[4, 0] = 1.0  # d=5
        eta[2, 1] = 1.0  # d=3
        m.mstep(eta)
        assert_allclose(m.lambda_[0], 5.0, atol=1e-6)
        assert_allclose(m.lambda_[1], 3.0, atol=1e-6)

    def test_gaussian_mstep(self):
        m = _GaussianDuration(2, 0, 10)
        m.initialize()
        eta = np.zeros((10, 2))
        eta[4, 0] = 1.0
        eta[2, 1] = 1.0
        m.mstep(eta)
        assert_allclose(m.means_[0], 5.0, atol=1e-6)
        assert_allclose(m.means_[1], 3.0, atol=1e-6)

    def test_get_set_params_roundtrip(self):
        m = _PoissonDuration(2, 0, 10)
        m.initialize()
        params = m.get_params()
        old = params['lambda_'].copy()
        m.set_params(lambda_=old * 2)
        assert_allclose(m.lambda_, old * 2)


# ---------------------------------------------------------------------------
# GaussianHSMM
# ---------------------------------------------------------------------------

class TestGaussianHSMM:

    @pytest.mark.parametrize("dur", ["poisson", "gaussian",
                                      "negative_binomial", "uniform"])
    def test_sample_shape(self, dur):
        model = GaussianHSMM(n_components=2, duration_distribution=dur,
                              max_duration=10, random_state=0, n_iter=1)
        model.startprob_ = np.array([0.6, 0.4])
        A = np.array([[0, 1], [1, 0]], dtype=float)
        model.transmat_ = A
        model.means_ = np.array([[0.0], [5.0]])
        model._covars_ = np.array([[1.0], [1.0]])
        model.n_features = 1
        model._duration_model_ = model._build_duration_model()
        model._duration_model_.initialize()

        X, Z = model.sample(50)
        assert X.shape == (50, 1)
        assert Z.shape == (50,)
        assert set(Z).issubset({0, 1})

    @pytest.mark.parametrize("cov_type", ["diag", "full", "spherical"])
    def test_fit_score_increases(self, cov_type):
        """Fit on generated data; score after fit should be finite."""
        rng = np.random.RandomState(7)
        X = np.concatenate([
            rng.randn(30, 1) + 0.0,
            rng.randn(30, 1) + 5.0,
        ])

        model = GaussianHSMM(n_components=2,
                              covariance_type=cov_type,
                              duration_distribution="poisson",
                              max_duration=8,
                              random_state=42,
                              n_iter=5,
                              tol=1e-3)
        model.fit(X)
        score_after = model.score(X)
        assert np.isfinite(score_after)

    def test_decode_returns_correct_shape(self):
        rng = np.random.RandomState(3)
        X = rng.randn(40, 2)
        model = GaussianHSMM(n_components=2,
                              covariance_type="diag",
                              duration_distribution="poisson",
                              max_duration=8,
                              random_state=0,
                              n_iter=3)
        model.fit(X)
        log_prob, states = model.decode(X)
        assert states.shape == (40,)
        assert np.isfinite(log_prob)

    def test_predict_proba_sums_to_one(self):
        rng = np.random.RandomState(4)
        X = rng.randn(30, 1)
        model = GaussianHSMM(n_components=2,
                              covariance_type="diag",
                              duration_distribution="poisson",
                              max_duration=8,
                              random_state=0,
                              n_iter=3)
        model.fit(X)
        proba = model.predict_proba(X)
        assert proba.shape == (30, 2)
        assert_allclose(proba.sum(axis=1), np.ones(30), atol=1e-5)

    def test_multiple_sequences(self):
        rng = np.random.RandomState(5)
        X = rng.randn(60, 1)
        lengths = [20, 20, 20]
        model = GaussianHSMM(n_components=2,
                              covariance_type="diag",
                              duration_distribution="poisson",
                              max_duration=8,
                              random_state=0,
                              n_iter=3)
        model.fit(X, lengths=lengths)
        log_prob = model.score(X, lengths=lengths)
        assert np.isfinite(log_prob)

    def test_viterbi_and_map_consistent(self):
        rng = np.random.RandomState(9)
        X = rng.randn(30, 1)
        model = GaussianHSMM(n_components=2,
                              covariance_type="diag",
                              duration_distribution="poisson",
                              max_duration=8,
                              random_state=0,
                              n_iter=3)
        model.fit(X)
        _, states_v = model.decode(X, algorithm="viterbi")
        _, states_m = model.decode(X, algorithm="map")
        assert states_v.shape == (30,)
        assert states_m.shape == (30,)

    def test_invalid_duration_distribution(self):
        model = GaussianHSMM(n_components=2,
                              duration_distribution="unknown_dist",
                              max_duration=10, random_state=0)
        rng = np.random.RandomState(0)
        X = rng.randn(30, 1)
        with pytest.raises(ValueError, match="Unknown duration_distribution"):
            model.fit(X)

    def test_transmat_no_self_loops_enforced(self):
        model = GaussianHSMM(n_components=2,
                              covariance_type="diag",
                              duration_distribution="poisson",
                              max_duration=8, random_state=0, n_iter=1)
        X = np.random.RandomState(0).randn(30, 1)
        model.fit(X)
        assert_allclose(np.diag(model.transmat_), np.zeros(2), atol=1e-10)

    def test_aic_bic(self):
        rng = np.random.RandomState(6)
        X = rng.randn(40, 1)
        model = GaussianHSMM(n_components=2,
                              covariance_type="diag",
                              duration_distribution="poisson",
                              max_duration=8, random_state=0, n_iter=3)
        model.fit(X)
        aic = model.aic(X)
        bic = model.bic(X)
        assert np.isfinite(aic)
        assert np.isfinite(bic)

    @pytest.mark.parametrize("dur", ["poisson", "gaussian",
                                      "negative_binomial", "uniform"])
    def test_fit_all_duration_distributions(self, dur):
        rng = np.random.RandomState(0)
        X = np.concatenate([rng.randn(20, 1), rng.randn(20, 1) + 4])
        model = GaussianHSMM(n_components=2,
                              covariance_type="diag",
                              duration_distribution=dur,
                              max_duration=8, random_state=0, n_iter=3)
        model.fit(X)
        assert np.isfinite(model.score(X))


# ---------------------------------------------------------------------------
# CategoricalHSMM
# ---------------------------------------------------------------------------

class TestCategoricalHSMM:

    def test_sample_and_fit(self):
        rng = np.random.RandomState(1)
        model = CategoricalHSMM(n_components=2, n_features=4,
                                 duration_distribution="poisson",
                                 max_duration=10, random_state=0, n_iter=5)
        # Manually set parameters to sample from
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.0, 1.0], [1.0, 0.0]])
        model.emissionprob_ = np.array([[0.7, 0.1, 0.1, 0.1],
                                         [0.1, 0.1, 0.1, 0.7]])
        model.n_features = 4
        model._duration_model_ = model._build_duration_model()
        model._duration_model_.initialize()

        X, Z = model.sample(100)
        assert X.shape[0] == 100
        assert set(Z.tolist()).issubset({0, 1})

    def test_fit_score(self):
        rng = np.random.RandomState(2)
        X = rng.randint(0, 3, size=(40, 1))
        model = CategoricalHSMM(n_components=2, n_features=3,
                                 duration_distribution="poisson",
                                 max_duration=8, random_state=0, n_iter=3)
        model.fit(X)
        assert np.isfinite(model.score(X))

    def test_predict_proba_shape(self):
        rng = np.random.RandomState(3)
        X = rng.randint(0, 4, size=(30, 1))
        model = CategoricalHSMM(n_components=2, n_features=4,
                                 duration_distribution="poisson",
                                 max_duration=8, random_state=0, n_iter=3)
        model.fit(X)
        proba = model.predict_proba(X)
        assert proba.shape == (30, 2)
        assert_allclose(proba.sum(axis=1), np.ones(30), atol=1e-5)

    def test_non_integer_raises(self):
        X = np.array([[0.5], [1.0], [2.3]])
        model = CategoricalHSMM(n_components=2, n_features=3,
                                 max_duration=5, random_state=0)
        with pytest.raises(ValueError, match="integers"):
            model.fit(X)


# ---------------------------------------------------------------------------
# PoissonHSMM
# ---------------------------------------------------------------------------

class TestPoissonHSMM:

    def test_sample_shape(self):
        model = PoissonHSMM(n_components=2,
                             duration_distribution="poisson",
                             max_duration=10, random_state=0, n_iter=1)
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        model.lambdas_ = np.array([1.0, 5.0])
        model.n_features = 1
        model._duration_model_ = model._build_duration_model()
        model._duration_model_.initialize()

        X, Z = model.sample(100)
        assert X.shape == (100, 1)
        assert Z.shape == (100,)

    def test_fit_score(self):
        rng = np.random.RandomState(0)
        X = rng.poisson(3, size=(50, 1))
        model = PoissonHSMM(n_components=2,
                             duration_distribution="poisson",
                             max_duration=8, random_state=0, n_iter=5)
        model.fit(X)
        assert np.isfinite(model.score(X))

    @pytest.mark.parametrize("dur", ["poisson", "gaussian",
                                      "negative_binomial", "uniform"])
    def test_fit_all_durations(self, dur):
        rng = np.random.RandomState(99)
        X = np.concatenate([rng.poisson(1, size=(25, 1)),
                             rng.poisson(6, size=(25, 1))])
        model = PoissonHSMM(n_components=2,
                             duration_distribution=dur,
                             max_duration=8, random_state=0, n_iter=3)
        model.fit(X)
        assert np.isfinite(model.score(X))

    def test_wrong_shape_raises(self):
        X = np.random.poisson(2, size=(50, 3))
        model = PoissonHSMM(n_components=2, max_duration=10, random_state=0)
        with pytest.raises(ValueError, match="univariate"):
            model.fit(X)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_sample_sequence(self):
        """A single-sample sequence with 1 component should not crash."""
        X = np.array([[1.5]])
        model = GaussianHSMM(n_components=1,
                              covariance_type="diag",
                              duration_distribution="poisson",
                              max_duration=5, random_state=0, n_iter=2)
        # Should not raise with n_components=1 and 1 sample
        model.fit(X)

    def test_single_component(self):
        rng = np.random.RandomState(0)
        X = rng.randn(20, 1)
        model = GaussianHSMM(n_components=1,
                              covariance_type="diag",
                              duration_distribution="poisson",
                              max_duration=5, random_state=0, n_iter=2)
        model.fit(X)
        assert np.isfinite(model.score(X))

    def test_check_fails_on_self_loop(self):
        model = GaussianHSMM(n_components=2,
                              covariance_type="diag",
                              duration_distribution="poisson",
                              max_duration=10, random_state=0)
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])  # non-zero diagonal
        model.means_ = np.array([[0.0], [1.0]])
        model._covars_ = np.array([[1.0], [1.0]])
        model.n_features = 1
        with pytest.raises(ValueError, match="zero diagonal"):
            model._check()
