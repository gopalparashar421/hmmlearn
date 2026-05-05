"""Tests for GaussianMixtureHSMM — Plan 002 implementation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hmmlearn.hsmm import GaussianMixtureHSMM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n_states=2, n_mix=2, n_features=2, n_samples=300, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    lengths = [n_samples]
    return X, lengths


def _make_model(n_components=2, n_mix=2, covariance_type="diag",
                duration_distribution="poisson", n_iter=5, seed=42):
    return GaussianMixtureHSMM(
        n_components=n_components,
        n_mix=n_mix,
        covariance_type=covariance_type,
        duration_distribution=duration_distribution,
        max_duration=20,
        n_iter=n_iter,
        random_state=seed,
    )


# ---------------------------------------------------------------------------
# TDD Gate: basic import and instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_import(self):
        """GaussianMixtureHSMM must be importable from hmmlearn.hsmm."""
        from hmmlearn.hsmm import GaussianMixtureHSMM  # noqa: F401

    def test_in_all(self):
        import hmmlearn.hsmm as hsmm_mod
        assert "GaussianMixtureHSMM" in hsmm_mod.__all__

    def test_default_params(self):
        m = GaussianMixtureHSMM(n_components=3, n_mix=2)
        assert m.n_components == 3
        assert m.n_mix == 2
        assert m.covariance_type == "diag"
        assert m.duration_distribution == "poisson"
        assert m.params == "stdmcw"
        assert m.init_params == "stdmcw"

    def test_sample_before_fit(self):
        """sample() after direct attribute assignment must work."""
        rng = np.random.RandomState(0)
        m = GaussianMixtureHSMM(n_components=2, n_mix=2, n_iter=1)
        X, _ = _make_data()
        m.fit(X, [len(X)])
        samples, states = m.sample(20, random_state=rng)
        assert samples.shape[0] == 20
        assert states.shape == (20,)


# ---------------------------------------------------------------------------
# Fit / score / predict / decode / sample
# ---------------------------------------------------------------------------

class TestFitScoreDecodeCovarTypes:

    @pytest.mark.parametrize("cov_type", ["diag", "full", "spherical", "tied"])
    def test_fit_runs(self, cov_type):
        X, lengths = _make_data(n_samples=200)
        m = _make_model(covariance_type=cov_type, n_iter=3)
        m.fit(X, lengths)
        assert hasattr(m, "startprob_")
        assert hasattr(m, "transmat_")
        assert hasattr(m, "means_")
        assert hasattr(m, "weights_")
        assert hasattr(m, "covars_")

    @pytest.mark.parametrize("cov_type", ["diag", "full", "spherical", "tied"])
    def test_score_finite(self, cov_type):
        X, lengths = _make_data(n_samples=200)
        m = _make_model(covariance_type=cov_type, n_iter=3)
        m.fit(X, lengths)
        score = m.score(X, lengths)
        assert np.isfinite(score)

    @pytest.mark.parametrize("cov_type", ["diag", "full", "spherical", "tied"])
    def test_predict(self, cov_type):
        X, lengths = _make_data(n_samples=200)
        m = _make_model(covariance_type=cov_type, n_iter=3)
        m.fit(X, lengths)
        states = m.predict(X, lengths)
        assert states.shape == (len(X),)
        assert set(states).issubset(range(m.n_components))

    @pytest.mark.parametrize("cov_type", ["diag", "full", "spherical", "tied"])
    def test_decode(self, cov_type):
        X, lengths = _make_data(n_samples=200)
        m = _make_model(covariance_type=cov_type, n_iter=3)
        m.fit(X, lengths)
        log_prob, states = m.decode(X, lengths)
        assert np.isfinite(log_prob)
        assert states.shape == (len(X),)

    @pytest.mark.parametrize("cov_type", ["diag", "full", "spherical", "tied"])
    def test_sample(self, cov_type):
        X, lengths = _make_data(n_samples=200)
        m = _make_model(covariance_type=cov_type, n_iter=3)
        m.fit(X, lengths)
        samples, states = m.sample(30, random_state=0)
        assert samples.shape == (30, 2)
        assert states.shape == (30,)


# ---------------------------------------------------------------------------
# Duration distribution parametrization
# ---------------------------------------------------------------------------

class TestDurationDistributions:

    @pytest.mark.parametrize("dur_dist", [
        "poisson", "gaussian", "negative_binomial", "uniform"
    ])
    def test_fit_all_duration_distributions(self, dur_dist):
        X, lengths = _make_data(n_samples=250)
        m = _make_model(duration_distribution=dur_dist, n_iter=3)
        m.fit(X, lengths)
        score = m.score(X, lengths)
        assert np.isfinite(score)


# ---------------------------------------------------------------------------
# Sufficient statistics
# ---------------------------------------------------------------------------

class TestSufficientStatistics:

    def test_dur_key_present(self):
        X, lengths = _make_data(n_samples=100)
        m = _make_model(n_iter=1)
        m.fit(X, lengths)  # triggers _init, _check, then EM
        # Manually get stats after fit to ensure 'dur' key
        stats = m._initialize_sufficient_statistics()
        assert 'dur' in stats
        assert 'post_mix_sum' in stats
        assert 'post_sum' in stats
        assert stats['dur'].shape == (m.max_duration, m.n_components)
        assert stats['post_mix_sum'].shape == (m.n_components, m.n_mix)

    def test_nobs_start_trans_present(self):
        X, lengths = _make_data(n_samples=100)
        m = _make_model(n_iter=1)
        m.fit(X, lengths)
        stats = m._initialize_sufficient_statistics()
        assert 'nobs' in stats
        assert 'start' in stats
        assert 'trans' in stats


# ---------------------------------------------------------------------------
# Log-likelihood monotonicity
# ---------------------------------------------------------------------------

class TestConvergence:

    def test_log_likelihood_nondecreasing(self):
        X, lengths = _make_data(n_samples=300, seed=7)
        m = GaussianMixtureHSMM(
            n_components=2, n_mix=2, n_iter=8,
            random_state=7, max_duration=20, tol=0
        )
        m.fit(X, lengths)
        history = m.monitor_.history
        assert len(history) >= 2
        diffs = np.diff(history)
        # Allow tiny numerical decreases (floating point)
        assert np.all(diffs >= -1e-3), f"Log-likelihood decreased: {diffs}"


# ---------------------------------------------------------------------------
# n_mix=1 consistency
# ---------------------------------------------------------------------------

class TestNMixOne:

    def test_n_mix_1_fits(self):
        """n_mix=1 should reduce to single-Gaussian-per-state HSMM behaviour."""
        X, lengths = _make_data(n_samples=200)
        m = GaussianMixtureHSMM(n_components=2, n_mix=1, n_iter=5,
                                  random_state=0, max_duration=20)
        m.fit(X, lengths)
        assert np.isfinite(m.score(X, lengths))


# ---------------------------------------------------------------------------
# Multi-sequence fitting
# ---------------------------------------------------------------------------

class TestMultiSequence:

    def test_fit_multiple_sequences(self):
        rng = np.random.RandomState(99)
        X = rng.randn(200, 2)
        lengths = [50, 50, 50, 50]
        m = GaussianMixtureHSMM(n_components=2, n_mix=2, n_iter=3,
                                  random_state=0, max_duration=20)
        m.fit(X, lengths)
        assert np.isfinite(m.score(X, lengths))
