"""
The :mod:`hmmlearn.hsmm` module implements Hidden Semi-Markov Models.

A Hidden Semi-Markov Model (HSMM) extends the HMM by explicitly modeling
the duration (sojourn time) that the process spends in each state.  The
standard HMM assumes a geometric duration distribution implicitly; an HSMM
replaces that with an explicit, per-state duration distribution chosen from
a family supplied by the user.

Supported duration distributions
---------------------------------
* ``"poisson"``   – Poisson(lambda_i), support {1, 2, ...}
* ``"gaussian"``  – Gaussian(mu_i, sigma_i) discretised and truncated to
                    {1, ..., D_max}, parameterised as mean / std
* ``"negative_binomial"`` – NegBin(r_i, p_i), support {1, 2, ...}
* ``"uniform"``   – Discrete uniform on {d_min_i, ..., d_max_i}
"""

import logging
import string

import numpy as np
from scipy import special
from scipy.stats import nbinom, poisson as sp_poisson
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state

from . import _utils
from .base import ConvergenceMonitor
from .utils import normalize, log_normalize


__all__ = [
    "GaussianHSMM",
    "CategoricalHSMM",
    "PoissonHSMM",
]

_log = logging.getLogger(__name__)

#: Maximum duration to consider (observations).  Any duration longer than this
#: is treated as having probability 0.  Users can override per-instance via the
#: ``max_duration`` constructor argument.
_DEFAULT_MAX_DURATION = 30


# ---------------------------------------------------------------------------
# Duration distribution helpers
# ---------------------------------------------------------------------------

class _PoissonDuration:
    """Poisson duration distribution (support starts at 1)."""

    name = "poisson"

    def __init__(self, n_components, random_state, max_duration):
        self.n_components = n_components
        self.random_state = random_state
        self.max_duration = max_duration

    def initialize(self, durations_hint=None):
        """Randomly initialise lambda_ (mean duration per state)."""
        rng = check_random_state(self.random_state)
        if durations_hint is not None:
            self.lambda_ = np.asarray(durations_hint, dtype=float)
        else:
            self.lambda_ = rng.uniform(2, max(3, self.max_duration // 2),
                                       size=self.n_components)

    def log_pmf(self):
        """Return log P(d | state) for d in 1 .. max_duration.

        Returns
        -------
        log_pmf : ndarray, shape (max_duration, n_components)
        """
        d = np.arange(1, self.max_duration + 1)  # (D,)
        # log Poisson pmf at d, but shifted to start at 1
        # P(D=d) ~ Poisson(d; lambda),  d >= 1  (unnormalised)
        lp = sp_poisson.logpmf(d[:, None], self.lambda_[None, :])  # (D, N)
        # Renormalise so that sum over d=1..D_max equals 1 per state.
        lp -= special.logsumexp(lp, axis=0, keepdims=True)
        return lp

    def sample(self, state, rng):
        """Sample a duration for the given state (at least 1)."""
        d = 0
        while d < 1:
            d = rng.poisson(self.lambda_[state])
        return d

    def mstep(self, expected_durations):
        """M-step: update lambda_ from expected duration counts.

        Parameters
        ----------
        expected_durations : ndarray, shape (max_duration, n_components)
            E[N(d, i)] – expected number of times state i had duration d.
        """
        d = np.arange(1, self.max_duration + 1)  # (D,)
        denom = expected_durations.sum(axis=0)  # (N,)
        denom = np.maximum(denom, 1e-10)
        self.lambda_ = (d[:, None] * expected_durations).sum(axis=0) / denom

    def get_params(self):
        return {"lambda_": self.lambda_.copy()}

    def set_params(self, lambda_):
        self.lambda_ = np.asarray(lambda_, dtype=float)

    def n_fit_scalars(self):
        return self.n_components  # one lambda per state


class _GaussianDuration:
    """Discretised Gaussian duration distribution (support 1..D_max)."""

    name = "gaussian"

    def __init__(self, n_components, random_state, max_duration):
        self.n_components = n_components
        self.random_state = random_state
        self.max_duration = max_duration

    def initialize(self, durations_hint=None):
        rng = check_random_state(self.random_state)
        if durations_hint is not None:
            arr = np.asarray(durations_hint, dtype=float)
            self.means_ = arr
            self.stds_ = np.maximum(arr * 0.3, 1.0)
        else:
            self.means_ = rng.uniform(2, max(3, self.max_duration // 2),
                                      size=self.n_components)
            self.stds_ = np.ones(self.n_components) * max(1.0,
                                                           self.max_duration / 10)

    def log_pmf(self):
        """Return log P(d | state) for d in 1 .. max_duration."""
        from scipy.stats import norm
        d = np.arange(1, self.max_duration + 1)  # (D,)
        # Discretise: P(D=d) ∝ phi((d - mu)/sigma)
        lp = norm.logpdf(d[:, None], self.means_[None, :],
                         self.stds_[None, :])  # (D, N)
        lp -= special.logsumexp(lp, axis=0, keepdims=True)
        return lp

    def sample(self, state, rng):
        d = 0
        while d < 1 or d > self.max_duration:
            d = int(round(rng.normal(self.means_[state], self.stds_[state])))
        return d

    def mstep(self, expected_durations):
        d = np.arange(1, self.max_duration + 1)  # (D,)
        denom = expected_durations.sum(axis=0)  # (N,)
        denom = np.maximum(denom, 1e-10)
        self.means_ = (d[:, None] * expected_durations).sum(axis=0) / denom
        self.stds_ = np.sqrt(
            ((d[:, None] - self.means_[None, :]) ** 2
             * expected_durations).sum(axis=0) / denom
        )
        self.stds_ = np.maximum(self.stds_, 0.5)  # avoid degeneracy

    def get_params(self):
        return {"means_": self.means_.copy(), "stds_": self.stds_.copy()}

    def set_params(self, means_, stds_):
        self.means_ = np.asarray(means_, dtype=float)
        self.stds_ = np.asarray(stds_, dtype=float)

    def n_fit_scalars(self):
        return 2 * self.n_components  # mean + std per state


class _NegativeBinomialDuration:
    """Negative-Binomial duration distribution (support starts at 1).

    Parameterised as (r, p) where r > 0 and 0 < p < 1, with mean r(1-p)/p.
    """

    name = "negative_binomial"

    def __init__(self, n_components, random_state, max_duration):
        self.n_components = n_components
        self.random_state = random_state
        self.max_duration = max_duration

    def initialize(self, durations_hint=None):
        rng = check_random_state(self.random_state)
        if durations_hint is not None:
            means = np.asarray(durations_hint, dtype=float)
        else:
            means = rng.uniform(2, max(3, self.max_duration // 2),
                                size=self.n_components)
        # Initialise r=2 and set p from desired mean: mean = r(1-p)/p => p = r/(mean+r)
        self.r_ = np.full(self.n_components, 2.0)
        self.p_ = self.r_ / (means + self.r_)

    def log_pmf(self):
        d = np.arange(1, self.max_duration + 1)  # (D,)
        # NegBin support is {0, 1, ...}; we normalise over {1...D_max}.
        lp = nbinom.logpmf(d[:, None] - 1, self.r_[None, :],
                           self.p_[None, :])  # (D, N)
        lp -= special.logsumexp(lp, axis=0, keepdims=True)
        return lp

    def sample(self, state, rng):
        d = 0
        while d < 1:
            d = int(nbinom.rvs(self.r_[state], self.p_[state],
                               random_state=rng)) + 1
        return d

    def mstep(self, expected_durations):
        d = np.arange(1, self.max_duration + 1)
        denom = expected_durations.sum(axis=0)
        denom = np.maximum(denom, 1e-10)
        means = (d[:, None] * expected_durations).sum(axis=0) / denom
        variances = (
            ((d[:, None] - means[None, :]) ** 2 * expected_durations).sum(axis=0)
            / denom
        )
        variances = np.maximum(variances, means + 1e-6)
        # Method of moments: r = mu^2 / (var - mu),  p = mu / var
        self.r_ = means ** 2 / np.maximum(variances - means, 1e-6)
        self.p_ = means / variances
        self.p_ = np.clip(self.p_, 1e-6, 1 - 1e-6)

    def get_params(self):
        return {"r_": self.r_.copy(), "p_": self.p_.copy()}

    def set_params(self, r_, p_):
        self.r_ = np.asarray(r_, dtype=float)
        self.p_ = np.asarray(p_, dtype=float)

    def n_fit_scalars(self):
        return 2 * self.n_components


class _UniformDuration:
    """Discrete Uniform duration distribution on {d_min_i .. d_max_i}.

    Parameters are learnt by tracking the empirical support.
    """

    name = "uniform"

    def __init__(self, n_components, random_state, max_duration):
        self.n_components = n_components
        self.random_state = random_state
        self.max_duration = max_duration

    def initialize(self, durations_hint=None):
        rng = check_random_state(self.random_state)
        if durations_hint is not None:
            means = np.asarray(durations_hint, dtype=float)
            half = np.maximum(means * 0.4, 1)
            self.d_min_ = np.maximum(np.round(means - half).astype(int), 1)
            self.d_max_ = np.minimum(
                np.round(means + half).astype(int), self.max_duration)
        else:
            self.d_min_ = np.ones(self.n_components, dtype=int)
            # Default: cover the entire max_duration range so all observations
            # have non-zero duration probability during initialisation.
            self.d_max_ = np.full(self.n_components, self.max_duration, dtype=int)

    def log_pmf(self):
        lp = np.full((self.max_duration, self.n_components), -np.inf)
        for i in range(self.n_components):
            lo, hi = self.d_min_[i], self.d_max_[i]
            lp[lo - 1: hi, i] = -np.log(hi - lo + 1)
        return lp

    def sample(self, state, rng):
        return int(rng.randint(self.d_min_[state], self.d_max_[state] + 1))

    def mstep(self, expected_durations):
        """Update support boundaries using the 5th and 95th percentile."""
        d = np.arange(1, self.max_duration + 1)
        denom = expected_durations.sum(axis=0)
        denom = np.maximum(denom, 1e-10)
        cum = np.cumsum(expected_durations / denom[None, :], axis=0)
        for i in range(self.n_components):
            lo_idx = np.searchsorted(cum[:, i], 0.05)
            hi_idx = np.searchsorted(cum[:, i], 0.95)
            self.d_min_[i] = max(1, d[lo_idx])
            self.d_max_[i] = min(self.max_duration, d[hi_idx])
            if self.d_min_[i] > self.d_max_[i]:
                self.d_max_[i] = self.d_min_[i]

    def get_params(self):
        return {"d_min_": self.d_min_.copy(), "d_max_": self.d_max_.copy()}

    def set_params(self, d_min_, d_max_):
        self.d_min_ = np.asarray(d_min_, dtype=int)
        self.d_max_ = np.asarray(d_max_, dtype=int)

    def n_fit_scalars(self):
        return 2 * self.n_components


_DURATION_DISTRIBUTIONS = {
    "poisson": _PoissonDuration,
    "gaussian": _GaussianDuration,
    "negative_binomial": _NegativeBinomialDuration,
    "uniform": _UniformDuration,
}


# ---------------------------------------------------------------------------
# Base HSMM
# ---------------------------------------------------------------------------

class BaseHSMM(BaseEstimator):
    """
    Base class for Hidden Semi-Markov Models trained via EM.

    Subclasses must implement the observation-model methods (same interface as
    :class:`~hmmlearn.base.BaseHMM` subclasses):

    * ``_compute_log_likelihood(X)``
    * ``_generate_sample_from_state(state, random_state)``
    * ``_init(X, lengths)``  – call ``super()._init(X, lengths)`` first
    * ``_check()``          – call ``super()._check()`` first
    * ``_get_n_fit_scalars_per_param()``
    * ``_initialize_sufficient_statistics()``
    * ``_accumulate_emission_sufficient_statistics(stats, X, posteriors)``
    * ``_do_emission_mstep(stats)``

    The HSMM forward-backward uses the so-called *explicit-duration* algorithm
    (Murphy 2002; Yu 2010).  The implementation works entirely in log-space
    for numerical stability.

    Parameters
    ----------
    n_components : int
        Number of hidden states.
    duration_distribution : {"poisson", "gaussian", "negative_binomial",
                             "uniform"}
        Family of the per-state duration distribution.
    max_duration : int
        Maximum duration to model explicitly.  Longer runs are ignored.
    startprob_prior : float or array, shape (n_components,)
        Dirichlet prior on start probabilities.
    transmat_prior : float or array, shape (n_components, n_components)
        Dirichlet prior on transition probabilities.  The diagonal must be
        zero for an HSMM (self-transitions are forbidden).
    algorithm : {"viterbi", "map"}
        Decoder algorithm.
    random_state : int or RandomState, optional
    n_iter : int
        EM iterations.
    tol : float
        Convergence threshold.
    verbose : bool
    params : str
        Characters that select which parameters are updated in the M-step.
        ``'s'`` start, ``'t'`` transmat, ``'d'`` duration, plus
        emission-specific characters defined in subclasses.
    init_params : str
        Characters that select which parameters are initialised before EM.
    """

    def __init__(self, n_components=1,
                 duration_distribution="poisson",
                 max_duration=_DEFAULT_MAX_DURATION,
                 startprob_prior=1.0,
                 transmat_prior=1.0,
                 algorithm="viterbi",
                 random_state=None,
                 n_iter=10,
                 tol=1e-2,
                 verbose=False,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters):
        self.n_components = n_components
        self.duration_distribution = duration_distribution
        self.max_duration = max_duration
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.params = params
        self.init_params = init_params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _needs_init(self, code, name):
        if code in self.init_params:
            if hasattr(self, name):
                _log.warning(
                    "Even though the %r attribute is set, it will be "
                    "overwritten during initialization because 'init_params' "
                    "contains %r", name, code)
            return True
        return not hasattr(self, name)

    def _build_duration_model(self):
        if self.duration_distribution not in _DURATION_DISTRIBUTIONS:
            raise ValueError(
                f"Unknown duration_distribution {self.duration_distribution!r}."
                f"  Supported options: {sorted(_DURATION_DISTRIBUTIONS)}")
        cls = _DURATION_DISTRIBUTIONS[self.duration_distribution]
        return cls(self.n_components,
                   self.random_state,
                   self.max_duration)

    # ------------------------------------------------------------------
    # Public API (sklearn-compatible)
    # ------------------------------------------------------------------

    def fit(self, X, lengths=None):
        """
        Estimate model parameters via EM.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        lengths : array-like of int, optional
            Lengths of individual sequences.  Defaults to one sequence.

        Returns
        -------
        self
        """
        X = check_array(X)
        if lengths is None:
            lengths = np.array([X.shape[0]])

        self._init(X, lengths)
        self._check()
        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)
        self.monitor_._reset()

        for _ in range(self.n_iter):
            stats, curr_logprob = self._do_estep(X, lengths)
            self._do_mstep(stats)
            self.monitor_.report(curr_logprob)
            if self.monitor_.converged:
                break
        return self

    def score(self, X, lengths=None):
        """
        Compute the log probability of ``X`` under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        lengths : array-like of int, optional

        Returns
        -------
        log_prob : float
        """
        check_is_fitted(self, "startprob_")
        self._check()
        X = check_array(X)
        log_prob = 0.0
        for sub_X in _utils.split_X_lengths(X, lengths):
            log_frameprob = self._compute_log_likelihood(sub_X)
            lp, _, _ = self._forward(log_frameprob)
            log_prob += lp
        return log_prob

    def score_samples(self, X, lengths=None):
        """
        Compute the log probability and per-frame state posteriors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        lengths : array-like of int, optional

        Returns
        -------
        log_prob : float
        posteriors : ndarray, shape (n_samples, n_components)
        """
        check_is_fitted(self, "startprob_")
        self._check()
        X = check_array(X)
        log_prob = 0.0
        all_posteriors = []
        for sub_X in _utils.split_X_lengths(X, lengths):
            log_frameprob = self._compute_log_likelihood(sub_X)
            lp, fwd, _scale = self._forward(log_frameprob)
            bwd = self._backward(log_frameprob)
            log_prob += lp
            log_gamma = fwd + bwd
            log_normalize(log_gamma, axis=1)
            with np.errstate(under="ignore"):
                all_posteriors.append(np.exp(log_gamma))
        return log_prob, np.concatenate(all_posteriors, axis=0)

    def predict(self, X, lengths=None):
        """
        Find most likely state sequence.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        lengths : array-like of int, optional

        Returns
        -------
        state_sequence : ndarray, shape (n_samples,)
        """
        _, state_sequence = self.decode(X, lengths)
        return state_sequence

    def predict_proba(self, X, lengths=None):
        """
        Compute per-frame posterior state probabilities.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        lengths : array-like of int, optional

        Returns
        -------
        posteriors : ndarray, shape (n_samples, n_components)
        """
        _, posteriors = self.score_samples(X, lengths)
        return posteriors

    def decode(self, X, lengths=None, algorithm=None):
        """
        Find the most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        lengths : array-like of int, optional
        algorithm : {"viterbi", "map"}, optional
            Overrides :attr:`algorithm`.

        Returns
        -------
        log_prob : float
        state_sequence : ndarray, shape (n_samples,)
        """
        check_is_fitted(self, "startprob_")
        self._check()
        algorithm = algorithm or self.algorithm
        X = check_array(X)
        log_prob = 0.0
        all_states = []
        for sub_X in _utils.split_X_lengths(X, lengths):
            log_frameprob = self._compute_log_likelihood(sub_X)
            if algorithm == "viterbi":
                lp, states = self._viterbi(log_frameprob)
            elif algorithm == "map":
                lp, _, _ = self._forward(log_frameprob)
                bwd = self._backward(log_frameprob)
                log_gamma = lp + bwd  # reuse fwd; but compute properly
                # Redo properly:
                fwd_lattice, _, _ = self._forward(log_frameprob)
                bwd_lattice = self._backward(log_frameprob)
                log_gamma = fwd_lattice + bwd_lattice
                log_normalize(log_gamma, axis=1)
                states = np.argmax(log_gamma, axis=1)
                lp = log_gamma.max(axis=1).sum()
            else:
                raise ValueError(f"Unknown algorithm {algorithm!r}")
            log_prob += lp
            all_states.append(states)
        return log_prob, np.concatenate(all_states)

    def sample(self, n_samples=1, random_state=None, currstate=None):
        """
        Generate random samples from the model.

        Parameters
        ----------
        n_samples : int
        random_state : int or RandomState, optional
        currstate : int, optional
            Initial state.

        Returns
        -------
        X : ndarray, shape (n_samples, n_features)
        state_sequence : ndarray, shape (n_samples,)
        """
        check_is_fitted(self, "startprob_")
        self._check()

        rng = check_random_state(random_state if random_state is not None
                                 else self.random_state)

        dur_model = self._duration_model_

        # Build CDF of transmat_ without self-loops (already forced to 0 diagonal)
        transmat_no_self = self.transmat_.copy()
        np.fill_diagonal(transmat_no_self, 0.0)
        row_sums = transmat_no_self.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        transmat_cdf = np.cumsum(transmat_no_self / row_sums, axis=1)

        if currstate is None:
            startprob_cdf = np.cumsum(self.startprob_)
            currstate = int((startprob_cdf > rng.rand()).argmax())

        X_list = []
        state_list = []
        while len(X_list) < n_samples:
            d = min(dur_model.sample(currstate, rng),
                    n_samples - len(X_list))
            for _ in range(d):
                X_list.append(
                    self._generate_sample_from_state(currstate, rng))
                state_list.append(currstate)
            # Transition to new state (no self-loop)
            currstate = int(
                (transmat_cdf[currstate] > rng.rand()).argmax())

        X = np.atleast_2d(X_list[:n_samples])
        states = np.array(state_list[:n_samples], dtype=int)
        return X, states

    # ------------------------------------------------------------------
    # Internal EM machinery
    # ------------------------------------------------------------------

    def _do_estep(self, X, lengths):
        stats = self._initialize_sufficient_statistics()
        curr_logprob = 0.0
        for sub_X in _utils.split_X_lengths(X, lengths):
            log_frameprob = self._compute_log_likelihood(sub_X)
            lp, fwd, dur_fwd = self._forward(log_frameprob)
            bwd = self._backward(log_frameprob)
            curr_logprob += lp

            T, N = log_frameprob.shape
            D = self.max_duration

            # --- state posteriors gamma[t, i] = log P(s_t = i | X) ---
            log_gamma = fwd + bwd
            with np.errstate(under="ignore", invalid="ignore"):
                log_normalize(log_gamma, axis=1)
                gamma = np.exp(log_gamma)
            # Guard against NaN rows (all-inf rows in log_gamma)
            gamma = np.where(np.isfinite(gamma), gamma, 0.0)

            # --- sufficient statistics for transitions ---
            # xi[t, i, j] = P(end segment of state i at t, next state j | X)
            # We compute the un-normalised version in log space.
            log_dur = self._duration_model_.log_pmf()  # (D, N)
            log_trans = np.log(
                np.maximum(self.transmat_, 1e-300))  # (N, N)

            log_xi_sum = np.full((N, N), -np.inf)
            for t in range(T - 1):
                for d in range(1, min(D, t + 1) + 1):
                    t0 = t - d + 1  # segment start
                    if t0 < 0:
                        continue
                    # Emission score for segment [t0..t] for each state
                    seg_log_emit = log_frameprob[t0:t + 1, :]  # (d, N)
                    log_emit_sum = seg_log_emit.sum(axis=0)  # (N,)
                    # fwd at t0-1 (or log startprob if t0==0)
                    if t0 == 0:
                        log_alpha_prev = np.log(
                            np.maximum(self.startprob_, 1e-300))
                    else:
                        log_alpha_prev = fwd[t0 - 1]
                    # bwd at t+1
                    log_beta_next = bwd[t + 1] if t + 1 < T else \
                        np.zeros(N)
                    # For each (i, j): alpha[t0-1, i]*dur[d,i]*emit[i]*trans[i,j]*beta[t+1,j]
                    log_seg_prob = (log_alpha_prev[:, None]
                                    + log_dur[d - 1, :, None]
                                    + log_emit_sum[:, None]
                                    + log_trans
                                    + log_beta_next[None, :])  # (N, N)
                    log_xi_sum = np.logaddexp(log_xi_sum, log_seg_prob)

            with np.errstate(under="ignore"):
                xi_sum = np.exp(log_xi_sum - lp)

            stats['nobs'] += 1
            if 's' in self.params:
                stats['start'] += gamma[0]
            if 't' in self.params:
                stats['trans'] += xi_sum

            # --- duration expected counts eta[d, i] ---
            # eta[d, i] = sum over all segments of state i with duration d
            eta = np.zeros((D, N))
            for t in range(T):
                for d in range(1, min(D, t + 1) + 1):
                    t0 = t - d + 1
                    if t0 < 0:
                        continue
                    seg_log_emit = log_frameprob[t0:t + 1, :]
                    log_emit_sum = seg_log_emit.sum(axis=0)
                    if t0 == 0:
                        log_alpha_prev = np.log(
                            np.maximum(self.startprob_, 1e-300))
                    else:
                        log_alpha_prev = fwd[t0 - 1]
                    log_seg = (log_alpha_prev
                               + log_dur[d - 1]
                               + log_emit_sum
                               + bwd[t])  # (N,)
                    with np.errstate(under="ignore"):
                        eta[d - 1] += np.exp(log_seg - lp)

            if 'd' in self.params:
                stats['dur'] += eta

            # Accumulate emission stats
            self._accumulate_emission_sufficient_statistics(
                stats, sub_X, gamma)

        return stats, curr_logprob

    def _do_mstep(self, stats):
        if 's' in self.params:
            startprob_ = np.maximum(
                self.startprob_prior - 1 + stats['start'], 0)
            self.startprob_ = np.where(
                self.startprob_ == 0, 0, startprob_)
            s = self.startprob_.sum()
            if s > 0:
                self.startprob_ /= s
            else:
                self.startprob_ = np.full(self.n_components,
                                          1.0 / self.n_components)
        if 't' in self.params and self.n_components > 1:
            transmat_ = np.maximum(
                self.transmat_prior - 1 + stats['trans'], 0)
            np.fill_diagonal(transmat_, 0.0)
            self.transmat_ = np.where(
                self.transmat_ == 0, 0, transmat_)
            row_sums = self.transmat_.sum(axis=1, keepdims=True)
            # Rows that sum to zero get uniform distribution (excluding self)
            zero_rows = (row_sums.squeeze() == 0)
            if zero_rows.any():
                for i in np.where(zero_rows)[0]:
                    self.transmat_[i] = 1.0 / max(self.n_components - 1, 1)
                    self.transmat_[i, i] = 0.0
            normalize(self.transmat_, axis=1)
        if 'd' in self.params:
            self._duration_model_.mstep(stats['dur'])
        self._do_emission_mstep(stats)

    # ------------------------------------------------------------------
    # Forward-backward (log-space, explicit duration)
    # ------------------------------------------------------------------

    def _forward(self, log_frameprob):
        """
        Explicit-duration forward algorithm.

        Parameters
        ----------
        log_frameprob : ndarray, shape (T, N)

        Returns
        -------
        log_prob : float
        fwd : ndarray, shape (T, N)
            log alpha[t, i] = log P(o_{1..t}, s_t = i)
        dur_fwd : ndarray, shape (T, D, N)  (kept for potential future use)
        """
        T, N = log_frameprob.shape
        D = self.max_duration
        log_dur = self._duration_model_.log_pmf()   # (D, N)
        log_pi = np.log(np.maximum(self.startprob_, 1e-300))   # (N,)
        log_trans = np.log(np.maximum(self.transmat_, 1e-300))  # (N, N)

        # fwd[t, i] = log P(o_{0..t}, q_t = i)
        fwd = np.full((T, N), -np.inf)

        for t in range(T):
            for d in range(1, min(D, t + 1) + 1):
                t0 = t - d + 1
                # Emission score for segment [t0 .. t]
                seg_emit = log_frameprob[t0:t + 1, :].sum(axis=0)  # (N,)
                if t0 == 0:
                    # Segment starts at beginning – use start distribution
                    incoming = log_pi  # (N,)
                else:
                    # Incoming: sum over previous states j at t0-1
                    # log P(o_{0..t0-1}, q_{t0-1}=j) + log a_{j,i}
                    incoming = special.logsumexp(
                        fwd[t0 - 1, :, None] + log_trans,
                        axis=0)  # (N,)
                fwd[t] = np.logaddexp(fwd[t],
                                      incoming + log_dur[d - 1] + seg_emit)

        log_prob = special.logsumexp(fwd[T - 1])
        return log_prob, fwd, None  # dur_fwd omitted for memory

    def _backward(self, log_frameprob):
        """
        Explicit-duration backward algorithm.

        Returns
        -------
        bwd : ndarray, shape (T, N)
            log beta[t, i] = log P(o_{t+1..T-1} | q_t = i)
        """
        T, N = log_frameprob.shape
        D = self.max_duration
        log_dur = self._duration_model_.log_pmf()   # (D, N)
        log_trans = np.log(np.maximum(self.transmat_, 1e-300))  # (N, N)

        bwd = np.full((T, N), -np.inf)
        bwd[T - 1] = 0.0  # log(1) = 0

        for t in range(T - 2, -1, -1):
            for d in range(1, min(D, T - t - 1) + 1):
                t1 = t + d  # end of next segment (inclusive)
                if t1 >= T:
                    continue
                seg_emit = log_frameprob[t + 1:t1 + 1, :].sum(axis=0)  # (N,)
                # bwd[t, i] = logsumexp_j a_{i,j} * dur[d,j] * emit_j * bwd[t1, j]
                contrib = (log_trans[:, :]  # (N, N): log a_{i,j}
                           + log_dur[d - 1, None, :]  # (1, N): log p_j(d)
                           + seg_emit[None, :]         # (1, N)
                           + bwd[t1, None, :])         # (1, N)
                bwd[t] = np.logaddexp(
                    bwd[t],
                    special.logsumexp(contrib, axis=1))  # sum over j -> (N,)

        return bwd

    def _viterbi(self, log_frameprob):
        """
        Explicit-duration Viterbi decoding.

        Returns
        -------
        log_prob : float
        state_sequence : ndarray, shape (T,)
        """
        T, N = log_frameprob.shape
        D = self.max_duration
        log_dur = self._duration_model_.log_pmf()   # (D, N)
        log_pi = np.log(np.maximum(self.startprob_, 1e-300))
        log_trans = np.log(np.maximum(self.transmat_, 1e-300))

        # delta[t, i] = best log-prob reaching state i ending AT time t
        delta = np.full((T, N), -np.inf)
        psi_state = np.zeros((T, N), dtype=int)  # best previous state
        psi_dur = np.ones((T, N), dtype=int)      # best duration

        for t in range(T):
            for d in range(1, min(D, t + 1) + 1):
                t0 = t - d + 1
                seg_emit = log_frameprob[t0:t + 1, :].sum(axis=0)
                if t0 == 0:
                    score = log_pi + log_dur[d - 1] + seg_emit
                    prev_state = np.zeros(N, dtype=int)
                else:
                    # delta[t0-1, j] + log_trans[j, i] for best j
                    combined = delta[t0 - 1, :, None] + log_trans  # (N, N)
                    best_j = np.argmax(combined, axis=0)  # (N,)
                    best_val = combined[best_j, np.arange(N)]
                    score = best_val + log_dur[d - 1] + seg_emit
                    prev_state = best_j

                improved = score > delta[t]
                delta[t] = np.where(improved, score, delta[t])
                psi_state[t] = np.where(improved, prev_state, psi_state[t])
                psi_dur[t] = np.where(improved, d, psi_dur[t])

        # Back-track
        state_sequence = np.empty(T, dtype=int)
        t = T - 1
        log_prob = np.max(delta[t])
        curr_state = int(np.argmax(delta[t]))
        while t >= 0:
            d = psi_dur[t, curr_state]
            t0 = t - d + 1
            state_sequence[t0:t + 1] = curr_state
            if t0 > 0:
                curr_state = psi_state[t, curr_state]
            t = t0 - 1

        return log_prob, state_sequence

    # ------------------------------------------------------------------
    # Initialisation & validation
    # ------------------------------------------------------------------

    def _init(self, X, lengths=None):
        """Initialise start, transmat, and duration model."""
        self._check_and_set_n_features(X)
        N = self.n_components
        rng = check_random_state(self.random_state)

        if self._needs_init("s", "startprob_"):
            self.startprob_ = rng.dirichlet(np.ones(N))

        if self._needs_init("t", "transmat_"):
            if N == 1:
                # Single-state HSMM: transmat must be all-zeros (no transitions)
                self.transmat_ = np.zeros((1, 1))
            else:
                # Initialise without self-loops
                A = rng.dirichlet(np.ones(N), size=N)
                np.fill_diagonal(A, 0.0)
                normalize(A, axis=1)
                self.transmat_ = A

        # Duration distribution
        if not hasattr(self, "_duration_model_"):
            self._duration_model_ = self._build_duration_model()
            self._duration_model_.initialize()

    def _check(self):
        self.startprob_ = np.asarray(self.startprob_)
        if len(self.startprob_) != self.n_components:
            raise ValueError("startprob_ must have length n_components")
        s = self.startprob_.sum()
        if not np.isclose(s, 1.0):
            raise ValueError(f"startprob_ must sum to 1 (got {s})")

        self.transmat_ = np.asarray(self.transmat_)
        if self.transmat_.shape != (self.n_components, self.n_components):
            raise ValueError(
                "transmat_ must have shape (n_components, n_components)")
        if np.any(np.diag(self.transmat_) != 0):
            raise ValueError(
                "HSMM transmat_ must have zero diagonal (no self-transitions)")
        # A single-state HSMM has no valid transitions (1x1 zero matrix).
        if self.n_components > 1:
            s = self.transmat_.sum(axis=1)
            if not np.allclose(s, 1.0):
                raise ValueError(
                    f"transmat_ rows must sum to 1 (got {s})")

        if not hasattr(self, "_duration_model_"):
            self._duration_model_ = self._build_duration_model()
            self._duration_model_.initialize()

    def _initialize_sufficient_statistics(self):
        N = self.n_components
        D = self.max_duration
        return {
            'nobs': 0,
            'start': np.zeros(N),
            'trans': np.zeros((N, N)),
            'dur': np.zeros((D, N)),
        }

    # ------------------------------------------------------------------
    # Abstract methods – must be overridden in concrete subclasses
    # ------------------------------------------------------------------

    def _check_and_set_n_features(self, X):
        raise NotImplementedError

    def _compute_log_likelihood(self, X):
        """Return log P(x_t | state i) for all t, i.

        Parameters
        ----------
        X : ndarray, shape (T, n_features)

        Returns
        -------
        log_prob : ndarray, shape (T, n_components)
        """
        raise NotImplementedError

    def _generate_sample_from_state(self, state, random_state):
        raise NotImplementedError

    def _get_n_fit_scalars_per_param(self):
        raise NotImplementedError

    def _accumulate_emission_sufficient_statistics(self, stats, X, posteriors):
        """Update emission-related sufficient statistics."""
        raise NotImplementedError

    def _do_emission_mstep(self, stats):
        """Update emission parameters from sufficient statistics."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Information criteria
    # ------------------------------------------------------------------

    def aic(self, X, lengths=None):
        """Akaike information criterion."""
        n_params = sum(self._get_n_fit_scalars_per_param().values())
        return -2 * self.score(X, lengths) + 2 * n_params

    def bic(self, X, lengths=None):
        """Bayesian information criterion."""
        n_params = sum(self._get_n_fit_scalars_per_param().values())
        return -2 * self.score(X, lengths) + n_params * np.log(len(X))


# ---------------------------------------------------------------------------
# Concrete HSMM subclasses
# ---------------------------------------------------------------------------

class GaussianHSMM(BaseHSMM):
    """
    Hidden Semi-Markov Model with Gaussian emissions.

    Each hidden state emits observations drawn from a multivariate Gaussian
    distribution parameterised by a state-specific mean and covariance.

    Parameters
    ----------
    n_components : int
        Number of hidden states.
    covariance_type : {"diag", "full", "spherical", "tied"}
        Type of covariance matrix.
    duration_distribution : {"poisson", "gaussian", "negative_binomial",
                             "uniform"}
        Duration distribution family.
    max_duration : int
        Maximum segment length to model explicitly.
    min_covar : float
        Floor on covariance eigenvalues for numerical stability.
    startprob_prior, transmat_prior : float
        Dirichlet priors.
    means_prior, means_weight : float
        Normal-Wishart prior on means.
    covars_prior, covars_weight : float
        Normal-Wishart prior on covariances.
    algorithm : {"viterbi", "map"}
    random_state : int or RandomState, optional
    n_iter : int
    tol : float
    verbose : bool
    params, init_params : str

    Attributes
    ----------
    means_ : ndarray, shape (n_components, n_features)
    covars_ : ndarray (shape depends on covariance_type)
    startprob_ : ndarray, shape (n_components,)
    transmat_ : ndarray, shape (n_components, n_components)

    Examples
    --------
    >>> from hmmlearn.hsmm import GaussianHSMM
    >>> model = GaussianHSMM(n_components=3, duration_distribution="poisson",
    ...                      max_duration=20, n_iter=20)
    >>> X = model.sample(200)[0]
    >>> model.fit(X)  # doctest: +ELLIPSIS
    GaussianHSMM(...)
    """

    def __init__(self, n_components=1,
                 covariance_type="diag",
                 duration_distribution="poisson",
                 max_duration=_DEFAULT_MAX_DURATION,
                 min_covar=1e-3,
                 startprob_prior=1.0,
                 transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 algorithm="viterbi",
                 random_state=None,
                 n_iter=10, tol=1e-2,
                 verbose=False,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters):
        super().__init__(
            n_components=n_components,
            duration_distribution=duration_distribution,
            max_duration=max_duration,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter, tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params)
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    # --- sklearn-style properties ---

    @property
    def covars_(self):
        return self._covars_

    @covars_.setter
    def covars_(self, covars):
        from .utils import fill_covars
        self._covars_ = fill_covars(
            covars, self.covariance_type, self.n_components,
            self.n_features if hasattr(self, "n_features") else 1)

    # --- feature dimension ---

    def _check_and_set_n_features(self, X):
        _, n_features = X.shape
        if hasattr(self, "n_features"):
            if self.n_features != n_features:
                raise ValueError(
                    f"Unexpected number of dimensions: got {n_features} "
                    f"but expected {self.n_features}")
        else:
            self.n_features = n_features

    # --- initialisation ---

    def _init(self, X, lengths=None):
        super()._init(X, lengths)
        from sklearn import cluster
        N, F = self.n_components, self.n_features

        if self._needs_init("m", "means_"):
            if X.shape[0] >= N:
                kmeans = cluster.KMeans(n_clusters=N,
                                        random_state=self.random_state,
                                        n_init=10)
                kmeans.fit(X)
                self.means_ = kmeans.cluster_centers_
            else:
                # Fewer samples than states: use random means from data range
                rng = check_random_state(self.random_state)
                self.means_ = rng.uniform(X.min(axis=0), X.max(axis=0),
                                           size=(N, F))

        if self._needs_init("c", "covars_"):
            if X.shape[0] > 1:
                cv = np.cov(X.T) + self.min_covar * np.eye(F)
            else:
                cv = self.min_covar * np.eye(F)
            if self.covariance_type == "full":
                self._covars_ = np.tile(cv, (N, 1, 1))
            elif self.covariance_type == "diag":
                self._covars_ = np.tile(np.diag(cv), (N, 1))
            elif self.covariance_type == "spherical":
                self._covars_ = np.full(N, cv.mean())
            elif self.covariance_type == "tied":
                self._covars_ = cv

    def _check(self):
        super()._check()
        from ._utils import _validate_covars
        if not hasattr(self, "n_features"):
            raise ValueError("n_features is not set")
        self.means_ = np.asarray(self.means_)
        if self.means_.shape != (self.n_components, self.n_features):
            raise ValueError(
                f"means_ must have shape (n_components, n_features)")
        _validate_covars(self._covars_, self.covariance_type, self.n_components)

    # --- emission model ---

    def _compute_log_likelihood(self, X):
        from .stats import log_multivariate_normal_density
        return log_multivariate_normal_density(
            X, self.means_, self._covars_, self.covariance_type)

    def _generate_sample_from_state(self, state, random_state):
        from .utils import fill_covars
        cov = fill_covars(self._covars_, self.covariance_type,
                          self.n_components, self.n_features)
        return random_state.multivariate_normal(self.means_[state], cov[state])

    # --- sufficient statistics ---

    def _get_n_fit_scalars_per_param(self):
        N, F = self.n_components, self.n_features
        d = self._duration_model_.n_fit_scalars() if hasattr(
            self, "_duration_model_") else N
        nc = {
            "full": N * F * (F + 1) // 2,
            "diag": N * F,
            "tied": F * (F + 1) // 2,
            "spherical": N,
        }[self.covariance_type]
        return {
            "s": N - 1,
            "t": N * (N - 1),
            "d": d,
            "m": N * F,
            "c": nc,
        }

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        N, F = self.n_components, self.n_features
        stats['obs'] = np.zeros((N, F))
        stats['obs**2'] = np.zeros((N, F))
        stats['obs*obs.T'] = np.zeros((N, F, F))
        stats['post'] = np.zeros(N)
        return stats

    def _accumulate_emission_sufficient_statistics(self, stats, X, posteriors):
        stats['post'] += posteriors.sum(axis=0)
        stats['obs'] += posteriors.T @ X
        if self.covariance_type in ('spherical', 'diag'):
            stats['obs**2'] += posteriors.T @ (X ** 2)
        if self.covariance_type in ('full', 'tied'):
            for t, o in enumerate(X):
                stats['obs*obs.T'] += posteriors[t, :, None, None] * np.outer(o, o)

    def _do_emission_mstep(self, stats):
        N, F = self.n_components, self.n_features
        denom = stats['post'][:, None]
        denom = np.maximum(denom, 1e-10)

        if 'm' in self.params:
            prior = self.means_weight * self.means_prior
            self.means_ = (prior + stats['obs']) / (self.means_weight + denom)

        if 'c' in self.params:
            meandiff = self.means_  # (N, F)
            if self.covariance_type == 'full':
                covars = np.zeros((N, F, F))
                for i in range(N):
                    n = stats['post'][i]
                    if n < 1e-10:
                        covars[i] = np.eye(F) * self.min_covar
                        continue
                    cv = (stats['obs*obs.T'][i] / n
                          - np.outer(meandiff[i], meandiff[i]))
                    cv += self.min_covar * np.eye(F)
                    covars[i] = cv
                self._covars_ = covars
            elif self.covariance_type == 'diag':
                avg_X2 = stats['obs**2'] / np.maximum(
                    stats['post'][:, None], 1e-10)
                avg_means2 = meandiff ** 2
                self._covars_ = np.maximum(
                    avg_X2 - avg_means2, self.min_covar)
            elif self.covariance_type == 'spherical':
                avg_X2 = stats['obs**2'] / np.maximum(
                    stats['post'][:, None], 1e-10)
                avg_means2 = meandiff ** 2
                diag_covars = np.maximum(avg_X2 - avg_means2, self.min_covar)
                self._covars_ = diag_covars.mean(axis=1)
            elif self.covariance_type == 'tied':
                total_post = stats['post'].sum()
                cv = np.zeros((F, F))
                for i in range(N):
                    n = stats['post'][i]
                    if n < 1e-10:
                        continue
                    cv += (stats['obs*obs.T'][i]
                           - n * np.outer(meandiff[i], meandiff[i]))
                cv = cv / total_post + self.min_covar * np.eye(F)
                self._covars_ = cv


class CategoricalHSMM(BaseHSMM):
    """
    Hidden Semi-Markov Model with categorical (discrete) emissions.

    Parameters
    ----------
    n_components : int
        Number of hidden states.
    n_features : int, optional
        Number of possible observation symbols.  Inferred from data if omitted.
    duration_distribution : {"poisson", "gaussian", "negative_binomial",
                             "uniform"}
        Duration distribution family.
    max_duration : int
        Maximum segment length to model explicitly.
    emissionprob_prior : float or ndarray, shape (n_components, n_features)
        Dirichlet prior on emission probabilities.
    startprob_prior, transmat_prior : float
    algorithm : {"viterbi", "map"}
    random_state : int or RandomState, optional
    n_iter : int
    tol : float
    verbose : bool
    params, init_params : str

    Attributes
    ----------
    emissionprob_ : ndarray, shape (n_components, n_features)
    startprob_ : ndarray, shape (n_components,)
    transmat_ : ndarray, shape (n_components, n_components)

    Examples
    --------
    >>> from hmmlearn.hsmm import CategoricalHSMM
    >>> model = CategoricalHSMM(n_components=2, n_features=4,
    ...                         duration_distribution="poisson", max_duration=15)
    >>> X, Z = model.sample(100)
    """

    def __init__(self, n_components=1,
                 n_features=None,
                 duration_distribution="poisson",
                 max_duration=_DEFAULT_MAX_DURATION,
                 emissionprob_prior=1.0,
                 startprob_prior=1.0,
                 transmat_prior=1.0,
                 algorithm="viterbi",
                 random_state=None,
                 n_iter=10, tol=1e-2,
                 verbose=False,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters):
        super().__init__(
            n_components=n_components,
            duration_distribution=duration_distribution,
            max_duration=max_duration,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter, tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params)
        self.n_features = n_features
        self.emissionprob_prior = emissionprob_prior

    def _check_and_set_n_features(self, X):
        if not np.issubdtype(X.dtype, np.integer):
            raise ValueError("Observations must be integers for CategoricalHSMM")
        if X.min() < 0:
            raise ValueError("Observations must be non-negative")
        if self.n_features is not None:
            if self.n_features - 1 < X.max():
                raise ValueError(
                    f"Largest symbol is {X.max()} but n_features={self.n_features}")
        else:
            self.n_features = int(X.max()) + 1

    def _init(self, X, lengths=None):
        super()._init(X, lengths)
        rng = check_random_state(self.random_state)
        if self._needs_init("e", "emissionprob_"):
            self.emissionprob_ = rng.dirichlet(
                np.ones(self.n_features), size=self.n_components)

    def _check(self):
        super()._check()
        self.emissionprob_ = np.asarray(self.emissionprob_)
        if self.emissionprob_.shape != (self.n_components, self.n_features):
            raise ValueError(
                f"emissionprob_ must have shape (n_components, n_features)")
        s = self.emissionprob_.sum(axis=1)
        if not np.allclose(s, 1.0):
            raise ValueError("emissionprob_ rows must sum to 1")

    def _compute_log_likelihood(self, X):
        # X shape: (T, 1) of integers
        X_flat = X.squeeze(axis=1) if X.ndim == 2 else X
        with np.errstate(divide="ignore"):
            return np.log(self.emissionprob_[:, X_flat].T)  # (T, N)

    def _generate_sample_from_state(self, state, random_state):
        return np.array([random_state.choice(self.n_features,
                                             p=self.emissionprob_[state])])

    def _get_n_fit_scalars_per_param(self):
        N, F = self.n_components, self.n_features
        d = self._duration_model_.n_fit_scalars() if hasattr(
            self, "_duration_model_") else N
        return {
            "s": N - 1,
            "t": N * (N - 1),
            "d": d,
            "e": N * (F - 1),
        }

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_emission_sufficient_statistics(self, stats, X, posteriors):
        X_flat = X.squeeze(axis=1) if X.ndim == 2 else X
        for t, obs in enumerate(X_flat):
            stats['obs'][:, obs] += posteriors[t]

    def _do_emission_mstep(self, stats):
        if 'e' in self.params:
            self.emissionprob_ = np.maximum(
                self.emissionprob_prior - 1 + stats['obs'], 0)
            normalize(self.emissionprob_, axis=1)


class PoissonHSMM(BaseHSMM):
    """
    Hidden Semi-Markov Model with Poisson emissions.

    Each hidden state emits scalar non-negative integer observations drawn from
    a Poisson distribution with a state-specific rate parameter.

    Parameters
    ----------
    n_components : int
        Number of hidden states.
    duration_distribution : {"poisson", "gaussian", "negative_binomial",
                             "uniform"}
        Duration distribution family.
    max_duration : int
        Maximum segment length to model explicitly.
    startprob_prior, transmat_prior : float
    lambdas_prior : float
        Gamma prior shape parameter for emission rates.
    lambdas_weight : float
        Gamma prior rate parameter for emission rates.
    algorithm : {"viterbi", "map"}
    random_state : int or RandomState, optional
    n_iter : int
    tol : float
    verbose : bool
    params, init_params : str

    Attributes
    ----------
    lambdas_ : ndarray, shape (n_components,)
        Emission rate for each state.
    startprob_ : ndarray, shape (n_components,)
    transmat_ : ndarray, shape (n_components, n_components)

    Examples
    --------
    >>> from hmmlearn.hsmm import PoissonHSMM
    >>> model = PoissonHSMM(n_components=2, duration_distribution="gaussian",
    ...                     max_duration=25, n_iter=30)
    >>> X, Z = model.sample(500)
    >>> model.fit(X)  # doctest: +ELLIPSIS
    PoissonHSMM(...)
    """

    def __init__(self, n_components=1,
                 duration_distribution="poisson",
                 max_duration=_DEFAULT_MAX_DURATION,
                 startprob_prior=1.0,
                 transmat_prior=1.0,
                 lambdas_prior=1.0,
                 lambdas_weight=1.0,
                 algorithm="viterbi",
                 random_state=None,
                 n_iter=10, tol=1e-2,
                 verbose=False,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters):
        super().__init__(
            n_components=n_components,
            duration_distribution=duration_distribution,
            max_duration=max_duration,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter, tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params)
        self.lambdas_prior = lambdas_prior
        self.lambdas_weight = lambdas_weight

    def _check_and_set_n_features(self, X):
        if X.shape[1] != 1:
            raise ValueError(
                "PoissonHSMM expects univariate observations: "
                "X must have shape (n_samples, 1)")
        self.n_features = 1

    def _init(self, X, lengths=None):
        super()._init(X, lengths)
        rng = check_random_state(self.random_state)
        if self._needs_init("l", "lambdas_"):
            self.lambdas_ = rng.uniform(0.5, max(1.0, X.mean() * 2),
                                         size=self.n_components)

    def _check(self):
        super()._check()
        self.lambdas_ = np.asarray(self.lambdas_)
        if len(self.lambdas_) != self.n_components:
            raise ValueError("lambdas_ must have length n_components")
        if np.any(self.lambdas_ <= 0):
            raise ValueError("lambdas_ must be positive")

    def _compute_log_likelihood(self, X):
        from scipy.stats import poisson as sp_poisson
        x = X.squeeze(axis=1) if X.ndim == 2 else X  # (T,)
        # (T, N): log P(x_t | state i)
        return sp_poisson.logpmf(x[:, None], self.lambdas_[None, :])

    def _generate_sample_from_state(self, state, random_state):
        return np.array([random_state.poisson(self.lambdas_[state])])

    def _get_n_fit_scalars_per_param(self):
        N = self.n_components
        d = self._duration_model_.n_fit_scalars() if hasattr(
            self, "_duration_model_") else N
        return {
            "s": N - 1,
            "t": N * (N - 1),
            "d": d,
            "l": N,
        }

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['obs'] = np.zeros(self.n_components)
        stats['post'] = np.zeros(self.n_components)
        return stats

    def _accumulate_emission_sufficient_statistics(self, stats, X, posteriors):
        x = X.squeeze(axis=1) if X.ndim == 2 else X
        stats['obs'] += posteriors.T @ x.astype(float)
        stats['post'] += posteriors.sum(axis=0)

    def _do_emission_mstep(self, stats):
        if 'l' in self.params:
            denom = np.maximum(self.lambdas_weight + stats['post'], 1e-10)
            self.lambdas_ = (self.lambdas_prior + stats['obs']) / denom
