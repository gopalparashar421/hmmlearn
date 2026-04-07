"""
Hidden Semi-Markov Model: Sampling, Fitting, and Decoding
----------------------------------------------------------

A **Hidden Semi-Markov Model** (HSMM) extends the standard HMM by replacing
the implicit geometric sojourn distribution with an explicit one.  This lets
each state have a characteristic *dwell time*, which is useful whenever the
process you are modelling stays in a state for a predictable number of steps
(e.g. machine faults, speech phonemes, behavioural bouts).

This example:

1. Generates synthetic observations from a known 3-state
   :class:`~hmmlearn.hsmm.GaussianHSMM` with Poisson durations.
2. Fits a new model on the data.
3. Decodes the hidden states with the Viterbi algorithm and compares them to
   the ground truth.
4. Plots the fitted duration probability mass functions alongside the true
   Poisson PMFs to show that the model has learnt the dwell-time structure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import permutations
from scipy.stats import poisson as sp_poisson

from hmmlearn.hsmm import GaussianHSMM

# ---------------------------------------------------------------------------
# 1. Ground-truth model and synthetic data
# ---------------------------------------------------------------------------

rng = np.random.RandomState(42)

N_COMPONENTS = 3
MAX_DURATION = 20
STATE_COLORS = ["tab:blue", "tab:orange", "tab:green"]

# Well-separated 1-D Gaussian emissions
TRUE_MEANS = np.array([[-4.0], [0.0], [4.0]])
TRUE_STDS  = np.array([[0.7],  [0.6], [0.8]])

# Poisson mean durations (steps) for each state
TRUE_LAMBDAS = np.array([5.0, 8.0, 4.0])

gen = GaussianHSMM(
    n_components=N_COMPONENTS,
    covariance_type="diag",
    duration_distribution="poisson",
    max_duration=MAX_DURATION,
    random_state=rng,
)
gen.startprob_ = np.array([1.0, 0.0, 0.0])
gen.transmat_  = np.array([
    [0.0, 0.7, 0.3],
    [0.4, 0.0, 0.6],
    [0.5, 0.5, 0.0],
])
gen.means_     = TRUE_MEANS
gen._covars_   = TRUE_STDS ** 2
gen.n_features = 1
gen._duration_model_ = gen._build_duration_model()
gen._duration_model_.initialize()
gen._duration_model_.set_params(lambda_=TRUE_LAMBDAS)

X, Z_true = gen.sample(120, random_state=rng)
N = len(X)

# %%
# 2. Plot the generated observations
# -----------------------------------

fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

axes[0].plot(X[:, 0], color="k", lw=0.8, alpha=0.85)
for s in range(N_COMPONENTS):
    axes[0].fill_between(
        np.arange(N), X.min() - 1, X.max() + 1,
        where=(Z_true == s), alpha=0.18, color=STATE_COLORS[s],
        label=f"State {s}")
axes[0].set_ylim(X.min() - 0.4, X.max() + 0.4)
axes[0].set_ylabel("Observation")
axes[0].legend(loc="upper right", ncol=N_COMPONENTS, fontsize=9)
axes[0].set_title("Synthetic observations (shaded by true hidden state)")

axes[1].step(np.arange(N), Z_true, where="post", color="k", lw=1.2)
axes[1].set_yticks(range(N_COMPONENTS))
axes[1].set_ylabel("State")
axes[1].set_xlabel("Time step")
axes[1].set_title("True state sequence")

fig.tight_layout()
plt.show()

# %%
# 3. Fit a GaussianHSMM
# ----------------------
# Run several random-restarts and keep the model with the highest
# log-likelihood to reduce sensitivity to initialisation.

best_score, best_model = -np.inf, None

for seed in range(4):
    m = GaussianHSMM(
        n_components=N_COMPONENTS,
        covariance_type="diag",
        duration_distribution="poisson",
        max_duration=MAX_DURATION,
        n_iter=20,
        tol=1e-3,
        random_state=seed,
    )
    m.fit(X)
    s = m.score(X)
    if s > best_score:
        best_score, best_model = s, m

print(f"Best log-likelihood: {best_score:.2f}  (true model: "
      f"{gen.score(X):.2f})")

# %%
# 4. Viterbi decoding
# --------------------
# Align recovered state labels with ground truth by finding the
# label permutation that minimises the squared distance between the
# fitted means and the true means.

_, Z_pred = best_model.decode(X, algorithm="viterbi")


def align_states(model, true_means):
    """Return index mapping: true_state[i] -> model_state[perm[i]]."""
    best_perm, best_err = None, np.inf
    for perm in permutations(range(model.n_components)):
        err = np.sum((model.means_[list(perm)] - true_means) ** 2)
        if err < best_err:
            best_err, best_perm = err, list(perm)
    return best_perm


perm = align_states(best_model, TRUE_MEANS)
Z_aligned = np.array([perm.index(s) for s in Z_pred])

accuracy = (Z_aligned == Z_true).mean()
print(f"Viterbi state-sequence accuracy: {accuracy:.1%}")

fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
for ax, Z, title in zip(
        axes,
        [Z_true, Z_aligned],
        ["True states", "Recovered states (Viterbi)"]):
    for s in range(N_COMPONENTS):
        ax.fill_between(
            np.arange(N), 0, 1,
            where=(Z == s),
            transform=ax.get_xaxis_transform(),
            alpha=0.65, color=STATE_COLORS[s])
    ax.set_yticks([])
    ax.set_ylabel("State")
    ax.set_title(title)
    patches = [mpatches.Patch(color=STATE_COLORS[s], label=f"State {s}")
               for s in range(N_COMPONENTS)]
    ax.legend(handles=patches, loc="upper right", fontsize=9)

axes[-1].set_xlabel("Time step")
fig.tight_layout()
plt.show()

# %%
# 5. Fitted duration distributions
# ---------------------------------
# Compare each state's fitted Poisson PMF with the true generating PMF.

dur_model = best_model._duration_model_
log_pmf   = dur_model.log_pmf()     # (max_duration, n_components)
pmf_fitted = np.exp(log_pmf)

d_values = np.arange(1, MAX_DURATION + 1)

fig, axes = plt.subplots(1, N_COMPONENTS, figsize=(12, 3), sharey=True)

for true_state, ax in enumerate(axes):
    model_state = perm[true_state]
    ax.bar(d_values, pmf_fitted[:, model_state],
           color=STATE_COLORS[true_state], alpha=0.65, label="Fitted")
    true_pmf = sp_poisson.pmf(d_values, TRUE_LAMBDAS[true_state])
    true_pmf /= true_pmf.sum()    # renormalise to the truncated support
    ax.plot(d_values, true_pmf, "k--", lw=1.5, label="True Poisson")
    ax.set_title(f"State {true_state}  (lambda = {TRUE_LAMBDAS[true_state]:.0f})")
    ax.set_xlabel("Duration (steps)")
    if true_state == 0:
        ax.set_ylabel("Probability")
    ax.legend(fontsize=8)

fig.suptitle("Fitted vs. true Poisson duration distributions", fontsize=11)
fig.tight_layout()
plt.show()
