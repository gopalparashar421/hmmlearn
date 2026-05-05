---
ID: 002
Origin: 002
UUID: b7e14c3a
Status: Active
Target Release: v0.4.0
---

# Plan 002 — Add GaussianMixtureHSMM (GMMHSMM)

| Field | Value |
|---|---|
| **ID** | 002 |
| **Target Release** | v0.4.0 |
| **Epic Alignment** | Feature expansion — richer emission models for HSMM |
| **Status** | Active (v2 — Critic revision) |
| **Created** | 2026-05-05 |
| **Depends On** | Plan 001 (HSMM emission mixin refactor) |

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-05-05 | Planner | Initial draft |
| 2026-05-05 | Planner | v2: Address Critic findings C-01, C-02, H-01, H-02, M-01 |

---

## Value Statement and Business Objective

> **As a researcher or practitioner using hmmlearn**, I want a Hidden Semi-Markov Model with Gaussian Mixture emissions (`GaussianMixtureHSMM`), so that I can model complex, multi-modal emission distributions with explicit duration control — a combination not currently available in the library.

---

## Background

`GMMHMM` (in `hmm.py`) uses `_emissions.BaseGMMHMM` which provides mixture-of-Gaussians emission logic but is currently tightly coupled to `BaseHMM`'s EM machinery. The HSMM framework (`BaseHSMM`) uses explicit-duration forward-backward that is incompatible with `BaseHMM`'s forward-backward.

The goal is to create a `GaussianMixtureHSMM` class that:
- Uses `BaseGMMHMM` emission logic (from `_emissions.py`)
- Uses `BaseHSMM` duration + structural logic
- Supports all four covariance types (spherical, diag, full, tied)
- Supports all four duration distributions (poisson, gaussian, negative_binomial, uniform)

---

## Assumptions

1. **Plan 001 is complete** before implementation of this plan begins. Specifically, `BaseHSMM` already exposes `self.implementation = "log"` (Plan 001 Milestone 2 Task 2), which is a prerequisite for `BaseGMMHMM` composition.
2. **[REVISED — C-01]** `_emissions.BaseGMMHMM` inherits from `BaseHMM`, NOT `_AbstractHMM`. It is NOT directly composable with `BaseHSMM` via simple multiple inheritance. The same `self.implementation` shim from Plan 001 is required; Plan 001 already adds this to `BaseHSMM.__init__`. However, `BaseGMMHMM.__init__` also calls `BaseHMM.__init__` which expects `implementation` as a constructor parameter, creating a constructor chain conflict. The implementer must ensure `GaussianMixtureHSMM.__init__` explicitly passes `implementation="log"` or handles the MRO constructor invocation carefully, calling `BaseHSMM.__init__` and `BaseGMMHMM`'s emission-relevant init separately.
3. **[REVISED — C-02 RESOLVED]** The open question about mixture responsibilities is now resolved: `_emissions.BaseGMMHMM._accumulate_sufficient_statistics()` computes mixture responsibilities **internally** from log-likelihoods and state posteriors. The bridge only needs to pass `gamma` (state posteriors, shape T × n_components) as `posteriors` in the correct position. This is achievable without modifying `_emissions.py`.
4. **[REVISED — M-01]** `_init_covar_priors()` and `_fix_priors_shape()` do NOT exist in the codebase. Covariance prior initialization logic from `GMMHMM._init()` must be inlined or extracted by the implementer; do not reference these as existing helper methods.
5. The sufficient statistics structure for GMM emissions is additive per-sequence and can be accumulated via HSMM's E-step.

**OPEN QUESTION [RESOLVED C-02]**: Does `_emissions.BaseGMMHMM._accumulate_sufficient_statistics()` compute mixture responsibilities internally? — **YES.** It computes responsibilities from `log_likelihood` (log per-component densities) and state posteriors. The bridge passes `posteriors=gamma` and the mixin handles the rest.

---

## Scope

**In scope**:
- `GaussianMixtureHSMM` class in `hsmm.py`
- Support for all four `covariance_type` values
- Support for all four duration distributions
- Initialization, validation, M-step delegation, sampling
- Export via `hsmm.__all__`
- Tests covering fit/predict/sample/score

**Out of scope**:
- Variational Bayes variant of GMMHSMM (future work)
- Multinomial mixture HSMM
- Changes to `GMMHMM` in `hmm.py`

---

## Milestones

### Milestone 1 — Verify BaseGMMHMM Composability

**Objective**: Confirm that `_emissions.BaseGMMHMM` can be composed with `BaseHSMM` using the mixin pattern established in Plan 001.

**Tasks**:
1. Apply the Milestone 1 audit methodology from Plan 001 to `_emissions.BaseGMMHMM`.
2. List every method that accesses `BaseHMM`-specific attributes or calls.
3. Confirm that the same adapter/shim strategy works; document the delta.

**Acceptance Criteria**:
- Written compatibility findings (can be inline code comments)
- No modifications to `_emissions.BaseGMMHMM` public API required

---

### Milestone 2 — Implement GaussianMixtureHSMM

**Objective**: Create the new class following the established pattern.

**Tasks**:

#### 2a — Class Declaration & `__init__`
1. Declare: `class GaussianMixtureHSMM(_emissions.BaseGMMHMM, BaseHSMM):`
2. Constructor must accept all `BaseHSMM` parameters plus all GMM-specific parameters (n_mix, min_covar, covariance_type, weights_prior, means_prior, means_weight, covars_prior, covars_weight).
3. **[H-02 fix]** Call `BaseHSMM.__init__` explicitly (not `super().__init__`) with the HSMM parameters. Then manually initialize GMM-specific attributes. Do NOT let `BaseGMMHMM.__init__` call `BaseHMM.__init__` through the MRO, as that constructor expects different parameters. The `self.implementation = "log"` attribute added to `BaseHSMM.__init__` by Plan 001 satisfies the mixin's dispatch needs.
4. Default `params='stdmcw'` (s=start, t=transmat, d=duration, m=means, c=covars, w=weights).
5. Default `init_params='stdmcw'`.

#### 2b — `_init(X, lengths)`
1. Call `BaseHSMM._init(self, X, lengths)` for start/transmat/duration initialization.
2. Initialize mixture weights, component means, covariances per state using the same K-means seeding strategy as `GMMHMM._init()`. **Do NOT call `_init_covar_priors()` or `_fix_priors_shape()` — these methods do not exist.** Inline the covariance prior setup from `GMMHMM._init()`.

#### 2c — `_check()`
1. Call `BaseHSMM._check(self)` for structural checks.
2. Call `_emissions.BaseGMMHMM._check(self)` explicitly if needed (or let `super()._check()` chain handle it through MRO — implementer to verify MRO resolution).
3. **[H-01 fix]** `GaussianMixtureHSMM._initialize_sufficient_statistics()` MUST explicitly merge the HSMM stats dict (`{'nobs': 0, 'start': ..., 'trans': ..., 'dur': ...}`) with the GMM emission stats dict. Do NOT rely on `super()` chaining alone because the MRO routes through `BaseGMMHMM → BaseHMM`, bypassing `BaseHSMM._initialize_sufficient_statistics()`. The implementer must call both explicitly: `stats = BaseHSMM._initialize_sufficient_statistics(self)` then merge in `_emissions.BaseGMMHMM._initialize_sufficient_statistics(self)`.

#### 2d — `_check_and_set_n_features(X)`
1. Delegate to `_emissions.BaseGMMHMM`'s implementation.

#### 2e — `_do_emission_mstep(stats)`
1. Delegate to `_emissions.BaseGMMHMM._do_mstep(stats)` for mixture weight, means, and covariance updates.

#### 2f — Docstring and `__all__` Export
1. Write a clear class docstring listing all attributes, parameters, and an example.
2. Add `"GaussianMixtureHSMM"` to `hsmm.__all__`.

**Acceptance Criteria**:
- `GaussianMixtureHSMM(n_components=3, n_mix=2).fit(X, lengths)` completes without error
- All four covariance types instantiate and fit
- All four duration distributions instantiate and fit
- `stats['dur']` key is present after `_initialize_sufficient_statistics()`

---

### Milestone 3 — Test Coverage

**Objective**: Verify correctness through tests.

**Tasks**:
1. Add `test_gmm_hsmm.py` (analogous to `test_gmm_hmm_new.py`) covering:
   - Fit on synthetic data (Gaussian mixture generating process)
   - Predict / decode
   - Sample
   - Score / log-likelihood
   - Round-trip: fit on generated samples → recovered params within tolerance
2. Parametrize over covariance types and duration distributions.
3. Run full test suite to confirm no regressions.

**Acceptance Criteria**:
- New test file passes completely
- No regressions in existing tests

---

### Milestone 4 — Documentation

**Objective**: Surface the new class in library documentation.

**Tasks**:
1. Add `GaussianMixtureHSMM` to `doc/source/api.rst`.
2. Add a minimal example to `examples/plot_hsmm.py` or a new `examples/plot_gmm_hsmm.py`.

**Acceptance Criteria**:
- API reference lists the new class
- Example runs without error

---

### Milestone 5 — Version and Release Artifacts

**Objective**: Update release artifacts for v0.4.0.

**Tasks**:
1. Add `GaussianMixtureHSMM` to `CHANGES.rst` under `Version 0.4.0`.
2. Verify `hsmm.__all__` is correct.

**Acceptance Criteria**:
- `CHANGES.rst` updated
- `from hmmlearn.hsmm import GaussianMixtureHSMM` works

---

## Testing Strategy

- **Unit**: Each `__init__` parameter accepted and stored correctly.
- **Integration**: Fit/predict/sample/score pipeline with all covariance types.
- **Regression**: Existing `test_hsmm.py` and `test_gmm_hmm*.py` unaffected.
- **Numerical**: Log-likelihood monotonically increases over EM iterations (convergence check).
- **Edge case**: `n_mix=1` should reduce to behavior consistent with `GaussianHSMM`.

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `BaseGMMHMM.__init__` chains to `BaseHMM.__init__` through MRO | **Confirmed** | Call `BaseHSMM.__init__` explicitly; do NOT route through `BaseGMMHMM.__init__` (Milestone 2a Task 3) |
| `stats['dur']` missing because `super()` chain bypasses `BaseHSMM._initialize_sufficient_statistics` | **Confirmed** | Call both `BaseHSMM._initialize_sufficient_statistics()` and `BaseGMMHMM._initialize_sufficient_statistics()` explicitly and merge (Milestone 2c Task 3) |
| `_init_covar_priors()` / `_fix_priors_shape()` do not exist | **Confirmed** | Inline covariance prior setup from `GMMHMM._init()` source (Milestone 2b Task 2) |
| GMM mixture responsibilities bridge to HSMM posteriors | **Resolved** | `BaseGMMHMM` computes responsibilities internally from posteriors; just pass `gamma` as `posteriors` |
| Numerical instability for short sequences + many mixtures | Low | Clipping already exists in base; ensure max_duration not too small |

---

## Handoff Notes

- **Prerequisite**: Plan 001 must be merged and tests green before implementation begins.
- **Implementer**: The key challenge is bridging HSMM state posteriors to GMM mixture responsibilities. Study `GMMHMM._accumulate_sufficient_statistics()` in `_emissions.py` carefully for how mixture posteriors are derived.
- **Scope guard**: Do not add `VariationalGaussianMixtureHSMM` in this plan. That is future work.
