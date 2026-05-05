---
ID: 002
Origin: 002
UUID: b7e14c3a
Status: OPEN
---

# Critique — Plan 002: Add GaussianMixtureHSMM

| Field | Value |
|---|---|
| **Artifact** | agent-output/planning/002-gmm-hsmm.md |
| **Critique Date** | 2026-05-05 |
| **Status** | OPEN |
| **Verdict** | REJECTED — two unresolved blockers; must be addressed before handoff |

## Changelog

| Date | Handoff | Request | Summary |
|------|---------|---------|---------|
| 2026-05-05 | Planner → Critic | Initial review | First review pass against actual source code |

---

## Value Statement Assessment

> *"As a researcher or practitioner using hmmlearn, I want a Hidden Semi-Markov Model with Gaussian Mixture emissions…"*

The value is well-articulated, outcome-focused, and genuinely fills a gap in the library. The combination of explicit duration control + multi-modal emission is not currently available. This is a legitimate feature request.

**Achievability concern**: The feature is achievable, but the plan underestimates the `BaseGMMHMM` coupling problem. The plan currently reads as if the mixin will "just work" once Plan 001's bridge pattern is established. Code inspection shows `BaseGMMHMM` is coupled at a deeper level than `BaseGaussianHMM`, which materially affects implementation complexity.

---

## Overview

The plan is logically structured and correctly depends on Plan 001. The risk table honestly flags the `BaseGMMHMM` coupling risk as "Medium." However, the plan then claims in Assumption 2 that `BaseGMMHMM` is analogous to `BaseGaussianHMM` — which is **factually wrong**. The analogy breaks at the inheritance level. This is the most important correction needed.

The plan also contains an unresolved OPEN QUESTION explicitly marked as "Implementer must resolve during Milestone 1 audit before starting Milestone 2." Per the Critic review protocol, plans with unresolved open questions cannot be handed off without explicit approval.

---

## Architectural Alignment

The plan depends on Plan 001 completing successfully — this dependency is correctly declared. If Plan 001 establishes a working bridge pattern for `BaseGaussianHMM` + `BaseHSMM`, the same bridge pattern should be applicable here. However, the differences in `BaseGMMHMM`'s MRO must be explicitly managed.

---

## Scope Assessment

Scope is appropriate. Excluding variational GMMHSMM and multinomial mixture HSMM is correct. The test coverage requirements are thorough.

---

## Technical Debt Risks

- The GMM M-step is complex (4 covariance types × mixture weights × component counts). If the bridge is wrong, it will produce numerically plausible but incorrect parameters with no obvious error signal.
- Double-initialization of `nobs`/`start`/`trans` in `_initialize_sufficient_statistics` (see H-01) could silently double-count structural stats.

---

## Findings

### Finding C-01 — CRITICAL | OPEN

**Issue**: `BaseGMMHMM` inherits from `BaseHMM`, not `_AbstractHMM` — the plan's Assumption 2 is factually wrong

**Description**: Assumption 2 states that `BaseGMMHMM` "inherits from `_AbstractHMM`, making it composable with `BaseHSMM` (analogous to how `BaseGaussianHMM` works after Plan 001)."

Inspecting `_emissions.py`:
```python
class BaseGaussianHMM(_AbstractHMM): ...  # Plan 001 target ✓
class BaseGMMHMM(BaseHMM):           ...  # Plan 002 target ✗
```

These are **not analogous**. `BaseGMMHMM` inherits from `BaseHMM`, which adds the full HMM forward-backward machinery. Specifically, `BaseGMMHMM._accumulate_sufficient_statistics` calls:
```python
super()._accumulate_sufficient_statistics(stats, X, lattice, post_comp, fwdlattice, bwdlattice)
```
This chains through `BaseHMM._accumulate_sufficient_statistics`, which dispatches on `self.implementation` — an attribute that `BaseHSMM` does not have.

For `GaussianMixtureHSMM(_emissions.BaseGMMHMM, BaseHSMM)`, the MRO is:
`GaussianMixtureHSMM → BaseGMMHMM → BaseHMM → _AbstractHMM → BaseHSMM → BaseEstimator → object`

When the bridge calls the mixin's emission accumulation, the `super()` chain inside `BaseGMMHMM` will hit `BaseHMM._accumulate_sufficient_statistics` and attempt `self.implementation` lookup → `AttributeError`.

**Impact**: The class cannot be instantiated and fitted without resolving this. This is the same class of bug as Plan 001's C-01 for `BasePoissonHMM`.

**Recommendation**: Assumption 2 must be corrected. The plan must explicitly acknowledge that `BaseGMMHMM` has deeper coupling than `BaseGaussianHMM` and prescribe one of the following:
(a) Add a shim in `GaussianMixtureHSMM` that overrides `_accumulate_sufficient_statistics` and does NOT call `super()` past the GMM-specific logic — essentially re-implementing the structural stats accumulation inline, or
(b) Refactor `BaseGMMHMM` to inherit `_AbstractHMM` instead of `BaseHMM` (scope expansion — touches `_emissions.py`), or
(c) Explicitly scope this plan to require option (a) and document the approach in a new Milestone 1.5.

---

### Finding C-02 — CRITICAL | OPEN

**Issue**: Unresolved OPEN QUESTION requires implementer decision before Milestone 2 can start

**Description**: The plan contains the following explicit marker:

> **OPEN QUESTION**: Does `_emissions.BaseGMMHMM._accumulate_sufficient_statistics()` compute mixture responsibilities internally from `log_likelihood`, or does it expect them to be pre-computed? — **Implementer must resolve during Milestone 1 audit before starting Milestone 2.**

This is an explicit deferral to the implementer. Per the Critic protocol, plans with unresolved open questions cannot be approved for handoff without explicit planner acknowledgement.

**Code resolution**: Inspecting `_emissions.py`, the answer is already determinable from the source:
```python
def _accumulate_sufficient_statistics(self, stats, X, lattice, post_comp, fwdlattice, bwdlattice):
    ...
    post_mix = np.zeros((n_samples, self.n_components, self.n_mix))
    for p in range(self.n_components):
        log_denses = self._compute_log_weighted_gaussian_densities(X, p)
        log_normalize(log_denses, axis=-1)
        post_mix[:, p, :] = np.exp(log_denses)
    post_comp_mix = post_comp[:, :, None] * post_mix
```
Mixture responsibilities **are computed internally** from the state posteriors (`post_comp = gamma`). The HSMM only needs to supply `gamma[t, i]` (state posteriors) — which it already has. The per-mixture decomposition is done inside the method.

**Impact**: The answer resolves in Plan 002's favor (the bridge is simpler than feared), but the question was never resolved in the plan document. This creates an ambiguous handoff.

**Recommendation**: Update the plan to mark this question `[RESOLVED]` with the answer: "Mixture responsibilities are computed internally from state posteriors. The bridge must pass `gamma` as `post_comp`. `fwdlattice`, `bwdlattice`, and `lattice` can be passed as `None` or zero arrays since `BaseGMMHMM._accumulate_sufficient_statistics` does not use them directly — only `post_comp` and `X` are used for the GMM-specific portion."

---

### Finding H-01 — HIGH | OPEN

**Issue**: Double-initialization risk in `_initialize_sufficient_statistics` chain

**Description**: `GaussianMixtureHSMM._initialize_sufficient_statistics` will trigger the `super()` chain in MRO order. The chain must produce exactly one `nobs`, `start`, `trans` initialization and exactly one GMM stats initialization. Currently:

- `BaseHSMM._initialize_sufficient_statistics()` returns `{'nobs': 0, 'start': ..., 'trans': ..., 'dur': ...}`
- `BaseGMMHMM._initialize_sufficient_statistics()` calls `super()._initialize_sufficient_statistics()` which goes through `BaseHMM._initialize_sufficient_statistics()` → `_AbstractHMM._initialize_sufficient_statistics()` → returns `{'nobs': 0, 'start': ..., 'trans': ...}`

In the `GaussianMixtureHSMM` MRO, when `super()._initialize_sufficient_statistics()` is called, it will hit `BaseGMMHMM` first, which calls its own `super()` going through `BaseHMM` → `_AbstractHMM`. It will NOT hit `BaseHSMM._initialize_sufficient_statistics`. This means `stats['dur']` — essential for HSMM — will not be initialized.

**Impact**: Missing `stats['dur']` will cause a `KeyError` in `BaseHSMM._do_estep` at `stats['dur'] += eta`.

**Recommendation**: Milestone 2 must explicitly address `_initialize_sufficient_statistics` override in `GaussianMixtureHSMM`: start from `BaseHSMM._initialize_sufficient_statistics()` result, then manually add GMM-specific keys. Do not rely on `super()` chaining through `BaseGMMHMM` for this method.

---

### Finding H-02 — HIGH | OPEN

**Issue**: `__init__` MRO parameter conflict — `BaseHMM.__init__` requires `implementation` parameter not present in `BaseHSMM`

**Description**: `GaussianMixtureHSMM.__init__` calling `super().__init__()` through the MRO will encounter `BaseHMM.__init__` (via `BaseGMMHMM`). `BaseHMM.__init__` has a required `implementation` parameter (default `"log"`) and also calls `super().__init__` with `implementation=...`. `BaseHSMM.__init__` does not accept `implementation` — a `TypeError` or silently swallowed kwarg may result.

**Impact**: The class constructor may fail or silently set unexpected attributes.

**Recommendation**: `GaussianMixtureHSMM.__init__` must explicitly manage the MRO by calling `BaseHSMM.__init__` directly (not via `super()` through the full MRO), similar to how `CategoricalHMM.__init__` calls `BaseHMM.__init__` directly. Alternatively, `BaseHSMM.__init__` can be augmented to absorb `**kwargs`.

---

### Finding M-01 — MEDIUM | OPEN

**Issue**: Milestone 2b references non-existent `_init_covar_priors()` and `_fix_priors_shape()` methods

**Description**: Milestone 2b Task 3 says: "Initialize covariance priors (delegate to `_init_covar_priors()` from `BaseGMMHMM` if available)." Inspecting `_emissions.py` and `hmm.py`, `BaseGMMHMM` does not define `_init_covar_priors()` or `_fix_priors_shape()`. These method names do not exist in the current codebase.

**Impact**: The implementer will spend time searching for methods that don't exist. The prior initialization logic is inline in `GMMHMM._init()` in `hmm.py`.

**Recommendation**: Replace the reference with the actual approach: copy the prior initialization pattern from `GMMHMM._init()` in `hmm.py`.

---

### Finding L-01 — LOW | OPEN

**Issue**: `n_mix=1` edge case equivalence claim needs qualification

**Description**: The testing strategy states: "Edge case: `n_mix=1` should reduce to behavior consistent with `GaussianHSMM`." This is approximately true but not exactly — `GaussianMixtureHSMM(n_mix=1)` will use a single mixture component with a weight of 1.0, which is numerically equivalent to a single Gaussian, but the parameter structure is different (`weights_` array, `means_` shape `(n_components, 1, n_features)` vs `(n_components, n_features)`). The test assertion needs to be probabilistic (score parity) rather than parameter-shape equality.

**Recommendation**: Clarify the test assertion: compare log-likelihoods on held-out data rather than checking parameter equivalence.

---

## Unresolved Open Questions

The plan contains **1 unresolved open question**:

> **OPEN QUESTION**: Does `_emissions.BaseGMMHMM._accumulate_sufficient_statistics()` compute mixture responsibilities internally?

**Decision required**: This question can be resolved from code (answer: yes, internally computed). Planner must mark it `[RESOLVED]` in the plan before implementation handoff.

---

## Questions for Planner

1. Given C-01, should Plan 002 scope expand to include refactoring `BaseGMMHMM` to inherit `_AbstractHMM`? Or will a shim approach be mandated?
2. How should the `__init__` MRO conflict (H-02) be resolved — direct constructor call, or `**kwargs` passthrough in `BaseHSMM`?
3. The open question is resolvable from code. Will the planner update the plan and mark it resolved, or should the implementer be trusted to discover it in Milestone 1?

---

## Risk Assessment

| Risk | Likelihood | Impact | Status |
|------|-----------|--------|--------|
| `BaseGMMHMM` MRO crash on `self.implementation` (C-01) | **Certain if not fixed** | Blocker | OPEN |
| Unresolved OPEN QUESTION on handoff (C-02) | Confirmed gap | Ambiguous handoff | OPEN |
| Missing `stats['dur']` in init chain (H-01) | High | KeyError at runtime | OPEN |
| `BaseHMM.__init__` MRO conflict (H-02) | High | Constructor failure | OPEN |
| Non-existent method references (M-01) | Confirmed | Implementer confusion | OPEN |

---

## Recommendations

1. **BLOCK on C-01**: Correct Assumption 2 and prescribe explicit implementation strategy for the `BaseGMMHMM` `BaseHMM` coupling.
2. **BLOCK on C-02**: Resolve the open question (answer is in the code) and update the plan.
3. **BLOCK on H-01**: Add explicit `_initialize_sufficient_statistics` override guidance to Milestone 2.
4. **BLOCK on H-02**: Add explicit `__init__` call strategy to Milestone 2a.
5. **MINOR**: Fix M-01 method name references.

---

## Revision History

| Date | Revision | Changes | Status Changes |
|------|----------|---------|----------------|
| 2026-05-05 | Initial | First pass | All findings OPEN |
