---
ID: 001
Origin: 001
UUID: a3f8c21d
Status: Active
Target Release: v0.4.0
---

# Plan 001 — Refactor HSMM to Use Emission Mixin Pattern

| Field | Value |
|---|---|
| **ID** | 001 |
| **Target Release** | v0.4.0 |
| **Epic Alignment** | Code quality / architectural consistency |
| **Status** | Active (v2 — Critic revision) |
| **Created** | 2026-05-05 |

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-05-05 | Planner | Initial draft |
| 2026-05-05 | Planner | v2: Address Critic findings C-01, H-01, H-02, M-01 |

---

## Value Statement and Business Objective

> **As a contributor or maintainer of hmmlearn**, I want the HSMM module to follow the same emission-mixin composition pattern as `hmm.py` and `vhmm.py`, so that emission logic is not duplicated, new emission types can be added consistently, and the codebase is easier to maintain and extend.

---

## Background and Problem Statement

`hsmm.py` currently implements three concrete classes — `GaussianHSMM`, `CategoricalHSMM`, and `PoissonHSMM` — by directly inheriting from `BaseHSMM` and **re-implementing emission logic** that already exists in `_emissions.py`. This violates DRY and creates a maintenance burden: any fix to a base emission class (e.g., `BaseGaussianHMM`) must be mirrored manually in the HSMM version.

By contrast, `hmm.py` composes emission behaviour through multiple inheritance:
```python
class GaussianHMM(_emissions.BaseGaussianHMM, BaseHMM):  ...
class CategoricalHMM(_emissions.BaseCategoricalHMM, BaseHMM):  ...
class PoissonHMM(_emissions.BasePoissonHMM):  ...
```

The `_emissions` mixin classes (`BaseCategoricalHMM`, `BaseGaussianHMM`, `BasePoissonHMM`) inherit from `_AbstractHMM` and implement:
- `_compute_log_likelihood()` / `_compute_likelihood()`
- `_accumulate_sufficient_statistics()` for emission stats
- `_generate_sample_from_state()`
- `_get_n_fit_scalars_per_param()` (partial)

`BaseHSMM` exposes the same abstract interface. The emission mixins are therefore **compatible** as long as `BaseHSMM` satisfies the `_AbstractHMM` contract for the methods the mixins call.

---

## Assumptions

1. **[REVISED — C-01]** NOT all `_emissions` mixins are `_AbstractHMM`-only. `BasePoissonHMM` and `BaseGaussianHMM` use `_AbstractHMM` but `BasePoissonHMM` references `self.implementation` which exists on `BaseHMM` but NOT on `BaseHSMM`. An adapter shim or `implementation` attribute on `BaseHSMM` is required. `BaseCategoricalHMM` similarly depends on `self.implementation` via the `_compute_likelihood` dispatch. All three concrete classes need shim handling.
2. `BaseHSMM._accumulate_emission_sufficient_statistics()` can delegate to the emission mixin's `_accumulate_sufficient_statistics()` after appropriately structuring the stats dict.
3. The duration M-step (`'d'` param) remains exclusive to `BaseHSMM` and does not conflict with emission mixin M-step characters.
4. `_needs_init()` is currently duplicated between `base.py` and `hsmm.py`; it can be moved directly into `BaseHSMM` as a self-contained method rather than relying on `_AbstractHMM` extraction.
5. **[REVISED — H-02]** `_check_sum_1()` is defined on `BaseHMM`, NOT on `_AbstractHMM`. It cannot be inherited from `_AbstractHMM`. The implementer must copy the method body into `BaseHSMM` or add a mixin — the simplest approach is copying it directly.

**OPEN QUESTION [RESOLVED]**: Are `_emissions` mixins safe to use without `BaseHMM` in the MRO without any changes? — **NO.** They reference `self.implementation` (log vs scaling dispatch). `BaseHSMM` must expose `self.implementation = "log"` as a fixed attribute (HSMM only supports log-space) to satisfy this dependency without modifying `_emissions.py`.

---

## Scope

**In scope**:
- Refactor `GaussianHSMM` to use `_emissions.BaseGaussianHMM` mixin
- Refactor `CategoricalHSMM` to use `_emissions.BaseCategoricalHMM` mixin
- Refactor `PoissonHSMM` to use `_emissions.BasePoissonHMM` mixin
- Remove duplicated emission implementations from `hsmm.py`
- Extract or reuse `_needs_init()` to eliminate duplication with `base.py`
- Ensure `BaseHSMM._check()` calls `_check_sum_1()` from the base instead of reimplementing

**Out of scope**:
- New emission types (handled in Plan 002)
- Online learning (Plan 003)
- Changes to `hmm.py`, `vhmm.py`, or `_emissions.py` public API

---

## Milestones

### Milestone 1 — Audit & Compatibility Verification

**Objective**: Confirm that the `_emissions` mixins are composable with `BaseHSMM` without modification, or identify the minimal interface changes required.

**Tasks**:
1. Read `_emissions.py` and list every method that references `BaseHMM`-specific attributes (e.g., `self.transmat_`, `self._fit_log()`, `self._do_estep()`).
2. For each such reference, determine if `BaseHSMM` already provides it or if an adapter is needed.
3. Confirm the MRO for `class GaussianHSMM(_emissions.BaseGaussianHMM, BaseHSMM)` resolves method calls correctly.
4. Document any method in `_emissions` that conflicts with `BaseHSMM` (same signature, different semantics).

**Acceptance Criteria**:
- A written compatibility matrix (can be a code comment or dev note, not a separate doc)
- At most 2 adapter shims needed; if more, escalate to re-evaluate approach

---

### Milestone 2 — Refactor BaseHSMM Interface

**Objective**: Align `BaseHSMM` abstract interface so that emission mixins can slot in.

**Tasks**:
1. **[H-02 fix]** Copy `_check_sum_1()` directly into `BaseHSMM` (it is defined on `BaseHMM`, NOT `_AbstractHMM`; extraction to a shared base is out of scope for this plan).
2. **[C-01 fix]** Add `self.implementation = "log"` as a fixed attribute in `BaseHSMM.__init__`. HSMM only supports log-space computations; this satisfies the `self.implementation` reference in all `_emissions` mixins without modifying `_emissions.py`.
3. Replace the inline `_needs_init()` in `hsmm.py` with a direct copy from `base.py` (do not try to inherit it — keep it self-contained on `BaseHSMM`).
4. Add a thin adapter `_accumulate_sufficient_statistics(stats, X, framelogprob, posteriors, ...)` on `BaseHSMM` that calls `_accumulate_emission_sufficient_statistics(stats, X, posteriors)` so emission mixins can extend the chain via `super()`.
5. Ensure `_initialize_sufficient_statistics()` in `BaseHSMM` chains via `super()` so emission mixins can add their own keys.

**Acceptance Criteria**:
- `BaseHSMM._check()` calls `_check_sum_1()` from the method on `BaseHSMM` (not reimplementing inline)
- `self.implementation` attribute exists on any `BaseHSMM` instance
- `_initialize_sufficient_statistics()` chains via `super()` consistently

---

### Milestone 3 — Refactor Concrete HSMM Classes

**Objective**: Replace hand-rolled emission code with mixin composition.

**Tasks**:

#### 3a — `CategoricalHSMM`
1. Change class declaration to `class CategoricalHSMM(_emissions.BaseCategoricalHMM, BaseHSMM):`
2. Remove any inline implementations of:
   - `_compute_log_likelihood()` / `_compute_likelihood()`
   - `_accumulate_emission_sufficient_statistics()` (use mixin via bridge)
   - `_generate_sample_from_state()`
3. Retain only `__init__`, `_init()` (init emission params), `_check()` (emission-specific checks), `_do_emission_mstep()` — delegating to mixin's `_do_mstep()` for the `'e'` character.
4. Confirm `params='stde'` (s=start, t=transmat, d=duration, e=emissionprob) covers all parameters.
5. **[M-01 note]** `BaseCategoricalHMM` uses `__init_subclass__` to wrap `fit`/`decode`/etc. with HMM-centric docstrings. These docstring overrides will appear on `CategoricalHSMM`. This is cosmetically undesirable but functionally harmless. The implementer should override the docstrings on the concrete class to reflect HSMM semantics.

#### 3b — `GaussianHSMM`
1. Change class declaration to `class GaussianHSMM(_emissions.BaseGaussianHMM, BaseHSMM):`
2. Remove inline emission implementations.
3. Retain Gaussian-specific `__init__` (covariance_type, min_covar, priors), `_init()`, `_check()`.
4. `_do_emission_mstep()` delegates to `BaseGaussianHMM`'s M-step for `'m'`, `'c'` params.
5. Confirm `params='stdmc'` (or equivalent) covers start, transmat, duration, means, covars.
6. **[H-01 fix]** `GaussianHSMM` must implement `_needs_sufficient_statistics_for_mean()` → return `'m' in self.params` and `_needs_sufficient_statistics_for_covars()` → return `'c' in self.params`. These are called by `BaseGaussianHMM._accumulate_sufficient_statistics()` and will raise `NotImplementedError` without them.

#### 3c — `PoissonHSMM`
1. Change class declaration to `class PoissonHSMM(_emissions.BasePoissonHMM, BaseHSMM):`
2. Remove inline emission implementations.
3. Retain Poisson-specific `__init__` (lambdas_prior, lambdas_weight), `_init()`, `_check()`.
4. `_do_emission_mstep()` delegates to `BasePoissonHMM`'s M-step for `'l'` param.

**Acceptance Criteria**:
- Each concrete class has zero inline emission logic (observation likelihood, sampling, stats accumulation)
- All existing tests in `test_hsmm.py` pass without modification

---

### Milestone 4 — Test Validation

**Objective**: Confirm no regressions.

**Tasks**:
1. Run `pytest src/hmmlearn/tests/test_hsmm.py -v` — all must pass.
2. Run `pytest src/hmmlearn/tests/` — no new failures introduced.
3. Verify sampling, scoring, and EM fitting work for all three refactored classes.

**Acceptance Criteria**:
- Full test suite green (or at parity with pre-refactor baseline)
- `GaussianHSMM`, `CategoricalHSMM`, `PoissonHSMM` remain importable from `hmmlearn.hsmm`

---

### Milestone 5 — Version and Release Artifacts

**Objective**: Record changes for v0.4.0 release.

**Tasks**:
1. Add entry to `CHANGES.rst` under a new `Version 0.4.0` heading documenting the refactor.
2. Confirm no version file changes needed (project uses `setuptools_scm` / git tags).
3. Verify `__all__` in `hsmm.py` still exports the same public names.

**Acceptance Criteria**:
- `CHANGES.rst` updated
- Public API unchanged (no removed classes or methods)

---

## Testing Strategy

- **Unit tests**: Existing `test_hsmm.py` suite is the primary regression guard.
- **Integration**: Smoke-test that each HSMM class can fit → predict → sample → score without error.
- **Parametric coverage**: Duration distributions (poisson, gaussian, negative_binomial, uniform) × emission types (Gaussian, Categorical, Poisson).
- Critical scenario: MRO resolution for `super()` calls in `_initialize_sufficient_statistics` and `_check` — must not raise `AttributeError`.

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `self.implementation` referenced by all `_emissions` mixins but absent from `BaseHSMM` | **Confirmed** | Add `self.implementation = "log"` in `BaseHSMM.__init__` (Milestone 2, Task 2) |
| `_needs_sufficient_statistics_for_mean/covars` missing from `GaussianHSMM` | **Confirmed** | Implement both in `GaussianHSMM` (Milestone 3b, Task 6) |
| `_check_sum_1` not on `_AbstractHMM`, only `BaseHMM` | **Confirmed** | Copy directly into `BaseHSMM` (Milestone 2, Task 1) |
| MRO conflicts between emission mixin and `BaseHSMM` | Low | C3 linearization is deterministic; verify in Milestone 1 |
| `_accumulate_sufficient_statistics` signature mismatch | Medium | Add bridge adapter in `BaseHSMM` (Milestone 2, Task 4) |
| Test breakage from `super()` chain changes | Low | Run full test suite immediately after each sub-task |
| `BaseCategoricalHMM.__init_subclass__` docstring side-effects | Low | Override docstrings on `CategoricalHSMM` (cosmetic only) |

---

## Handoff Notes

- **Implementer**: Start with Milestone 1 (audit). Do NOT modify `_emissions.py` public API; adapt only `hsmm.py`.
- **Dependencies for Plan 002**: This refactor is a hard prerequisite for `GaussianMixtureHSMM`. Plan 002 should not start until this plan's tests pass.
