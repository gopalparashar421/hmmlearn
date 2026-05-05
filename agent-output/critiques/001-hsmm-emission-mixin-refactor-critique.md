---
ID: 001
Origin: 001
UUID: a3f8c21d
Status: OPEN
---

# Critique — Plan 001: Refactor HSMM to Use Emission Mixin Pattern

| Field | Value |
|---|---|
| **Artifact** | agent-output/planning/001-hsmm-emission-mixin-refactor.md |
| **Critique Date** | 2026-05-05 |
| **Status** | OPEN |
| **Verdict** | REJECTED — must address Critical and High findings before handoff |

## Changelog

| Date | Handoff | Request | Summary |
|------|---------|---------|---------|
| 2026-05-05 | Planner → Critic | Initial review | First review pass against actual source code |

---

## Value Statement Assessment

> *"As a contributor or maintainer of hmmlearn, I want the HSMM module to follow the same emission-mixin composition pattern…"*

The value is clear, outcome-focused, and achievable **in principle** for `GaussianHSMM` and `CategoricalHSMM`. However, as detailed below, `PoissonHSMM` cannot use `BasePoissonHMM` directly in the current codebase without a deeper change to `_emissions.py` that is currently out-of-scope for this plan. The value statement cannot be delivered as written until the inheritance discrepancy is resolved.

---

## Overview

The plan correctly diagnoses the DRY problem and correctly identifies the mixin pattern used in `hmm.py`. The milestone structure is logical. However, the plan's central assumption — that all `_emissions.Base*` mixins are `_AbstractHMM`-derived and therefore safely composable with `BaseHSMM` — is factually incorrect for two of the three target classes. This is not a minor gap; it changes the implementation approach for `PoissonHSMM` entirely.

---

## Architectural Alignment

The plan is structurally aligned with the codebase's existing composition pattern. `BaseCategoricalHMM` and `BaseGaussianHMM` are confirmed `_AbstractHMM` subclasses and are composable as described. The bridge approach for `_accumulate_sufficient_statistics` (Milestone 2 Task 3) is the correct design. The `super()` chaining discipline for `_initialize_sufficient_statistics` is the right call.

---

## Scope Assessment

Scope is appropriate. The out-of-scope list is correctly bounded. The plan wisely avoids touching `_emissions.py`'s public API.

---

## Technical Debt Risks

- The refactor deliberately removes duplication. If done correctly, this reduces long-term debt.
- **Incorrect**: If `BasePoissonHMM` coupling to `BaseHMM` is not addressed, the result would be a new category of subtle runtime failures (AttributeError on `self.implementation`) rather than eliminated debt.

---

## Findings

### Finding C-01 — CRITICAL | OPEN

**Issue**: Assumption #1 is factually wrong for `PoissonHSMM`

**Description**: Assumption 1 states that all `_emissions.Base*` mixins depend only on `_AbstractHMM`. This is **wrong for `BasePoissonHMM`**.

Inspecting `_emissions.py`:
```python
class BaseCategoricalHMM(_AbstractHMM): ...  # safe ✓
class BaseGaussianHMM(_AbstractHMM):    ...  # safe ✓
class BasePoissonHMM(BaseHMM):          ...  # UNSAFE ✗
```

`BasePoissonHMM` inherits from `BaseHMM`, not `_AbstractHMM`. Its `_accumulate_sufficient_statistics` calls:
```python
super()._accumulate_sufficient_statistics(stats, obs, lattice, posteriors, fwdlattice, bwdlattice)
```
This chains into `BaseHMM._accumulate_sufficient_statistics`, which dispatches:
```python
impl = { "scaling": ..., "log": ... }[self.implementation]
```
`BaseHSMM` has **no `implementation` attribute** — it will crash with `AttributeError: 'PoissonHSMM' object has no attribute 'implementation'` at the first `fit()` call.

**Impact**: `PoissonHSMM` cannot use the `BasePoissonHMM` mixin pattern without either:
(a) Adding `implementation` to `BaseHSMM.__init__` (out of scope, risks breaking HSMM), or
(b) Refactoring `BasePoissonHMM` to inherit from `_AbstractHMM` instead of `BaseHMM` (modifies `_emissions.py` public chain — currently out of scope), or
(c) Writing a full custom bridge that bypasses `super()` from `BasePoissonHMM`, effectively reimplementing its accumulation logic — negating the DRY benefit.

**Recommendation**: Planner must explicitly re-evaluate whether `PoissonHSMM` belongs in this plan's scope, or whether it is deferred to a follow-on. If included, the plan must add a task that either: (i) moves `BasePoissonHMM` to inherit `_AbstractHMM` (touching `_emissions.py`) and update the out-of-scope list, or (ii) maintains a hand-written mixin-less `PoissonHSMM` for this release.

---

### Finding H-01 — HIGH | OPEN

**Issue**: `BaseGaussianHMM` has two abstract methods not mentioned in Milestone 3b

**Description**: `BaseGaussianHMM._accumulate_sufficient_statistics` calls:
```python
if self._needs_sufficient_statistics_for_mean():
    ...
if self._needs_sufficient_statistics_for_covars():
    ...
```
Both `_needs_sufficient_statistics_for_mean` and `_needs_sufficient_statistics_for_covars` are abstract in `BaseGaussianHMM` (they raise `NotImplementedError`). They are implemented by `GaussianHMM` in `hmm.py`. Milestone 3b lists tasks for `GaussianHSMM` but **does not mention implementing these two methods**.

**Impact**: `GaussianHSMM` will raise `NotImplementedError` during the E-step accumulation the first time `fit()` runs. This would cause Milestone 3's acceptance criterion ("all existing tests pass") to fail.

**Recommendation**: Add tasks to Milestone 3b:
- Implement `_needs_sufficient_statistics_for_mean(self) → bool` — return `True` if `'m' in self.params`
- Implement `_needs_sufficient_statistics_for_covars(self) → bool` — return `True` if `'c' in self.params`

---

### Finding H-02 — HIGH | OPEN

**Issue**: `_check_sum_1` lives on `BaseHMM`, not `_AbstractHMM`

**Description**: Milestone 2 Task 1 states: "Ensure `BaseHSMM` exposes `_check_sum_1()` (inherit from `_AbstractHMM` or add delegation)." However, inspecting `base.py`, `_check_sum_1` is defined on `BaseHMM` (line ~950), not on `_AbstractHMM`. `_AbstractHMM` declares `_check` as abstract but does not provide `_check_sum_1`.

**Impact**: The stated approach ("inherit from `_AbstractHMM`") will not expose `_check_sum_1` because `_AbstractHMM` doesn't have it. The task will silently fail unless the implementer notices.

**Recommendation**: The plan must be corrected. Options:
(a) Extract `_check_sum_1` from `BaseHMM` up to `_AbstractHMM` (a change to `base.py`, currently not in scope),
(b) Explicitly add `_check_sum_1` directly to `BaseHSMM` (a safe copy, noted as intentional duplication until extraction),
(c) Expand scope to include extracting `_check_sum_1` to `_AbstractHMM`. Update out-of-scope list accordingly.

---

### Finding M-01 — MEDIUM | OPEN

**Issue**: `CategoricalHSMM` + `BaseCategoricalHMM` MRO wraps public API methods via `__init_subclass__`

**Description**: `BaseCategoricalHMM.__init_subclass__` (lines 33–47 of `_emissions.py`) wraps `decode`, `fit`, `predict`, `predict_proba`, `sample`, `score`, `score_samples` with docstring modifiers for categorical HMM doc. For `CategoricalHSMM`, these wraps will apply but the doc change ("(n_samples, 1)" replacement) may produce confusing API documentation for HSMM users. More importantly, the wraps use `getattr(cls, name)`, which will pick up HSMM's own implementations of `score`, `fit`, etc.

**Impact**: Low functional risk (the wraps are pure doc wraps using `functools.wraps`), but the docstring injection of `_CATEGORICALHMM_DOC_SUFFIX` will appear in all HSMM methods, which is misleading for HSMM users.

**Recommendation**: Note this side-effect in Milestone 3a and decide whether `CategoricalHSMM` should override `__init_subclass__` to suppress the doc wrap, or accept the docs as-is.

---

### Finding L-01 — LOW | OPEN

**Issue**: Milestone 5 omits checking for `_duration_model_` in `__all__` and naming consistency

**Description**: `BaseHSMM` uses `_duration_model_` as an internal attribute. If mixin classes add their own `_check()` methods that expect attributes also named `_duration_model_` from the base, there's no conflict. This is fine but worth noting in the acceptance criteria.

**Recommendation**: No action required, but add a smoke test that `GaussianHSMM._duration_model_` is accessible after `fit()` to prevent regression.

---

## Questions for Planner

1. Is `PoissonHSMM` mixin refactoring in scope for v0.4.0, given that `BasePoissonHMM(BaseHMM)` requires a deeper change? Or should it be explicitly deferred and `PoissonHSMM` left as-is?
2. Should extracting `_check_sum_1` to `_AbstractHMM` be added to scope, given it's needed for a clean implementation?
3. Is modifying `_emissions.py` to move `BasePoissonHMM` to `_AbstractHMM` acceptable, given it would unlock both Plan 001 and Plan 002 (Poisson-mixture future work)?

---

## Risk Assessment

| Risk | Likelihood | Impact | Status |
|------|-----------|--------|--------|
| `BasePoissonHMM` MRO crash (C-01) | **Certain if not fixed** | Blocker | OPEN |
| Missing abstract methods in GaussianHSMM (H-01) | High | Failing tests | OPEN |
| `_check_sum_1` not on `_AbstractHMM` (H-02) | High | Silent wrong behaviour | OPEN |
| `__init_subclass__` doc wrap on CategoricalHSMM (M-01) | Low | Misleading docs only | OPEN |

---

## Recommendations

1. **BLOCK on C-01**: Explicitly state the PoissonHSMM strategy before implementation begins.
2. **BLOCK on H-01**: Add the two missing abstract method implementations to Milestone 3b.
3. **RESOLVE H-02**: Clarify the `_check_sum_1` extraction strategy and update the out-of-scope list.
4. **OPTIONAL**: Consider refactoring `BasePoissonHMM` to inherit `_AbstractHMM` (a small but breaking-if-MRO-depended change) to properly unify the pattern.

---

## Revision History

| Date | Revision | Changes | Status Changes |
|------|----------|---------|----------------|
| 2026-05-05 | Initial | First pass | All findings OPEN |
