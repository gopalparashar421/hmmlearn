---
ID: 003
Origin: 003
UUID: c9d27f4b
Status: Active
Target Release: v0.4.0
---

# Plan 003 — Online / Incremental Learning via `partial_fit`

| Field | Value |
|---|---|
| **ID** | 003 |
| **Target Release** | v0.4.0 |
| **Epic Alignment** | Feature expansion — streaming and incremental learning |
| **Status** | Active |
| **Created** | 2026-05-05 |

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-05-05 | Planner | Initial draft |

---

## Value Statement and Business Objective

> **As a practitioner working with streaming or large-scale time-series data**, I want a `partial_fit(X, lengths)` API on HSMM models (and stubs on HMM models), so that I can incrementally update a fitted model as new data arrives — without refitting from scratch — enabling online learning and memory-efficient training on large datasets.

---

## Background

The current `fit()` method on all hmmlearn models:
1. Reinitializes parameters (`_init()`) unless called with previously fitted parameters
2. Runs a complete EM loop over all provided data
3. Does not accumulate state across calls

This makes it impossible to:
- Update a model as new sequences arrive one at a time
- Train on data too large to fit in memory
- Adapt a deployed model to concept drift

`partial_fit` follows the scikit-learn convention for online estimators. The implementation strategy is **mini-batch EM with a forgetting factor**, where:
- Each `partial_fit` call runs one E-step + one M-step on the provided batch
- A configurable forgetting factor `alpha ∈ (0, 1]` controls how much weight prior statistics receive relative to new data
- `alpha=1.0` is pure batch accumulation (no forgetting); `alpha < 1.0` allows adaptation to drift

---

## Scope

**Priority implementation** (full):
- `BaseHSMM.partial_fit(X, lengths, alpha=1.0)`
- All three existing HSMM concrete classes (`GaussianHSMM`, `CategoricalHSMM`, `PoissonHSMM`) gain `partial_fit` automatically via inheritance

**Stubs** (raise `NotImplementedError` with informative message):
- `BaseHMM.partial_fit(X, lengths, alpha=1.0)`
- All `BaseHMM` subclasses inherit the stub

**Out of scope**:
- Full online HMM implementation (future work)
- Variational HSMM online learning
- Distributed or multi-worker learning
- Concept drift detection

---

## Assumptions

1. `partial_fit` does NOT call `_init()` on subsequent invocations — parameters are preserved between calls.
2. The first call to `partial_fit` on an unfitted model should initialize parameters (call `_init()`), equivalent to one EM iteration.
3. The forgetting factor applies to **accumulated sufficient statistics**: `stats_total = alpha * stats_old + stats_new`.
4. One EM step per `partial_fit` call is the correct granularity. Multi-iteration partial fits are out of scope.
5. `partial_fit` does not update `monitor_` in the same way as `fit` — convergence checking across calls is the caller's responsibility.
6. `check_is_fitted` from sklearn handles the "first call" detection.

---

## Milestones

### Milestone 1 — Design the Forgetting Factor Statistics Accumulation

**Objective**: Define how `partial_fit` accumulates and decays sufficient statistics across calls.

**Tasks**:
1. Identify all sufficient statistics structures in `BaseHSMM._initialize_sufficient_statistics()` (nobs, start, trans, dur, plus emission-specific).
2. Define which statistics are accumulated (sum over sequences) vs. overwritten (e.g., nobs).
3. Specify the forgetting factor update rule:
   - Before new batch: `stats_accumulated[key] *= alpha` for all relevant keys
   - After E-step on batch: `stats_accumulated[key] += stats_batch[key]`
4. Decide where to store `stats_accumulated_` — as a model attribute (persisted between calls).
5. Define first-call behaviour: if `stats_accumulated_` does not exist, initialize fresh and call `_init()`.

**Acceptance Criteria**:
- Written spec (can be inline docstring in `partial_fit`)
- All stat fields clearly categorized as forgettable vs. structural

---

### Milestone 2 — Implement `BaseHSMM.partial_fit`

**Objective**: Implement the core online learning loop in `BaseHSMM`.

**Tasks**:

#### 2a — Method Signature
```
partial_fit(self, X, lengths=None, alpha=1.0)
```
- `X`: observation array (same format as `fit`)
- `lengths`: sequence lengths (same as `fit`)
- `alpha`: forgetting factor, float in (0, 1]. `1.0` = no forgetting (accumulate forever).

#### 2b — First-Call Initialization
1. Detect first call: check for absence of `stats_accumulated_` attribute (or `is_fitted` check).
2. On first call: call `_check()`, call `_init(X, lengths)`, call `_check()` again post-init.
3. Initialize `self.stats_accumulated_` with `_initialize_sufficient_statistics()`.

#### 2c — E-Step on Batch
1. Run one pass of `_do_estep(X, lengths)` on the provided batch → produces `stats_batch`.
2. No loop, no convergence check — just one E-step.

#### 2d — Forgetting Factor Application
1. For each stat key in `stats_accumulated_`: `stats_accumulated_[key] *= alpha`
2. Merge batch stats: `stats_accumulated_[key] += stats_batch[key]`
3. Use the same merge logic as M-step expects.

#### 2e — M-Step
1. Call `_do_mstep(self.stats_accumulated_)`.
2. This updates all model parameters in-place.

#### 2f — Return Value
1. Return `self` (sklearn convention for `partial_fit`).

#### 2g — Input Validation
1. Validate `X` with `check_array`.
2. Validate `alpha` is a float in (0, 1].
3. Validate sequence lengths consistency.

**Acceptance Criteria**:
- `model.partial_fit(X1, lengths1).partial_fit(X2, lengths2)` works
- Calling `partial_fit` once on all data produces similar (not necessarily identical) result to `fit` with `n_iter=1`
- Model state (`startprob_`, `transmat_`, duration params, emission params) updates after each call

---

### Milestone 3 — Stub `BaseHMM.partial_fit`

**Objective**: Reserve the API surface for HMM without implementing.

**Tasks**:
1. Add `partial_fit(self, X, lengths=None, alpha=1.0)` to `BaseHMM` in `base.py`.
2. Method body: raise `NotImplementedError` with message: `"partial_fit is not yet implemented for HMM models. It is available for HSMM models. Contributions welcome."`
3. Add the stub signature to the API docstring in `doc/source/api.rst`.

**Acceptance Criteria**:
- `GaussianHMM().partial_fit(X)` raises `NotImplementedError` with the above message
- No crash, no silent wrong behaviour

---

### Milestone 4 — Test Coverage

**Objective**: Verify correctness of `partial_fit` for all HSMM classes.

**Tasks**:
1. Add `test_partial_fit` section to `test_hsmm.py` (or create `test_hsmm_online.py`):
   - Single call: `partial_fit` on full dataset ≈ `fit(n_iter=1)` result
   - Sequential calls: fitting one sequence at a time eventually converges
   - Forgetting factor: `alpha=0.5` causes old statistics to decay (test that parameters drift toward new data distribution)
   - First-call initialization: calling on unfitted model initializes parameters
   - Return value: `partial_fit` returns `self`
2. Test all three HSMM emission types.
3. Verify `BaseHMM.partial_fit` stub raises correct error.

**Acceptance Criteria**:
- Tests pass for GaussianHSMM, CategoricalHSMM, PoissonHSMM
- Stub test passes for GaussianHMM
- No regressions in existing tests

---

### Milestone 5 — Documentation

**Objective**: Surface the new method in API docs and tutorial.

**Tasks**:
1. Add `partial_fit` to `doc/source/api.rst` under the HSMM section.
2. Add a short usage note to `doc/source/tutorial.rst` showing the online learning pattern.
3. Optionally add a minimal example in `examples/plot_hsmm.py` demonstrating incremental fitting.

**Acceptance Criteria**:
- API docs list `partial_fit` with correct signature and docstring
- Tutorial section explains the forgetting factor concept

---

### Milestone 6 — Version and Release Artifacts

**Objective**: Update release artifacts for v0.4.0.

**Tasks**:
1. Add `partial_fit` (HSMM) and stub (HMM) to `CHANGES.rst` under `Version 0.4.0`.

**Acceptance Criteria**:
- `CHANGES.rst` updated

---

## Testing Strategy

- **Unit**: `partial_fit` signature, return value, first-call initialization, `alpha` validation.
- **Behavioural**: Multiple sequential calls should monotonically improve log-likelihood on held-out data (with `alpha=1.0` and enough data).
- **Equivalence**: `partial_fit(all_data)` with `n_iter=1` baseline comparison against `fit(all_data, n_iter=1)`.
- **Forgetting**: With `alpha < 1.0`, a model fitted on distribution A that sees distribution B data should shift toward B.
- **Stub**: `BaseHMM.partial_fit` raises `NotImplementedError`.

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Single E-step per call may not converge for complex models | Medium | Document this is intentional; users control convergence via call frequency |
| Forgetting factor makes statistics non-integer (Dirichlet priors expect integers) | Low | Statistics are expected values (floats throughout), so this is fine |
| HSMM E-step is expensive; `partial_fit` won't hide that | Low | Document expected cost; no optimization required in this plan |
| `stats_accumulated_` grows stale on refitting | Low | If `fit()` is called, `stats_accumulated_` should be reset; add reset logic in `_init()` |

---

## Open Questions

**OPEN QUESTION [RESOLVED]**: Should `partial_fit` support `n_iter > 1`? — No. One E-step + M-step per call is standard for online EM. Multi-iteration is just calling `partial_fit` multiple times.

**OPEN QUESTION [RESOLVED]**: Should this plan also implement `warm_start=True` on `fit()`? — Out of scope. `warm_start` is a different pattern (reuse params as initial point for full EM). Can be addressed separately.

---

## Handoff Notes

- **Implementation order**: Milestone 3 (stub) can be done independently and in parallel with Milestones 1-2.
- **HSMM E-step reuse**: `partial_fit` should call the existing `_do_estep()` directly — do not duplicate E-step logic.
- **Plan 001 dependency**: `partial_fit` depends on the emission mixin refactor being in place so that emission sufficient statistics are accumulated correctly via the mixin chain.
- **Scope discipline**: Do NOT implement online HMM. Only stub it.
