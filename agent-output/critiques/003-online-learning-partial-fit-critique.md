---
ID: 003
Origin: 003
UUID: c9d27f4b
Status: OPEN
---

# Critique — Plan 003: Online / Incremental Learning via `partial_fit`

| Field | Value |
|---|---|
| **Artifact** | agent-output/planning/003-online-learning-partial-fit.md |
| **Critique Date** | 2026-05-05 |
| **Status** | OPEN |
| **Verdict** | APPROVED WITH CONDITIONS — address High findings before handoff; Mediums can be resolved in implementation |

## Changelog

| Date | Handoff | Request | Summary |
|------|---------|---------|---------|
| 2026-05-05 | Planner → Critic | Initial review | First review pass against actual source code |

---

## Value Statement Assessment

> *"As a practitioner working with streaming or large-scale time-series data, I want a `partial_fit(X, lengths)` API on HSMM models…"*

The value is clear, outcome-focused, and fills a legitimate gap. Mini-batch EM with forgetting factor is a well-established algorithm (Cappé & Moulines, 2009; Neal & Hinton, 1998 for incremental EM). The scoping choice — full implementation on HSMM, stubs on HMM — is pragmatic and honest. The value is achievable.

**Concern**: The `monitor_` integration gap (see M-02) may create a confusing user experience if users call `fit()` then `partial_fit()`. This doesn't block delivery but should be documented.

---

## Overview

This plan is the most self-contained of the three. It does not depend on Plans 001 or 002 directly (though it will benefit from them for `GaussianHSMM`/etc.). The HSMM's existing `_do_estep` / `_do_mstep` infrastructure maps cleanly to one-pass EM. The plan's core design is sound.

The main issues are:
1. Assumption 6 is misleading and could cause the implementer to use the wrong guard mechanism.
2. Input validation on subsequent calls is missing, creating a silent data-shape mismatch risk.
3. The test specification contains an assertion that cannot hold as written (`partial_fit` once ≈ `fit(n_iter=1)`).

---

## Architectural Alignment

- `BaseHSMM._do_estep` already accumulates stats internally and returns them — Plan 003 can call it cleanly.
- `BaseHSMM._do_mstep` takes a stats dict and updates in-place — Plan 003 can call it after forgetting factor application.
- `BaseHSMM` does not inherit from `BaseHMM` or `_AbstractHMM` — placing `partial_fit` on `BaseHSMM` is independent of where the HMM stub lives.
- Placing the HMM stub on `BaseHMM` (not `_AbstractHMM`) is correct: it's an HMM-specific concern. `_AbstractHMM` also has VHMMs and other non-EM models.

---

## Scope Assessment

Appropriate. Deferring the full HMM implementation is the right call. The forgetting factor approach is the simplest valid online EM strategy. `alpha ∈ (0, 1]` validation is correctly bounded.

---

## Technical Debt Risks

- Storing `stats_accumulated_` as an instance attribute means the model becomes stateful in a way that `fit()` doesn't reset. A user who calls `partial_fit()` then `fit()` may be confused about whether `stats_accumulated_` is stale. The plan should document this.
- Mini-batch EM on HSMMs with `O(T × D × N²)` per-sequence complexity means large batches are expensive. This is not a code debt but a user expectation gap worth noting in docs.

---

## Findings

### Finding H-01 — HIGH | OPEN

**Issue**: Assumption 6 is incorrect — `check_is_fitted` does not detect first `partial_fit` call

**Description**: Assumption 6 states: "`check_is_fitted` from sklearn handles the 'first call' detection."

This is wrong. `check_is_fitted` raises `NotFittedError` if the model has not been fitted (no attributes ending with `_`). It cannot distinguish between "never had `partial_fit` called" and "was fitted with `fit()` but not `partial_fit()`". It is not the right tool for detecting whether `stats_accumulated_` exists.

Milestone 2b correctly uses `if not hasattr(self, 'stats_accumulated_')` — which is the right approach. But Assumption 6 contradicts this and could mislead the implementer into adding a redundant or incorrect `check_is_fitted` call.

**Impact**: If the implementer follows Assumption 6, they may guard with `check_is_fitted` and raise `NotFittedError` on the first `partial_fit` call on an unfitted model — which is the exact use case `partial_fit` is designed to handle. This would break the primary online learning workflow.

**Recommendation**: Delete or correct Assumption 6. Correct text: "First-call detection uses `hasattr(self, 'stats_accumulated_')`. The model may be unfitted when `partial_fit` is first called — `_init()` is called in that case."

---

### Finding H-02 — HIGH | OPEN

**Issue**: No input validation on non-first calls — silent shape mismatch risk

**Description**: Milestone 2b specifies `_check()`, `_init()`, `_check()` only on the first call. Subsequent calls have no validation of `X`'s feature dimension against the model's `n_features`. If a user passes data with a different `n_features` on a subsequent call, `_compute_log_likelihood(X)` will produce silently wrong results or raise a cryptic numpy broadcasting error deep in the E-step.

**Impact**: Silent data corruption or unhelpful runtime errors. Per the OWASP principle of "fail loudly at boundaries," input validation must occur on every call.

**Recommendation**: On every `partial_fit` call (not just the first), add:
```python
X = check_array(X)
self._check_and_set_n_features(X)  # raises ValueError on mismatch
```
(Note: `_check_and_set_n_features` already handles dimension mismatch correctly on subsequent calls — it raises `ValueError` if `n_features` was already set and the new X differs.)

---

### Finding M-01 — MEDIUM | OPEN

**Issue**: Test assertion "single call ≈ fit(n_iter=1)" is not reproducible as stated

**Description**: Milestone 4 Task 1 specifies: "Single call: `partial_fit` on full dataset ≈ `fit(n_iter=1)` result."

This cannot hold as written. `fit()` calls `_init(X, lengths)` which randomly initializes parameters (if `init_params` includes relevant chars). `partial_fit` on a fresh model also calls `_init(X, lengths)`. Even with the same `random_state`, the M-step output depends on the initialized values, and the two calls will likely diverge due to internal PRNG state differences between calls.

**Impact**: A flaky test that sometimes passes and sometimes fails, or a test that only works with a very specific setup not documented in the plan.

**Recommendation**: Rewrite the test as: "Fit a model with `fit(n_iter=1)`. Then create an identical model with the same parameters, call `partial_fit` once on the same data. Verify the log-likelihood values are equal within tolerance." The key is starting from the same initialized state, not from scratch.

---

### Finding M-02 — MEDIUM | OPEN

**Issue**: `monitor_` lifecycle after `partial_fit` is unaddressed

**Description**: `BaseHSMM.fit()` creates `self.monitor_` as a `ConvergenceMonitor`. Assumption 5 says `partial_fit` does not update `monitor_`. But if a user calls `fit()` followed by `partial_fit()`, `monitor_` reflects the `fit()` convergence history, not the online update history. If a user calls `partial_fit()` on a never-fitted model (first call), there is no `monitor_` at all — downstream code that accesses `model.monitor_.history` will raise `AttributeError`.

**Impact**: `AttributeError` for users who access `monitor_` after `partial_fit`-only workflows.

**Recommendation**: Add to Milestone 2 or Milestone 5 docs: on first `partial_fit` call, initialize `self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)` (without resetting it between calls). Alternatively, explicitly document that `monitor_` is not set by `partial_fit` and users should not access it.

---

### Finding M-03 — MEDIUM | OPEN

**Issue**: `nobs` stats field semantics clash with forgetting factor

**Description**: `BaseHSMM._initialize_sufficient_statistics` includes `'nobs': 0` (a count of sequences, incremented by 1 per sequence in `_do_estep`). Milestone 1 says to categorize stats as "forgettable vs. structural" but does not provide the categorization.

Applying `alpha` to `nobs` (multiplying it by `alpha < 1.0`) produces a fractional sequence count, which is semantically meaningless and could affect downstream logic if any code uses `stats['nobs']` for normalization (currently none does, but it's a maintenance hazard).

**Impact**: Low immediate risk (no code currently reads `nobs` for normalization in HSMM), but the deferred categorization is an incomplete design.

**Recommendation**: Complete the categorization in Milestone 1. Suggested answer: `nobs` should be treated as structural (accumulate without forgetting — it's a count of how many sequences have been processed). All other stats (`start`, `trans`, `dur`, emission stats) are forgettable. Document this explicitly in `partial_fit`'s docstring.

---

### Finding L-01 — LOW | OPEN

**Issue**: Alpha edge case `alpha=0.0` behavior is undocumented

**Description**: The plan specifies `alpha ∈ (0, 1]` (exclusive lower bound). At `alpha=0`, all prior stats would be zeroed before adding the new batch — effectively, only the current batch contributes to the M-step. This is valid "full forgetting" behavior but the plan rejects `alpha=0.0` via the validation constraint.

**Impact**: Minor — the validation is correct (`alpha=0` would make `partial_fit` stateless, defeating the purpose). But the validation message should explain why.

**Recommendation**: In the `alpha` validation, use: `raise ValueError(f"alpha must be in (0, 1], got {alpha}. Use alpha=1.0 for no forgetting.")` — makes the valid range intuitive.

---

### Finding L-02 — LOW | OPEN

**Issue**: `fit()` followed by `partial_fit()` leaves `stats_accumulated_` stale

**Description**: If a user calls `model.fit(X1)` then `model.partial_fit(X2)`, the `fit()` call does not create `stats_accumulated_`. The first `partial_fit` call will detect `not hasattr(self, 'stats_accumulated_')` and call `_init()` again, re-initializing parameters from scratch rather than continuing from the fitted model.

**Impact**: Confusing behavior — user expects `partial_fit` to continue from the fitted state, but gets a fresh initialization.

**Recommendation**: Milestone 2b first-call logic should check both `stats_accumulated_` absence AND whether the model is already fitted (has `startprob_`). If fitted but no `stats_accumulated_`, skip `_init()` and only initialize `stats_accumulated_`.

---

## Questions for Planner

1. Should `partial_fit()` on an already-`fit()` model skip `_init()`? (Currently it would re-initialize per Finding L-02.)
2. Should `monitor_` be created and updated in `partial_fit`, or explicitly left unset? (Finding M-02.)
3. The `nobs` categorization (forgettable vs. structural) — can this be resolved in the plan rather than deferred to the implementer? (Finding M-03.)

---

## Risk Assessment

| Risk | Likelihood | Impact | Status |
|------|-----------|--------|--------|
| Wrong first-call guard (H-01) | High | Breaks primary workflow | OPEN |
| Missing shape validation on subsequent calls (H-02) | Medium | Silent wrong results | OPEN |
| Flaky test assertion (M-01) | High | Unreliable CI | OPEN |
| `monitor_` AttributeError (M-02) | Medium | Runtime error | OPEN |
| `nobs` forgetting semantics (M-03) | Low | Maintenance hazard | OPEN |
| `fit()` then `partial_fit()` re-init (L-02) | Medium | Confusing behavior | OPEN |

---

## Recommendations

1. **BLOCK on H-01**: Correct Assumption 6. The guard is `hasattr(self, 'stats_accumulated_')`, not `check_is_fitted`.
2. **BLOCK on H-02**: Add `_check_and_set_n_features(X)` to the per-call validation path.
3. **RESOLVE before handoff (M-01)**: Rewrite the flaky test assertion.
4. **RESOLVE before handoff (M-03)**: Complete the forgettable/structural categorization in Milestone 1.
5. **DOCUMENT (M-02, L-02)**: Clarify the `fit()` + `partial_fit()` interaction in the method docstring.

---

## Revision History

| Date | Revision | Changes | Status Changes |
|------|----------|---------|----------------|
| 2026-05-05 | Initial | First pass | All findings OPEN |
