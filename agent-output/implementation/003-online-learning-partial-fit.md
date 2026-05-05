---
ID: 003
Origin: 003
UUID: c9d27f4b
Status: Active
---

# Implementation 003 — Online / Incremental Learning via `partial_fit`

| Field | Value |
|---|---|
| **Plan Reference** | agent-output/planning/003-online-learning-partial-fit.md |
| **Date** | 2026-05-05 |

## Changelog

| Date | Handoff | Request | Summary |
|------|---------|---------|---------|
| 2026-05-05 | Implementer | Plan 003 | Initial implementation |

---

## Implementation Summary

Added `partial_fit(X, lengths, alpha)` to `BaseHSMM` for incremental/online learning via mini-batch EM with forgetting factor. Added `stats_accumulated_` persistence to `BaseHSMM.fit()`. Added `NotImplementedError` stub to `BaseHMM.partial_fit`. All three HSMM subclasses (`GaussianHSMM`, `CategoricalHSMM`, `PoissonHSMM`) inherit `partial_fit` automatically.

---

## Milestones Completed

- [x] M1 — Sufficient statistics categorisation (forgettable vs structural)
- [x] M2 — `BaseHSMM.partial_fit` implemented
- [x] M3 — `BaseHMM.partial_fit` stub added
- [x] M4 — Test coverage (15 tests, all pass)

---

## Files Modified

| Path | Changes | Lines |
|------|---------|-------|
| src/hmmlearn/hsmm.py | Added `partial_fit` method to `BaseHSMM`; added `self.stats_accumulated_ = stats` at end of `fit()` | ~90 lines added |
| src/hmmlearn/base.py | Added `partial_fit` stub to `BaseHMM` | ~15 lines added |
| src/hmmlearn/tests/test_hsmm.py | Added `TestPartialFit` class with 15 tests | ~125 lines added |

---

## Code Quality Validation

- [x] No syntax errors (all tests run)
- [x] Existing 383 tests pass, 0 failures
- [x] 15 new `partial_fit` tests pass
- [x] Smoke test passes (all 4 scenarios)

---

## Value Statement Validation

> **As a practitioner working with streaming or large-scale time-series data**, I want a `partial_fit(X, lengths)` API on HSMM models, so that I can incrementally update a fitted model as new data arrives.

**Delivers**: `GaussianHSMM`, `CategoricalHSMM`, `PoissonHSMM` all support `model.partial_fit(X, lengths)`. Sequential calls work. `alpha` forgetting factor works. `partial_fit` after `fit()` preserves parameters. HMM stub raises `NotImplementedError` with informative message.

---

## TDD Compliance

| Function/Class | Test File | Test Written First? | Failure Verified? | Failure Reason | Pass After Impl? |
|---|---|---|---|---|---|
| `BaseHSMM.partial_fit` | test_hsmm.py::TestPartialFit | ✅ Yes | ✅ Yes | AttributeError: no attribute 'partial_fit' | ✅ Yes |
| `BaseHMM.partial_fit` (stub) | test_hsmm.py::TestPartialFit::test_hmm_stub_raises_not_implemented | ✅ Yes | ✅ Yes | AttributeError: no attribute 'partial_fit' | ✅ Yes |

---

## Test Coverage

**Unit tests added** (15 total in `TestPartialFit`):
- `test_partial_fit_returns_self`
- `test_partial_fit_fresh_model_initializes`
- `test_partial_fit_sequential_calls`
- `test_partial_fit_chaining`
- `test_partial_fit_after_fit_preserves_parameters`
- `test_stats_accumulated_set_after_fit`
- `test_stats_accumulated_set_after_partial_fit`
- `test_monitor_accessible_after_partial_fit_only`
- `test_alpha_forgetting_factor`
- `test_alpha_invalid_zero_raises`
- `test_alpha_invalid_negative_raises`
- `test_alpha_invalid_above_one_raises`
- `test_categorical_hsmm_partial_fit`
- `test_poisson_hsmm_partial_fit`
- `test_hmm_stub_raises_not_implemented`

## Test Execution Results

```
python -m pytest src/hmmlearn/tests/test_hsmm.py::TestPartialFit -v
15 passed in 9.80s

python -m pytest src/hmmlearn/tests/ -q
383 passed, 22 xfailed, 4 xpassed in 289.62s
```

---

## Outstanding Items

None.

## Next Steps

QA → UAT.
