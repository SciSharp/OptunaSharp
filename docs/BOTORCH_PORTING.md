# BoTorch Porting — Completion Plan

> **STATUS: COMPLETE ✅ (all phases done).** `OptunaSharp.Interop.BoTorch` now covers the entire
> `optuna_integration.BoTorchSampler` surface — single-objective (Phase A), **multi-objective (Phase B:
> qEHVI / EHVI / qNEHVI / qParEGO / qLogEI-ParEGO), constraints (Phase C), knowledge-gradient (Phase D:
> qKG / qHVKG), and CPU/GPU `device` (Phase E)** — all selected through the `BoTorchCandidatesFunc` enum
> (the public surface stays `PyObject`-free). Every working `candidates_func` is **0-ULP** vs its
> `optuna_integration.botorch.*` twin; the only non-parity spot is *constrained qKG*, which raises the
> **identical** botorch `RuntimeError` on both sides (a real botorch limitation, faithfully reproduced).
> Gated by `BoTorchApiParityTests` (31, function-level byte-exact) + `BoTorchSamplerParityTests` (15,
> end-to-end replay incl. the CUDA device smoke test) + `BoTorchIntegrationParityTests` (5, the
> higher-level end-to-end paths: running-trial `X_pending` parallel optimization for qEI & qEHVI, the
> deprecated `constraintsFunc` path, and a mixed int+float multi-objective space — all bit-exact), all on
> `net8.0` + `net10.0`. The custom-callable `candidates_func` (§7) remains deliberately deferred (enum-only
> surface).

Governs completing **`OptunaSharp.Interop.BoTorch`** to full parity with
`optuna_integration.BoTorchSampler`. This is the detailed sub-plan for **PORTING_PLAN.md §6.5**.
Reference source of truth:
`refs/optuna-integration/optuna_integration/botorch/botorch.py` (1357 lines) and the live
`botorch` package via the pythonnet interop.

> **Design (unchanged from Phase A).** `BoTorchSampler` is a native OptunaSharp `BaseSampler` that
> ports botorch.py's **glue** (sampler protocol + `candidates_func` selection) to C#, while every
> numeric step is a **direct `botorch` call** through pythonnet — so it is **0-ULP** vs live
> `optuna_integration.BoTorchSampler` and needs **no** `optuna_integration` package (install only
> `botorch`). The public surface stays **pythonnet-free**: acquisitions are chosen by the
> `BoTorchCandidatesFunc` enum, not by handing the user `PyObject`s. Every completion phase keeps
> that contract. Tensor marshaling is already zero-copy through `NumSharp.Interop.pythonnet`
> (`np.array(x, copy:false).ToTorch()` pins + views; results via `ToTorchNDArray`) — **no
> per-phase marshaling work**; the objective matrix, the constraint matrix and the bounds are all
> float64, which that path already handles.

---

## 1. What Phase A shipped (DONE)

Single-objective, in `src/OptunaSharp.Interop.BoTorch/`:

- **4 candidates_func** ported near-verbatim as Python snippets run through `botorch.evalStmts`
  (`BoTorchCandidates.cs`): `LogEi` (`logei_candidates_func`, the SO default), `Qei`, `QLogEi`,
  `QNei`.
- **`BoTorchCandidatesFunc` enum**: `Auto, LogEi, Qei, QLogEi, QNei` (`BoTorchCandidates.cs:14`).
- **Sampler protocol** (`BoTorchSampler.cs`): intersection search space, one-hot
  `SearchSpaceTransform`, startup trials via the bit-exact `RandomSampler`, running-trial handling,
  maximization sign-flip, per-trial `manual_seed` + seed advance. **The MO value matrix is already
  built** (`completedValues[i,j]` with per-direction flip) and `nObjectives` is already passed to
  `Resolve` — MO is gated only by the throw at `BoTorchCandidates.cs:79`.
- **`Resolve`** = a partial `_get_default_candidates_func`: SO branch only; throws for
  `nObjectives > 1`.
- **Constraints**: the sampler passes `hasConstraint:false` and `train_con = null` always; each
  candidates_func rejects a non-null `train_con` (`ConstraintsNotYetSupported`).
- **Gate**: `test/OptunaSharp.Tests.Interop/BoTorchApiParityTests.cs` (11, function-level byte-exact
  vs each `optuna_integration.botorch.*_candidates_func`) + `BoTorchSamplerParityTests.cs` (8,
  end-to-end), both TFMs.

---

## 2. The gap — full `candidates_func` inventory (botorch.py)

| Python `candidates_func` | Nature | `@experimental` | botorch req. | Phase | Enum member (proposed) |
|---|---|---|---|---|---|
| `logei_candidates_func` | SO | 3.3.0 | ≥0.8.1 | **A ✅** | `LogEi` |
| `qei_candidates_func` | SO | — | — | **A ✅** | `Qei` |
| `qlogei_candidates_func` | SO | — | ≥0.9.0 | **A ✅** | `QLogEi` |
| `qnei_candidates_func` | SO noisy | — | — | **A ✅** | `QNei` |
| `qehvi_candidates_func` | MO (≤3 obj) | — | — | **B** | `Qehvi` |
| `ehvi_candidates_func` | MO, no constraint | 3.5.0 | — | **B** | `Ehvi` |
| `qnehvi_candidates_func` | MO noisy | 3.1.0 | — | **B** | `Qnehvi` |
| `qparego_candidates_func` | MO (>3 obj / constrained) | — | — | **B** | `QParEgo` |
| `qlogei_parego_candidates_func` | MO, log-stable ParEGO | 4.9.0 | ≥0.9.0 | **B** | `QLogEiParEgo` |
| `qkg_candidates_func` | SO knowledge-gradient | 3.3.0 | — | **D** | `Qkg` |
| `qhvkg_candidates_func` | MO hypervolume-KG | 4.0.0 | ≥0.9.5 | **D** | `Qhvkg` |

Cross-cutting, not a new func:
- **Constraints (`train_con`)** — a branch inside every func (`_get_constraint_funcs`, +
  `constraints_func` sampler param) → **Phase C**.
- **`device`** (CPU/GPU) — sampler param, `.to(device)` on every tensor → **Phase E**.
- **User-supplied `candidates_func`** — Python accepts any callable → **Deferred** (§7).

The `Auto` selector (`_get_default_candidates_func`, botorch.py:1043) — port in full:
```
n_obj>3 && !con && !running -> ehvi
n_obj>3                     -> qparego
n_obj>1                     -> qehvi
running                     -> qei
else                        -> logei
```

---

## 3. Phase B — Multi-objective (qEHVI / EHVI / qNEHVI / qParEGO / qLogEI-ParEGO) ✅ DONE

**Goal:** `nObjectives > 1` studies work; the `Auto` MO branches resolve exactly like Python.

**Work**
1. **Enum** (`BoTorchCandidates.cs`): add `Qehvi, Ehvi, Qnehvi, QParEgo, QLogEiParEgo`.
2. **`Resolve`**: replace the `nObjectives > 1` throw with the full `_get_default_candidates_func`
   selector; add the explicit-selection switch arms.
3. **Sampler** (`BoTorchSampler.cs`): drop the "single-objective only" implication — the value
   matrix already carries `nObjectives` columns; `RequireSingleObjective` guards stay only on the SO
   funcs. Nothing else in `SampleRelative` changes (bounds/values/running already generic).
4. **Bodies** (`BoTorchCandidates.cs`, near-verbatim Python via `botorch.evalStmts`): port
   `qehvi` (botorch.py:479), `ehvi` (:574), `qnehvi` (:639), `qparego` (:730), `qlogei_parego`
   (:802). Note the per-func `num_restarts=20, raw_samples=1024` (vs SO's 10/512) and the
   `nonnegative: True` option on qEHVI/qNEHVI.
5. **`Imports`** additions (the botorch symbols the MO bodies resolve against):
   `NondominatedPartitioning`, `monte_carlo.qExpectedHypervolumeImprovement`,
   `ExpectedHypervolumeImprovement`, `qLogExpectedHypervolumeImprovement` (version-gated),
   `IdentityMCMultiOutputObjective`, `get_chebyshev_scalarization`, `sample_simplex`,
   `GenericMCObjective`. Also port the **`_get_sobol_qmc_normal_sampler(n)`** helper (botorch.py has
   a version-aware constructor) — the MO funcs call it with 256; keep the SO funcs' inline
   `SobolQMCNormalSampler(torch.Size([256]))` unless unifying.
6. **qEHVI version fallback** (botorch.py:537–545): use `qLogExpectedHypervolumeImprovement` when
   available **and** `partitioning.get_hypercell_bounds().shape[-2] > 0`, else fall back to
   `qExpectedHypervolumeImprovement`. Reproduce the guard so byte-parity holds on the installed
   botorch.

**Gate:** extend `BoTorchApiParityTests` with a byte-exact case per MO func over pinned
multi-objective tensors + seed; add MO scenarios (2-obj and >3-obj) to `BoTorchSamplerParityTests`
so the `Auto` selector is exercised end-to-end. `qEHVI`/`qNEHVI` with **large** objective counts
exercise the `alpha` box-decomposition branch (botorch.py:526).

---

## 4. Phase C — Constraints (`constraints_func` + the `train_con` branch) ✅ DONE

> Note: contrary to an earlier draft of this plan, **`logei` supports constraints** (via
> `LogConstrainedExpectedImprovement`); only **`ehvi`** rejects them. Ported from
> `botorch.py` directly, per the "source of truth" rule.

**Reuse Core's existing constraint infrastructure** — the exact pattern `GPSampler` uses:

- ctor param `Func<FrozenTrial, IEnumerable<double>>? constraintsFunc = null`
  (`GPSampler.cs:96`), stored as `_constraintsFunc`, with the experimental warning.
- `AfterTrial` → `SamplerHelpers.ProcessConstraintsAfterTrial(_constraintsFunc, study, trial, state)`
  (`GPSampler.cs:581`) — computes and stores constraints.
- read back via `ConstrainedOptimization` (`Internal/ConstrainedOptimization.cs`:
  `IsConstrainedOptimization`, `GetConstraintsFromSystemAttrs`, `ConstraintsKey = "constraints"`) and
  `FrozenTrial.Constraints`.

**Work**
1. **Sampler ctor**: add `constraintsFunc` param; store; `AfterTrial` wires
   `ProcessConstraintsAfterTrial`; `BeforeTrial` already delegates.
2. **`SampleRelative`**: build the `con` matrix from completed trials'
   `Constraints` (botorch.py:1238–1274) — `(n_completed, n_constraints)` float64, NaN where a trial
   lacks a value; set `hasConstraint = con != null`; pass the con tensor (via `ToTensor`) as
   `train_con` (was always null). Reproduce the *"constraints_func was given but no call…"* warning
   (botorch.py:1265) and the *"Expected the constraints named…"* RuntimeError (botorch.py:1250).
3. **`_get_constraint_funcs`** (botorch.py:95) — the `Z[..., -n_constraints + i]` slicer. It lives
   inside the Python bodies, so it ports as part of each body's `Imports`/setup (a small Python
   helper defined in the snippet, or inlined per func).
4. **Constrained branch of each candidates_func** — remove `ConstraintsNotYetSupported`; port the
   `if train_con is not None:` branch of each (SO: `qei`/`qnei`/`qlogei` build
   `ConstrainedMCObjective`/constraint kwargs; MO: `qehvi`/`qparego` concat `train_con`, feasibility
   mask, `IdentityMCMultiOutputObjective` + `constraints`). Note `logei`/`ehvi` **reject**
   constraints (botorch.py:156/593) — reproduce those `ValueError`s verbatim.
5. **Deprecation**: `constraints_func` is **deprecated in optuna_integration 5.0.0 (removal 7.0.0)**
   (botorch.py:1154). Port with the deprecation warning to match — do not silently accept it.
6. **Version validation**: `_validate_botorch_version_for_constrained_opt` (botorch.py:86) — the
   per-func botorch-version guard on the constrained path; reproduce the messages.

**Dependency check before starting:** confirm `SamplerHelpers.ProcessConstraintsAfterTrial`,
`FrozenTrial.Constraints`, and the `constraints` system-attr round-trip behave as GPSampler expects
(they do for GPSampler/NSGA-II/CMA-ES). No new Core work is expected.

**Gate:** constrained SO + constrained MO scenarios in both parity suites; the two rejection paths
(`logei`/`ehvi` + constraints) assert the verbatim `ValueError`; a scenario with a trial missing a
constraint value exercises the NaN-fill + warning.

---

## 5. Phase D — Knowledge Gradient (qKG / qHVKG) ✅ DONE

> Note: **constrained qKG** raises botorch's own `RuntimeError` (`shape '[256, 1, 2, 1]' is invalid …`)
> on the live library — reproduced identically by the port (verified both sides raise the same error).
> It is therefore not a byte-exact test case; SO-unconstrained qKG and the multi-value guard are gated.

**Work**
1. **Enum**: add `Qkg, Qhvkg`.
2. **`Resolve`**: explicit-selection arms only (KG is never an `Auto` default).
3. **Bodies**: `qkg` (botorch.py:880, SO — `qKnowledgeGradient`, `num_fantasies=256`,
   `num_restarts=10, raw_samples=512, batch_limit:8`; single-objective guard) and `qhvkg`
   (botorch.py:946, MO — `qHypervolumeKnowledgeGradient` + its current-value helpers).
4. **`Imports`**: `qKnowledgeGradient` (`botorch.acquisition.knowledge_gradient`),
   `qHypervolumeKnowledgeGradient` + the `botorch.acquisition.multi_objective.hypervolume_knowledge_gradient`
   helpers; `ConstrainedMCObjective`.
5. **Version gates**: `qhvkg` requires botorch ≥0.9.5 (botorch.py:967) — reproduce the check +
   message.

**Gate:** byte-exact `qkg`/`qhvkg` cases (SO / MO respectively) in both suites; the qKG SO guard
(`Objective may only contain single values with qKG.`) asserted verbatim.

---

## 6. Phase E — `device` (CPU / GPU) ✅ DONE

> Shipped as a **pythonnet-free `string? device`** ctor param (`"cpu"` default / `"cuda"` / `"cuda:0"`).
> `BoTorchInterop.ToTensor(matrix, device)` keeps the zero-copy pinned view for CPU and does
> `tensor.to(device)` (a copy — a device tensor cannot alias host pinned memory) otherwise; the result
> crosses back through `ToTorchNDArray(force:true)`'s `.cpu()`. Gated by a CUDA smoke test
> (self-Inconclusive without a device); byte-parity holds only for identical device + torch build.

Python threads a `torch.device` (default CPU) and does `.to(device)` on values/params/con/bounds
(botorch.py:1150,1271–1275).

**Work**
1. **Sampler ctor**: add a **pythonnet-free** `device` selector — a `string?` (e.g. `"cpu"`,
   `"cuda"`, `"cuda:0"`) or a small `BoTorchDevice` enum — default CPU.
2. **Marshaling**: our `ToTensor` yields CPU float64 tensors (pinned NumSharp view). For a non-CPU
   device, add a `.to(device)` after `torch.from_numpy(...)` (a copy to device — the pin/view is CPU
   only, which is correct: device tensors cannot alias host pinned memory). Keep CPU as the pure
   zero-copy path.
3. **Result**: `ToTorchNDArray(force:true)` already does `detach().cpu()...numpy()`, so the result
   crosses back from any device unchanged.

**Gate:** CPU path is the existing suite. GPU is **environment-gated** (self-Inconclusive when no
CUDA device) — a single smoke scenario asserting a `cuda` study runs when a device is present;
document that byte-parity vs Python holds only for identical device + torch build.

---

## 7. Deferred — user-supplied `candidates_func`

Python's headline feature is *"use your own `candidates_func`"* (any callable). C# exposes only the
built-in enum. Options, in preference order:

- **(a, default) Enum-only.** Phases B–D ship the **entire built-in suite**, which is the bulk of
  the value. Document that a custom acquisition is not exposed on the managed surface.
- **(b) Python-source escape hatch.** Let advanced users pass a Python snippet (a `candidates_func`
  body string) run through `botorch.evalStmts` — exposes Python as *text*, not `PyObject`, so the
  clean surface is preserved. A separable, optional add-on.
- **(c) Typed native delegate** over a tensor abstraction — rejected: it would leak `PyObject`/torch
  into the public API, the exact thing the design hides.

Recommend (a) now; revisit (b) if a user asks. Not on the critical path.

---

## 8. Cross-cutting rules (apply to every phase)

- **0-ULP contract.** The numerics *are* botorch; a phase is correct iff each new func byte-matches
  `optuna_integration.botorch.<name>` over the **same** tensors + `manual_seed`. Keep the bodies
  **near-verbatim Python** (diff 1:1 against botorch.py) run via `botorch.evalStmts` — do **not**
  translate them to C#/`PyObject` chains (see the Pythonic.cs rationale).
- **Version gating.** Each func carries `@experimental_func` and/or a botorch-version requirement.
  Reproduce the guard + **verbatim** error message (the Phase-A `_imports.check()` pattern), so a too-
  old botorch fails the same way Python does.
- **Marshaling is done.** No per-phase tensor-plumbing work — MO values, the con matrix and bounds
  are float64 and ride the existing pinned zero-copy `ToTensor` / `ToNDArray`.
- **Enum is the surface.** Every new acquisition is a `BoTorchCandidatesFunc` member; the public API
  stays `PyObject`-free.
- **`ExtractCandidate` is unchanged** — the shape validation (`Python.torch.Tensor.dim_t(t)` /
  `.size_t(t, dim)` / `.squeeze(t, dim)`) + `ToNDArray` result crossing already handles any acquisition's
  output. Those tensor members live in the pythonic tree (`Python.torch.Tensor` in `Pythonic.cs`), exposed
  as a **nested class** (not extension members), mirroring Python's own `torch.Tensor.size(t, -1)`.

## 9. Definition of Done (per phase)

1. Enum members added; `Resolve` matches `_get_default_candidates_func`/explicit selection exactly.
2. Each new body byte-exact vs its `optuna_integration.botorch.*` twin in `BoTorchApiParityTests`
   (function-level) **and** exercised end-to-end in `BoTorchSamplerParityTests` (the `Auto`
   selector), both `net8.0` + `net10.0`.
3. Error/version taxonomy verbatim (rejections, guards, deprecation warnings).
4. `dotnet build` + the unit filter green; interop suite green with `botorch` installed
   (self-Inconclusive without it).
5. PORTING_PLAN.md §6.5 updated (phase → DONE); memory note refreshed.

## 10. Reference & file map

| Piece | Reference (botorch.py) | Port location |
|---|---|---|
| candidates_func bodies | :109–1028 | `BoTorchCandidates.cs` (Python-string consts + `evalStmts`) |
| `_get_default_candidates_func` | :1029 | `BoTorchCandidates.Resolve` |
| `_get_constraint_funcs` | :95 | inside the constrained bodies |
| `_validate_botorch_version_for_constrained_opt` | :86 | a `botorch`-facade guard (Pythonic.cs) |
| sampler protocol / con matrix / device | :1056,1194–1317 | `BoTorchSampler.cs` |
| `Imports` (botorch symbols) | per-body | `botorch.Imports` (Pythonic.cs) |
| constraints infra (Core) | — | `Internal/ConstrainedOptimization.cs`, `SamplerHelpers.ProcessConstraintsAfterTrial`, `FrozenTrial.Constraints` (GPSampler pattern) |

## 11. Risks & decisions

- **`constraints_func` is deprecated (5.0.0→7.0.0).** Still port it (with the warning) for parity;
  it is on its way out — do not over-invest, and mirror Python's exact deprecation text.
- **qEHVI ↔ qLogEHVI fallback** is version- **and** data-dependent (empty box decomposition). Port
  the exact guard or byte-parity breaks on some inputs.
- **GPU is untestable in CI** (no CUDA). Ship CPU zero-copy as the parity path; GPU is a documented,
  environment-gated smoke test only.
- **botorch version drift.** The bodies pin behavior to the installed botorch; parity is defined
  against *that* botorch (same lever as Phase A). Pin the interop interpreter's botorch version in
  the test docs.
- **Multi-objective RNG lock-step.** The sampler-replay gates must leave trials RUNNING / advance the
  seed exactly as Phase A does, now across `nObjectives > 1` — reuse the existing ask/tell schedule.

## 12. Suggested order

**B (multi-objective)** → **C (constraints)** → **D (knowledge gradient)** → **E (device)**;
**(b) custom candidates_func** deferred. B unlocks the most user value (MO is a headline BoTorch use)
and needs no Core changes. C reuses Core's constraint infra (verify first, no new Core work expected).
D is additive (explicit-selection only). E is small and CPU-parity is already met. Each phase is
independently shippable behind its own interop gate.
