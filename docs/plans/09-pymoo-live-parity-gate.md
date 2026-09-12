# Plan 09 — Live pymoo parity gate for AGE-MOEA-2 (and pymoo-sourced ports)

> **Status: DONE.** Shipped `test/OptunaSharp.Tests.Interop/AgeMoea2ParityTests.cs` — 21 cases
> (5 driver grid rows + 16 operator spot-checks) **green on net8.0 + net10.0** against live pymoo
> 0.6.1.3 (numba 0.65.1 / numpy 2.4.2), and the +1-ULP smoke perturbation confirmed the gate bites.
> Depends on: the committed AGE-MOEA-2 port (Layer 1 `AgeMoea2Optimizer`, done — see the
> [`age-moea-2-port`](../../) memory); the pymoo-on-numpy-2.4.2 migration (done, commit `9c7aec2`).
> Drives pymoo **directly via the interop suite's `PythonSession`/pythonnet** (does NOT hard-depend on
> [08](08-pymoo-interop-sampler-bridge.md)). **Tier X** (test-time foreign runtime). Value ★★ / effort low.
> The live counterpart to the committed differential corpus.

## What landed

- **`AgeMoea2ParityTests`** (`test/OptunaSharp.Tests.Interop`, `[TestCategory("RequiresPython")]`):
  - **`Driver_IsGenerationByGenerationBitExact_VsLivePymoo`** — drives live `AGEMOEA2` via
    `algo.setup(("n_gen", n_gen), seed)` + `while has_next(): next()` (exactly `gen_agemoea2_oracle.py`'s
    `dump_driver`), captures each generation's `X`/`F`, and asserts the native `AgeMoea2Optimizer` matches
    **bit-for-bit** (`BitConverter.DoubleToInt64Bits`) over the corpus grid
    `(42,6,2,8,5)/(7,5,2,10,6)/(0,4,3,12,5)/(2024,8,2,16,4)` **plus a larger 3-obj front
    `(2024,10,3,40,5)`** as the fastmath stress case. Reuses the polynomial box objective
    (`MathUtil.PairwiseSum`, numpy-`np.sum`-faithful).
  - **Operator spot-checks** — `FloatRandomSampling` / `SBX` / `PM` / `TournamentSelection`, live, bit-exact
    (the corpus sections against the live library; localizes any RNG-stream drift).
  - **Decision-equivalence ledger** — the driver accumulates the max ULP gap over the whole run; green ⇒
    0 ULP (which trivially preserves every survivor/argmin/argmax/tournament decision), and a drift makes
    the single final assertion bite and name the worst offender. JIT-off is unavailable (pymoo's
    `find_corner_solutions` uses `numba.intp` as a numpy dtype), so we run **stock JIT-on pymoo** — what
    Mimicry runs and what generated the corpus.
  - **Host-pinned**: uses `PythonSession.EnsureBitExactHostOrInconclusive()` (x64; Inconclusive on the
    macOS arm64 runner, matching CMA-ES / linalg), and `Assert.Inconclusive` without pymoo — hard-fails
    under `OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE=1`.
- **CI**: the interop-test job now installs `pymoo==0.6.1.3 --no-deps` + `autograd alive_progress numba`
  (numpy re-pinned to 2.4.2 after, byte-exact BLAS invariant preserved). This also lights up
  [08](08-pymoo-interop-sampler-bridge.md)'s `PymooSamplerParityTests` / `PymooApiUnitTests`, which
  previously self-skipped in CI for lack of pymoo.
- Data crosses as round-trippable-`repr` JSON (bit-lossless for the bounded finite doubles the box
  produces; `allow_nan=False` would raise loudly otherwise) — the same channel the committed corpus uses.

## Goal

Add a **live** pymoo interop parity test — drive the real pymoo AGE-MOEA-2 via pythonnet in-process and
assert OptunaSharp's native `AgeMoea2Optimizer` reproduces it **generation-by-generation bit-exact**,
the same equality the committed corpus (`test/OptunaSharp.Tests/oracle/agemoea2_operators.json`) checks
offline — but against the *live* library, every CI run. Lives at
`test/OptunaSharp.Tests.Interop/AgeMoea2ParityTests.cs`, `[TestCategory("RequiresPython")]`.

## Motivation

- **The committed corpus can go stale.** It was captured once (pymoo 0.6.1.3). A live gate catches
  pymoo-version drift, a NumSharp MT19937 regression, or a libm change the moment it happens, instead of
  the next time someone regenerates the oracle. This is the differential gate the repo's testing
  philosophy calls for (the BoTorch/PyCma bridges each carry a live parity test); AGE-MOEA-2 shipped with
  only the offline corpus because the numpy pin blocked live pymoo — now resolved.
- **It closes the AGE-MOEA-2 Phase 5 gate** named in `docs/AGE_MOEA_2_PORTING.md` §7 ("live interop parity
  gate … drive live pymoo via pythonnet"), which was explicitly deferred.
- **It de-risks the fastmath question empirically, continuously.** §1.1 of the porting doc predicts the
  `@jit(fastmath=True)` kernels are decision-preserving and (at 2–3 objectives) bit-identical; the gate
  turns that from a one-time measurement into a standing check, and records the max observed ULP gap in a
  `MisalignedRegistry`-style ledger if a config ever trips it.

## Scope / deliverables

- **`AgeMoea2ParityTests`** (`test/OptunaSharp.Tests.Interop`, which is already a Core `InternalsVisibleTo`
  friend, so it can construct the internal `AgeMoea2Optimizer`):
  - **Per-generation bit-exact:** for a grid of `(seed, n_var, n_obj, pop_size, n_gen)`, drive live
    `pymoo.algorithms.moo.age2.AGEMOEA2` through `minimize`/manual `ask`-`tell` capturing each
    generation's `X` and `F`; run the native `AgeMoea2Optimizer` with the same config; assert every
    generation's `X`/`F` **bit-identical** (`BitConverter.DoubleToInt64Bits`), reusing the polynomial box
    objective (bit-exact across numpy/C#, `MathUtil.PairwiseSum` for `n_var ≥ 8`).
  - **Operator spot-checks (optional):** drive live `SBX.do` / `PM.do` / `TournamentSelection.do` /
    `Mating.do` with fixed seeds+inputs and assert the native operators match — the corpus sections, live.
  - **Decision-equivalence ledger:** also run stock (JIT-on) pymoo and record the max ULP gap in the
    survival scores; assert it never flips a survivor/argmin/argmax/tournament decision on the grid
    (§1.1). (JIT-off is unavailable — `numba.intp`-as-dtype breaks un-jitted; documented in the port.)
  - Self-skips **Inconclusive** without a usable Python+pymoo; hard-fails under
    `OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE=1` (the interop-suite contract).
- **Reuse the driving code**: either [08](08-pymoo-interop-sampler-bridge.md)'s `PymooRuntime` +
  `Python.pymoo` facade (preferred — one pymoo entry point), or, if 08 is not yet built, drive pymoo
  directly in the test via the existing `PythonSession`/pythonnet (the gate does **not** hard-depend on
  08). Data crosses via `NumSharp.Interop.pythonnet` (no element loops).
- **CI**: runs in the existing `interop-test` matrix job, which installs Python + optuna; add
  `pip install pymoo==0.6.1.3 --no-deps autograd alive_progress` there (numpy/numba/optuna untouched, as
  proven locally). Document the one-line install.
- **(Optional) `PymooSampler` value-parity**: if 08 is built, add a behavioral + value-parity check of the
  bridge sampler vs live `pymoo.minimize` here too, so both layers share one interop file.

## Design notes

- **Why bit-exactness is reachable live** (it is, for Layer 1): MT19937 is numpy-version-stable, libm is
  the machine's, and the fastmath kernels reduce over 2–3 objectives (no reassociation) — the same reasons
  the corpus proved **byte-identical across numpy 2.0.2 → 2.4.2** (commit `9c7aec2`, 0 ULP over all 14
  sections). The live gate re-establishes this against whatever pymoo/numpy the CI box actually has.
- **GIL discipline**: compute the C# generations first, then open one `PythonRuntime.Run` scope for the
  pymoo run + the comparison; every `PyObject` (incl. `Population`) disposed under the GIL.
- **Grid**: reuse the corpus's `(seeds 42/7/0/2024, pop 8–16, 2–3 obj, n_gen 4–6)` plus at least one
  larger front (pop 40, 3 obj) as the fastmath stress case for the ledger.

## Validation

- The gate itself is the validation; it must be **green on net8.0 + net10.0** in the interop job, and
  **Inconclusive (never red)** on a machine without pymoo. A deliberate one-ULP perturbation of a native
  kernel must turn it red (a smoke check that the assertion bites).

## Out of scope

- Re-deriving or replacing the committed offline corpus — it stays as the no-Python gate for
  `OptunaSharp.Tests`; this plan is the *live* complement, not a replacement.
- Parity for the Layer-2 `AgeMoea2Sampler` beyond behavioral/value-parity (it uses Optuna's operators, not
  pymoo's — not bit-exact by design; see [08](08-pymoo-interop-sampler-bridge.md)).
- A generic pymoo-vs-native parity harness for other algorithms — revisit if/when another pymoo algorithm
  is ported natively (e.g. a native SPEA2/MOEA-D); this plan is AGE-MOEA-2-scoped.
