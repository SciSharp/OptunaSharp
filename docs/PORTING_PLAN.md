# OptunaSharp — Optuna → C# Conversion Plan

**Source of truth:** `optuna/optuna` @ `f6898da` (v5.0.0.dev, cloned at `K:/source/optuna`, vendored as a pinned submodule at `src/optuna`) — 24,474 core SLOC
(tokenizer-counted, comments/docstrings excluded). Secondary reference: `optuna/rustuna` @ `8acb2dd`
(`K:/source/rustuna`) — used ONLY where it accelerates (SQLite schema, Sobol tables, static-type designs);
Optuna's Python behavior always wins on conflict.

**Ordering directives (two, composed):**
1. Core / infra / storage / persistence before features; features by difficulty, easiest first.
2. **Native before external:** everything implementable with **BCL + NumSharp.Core only** (Tier N,
   Phases 0–3) is completed before anything pulling a third-party NuGet, native runtime, or network
   service (Tier X, Phases 4–5). Target: **almost all features**.

**The tier rule (what counts as "external"):** a work item is Tier X iff it requires any dependency
beyond the .NET BCL and `NumSharp.Core` — third-party managed packages (ML.NET, StackExchange.Redis,
Grpc.Net, System.CommandLine, YamlDotNet, Plotly.NET, Microsoft.Data.Analysis, EF Core), native
binaries (Microsoft.Data.Sqlite's e_sqlite3, TorchSharp, `NumSharp.Interop.OpenBLAS`), or external
services (Redis, PostgreSQL/MySQL, S3/GCS, gRPC servers). Everything else is Tier N.
`NumSharp.Interop.OpenBLAS` is first-party but bundles native binaries → Tier X by rule; the native
tier must be fully functional without it (it appears only as an optional byte-exactness upgrade).

**Tier N headline (the release claim this ordering buys):** `OptunaSharp` Core ships **Optuna's
entire algorithm surface — every sampler incl. TPE/NSGA/CMA-ES/GP, every pruner, MO machinery,
PED-ANOVA, terminators, in-memory + journal-file persistence — pure managed, zero dependencies
beyond NumSharp**. Cross-language persistence is ALREADY in this tier: Optuna's journal-file format
is JSON-lines over plain file IO, so Python Optuna and OptunaSharp can share a live journal file
with no external package on either side.

---

## 0. Progress ledger — where we are, where to continue

> **Status as of 2026-09-03.** Legend: ✅ done & green · 🟩 partially landed (some sub-parts done as
> byproducts) · ⬜ not started. Update this section as items land — it is the first thing the next
> agent should read. Per-item detail is in the phase tables below (the same `#` ids).

### ▶ CONTINUE HERE: **Phase 2.9** — TPE stage 2 (`group`, MOTPE refinements)

Phases 0, 1 and 2.1–2.8 are complete, committed, and proven green (303 unit + 97 live-interop
tests, both TFMs, none skipped). **2.8 landed the whole TPE numeric kernel plus the univariate AND
non-group multivariate sampler** — a default `TPESampler()` now optimizes single-objective studies
end-to-end, matching Optuna within the erf tolerance (see the 2.8 entry below). The remaining TPE
work is small:
- **`group=true`** — the group-decomposed multivariate TPE. Constructor currently throws
  `NotImplementedException`; `SearchSpace/GroupDecomposedSearchSpace.cs` is already ported, so this
  is mostly wiring `infer_relative_search_space`/`sample_relative`'s group branches.
- **MO/MOTPE** — MO studies already run through the independent path (`multivariate` defaults to
  False for MO) and the `handle_below` uniform-weights path is ported; what remains rides on **2.10**
  (`_hypervolume/*` — the `_split_complete_trials_multi_objective` HSSP tie-break throws until then)
  and **2.11** (the MOTPE split).
- Everything else in the sampler (`constant_liar` incl. the `tpe:relative_params` sharing, the
  deprecated `weights`/`gamma`/`consider_*` args, `n_startup_trials`, `reseed_rng`) is done.

**Read BOTH divergence notes below** (erf/ucrt AND the new argsort tie-break) — together they bound
TPE's parity exactly.

### Phase status

| Phase | Status | Notes |
|---|---|---|
| 0 — Core framework (0.1–0.9) | ✅ | walking-skeleton gate (0.9) bit-identical to Python for shared seed |
| 1 — Native persistence (1.1–1.5) | ✅ | journal Python↔C# live interop green; **M2 reached** |
| 2 — Native features | 🟩 | **2.1–2.8 done**; 2.9 (`group`) not started; 2.10/2.11 partially landed (see below) |
| 3 — Native heavy tail | ⬜ | not started |
| 4 — External storage (Tier X) | ⬜ | not started; `ICachedStorageBackend` seam for 4.1 already declared in 1.2 |
| 5 — External integrations (Tier X) | ⬜ | not started |

### Done work items (with the commit that landed them)

- ✅ **0.1–0.9** — `feat(core): Phase 0 complete` (`2763190`). Exceptions/enums/logging seam,
  distributions + JSON round-trip, `LazyRandomState`, `SearchSpaceTransform`, Trial family,
  `BaseStorage`+`InMemoryStorage`, sampler contract + `RandomSampler` + intersection/group search
  spaces, `Study`+optimize/ask/tell, `Optuna.*` module fns. **0.9 gate passed bit-exact.**
- ✅ **1.1–1.5** — `feat(storage): Phase 1 complete` (`5323849`). `JournalStorage`+file backend+locks
  (byte-compatible JSON-lines), `CachedStorage`+`ICachedStorageBackend` seam, heartbeat+retry
  callbacks, artifacts (protocol/FileSystem/backoff/upload-download), copy/delete/summaries.
- ✅ **2.1–2.3** — `feat(pruners): ...` (`0504dab`). `Nop`/`Threshold`/`Patient`/`Median`/`Percentile`
  (Median/Percentile/Nop actually landed in Phase 0 since `Study` defaults to `MedianPruner`),
  `SuccessiveHalving`(ASHA)/`Hyperband` (CRC-32 bracket routing), `Wilcoxon` (self-contained
  signed-rank test, no scipy).
- ✅ **2.4** — `feat(samplers): Phase 2.4 — PartialFixedSampler + GridSampler` (`cbaf90c`).
  `PartialFixedSampler` (delegating wrapper; base sampler's RNG continues across the fix boundary,
  bit-exact). `GridSampler` (`itertools.product` enumeration + seeded index-shuffle bit-identical
  to numpy's list shuffle — `random_interval(i)` == `MT19937.Next(i+1)`; `grid_id`/`search_space`
  system attrs byte-compatible with Python `json.dumps`; self-stops at grid exhaustion). Grid
  values use `CategoricalChoice` as the image of Optuna's `GridValueType`. Full-state parity vs
  live Optuna across seeds/space shapes.
- ✅ **2.5** — `feat(samplers): Phase 2.5 — BruteForceSampler` (`758189f`). Exhaustive define-by-run
  search via a per-suggestion `_TreeNode`/`_UnexpandedTreeNode` tree (four node states), the
  α=0.5 exact-uniform + flat-uniform `sample_child` weight blend, and self-stop at exhaustion
  (looser vs. strict `avoid_premature_stop`). Bit-exact `rng.choice(p=weights)`: NumPy
  pairwise-summation weight normalization + cdf `searchsorted(side='right')` over MT19937 —
  NumSharp's `choice` uses `side='left'`, so the selection is replicated directly. `_is_nan` subset
  filter recognizes NaN/±inf categorical choices as visited (the fix that converges the 4**3
  weird-choice space to 64 trials, not endless re-sampling). `_enumerate_candidates` decimal path
  normalizes `-0.0 → +0.0` (.NET `decimal` signs a zero crossing where Python's `Decimal` does not,
  which also matters for `OrderedDict<double,_>` keys). 20 unit tests (a port of Optuna's own
  `test_brute_force.py`, incl. the hardcoded seed=42 ask/tell sequences) + 9 live full-state parity
  tests across seeds/spaces.
- ✅ **2.6** — `feat(terminator): Phase 2.6 — Terminator framework + non-GP evaluators` (`cc7c6e2`).
  New `OptunaSharp.Terminator` namespace porting `terminator/*` minus the GP improvement evaluators:
  `BaseTerminator`/`Terminator` (deprecation `FutureWarning` + `min_n_trials` gate + the
  `BestValueStagnation → StaticErrorEvaluator(0)` error-evaluator selection), `TerminatorCallback`
  (implicit-convertible to `Action<Study,FrozenTrial>`, calls `study.Stop()`), `BaseErrorEvaluator`/
  `CrossValidationErrorEvaluator` (best-trial cv-score scaled std; `np.var` reproduced with
  sequential two-pass summation — bit-exact for the <128-element cv arrays) / `StaticErrorEvaluator`
  + `TerminatorFunctions.ReportCrossValidationScores`, `BaseImprovementEvaluator`/
  `BestValueStagnationEvaluator`, `MedianErrorEvaluator` (cached median×ratio; `-sys.float_info.min`/
  `sys.float_info.max` sentinels reproduced bit-exactly). `RegretBoundEvaluator` (the default
  improvement evaluator) and `EMMREvaluator` are benign-construction placeholders that throw on
  `Evaluate` until the GP stack (**3.3**) — same pattern as the `TPESampler` placeholder. 19 unit
  tests (a port of Optuna's own `tests/terminator_tests/*`) + 6 live parity tests (decision sequence,
  scaled CV error bit-exact, median threshold) across both directions.
- ✅ **2.7** — `feat(importance): Phase 2.7 — Importance framework + PED-ANOVA` (`3096aba`). New
  `OptunaSharp.Importance` + `OptunaSharp.Importance.PedAnova` namespaces: `BaseImportanceEvaluator`
  + `get_param_importances` (the `PedAnovaImportanceEvaluator` default; normalization; experimental
  `normalize=false` warning), `PedAnovaImportanceEvaluator` (the full `evaluator.py` — `_QuantileFilter`,
  `_get_top_quantile_trials`, Pearson-divergence, regime partitioning, conditional/dynamic-space
  handling), and `ScottParzenEstimator` + grid-count/build helpers. Pulled forward as tested
  byproducts for 2.8: `Internal/TruncNorm.cs` (`_log_gauss_mass` + `_log_ndtr`/`_ndtr`/`_log_diff`)
  and `Samplers/Tpe/ProbabilityDistributions.cs` (`_MixtureOfProductDistribution.LogPdf`, the
  categorical + discrete-truncnorm batched kinds, `_unique_inverse_2d`/`_log_gauss_mass_unique`).
  `Internal/TpeTrialSplit.cs` hosts `_split_complete_trials_multi_objective` (the MO `target=None`
  path); its HSSP tie-break throws pending **2.10** (`_hypervolume/*`) — the exact-rank-partition
  case works, and MO-with-explicit-`target` uses the QuantileFilter path fully. **Numeric strategy
  (all verified against live Optuna before porting):** NumPy `@`/`np.dot` reproduced as a *sequential*
  dot (proven bit-identical end-to-end over 300 mixed studies — the BLAS order doesn't reach the
  answer's bits), `.sum()` reductions via NumPy-pairwise (`MathUtil.PairwiseSum`), `np.cumsum`
  sequential, scalar `x**2` via `Math.Pow` (Python's scalar `**` ≠ `x*x` in ~0.05% of cases; NumPy
  array `**2` == `x*x`), and a new bit-exact `MathUtil.Log1p` (`log(u) - ((u-1)-x)/u`, matches NumPy
  over 220k points — `.NET Math.Log`/`Math.Exp` already equal NumPy's `np.log`/`np.exp` bit-for-bit).
  29 unit tests (a port of Optuna's own `pedanova_tests/*` — grid counts, Scott bandwidth, quantile
  filter, PDFs — plus end-to-end importances built via the bit-exact `RandomSampler`) + 7 live parity
  tests (SO mixed float/log-float/int/log-int/categorical across seeds, `params=` subset, MO with
  explicit target).
  - **[Misaligned] erf/erfc — the one divergence from Python (documented on `TruncNorm`).** The
    numerical Parzen density routes through `_ndtr` → `erf`, and Optuna's `math.erf` is the platform
    C library (**ucrt** on this Windows box), which matches neither FreeBSD `s_erf` (what
    `Internal/Erf.cs` ports), CPython's own `m_erf`, nor scipy — they differ by ~1-3 ULP. So PED-ANOVA
    importances (and anything on the truncnorm kernel, incl. **TPE in 2.8**) agree with Python only to
    a few ULP, **not bit-for-bit** — the same system-libm class the plan already excuses for LAPACK.
    Everything NOT through `erf` stays bit-exact: the categorical importance path, trial selection,
    Scott bandwidth, grid counts, and the descending importance ordering all match Optuna exactly, and
    Optuna's ≥2000-element vectorized-erf approximation IS reproduced bit-for-bit by `Erf.ComputeArray`.
    Unit/interop tests assert a `1e-10`…`1e-13` tolerance on erf-dependent values, exact elsewhere.
- ✅ **2.8** — `feat(samplers): Phase 2.8 — TPESampler (TPE, the default sampler)` (`c1f0b63`). Replaced the
  placeholder with the full `TPESampler`: univariate (independent) AND non-group multivariate paths,
  so a default `TPESampler()` optimizes single-objective studies end-to-end. Completed the numeric
  kernel — `Internal/TruncNorm.cs` sampling half (`_ndtri_exp` Newton iter, `ppf`, vectorized
  `_log_ndtr`, `_norm_logpdf`), new bit-exact `MathUtil.Expm1` (Kahan `(u-1)x/log(u)`) + `LogAddExp`
  (`npy_logaddexp`), `Samplers/Tpe/ProbabilityDistributions.cs` (all four numerical batched kinds,
  `_MixtureOfProductDistribution.sample` incl. the bit-exact `rng.choice(p=)`/`rand`/`uniform` draw
  order, and the continuous branch of `log_pdf`), new `Samplers/Tpe/ParzenEstimator.cs`
  (`_ParzenEstimator` + params, `compute_sigmas`, `_call_weights_func`, categorical/numerical
  distributions), and `Internal/TpeTrialSplit.cs` gained `_split_trials`/complete/pruned/infeasible +
  `default_gamma`/`default_gamma_multiobjective`/`default_weights` (bit-exact `np.linspace`). The
  sampler ports `_sample`/`_compare` (`np.argmax` first-max), `_get_internal_repr`, `_get_params`,
  `constant_liar` (the `tpe:relative_params` system-attr sharing, byte-compatible with Python
  `json.dumps`), the deprecated-arg `FutureWarning`s, and `reseed_rng`. `group=true` throws pending
  **2.9**; MO uses the independent path (the MO-split HSSP tie-break still throws pending **2.10**).
  14 unit tests + 8 live trajectory-parity tests (multivariate continuous/log-float/categorical,
  independent mode, sum-of-squares, bit-exact startup) across seeds.
  - **Numeric strategy (all verified against live Optuna before porting):** `Expm1`/`LogAddExp` are
    **bit-exact** vs NumPy (checked over the truncnorm input range); `_ndtri_exp` is bit-exact for its
    own inputs but the sampling `ppf`/density ride the scalar `erf` → the same **erf/ucrt few-ULP
    tolerance** as 2.7 (the flagship parity is tolerance-bounded, not bit-for-bit, exactly as the plan
    anticipates). RNG draw **count and order** are identical regardless, so the study stays in
    lock-step. Confirmed exact-to-tolerance over full trajectories: multivariate continuous +
    categorical, log-float, and independent mode.
  - **[Misaligned] argsort tie order (the second documented divergence, on `ParzenEstimator`).**
    `compute_sigmas` calls `np.argsort(mus)`; NumPy 2.x sorts float64 with **x86-simd-sort**, whose
    tie order is a SIMD-network permutation that is not portably reproducible (not stable, not classic
    quicksort/heapsort — `kind=` is ignored for float64 — and ISA-dependent). OptunaSharp uses a
    **stable** sort, which is exact whenever the observation values are **distinct** — so every
    all-continuous-float study matches Optuna, as do categorical params and the independent path with
    <25 "above" observations (uniform weights make the 1-D tie order cancel). The residual case —
    **repeated discrete/int observations in a multivariate study** (there each dim's bandwidth pairs
    with a joint kernel, so the tie order does not cancel) — matches Optuna's algorithm but can pick a
    different sigma→kernel pairing and diverge; confirmed by forcing Python's argsort to `kind=stable`,
    which makes an int-param study match bit-for-bit. Documented like the erf one; TPE parity tests
    exclude that path on purpose.
- ✅ Reference vendored — `chore: vendor optuna submodule` (`474db6d`), `src/optuna` @ `f6898da`.
- ✅ Test suites — `test: Phase 0+1 proven` (`cdb3b7b`) + `test(parity): deep parity engine` (`21ff67f`).

### 🟩 Byproducts already landed that later items depend on (don't re-port these)

- **`Internal/Erf.cs`** — full FreeBSD `s_erf.c` `erf`/`erfc` + Optuna's vectorized `_tpe/_erf.py`
  variant + norm cdf/sf. Built for Wilcoxon; it is also the **2.8 TPE** `_erf` prerequisite — done.
  Its ≥2000-element `ComputeArray` path reproduces Optuna's `_erf.py` approximation bit-for-bit; the
  scalar/`<2000` path is FreeBSD `s_erf`, ~1-3 ULP from Python's `math.erf` (ucrt) — see the 2.7
  erf/ucrt `[Misaligned]` note. (`Internal/MathUtil.cs` gained a bit-exact `Log1p` + shared
  `PairwiseSum` in 2.7.)
- **`Internal/TruncNorm.cs`** — the full `_tpe/_truncnorm.py`: `_log_gauss_mass` (`_log_ndtr`/`_ndtr`/
  `_log_diff` + case logic, from 2.7) plus the **2.8 sampling half** (`_ndtri_exp`, `ppf`, vectorized
  `LogNdtr`, `NormLogPdf`). Complete.
- **`Samplers/Tpe/ProbabilityDistributions.cs`** — the full `_tpe/probability_distributions.py`:
  `_MixtureOfProductDistribution.LogPdf` (categorical, discrete, and — **2.8** — continuous branches),
  all five batched kinds (categorical + the **2.8** continuous/discrete trunc-norm/log-norm), the
  **2.8** `Sample` (bit-exact `rng.choice(p=)`/`rand`/`uniform` draw order), and
  `_unique_inverse_2d`/`_log_gauss_mass_unique`. Complete.
- **`Samplers/Tpe/ParzenEstimator.cs`** — **new in 2.8**. `_ParzenEstimator` + `_ParzenEstimatorParameters`,
  `compute_sigmas` (stable argsort — see the 2.8 argsort `[Misaligned]`), categorical/numerical
  distribution builders, `_call_weights_func`, transform/untransform, sample/log_pdf.
- **`Internal/TpeTrialSplit.cs`** — the trial-splitting + defaults from `_tpe/sampler.py`:
  `_split_complete_trials_multi_objective` (from 2.7; HSSP tie-break throws pending **2.10**) plus the
  **2.8** `_split_trials`/`_split_complete_trials_single_objective`/`_split_pruned_trials`/
  `_split_infeasible_trials` and `default_gamma`/`default_gamma_multiobjective`/`default_weights`
  (bit-exact `np.linspace`).
- **`Internal/SignedRankTest.cs`** — scipy-1.16-exact Wilcoxon signed-rank (exact/permutation/
  asymptotic branches). Done for 2.3.
- **`Internal/MultiObjective.cs`** — `study/_multi_objective.py` port: non-dominated sort,
  `_dominates`, `_fast_non_domination_rank`, `_calculate_nondomination_rank`, 2-D/N-D Pareto front,
  `np.unique(axis=0)` semantics. Landed for Phase 0.8 `best_trials`. **This is the first half of
  2.10** — what remains for 2.10 is `_hypervolume/*` (WFG, HSSP, box decomposition).
- **`Samplers/TPESampler.cs`** — **the real sampler as of 2.8** (was a throwing placeholder).
  Univariate + non-group multivariate; `group=true` throws pending **2.9**.
- `trials_dataframe` ships as a dependency-free `TrialsTable` (Core stays pure); the
  `Microsoft.Data.Analysis` adapter remains Tier X (5.4).

### Validation infra — current state vs. §4 target

- ✅ **Live differential interop** (`OptunaSharp.Tests.Interop`) is the working differential gate and
  is now substantial: a NumSharp-style Pythonic harness (`OptunaInteropTestBase`), a full-state
  **`StudySnapshot`** capture/parse/`AssertParity` (bit-exact floats, distribution-JSON equality,
  instant-datetime parity), and a seeded **`ParityScenario`** differential fuzzer (mixed spaces;
  NaN/±inf; unicode/nested attrs; constraints; complete/fail/prune; enqueue+waiting). Includes a
  **raw journal line-format gate** (it already caught one real byte-format bug: categorical
  `param_value_internal` is an int in Python's journal, not a float — fixed). A 2026-09-02
  parity audit added live decision-parity for the **full pruner surface** (Median/Percentile/
  Patient/Threshold/Hyperband/Wilcoxon — previously only Median+SuccessiveHalving were live) and
  for the derived **MO Pareto `best_trials`** set; the audit found parity bit-identical everywhere
  except passing a NaN distribution bound (pathological input → numpy's incidental `OverflowError`
  in Python vs a clean `ValueError` in C#; both raise, not an Optuna typed error).
- ✅ A 2026-09-02 **BruteForceSampler audit** (`f0fdb12`) ran an offline differential fuzz — 205
  studies / ~7k trials over conditional dynamic spaces, big-int (pairwise-sum n>128),
  multi-objective, log-int, deep nesting, mixed float/int/categorical, `avoid_premature_stop`, and
  bool/int/float categoricals — asserting per-trial order/state/values/params (external + internal,
  bit-level IEEE) against live Optuna 5.0.0.dev, plus verbatim parity of all four error messages and
  the `reseed_rng`/`before_trial` structural behaviors. Everything matched except one message-format
  bug (the `search_space mismatch` `choices_new` must render as a Python tuple, not a list), now
  fixed and regression-tested.
- ⬜ **Committed-corpus oracle** (`test/oracle/gen_optuna_oracle.py` + `OptunaSharp.Tests.Oracle`,
  no-Python-at-test-time replay) — **still deferred**, exactly as §4/CLAUDE.md anticipate. The live
  suite covers the gap for now. Build it when the algorithm surface is large enough to snapshot
  (natural point: alongside/after TPE 2.8, the flagship bit-exact claim).
- ⬜ **`MisalignedRegistry`** — not yet a formal ledger; documented divergences currently live as
  in-source `[Misaligned]` comments: journal snapshot bytes are OptunaSharp JSON, not pickle; the
  2.7 **erf/ucrt** ~1-3 ULP tolerance on the truncnorm density, which also bounds 2.8 TPE; and the
  2.8 **argsort tie order** in `compute_sigmas` (stable vs NumPy's unreproducible x86-simd-sort —
  diverges only on repeated numerical observations in a multivariate study; continuous/categorical
  studies are exact). Both TPE divergences are on `Samplers/Tpe/ParzenEstimator.cs`/`TruncNorm.cs`.

### Build / test notes for the next agent

- **Build:** `dotnet build SciSharp.OptunaSharp.sln -c Release -p:SkipNDScopeWeave=true
  -p:GeneratePackageOnBuild=false`. NumSharp resolves as sibling `ProjectReference` → NuGet fallback
  (see main §"NumSharp reference" in CLAUDE.md). A new MSBuild property **`NumSharpCoreProject`**
  (settable via env var of the same name) pins an alternate NumSharp checkout — added because the
  sibling working tree was temporarily mid-edit and wouldn't compile during this work. **If a build
  fails *inside* `NumSharp.Core`, that's the sibling checkout, not OptunaSharp** — point
  `NumSharpCoreProject` at a good NumSharp worktree/commit, or delete the sibling to force the NuGet
  fallback.
- **Test:** `dotnet test test/OptunaSharp.Tests/...` (303 unit, both TFMs) and
  `dotnet test test/OptunaSharp.Tests.Interop/...` (97 live-Optuna tests; self-skip Inconclusive
  without a discoverable Python+optuna). All green on net8.0 + net10.0 as of this ledger.
- **Interop Python is pinned to the vendored reference.** The live-interop suite runs against the
  submodule's Optuna **5.0.0.dev** (`pip install -e src/optuna --no-deps`), not the stale PyPI build.
  This matters from 2.5 on: `_brute_force.py` was rewritten between the last PyPI release (4.6.0) and
  5.0.0.dev (α-blended `sample_child`, the `_is_nan` fix), so an interop diff against 4.6.0 would
  validate the wrong implementation. All 76 tests — including the pre-existing 67 — pass against
  5.0.0.dev with no regression, so the submodule is now the single source of truth for both the port
  and the live gate. (CI installs plain `optuna` from PyPI; align it to the pin, or those specific
  interop cases self-skip when the reference version is absent.)
- **Reuse the parity engine** for every new sampler/pruner: extend `ParityScenario`'s space/op pools
  and assert with `StudySnapshot.AssertParity` — that is the intended "prove it bit-exact" path.

---

## 1. Goals & non-goals

**Goals**
- API-compatible .NET port: an Optuna user reads Optuna's docs and writes the same call in C#.
- Behavior parity proven the NumSharp way: Python Optuna is the **oracle**; committed corpora are
  replayed bit-exact in CI (no Python at test time). NumSharp's byte-identical MT19937 `RandomState`
  makes sampler sequences reproducible bit-for-bit.
- Storage-level interop: journal-file (Tier N) and SQLite `.db` / gRPC (Tier X) studies open in
  Python Optuna and vice versa.
- NumSharp is the numeric backbone; plain C# where arrays add nothing.

**Non-goals (explicit cuts)**
- `optuna.integration.*` (LightGBM/PyTorch/Keras callbacks) — framework adapters, not ports; revisit
  per-framework (ML.NET/TorchSharp) after v1.
- Interactive visualization rendering in Core — the **data layer** (pure-numpy `_get_*_info`
  functions) is Tier N; rendering is a Tier X package.
- OptunaHub, Dashboard, `optuna.testing` (we build our own test infra).

---

## 2. Ground rules

1. **Match Optuna exactly** — run the Python first, observe, replicate; edge cases (NaN, single-value
   distributions, dynamic search spaces, `None` choices) matter. Verbatim error messages where Optuna
   raises typed errors.
2. **RNG discipline:** every sampler consumes `np.random.RandomState` through a `LazyRandomState`
   port. Never `System.Random`. This is what makes differential testing bit-exact.
3. **Categorical value semantics:** `CategoricalChoiceType = None|bool|int|float|str` maps to a
   `CategoricalChoice` value type implementing **Python equality** (`True == 1`, `1.0 == 1` — see
   `distributions.py:509` note). Pinned by tests; the one place C# object equality silently diverges.
4. **Tier N is dependency-pure:** Core references NumSharp.Core and nothing else. Even logging is a
   self-contained seam (port of Optuna's own `logging.py` verbosity API); the
   `Microsoft.Extensions.Logging` adapter is Tier X. `trials_dataframe` returns a plain typed row
   table in Core; the `Microsoft.Data.Analysis` adapter is Tier X.
5. **Package optionality mirrors Optuna's:** anything needing a foreign runtime is a separate NuGet
   package — the NumSharp.Interop.OpenBLAS pattern.
6. **Threading:** trial-level parallelism (`n_jobs`, distributed workers) is supported and tested;
   numeric kernels stay single-threaded (NumSharp house rule).
7. **Every phase ends green:** its oracle tier committed + replayed in CI, unit tests passing, and —
   for storage phases — a live Python↔C# interop script (the `verify_npy_interop.py` model).

---

## 3. Package map

| Tier | Package | Contents | Deps beyond BCL |
|---|---|---|---|
| N | `OptunaSharp` (Core) | distributions, trial, study, ALL samplers (Random/Grid/PartialFixed/BruteForce/TPE/QMC/NSGA-II/NSGA-III/CMA-ES/**GP**), all 9 pruners, transform, MO machinery, PED-ANOVA, terminators (incl. GP-based), InMemory + **journal-file** storage, filesystem artifacts, viz data layer, own managed `eigh`(Jacobi)/`cholesky`/L-BFGS-B | **NumSharp.Core only** |
| X | `OptunaSharp.Storage.Sqlite` | RDB storage, Optuna-compatible schema + alembic version pin | Microsoft.Data.Sqlite |
| X | `OptunaSharp.Storage.EFCore` | same schema on PostgreSQL/MySQL | EF Core + providers |
| X | `OptunaSharp.Storage.Redis` | journal Redis backend | StackExchange.Redis |
| X | `OptunaSharp.Storage.Grpc` | storage proxy client + server | Grpc.Net, Google.Protobuf |
| X | `OptunaSharp.Importance.Forests` | fANOVA, MDI | ML.NET (managed-CART exact port = later Tier N option) |
| X | `OptunaSharp.Gp.Torch` | TorchSharp GP backend — ONLY if the native GP gate fails, or as alt backend | TorchSharp |
| X | `OptunaSharp.Artifacts.S3` / `.Gcs` | cloud artifact stores | AWSSDK / Google.Cloud.Storage |
| X | `OptunaSharp.Visualization.Plotly` | rendering over the Core data layer | Plotly.NET |
| X | `OptunaSharp.Cli` | `optuna` CLI | System.CommandLine, YamlDotNet |
| X | `OptunaSharp.Extensions` | M.E.L logging adapter, Data.Analysis adapter, OpenBLAS byte-exactness switches | various |

---

## 4. Validation strategy (built in Phase 0, extended every phase)

Copy of NumSharp's differential-fuzz pipeline, adapted:

- `test/oracle/gen_optuna_oracle.py` — pinned `optuna==<release>` + pinned clone SHA; generates JSONL
  corpora: `(seed, search_space, observed_trials) → expected (params | pruning decision | importance
  values | storage bytes)` per feature tier.
- `OptunaSharp.Tests.Oracle` — replays corpora with **no Python at test time**; bit-compare where the
  path is deterministic (everything on MT19937), tolerance-compare only where the native tier
  legitimately diverges (managed `eigh`/`cholesky` vs LAPACK — documented per case, upgradeable to
  byte-exact via the Tier X OpenBLAS switch).
- `OptunaSharp.Tests.Interop` (scaffolded) — live gates run by hand/nightly: Python writes study →
  C# continues it → Python reads back (journal file in Tier N; SQLite/gRPC in Tier X).
- `MisalignedRegistry` — the documented-divergence ledger, same rules as NumSharp: excused, never silent.

---

# TIER N — pure managed (BCL + NumSharp only)

## Phase 0 — Core framework — ✅ COMPLETE (`2763190`)

Prerequisite to everything; order within the phase is the dependency order.

| # | Work item | Optuna source | py SLOC | est. C# | Notes |
|---|---|---|--:|--:|---|
| 0.1 | Exceptions, enums, own logging seam | `exceptions.py`, `logging.py`, `study/_study_direction.py`, `trial/_state.py` | ~200 | 450 | `TrialPruned`, `DuplicatedStudyError`…; logging = self-contained port of Optuna's verbosity API (no M.E.L — that adapter is 5.4) |
| 0.2 | **Distributions** + JSON round-trip | `distributions.py` | 381 | 900 | 9 classes incl. deprecated aliases; `to_internal_repr`/`to_external_repr`; `distribution_to_json`/`from_json` (storage depends on it); `CategoricalChoice` Python-equality (§2.3) |
| 0.3 | `LazyRandomState` | `samplers/_lazy_random_state.py` | 24 | 60 | wraps NumSharp `np.random.RandomState` |
| 0.4 | **SearchSpaceTransform** | `_transform.py` | 198 | 450 | float64 bounds matrix, one-hot categorical encode/decode — the numeric boundary of the whole system |
| 0.5 | Trial family | `trial/*` | 853 | 1,900 | suggest_* API, `report`/`should_prune`, attrs, `FrozenTrial`, `create_trial` |
| 0.6 | **BaseStorage contract** + InMemoryStorage | `storages/_base.py`, `_in_memory.py` | 879 | 1,900 | the contract every backend implements; thread-safe in-memory impl; conformance suite born here |
| 0.7 | Sampler contract + RandomSampler + search-space intersection | `samplers/_base.py`, `_random.py`, `search_space/*` | 727 | 1,500 | `infer_relative_search_space`/`sample_relative`/`sample_independent`; `IntersectionSearchSpace` |
| 0.8 | **Study** + optimize loop + ask/tell | `study/study.py`, `_optimize.py`, `_tell.py`, `_dataframe.py` (plain table) | 1,256 | 2,700 | callbacks, `n_jobs` threading, enqueue, best_trial(s), directions |
| 0.9 | **Walking skeleton gate** | — | — | — | quadratic objective E2E: create_study → optimize(RandomSampler) → best_trial **bit-identical to Python** for shared seed; oracle harness v1 lands here |

**Phase 0 ≈ 4.5k py → ~9.9k C#. Estimate: 5–6 weeks.**

## Phase 1 — Native persistence & infra (pure file/memory) — ✅ COMPLETE (`5323849`)

| # | Work item | Optuna source | py SLOC | est. C# | Difficulty | Notes |
|---|---|---|--:|--:|---|---|
| 1.1 | **Journal storage**: file backend, `OpenLock`/`SymlinkLock` → `FileStream` locks, snapshot/replay | `storages/journal/*` minus `_redis.py` | ~650 | 1,400 | 🟢 | JSON-lines op log, byte-compatible with Python's format. **Interop gate (Tier N!):** Python appends → C# replays → C# appends → Python replays, same file |
| 1.2 | Cached storage layer | `storages/_cached_storage.py` | ~300 | 600 | 🟢 | read-through cache (used later by RDB too) |
| 1.3 | Heartbeat + failed-trial retry | `storages/_heartbeat.py`, `_callbacks.py` | ~250 | 550 | 🟢 | `HeartbeatThread`, `RetryFailedTrialCallback`, `fail_stale_trials` |
| 1.4 | Artifacts: protocol + FileSystem + backoff + upload/download | `artifacts/*` minus boto3/GCS | ~250 | 550 | 🟢 | cloud backends are 5.3 |
| 1.5 | Copy/delete/summaries API | `copy_study`, `get_all_study_summaries`, delete | ~200 | 450 | 🟢 | composes on storage contract |

**Phase 1 ≈ 1.65k py → ~3.55k C#. Estimate: 3–4 weeks.**
Exit gate: journal-file Python↔C# live interop green — **cross-language persistence with zero external deps on either side**.

## Phase 2 — Native features, easiest first — 🟩 IN PROGRESS (2.1–2.8 done; **next: 2.9**)

Status column: ✅ done · 🟩 partially landed · ⬜ not started (see §0 ledger).

| # | St | Feature | Optuna source | py SLOC | est. C# | Difficulty | Depends on |
|---|---|---|---|--:|--:|---|---|
| 2.1 | ✅ | Pruners batch 1: `Nop`, `Threshold`, `Patient`, `Median`, `Percentile` | `pruners/*` | ~300 | 700 | 🟢 trivial | 0.5, 0.8 |
| 2.2 | ✅ | Pruners batch 2: `SuccessiveHalving` (ASHA), `Hyperband` | `pruners/_successive_halving.py`, `_hyperband.py` | ~250 | 650 | 🟢 | 2.1 |
| 2.3 | ✅ | `WilcoxonPruner` | `pruners/_wilcoxon.py` | ~100 | 300 | 🟢 | self-contained signed-rank test (`Internal/SignedRankTest.cs`, no scipy) |
| 2.4 | ✅ | `PartialFixedSampler`, `GridSampler` | `samplers/_partial_fixed.py`, `_grid.py` | ~400 | 900 | 🟢 | 0.7 |
| 2.5 | ✅ | `BruteForceSampler` | `samplers/_brute_force.py` | ~480 | 1,000 | 🟢+ | tree enumeration over dynamic spaces; the categorical-NaN quirk (`_is_nan`); bit-exact `rng.choice(p=)` via NumPy pairwise-sum + `searchsorted(side='right')`. `758189f` |
| 2.6 | ✅ | Terminator framework + non-GP evaluators (`StaticError`, `BestValueStagnation`, `MedianError`, `CrossValidationError`) | `terminator/*` minus regret/EMMR | ~300 | 700 | 🟢 | `OptunaSharp.Terminator` namespace; regret/EMMR are GP placeholders (throw until 3.3); `np.var`/`sys.float_info` sentinels bit-exact. `cc7c6e2` |
| 2.7 | ✅ | Importance framework + **PED-ANOVA** | `importance/__init__.py`, `_ped_anova/*` | ~350 | 800 | 🟢+ | pure-numpy Parzen; NumSharp throughout. `_truncnorm._log_gauss_mass` + `probability_distributions.log_pdf` landed as byproducts. **erf/ucrt tolerance** (`[Misaligned]`, see §0) |
| 2.8 | ✅ | **TPE — the default sampler (univariate + non-group multivariate)** | `_tpe/_erf.py`, `_truncnorm.py`, `parzen_estimator.py`, `probability_distributions.py`, `sampler.py` core | ~800 | 2,600 | 🟡 | 0.4. Full numeric kernel + sampler; `Expm1`/`LogAddExp` bit-exact, `ppf`/density erf-tolerance-bounded. Trajectory parity vs live Optuna (continuous/log/categorical multivariate + independent). **Two `[Misaligned]`: erf/ucrt few-ULP, and stable-vs-x86-simd-sort argsort tie order on repeated numerical obs.** `constant_liar`+deprecated args done |
| 2.9 | ⬜ **NEXT** | TPE stage 2 — `group` (group-decomposed multivariate) | `_tpe/sampler.py` group branches | ~150 | 450 | 🟡 | 2.8. `multivariate`/`constant_liar`/`weights` already landed in 2.8; `GroupDecomposedSearchSpace` already ported. `group=true` currently throws |
| 2.10 | 🟩 | **MO machinery**: non-dominated sort, WFG hypervolume, HSSP, box decomposition, Pareto front | `study/_multi_objective.py`, `_hypervolume/*` | ~400 | 1,100 | 🟡 | **`study/_multi_objective.py` half done** (`Internal/MultiObjective.cs`, from 0.8); remaining = `_hypervolume/*` |
| 2.11 | ⬜ | TPE stage 3 — MOTPE + constraints | `_tpe/sampler.py` MO split | ~150 | 450 | 🟡 | 2.9, 2.10 |
| 2.12 | ⬜ | **QMC sampler** (Sobol + Halton, scrambling) | `samplers/_qmc.py` + scipy behavior | ~172 | 1,600 | 🟡 | Sobol direction numbers lifted from `rustuna_sampler/src/qmc/sobol/direction_numbers.rs`; scrambled sequences oracle-pinned vs scipy |
| 2.13 | ⬜ | **NSGA-II** (6 crossovers, polynomial mutation, elite selection, constraints) | `samplers/nsgaii/*`, `_ga/_base.py` | ~1,000 | 2,400 | 🟡 | 2.10; crossover object-array site → `CategoricalChoice[]` |
| 2.14 | ⬜ | NSGA-III (reference points, niching) | `samplers/_nsgaiii/*` | ~390 | 950 | 🟡+ | 2.13 |
| 2.15 | ⬜ | Managed numeric primitives for 2.16/3.2: symmetric `eigh` (cyclic Jacobi), unblocked `cholesky`, `solve_triangular` | — (NumSharp.Core has no managed eigen/cholesky) | — | 500 | 🟡 | small dims (CMA-ES: search-space rank; GP: ≤ trial count) → O(n³) managed is fine. Tolerance oracle vs LAPACK; **byte-exact upgrade switch lives in Tier X (5.5)** |
| 2.16 | ⬜ | **CMA-ES** — port the external `cmaes` package (CMA, sep-CMA, IPOP/BIPOP restarts, margin, lr-adapt) + `CmaEsSampler` | `samplers/_cmaes.py` + `CyberAgentAILab/cmaes` (~1.8k py) | ~2,300 | 4,500 | 🟠 | 2.15. Trajectory oracle is tolerance-based on the native `eigh` (byte-exact under 5.5). Optimizer state → own JSON (pickle non-goal, `[Misaligned]`) |

**Phase 2 ≈ 7.7k py → ~20.1k C#. Estimate: 14–17 weeks.**
Milestone after 2.11: **"Optuna default experience" complete** — TPE + MedianPruner bit-exact with
Python, persisted to a journal file Python can open. Zero external deps.

## Phase 3 — Native heavy tail — ⬜ NOT STARTED

| # | Feature | Optuna source | py SLOC | est. C# | Difficulty | Notes |
|---|---|---|--:|--:|---|---|
| 3.1 | Visualization **data layer** | `visualization/_*.py` info builders | ~1,500 | 2,500 | 🟡 | pure numpy → NumSharp; rendering pkg is 5.2 |
| 3.2 | **GP stack, native path**: `gp.py` (Matérn-5/2 + MLL fit via hand-derived gradients), `acqf.py` (LogEI/qLogEI/LogPI), `optim_mixed`, `batched_lbfgsb` (already pure numpy), `prior`, `search_space` → `GPSampler` | `_gp/*`, `samplers/_gp/sampler.py` | 1,766 | 8,500 | 🔴 | **Decision gate first (§7), prototyped during Phase 2.** Uses 2.15 cholesky/solve_triangular. Port order: search_space → prior → kernel+MLL → L-BFGS-B → acqf → optim_mixed → sampler |
| 3.3 | GP terminators: `RegretBoundEvaluator` (default), `EMMREvaluator` | `terminator/improvement/*` | ~230 | 700 | 🔴 | rides 3.2 |

**Phase 3 ≈ 3.5k py → ~11.7k C#. Estimate: 10–13 weeks (GP is 8–10).**
Exit gate = **Tier N complete**: the full Optuna algorithm surface, pure managed, NumSharp-only.

---

# TIER X — external dependencies & integrations (last)

## Phase 4 — External storage & transport — ⬜ NOT STARTED (Tier X)

| # | Work item | Optuna source | py SLOC | est. C# | Package | Notes |
|---|---|---|--:|--:|---|---|
| 4.1 | **RDB storage on SQLite, Optuna-compatible schema** | `storages/_rdb/*` | 2,339 | 3,000 | `.Storage.Sqlite` | hand SQL on Microsoft.Data.Sqlite mirroring the SQLAlchemy models + `alembic_version` pin so Python opens our `.db`. Reference: `rustuna_storage/sqlite3.rs`. **Interop gate:** Python↔C# same-file round-trip. Reuses 1.2 cache + 0.6 conformance suite |
| 4.2 | EF Core providers (PostgreSQL/MySQL) | (schema from 4.1) | — | 1,500 | `.Storage.EFCore` | conformance suite + testcontainers |
| 4.3 | Redis journal backend | `storages/journal/_redis.py` | ~200 | 450 | `.Storage.Redis` | thin: 1.1 already carries the journal semantics |
| 4.4 | gRPC storage proxy (client + server) | `storages/_grpc/*` | 1,719 | 2,400 | `.Storage.Grpc` | reuse Optuna's `.proto` verbatim; wire-compat gate: C# client ↔ Python `run_grpc_proxy_server` |
| 4.5 | Cloud artifacts (S3, GCS) | `artifacts/_boto3.py`, `_gcs.py` | ~100 | 300 | `.Artifacts.*` | protocol landed in 1.4 |

**Phase 4 ≈ 4.4k py → ~7.65k C#. Estimate: 5–6 weeks.**

## Phase 5 — External integrations & upgrades — ⬜ NOT STARTED (Tier X)

| # | Work item | py SLOC | est. C# | Package | Notes |
|---|---|--:|--:|---|---|
| 5.1 | **fANOVA + MDI importance** via ML.NET FastForest | 930 | 4,000 | `.Importance.Forests` | tolerance oracle (forests not tree-identical to sklearn), divergence documented; exact managed-CART port stays on the backlog as a Tier N follow-up |
| 5.2 | Visualization rendering | — | 1,500 | `.Visualization.Plotly` | over the 3.1 data layer |
| 5.3 | CLI | 806 | 1,300 | `.Cli` | System.CommandLine + YamlDotNet |
| 5.4 | Adapters: M.E.L logging, `Microsoft.Data.Analysis` `trials_dataframe` | — | 400 | `.Extensions` | thin bridges over Core seams |
| 5.5 | **Byte-exactness upgrades via `NumSharp.Interop.OpenBLAS`**: CMA-ES `eigh` → LAPACK `syevd`, GP `cholesky` → `potrf` | — | 200 | `.Extensions` | flips the 2.15 primitives to the backend seam → CMA-ES/GP oracles upgrade from tolerance to byte-exact (NumSharp's linalg_parity model) |
| 5.6 | `OptunaSharp.Gp.Torch` (contingency) | — | (2,000) | `.Gp.Torch` | ONLY if the §7 native gate failed — or as an alternative backend for large-trial GP perf |
| 5.7 | Framework pruning-callback adapters (ML.NET, TorchSharp) | — | (backlog) | — | the `optuna.integration` analog; per-framework, post-v1 |

**Phase 5 ≈ 1.7k py → ~7.4k C#. Estimate: 6–8 weeks.**

---

## 7. GP autodiff decision gate (opens 3.2; prototyped during Phase 2)

The only genuinely hard porting problem. Preference order **reflects the native-first directive**:

1. **Hand-derived analytic gradients** (pure managed — keeps GP in Tier N): Matérn-5/2
   marginal-likelihood gradients are closed-form; Optuna's own `Matern52Kernel(torch.autograd.Function)`
   already writes the backward by hand — evidence the surface is small. Optimizer = ported
   `batched_lbfgsb` (already numpy). Risk: derivation errors → finite-difference checks in the oracle.
2. **TorchSharp** (`OptunaSharp.Gp.Torch`, Tier X): fallback if (1) stalls; isolated to the optional
   package, ~600 MB native dep.
3. Micro reverse-AD library: only if (1) proves error-prone AND (2) unacceptable.

Prototype task (1 week, during Phase 2): fit a 1-D GP with hand gradients + ported L-BFGS-B; compare
log-marginal-likelihood trajectory vs Python. Green → GP stays native; red → 5.6.

---

## 8. Cross-cutting workstreams

- **Oracle corpus growth** — every feature lands with its tier; nightly soak sweeps seeds.
- **Docs** — DocFX site (scaffolded); per-API pages mirror Optuna's doc structure.
- **Benchmarks** — `OptunaSharp.Benchmark` (scaffolded): suggest-throughput, storage ops/s,
  sampler wall-clock vs Python Optuna AND vs Rustuna.
- **NumSharp feedback loop** — gaps found get fixed upstream (done already: `logaddexp`, `nextafter`,
  `interp`; watch the `erf` family in 2.8). Managed `eigh`/`cholesky` (2.15) live in OptunaSharp
  first; graduating them into NumSharp.Core as managed linalg fallbacks is a candidate upstream
  contribution.

---

## 9. Timeline & totals

| Phase | Tier | Scope | est. C# SLOC | Duration |
|---|---|---|--:|---|
| 0 | N | Core framework | ~9.9k | 5–6 wk |
| 1 | N | Native persistence (journal/heartbeat/artifacts) | ~3.6k | 3–4 wk |
| 2 | N | Native features easy→hard (pruners → TPE → … → CMA-ES) | ~20.1k | 14–17 wk |
| 3 | N | Native heavy tail (viz data, GP) | ~11.7k | 10–13 wk |
| — | — | **Tier N complete — "pure managed full Optuna"** | **~45.3k** | **~7.5–9 mo** |
| 4 | X | External storage (SQLite/EFCore/Redis/gRPC/cloud) | ~7.7k | 5–6 wk |
| 5 | X | External integrations (forests, rendering, CLI, adapters, exactness) | ~7.4k | 6–8 wk |
| **Total** | | | **~60.4k** | **~10–12.5 mo** |

Tier X has no dependency on Tier N's *late* items (4.1 needs only Phase 0/1 contracts) — with a second
contributor, Phase 4 can run concurrently with Phases 2–3 and the total compresses toward ~9 months.

Checkpoints: **M1** (end P0) walking skeleton bit-exact · **M2** (end P1) **cross-language journal
persistence, zero external deps** · **M3** (after 2.11) default-experience parity — TPE + pruning,
the announceable release · **M4** (end P3) Tier N complete: full algorithm surface pure-managed ·
**M5** (end P4) Python-interoperable SQLite/gRPC · **M6** (end P5) near-full Optuna.

---

## 10. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| GP gradients wrong/slow | 3.2 slips | decision-gate prototype in Phase 2; TorchSharp fallback (5.6) |
| Managed `eigh`/`cholesky` drift vs LAPACK | CMA-ES/GP oracles only tolerance-exact in Tier N | documented per-case; 5.5 OpenBLAS switch restores byte-exactness; small-dim Jacobi is deterministic |
| fANOVA forest not tree-identical to sklearn | importance values differ | ML.NET + tolerance oracle + documented divergence; exact CART port as Tier N follow-up |
| Journal-file format drift across Optuna versions | Tier N interop breaks on their upgrades | format pinned to Optuna release; interop gate runs against pinned + latest |
| SQLAlchemy schema drift | Tier X interop breaks | schema pinned to alembic revision; same dual-version gate |
| Python-equality of `CategoricalChoice` edge cases | silent wrong index → wrong param | dedicated property tests (`True/1`, `1.0/1`, `nan`, mixed lists) from 0.2 |
| Pickle-serialized sampler state (CMA-ES attrs) | cross-language studies can't resume mid-CMA | own JSON state, `[Misaligned]`, migration notes |
| `frompyfunc`/object-array sites | none — audited: 1 site (NSGA-II crossover) + 1 intermediate | plain C# arrays (dtype audit, 2026-08-29) |
