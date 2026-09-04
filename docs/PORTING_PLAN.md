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
PED-ANOVA, terminators, in-memory + journal-file persistence**. Cross-language persistence is ALREADY
in this tier: Optuna's journal-file format is JSON-lines over plain file IO, so Python Optuna and
OptunaSharp can share a live journal file with no external package on either side.

> **⚠️ Dependency-model update (2026-09-03, maintainer decision — supersedes the original "pure
> managed, zero deps beyond NumSharp" framing).** Core now also references **`NumSharp.Interop.OpenBLAS`**
> as a first-class dependency: *"use NumSharp + OpenBLAS any time Optuna uses NumPy."* That package
> ships the same scipy-openblas 0.3.31 binaries NumPy 2.4.2 bundles, so NumSharp's `np.linalg`
> (cholesky/eigh/solve/matrix_rank/svd/qr) runs through LAPACK and is **byte-identical to NumPy** —
> the linalg-backed samplers (NSGA-III's hyperplane solve, CMA-ES's `eigh`, GP's Cholesky) are
> byte-exact rather than managed-tolerance, and the plan's separate "5.5 OpenBLAS byte-exactness
> upgrade switch" is effectively pulled forward into Core. Consequences: Core carries per-RID native
> binaries (no longer zero-native-deps); `2.15` is now "wire OpenBLAS + thin `LinAlg` adapters," not
> "hand-rolled managed eigh/cholesky." **Not yet on nuget.org** — the in-repo sibling ProjectReference
> is the working path; the `PackageReference` fallback (and CI/pack) needs the package published.

---

## 0. Progress ledger — where we are, where to continue

> **Status as of 2026-09-04.** Legend: ✅ done & green · 🟩 partially landed (some sub-parts done as
> byproducts) · ⬜ not started. Update this section as items land — it is the first thing the next
> agent should read. Per-item detail is in the phase tables below (the same `#` ids).

### ▶ CONTINUE HERE: **Phase 4 — external storage (Tier X)** — this begins the Tier X external-dependency work (RDB/Redis/gRPC storage, etc.); the `ICachedStorageBackend` seam for 4.1 was declared back in 1.2. **Phase 4.5 (cloud artifacts) is already DONE** — S3/GCS ports + Azure Blob / Cloudflare R2 extensions landed as 4 satellite packages (feat `3923fcd`); the remaining Phase 4 work is 4.1 SQLite RDB, 4.2 EF Core, 4.3 Redis journal, 4.4 gRPC proxy. **TIER N IS COMPLETE — the entire native Optuna algorithm surface is ported (every sampler incl. TPE/NSGA/CMA-ES/GP, every pruner, MO machinery, PED-ANOVA, terminators incl. the GP evaluators, the visualization data layer, in-memory + journal-file persistence), pure managed on NumSharp(+OpenBLAS).** The 2.12 `QMCSampler` `scramble=true` wire-up is **DONE for Sobol'** (the default engine; full-state bit-identical vs live Optuna — see the 2.12 done-item). The one remaining native sliver is **scrambled Halton** (`qmc_type="halton", scramble=true`, throws with a clear message): it needs numpy's PCG64 `shuffle`/`random_interval` (scipy's `_van_der_corput_permutations`), which NumSharp does not yet expose publicly → a small NumSharp upstream primitive is the prerequisite.

> **Phase 3.3 is COMPLETE (2026-09-04).** The two GP improvement evaluators are ported and un-throw:
> `RegretBoundEvaluator` (the default, `terminator/improvement/evaluator.py` — `_get_beta`,
> `_compute_standardized_regret_bound`, `_get_top_n`) and `EMMREvaluator` (`emmr.py` — the EMMR
> Theorem-1/Eq-4 criterion). New `UCB`/`LCB` acqfs + `Gp/OptimSample.cs` (`optimize_acqf_sample`) are
> the only new GP primitives they needed; everything else rides the Phase 3.2 GP stack (`FitKernelParams`,
> `Posterior`/`PosteriorJoint`, `GpSearchSpace`). The deferred viz builder `_get_improvement_info` is
> wired as `Visualization/TerminatorImprovement.cs` (the plotly `_get_improvement_plot` stays Tier X 5.2).
> **Tolerance parity vs live Optuna** (the documented GP boundary — the fit's ~1e-3 hyperparameter drift
> propagates through the posterior; RegretBound/EMMR criteria agree to a few 1e-3 relative, asserted at
> 3e-2). Commit `28e517f`. Suite: **544 unit + 259 interop**, both TFMs, none skipped.

> **The whole `GPSampler` is ported and validated vs live Optuna/torch/scipy:** the fit (parts 1–2),
> transforms + posterior (3a), LogEI (3b), scrambled Sobol' preliminary sampling (3c), the mixed acqf
> optimizer + `GPSampler` (3d), **part 3e sub-parts 1–5** (the sequential MO/constrained acqfs), and
> **part 3e-6** (the running-trial q-batch acqfs):
> - **3e-1** — `box_decomposition.py: get_non_dominated_box_bounds` (`Internal/Hypervolume.cs`, the 2.10
>   remainder; bit-exact vs Optuna).
> - **3e-2** — `LogPI` + `LogCEI` (`Gp/Acqf.cs`) → the **SO constrained** branch. No QMC.
> - **3e-3** — `qmc.sample_from_normal_sobol` (`Gp/Qmc.cs` + `Internal/TorchMt19937.cs` + cephes
>   `TruncNorm.Ndtri`): **torch's** scrambled Sobol' (classic MT19937 + unit-diagonal LMS on the
>   scipy-identical unscrambled direction numbers), matched to torch to ~1e-15. Note the row-0 float32
>   rounding quirk. This is the QMC base for all MO/q-batch acqfs.
> - **3e-4** — `LogEHVI` (`Gp/Acqf.cs`) → the **MO** branch (rides 3e-1 + 3e-3; hand-derived logsumexp/
>   box-clamp gradient).
> - **3e-5** — `LogCEHVI` (`Gp/Acqf.cs`) → the **MO constrained** branch (= LogEHVI over feasible + Σ LogPI).
> - **3e-6** — the **running-trial** branches (`ConditionalGPRegressor` + `qLogEI`/`qLogPI`/`qLogCEI`/
>   `qLogEHVI` + the Kriging-Believer `append_running_data` path). Committed in four steps `741b793`
>   (3e-6a: `append_running_data`/`_extend_cholesky`/`PosteriorJoint` + KB `LogEHVI`/`LogPI`/`LogCEHVI`
>   + MO-constrained running), `a75b14e` (3e-6b: `ConditionalGPRegressor` + `qLogEI`), `c33d7b3` (3e-6c:
>   `qLogPI` + `qLogCEI`), `51ec724` (3e-6d: `qLogEHVI`). The `sample_joint_posterior` gradient +
>   `amax`/`logsumexp` reduction gradients are hand-derived (the hardest gradient in the GP stack).
>
> Each sub-part: eval **and** hand-derived gradient parity vs live Optuna autograd (fixed-hyperparameter
> acqf sweeps) + a per-suggestion **replay** parity of the sampler branch (for 3e-6, an ask/tell schedule
> that leaves trials RUNNING before the trial under test, RNG in lock-step). Tolerance, not bit-exact —
> the documented GP boundary. Suite after 3e-6: **533 unit + 252 interop**, both TFMs, none skipped.
>
> **Still open (small, independent — the last native sliver):** the 2.12 `QMCSampler`
> `scramble=true` is **wired for Sobol'** (the default; bit-identical vs live Optuna). Only scrambled
> **Halton** remains (throws), pending a NumSharp PCG64 `shuffle` primitive.

Phases 0, 1, **2.1–2.16 and 3.1–3.3 are complete** — **Phase 2 (all native features) is DONE** and
**Phase 3 (native heavy tail) is DONE, so TIER N is COMPLETE**. The whole TPE surface, pruners, PED-ANOVA, terminators, the MO hypervolume/HSSP
machinery, the non-scrambled QMC sampler, the full NSGA-II **and** NSGA-III genetic-algorithm samplers,
the OpenBLAS-backed linalg primitives, and the complete `CmaEsSampler` over **all four** cmaes backends —
full-covariance `cmaes.CMA`, separable `cmaes.SepCMA`, Warm-Starting via `get_warm_start_mgd`, and CMAwM
(`with_margin`) — are committed and proven green. **Phase 3.1 (the visualization data layer) is now
COMPLETE** — all 13 `_get_*_info` builders ported and **bit-exact vs live Optuna** (**465 unit + 205
live-interop tests**, both TFMs, none skipped): `_utils` helpers + optimization-history/EDF/
intermediate-values (part 1) + hypervolume-history + param-importances (part 2) + slice + pareto-front
(part 3) + contour (part 4) + rank (part 5) + parallel_coordinate (part 6) + timeline (part 7). The new
`OptunaSharp.Visualization` namespace exposes public info records the Tier X rendering package (**5.2**) will
consume across the assembly boundary. Only three deliberate boundaries/divergences: rank's RGB `colors`
(plotly's `sample_colorscale`; exposed as the pure `ColorIdxs` + `Infeasible` mask, RGB → 5.2); the MO
parallel_coordinate `draw_order` (`np.argsort` over tied Pareto ranks — the unstable-quicksort `[Misaligned]`,
same class as 2.8); and `_terminator_improvement`, which rides the GP improvement evaluators → deferred to
**3.3**. New shared primitives: `MathUtil.Linspace`/`Mean`/`Std`/`FormatG3` (all byte-exact vs numpy/CPython)
and `PyJson.DumpsIndented` (byte-exact `json.dumps(indent=2)`, used by the timeline hovertext). **Phase 3.2
is now in progress — the §7 autodiff decision gate is GREEN and CLOSED.** The GP kernel-parameter fit is
**fully ported and validated end-to-end vs live Optuna** — the hand-derived Matérn-5/2 marginal-likelihood
gradient (part 1) plus a from-scratch managed **L-BFGS-B** (part 2, the port of scipy's `__lbfgsb.c`) — so
`GPRegressor.FitKernelParams` reproduces Optuna's `fit_kernel_params` to **~1e-14** on the canonical case and
to a loose-`gtol` tolerance across a random sweep. **GP stays native (Tier N); TorchSharp 5.6 is NOT needed.**
⚠️ **Plan correction (settled):** `batched_lbfgsb.py` is NOT "already numpy" as §7's parenthetical assumed —
it is a greenlet *wrapper* around `scipy.optimize.fmin_l_bfgs_b`, and `_fit_kernel_params` itself calls
`scipy.optimize.minimize(method="l-bfgs-b")`; the managed L-BFGS-B port (`Internal/LBfgsB.cs`) is what keeps
GP native. **Phase 3.2 parts 3a–3d are done — the single-objective `GPSampler` runs end-to-end** (LogEI acqf,
the `optim_mixed.py` mixed optimizer, and the sampler wiring; **next is 3e**, the MO/constrained/q-batch acqf,
then 3.3 GP terminators). The GP
forward/fit stays **tolerance-exact, not bit-for-bit** (managed `SolveTriangular` + hand-written L-BFGS-B
BLAS/LAPACK drift from scipy's by ULPs — the documented GP boundary). The plotly rendering
`plot_*`/`_get_*_plot` stays Tier X **5.2**.
- ✅ **3.3 — GP terminators (`RegretBoundEvaluator` default + `EMMREvaluator`) → Tier N COMPLETE.**
  Un-throws the two Phase 2.6 GP placeholders in `Terminator/ImprovementEvaluator.cs` +
  `Terminator/EmmrEvaluator.cs`. **`RegretBoundEvaluator`** (`terminator/improvement/evaluator.py`):
  `_get_top_n` (`np.clip(int(n·ratio), min_n, n)` + the `np.partition` order-statistic threshold + the
  `values ≥ thr` mask), the standardize `(v−mean)/max(1e-10, std)`, `fit_kernel_params` on the top-n,
  then `_compute_standardized_regret_bound` = `max(UCB) − max(LCB)` where max(UCB) also takes the
  random-search `optimize_acqf_sample` sample, times the top-n std. **`EMMREvaluator`** (`emmr.py`): the
  insufficient-trials / empty-space `float_info.max·0.1` sentinels, the 1-D `warn_and_convert_inf`, two
  fits (t−1 with `gpr_cache=None`, t with `gpr_cache=gpr_t1`), the Theorem-1 Δμ*/v/g terms
  (`norm.pdf`/`norm.cdf`), the Eq-4 KL terms (`λ = DEFAULT_MINIMUM_NOISE_VAR⁻¹`), and the
  `min(float_info.max·0.5, Σ)` result. New GP primitives they needed: **`UCB`/`LCB`** acqfs (`Gp/Acqf.cs`,
  `mean ± √(β·var)`) + **`Gp/OptimSample.cs`** (`optimize_acqf_sample` — scrambled-Sobol' sample →
  `argmax`); everything else rides the 3.2 stack (`FitKernelParams`, `Posterior`/`PosteriorJoint`,
  `GpSearchSpace`, `Prior.DefaultMinimumNoiseVar`). **Wired the deferred viz builder** as
  `Visualization/TerminatorImprovement.cs` (`_get_improvement_info`: the MO guard, default
  RegretBound/CrossValidation—or—StaticError(0)-for-BestValueStagnation evaluator selection, the
  completed-prefix walk) — the plotly `_get_improvement_plot`/`_get_improvement_scatter` stay Tier X
  **5.2**. RNG discipline: the evaluators' `LazyRandomState` seed draws the Sobol' seed identically to
  Python, so the sampled-point sequence is in lock-step. **Tolerance parity vs live Optuna** (the
  documented GP boundary: managed L-BFGS-B + triangular solve drift the fitted hyperparameters ~1e-3,
  which propagate through the posterior; RegretBound/EMMR criteria agree to a few 1e-3 relative,
  asserted at 3e-2 with ~13x headroom). 11 unit (`GpTerminatorTests` ×7 — a port of Optuna's own
  `test_evaluator.py`/`test_emmr_evaluator.py` GP cases; `TerminatorImprovementTests` ×4 — the viz
  builder control flow) + 7 interop (`GpTerminatorParityTests`: RegretBound ×3 + EMMR ×3 sweeps +
  the full `_get_improvement_info` walk vs `optuna.visualization._terminator_improvement`). Suite:
  **544 unit + 259 interop**, both TFMs, none skipped.
- ✅ **3.2 part 1 — GP numeric core + the §7 autodiff decision gate (`OptunaSharp.Gp`).** New
  `src/OptunaSharp.Core/Gp/` namespace porting the forward + hand-derived backward of `optuna/_gp/gp.py`'s
  `GPRegressor`: `Prior.cs` (`default_log_prior` — the lengthscale hand-crafted term + `Gamma(2,1)` scale /
  `Gamma(1.1,30)` noise priors, constants omitted; `DEFAULT_MINIMUM_NOISE_VAR`) and `GPRegressor.cs`
  (`Kernel` — Matérn-5/2 ARD with categorical Hamming distance; `MarginalLogLikelihood` —
  `-½logdet(C) - ½yᵀC⁻¹y` via `LinAlg.Cholesky` + `LinAlg.SolveTriangular`; and **`LossAndGrad`** — the
  analytic gradient of `-MLL - log_prior` w.r.t. the raw log-params that replaces PyTorch autograd). **This
  resolves the plan's single genuinely-hard porting problem: the closed-form Matérn-5/2 gradient (∂MLL/∂w =
  s·Σ(G⊙M')⊙D, ∂MLL/∂s = ΣG⊙M(S), ∂MLL/∂ν = ½(α·α − tr C⁻¹), G = ½(ααᵀ − C⁻¹)), computed in pure managed
  code with `LinAlg.Dot`/`SolveTriangular` + `MathUtil.PairwiseSum`, matches live PyTorch autograd's loss AND
  full gradient to ~1e-13** — far tighter than needed; a finite-difference self-check confirms correctness
  independently. Tolerance (not bit-exact) by design: the analytic backward is a different float-op sequence
  than autograd's reverse pass and the forward rides the managed triangular solve — the documented GP
  boundary. 7 unit tests (kernel numerical + categorical-Hamming vs Optuna, MLL forward on 3 fixed cases,
  prior value, finite-difference gradient check, deterministic noise-grad = 0, reproducibility) + 3 live
  parity sweeps (`GpFitParityTests`: numerical / categorical / deterministic — 60 random (X,Y,raw) configs,
  loss+gradient vs Optuna's own `_fit_kernel_params.loss_func`, magnitude-relative tol 1e-9). Suite:
  **472 unit + 208 interop**, both TFMs, none skipped.
- ✅ **3.2 part 2 — managed L-BFGS-B + `fit_kernel_params` (closes the §7 gate).** New
  `src/OptunaSharp.Core/Internal/LBfgsB.cs` — a faithful port of scipy 1.16.3's `scipy/optimize/__lbfgsb.c`
  (L-BFGS-B v3.0 by Zhu/Byrd/Lu/Nocedal/Morales, ~3.8k lines of C). Structure preserved for diffing: the
  compact-BFGS matrices keep the column-major flat layout (`a[i + lda*j]`) and 0-based indexing; the
  reverse-communication driver (`Mainlb`) keeps the original `goto` control flow and the `isave`/`dsave`/
  `lsave` persistent-state arrays; C pointer arithmetic (`&wa[2*m]`) → `Span<T>` slices; every subroutine
  ported (`active`/`bmv`/`cauchy`/`cmprlb`/`errclb`/`formk`/`formt`/`freev`/`hpsolb`/`lnsrlb`/`matupd`/
  `projgr`/`subsm`/`dcsrch`/`dcstep`); the small dense kernels (`ddot`/`dnrm2`/`daxpy`/`dscal`/`dcopy` +
  the `dpotf2`/`dtrsm` behind `dpotrf`/`dtrtrs`) are hand-written sequential routines (dims ≤ m = 10).
  Then `GPRegressor.FitKernelParams` + `FitInner` + `CacheMatrix` port `gp.fit_kernel_params` /
  `_fit_kernel_params` / `_cache_matrix` — build the initial raw params, drive `LBfgsB.Minimize` with
  Optuna's exact options (`gtol`→`pgtol`, `factr=1e7`, `m=10`), the both-caches robustness retry, and cache
  `C = LLᵀ` + `C⁻¹y` for the posterior. **Validated vs scipy AND live Optuna:** on identical pure test
  functions (quadratic/Rosenbrock/bounded) `LBfgsB` reproduces scipy's **iteration and evaluation counts
  exactly**, the bounded run **bit-for-bit**, the others bit-identical for the first several iterations then
  ULP-drift to the same optimum (tolerance-faithful — managed BLAS/LAPACK vs scipy's OpenBLAS); and
  `FitKernelParams` matches Optuna's `fit_kernel_params` to **~1e-14** on the canonical 1-D case, with the
  achieved **loss** agreeing to <1e-4 across a 20-config random sweep (the hyperparameters settle ~1e-3 apart
  where Optuna's loose `gtol=1e-2` leaves the argmin in a flat region — the objective is the well-defined
  "same optimum" metric). 4 unit (`LBfgsBTests`: quadratic/Rosenbrock/bounded convergence, determinism) +
  4 interop (`LBfgsBParityTests` quadratic/Rosenbrock/bounded vs scipy nit/nfev/x/f; `GpFitParityTests`
  fit-result sweep vs live Optuna). Suite: **476 unit + 212 interop**, both TFMs, none skipped.
- ✅ **3.2 part 3a — GP search-space transforms + posterior (the acqf foundations).** Two pieces the
  acquisition functions consume. `Gp/SearchSpace.cs` — `GpSearchSpace` (port of `_gp/search_space.py`'s
  `SearchSpace`; **renamed to avoid the `OptunaSharp.SearchSpace` namespace collision**, the `SlicePlot`
  precedent): scale-type/bounds/steps construction, `NormalizeOneParam`/`UnnormalizeOneParam`/
  `RoundOneNormalizedParam` (affine + log/exp + floor-to-grid), `GetChoicesOfDiscreteParams`,
  `GetNormalizedParams`, `GetUnnormalizedParam`. **Bit-exact vs Optuna** (pure numpy — no erf/LAPACK;
  `log`/`exp`/`floor` match Python bit-for-bit): scale-types/steps/indices, all discrete choice grids,
  and the one-param transforms over a 60-config sweep (`Should().Be(0.0)`), plus the boxed int/log-float/
  categorical `get_unnormalized_param`. `SampleNormalizedParams` is a documented `NotImplementedException`
  (needs the 2.12 scrambled Sobol'). `GPRegressor.Kernel(X1,X2)` + `Posterior(x)` — the two-argument cross
  kernel and `posterior(x, joint=False)` (`mean = k·C⁻¹y`, `var = kernel_scale − k·(C⁻¹kᵀ)` clamped ≥ 0),
  reusing `CacheMatrix`'s `L`/`C⁻¹y`. **Tolerance-exact vs Optuna** (~1e-14; managed `SolveTriangular` +
  sequential dots vs torch): posterior mean/var over a 20-config fixed-hyperparameter sweep (isolating the
  posterior math from fit drift). 8 unit (`SearchSpaceTests` ×6, `GPRegressorTests.Posterior`, +) + 2
  interop (`GpSearchSpaceParityTests` one-param sweep; `GpFitParityTests.Posterior` sweep). Suite:
  **483 unit + 214 interop**, both TFMs, none skipped.
- ✅ **3.2 part 3b — LogEI acquisition function (eval + hand-derived gradient).** `Gp/Acqf.cs` — the
  log-expected-improvement acqf `GPSampler` maximizes for single-objective optimization (port of
  `acqf.py`'s `standard_logei`/`logei`/`LogEI`). `Acqf.StandardLogEi` (the stable log-EI: primary
  `log(z·Φ(z)+φ(z))` branch + the `z<-25` `erfcx` branch, Eq. 9 of arxiv 2310.20708) + `LogEi`; `LogEI`
  with `EvalAcqfNoGrad` (batched, for the preliminary samples + discrete search) and `EvalAcqfWithGrad`
  (value + the **hand-derived gradient** that replaces autograd — chains `standard_logei'` through the new
  `GPRegressor.PosteriorWithGrad`, whose `∂mean/∂x`, `∂var/∂x` are analytic, with the categorical Hamming
  dims and the `clamp_min` variance-gradient handled as autograd does). New `Erf.Erfcx` (scaled
  complementary error function: direct `exp(x²)·erfc(x)` for `|x|≤25`, asymptotic series beyond) + a
  `GPRegressor.LengthScales` property. **Tolerance-exact vs live Optuna** (LogEI rides `erfc`/`erfcx` → the
  erf/ucrt class): `standard_logei`/`logei` over a 40-config both-branch sweep (rel <1e-9), and LogEI
  `eval` (rel <1e-8) + `gradient` (rel <1e-6) over a 15-config fixed-hyperparameter sweep — the gradient
  asserted only where Optuna is finite, since **Optuna's own autograd `nan`s at pathological points**
  (`z ≲ −35`, acqf ≈ −10³, never roulette-selected) where OptunaSharp's analytic gradient stays finite and
  correct (finite-difference-checked). 6 unit (`AcqfTests`: erfcx, standard_logei, logei, eval vs Optuna,
  finite-difference gradient, −∞-threshold = 0) + 2 interop (`GpAcqfParityTests`). Suite: **489 unit +
  216 interop**, both TFMs, none skipped.
- ✅ **3.2 part 3c — scrambled Sobol' preliminary sampling (closes the Phase 2.12 gap).** New
  `Internal/ScrambledSobol.cs` — a **bit-exact** port of `scipy.stats.qmc.Sobol(scramble=True)`
  (`scipy/stats/_sobol.pyx` + `_qmc.py`): the LMS (`_cscramble`) + digital shift seeded by
  `numpy.random.default_rng(seed)`, then the Gray-code `draw`. **Key enabler discovered:** NumSharp's
  `np.random.default_rng`/`Generator.integers`/`random` reproduce NumPy's **PCG64 stream byte-for-byte**
  (verified) — the 2.12 "not reproducible without scipy's Cython" concern is resolved. Reuses
  `QuasiMonteCarlo.SobolSv` (2.12's `_initialize_v`, already bit-exact). Wired
  `GpSearchSpace.SampleNormalizedParams` (draw a Sobol' seed from the sampler's MT19937 via
  `randint(int32max)`, scrambled box-fill, then categorical-floor / discrete-round per dim). **Bit-exact
  vs scipy AND Optuna:** the 6-point d=2/seed=42 scramble matches scipy to the last bit; a 25-config
  random `d×n×seed` sweep is byte-identical (`Should().Be(0.0)`); and `SampleNormalizedParams` over a
  mixed float/log-float/int/discretized-float/categorical space matches Optuna's own
  `_sample_normalized_params` (seed draw + Sobol' + rounding) byte-for-byte across 5 seeds. **Also
  unblocks the 2.12 `QMCSampler` `scramble=true`** (a follow-up). 3 unit (`ScrambledSobolTests` ×2,
  `SearchSpaceTests.SampleNormalizedParams`) + 2 interop (`GpSobolParityTests`: scramble sweep +
  SampleNormalizedParams). Suite: **492 unit + 218 interop**, both TFMs, none skipped.
- ✅ **3.2 part 3d — mixed acqf optimizer + `GPSampler` (single-objective path). GP-based optimization
  runs end-to-end.** Three new files. `Gp/OptimMixed.cs` ports `optim_mixed.py`: `OptimizeAcqfMixed`
  (scrambled-Sobol' preliminary draw → `argmax` + the roulette `rng.choice(replace=False, p=)` start
  points → `LocalSearchMixedBatched`), `GradientAscentBatched` (the lengthscale-preconditioned L-BFGS-B
  gradient ascent over the continuous dims — `z = x/l`, bounds `(0, 1/l)`, `pgtol=√tol`, `maxiter=200` —
  driving the managed `LBfgsB` per row), and the discrete `ExhaustiveSearch`/`DiscreteLineSearch` +
  the `LocalSearchMixedBatched` convergence bookkeeping (the `remaining_inds`/`last_changed_dims` masks,
  index-list-faithful). `Internal/BrentMinimizer.cs` ports scipy's `minimize_scalar(method="brent")`
  (the 3-point-bracket `Brent` class — `_mintol`/`_cg`, parabolic-vs-golden step) for the discrete line
  search. `Samplers/GPSampler.cs` ports `samplers/_gp/sampler.py`: `infer_relative_search_space`
  (intersection minus singles), `_standardize_values`/`warn_and_convert_inf`, per-objective
  `GPRegressor.FitKernelParams` with the cross-trial cache, `LogEI` build (`threshold =
  standardized.max()`, `best_params = normalized[argmax]`), `_optimize_acqf` → `OptimMixed`, and
  `get_unnormalized_param`; the `gp:relative_params` chunked system-attr sharing is byte-compatible
  (`PyJson.Dumps`). **The MO (`LogEHVI`), constrained (`LogCEI/LogCEHVI`), and running-trial q-batch
  (`qLogEI/qLogEHVI`) branches throw `NotImplementedException`** pending 3e (their acqf classes ride the
  deferred `box_decomposition` + `qmc.sample_from_normal_sobol`) — same placeholder pattern as the
  earlier phases. **Key numeric enabler:** `ChoiceNoReplaceWeighted` reproduces numpy's
  `RandomState.choice(replace=False, p=)` — the `random_sample(size−n_uniq)` loop + `cumsum`/`searchsorted`
  cdf inversion + first-appearance dedup — bit-exact **and** advancing the RNG state identically
  (verified vs numpy), keeping the sampler in per-suggestion RNG lock-step. **Validated vs live Optuna
  (tolerance, the documented GP boundary):** `GpOptimMixedParityTests` runs `optimize_acqf_mixed` on an
  identical fitted GPR + space + seed across 5 mixed spaces (float/int-small/int-big→Brent/categorical/
  log-float/all-continuous) — the achieved acqf **value** matches to <1e-6 and OptunaSharp's optimum
  scored by Optuna's own acqf to <1e-5 (the argmin itself drifts ~1e-4 in flat regions at the loose
  `pgtol=1e-2` — validate by achieved value, not argmin). `GpSamplerParityTests` **replays** each C#
  study trial-by-trial through Optuna's `GPSampler.sample_relative` on the identical history (re-synced
  every trial → cascade-free, RNG locked): every GP suggestion matches to **<1e-2 in the normalized box**
  (fit-drift envelope; discrete/int/categorical suggestions **exact**, `int_mismatches == 0`), plus an
  end-to-end convergence sanity. **Why per-suggestion, not raw-trajectory:** the fit's ~1e-3 kernel-
  hyperparameter drift moves the LogEI argmax ~1e-2, which run free occasionally flips a roulette/argmax
  choice and diverges the tail — the sensitivity IS the GP boundary; re-syncing isolates the bound it
  actually holds. 7 unit (`OptimMixedTests`: Brent min + invalid-bracket guard, numpy-pinned
  `ChoiceNoReplaceWeighted` ×2, determinism, in-bounds/on-grid, all-categorical no-continuous) + 7 interop
  (`GpOptimMixedParityTests` 5-space sweep; `GpSamplerParityTests` float-sphere ×3 + mixed ×2 replay +
  convergence). Suite: **499 unit + 225 interop**, both TFMs, none skipped.
- ✅ **3.2 part 3e — the MO/constrained + running-trial acqf surface (all 6 sub-parts done; the entire
  `GPSampler` runs end-to-end).** Un-throws every `GPSampler` branch. **3e-1** `box_decomposition.py:
  get_non_dominated_box_bounds` → `Internal/Hypervolume.cs` (the 2.10 remainder; the Lacour17 Alg. 2
  upper-bound-set recurrence; **bit-exact vs Optuna** — only coordinates/negation/max/lexsort, no
  summation). **3e-2** `LogPI` + `LogCEI` (`Gp/Acqf.cs`) → the **SO-constrained** branch: LogPI =
  `log_ndtr((mean−thr)/σ)` (rides `TruncNorm.LogNdtr`, hand-derived φ/Φ gradient), LogCEI = LogEI + Σ
  LogPI; the sampler's `_get_constraint_vals_and_feasibility`/`_get_constraints_acqf_args`. No QMC.
  **3e-3** `qmc.sample_from_normal_sobol` → `Gp/Qmc.cs` + `Internal/TorchMt19937.cs` + cephes
  `TruncNorm.Ndtri`: **torch's** scrambled Sobol' reproduced bit-for-bit (torch unscrambled ≡ scipy ≡
  `QuasiMonteCarlo.SobolSv`; torch CPU RNG = classic MT19937 init_genrand, `randint(2)` = low bit; the
  unit-diagonal MSB LMS = scipy's `_cscramble`), then Φ⁻¹ via cephes ndtri → matches torch's `erfinv`
  path to ~1e-15. **Gotcha:** torch computes row 0 (`_first_point`) in float32 then upcasts (a ~1e-8
  rounding on the first point only; rows 1+ are float64). **3e-4** `LogEHVI` (`Gp/Acqf.cs`) → the **MO**
  branch: the Sobol'-QMC EHVI estimate (Eq. (1) of arxiv 2006.05078) over the box decomposition +
  `_get_reference_point` (`nextafter`) + `_get_non_dominated_box_bounds` (maximization wrapper);
  hand-derived gradient (softmax over boxes/samples × box-clamp mask, chained through each objective's
  posterior gradient). `_get_best_params_for_multi_objective` (`choice(replace=False)` =
  `permutation[:size]`, bit-exact via the shuffle). **3e-5** `LogCEHVI` (`Gp/Acqf.cs`) → the **MO-
  constrained** branch: LogEHVI over the feasible subset (or None when all infeasible) + Σ LogPI.
  **Validated:** each acqf's eval **and** hand-derived gradient vs live Optuna autograd (fixed-
  hyperparameter sweeps, isolating the acqf math from fit drift; eval <1e-7…1e-8, grad <1e-5…1e-6) +
  per-suggestion **replay** parity of each sampler branch (SO-constrained/MO/MO-constrained float-sphere
  + mixed spaces; continuous within a 2e-2…5e-2 normalized envelope, discrete/int **exact**). Tolerance,
  the documented GP boundary (posterior rides the managed triangular solve; QMC samples match torch
  ~1e-15). Commits `8381d37`/`4873140`/`0d731c8`/`055747c`/`6284a8f`. Suite after 3e-5: **525 unit + 238
  interop**, both TFMs, none skipped. **3e-6 — the running-trial q-batch surface (un-throws the last four
  branches; the ENTIRE `GPSampler` now runs end-to-end).** `ConditionalGPRegressor`
  (`Gp/ConditionalGPRegressor.cs`, port of `gp.py`) conditions the GP on pre-sampled running fantasies:
  `posterior(joint=True)` (`PosteriorJoint`), `append_running_data` (Kriging Believer — bordered/block
  `_extend_cholesky` + `_solve_cholesky` over the augmented data), `get_fantasy_samples`,
  `sample_joint_posterior` (forward + the query-sample gradient — only the query column depends on `x`).
  The four q-acqfs (`Gp/Acqf.cs`): **3e-6a** the Kriging-Believer path in `LogPI`/`LogEHVI`/`LogCEHVI`
  (`normalizedParamsOfRunningTrials`) → the MO-constrained running branch (which stays `LogCEHVI`, not a
  q-acqf); **3e-6b** `qLogEI` + `_compute_mean_max_utility` (`Acqf.MeanMaxUtility[WithGrad]`, the
  `amax`/`logsumexp` reduction) → the SO running branch; **3e-6c** `qLogPI` (`logsigmoid`) + `qLogCEI`
  (composes the qLogEI + Σ qLogPI per-sample log utilities, each constraint seeded `qmc_seed+i+1`) → the
  SO-constrained running branch; **3e-6d** `qLogEHVI` (per-QMC-draw box decomposition from that draw's
  fantasies + `_per_sample_log_hvi`, nested-softmax gradient) → the MO running branch. **The
  `sample_joint_posterior` gradient and the `amax`/`logsumexp` reduction gradients are hand-derived — the
  hardest gradient in the GP stack** (autograd's `amax` selects the per-sample argmax; the gradient flows
  only where the query column wins). Validated: each acqf eval+gradient vs live Optuna autograd
  (fixed-hyperparameter sweeps, eval <1e-7, grad <1e-5, finite-only to skip autograd's `amax`-kink nans)
  + a per-suggestion **replay** parity for each running branch — an ask/tell schedule that leaves trials
  RUNNING before the trial under test, replayed through live Optuna (running trials left RUNNING with
  their shared params, RNG in lock-step; continuous within 5e-2 normalized, int/discrete exact) — plus
  KB-acqf and qLogEI/qLogEHVI unit tests (FD gradients). Commits `741b793`/`a75b14e`/`c33d7b3`/`51ec724`.
  Suite after 3e-6: **533 unit + 252 interop**, both TFMs, none skipped.
- ✅ **3.1 part 1 — viz data layer foundations (`OptunaSharp.Visualization`).** Ported the pure (non-plotly)
  half of `_utils.py` (`VisualizationUtils`: `CheckPlotArgs`/`IsLogScale`/`IsNumerical`/`FilterNonfinite`/
  `IsReverseScale`) plus the three simplest `_get_*_info` builders — `_optimization_history.py`
  (`OptimizationHistory.GetInfoList`: raw + best-value `np.minimum/maximum.accumulate`, the Feasible/
  Infeasible/Incomplete states, and the `error_bar` `np.mean`/`np.std` aggregate), `_edf.py`
  (`Edf.GetInfo`: `np.linspace` grid + `<=`-count EDF), and `_intermediate_values.py`
  (`IntermediateValues.GetPlotInfo`). Info records are **public** (`OptimizationHistoryInfo`/`ValuesInfo`/
  `ValueState`/`EdfInfo`/`EdfLineInfo`/`IntermediatePlotInfo`/`TrialInfo`) — the Tier X rendering package
  (5.2) consumes them across the assembly boundary. Added bit-exact numpy `MathUtil.Linspace`/`Mean`/`Std`.
  **Bit-exact vs live Optuna 5.0.0.dev** (no erf, no LAPACK on these paths — objectives use `x*x` not
  `x**2` so the stored values stay bit-identical too): 13 live parity tests (optimization-history single/
  constraints/error-bar, EDF single/multi, intermediate-values — every value/std/count/x-grid byte-identical)
  + 15 unit tests.
- ✅ **2.16 part 1 — base CMA-ES (`CmaEsSampler` + `Cma`).** Ported `cmaes.CMA` (`Samplers/CmaEs/Cma.cs`,
  CMA + `lr_adapt`) and `samplers/_cmaes.py`'s `CmaEsSampler` (default + `lr_adapt`, deprecated
  `x0`/`sigma0`, `popsize`, `consider_pruned_trials`, `n_startup_trials`, independent-sampler fallback,
  MO guard, verbatim validation errors). **Bit-exact vs cmaes 0.13.0** (`D2,B = eigh(C)` →
  `LinAlg.Eigh`/OpenBLAS), NOT tolerance-bounded. Optimizer state → self-describing own JSON
  (`[Misaligned]`, pickle non-goal). See the 2.16 done-item for the two subtle bit-exactness fixes
  (upper-triangle covariance symmetry across the state round-trip; the `y_w` sequential reduction).
- ✅ **2.16 part 2a — SepCMA (`use_separable_cma`).** Ported `cmaes.SepCMA` (`Samplers/CmaEs/SepCma.cs`,
  diagonal covariance, no eigendecomposition) behind the shared `ICmaOptimizer` seam; the sampler's
  `InitOptimizer` now returns SepCMA (or falls back to CMA + a `UserWarning` on a 1-D space, matching
  Optuna). **Bit-exact vs cmaes 0.13.0** (verified over an 11-config sampler differential incl. dims
  2/3/5/8 and popsize 10/20). Note SepCMA's constructor uses Python `sum()` (sequential) for the
  weights, unlike CMA's `np.sum`.
- ✅ **2.16 part 2b — Warm-Starting (`source_trials`).** Ported `get_warm_start_mgd`
  (`Samplers/CmaEs/WarmStart.cs`) + the sampler's `source_solutions` gathering and
  `_is_compatible_search_space`; added `LinAlg.Det` (byte-exact vs numpy over a 50-case sweep).
  `InitOptimizer` now estimates the prior (mean, sigma, cov) from the top-γ source trials and seeds the
  CMA with it. **Bit-exact vs cmaes 0.13.0** (a source `RandomSampler` study → `get_warm_start_mgd` → CMA
  trajectory, verified over a 4-config differential incl. dims 2/3/5). The rank-one outer products carry
  no summation and the determinant rides LAPACK, so it is byte-exact.
- ✅ **2.16 part 2c — CMAwM (`with_margin`).** Ported `cmaes.CMAwM` (`Samplers/CmaEs/CmaWm.cs`) — it
  wraps a full-covariance `Cma` and, on every `tell`, applies the margin correction that lower-bounds each
  discrete dim's marginal probability (Hamano et al., GECCO 2022). `ask()` returns two vectors
  (`AskEncoded()` → encoded + raw); the sampler branches `optimizer is CmaWm` to store the raw x as the
  `x_for_tell` trial system attr and tell with it, exactly like Optuna's `isinstance(…, cmaes.CMAwM)`
  path; `InitOptimizer` builds the per-dim `steps` and constructs the CMAwM (attr prefix `cmawm:`).
  **The one non-bit-exact CMA-ES backend** — the margin correction rides scipy's `norm.cdf` →
  `Erf.NormCdf` and `chi2.ppf(q, df=1) = Φ⁻¹((1+q)/2)²` → `TruncNorm.NdtriExp` (the closed-form df=1
  identity; verified to agree with scipy's `chi2.ppf` end-to-end at machine epsilon — the identity's
  huge relative error near `q→1` cancels because those χ values only ever sit in denominators/ratios).
  **But the divergence stays invisible in practice:** the correction leaves `mean[continuous]`/`C`/`sigma`
  untouched, so the continuous dims never touch erf (bit-exact), and the discrete dims snap to an exact
  grid — so the erf drift lives only inside `mean[discrete]`/`_A` and never flips an encoding decision.
  Verified over a **40-config / 1040-trial offline differential** (4 mixed-space templates — int,
  discretized-float, log-float, all-discrete, dims 2–5 × 10 seeds vs live Optuna): every discrete param
  and `cmawm:generation` **byte-exact** (0 mismatches), continuous params to **1.7e-13**, values to
  **1.3e-12**. `np.arange` grid + `searchsorted(side='left')` reproduced bit-exact; own-JSON state wraps
  the inner `CmaState` + steps/margin/`_A` (`[Misaligned]`, pickle non-goal). IPOP/BIPOP restarts are
  deprecated→None in cmaes 0.13.0 and drop out. 6 unit tests (seed=42 mixed trajectory, reproducibility,
  minimization, `cmawm:`-prefix + `x_for_tell` attribution, `CmaWmState` round-trip idempotency) + 3 live
  tolerance-trajectory parity seeds.
- **2.10 remainder — `box_decomposition.py`** ✅ **DONE** (Phase 3.2 part 3e-1): `get_non_dominated_box_bounds`
  ported into `Internal/Hypervolume.cs`, bit-exact vs Optuna, consumed by the GP `LogEHVI`/`LogCEHVI` acqfs.
- **2.12 remainder — QMC scrambling** ✅ **DONE for Sobol'** (see the 2.12 done-item): `QMCSampler`
  `scramble=true` with the default `qmc_type="sobol"` is wired through `Internal/ScrambledSobol.cs`
  (bit-exact vs scipy; full-state bit-identical vs live Optuna). The `seed` is handled exactly as Optuna
  (`np.random.PCG64().random_raw()` when unseeded), and the `qmc:<sha256>` counter hash folds in
  `scramble`/`seed` byte-for-byte. **Only scrambled Halton stays open** (`qmc_type="halton",
  scramble=true` throws): it needs numpy's PCG64 `shuffle`/`random_interval` (scipy's
  `_van_der_corput_permutations`), which NumSharp exposes no public primitive for — a NumSharp upstream
  addition is the prerequisite.
- After 2.16: Phase 3 (viz data layer, GP). GP's `torch.linalg.cholesky`/`solve_triangular` map to
  `LinAlg.Cholesky` (byte-exact) + `LinAlg.SolveTriangular` (the one still-managed primitive — NumSharp
  exposes no LAPACK `dtrtrs`; route it through NumSharp once it gains the binding).

**Read the divergence notes below** (erf/ucrt, argsort tie-break, HSSP ≥3-obj tie, UNDX-QR sign). The
2.14 solve note is **resolved** by 2.15's OpenBLAS route. The NSGA-III niching `rng.shuffle` reproduces
numpy's Python-list Fisher-Yates via an index permutation (bit-exact), exactly like the GridSampler shuffle.

**Wiring gotcha for OpenBLAS (important for every future np.linalg user):** a bare package/project
reference is **not** enough — the CLR runs the OpenBLAS backend's self-registering `[ModuleInitializer]`
only when code first touches one of its types, so an unreferenced assembly stays unloaded and
`np.linalg.cholesky`/`eigh` throw `OpenBlasMissingBackendException`. `Internal/OpenBlasActivation.cs` is
Core's own `[ModuleInitializer]` that touches `OpenBlasEngine.LapackAvailable` to force the load the
moment OptunaSharp.Core loads. Keep it.

### Phase status

| Phase | Status | Notes |
|---|---|---|
| 0 — Core framework (0.1–0.9) | ✅ | walking-skeleton gate (0.9) bit-identical to Python for shared seed |
| 1 — Native persistence (1.1–1.5) | ✅ | journal Python↔C# live interop green; **M2 reached** |
| 2 — Native features | ✅ | **2.1–2.16 COMPLETE** (full TPE; MO hypervolume/HSSP; QMC — Sobol' both scrambled & unscrambled + Halton unscrambled; NSGA-II; NSGA-III; OpenBLAS linalg; **CMA-ES `CmaEsSampler` — full CMA + separable SepCMA + Warm-Starting + CMAwM `with_margin`**); box-decomp landed in 3.2; only scrambled **Halton** remains (needs a NumSharp PCG64 `shuffle`) |
| 3 — Native heavy tail | ✅ | **COMPLETE (3.1 + 3.2 + 3.3).** 3.1 viz data layer; 3.2 §7 autodiff gate **GREEN & CLOSED** — the GP fit fully ported (hand-derived MLL gradient ~1e-13 vs autograd; managed L-BFGS-B port of scipy's `__lbfgsb.c`) and **the ENTIRE `GPSampler`** (every sequential 3a–3e-5 + running-trial q-batch 3e-6 path), no TorchSharp; **3.3 GP terminators** — `RegretBoundEvaluator` (default) + `EMMREvaluator` un-thrown (`UCB`/`LCB` acqfs + `OptimSample` + the deferred `_get_improvement_info` viz builder), tolerance parity vs live Optuna. **Tier N is COMPLETE.** Rendering `plot_*`→5.2 |
| 4 — External storage (Tier X) | 🟩 | **4.5 cloud artifacts DONE** (S3/GCS ports + Azure Blob/Cloudflare R2 extensions — 4 `.Artifacts.*` satellite pkgs, feat `3923fcd`); 4.1–4.4 pending; `ICachedStorageBackend` seam for 4.1 declared in 1.2 |
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
  improvement evaluator) and `EMMREvaluator` shipped here as benign-construction placeholders that
  threw on `Evaluate` until the GP stack — **now ported and un-thrown in 3.3** (see the 3.3 done-item).
  19 unit
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
  path); its HSSP tie-break threw pending **2.10** (`_hypervolume/*`, since landed) — the
  exact-rank-partition case works, and MO-with-explicit-`target` uses the QuantileFilter path fully. **Numeric strategy
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
  `json.dumps`), the deprecated-arg `FutureWarning`s, and `reseed_rng`. `group=true` threw pending
  **2.9** (landed there); MO uses the independent path (the MO-split HSSP tie-break landed in **2.10**).
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
- ✅ **2.9** — `feat(samplers): Phase 2.9 — group-decomposed multivariate TPE (group=true)` (`66064a4`).
  Replaced the `group=true` `NotImplementedException` placeholder with the full group-decomposed
  multivariate TPE (the `group` branches of `_tpe/sampler.py`), wired over the already-ported
  `_GroupDecomposedSearchSpace`/`_SearchSpaceGroup`. `infer_relative_search_space` computes the
  decomposed `_search_space_group` and unions the non-single distributions across subspaces;
  `sample_relative` iterates the subspaces, builds a per-subspace search space (`StringComparer.Ordinal`
  key sort == Python `sorted(sub_space.items())`; drops single dists and keys absent from the inferred
  space — the PartialFixedSampler shrink case), and calls `_sample_relative` once per subspace, merging
  the results (the `constant_liar` `tpe:relative_params` sharing now covers the merged params). The
  `_is_multivariate` group branch returns True and, when `multivariate` is left at its default for a
  multi-objective study, emits the "group overrides the MO default" `UserWarning`; construction emits
  `warn_experimental_argument("group")`. **Parity (verified vs live Optuna 5.0.0.dev):** the group path
  stays in exact RNG lock-step — branch decisions + trial state/number/param-set bit-for-bit, continuous
  values to the erf/ucrt tolerance. Offline differential sweeps: a branching float space (subspaces
  {x}/{y}/{z}) over seeds 0/7/42 → max abs diff 8.9e-15; a joint multi-param subspace ({a,b,c} sampled
  together + {d}) over seeds 0/7/42 → max abs diff 2.4e-13; zero violations. 5 new unit tests
  (experimental warning, {x}/{y,z} decomposition → union-infer, SO/MO default-multivariate incl. the
  MO UserWarning, group reproducibility) replacing the obsolete throw-test + 3 live branching-space
  trajectory-parity seeds. The two 2.8 `[Misaligned]` notes (erf/ucrt, argsort tie order) apply
  unchanged.
- ✅ **2.10 core** — `feat(hypervolume): Phase 2.10 core — WFG hypervolume + HSSP, unblock MO TPE split`
  (`c7bd648`). New `Internal/Hypervolume.cs`: `compute_hypervolume` (the `wfg.py` calculator —
  `_compute_2d`/`_compute_3d`/`_compute_hv`/`_compute_exclusive_hv`, the `<=`/non-finite/empty guards,
  assume_pareto-vs-unique-Pareto dispatch) and `_solve_hssp` (the `hssp.py` greedy — `_solve_hssp_2d`,
  `_lazy_contribs_update`'s incremental/decremental contribution bounds + upper-bound skip,
  `_solve_hssp_on_unique_loss_vals`, the `np.unique(return_index=True)` wrapper via new
  `MultiObjective.UniqueRowsFirstIndex`). Wired the HSSP tie-break into
  `TpeTrialSplit.SplitCompleteTrialsMultiObjective` (`_get_reference_point` + `SolveHssp`), **removing
  the `NotImplementedException`** — multi-objective TPE splitting and PED-ANOVA with `target=null` now
  run end-to-end. **Numeric:** NumPy `@`/`np.dot` as a sequential dot, `.prod` sequential;
  `compute_hypervolume` bit-exact-to-tolerance vs Optuna over 120 mixed 2–6D cases (max rel err 2e-16);
  end-to-end 2-/3-objective MO TPE trajectories match live Optuna in RNG lock-step to the erf tolerance
  (max abs diff 5.7e-14); MO PED-ANOVA `target=null` matches (exact selection path, erf-tolerance
  values). 25 hypervolume unit tests (ports of `tests/hypervolume_tests/*`) + 4 MO TPE unit tests + 5
  MO TPE trajectory-parity seeds + 2 MO importance `target=null` seeds. **box_decomposition deferred to
  3.2** (its only consumer is `_gp/acqf.py`). See the 2.10 `[Misaligned]` HSSP note below.
- ✅ **2.11** — `test(tpe): Phase 2.11 — verify MOTPE + constraints already work, lock with tests`
  (`9a6ffe1`). **No new production code** — multivariate multi-objective TPE (MOTPE) and constrained
  TPE already compose from earlier phases: the MO uniform-weight `l(x)` branch (2.7), the sampler +
  `default_gamma_multiobjective` + `_split_infeasible_trials`/`_get_infeasible_trial_score` (2.8), the
  group path (2.9), the non-domination/HSSP split (2.10), and the already-ported constraint infra
  (`ConstrainedOptimization`, `_process_constraints_after_trial`, `trial.set_constraint`,
  `FrozenTrial.Constraints`). Verified end-to-end vs live Optuna in full RNG lock-step to the erf
  tolerance: multivariate=True MO TPE (max abs diff 4.6e-14), constrained SO/MO TPE via
  `constraints_func` (7.1e-14, with genuine infeasible trials — 16/140 — exercising the infeasible
  split and matching stored constraint values), and `trial.set_constraint` (7.1e-14). 2 unit tests
  (multivariate MO reproducibility; constrained-TPE constraint storage + reproducibility) + 7 interop
  seeds (3 multivariate MO + 4 constrained SO/MO). **TPE is now complete** — univariate, multivariate,
  group, MOTPE, constrained.
- ✅ **2.12** — `feat(samplers): Phase 2.12 — QMCSampler (non-scrambled Sobol' + Halton)` (`217af06`).
  New `Samplers/QMCSampler.cs` + `Internal/QuasiMonteCarlo.cs` + `Internal/SobolDirectionNumbers.cs`.
  Reproduces `scipy.stats.qmc` bit-for-bit for `scramble=false` (the default) — the first sampler
  validated against **scipy**, not just NumPy/Optuna. Sobol': scipy's `_initialize_v` recurrence over
  the Joe-Kuo `new-joe-kuo-6.21201` direction numbers (first 1024 dims, matching rustuna; embedded
  base64) + BITS=30 Gray-code XOR (`_sv` verified identical through d=1024). Halton: van der Corput
  radical inverse in the first primes (runtime sieve). The sampler ports `infer_relative_search_space`,
  `sample_relative` (categorical → `FloatDistribution(0, C)` pseudo-space + `SearchSpaceTransform`
  transform01 + `int()` floor), `sample_independent` (delegates to the independent `RandomSampler` →
  first trial + non-relative params stay bit-exact on MT19937), and `_find_sample_id` (the
  `qmc:`+sha256(`json.dumps(qmc_vars)`) study-attr counter — **byte-identical to Python**, so full study
  state matches and cross-language journals share the counter). **Verified vs live Optuna: bit-exact
  (max abs diff 0.0)** over Sobol+Halton × float/int/categorical/log-float × seeds. 10 unit + 6
  live-interop full-state-parity seeds. **Scrambling deferred at the time** — scipy scrambles with its
  internal PCG64 inside compiled Cython; `scramble=true` threw (NumSharp has PCG64, but scipy's compiled
  scramble looked non-reproducible without its source). Superseded by the follow-up below.
- ✅ **2.12 follow-up — scrambled Sobol' `QMCSampler(scramble=true)`** (`d0f68eb`). Un-throws
  the scramble path for the default `qmc_type="sobol"` by routing `_sample_qmc` through
  `Internal/ScrambledSobol.cs` (the Phase 3.2 part-3c bit-exact port of `scipy.stats.qmc.Sobol(scramble=True)`
  — LMS + digital shift seeded by `numpy.random.default_rng(seed)`; the earlier "not reproducible without
  scipy's Cython" concern was resolved once NumSharp's `default_rng`/PCG64 was shown byte-identical to numpy).
  `Sobol(d, seed, scramble=True).fast_forward(sample_id).random(1)` is the point at index `sample_id` of the
  scrambled sequence, i.e. `ScrambledSobol.Random(d, sample_id+1, seed)[sample_id]` (verified bit-exact vs
  scipy over the `fast_forward` cases). The `seed` follows Optuna exactly (`np.random.PCG64().random_raw()`
  when unseeded — drawn from NumSharp entropy, never `System.Random`; the explicit `seed` otherwise), and
  `_find_sample_id`'s `qmc:`+sha256 counter now folds in `scramble=true`/`seed` **byte-identically to Python**
  (JSON + hash verified equal). **Only scrambled Halton stays open** — `qmc_type="halton", scramble=true`
  throws a clear `NotImplementedException`: it needs numpy's PCG64 `shuffle`/`random_interval` (scipy's
  `_van_der_corput_permutations`), which NumSharp exposes no public primitive for (its PCG64 `NextUInt32` is
  `internal`, no public `random_raw`) — a small NumSharp upstream addition is the prerequisite, so this is
  the last native sliver. 5 new unit (`QmcSamplerTests`: Halton-scramble throws, Sobol-scramble constructs,
  the pinned seed=42 scrambled trajectory vs Optuna, same-seed reproducibility, scrambled≠unscrambled) + 3
  live-interop full-state parity (`QmcSamplerParityTests`: mixed-space seed 0/42 + a power-of-two run vs live
  `QMCSampler(scramble=True)`). Suite: **548 unit + 262 interop**, both TFMs, none skipped.
- ✅ **2.13** — `feat(samplers): Phase 2.13 — NSGAIISampler (NSGA-II)`. The full genetic-algorithm
  MO sampler, ported structure-for-structure into `Samplers/Nsgaii/*` + `Samplers/BaseGASampler.cs`
  + `Samplers/NSGAIISampler.cs`. `BaseGASampler` (generation attribution + parent-population cache in
  study/trial system attrs — `NSGAIISampler:generation`, `NSGAIISampler:parent:<gen>`, byte-identical
  to Python), `NSGAIIElitePopulationSelectionStrategy` (`_rank_population` over `MultiObjective.
  FastNonDominationRank` + `_calc_crowding_distance`/`_crowding_distance_sort`), `NSGAIIChildGeneration
  Strategy` (binary-tournament parent selection, crossover/no-crossover branch, per-param mutation
  decision), `perform_crossover`/`perform_mutation`, all **6 crossovers** (`UniformCrossover`,
  `BLXAlphaCrossover`, `SBXCrossover`, `VSBXCrossover`, `SPXCrossover`, `UNDXCrossover`) and both
  mutations (`BaseMutation`, `PolynomialMutation`), the constrained-domination path
  (`_constrained_dominates`/`_evaluate_penalty`/`_validate_constraints`), and the two strategy delegates
  + `NSGAIIAfterTrialStrategy`. Two separate MT19937 streams from one seed (child generation on
  `self._rng`; initial-population / mutated-away / non-intersection params on the independent
  `RandomSampler`), replicated exactly. **Parity (verified vs live Optuna 5.0.0.dev + a 205-config
  offline differential):** the default path AND every arithmetic crossover (blx/sbx/vsbx/spx) are
  **bit-exact** — full study state (each trial's params-internal-repr at IEEE-bit level, values,
  per-trial generation, and the parent-cache trial-id lists) matches byte-for-byte, SO and MO, with
  `PolynomialMutation`, `set_constraint`, and the deprecated `constraints_func` (old-format list
  constraint). RNG draw count/order verified across the categorical uniform crossover
  (`rng.rand(n)` masks), numerical crossover (`rng.rand`/`np.power`/min-max/mean, all bit-exact via
  `Math.Pow`), and the `rng.choice`→`randint(0,n)` tournament draws. 20 unit tests (hardcoded
  seed=42 sequence from real Optuna, 6-crossover reproducibility, MO reproducibility, validation +
  warnings) + 14 live full-state parity tests (SO/MO defaults, 5 bit-exact crossovers, poly mutation,
  set_constraint, constraints_func). The crowding-distance sort is a **stable Python `list.sort`** (not
  `np.argsort`), so it is bit-reproducible; the one `frompyfunc`/object-array site (categorical
  crossover) maps to a plain `object?[]`.
  - **[Misaligned] UNDX QR column signs (the fifth documented divergence, on `UNDXCrossover`).**
    `UNDXCrossover` (n_parents=3) builds the orthonormal basis orthogonal to the primary search line
    via `np.linalg.qr` (LAPACK `dgeqrf`/`dorgqr`). OptunaSharp uses a managed Householder QR that
    reproduces numpy's basis vectors to ~1e-14 in **magnitude** but not their **per-column signs**
    (LAPACK's column-sign bookkeeping is not portably reproducible — the same managed-vs-LAPACK class
    §10 excuses). Because `etas ~ N(0, σ²)` is symmetric, a sign-flipped basis is still a *correct*
    UNDX sampler, but with **≥2 numerical parameters** the child values differ from Optuna by O(1) and
    that difference cascades through the crowding/non-domination selection, so a multi-parameter UNDX
    study's trajectory diverges from Optuna's (it stays RNG-lock-step and reproducible within
    OptunaSharp). **UNDX with a single numerical parameter is bit-exact** (the basis is unused). The
    other 5 crossovers have no QR/linalg and are exact; UNDX ≥2-D is excluded from the bit-parity
    tests (reproducibility only).
- ✅ **2.14** — `feat(samplers): Phase 2.14 — NSGAIIISampler (NSGA-III)`. The many-objective GA sampler,
  ported structure-for-structure into `Samplers/Nsgaiii/*` + `Samplers/NSGAIIISampler.cs`. `NSGAIIISampler`
  mirrors `NSGAIISampler` (same `BaseGASampler`, `NSGAIIChildGenerationStrategy`, crossovers, mutations,
  `NSGAIIAfterTrialStrategy`, two-stream RNG) and only swaps the elite selection for
  `NSGAIIIElitePopulationSelectionStrategy` — `@experimental_class("3.2.0")` ExperimentalWarning on
  construction, `reference_points`/`dividing_parameter` args, no experimental-argument warnings (unlike
  NSGA-II). The strategy is the full `_nsgaiii/_elite_population_selection_strategy.py`: rank by shared
  `_rank_population` (now `internal static`), fill by rank, and truncate the borderline front by
  **reference-point niching** — `_generate_default_reference_point` (`combinations_with_replacement` +
  `np.add.at`), `_filter_inf` (nanmax/nanmin/clip), `_normalize_objective_values` (ideal-point subtract,
  ASF `einsum`/argmin extreme points, hyperplane `solve`/`matrix_rank` branch), `_associate_individuals_
  with_reference_points`, and `_preserve_niche_individuals` (the `defaultdict` niche sweep with
  `rng.shuffle`/`sort(reverse=True)`/`pop()`). New `Samplers/Nsgaiii/NsgaiiiLinAlg.cs` supplies the small
  managed `Solve` (reciprocal-multiplier LU, LAPACK `dgetf2` shape) + `MatrixRank` (GE pivot count).
  **Parity (verified vs live Optuna 5.0.0.dev):** a **400-config offline differential** — 5 crossovers ×
  {2,3,4} objectives × 25 seeds + 25 constrained (set_constraint), ~10k trials — matched **byte-for-byte
  with zero divergence** (full study state: every trial's params/values, per-trial generation, parent-cache
  trial-id lists). 15 unit tests (hardcoded seed=42 SO sequence, default reference points for M=2/3,
  MO/crossover/custom-reference-point reproducibility, validation, experimental+deprecation warnings) + 11
  live full-state parity tests (2-/3-objective defaults, 4 crossovers, custom reference points, poly
  mutation, set_constraint). **Numeric due-diligence (all verified before porting):** `_filter_inf` and
  the reference-point association are bit-exact (the 4-operand `einsum("ni,pi,p,pm->npm")` reproduced
  term-by-term — 2000/2000, vs 46% for the naive factored form); the ASF argmin and `matrix_rank`
  degeneracy branch are exact; the reciprocal-multiplier `Solve` matches numpy on ~96% of the real
  extreme-point systems (vs ~87% for direct-division), the residual ~1 ULP being the only managed-vs-LAPACK
  gap — and it flipped **no** niche selection across the 400 studies. `rng.shuffle` on the niche lists
  reproduces numpy's Python-list Fisher-Yates via a bit-exact index permutation (the GridSampler pattern).
  - **[Misaligned] hyperplane-fit solve (a theoretical managed-vs-LAPACK ULP point that did not surface).**
    `_normalize_objective_values` fits the ASF-extreme-point hyperplane with `np.linalg.solve` (LAPACK
    `dgesv`). `NsgaiiiLinAlg.Solve` is the closest managed match (unblocked reciprocal-multiplier LU) but
    drifts ~1 ULP on a few percent of the small systems (numpy's LAPACK rides an arch-specific FMA `gemm`).
    A 1-ULP normalization difference only changes the outcome when it flips a niche-distance decision
    within ULPs of a tie; across the 400-config differential (incl. 4×4 fits) it never did, so NSGA-III is
    effectively bit-exact. Documented like the HSSP ≥3-obj note. `matrix_rank` uses a `max|a|·n·eps`
    tolerance vs numpy's SVD `σmax·n·eps`; the two agree except at the degeneracy boundary (rare, and the
    same tolerance class). **RESOLVED in 2.15:** the solve/matrix_rank now route through
    `LinAlg`→`np.linalg` over OpenBLAS (byte-identical to numpy), so this note is history — the 400-config
    differential still passes, now with a genuinely byte-exact hyperplane fit.
- ✅ **2.15** — `feat(linalg): Phase 2.15 — OpenBLAS-backed linalg primitives (LinAlg) for CMA-ES/GP`.
  **Redirected mid-flight by the maintainer** from hand-rolled managed primitives to
  **`NumSharp.Interop.OpenBLAS`** (*"use NumSharp + OpenBLAS any time Optuna uses NumPy"*). Core's csproj
  now references the OpenBLAS project the same conditional "csproj now / nuget later" way as NumSharp.Core;
  `Internal/OpenBlasActivation.cs` is a `[ModuleInitializer]` that force-loads it (a bare reference is not
  enough — the backend self-registers from its own module initializer, which the CLR runs only on first
  type touch). `Internal/LinAlg.cs` is a thin `double[,]`/`double[]` adapter whose `Solve`/`MatrixRank`/
  `Cholesky`/`Eigh` delegate to `np.linalg` — **byte-identical to NumPy** (the same scipy-openblas 0.3.31
  binaries numpy 2.4.2 bundles): verified cholesky, eigh values **and eigenvectors incl. signs**, and solve
  match numpy 2.4.2 bit-for-bit on a fixed 3×3 and over a 90-case random sweep (`Should().Be(0.0)`).
  `SolveTriangular` stays managed (forward/back substitution — NumSharp exposes no LAPACK `dtrtrs`; the one
  primitive not yet LAPACK-backed). Graduated NSGA-III's solve/matrix_rank onto `LinAlg` (deleting
  `NsgaiiiLinAlg.cs`), retro-upgrading the 2.14 hyperplane fit to byte-exact — the 400-config differential
  still passes 400/400. 12 unit tests (known cholesky/eigh/solve/triangular cases + a 2000-case
  reconstruction/residual/orthonormality self-consistency sweep) + 6 live interop tests (byte-exact fixed
  3×3, byte-exact cholesky/solve/eigenvalue sweeps, managed solve_triangular tolerance vs scipy). Suite:
  **393 unit + 151 interop**, both TFMs, none skipped. **Caveat:** `NumSharp.Interop.OpenBLAS` is not on
  nuget.org yet, so the CI/pack PackageReference fallback needs it published; the sibling ProjectReference
  is the working path (pack already records the dependency cleanly for both TFMs).
- ✅ **2.16 part 1** — `feat(samplers): Phase 2.16 part 1 — base CMA-ES (CmaEsSampler + Cma)`. Ported the
  external `cmaes.CMA` optimizer into `Samplers/CmaEs/Cma.cs` (ask/tell/should_stop, CMA + learning-rate
  adaptation, bounds resampling/repair, own-JSON state serialization) and `samplers/_cmaes.py`'s
  `CmaEsSampler` into `Samplers/CmaEsSampler.cs` (`infer_relative_search_space` — intersection space,
  drop single/categorical; `sample_relative` — restore-or-init optimizer, generation attribution +
  solution-trial gathering, tell + store, per-trial reseed `randint(1,2**16)+trial.number`, ask,
  untransform; `sample_independent`/`before_trial`/`after_trial` delegating to the seeded independent
  `RandomSampler`; deprecated `x0`/`sigma0` `FutureWarning`s; experimental-argument `ExperimentalWarning`s;
  verbatim `ValueError` validation for the incompatible-argument combinations; the MO guard). `LinAlg`
  gained `Dot(m,m)`/`Dot(m,v)`/`Outer`/`Norm` over `np.dot`/`np.outer`/`np.linalg.norm` (CBLAS/OpenBLAS).
  **Bit-exact vs cmaes 0.13.0** — the plan's byte-exact expectation held (eigh via `LinAlg.Eigh`), NOT
  tolerance-bounded. Verified over a 23-config offline differential (12 seeds × dim 3 × 50 trials; dims
  2/6/8/10; lr_adapt; popsize 6/12/20 — every trial's params/values/generation byte-identical) plus live
  interop. 18 unit tests (hardcoded seed=42 trajectory, reproducibility, minimization, generation
  attribution, all validation errors, experimental/deprecation warnings, the guarded-backend
  `NotSupportedException`s, and Cma-state round-trip idempotency/determinism) + 9 live full-state parity
  tests (float dim 2/3/5, lr_adapt, popsize 10/20, mixed float/int/categorical). Suite: **411 unit + 160
  interop**, both TFMs, none skipped.
  - **[Misaligned] optimizer state is our own JSON, not Python `pickle`.** Optuna stores
    `pickle.dumps(optimizer).hex()` chunked into `cma:optimizer:{i}` system attrs; OptunaSharp stores an
    equivalent `CmaState` JSON under the same chunked keys, so a study can NOT resume CMA-ES mid-run
    across the Python/C# boundary. The interop parity tests exclude the `*:optimizer:*` attrs and compare
    everything else (params, values, distributions, `cma:generation`) bit-for-bit.
  - **Two subtle numpy-fidelity fixes were required (both verified against live cmaes).** (1) **Covariance
    symmetry across the state round-trip.** `_C = B·diag(D²)·Bᵀ` via CBLAS is ~1 ULP asymmetric; Optuna's
    pickle uses `_compress_symmetric`/`_decompress_symmetric` (store the UPPER triangle, mirror into both),
    so a restored covariance is EXACTLY symmetric-from-upper — otherwise the next eigendecomposition's
    `(C+Cᵀ)/2` averages `(upper+lower)/2 ≠ upper` and the eigenvectors (hence every ask) diverge. `ToState`
    stores the covariance symmetrized-from-upper to match. (2) **`y_w` is a sequential reduction.**
    `np.sum(y_k[:μ].T * w[:μ], axis=1)` reduces over the F-contiguous product's STRIDED axis, so numpy sums
    it sequentially (its pairwise 8-way only fires on a contiguous/last-axis reduction); a `PairwiseSum`
    there diverges once μ ≥ 8. `rank_mu` (axis=0 strided) and the `w_io` column norm are likewise
    sequential, while contiguous 1-D `np.sum`s (weights, `sum_weights`) stay pairwise. General rule for the
    remaining variants: **numpy `np.sum(..., axis=k)` over a strided axis is sequential; only a contiguous
    last-axis / 1-D reduction is pairwise.**
- ✅ **2.16 part 2a** — `feat(samplers): Phase 2.16 part 2a — SepCMA (use_separable_cma)`. Ported the
  external `cmaes.SepCMA` (separable / diagonal-covariance CMA-ES) into `Samplers/CmaEs/SepCma.cs` and
  wired `use_separable_cma=True` into `CmaEsSampler` behind a new shared `ICmaOptimizer` seam (`Cma` and
  `SepCma` both implement it; the sampler's optimizer-state JSON is now self-describing — a `cma`/`sepcma`
  tag prefix — so restore reconstructs the right type even for the 1-D `use_separable_cma` fallback that
  stores a `Cma` under the `sepcma:` prefix). `InitOptimizer` returns `SepCma` for a ≥2-D space, or falls
  back to `Cma` with the verbatim "does not operate meaningfully on single-dimensional" `UserWarning` on a
  1-D space (matching Optuna). **Bit-exact vs cmaes 0.13.0** — SepCMA has a diagonal `_C` (1-D, `D=√C`, no
  eigendecomposition, no LAPACK) so parity is byte-for-byte, verified over an 11-config sampler
  differential (dims 2/3/5/8 across seeds; popsize 10/20; ~450 trials) plus the standalone-optimizer
  differential (5 configs). **Numeric note:** SepCMA's constructor forms the recombination weights with
  Python's builtin `sum()` (SEQUENTIAL), not `np.sum` — distinct from CMA — while `y_w`/`rank_mu` are the
  same strided-sequential axis reductions and `np.sum(weights)` in `tell` stays contiguous-pairwise. 2
  unit tests (hardcoded seed=42 SepCMA trajectory; 1-D fallback `UserWarning`) + 3 live full-state parity
  seeds (dims 2/3/5). Suite: **412 unit + 163 interop**, both TFMs, none skipped.
- ✅ **2.16 part 2b** — `feat(samplers): Phase 2.16 part 2b — Warm-Starting CMA-ES (source_trials)`. Ported
  `cmaes.get_warm_start_mgd` into `Samplers/CmaEs/WarmStart.cs` (estimate a prior gaussian from the top-γ
  source-task solutions) and wired the `source_trials` path in `CmaEsSampler.InitOptimizer`: gather
  `source_solutions` from the source trials whose state is in the expected set and whose distributions are
  key-compatible with the relative space (`_is_compatible_search_space`), `sign`-flip the values for the
  study direction, raise the verbatim `"No compatible source_trials"` `ValueError` on an empty set, and
  seed the `Cma` with `get_warm_start_mgd`'s (mean, sigma, cov). Added `LinAlg.Det` (`np.linalg.det`,
  **byte-exact vs numpy** over a 50-case 2–6-D sweep). **Bit-exact vs cmaes 0.13.0** — the rank-one outer
  products (`np.dot(v.reshape(d,1), v.reshape(1,d))`) carry no summation, the accumulations are plain
  sequential `+=`, and the determinant/scalar-powers ride LAPACK/`Math.Pow`; verified over a 4-config
  differential (a source `RandomSampler` study → warm-start → CMA trajectory, dims 2/3/5, mixed seeds).
  2 unit tests (reproducibility; the `No compatible source_trials` `ValueError`) + 3 live full-state parity
  seeds (dims 2/3/5). Suite: **414 unit + 166 interop**, both TFMs, none skipped.
- ✅ **3.1 part 1** — `feat(viz): Phase 3.1 part 1 — visualization data layer foundations`. New
  `OptunaSharp.Visualization` namespace (`src/OptunaSharp.Core/Visualization/`) porting the pure
  (non-plotly) half of the viz modules — the plotly rendering (`plot_*`/`_get_*_plot`) stays a Tier X
  concern (**5.2**). Ported `_utils.py`'s pure helpers into `VisualizationUtils` (`CheckPlotArgs` — the MO
  `ValueError` + default-`target_name` `UserWarning`; `IsLogScale`; `IsNumerical`; `FilterNonfinite`;
  `IsReverseScale`) and the three simplest `_get_*_info` builders: `_optimization_history.py`
  (`OptimizationHistory.GetInfoList` — per-trial values, running best via `np.minimum/maximum.accumulate`
  with `±inf` sentinels for infeasible/incomplete, the `_ValueState` Feasible/Infeasible/Incomplete
  classification, single-/multi-study labels, and the `error_bar` aggregate over `np.mean`/`np.std`),
  `_edf.py` (`Edf.GetInfo` — complete-trial filter, `np.linspace(min,max,100)` x-grid, and the
  `sum(values[:,None] <= x, axis=0)/size` EDF counts), and `_intermediate_values.py`
  (`IntermediateValues.GetPlotInfo` — sorted intermediate values + constraint feasibility). The `_*Info`
  NamedTuples become **public records** (`OptimizationHistoryInfo`/`ValuesInfo`/`ValueState`/`EdfInfo`/
  `EdfLineInfo`/`IntermediatePlotInfo`/`TrialInfo`) — the data-layer contract the Tier X rendering package
  reads across the assembly boundary. Added bit-exact numpy primitives `MathUtil.Linspace`/`Mean`/`Std`
  (population std, `x*x` residuals + `PairwiseSum`, matching NumPy's `add.reduce` order). **Bit-exact vs
  live Optuna 5.0.0.dev** — these paths touch no `erf` and no LAPACK, and the parity objectives use `x*x`
  (not scalar `x**2`) so the stored trial values are byte-identical too; every value/std/EDF-count/x-grid
  matches byte-for-byte. 15 unit tests (constructed-study builders + the `MathUtil` helpers) + 13 live
  parity tests (optimization-history single/constraints/error-bar-multistudy, EDF single/multi-study,
  intermediate-values with mixed feasibility). Suite: **433 unit + 182 interop**, both TFMs, none skipped.
- ✅ **3.1 part 2** — `feat(viz): Phase 3.1 part 2 — hypervolume-history + param-importances data layer`.
  Two more `_get_*_info` builders. `_hypervolume_history.py` → `HypervolumeHistory.GetInfo` (the incremental
  running-hypervolume over completed feasible trials — sign-flip to minimization, the weak-domination skip,
  the out-of-reference skip, `np.prod(ref - v)` box volume, and the inclusion-exclusion correction via
  `Internal/Hypervolume.ComputeHypervolume` on `np.maximum`-limited solutions; the two `plot_*` argument
  validations — MO-required + reference-dimension `ValueError`s — folded into `GetInfo`). **Byte-exact vs
  live Optuna for 2 objectives** (compute_hypervolume rides its dot-free 2-D path); ≥3-obj inherits the 2.10
  compute_hypervolume machine-eps tolerance. `_param_importances.py` → `ParamImportances.GetInfo`/`GetInfos`
  (reuses `ImportanceFunctions.GetParamImportances`; `dict(reversed(items))` → ascending; the
  `f"{v:.2f}"`/`"<0.01"` labels; the SO-vs-MO / `metric_names` `_get_importances_infos` dispatch, one info
  per objective). Importance values inherit PED-ANOVA's **erf/ucrt few-ULP tolerance** (1e-10 in the parity
  tests); the param names, ordering, `target_name` and the formatted labels are exact. Info records
  `HypervolumeHistoryInfo`/`ImportancesInfo` are public. 7 unit tests (hand-computed HV sequence [3,5,6],
  infeasible-carry, both `ValueError` guards, reverse/label parity vs `ImportanceFunctions`, MO one-info-per-
  objective, empty-study) + 6 live parity (HV 2-obj minimize + mixed-direction-with-constraints, importances
  SO × 2 seeds + MO). Suite: **440 unit + 188 interop**, both TFMs, none skipped.
- ✅ **3.1 part 3** — `feat(viz): Phase 3.1 part 3 — slice + pareto-front data layer`. `_slice.py` →
  `SlicePlot.GetPlotInfo`/`GetSliceSubplotInfo` (per-parameter x [external values]/y [target]/trial-numbers/
  feasibility, `is_log` via `VisualizationUtils.IsLogScale`, `is_numerical` = "not categorical", categorical
  `x_labels`; `sorted(all_params)` ordinal; the `params=` subset validation → verbatim `Parameter {name}
  does not exist in your study.` `ValueError`; empty-study guard). **Named `SlicePlot`, not `Slice`, to
  avoid the `NumSharp.Slice` type collision** (a real API smell for a NumSharp-based library) — the info
  record stays `SlicePlotInfo`. `_pareto_front.py` → `ParetoFront.GetInfo` (partition COMPLETE trials into
  feasible/infeasible via `trial.constraints` or the deprecated `constraints_func` `FutureWarning`;
  best-trials Pareto front over the done `MultiObjective.GetParetoFrontTrialsByTrials`; `include_dominated`
  non-best set via a `HashSet` "not in"; `_targets` default = `trial.values`, the 2-/3-target inference +
  the four `ValueError` guards; `metric_names`/`"Objective {i}"` target names; `axis_order`). Both **bit-exact
  vs live Optuna** — no erf/LAPACK, trials bit-identical (`x*x` objectives). Info records
  `SliceSubplotInfo`/`SlicePlotInfo`/`ParetoFrontInfo` public; `X` holds unwrapped external values
  (`double`/`long`/`string`/…), matching Python `t.params[p]`. 6 unit tests (slice sort/classify/subset-
  validate/empty, pareto best-non-best-infeasible partition, include-dominated-false, unsupported-objective-
  count `ValueError`) + 5 live parity (slice mixed float/log-float/int/categorical + subset, pareto 2-obj
  with constraints, pareto 3-obj mixed-direction). Suite: **446 unit + 193 interop**, both TFMs, none skipped.
- ✅ **3.1 part 4** — `feat(viz): Phase 3.1 part 4 — contour data layer`. `_contour.py` → `Contour.GetInfo`/
  `GetContourSubplotInfo`/`GetAxisInfo` (the full pure-numpy half — `_contour.py`'s data layer has no plotly
  leak, unlike `_rank.py`). Per-axis `ContourAxisInfo` = name, padded `(min,max)` range, `is_log`/`is_cat`,
  the sorted-unique `indices` (with the padded numeric endpoints prepended/appended for numerical axes) and
  the per-trial `values` (numeric / `str()` / null); `SubContourInfo` = the sparse `(x_i, y_i) → z` grid
  (keyed by the `indices` positions, deduped by study-direction min/max when `target` is null) plus per-trial
  feasibility; `ContourInfo` = sorted params + the 2-param single cell or the N×N param-pair matrix +
  `reverse_scale`. `PADDING_RATIO=0.05`; ordinal param sort; the `Parameter {name} does not exist`
  `ValueError`; the <2-unique/same-param empty-grid warnings. **Bit-exact vs live Optuna** — including the
  **log-axis range** (`Math.Log10`/`Math.Pow` match Python's `math.log10`/`math.pow` here) and the boxed
  int/float/str index lookups. 4 unit tests (two-param padded grid + min/max z dedup, maximize→not-reversed,
  unknown-param `ValueError`, 3×3 grid with empty diagonals) + 2 live parity (float/log-float/categorical,
  9-cell grid, every axis range/indices/values + z-grid + constraints byte-identical). Suite: **450 unit +
  195 interop**, both TFMs, none skipped.
- ✅ **3.1 part 5** — `feat(viz): Phase 3.1 part 5 — rank data layer`. `_rank.py` → `Rank.GetInfo`/
  `GetRankSubplotInfo`/`GetAxisInfo`/`OrderWithSameOrderAveraging` — the pure half. Reproduces the
  tie-averaged ranking (`_get_order_with_same_order_averaging`: each value's rank = the mean of its
  positions in the sorted array, so ties share it — bit-exact regardless of sort tie order since equal
  values occupy a contiguous block), the `color_idxs = raw_ranks/(n-1)` scaling (`[0.5]` for n<2), the rank
  `_AxisInfo` (name/range/`is_log`/`is_cat`, no indices), the per-subplot xs/ys (raw external values)/zs +
  the filtered-trial list, the 2-param single cell vs N×N matrix, and rank's quirks: **no non-finite filter**
  (unlike slice/contour) and **no param sort/dedupe** in the `params=` path (the given order is kept).
  **The one deliberate Tier N boundary:** Optuna's RGB `colors` come from plotly's
  `sample_colorscale("RdYlBu_r")` (its own TODO wants this removed from `_get_rank_info`), so the port
  exposes the pure `ColorIdxs` (the [0,1] colormap input) + a per-trial `Infeasible` mask instead — the RGB
  mapping (and infeasible→grey) is Tier X rendering (**5.2**). **Bit-exact vs live Optuna** on everything
  ported: the global `color_idxs` (compared against Python's own `_get_order_with_same_order_averaging`),
  the zs, the axis ranges, the per-subplot xs/ys/zs/trial-numbers/infeasible, and the per-subplot
  `color_idxs` self-consistency. 5 unit tests (scaled average ranks, tie-sharing, given-param-order kept,
  infeasible flags, unknown-param `ValueError`) + 2 live parity (mixed float/log-float/categorical, 3×3
  grid). Suite: **455 unit + 197 interop**, both TFMs, none skipped.
- ✅ **3.1 part 6** — `feat(viz): Phase 3.1 part 6 — parallel-coordinate data layer`. `_parallel_coordinate.py`
  → `ParallelCoordinate.GetInfo` — the hardest viz builder. Per-axis `DimensionInfo` (label, per-trial
  numeric `values`, range, `is_log`/`is_cat`, `tickvals`/`ticktext`, `has_missing`) with the three dim
  builders: **log** (`log10` transform, integer decade `tickvals` + min/max, `f"{pow(10,x):.3g}"` text),
  **categorical** (an incremental `vocab` mapping choices→codes; numeric-cat sorts the vocab and is flagged
  for the lexsort reorder; the `offset`/`None`-tick handling for missing), and **numerical** (`np.linspace`
  ticks, `has_missing` synthetic below-range value). The MO path (`target=null`, ≥2 objectives) colours by
  **Pareto rank** via `MultiObjective.FastNonDominationRank` over sign-flipped losses, and — when numeric-cat
  params exist — reorders every column by `np.lexsort` (a stable multi-key sort) and recomputes `draw_order`
  with Python's stable `sorted(reverse=True)`. New `MathUtil.FormatG3` reproduces CPython's `f"{x:.3g}"`
  byte-for-byte (.NET `"G3"` with `E`→`e`; validated incl. the round-up-bumps-exponent cases). **SO
  parallel_coordinate is fully bit-exact** (dims/values/ranges/tickvals/ticktext/color_values/feasibility/
  draw_order); **[Misaligned] — the MO `draw_order` rides `np.argsort` over tied Pareto ranks** (numpy's
  unstable quicksort tie order isn't portably reproducible; same class as the 2.8 argsort note) — everything
  else in the MO path is byte-identical, and the port uses a stable argsort + documents it, verifying instead
  that `draw_order` orders the ranks non-increasingly. 4 unit tests (`FormatG3` vs CPython, SO obj+param dims,
  constraints→feasibility, MO Pareto-rank) + 6 live parity (SO mixed float/log/categorical, SO with a
  conditional/missing param, MO Pareto rank — all bit-exact save the noted MO `draw_order`). Suite: **459 unit
  + 203 interop**, both TFMs, none skipped.
- ✅ **3.1 part 7 — COMPLETES Phase 3.1** — `feat(viz): Phase 3.1 part 7 — timeline data layer`. `_timeline.py`
  → `Timeline.GetInfo` (`_get_timeline_info` + `_get_max_datetime_complete` + `_is_running_trials_in_study`):
  per-trial lifetime `TimelineBarInfo` (number, start, complete, state, hovertext, infeasible), the running-
  trial `max_datetime + 1s` bar, the `n_recent_trials` tail slice (with the `plot_timeline` `> 0` `ValueError`
  folded in), the reversed-time warning. The `_make_hovertext`/`_make_json_compatible` helpers land in
  `VisualizationUtils`, backed by a new **`PyJson.DumpsIndented`** (additive) that reproduces
  `json.dumps(indent=2)` byte-for-byte (2-space step, `(",", ": ")` separators, `{}`/`[]` empties; leaves
  reuse the existing Python-float/string writers). **Hovertext is bit-exact vs live Optuna** — it is
  `json.dumps({number, values, params, [user_attrs]}, indent=2)` with `<br>` newlines and touches no
  datetimes, so it matches byte-for-byte across separately-run studies; the parity test compares
  number/state/infeasible/hovertext exactly (datetimes are wall-clock, checked structurally start≤complete).
  6 unit tests (fixed-datetime bars, exact indented-JSON+`<br>` hovertext, `n_recent_trials` slice,
  non-positive `ValueError`, running-trial `complete>start`, empty study) + 2 live parity (hovertext + n_recent
  slice). Suite: **465 unit + 205 interop**, both TFMs, none skipped. **Phase 3.1 done — 13/13 viz builders.**
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
  `_split_complete_trials_multi_objective` (from 2.7; HSSP tie-break wired to `Internal/Hypervolume.cs`
  in **2.10** + `_get_reference_point`) plus the
  **2.8** `_split_trials`/`_split_complete_trials_single_objective`/`_split_pruned_trials`/
  `_split_infeasible_trials` and `default_gamma`/`default_gamma_multiobjective`/`default_weights`
  (bit-exact `np.linspace`).
- **`Internal/SignedRankTest.cs`** — scipy-1.16-exact Wilcoxon signed-rank (exact/permutation/
  asymptotic branches). Done for 2.3.
- **`Internal/MultiObjective.cs`** — `study/_multi_objective.py` port: non-dominated sort,
  `_dominates`, `_fast_non_domination_rank`, `_calculate_nondomination_rank`, 2-D/N-D Pareto front,
  `np.unique(axis=0)` semantics (both `return_inverse` and — added in 2.10 — `return_index`). Landed
  for Phase 0.8 `best_trials`; the `_hypervolume/*` other half is now `Internal/Hypervolume.cs`.
- **`Internal/Hypervolume.cs`** — **new in 2.10 core**. `_hypervolume/wfg.py: compute_hypervolume`
  (2-D/3-D/N-D WFG) + `hssp.py: _solve_hssp` (2-D + n-D greedy). Consumed by the TPE MO split; the
  remaining `box_decomposition.py` (`get_non_dominated_box_bounds`) is deferred to **3.2** with its GP
  consumer. Carries the 2.10 `[Misaligned]` HSSP ≥3-obj zero-contribution tie note.
- **`Samplers/TPESampler.cs`** — **the complete sampler**: numeric kernel + univariate + non-group
  multivariate (2.8), the group-decomposed multivariate `group=true` path (2.9), the MO split via
  `Internal/Hypervolume.cs` (2.10), and — verified in **2.11** — multivariate multi-objective (MOTPE)
  and constrained TPE (`constraints_func`/`set_constraint`, the infeasible split). Nothing in TPE is
  pending.
- **`Internal/QuasiMonteCarlo.cs` + `Internal/SobolDirectionNumbers.cs`** — **new in 2.12**. The
  non-scrambled Sobol' (Gray-code over Joe-Kuo direction numbers, base64-embedded, first 1024 dims) and
  Halton (van der Corput) engines, bit-exact vs `scipy.stats.qmc`. Reuse for any future QMC work; note
  the scrambling gap (needs scipy's compiled Cython scramble over PCG64).
- **`Samplers/QMCSampler.cs`** — **new in 2.12**. Non-scrambled QMC sampler; `scramble=true` throws.
- **`Samplers/BaseGASampler.cs` + `Samplers/Nsgaii/*` + `Samplers/NSGAIISampler.cs`** — **new in 2.13**.
  The GA framework (generation attribution + parent-population cache) and the whole NSGA-II strategy
  stack: elite selection (rank + crowding distance), child generation, `perform_crossover`/
  `perform_mutation`, the 6 crossovers, `BaseMutation`/`PolynomialMutation`, constrained domination.
  **NSGA-III (2.14) reuses all of this** — it only swaps `_elite_population_selection_strategy` for a
  reference-point/niching one. Carries the 2.13 UNDX-QR-sign `[Misaligned]`. `MultiObjective.
  FastNonDominationRank` (0.8) is the ranking engine. `NSGAIIElitePopulationSelectionStrategy.
  RankPopulation` is now `internal static` (Optuna's shared `_rank_population`), consumed by NSGA-III.
- **`Samplers/NSGAIIISampler.cs` + `Samplers/Nsgaiii/*`** — **new in 2.14**. `NSGAIIISampler` (mirrors
  `NSGAIISampler` over the shared GA stack) + `NSGAIIIElitePopulationSelectionStrategy` (reference-point
  niching: default-reference-point generation, filter-inf, ideal-point/ASF normalization, association,
  the `defaultdict` niche sweep). Its hyperplane fit calls `Internal/LinAlg` (2.15). Bit-exact vs Optuna.
- **`Internal/LinAlg.cs` + `Internal/OpenBlasActivation.cs`** — **new in 2.15**, extended in 2.16. `LinAlg`
  is the `double[,]`/`double[]` adapter over NumSharp's `np.linalg`: `Solve`/`MatrixRank`/`Cholesky`/`Eigh`
  delegate to LAPACK via `NumSharp.Interop.OpenBLAS` → **byte-identical to NumPy** (used by NSGA-III and
  CMA-ES's `Eigh`; GP `Cholesky` in 3.2). **2.16 added** `Dot(m,m)`/`Dot(m,v)` (`np.dot` — CBLAS `gemm`),
  `Outer` (`np.outer`), `Norm` (`np.linalg.norm`), all byte-exact. `SolveTriangular` is the one
  still-managed primitive (forward/back substitution — no NumSharp `dtrtrs`); a `Det` (`np.linalg.det`) is
  still to add for warm-start (2.16 part 2). `OpenBlasActivation` is a `[ModuleInitializer]` that
  force-loads the OpenBLAS backend (a bare reference does not trigger its self-registration — keep it).
  **Any future numpy-linalg site should use `LinAlg`/`np.linalg`, not a hand-rolled managed kernel.**
- **`Samplers/CmaEs/{Cma,SepCma,WarmStart,CmaWm}.cs` + `Samplers/CmaEs/ICmaOptimizer.cs` +
  `Samplers/CmaEsSampler.cs`** — the complete CMA-ES stack (2.16). `Cma` (`cmaes.CMA`, part 1), `SepCma`
  (`cmaes.SepCMA`, part 2a) and `CmaWm` (`cmaes.CMAwM`, part 2c) share the `ICmaOptimizer` seam;
  `WarmStart` is `get_warm_start_mgd` (part 2b). All reuse the same numeric idioms (`LinAlg.Dot/Eigh/
  Outer/Norm/Det`, the strided-sequential vs contiguous-pairwise sum rule, scalar `**2`→`Math.Pow`,
  `a/scalar`→elementwise divide) and each serializes its evolving state to self-describing own JSON
  (tag prefix `cma`/`sepcma`/`cmawm`). `CmaEsSampler.InitOptimizer` dispatches to the right backend;
  **all four are ported — none throw `NotSupportedException` any more.** `CmaWm` is the one erf-tolerance
  backend (its `AskEncoded()` returns two vectors; the sampler branches `optimizer is CmaWm` for the
  `x_for_tell` round-trip) — see its `[Misaligned]` note.
- `trials_dataframe` ships as a dependency-free `TrialsTable`; the `Microsoft.Data.Analysis` adapter
  remains Tier X (5.4).

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
  2.7 **erf/ucrt** ~1-3 ULP tolerance on the truncnorm density, which also bounds 2.8 TPE; the
  2.8 **argsort tie order** in `compute_sigmas` (stable vs NumPy's unreproducible x86-simd-sort —
  diverges only on repeated numerical observations in a multivariate study; continuous/categorical
  studies are exact); and the 2.10 **HSSP ≥3-objective zero-contribution tie** on `Internal/Hypervolume.cs`
  (the sequential dot cancels a dominated point's marginal HV to exactly 0 where Optuna's BLAS `np.dot`
  leaves a sub-ULP residual, so `np.argmax` can pick a different but hypervolume-equivalent point — 2-obj
  HSSP is dot-free and exact; does not surface in realistic MO TPE); and the 2.13 **UNDX QR column
  signs** on `Samplers/Nsgaii/Crossovers/UNDXCrossover.cs` (managed Householder QR matches numpy's basis
  magnitudes to ~1e-14 but not LAPACK's per-column signs → a ≥2-parameter UNDX study's values diverge
  O(1) and cascade through selection; still a correct, reproducible UNDX; 1-parameter UNDX and the other
  5 crossovers are bit-exact); and the 2.14 **NSGA-III hyperplane-fit solve** on
  `Samplers/Nsgaii/../Nsgaiii/NsgaiiiLinAlg.cs` (a reciprocal-multiplier LU ~1 ULP from numpy's LAPACK
  `dgesv` on a few percent of the small extreme-point systems — a theoretical niche-selection tie-break
  risk that flipped nothing across the 400-config differential, incl. 4×4 fits; NSGA-III is effectively
  bit-exact); and the 2.16 **CMAwM margin correction** on `Samplers/CmaEs/CmaWm.cs` (scipy's
  `norm.cdf`→`Erf.NormCdf` and `chi2.ppf(q,df=1)=Φ⁻¹((1+q)/2)²`→`TruncNorm.NdtriExp`, both the erf
  class ~1-3 ULP; the drift is confined to `mean[discrete]`/`_A`, so the continuous dims stay bit-exact
  and the discrete dims snap to an exact grid — an *encoding-boundary* tie is the only way it could
  surface, and it flipped nothing across the 40-config/1040-trial differential: every discrete param +
  `cmawm:generation` byte-exact, floats to 1.7e-13). The TPE divergences are on
  `Samplers/Tpe/ParzenEstimator.cs`/`TruncNorm.cs`.

### Build / test notes for the next agent

- **Build:** `dotnet build SciSharp.OptunaSharp.sln -c Release -p:SkipNDScopeWeave=true
  -p:GeneratePackageOnBuild=false`. NumSharp resolves as sibling `ProjectReference` → NuGet fallback
  (see main §"NumSharp reference" in CLAUDE.md). A new MSBuild property **`NumSharpCoreProject`**
  (settable via env var of the same name) pins an alternate NumSharp checkout — added because the
  sibling working tree was temporarily mid-edit and wouldn't compile during this work. **If a build
  fails *inside* `NumSharp.Core`, that's the sibling checkout, not OptunaSharp** — point
  `NumSharpCoreProject` at a good NumSharp worktree/commit, or delete the sibling to force the NuGet
  fallback.
- **Test:** `dotnet test test/OptunaSharp.Tests/...` (393 unit, both TFMs) and
  `dotnet test test/OptunaSharp.Tests.Interop/...` (151 live-Optuna tests; self-skip Inconclusive
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
| N | `OptunaSharp` (Core) | distributions, trial, study, ALL samplers (Random/Grid/PartialFixed/BruteForce/TPE/QMC/NSGA-II/NSGA-III/CMA-ES/**GP**), all 9 pruners, transform, MO machinery, PED-ANOVA, terminators (incl. GP-based), InMemory + **journal-file** storage, filesystem artifacts, viz data layer, `eigh`/`cholesky`/`solve` via `np.linalg` (LAPACK), L-BFGS-B | **NumSharp.Core + NumSharp.Interop.OpenBLAS** (see the ⚠️ dependency-model note in §0) |
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

## Phase 2 — Native features, easiest first — ✅ COMPLETE (2.1–2.16 done)

Status column: ✅ done · 🟩 partially landed · ⬜ not started (see §0 ledger).

| # | St | Feature | Optuna source | py SLOC | est. C# | Difficulty | Depends on |
|---|---|---|---|--:|--:|---|---|
| 2.1 | ✅ | Pruners batch 1: `Nop`, `Threshold`, `Patient`, `Median`, `Percentile` | `pruners/*` | ~300 | 700 | 🟢 trivial | 0.5, 0.8 |
| 2.2 | ✅ | Pruners batch 2: `SuccessiveHalving` (ASHA), `Hyperband` | `pruners/_successive_halving.py`, `_hyperband.py` | ~250 | 650 | 🟢 | 2.1 |
| 2.3 | ✅ | `WilcoxonPruner` | `pruners/_wilcoxon.py` | ~100 | 300 | 🟢 | self-contained signed-rank test (`Internal/SignedRankTest.cs`, no scipy) |
| 2.4 | ✅ | `PartialFixedSampler`, `GridSampler` | `samplers/_partial_fixed.py`, `_grid.py` | ~400 | 900 | 🟢 | 0.7 |
| 2.5 | ✅ | `BruteForceSampler` | `samplers/_brute_force.py` | ~480 | 1,000 | 🟢+ | tree enumeration over dynamic spaces; the categorical-NaN quirk (`_is_nan`); bit-exact `rng.choice(p=)` via NumPy pairwise-sum + `searchsorted(side='right')`. `758189f` |
| 2.6 | ✅ | Terminator framework + non-GP evaluators (`StaticError`, `BestValueStagnation`, `MedianError`, `CrossValidationError`) | `terminator/*` minus regret/EMMR | ~300 | 700 | 🟢 | `OptunaSharp.Terminator` namespace; regret/EMMR shipped as GP placeholders here, **ported in 3.3**; `np.var`/`sys.float_info` sentinels bit-exact. `cc7c6e2` |
| 2.7 | ✅ | Importance framework + **PED-ANOVA** | `importance/__init__.py`, `_ped_anova/*` | ~350 | 800 | 🟢+ | pure-numpy Parzen; NumSharp throughout. `_truncnorm._log_gauss_mass` + `probability_distributions.log_pdf` landed as byproducts. **erf/ucrt tolerance** (`[Misaligned]`, see §0) |
| 2.8 | ✅ | **TPE — the default sampler (univariate + non-group multivariate)** | `_tpe/_erf.py`, `_truncnorm.py`, `parzen_estimator.py`, `probability_distributions.py`, `sampler.py` core | ~800 | 2,600 | 🟡 | 0.4. Full numeric kernel + sampler; `Expm1`/`LogAddExp` bit-exact, `ppf`/density erf-tolerance-bounded. Trajectory parity vs live Optuna (continuous/log/categorical multivariate + independent). **Two `[Misaligned]`: erf/ucrt few-ULP, and stable-vs-x86-simd-sort argsort tie order on repeated numerical obs.** `constant_liar`+deprecated args done |
| 2.9 | ✅ | TPE stage 2 — `group` (group-decomposed multivariate) | `_tpe/sampler.py` group branches | ~150 | 300 | 🟡 | `infer_relative_search_space`/`sample_relative` group branches over the already-ported `GroupDecomposedSearchSpace`; experimental warning; MO-default override. Branching-space trajectory parity vs live Optuna (erf tolerance). `66064a4` |
| 2.10 | 🟩 | **MO machinery**: non-dominated sort, WFG hypervolume, HSSP, box decomposition, Pareto front | `study/_multi_objective.py`, `_hypervolume/*` | ~400 | 1,100 | 🟡 | **core done**: `MultiObjective.cs` (0.8) + `Hypervolume.cs` (`compute_hypervolume`+`_solve_hssp`, `c7bd648`) → MO TPE split unblocked. **box_decomposition deferred to 3.2** (GP-only consumer) |
| 2.11 | ✅ | TPE stage 3 — MOTPE + constraints | `_tpe/sampler.py` MO split | ~150 | 0 | 🟡 | **no new code** — multivariate MO + constrained TPE already compose from 2.7–2.10 + the constraint infra. Verified vs live Optuna (mv MO 4.6e-14; constrained SO/MO 7.1e-14, infeasible split exercised). `9a6ffe1` |
| 2.12 | 🟩 | **QMC sampler** (Sobol + Halton, scrambling) | `samplers/_qmc.py` + scipy behavior | ~172 | 1,600 | 🟡 | **non-scrambled done** — Sobol'+Halton bit-exact vs scipy, byte-exact sample-id key (`217af06`). Sobol numbers from `rustuna .../direction_numbers.rs`. **Scrambling deferred** (scipy's compiled Cython scramble over PCG64) |
| 2.13 | ✅ | **NSGA-II** (6 crossovers, polynomial mutation, elite selection, constraints) | `samplers/nsgaii/*`, `_ga/_base.py` | ~1,000 | 2,400 | 🟡 | 2.10. Default + blx/sbx/vsbx/spx **bit-exact** vs live Optuna (full state); UNDX ≥2-D `[Misaligned]` (QR column signs). Crossover object-array → `object?[]`. `NSGAIISampler`+`BaseGASampler`+`Samplers/Nsgaii/*` |
| 2.14 | ✅ | NSGA-III (reference points, niching) | `samplers/_nsgaiii/*` | ~390 | 950 | 🟡+ | 2.13. Reference-point niching over the reused `BaseGASampler`/crossovers/mutations. **Bit-exact** vs live Optuna — 400-config offline differential (0 divergence) + 11 live parity seeds. Reciprocal-multiplier LU `Solve`/`MatrixRank` (`NsgaiiiLinAlg`); numpy association `einsum` replicated term-by-term; `rng.shuffle` via index permutation. The flagged managed-linalg tolerance point did not surface |
| 2.15 | ✅ | Numeric primitives for 2.16/3.2: `eigh`, `cholesky`, `solve`, `matrix_rank`, `solve_triangular` | `Internal/LinAlg.cs` + `OpenBlasActivation.cs` | — | 250 | 🟡 | **Redirected to OpenBLAS** (maintainer decision): Core references `NumSharp.Interop.OpenBLAS`; `LinAlg` delegates cholesky/eigh/solve/matrix_rank to `np.linalg` → **byte-identical to NumPy** (verified 90-case sweep). `solve_triangular` stays managed (no NumSharp `dtrtrs`). NSGA-III solve retro-upgraded to byte-exact. `[ModuleInitializer]` force-load required |
| 2.16 p1 | ✅ | **base CMA-ES** — `cmaes.CMA` (CMA + lr-adapt) + `CmaEsSampler` (default/lr_adapt/x0/sigma0/popsize/consider_pruned/startup) | `samplers/_cmaes.py` + `cmaes/_cma.py` (v0.13.0) | ~900 | 2,300 | ✅ | **byte-exact** vs cmaes 0.13.0 as predicted (`eigh`→`LinAlg.Eigh`). Optimizer state → own JSON (`[Misaligned]`). Fixes: upper-triangle covariance symmetry across round-trip; `y_w` sequential (F-contiguous axis sum) |
| 2.16 p2a | ✅ | **SepCMA** (`use_separable_cma`) — diagonal-covariance separable CMA-ES behind `ICmaOptimizer` | `cmaes/_sepcma.py` | ~350 | 550 | ✅ | **byte-exact** vs cmaes 0.13.0 (11-config differential). 1-D falls back to CMA + `UserWarning` |
| 2.16 p2b | ✅ | **Warm-Starting** (`source_trials` → `get_warm_start_mgd`) | `cmaes/_warm_start.py` | ~120 | 300 | ✅ | **byte-exact** vs cmaes 0.13.0 (4-config differential). Added `LinAlg.Det` (byte-exact) |
| 2.16 p2c | ✅ | **CMAwM** (`with_margin`) — CMA-ES with margin for discrete dists | `cmaes/_cmawm.py` | ~320 | 500 | 🟠 | `Samplers/CmaEs/CmaWm.cs` (wraps `Cma`; two-vector `AskEncoded()`; margin correction). Rides scipy `chi2.ppf(df=1)=Φ⁻¹((1+q)/2)²`/`norm.cdf` → erf `[Misaligned]`, but the drift stays inside `mean[disc]`/`_A` (continuous dims bit-exact, discrete snap to exact grid). **40-config/1040-trial differential: discrete+generation byte-exact, floats 1.7e-13.** `ask()`→2 vectors, sampler stores `x_for_tell` |

**Phase 2 ≈ 7.7k py → ~20.1k C#. Estimate: 14–17 weeks.**
Milestone after 2.11 — **✅ REACHED (M3): "Optuna default experience" complete** — TPE (univariate/
multivariate/group/MOTPE/constrained) + MedianPruner, parity-verified against Python (bit-exact off
the erf path, erf-tolerance on the truncnorm density), persisted to a journal file Python can open.
Zero external deps.

## Phase 3 — Native heavy tail — ✅ COMPLETE (3.1 viz data layer; 3.2 GPSampler; 3.3 GP terminators)

| # | Feature | Optuna source | py SLOC | est. C# | Difficulty | Notes |
|---|---|---|--:|--:|---|---|
| 3.1 | Visualization **data layer** | `visualization/_*.py` info builders | ~1,500 | 2,500 | ✅ | **COMPLETE** — all 13 `_get_*_info` builders in `OptunaSharp.Visualization` (optimization-history/EDF/intermediate-values/hypervolume-history/param-importances/slice/pareto-front/contour/rank/parallel_coordinate/timeline + `_utils`), bit-exact/erf-tolerance vs live Optuna. rank RGB→5.2, MO PC draw_order `[Misaligned]`, `_terminator_improvement`→3.3. Rendering `plot_*`→5.2 |
| 3.2 | **GP stack, native path**: `gp.py` (Matérn-5/2 + MLL fit via hand-derived gradients), `acqf.py` (LogEI/qLogEI/LogPI), `optim_mixed`, L-BFGS-B (managed port of scipy's `__lbfgsb.c` — NOT "already numpy"; see §0), `prior`, `search_space` → `GPSampler` | `_gp/*`, `samplers/_gp/sampler.py` | 1,766 | 8,500 | ✅ | **✅ COMPLETE — the entire `GPSampler` is ported. Gate GREEN & CLOSED (parts 1–2):** prior + kernel + MLL + **hand-derived gradient** (~1e-13 vs autograd) + **managed L-BFGS-B** (`Internal/LBfgsB.cs`, port of scipy `__lbfgsb.c`) + `FitKernelParams` (~1e-14 vs Optuna). Native confirmed; tolerance-exact via 2.15 cholesky/solve_triangular + hand-written BLAS. **Parts 3a–3d done — single-objective `GPSampler` runs end-to-end:** `GpSearchSpace` + `Posterior`; LogEI eval+gradient (`Gp/Acqf.cs`); scrambled Sobol' (`Internal/ScrambledSobol.cs`, bit-exact vs scipy — NumSharp PCG64 matches numpy); `Gp/OptimMixed.cs` (`optimize_acqf_mixed` + gradient ascent + exhaustive/Brent discrete search; numpy `choice(replace=False,p=)` reproduced) + `Internal/BrentMinimizer.cs` + `Samplers/GPSampler.cs`. Tolerance-parity vs live Optuna: `optimize_acqf_mixed` achieved-value <1e-6; per-suggestion GP replay <1e-2 normalized (discrete exact). **Parts 3e-1…3e-5 done — every sequential `GPSampler` path runs end-to-end:** box decomposition (`Internal/Hypervolume.cs`, bit-exact); `LogPI`+`LogCEI` (SO constrained); `qmc.sample_from_normal_sobol` (`Gp/Qmc.cs`+`Internal/TorchMt19937.cs`+cephes `TruncNorm.Ndtri` — torch's scrambled Sobol' reproduced to ~1e-15); `LogEHVI` (MO); `LogCEHVI` (MO constrained). Each: eval+hand-derived-gradient parity vs Optuna autograd + per-suggestion sampler replay. **Part 3e-6 done — every running-trial path runs end-to-end:** `ConditionalGPRegressor` (`Gp/ConditionalGPRegressor.cs`: `sample_joint_posterior`/`append_running_data`/`_extend_cholesky`, forward + query-sample gradient) + `qLogEI`/`qLogPI`/`qLogCEI`/`qLogEHVI` (`Gp/Acqf.cs`, the `amax`/`logsumexp` reduction gradients hand-derived) + the Kriging-Believer `LogCEHVI` for MO-constrained-with-running. Per-suggestion replay uses an ask/tell schedule that leaves trials RUNNING before the trial under test (RNG in lock-step). Suite: **533 unit + 252 interop**, both TFMs |
| 3.3 | GP terminators: `RegretBoundEvaluator` (default), `EMMREvaluator` | `terminator/improvement/*` | ~230 | 700 | ✅ | **COMPLETE** — both evaluators un-thrown (`Terminator/{ImprovementEvaluator,EmmrEvaluator}.cs`); new `UCB`/`LCB` acqfs (`Gp/Acqf.cs`) + `Gp/OptimSample.cs` (`optimize_acqf_sample`); deferred `_get_improvement_info` viz wired (`Visualization/TerminatorImprovement.cs`). Rides the 3.2 GP stack → tolerance parity vs live Optuna (few-1e-3, asserted 3e-2 — the GP boundary). +11 unit +7 interop. Rendering `_get_improvement_plot`→5.2 |

**Phase 3 ≈ 3.5k py → ~11.7k C#. Estimate: 10–13 weeks (GP is 8–10).**
Exit gate = **Tier N complete**: the full Optuna algorithm surface, pure managed, NumSharp-only.

---

# TIER X — external dependencies & integrations (last)

## Phase 4 — External storage & transport — 🟩 IN PROGRESS (Tier X) — 4.5 cloud artifacts DONE; 4.1–4.4 pending

| # | Work item | Optuna source | py SLOC | est. C# | Package | Notes |
|---|---|---|--:|--:|---|---|
| 4.1 | **RDB storage on SQLite, Optuna-compatible schema** | `storages/_rdb/*` | 2,339 | 3,000 | `.Storage.Sqlite` | hand SQL on Microsoft.Data.Sqlite mirroring the SQLAlchemy models + `alembic_version` pin so Python opens our `.db`. Reference: `rustuna_storage/sqlite3.rs`. **Interop gate:** Python↔C# same-file round-trip. Reuses 1.2 cache + 0.6 conformance suite |
| 4.2 | EF Core providers (PostgreSQL/MySQL) | (schema from 4.1) | — | 1,500 | `.Storage.EFCore` | conformance suite + testcontainers |
| 4.3 | Redis journal backend | `storages/journal/_redis.py` | ~200 | 450 | `.Storage.Redis` | thin: 1.1 already carries the journal semantics |
| 4.4 | gRPC storage proxy (client + server) | `storages/_grpc/*` | 1,719 | 2,400 | `.Storage.Grpc` | reuse Optuna's `.proto` verbatim; wire-compat gate: C# client ↔ Python `run_grpc_proxy_server` |
| 4.5 | Cloud artifacts (S3, GCS) + **AzureBlob/R2 extensions** | `artifacts/_boto3.py`, `_gcs.py` | ~100 | 300 | `.Artifacts.{S3,Gcs,AzureBlob,R2}` | ✅ **DONE** (feat `3923fcd`) — 4 satellite pkgs on the 1.4 `IArtifactStore`. `Boto3ArtifactStore` (AWSSDK.S3 4.0.102.4) + `GCSArtifactStore` (Google.Cloud.Storage.V1 4.15.0) **port Optuna**; `AzureBlobArtifactStore` (Azure.Storage.Blobs 12.29.2 + Azure.Identity 1.21.0) + `R2ArtifactStore` (Cloudflare R2 — S3-compatible, reuses `.S3`) are **extensions beyond Optuna** (no live-interop parity). 15 mock unit tests + 8 dockerized emulator integration tests (MinIO for S3/R2, Azurite, fake-gcs-server; self-skipping via `docker-compose.yml`) ×2 TFMs; strong-name + recorded-deps verified. CI wiring of the new test project + packages pending |

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
