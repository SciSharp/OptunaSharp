# OptunaSharp Expansion Plan — SOTA optimizers to add

> **Purpose.** A phase-by-phase plan for porting a curated set of state-of-the-art black-box /
> hyperparameter optimizers into OptunaSharp. Each method below was surfaced by a citation-ranked
> paper search (OpenAlex + arXiv via the `paper-gateway` skill), its source paper **fetched, converted
> and read**, and an integration design written against OptunaSharp's actual abstractions.
>
> **Relationship to the governing roadmap.** This document is *additive* to
> [`docs/PORTING_PLAN.md`](../docs/PORTING_PLAN.md) (the authoritative Optuna-parity roadmap). Every
> method here is either an **Optuna-parity gap** (matches an Optuna 5.x feature we are missing) or a
> **beyond-core, convergence-validated** sampler with **no Optuna counterpart** — the same category as
> the already-shipped LM-CMA / VkD-CMA / SMS-EMOA / AGE-MOEA-2 / MOEA/D / RVEA / PSO / DE / JANUS
> samplers. It follows the same conventions as [`docs/AGE_MOEA_2_PORTING.md`](../docs/AGE_MOEA_2_PORTING.md)
> and [`docs/BOTORCH_PORTING.md`](../docs/BOTORCH_PORTING.md).

---

## 0. Verdict up front

The candidate set was filtered against what OptunaSharp **already ships** (see §2 gap analysis). Two
"candidates" turned out to be **already implemented** and are dropped. The rest split across four
parity tiers and two homes (pure-managed **Core** vs a **satellite bridge package**):

| Method | Tier | Home | Parity target | Priority |
|---|---|---|---|---|
| **OpenAI-ES** (isotropic NES) | Beyond-core | Core | Behavioral (+ optional bit-parity vs pinned numpy ref) | **P1 ✅ DONE** |
| **SNES / xNES** (separable / exponential NES) | Beyond-core | Core | Behavioral | **P1 ✅ DONE** |
| **REMBO** (random-embedding wrapper) | Beyond-core | Core | Behavioral | **P2 ✅ DONE** |
| **TuRBO-1** (trust-region BO) | Beyond-core | Core | Achieved-value + per-step replay tolerance | **P2 ✅ DONE** |
| **TuRBO-m** (multi-TR bandit) | Beyond-core | Core | Achieved-value + per-step replay tolerance | P2 (follow-on) |
| **SAASBO** (fully-Bayesian sparse GP) | Optuna-integration analog | **Bridge** (`Interop.BoTorch`) | **0-ULP vs `optuna_integration`/BoTorch** | P3 |
| **MAP-Elites** (quality-diversity) | New paradigm | Core (+ new QD surface) | Behavioral (archive coverage / QD-score) | P4 |
| **MO-CMA-ES** (multi-objective CMA) | Beyond-core | Core | Behavioral | P5 (paper access gap — see §9) |
| ~~CMA-ES-LRA~~ | — | — | **ALREADY DONE** — `CmaEsSampler(lrAdapt: true)` | — |
| ~~Hyperband / BOHB~~ | — | — | **ALREADY DONE** — `HyperbandPruner` + `SuccessiveHalvingPruner` (BOHB = TPE + Hyperband recipe) | — |

**Two load-bearing rules inherited from the plan** (`docs/PORTING_PLAN.md` §2) that every item here
must respect:

- **RNG discipline.** All stochastic samplers consume `np.random.RandomState`/`Generator` through the
  `LazyRandomState` port (MT19937 / PCG64). **Never `System.Random`.** This is what makes any
  bit-parity claim testable against a reference.
- **Core stays pure-managed (NumSharp only).** Anything needing a foreign runtime (torch, pyro/NUTS,
  JAX) is a **separate satellite package** — the `NumSharp.Interop.OpenBLAS` pattern. This is why
  SAASBO is a **bridge**, not a Core sampler (see §6).

---

## 1. Reference sources (fetched & read)

All papers were fetched with the `paper-gateway` CLI and converted into the local collection
`docs/papers/expansion-optimizers/library/<conversion-id>/paper.md` (git-ignored PDFs under
`docs/papers/paper-gateway/downloads/`). Regenerate with
`gateway.py fetch <id> --convert --library docs/papers/expansion-optimizers/library`.

| Method | Paper | Identifier | Local conversion | Read |
|---|---|---|---|---|
| OpenAI-ES | Salimans et al., *Evolution Strategies as a Scalable Alternative to RL* (2017) | arXiv:1703.03864 | `16ed1ee42f14c9d7` | **full** |
| NES / SNES / xNES | Wierstra et al., *Natural Evolution Strategies* (JMLR 2014) | arXiv:1106.4487 | `4d8153cf01929bb3` | framework + algorithm boxes |
| TuRBO | Eriksson et al., *Scalable Global Optimization via Local Bayesian Optimization* (NeurIPS 2019) | arXiv:1910.01739 | `9327fd2ca2c8c350` | §2 method |
| SAASBO | Eriksson & Jankowiak, *High-Dim BO with Sparse Axis-Aligned Subspaces* (UAI 2021) | arXiv:2103.00349 | `cc5d076ae6f73ab7` | §4 prior + inference |
| REMBO | Wang et al., *Bayesian Optimization in a Billion Dimensions via Random Embeddings* (JAIR 2016) | doi:10.1613/jair.4806 | `e26c42a723a9daac` | §3 embedding |
| MAP-Elites | Mouret & Clune, *Illuminating search spaces by mapping elites* (2015) | arXiv:1504.04909 | `df7344343b14893a` | full algorithm |
| QD framing | Pugh et al., *Quality Diversity: A New Frontier for Evolutionary Computation* (2016) | doi:10.3389/frobt.2016.00040 | `958a7bc41660a461` | context |
| CMA-ES-LRA (already done) | Nomura et al., *CMA-ES with Learning Rate Adaptation* (TELO 2024) | arXiv:2401.15876 | `34308182bd84fa5f` | full algorithm (validation) |
| MO-CMA-ES | Igel et al., *Covariance Matrix Adaptation for Multi-objective Optimization* (Evol. Comput. 2007) | doi:10.1162/evco.2007.15.1.1 | **paywalled — not retrieved** (§9) | — |

**Excluded on purpose — metaphor metaheuristics.** Grey Wolf Optimizer (≈19 000 citations), Whale/
Harris-Hawks/etc. are the field's cautionary tale: extraordinary "outperforms everything" claims that
the critique literature (Sörensen, *Metaheuristics — the metaphor exposed*, 2015, ≈1000c; Aranha/
Campelo et al., *the elephant in the room*, 2021) shows are re-branded classics. **Do not port these.**
Cite the critiques if asked.

---

## 2. Gap analysis — what OptunaSharp already has (do NOT re-plan)

Verified against the current tree before writing this plan:

- **Samplers (Core):** Random, Grid, TPE, CMA-ES, GP, QMC, BruteForce, NSGA-II/III, PartialFixed;
  beyond-core LM-CMA, VkD-CMA, DE, PSO, SMS-EMOA, AGE-MOEA-2, MOEA/D, RVEA, JANUS family, + CPT variants.
- **CMA-ES-LRA is already implemented:** `CmaEsSampler(lrAdapt: true)` →
  `Samplers/CmaEs/Cma.LrAdaptation(...)` + `CmaState.LrAdapt` (the Nomura 2024 SNR learning-rate
  adaptation). The paper was still read to *confirm* our implementation matches Algorithm 1
  (one-step diffs → local Fisher coordinates → EMA `E,V` → `SNR = αη` feedback → `η ← min(η,1)`).
- **Bandit / multi-fidelity is already covered by pruners:** `HyperbandPruner`,
  `SuccessiveHalvingPruner`, `MedianPruner`, `PercentilePruner`, `PatientPruner`, `ThresholdPruner`,
  `WilcoxonPruner`. **BOHB** is not a separate object — it is the **TPESampler + HyperbandPruner**
  usage recipe (as in Optuna). Add a Recipes/docs page rather than code.
- **Bayesian optimization:** native `GPSampler` (GP stack in `Internal/LinAlg` + `Internal/Gp`,
  OpenBLAS Cholesky/eigh) **and** the `OptunaSharp.Interop.BoTorch` bridge (LogEI/qEI/qLogEI/qNEI +
  MO qEHVI/qNEHVI/qParEGO). **No SAASBO** in either → §6.

**Net new work = the seven rows in §0.** SMAC (random-forest surrogate BO) is deliberately **out of
scope**: Optuna itself has no SMAC sampler, and we already have a native RF (the fANOVA/MDI forests in
`Core/Importance`) if a future beyond-core SMAC is ever wanted.

---

## 3. Phase 1 — Natural Evolution Strategies (OpenAI-ES, SNES, xNES)

> **✅ DONE.** All three landed in Core under `Samplers/EvolutionStrategy/`: the numeric cores
> (`OpenEsOptimizer` / `SnesOptimizer` / `XnesOptimizer` behind `IEvolutionStrategyOptimizer`), the shared
> fitness-shaping (`EsFitnessShaping`: centered ranks + NES utilities) and deterministic per-generation
> Gaussian sampling (`EsSampling`), and the shared Optuna-protocol base `EvolutionStrategySampler` driving
> the public `OpenAiEsSampler` / `SnesSampler` / `XnesSampler`. Each mirrors the `CmaEsSampler` ask/tell
> idiom (reconstruct-from-x, chunked state in system attrs, independent-sampler fallback). Validated by 46
> unit tests (both TFMs) — convergence, same-seed determinism, state round-trip, antithetic pairing — plus
> a live-numpy parity gate (`EvolutionStrategyNumpyParityTests`) for the natural-gradient math. Deep-dive
> docs at `methods/{openai-es,snes,xnes}.md`. **Design notes worth carrying forward:** σ is annealed
> (estool-style) so OpenAI-ES converges in the Optuna regime, but it still settles to a σ-neighborhood
> (SNES/xNES converge tight); `weightDecay` defaults to 0 because the transformed box's origin is a corner;
> xNES uses `B` consistently in sample and mean-update (the JMLR box's `Bᵀ` in the sample line is a
> transpose slip that only holds for symmetric `B`), and its covariance exponential rides OpenBLAS `eigh`.

**Why.** The extraordinary-claim tier of the search: OpenAI-ES ("scalable alternative to RL", ≈1000c)
is a dead-simple, embarrassingly-parallel, gradient-free optimizer that fills a real gap — OptunaSharp
has CMA-family and DE/PSO, but no **natural-gradient search-distribution** sampler. NES is also the
theoretical parent of CMA-ES, so it rounds out the ES story.

**Home / shape.** Core, beyond-core. Mirror the existing `DifferentialEvolutionSampler` /
`PsoSampler` design (a serialized population optimizer wrapped by a `BaseSampler`, RNG via
`LazyRandomState`, per-trial state persisted in `study.system_attrs`). Proposed:

- `Samplers/EvolutionStrategy/OpenEsOptimizer.cs` — the numeric core.
- `Samplers/OpenAiEsSampler.cs` — the `BaseSampler` wrapper (public API mirrors DE/PSO).
- `Samplers/SnesSampler.cs`, `Samplers/XnesSampler.cs` — thin variants sharing the optimizer +
  utility/fitness-shaping helpers under `Samplers/EvolutionStrategy/`.

### 3.1 Algorithm contract (from the papers)

**OpenAI-ES** (Algorithm 1, arXiv:1703.03864 p.3), isotropic Gaussian `N(θ, σ²I)`, ascent on the mean:

- Sample `ε₁…εₙ ~ N(0,I)`; evaluate `Fᵢ = f(θ + σεᵢ)`; update `θ ← θ + α · 1/(nσ) · Σ Fᵢ εᵢ`.
- **Mirrored/antithetic sampling** (p.4): evaluate pairs `ε, −ε` (halves variance; popsize even).
- **Fitness shaping** (p.4): rank-transform returns before the update (centered ranks in `[-0.5, 0.5]`);
  removes outlier influence and scale dependence. **OptunaSharp minimizes → negate** so higher rank =
  lower objective.
- **σ is fixed** (they saw no benefit adapting it); **weight decay** on `θ`; an **Adam** step on the
  estimated gradient is used by the canonical `estool`/`evosax` `OpenES` implementations.

**SNES** (separable NES, Schaul et al. via Wierstra 2014 §5): per-coordinate `σ` (diagonal), utility
weights, natural-gradient updates with learning rates `η_μ = 1`, `η_σ = (3 + ln d)/(5√d)`. Scales to
high `d` at `O(d)`/sample — the practical NES for hyperparameter spaces.

**xNES** (exponential NES, Wierstra 2014 §4): full covariance via the **exponential map** on the
`(μ, A)` parameterization (`Σ = AᵀA`); closely related to CMA-ES. Lower priority than SNES (we already
have full-covariance adaptation in CMA-ES).

**Shared NES utility function** `uₖ` (rank-based, monotone, sums to the natural-gradient baseline):
`uₖ = max(0, ln(λ/2 + 1) − ln k) / Σⱼ max(0, ln(λ/2+1) − ln j) − 1/λ`, with `λ = 4 + ⌊3 ln d⌋` default.

### 3.2 Integration notes

- **Search space:** continuous only (float params, log/step handled by transform to an internal `z`
  space like CMA-ES does). Fall back to `RandomSampler` for the first trial and for
  categorical/int-only spaces (mirror `CmaEsSampler`'s `_sample_relative` guard).
- **Ask/tell over Optuna trials:** the population is a generation of trials; hold pending members in
  `system_attrs` (serialized `θ, σ, generation, popsize, Adam moments`) exactly as DE/PSO do so
  parallel `n_jobs` and storage round-tripping work.
- **Minimization:** negate the objective for the ascent; keep the public contract "minimize" like
  every other sampler.

### 3.3 Reference oracle & Definition of Done

- **Reference impl:** `evosax` `OpenES`/`SNES`/`xNES` (JAX) and the numpy `estool` `OpenES`. JAX RNG
  is not MT19937, so **default tier = behavioral** (like `LmCmaSamplerTests` / `DifferentialEvolutionSamplerTests`):
  convergence on sphere/Rosenbrock/Rastrigin within calibrated thresholds, **same-seed determinism**,
  and **optimizer-state round-trip fidelity**. *Optional bit-parity* against a pinned **numpy** reference
  we write with `RandomState` (mirrored ε, rank shaping, Adam) — achievable because the whole update is
  MT19937-driven; use the `[Misaligned]` ledger only if erf/log tolerances bite.
- **DoD:** default-param constructor matches the reference's defaults; XML docs on every public member
  (per repo rule); unit tests both TFMs; one `[RequiresPython]` interop convergence sanity check;
  Recipes/docs page. No `System.Random`.

---

## 4. Phase 2a — REMBO (random-embedding meta-sampler)

> **✅ DONE.** Landed in Core as `Samplers/RandomEmbedding/RandomEmbeddingProjection.cs` (the geometry — the
> deterministic per-`(run, name)`-seeded Gaussian embedding matrix `A`, the latent box `Y = [−√d, √d]^d`,
> the round-robin interleaved-run assignment, and the box-projected mapping `x = p_X(A·y)` reusing
> `SearchSpaceTransform` for the `[0,1]→external` de-normalization) + the public `RandomEmbeddingSampler`
> (the meta-sampler: it drives an inner sampler — a `GPSampler` by default, matching the paper — over a
> private in-memory **shadow study** of latent trials, mirroring the real study's finished trials via
> `create_trial`/`add_trial`, `Ask`-ing the latent point and immediately `Tell`-ing the ask-trial `Fail` so
> the inner GP never treats it as a pending point, then mapping back to the real parameters). Every trial is
> embedded through `SampleIndependent` from one cached latent point; single-objective only. **Both §3.2
> kernels are implemented:** the default **low-dimensional** kernel (inner GP over the latent `y`, categoricals
> delegated) and a **high-dimensional** kernel (`Samplers/RandomEmbedding/RandomEmbeddingGpOptimizer.cs`) that
> fits a `GPRegressor` in the real (projected) parameter space and maximizes `Acqf.LogEi` over projected
> Sobol candidate `y`'s — reusing the GP stack unchanged, exactly like `TurboOptimizer`. The high-dimensional
> kernel removes the low-dimensional kernel's boundary over-exploration (validated below) and **embeds
> categoricals** via the §3.2 scale-and-round `s(x)` + `GpSearchSpace`'s categorical kernel. Validated by 34
> unit tests (both TFMs): the embedding geometry (matrix determinism / `N(0,1)` entries / box / clip /
> int+log+step+categorical de-normalization) directly; end-to-end convergence (interleaved & high-dim),
> determinism, bounds, mixed types, TPE inner, latent persistence, MO guard; **and a dedicated claims suite
> proving Theorems 2/3/4/6, Lemma 5, §3.1 (δ^k capture), and §3.2 (high-dim kernel escapes a low-dim stall).**
> Deep-dive doc `methods/rembo.md`. **Design notes worth carrying forward:** a *single* embedding is genuinely
> fragile — a **box miss** (fixed by §3.1 interleaving / larger `d`) and, under the low-dim kernel, **boundary
> over-exploration** (fixed by the high-dim kernel); the embedding matrix is generated per-`(run, name)` from a
> stable FNV+SplitMix seed rather than persisted, so it survives parameter-discovery order / resumption /
> parallel workers, and this makes Theorem 4 (invariance to added dims) and D-independence *exact* identities;
> `nEmbeddingDims` (d) is **required** (no honest default). **Not ported (documented divergence):** only the
> adaptive length-scale scheme (Algorithm 3), which would need bounds injected into the GP fit — the high-dim
> kernel addresses the same clipped-boundary over-exploration from another angle. **Still to do under P2:**
> **TuRBO-m** (the multi-trust-region bandit).

**Why.** The "billion dimensions" extraordinary claim (≈390c). It is the canonical way to run BO on a
high-`D` space that has a low **effective dimension** `dₑ` — common in hyperparameter tuning where only
a few knobs matter.

**Home / shape.** Core, beyond-core, **meta-sampler** (wraps an inner sampler) — architecturally like
`PartialFixedSampler`. Proposed `Samplers/RandomEmbeddingSampler.cs`.

### 4.1 Algorithm contract (Wang 2016 §3, Algorithm 2)

- Draw a fixed random matrix `A ∈ ℝ^{D×d}` with i.i.d. `N(0,1)` entries (`D` = real dims, `d` = embedding
  dim ≥ guessed `dₑ`).
- Optimize `g(y) = f(p_X(A y))` with an inner BO in a **low-dim box** `Y = [−√d, √d]^d` (Theorem 3
  bound → ball radius `√(dₑ)/ε`; `[−√d,√d]^d` is the standard practical choice).
- **Convex projection** `p_X`: the objective's real box is rescaled to `X = [−1,1]^D`; if `Ay` lands
  outside, clip per-coordinate into `X` **before** evaluating (the paper's `p_X(y)=argmin_{z∈X}‖z−y‖`
  is per-dim clamping for a box).

### 4.2 Integration notes

- The study's **real** parameters stay the user's `D` params. REMBO maintains `d` **latent**
  parameters internally; `SampleRelative` proposes `y ∈ Y` via the inner sampler, computes
  `x = clip(A·y)` mapped back through each distribution's `[low,high]` (+ log/step) transform, and
  returns the real params. Persist `A` and the latent history in `system_attrs`.
- **Inner sampler:** default to the native `GPSampler` (matches the paper); allow any `BaseSampler`
  (TPE for a pure-managed, cheaper variant).
- **Successors to note in docs (not porting now):** HeSBO (hashing embedding — TuRBO's high-dim
  baseline) and ALEBO fix REMBO's projection/box pathologies; REMBO is the readable canonical base.

### 4.3 Oracle & DoD

Behavioral: on a `D=100` Branin/Hartmann embedded in a 2–6 effective-dim subspace, REMBO reaches the
optimum in far fewer trials than plain GP on the full `D`. Determinism under fixed seed. Optional
parity vs the reference REMBO (`ceberio`/`Bayesian-optimization` implementations). Full XML docs;
both TFMs.

---

## 5. Phase 2b — TuRBO / TuRBO-m (trust-region Bayesian optimization)

> **✅ TuRBO-1 DONE.** Landed in Core as `Samplers/Turbo/TurboOptimizer.cs` (the TR mechanics — the
> deterministic expand/shrink/restart state fold, the length-scale-weighted TR box, the masked Sobol
> perturbation candidate set, and joint-Cholesky Thompson sampling) + the public `TurboSampler`, driven
> through the Optuna protocol reusing the native GP stack unchanged (`GPRegressor` Matérn-5/2 ARD +
> `PosteriorJoint` + OpenBLAS Cholesky, `GpSearchSpace` normalization, `ScrambledSobol`). Validated by 26
> unit tests (both TFMs): the fold's expand→1.6 / shrink→0.4 / restart-after-7-halvings, the anisotropic
> box, the perturbation mask, Thompson argmax + jitter fallback, and end-to-end convergence — TuRBO beats
> plain `GPSampler` on multimodal Ackley (≈0.87 vs ≈2.23) and matches it on the sphere. 935/935 unit suite.
> Deep-dive doc `methods/turbo.md`. **Design notes:** single-objective, Float/Int only (categorical →
> independent sampler, like CMA); state is RECONSTRUCTED by a deterministic fold over the completed-trial
> value sequence (no serialized blob); Thompson sampling draws the EXACT joint posterior via Cholesky, so
> `numCandidates` defaults to `min(100·d, 500)` (capped below the paper's 5000, which relies on BoTorch's
> scalable sampler) with cost `O(numCandidates³)`/trial; per-restart Sobol init (`2·d` points). **Still to
> do under P2:** **TuRBO-m** (the multi-trust-region bandit) — **REMBO** is now done (§4).

**Why.** The SOTA high-dimensional BO baseline (≈150c and rising; the very baseline **JANUS** cites).
It reuses our existing GP stack, so marginal cost is moderate and it materially strengthens BO on the
1000s-of-evaluations, high-`d` regime where plain `GPSampler` degrades.

**Home / shape.** Core, beyond-core. Proposed `Samplers/Turbo/TurboOptimizer.cs` +
`Samplers/TurboSampler.cs`. Builds directly on `Internal/Gp` + `Internal/LinAlg`.

### 5.1 Algorithm contract (Eriksson 2019 §2)

- **Trust region** = hyperrectangle centered at the incumbent `x⋆` (best observed; smallest posterior
  mean under noise). Base side length `L`; per-dim side `Lᵢ = λᵢ L / (Πⱼ λⱼ)^{1/d}` using GP
  lengthscales `λᵢ`, total volume `L^d`.
- **Candidates by Thompson sampling:** draw posterior realizations from the **local** GP over a
  discrete candidate set inside the TR (Sobol/perturbation set, `n_cand = min(100·d, 5000)`), pick the
  batch-`q` argmin across draws. TuRBO-m draws across `m` TRs simultaneously (implicit multi-armed
  bandit).
- **TR resizing:** after `τ_succ` consecutive successes `L ← min(L_max, 2L)`; after `τ_fail`
  consecutive failures `L ← L/2`; reset counters on change; if `L < L_min` **discard & restart** the TR
  at `L_init`. A *success* = candidate improves on `x⋆`.
- **Defaults** (paper supplement / BoTorch TuRBO tutorial — pin against the tutorial, not the paper
  body): `L_init = 0.8`, `L_min = 2⁻⁷ ≈ 0.0078`, `L_max = 1.6`, `τ_succ = 3`,
  `τ_fail = ⌈max(4/q, d/q)⌉`.

### 5.2 Integration notes

- Reuses `GPSampler`'s kernel/fit path (Matérn-5/2, ARD lengthscales, OpenBLAS Cholesky). The **new**
  code is TR bookkeeping + the Sobol-in-TR candidate set (we have scrambled Halton/Sobol in
  `Internal/ScrambledHalton` + QMC) + Thompson draws (posterior sample = `μ + L·z`, reuse the GP
  Cholesky).
- Start with **TuRBO-1** (single TR); add TuRBO-m behind a `numTrustRegions` ctor arg.
- Categorical/int handling: same guard/fallback as `GPSampler`.

### 5.3 Oracle & DoD

GP + Thompson sampling is stochastic and (like our GP terminators) **not bit-exact vs torch** →
**achieved-value + per-suggestion replay tolerance** tier (see the GP-stack tolerance precedent in the
repo). Validate on the paper's spirit: TuRBO-1 beats plain GP on Ackley-100D / Levy-50D within a fixed
budget; determinism under fixed seed; TR expand/shrink/restart unit-tested directly. Optional replay
parity vs the BoTorch TuRBO tutorial (tolerance). Full XML docs; both TFMs.

---

## 6. Phase 3 — SAASBO (fully-Bayesian sparse-GP BO) — BoTorch bridge

**Why.** SOTA sample-efficient high-dim BO ("outperforms strong baselines"); it is the missing model in
our existing bridge.

**Home / shape — BRIDGE, not Core.** SAASBO's engine is a **fully-Bayesian GP** whose hierarchical
sparsity prior is sampled by **NUTS/HMC** (Eriksson & Jankowiak §4.2, via NumPyro). Re-implementing
NUTS-over-a-GP byte-exact in managed code is infeasible and low-value; the correct move is to **extend
the existing `OptunaSharp.Interop.BoTorch` pythonnet bridge** — which already drives real torch/botorch
at **0-ULP vs `optuna_integration`**.

### 6.1 Algorithm contract (Eriksson & Jankowiak §4.1–4.3)

- **SAAS prior:** `σ²_k ~ LogNormal(0, 10²)` (kernel variance); global shrinkage `τ ~ HalfCauchy(α)`,
  **default `α = 0.1`**; per-dim inverse-squared lengthscales `ρᵢ ~ HalfCauchy(τ)` (ARD → most dims
  "off" unless data turns them on); RBF or Matérn-5/2 kernel; noise `σ² → 1e-6` (noise-free) or a weak
  `LN(0,10²)` prior.
- **Inference:** NUTS → `L` posterior hyperparameter samples (BoTorch default 256 warmup / 256 samples,
  thinning 16); or MAP over pre-selected `τ ∈ {1e-1,1e-2,1e-3}`. Cost `O(N³D)`, best for `N ≲ 500`.
- **Acquisition:** EI (single-obj) / qLogEI over the sampled-hyperparameter GP; qLogNEHVI for MO.

### 6.2 Integration notes

- Add `SaasFullyBayesian` to `BoTorchCandidatesFunc` (enum) + a `saas_candidates_func` in
  `BoTorchCandidates.cs` that builds `SaasFullyBayesianSingleTaskGP`, runs
  `fit_fully_bayesian_model_nuts`, then `qLogExpectedImprovement` (single) / `qLogNEHVI` (MO), mirroring
  the existing candidate funcs. Expose NUTS knobs (`warmup_steps`, `num_samples`, `thinning`) on
  `BoTorchSampler`.
- Public API stays pythonnet-free (enum + sampler options), exactly like the current bridge.

### 6.3 Oracle & DoD

**0-ULP vs `optuna_integration.BoTorchSampler` with a SAAS candidates func** (the bridge's existing
byte-exact contract) — cached-module discipline, `PYTHONNET_PYDLL`, try-import guard. `[RequiresPython]`
interop tests both TFMs. Full XML docs.

---

## 7. Phase 4 — MAP-Elites (Quality-Diversity) — new paradigm

**Why.** A genuinely different objective: not "the single best", but an **archive of the best solution
per behavior niche" (illumination). No Optuna analog. High-value for design-space exploration and as the
substrate for CMA-ME and downstream JANUS-style MO work.

**Home / shape — Core, beyond-core, requires new study-level surface.** MAP-Elites does **not** fit the
plain `Study.best_trial` contract; it needs (a) a user **behavior/feature descriptor** per trial and (b)
an **archive** accessor. This is the most API-invasive item — scope it deliberately.

### 7.1 Algorithm contract (Mouret & Clune 2015, Fig. 2)

```
P ← ∅ (performances), X ← ∅ (solutions)        # N-dim discretized feature map
for iter = 1 → I:
    if iter < G:  x' ← random_solution()        # G random bootstrap solutions
    else:         x' ← random_variation(random_selection(X))   # mutate a uniformly-picked elite
    b' ← feature_descriptor(x');  p' ← performance(x')
    if P(b') empty or P(b') < p':  P(b') ← p';  X(b') ← x'      # keep the niche elite
return (P, X)
```

### 7.2 Integration notes

- **New surface:** a `MapElitesSampler` plus a `QualityDiversityArchive` result object, and a
  descriptor delegate `Func<FrozenTrial, IReadOnlyList<double>>` (behavior coordinates) with a
  discretization spec (per-dim bin edges). Retrieve the archive off the study (e.g., an extension method
  `study.GetQualityDiversityArchive()`), since `BestTrial` semantics don't apply.
- **Variation operator:** reuse the SBX/polynomial-mutation kernels already in `BaseGASampler`
  (NSGA/AGE-MOEA share them) for `random_variation`.
- **Persistence:** archive lives in `system_attrs` (cell → best trial id + performance + descriptor).
- **Follow-on (note only):** CMA-ME / CMA-MEGA replace random variation with CMA-ES emitters — natural
  once MAP-Elites + our CMA core coexist.

### 7.3 Oracle & DoD

Behavioral vs Mouret's `pymap_elites` / `qdpy`: **archive coverage** (fraction of cells filled) and
**QD-score** (Σ elite performances) on Rastrigin-with-2D-descriptor reach reference ranges; determinism
under fixed seed. Full XML docs; both TFMs. **Gate the API design** with the maintainer before building
(it touches study-level surface).

---

## 8. Phase 5 — MO-CMA-ES (multi-objective CMA) — lower priority

**Why.** A 794-citation MO extension of CMA. Marginal value is lower than the single-objective ES/BO
gaps because our MO suite is already deep (NSGA-II/III, SMS-EMOA, AGE-MOEA-2, MOEA/D, RVEA).

**Home / shape.** Core, beyond-core. `Samplers/MoCmaEsSampler.cs` reusing the CMA `(1+1)`/rank-μ
machinery + non-dominated sorting (NSGA) + hypervolume contribution (already in SMS-EMOA).

**⚠️ Paper access gap.** `doi:10.1162/evco.2007.15.1.1` is paywalled (MIT Press) and recovery found no
OA copy. **Do not implement from memory.** Before this phase: obtain the paper (institutional access,
or the authors' tech-report version), *then* port against a reference impl (pymoo `MOCMAES` /
`deap.cma.StrategyMultiObjective` / pycma). Behavioral tier (hypervolume vs the reference on ZDT/DTLZ).
Until the paper is read, this phase is **blocked**.

---

## 9. Deferred frontier tier (survey only — no integration plan yet)

Surfaced by the search as SOTA/extraordinary but **out of scope for a define-by-run library** or
**bridge-only** (foreign runtime). Recorded so they are not re-discovered from scratch:

- **Meta-black-box optimization (learned optimizers):** LES / DES (Lange 2023), SYMBOL (Chen 2024),
  GLHF / B2Opt (Li 2024/25), RLDEAFL (Guo 2025). Require **offline meta-training over a task
  distribution** — antithetical to on-instance define-by-run. Possible future
  `OptunaSharp.Interop.*` bridge to `evosax` / `learned_optimization`. (JANUS, already shipped,
  deliberately occupies the *on-instance* complement of this space.)
- **LLM-as-optimizer:** OPRO (Yang 2023), FunSearch (Nature 2023). Need an LLM runtime → bridge only.
- **Diffusion-BBO** (offline BBO) and **PFNs4BO / in-context BO** (TabPFN-style transformers): torch
  bridge only.

---

## 10. Suggested sequencing

1. ~~**P1 — OpenAI-ES + SNES**~~ **✅ DONE** (self-contained, pure-managed, high extraordinary-claim value,
   mirrors existing DE/PSO plumbing → lowest risk). xNES was also completed (rides OpenBLAS `eigh`).
2. **P2 — TuRBO-1** **✅ DONE** (reused the GP stack; beats global GP on multimodal) and **REMBO**
   **✅ DONE** (random-embedding meta-wrapper; interleaved runs recover a low-effective-dim optimum) →
   **TuRBO-m** (multi-TR bandit) remains.
3. **P3 — SAASBO** (bridge extension; small, high-parity, once BoTorch bridge is touched anyway).
4. **P4 — MAP-Elites** (new QD surface; design-gated).
5. **P5 — MO-CMA-ES** (blocked on paper access).

**Per-item Definition of Done** (uniform, from `CLAUDE.md` §"Adding a ported feature"): read the
reference source; match signatures/defaults/edge-cases; `LazyRandomState` only; full XML docs on every
public/internal member (summary conveys consequence, every param/return/exception documented); unit
tests both TFMs derived from reference output; a `[RequiresPython]` parity/convergence check where
practical; keep `dotnet build` + the unit filter green; document any deliberate divergence with
`[Misaligned]`. Commit per repo convention (scratchpad message file, single add+commit, extensive body).
