# Methods

This section is the per-method deep-dive reference for OptunaSharp: one page per algorithm, each
with its paper citation, the idea (with pseudocode), the OptunaSharp class and a runnable example,
its lineage, and cross-links. It complements the [Guides](../samplers.md) (which explain *how to use*
a family) by explaining *how each method works*.

## Why "Methods" and not "Algorithms"

Everything here is a named, published algorithm — but the set spans **two tiers**, which is why the
umbrella is "methods" rather than the flatter "algorithms":

- **Tier 1 — methods you choose.** The things you plug into a [`Study`](../study-and-trials.md)
  ([samplers](../samplers.md), [pruners](../pruners.md)) or run to analyze one
  ([importance evaluators](../importance.md), [terminators](../terminators.md)). A sampler is a
  *search method*; a pruner is an *early-stopping method*.
- **Tier 2 — building blocks.** The **operators** (crossovers, mutations) and numerical **kernels**
  (hypervolume, reference directions, QMC sequences, GP acquisition functions, CMA optimizers) that
  Tier 1 is *composed of*. You rarely select these directly; they are the machinery the methods share.

## Legend

Pages are written in **batches of three**, so this catalog is the source of truth for what exists and
what is planned:

- ✅ — page available (link is live).
- ◻ — page planned; the link points to its final location and goes live when the page is written.

Every method below already ships in code today (see the [API reference](../../api/index.md)); the ◻
marks only the *documentation* page, not the feature.

---

## Tier 1 — methods you choose

### Samplers — search methods

Faithful ports of Optuna's own samplers (`OptunaSharp` core package):

| Method | Class | Paper |
|---|---|---|
| ◻ [Random](random.md) | `RandomSampler` | — (independent uniform sampling) |
| ◻ [Grid](grid.md) | `GridSampler` | — (exhaustive grid search) |
| ◻ [Brute-Force](brute-force.md) | `BruteForceSampler` | — (exhaustive define-by-run search) |
| ◻ [Partial-Fixed](partial-fixed.md) | `PartialFixedSampler` | — (meta-sampler; fixes a subset, delegates the rest) |
| ◻ [TPE (2011)](tpe.md) | `TPESampler` | Bergstra et al., *Algorithms for Hyper-Parameter Optimization*, NeurIPS 2011 |
| ◻ [CMA-ES (2001/2016)](cma-es.md) | `CmaEsSampler` | Hansen, *The CMA Evolution Strategy* (arXiv:1604.00772) — + sep-CMA, margin, LR-adapt, warm-start |
| ◻ [GP-BO](gp.md) | `GPSampler` | Gaussian-process Bayesian optimization; LogEI acq: Ament et al., NeurIPS 2023 |
| ◻ [QMC](qmc.md) | `QMCSampler` | Quasi-Monte Carlo — Sobol′ 1967 / Halton 1960 |
| ✅ [NSGA-II (2002)](nsga-ii.md) | `NSGAIISampler` | Deb et al., IEEE TEC 6(2), 2002 |
| ✅ [NSGA-III (2014)](nsga-iii.md) | `NSGAIIISampler` | Deb & Jain, IEEE TEC 18(4), 2014 |

Beyond-core native samplers (in `OptunaSharp` core, but absent from Optuna itself):

| Method | Class | Paper |
|---|---|---|
| ◻ [LM-CMA (2015)](lm-cma.md) | `LmCmaSampler` | Loshchilov, *LM-CMA*, Evolutionary Computation 2015 (arXiv:1511.00221) |
| ◻ [VkD-CMA (2016)](vkd-cma.md) | `VkdCmaSampler` | Akimoto & Hansen, GECCO 2016 |
| ◻ [Differential Evolution (1997)](differential-evolution.md) | `DifferentialEvolutionSampler` | Storn & Price, J. Global Optimization 11, 1997 |
| ◻ [Particle Swarm (1995)](pso.md) | `PsoSampler` | Kennedy & Eberhart 1995; Shi & Eberhart 1998 (inertia) |
| ◻ [MOEA/D (2007)](moea-d.md) | `MoeaDSampler` | Zhang & Li, IEEE TEC 11(6), 2007 |
| ◻ [RVEA (2016)](rvea.md) | `RveaSampler` | Cheng et al., IEEE TEC 20(5), 2016 |
| ✅ [AGE-MOEA (2019)](age-moea.md) | (survival) | Panichella, GECCO 2019 |
| ✅ [AGE-MOEA-2 (2022)](age-moea-2.md) | `AgeMoea2Sampler` | Panichella, GECCO 2022 |
| ✅ [SMS-EMOA (2007)](sms-emoa.md) | `SmsEmoaSampler` | Beume, Naujoks & Emmerich, EJOR 181(3), 2007 |

JANUS family (beyond-core; Yu et al., *JANUS: Online Jacobian-Aligned Infill for Black-Box
Optimization*, arXiv:2608.22862, 2026):

| Method | Class | Role |
|---|---|---|
| ◻ [JANUS](janus.md) | `JanusSampler` | single-objective infill sampler (CMA / DE population hosts) |
| ◻ [JANUS-DE](janus-de.md) | `JanusDeSampler` | JANUS over a differential-evolution host |
| ◻ [JANUS-MOO-CMIX](janus-moo-cmix.md) | `JanusMooCmixSampler` | multi-objective CMIX over NSGA-II |
| ◻ [JANUS-SMS-EMOA-CMIX](janus-sms-emoa-cmix.md) | `JanusSmsEmoaCmixSampler` | CMIX over SMS-EMOA survival |
| ◻ [JANUS-AGE-MOEA-2-CMIX](janus-age-moea-2-cmix.md) | `JanusAgeMoea2CmixSampler` | CMIX over AGE-MOEA-2 survival |

### Pruners — early-stopping methods

`OptunaSharp` core. See the [Pruners guide](../pruners.md) for usage.

| Method | Class | Paper |
|---|---|---|
| ◻ [Median](median.md) | `MedianPruner` | median stopping rule — Golovin et al. (Vizier), KDD 2017 |
| ◻ [Percentile](percentile.md) | `PercentilePruner` | — (generalized median rule) |
| ◻ [Successive Halving / ASHA](successive-halving.md) | `SuccessiveHalvingPruner` | Li et al., *A System for Massively Parallel HP Tuning*, MLSys 2020 |
| ◻ [Hyperband](hyperband.md) | `HyperbandPruner` | Li et al., JMLR 18, 2017 |
| ◻ [Threshold](threshold.md) | `ThresholdPruner` | — (outlier thresholding) |
| ◻ [Patient](patient.md) | `PatientPruner` | — (patience wrapper over another pruner) |
| ◻ [Wilcoxon](wilcoxon.md) | `WilcoxonPruner` | Wilcoxon signed-rank test, 1945 |
| ◻ [Nop](nop.md) | `NopPruner` | — (never prunes) |

### Importance evaluators — attribution methods

`OptunaSharp` core (pure-managed). See the [Parameter Importance guide](../importance.md).

| Method | Class | Paper |
|---|---|---|
| ◻ [fANOVA (2014)](fanova.md) | `FanovaImportanceEvaluator` | Hutter et al., *Assessing Hyperparameter Importance*, ICML 2014 |
| ◻ [Mean Decrease Impurity](mdi.md) | `MeanDecreaseImpurityImportanceEvaluator` | Breiman, *Random Forests*, 2001 |
| ◻ [PED-ANOVA (2023)](ped-anova.md) | `PedAnovaImportanceEvaluator` | Watanabe et al., IJCAI 2023 (+ Conditional PED-ANOVA, KDD 2026) |
| ◻ [SHAP (2017)](shap.md) | `ShapleyImportanceEvaluator` | Lundberg & Lee, *A Unified Approach…*, NeurIPS 2017 |

### Terminators — stopping-criterion methods

`OptunaSharp` core. See the [Terminators guide](../terminators.md).

| Method | Class | Paper |
|---|---|---|
| ◻ [Regret-Bound](regret-bound.md) | `RegretBoundEvaluator` | Makarova et al., *Automatic Termination for HPO*, AutoML 2022 |
| ◻ [EMMR (2023)](emmr.md) | `EMMREvaluator` | Ishibashi et al., *A stopping criterion for BO…*, PMLR 206, 2023 |
| ◻ [Best-Value Stagnation](best-value-stagnation.md) | `BestValueStagnationEvaluator` | — (Optuna heuristic) |
| ◻ [Cross-Validation Error](cross-validation-error.md) | `CrossValidationErrorEvaluator` | — (CV-variance statistical error) |
| ◻ [Static Error](static-error.md) | `StaticErrorEvaluator` | — (constant error term) |
| ◻ [Median Error](median-error.md) | `MedianErrorEvaluator` | ratio-to-initial-median heuristic (Ishibashi et al. 2023) |

---

## Tier 2 — building blocks

### Operators — evolutionary recombination & mutation

Shared by the NSGA-II/III, AGE-MOEA, SMS-EMOA, MOEA/D and RVEA samplers (`OptunaSharp` core).

| Operator | Class | Paper |
|---|---|---|
| ◻ [SBX — Simulated Binary Crossover](sbx.md) | `SBXCrossover` | Deb & Agrawal, 1995 |
| ◻ [vSBX — Modified SBX](vsbx.md) | `VSBXCrossover` | modified-SBX variant (Optuna `_vsbx`) |
| ◻ [BLX-α — Blend Crossover](blx-alpha.md) | `BLXAlphaCrossover` | Eshelman & Schaffer, 1993 |
| ◻ [SPX — Simplex Crossover](spx.md) | `SPXCrossover` | Tsutsui, Yamamura & Higuchi, GECCO 1999 |
| ◻ [UNDX — Unimodal Normal Distribution Crossover](undx.md) | `UNDXCrossover` | Ono & Kobayashi, 1997 |
| ◻ [Uniform Crossover](uniform-crossover.md) | `UniformCrossover` | Syswerda, 1989 |
| ◻ [Polynomial Mutation](polynomial-mutation.md) | `PolynomialMutation` | Deb & Goyal (GeneAS), 1996 |

### Kernels — numerical primitives

| Kernel | Location | Paper |
|---|---|---|
| ◻ [Hypervolume (WFG)](hypervolume.md) | `Internal/Hypervolume.cs` | While, Bradstreet & Barone, IEEE TEC 16(1), 2012 |
| ◻ [Reference Directions](reference-directions.md) | `Internal/ReferenceDirections.cs` | Das & Dennis 1998; Riesz s-energy: Blank et al. 2021 |
| ◻ [QMC Sequences](qmc-sequences.md) | `Internal/{QuasiMonteCarlo,ScrambledSobol,ScrambledHalton}.cs` | Sobol′ 1967 / Halton 1960; Owen scrambling 1998 |
| ◻ [GP Acquisition Functions](gp-acquisitions.md) | `Gp/Acqf.cs` | LogEI: Ament et al. NeurIPS 2023; qEHVI: Daulton et al. NeurIPS 2020 |
| ◻ [CMA Optimizers](cma-optimizers.md) | `Samplers/CmaEs/{Cma,SepCma,CmaWm,LmCma,VkdCma}.cs` | see [CMA-ES](cma-es.md) / [LM-CMA](lm-cma.md) / [VkD-CMA](vkd-cma.md) |

---

## Via integration bridges

These methods are driven through a foreign runtime and live in optional satellite packages, so they
are documented with their integrations (see the [Samplers guide](../samplers.md)) rather than as
native deep-dives:

- **BoTorch** — `BoTorchSampler` (`OptunaSharp.Interop.BoTorch`): Bayesian optimization via the real
  BoTorch (Balandat et al., NeurIPS 2020).
- **PyCma** — `PyCmaSampler` (`OptunaSharp.Interop.PyCma`): CMA-ES via Hansen's `cma` package.
- **Pymoo** — `PymooSampler` (`OptunaSharp.Interop.Pymoo`): drives 11 pymoo multi-objective
  algorithms (NSGA-II/III, R-NSGA-II/III, U-NSGA-III, AGE-MOEA/2, RVEA, SMS-EMOA, SPEA2, C-TAEA). See
  the [pymoo Integration guide](../pymoo.md).
- **OptunaHub** — `OptunaHubSampler` / `OptunaHubPruner` (`OptunaSharp.Interop.pythonnet`): wraps any
  of the ~85 samplers and the pruners on [hub.optuna.org](https://hub.optuna.org/).
- **Forest-backend importances** — ML.NET FastForest/FastTree and LightGBM variants of fANOVA / MDI
  (`OptunaSharp.Integration.ML.NET`, `OptunaSharp.Integration.LightGbm`).

## See also

[Samplers](../samplers.md) · [Pruners](../pruners.md) · [Multi-objective Optimization](../multi-objective.md) ·
[Parameter Importance](../importance.md) · [Terminators](../terminators.md) ·
the [API reference](../../api/index.md).
