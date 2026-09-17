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
  ([importance evaluators](../importance.md), [terminators](../terminators.md)).
- **Tier 2 — building blocks.** The **operators** (crossovers, mutations) and numerical **kernels**
  (hypervolume, reference directions, QMC sequences, GP acquisition functions, CMA optimizers) that
  Tier 1 is *composed of*.

## How this list is ordered

Within each role, methods are **grouped by lineage** — papers that build on one another sit together —
and each group is sorted **newest first**. So a reader sees each family as a chain from its latest
refinement back to its origin.

- ✅ — full deep-dive page.
- 🚧 — stub (header + paper link); the write-up is being filled in, in batches.

Every method below already ships in code today (see the [API reference](../../api/index.md)); 🚧 marks
only the *documentation*, not the feature.

---

## Tier 1 — methods you choose

### Samplers — search methods

**JANUS infill family** — a single online-Jacobian-aligned infill idea specialized per host optimizer.

| | Method | Paper |
|---|---|---|
| 🚧 | [JANUS](janus.md) | [Yu et al., arXiv:2608.22862 (2026)](https://arxiv.org/abs/2608.22862) |
| 🚧 | [JANUS-DE](janus-de.md) | [Yu et al., arXiv:2608.22862 (2026)](https://arxiv.org/abs/2608.22862) |
| 🚧 | [JANUS-MOO-CMIX](janus-moo-cmix.md) | [Yu et al., arXiv:2608.22862 (2026)](https://arxiv.org/abs/2608.22862) |
| 🚧 | [JANUS-SMS-EMOA-CMIX](janus-sms-emoa-cmix.md) | [Yu et al., arXiv:2608.22862 (2026)](https://arxiv.org/abs/2608.22862) |
| 🚧 | [JANUS-AGE-MOEA-2-CMIX](janus-age-moea-2-cmix.md) | [Yu et al., arXiv:2608.22862 (2026)](https://arxiv.org/abs/2608.22862) |

**Dominance & indicator-based MOEA** — the NSGA-II lineage and its descendants.

| | Method | Paper |
|---|---|---|
| ✅ | [AGE-MOEA-2 (2022)](age-moea-2.md) | [Panichella, GECCO 2022](https://doi.org/10.1145/3512290.3528732) |
| ✅ | [AGE-MOEA (2019)](age-moea.md) | [Panichella, GECCO 2019](https://doi.org/10.1145/3321707.3321839) |
| ✅ | [NSGA-III (2014)](nsga-iii.md) | [Deb & Jain, IEEE TEC 18(4), 2014](https://doi.org/10.1109/TEVC.2013.2281535) |
| ✅ | [SMS-EMOA (2007)](sms-emoa.md) | [Beume, Naujoks & Emmerich, EJOR 181(3), 2007](https://doi.org/10.1016/j.ejor.2006.08.008) |
| ✅ | [NSGA-II (2002)](nsga-ii.md) | [Deb et al., IEEE TEC 6(2), 2002](https://doi.org/10.1109/4235.996017) |

**Decomposition & reference-vector MOEA.**

| | Method | Paper |
|---|---|---|
| 🚧 | [RVEA (2016)](rvea.md) | [Cheng et al., IEEE TEC 20(5), 2016](https://doi.org/10.1109/TEVC.2016.2519378) |
| 🚧 | [MOEA/D (2007)](moea-d.md) | [Zhang & Li, IEEE TEC 11(6), 2007](https://doi.org/10.1109/TEVC.2007.892759) |

**CMA-ES / evolution-strategy family.**

| | Method | Paper |
|---|---|---|
| 🚧 | [VkD-CMA (2016)](vkd-cma.md) | [Akimoto & Hansen, GECCO 2016](https://doi.org/10.1145/2908812.2908863) |
| 🚧 | [LM-CMA (2015)](lm-cma.md) | [Loshchilov, arXiv:1511.00221 (2015)](https://arxiv.org/abs/1511.00221) |
| 🚧 | [CMA-ES (2001)](cma-es.md) | [Hansen & Ostermeier, Evol. Comput. 9(2), 2001](https://doi.org/10.1162/106365601750190398) |

**Natural Evolution Strategies (NES)** — natural-gradient search over a Gaussian distribution, from
isotropic to full-covariance.

| | Method | Paper |
|---|---|---|
| ✅ | [OpenAI-ES (2017)](openai-es.md) | [Salimans et al., arXiv:1703.03864 (2017)](https://arxiv.org/abs/1703.03864) |
| ✅ | [SNES (2014)](snes.md) | [Wierstra et al., JMLR 15, 2014](https://jmlr.org/papers/v15/wierstra14a.html) |
| ✅ | [xNES (2014)](xnes.md) | [Wierstra et al., JMLR 15, 2014](https://jmlr.org/papers/v15/wierstra14a.html) |

**Classic population-based & swarm.**

| | Method | Paper |
|---|---|---|
| 🚧 | [Differential Evolution (1997)](differential-evolution.md) | [Storn & Price, J. Global Optim. 11, 1997](https://doi.org/10.1023/A:1008202821328) |
| 🚧 | [Particle Swarm Optimization (1995)](pso.md) | [Kennedy & Eberhart, Proc. IEEE ICNN 1995](https://doi.org/10.1109/ICNN.1995.488968) |

**Model-based (Bayesian / surrogate).**

| | Method | Paper |
|---|---|---|
| ✅ | [TuRBO — Trust-Region Bayesian Optimization (2019)](turbo.md) | [Eriksson et al., NeurIPS 2019 (arXiv:1910.01739)](https://arxiv.org/abs/1910.01739) |
| 🚧 | [GP-BO — Gaussian-Process Bayesian Optimization (2012)](gp.md) | [Snoek et al., NeurIPS 2012 (arXiv:1206.2944)](https://arxiv.org/abs/1206.2944) |
| 🚧 | [TPE — Tree-structured Parzen Estimator (2011)](tpe.md) | [Bergstra et al., NeurIPS 2011](https://papers.nips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html) |

**Sampling & search baselines.**

| | Method | Paper |
|---|---|---|
| 🚧 | [Random Search (2012)](random.md) | [Bergstra & Bengio, JMLR 13, 2012](https://www.jmlr.org/papers/v13/bergstra12a.html) |
| 🚧 | [QMC — Quasi-Monte Carlo (1967)](qmc.md) | [Sobol′, USSR Comput. Math. Math. Phys. 7(4), 1967](https://doi.org/10.1016/0041-5553(67)90144-9) |
| 🚧 | [Grid Search](grid.md) | — (no canonical paper) |
| 🚧 | [Brute-Force Search](brute-force.md) | — (no canonical paper) |
| 🚧 | [Partial-Fixed](partial-fixed.md) | — (meta-sampler; no canonical paper) |

### Pruners — early-stopping methods

**Bandit-based (successive-halving lineage).**

| | Method | Paper |
|---|---|---|
| 🚧 | [Successive Halving / ASHA (2020)](successive-halving.md) | [Li et al., MLSys 2020 (arXiv:1810.05934)](https://arxiv.org/abs/1810.05934) |
| 🚧 | [Hyperband (2017)](hyperband.md) | [Li et al., JMLR 18, 2017 (arXiv:1603.06560)](https://arxiv.org/abs/1603.06560) |

**Statistical stopping rules.**

| | Method | Paper |
|---|---|---|
| 🚧 | [Median (2017)](median.md) | [Golovin et al. (Google Vizier), KDD 2017](https://doi.org/10.1145/3097983.3098043) |
| 🚧 | [Wilcoxon (1945)](wilcoxon.md) | [Wilcoxon, Biometrics Bulletin 1(6), 1945](https://doi.org/10.2307/3001968) |
| 🚧 | [Percentile](percentile.md) | — (generalizes the median rule) |
| 🚧 | [Threshold](threshold.md) | — (outlier thresholding) |
| 🚧 | [Patient](patient.md) | — (patience wrapper) |
| 🚧 | [Nop](nop.md) | — (never prunes) |

### Importance evaluators — attribution methods

**Density-based.**

| | Method | Paper |
|---|---|---|
| 🚧 | [PED-ANOVA (2023)](ped-anova.md) | [Watanabe et al., IJCAI 2023 (arXiv:2304.10255)](https://arxiv.org/abs/2304.10255) |

**Random-forest-surrogate based.**

| | Method | Paper |
|---|---|---|
| 🚧 | [SHAP (2017)](shap.md) | [Lundberg & Lee, NeurIPS 2017 (arXiv:1705.07874)](https://arxiv.org/abs/1705.07874) |
| 🚧 | [fANOVA (2014)](fanova.md) | [Hutter et al., ICML 2014](https://proceedings.mlr.press/v32/hutter14.html) |
| 🚧 | [Mean Decrease Impurity (2001)](mdi.md) | [Breiman, Machine Learning 45(1), 2001](https://doi.org/10.1023/A:1010933404324) |

### Terminators — stopping-criterion methods

**GP-based regret.**

| | Method | Paper |
|---|---|---|
| 🚧 | [EMMR — Expected Minimum Model Regret (2023)](emmr.md) | [Ishibashi et al., PMLR 206, 2023](https://proceedings.mlr.press/v206/ishibashi23a.html) |
| 🚧 | [Median Error Evaluator (2023)](median-error.md) | [Ishibashi et al., PMLR 206, 2023](https://proceedings.mlr.press/v206/ishibashi23a.html) |
| 🚧 | [Regret-Bound Evaluator (2022)](regret-bound.md) | [Makarova et al., AutoML-Conf 2022 (arXiv:2104.08166)](https://arxiv.org/abs/2104.08166) |

**Utility evaluators.**

| | Method | Paper |
|---|---|---|
| 🚧 | [Best-Value Stagnation](best-value-stagnation.md) | — (Optuna heuristic) |
| 🚧 | [Cross-Validation Error](cross-validation-error.md) | — (no canonical paper) |
| 🚧 | [Static Error](static-error.md) | — (no canonical paper) |

---

## Tier 2 — building blocks

### Operators — recombination & mutation

**SBX family (simulated binary).**

| | Operator | Paper |
|---|---|---|
| 🚧 | [vSBX — Modified Simulated Binary Crossover (2002)](vsbx.md) | [Modified SBX, PPSN VII (LNCS 2439), 2002](https://doi.org/10.1007/3-540-45105-6_86) |
| 🚧 | [SBX — Simulated Binary Crossover (1995)](sbx.md) | [Deb & Agrawal, Complex Systems 9(2), 1995](https://www.complex-systems.com/abstracts/v09_i02_a02/) |

**Simplex & normal-distribution recombination (multi-parent).**

| | Operator | Paper |
|---|---|---|
| 🚧 | [SPX — Simplex Crossover (2001)](spx.md) | [Tsutsui, Goldberg & Sastry, 2001](https://www.researchgate.net/publication/2388486_Progress_Toward_Linkage_Learning_in_Real-Coded_GAs_with_Simplex_Crossover) |
| 🚧 | [UNDX — Unimodal Normal Distribution Crossover (1999)](undx.md) | [Kita, Ono & Kobayashi, CEC 1999](https://doi.org/10.1109/CEC.1999.782672) |

**Blend & discrete recombination.**

| | Operator | Paper |
|---|---|---|
| 🚧 | [BLX-α — Blend Crossover (1993)](blx-alpha.md) | [Eshelman & Schaffer, FOGA 1993](https://doi.org/10.1016/B978-0-08-094832-4.50018-0) |
| 🚧 | [Uniform Crossover (1989)](uniform-crossover.md) | [Syswerda, ICGA 1989](https://www.researchgate.net/publication/201976488_Uniform_Crossover_in_Genetic_Algorithms) |

**Mutation.**

| | Operator | Paper |
|---|---|---|
| 🚧 | [Polynomial Mutation (1996)](polynomial-mutation.md) | [Deb & Goyal (GeneAS), 1996](https://doi.org/10.1109/4235.996017) |

### Kernels — numerical primitives

| | Kernel | Paper |
|---|---|---|
| 🚧 | [GP Acquisition Functions (2023)](gp-acquisitions.md) | [Ament et al. (LogEI), NeurIPS 2023 (arXiv:2310.20708)](https://arxiv.org/abs/2310.20708) |
| 🚧 | [Hypervolume (WFG) (2012)](hypervolume.md) | [While, Bradstreet & Barone, IEEE TEC 16(1), 2012](https://doi.org/10.1109/TEVC.2010.2077298) |
| 🚧 | [CMA Optimizers (2016)](cma-optimizers.md) | [Hansen, The CMA Evolution Strategy: A Tutorial (arXiv:1604.00772)](https://arxiv.org/abs/1604.00772) |
| 🚧 | [Reference Directions — Das–Dennis (1998)](reference-directions.md) | [Das & Dennis, SIAM J. Optim. 8(3), 1998](https://doi.org/10.1137/S1052623496307510) |
| 🚧 | [QMC Sequences — Sobol′ / Halton (1967)](qmc-sequences.md) | [Sobol′, USSR Comput. Math. Math. Phys. 7(4), 1967](https://doi.org/10.1016/0041-5553(67)90144-9) |

---

## Via integration bridges

These methods are driven through a foreign runtime and live in optional satellite packages, so they
are documented with their integrations (see the [Samplers guide](../samplers.md)) rather than as
native deep-dives:

- **BoTorch** — `BoTorchSampler` (`OptunaSharp.Interop.BoTorch`): Bayesian optimization via the real
  BoTorch ([Balandat et al., NeurIPS 2020](https://arxiv.org/abs/1910.06403)).
- **PyCma** — `PyCmaSampler` (`OptunaSharp.Interop.PyCma`): CMA-ES via Hansen's `cma` package.
- **Pymoo** — `PymooSampler` (`OptunaSharp.Interop.Pymoo`): drives 11 pymoo multi-objective
  algorithms (NSGA-II/III, R-NSGA-II/III, U-NSGA-III, AGE-MOEA/2, RVEA, SMS-EMOA, SPEA2, C-TAEA).
- **OptunaHub** — `OptunaHubSampler` / `OptunaHubPruner` (`OptunaSharp.Interop.pythonnet`): wraps any
  of the ~85 samplers and the pruners on [hub.optuna.org](https://hub.optuna.org/).
- **Forest-backend importances** — ML.NET FastForest/FastTree and LightGBM variants of fANOVA / MDI
  (`OptunaSharp.Integration.ML.NET`, `OptunaSharp.Integration.LightGbm`).

## See also

[Samplers](../samplers.md) · [Pruners](../pruners.md) · [Multi-objective Optimization](../multi-objective.md) ·
[Parameter Importance](../importance.md) · [Terminators](../terminators.md) ·
the [API reference](../../api/index.md).
