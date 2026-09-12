# `OptunaSharp.Benchmark.VolleyEval` — an algorithm-quality evaluation suite

> **Status: DESIGN (not started).** This document establishes the design in words. VolleyEval is a
> standing apparatus that fires a *volley* of optimization **challenges** at any sampler (or whole
> study configuration), **auto-hyperoptimizes** that sampler against each challenge, and records a
> normalized score — accumulating, over time, a **sampler × challenge profile matrix** that proves
> where each study is strong and where it is weak.
>
> **Distinct from `OptunaSharp.Benchmark`.** The existing `test/OptunaSharp.Benchmark` is a
> *BenchmarkDotNet micro-performance* harness (per-operation wall-clock: how fast is one `sampler.step()`).
> VolleyEval measures the orthogonal thing: **optimization quality** — how *good* are the points a
> sampler proposes, per unit of budget. Speed and quality are different axes and we keep them in
> different projects.
>
> **Grounded in evidence.** Every design choice below is motivated by measurements taken during the
> 2026-09-10 sampler investigation (see §12). The headline lesson — *there is no single best sampler,
> only a best sampler per regime* — is the reason this suite exists.

---

## 0. Why VolleyEval

A single leaderboard is a lie. In one session we watched the ranking **completely invert** with the
problem regime:

- On a **128×1024 categorical** reward space, cheap **QMC** beat model-based **TPE** at low budget,
  a **tuned NSGA-III** beat everything (incl. `pycma`), and **CMA-ES was bit-identical to random**.
- On **128-D continuous**, **LM-CMA** beat NSGA-III by **12–102×** and beat full CMA by 3–9×.
- **BoTorch/GP** could not complete a useful budget at high categorical cardinality (a 131k-dim
  one-hot); they are only viable in their design regime (low-D, expensive objective, tiny budget).

So "which sampler is best?" is unanswerable in the abstract, and any benchmark that reports one number
is misleading. VolleyEval answers the *useful* question instead: **for each kind of problem, which
sampler wins, by how much, at what cost, and with what hyperparameters — proven, reproducibly, and
tracked over time.**

---

## 1. Vocabulary

| term | meaning |
|------|---------|
| **Challenge** (`IChallenge`) | One benchmark problem: a define-by-run objective, a direction, a budget, a known/estimated optimum, a random baseline, and difficulty **tags**. |
| **Contender** (`Contender`) | A sampler under test, plus the **hyperparameter search space** used to tune it (its knobs and their ranges). |
| **Round** | One contender evaluated on one challenge (inner runs + hyperopt + validation). |
| **Volley** | A full sweep: every contender × every challenge → the complete scoreboard. |
| **Scoreboard / Profile matrix** | contenders × challenges → normalized score (+ rank, + cost). The primary artifact. |
| **Recipe** | The winning hyperparameters a contender's hyperopt found for a challenge — a reusable library entry. |

The volleyball metaphor is deliberately light: a *volley* is the exchange where each challenge is
served at each contender and the rally is scored. Code uses the plain technical names above.

---

## 2. Challenge taxonomy — the axes of "difficulty space"

Samplers diverge along measurable axes. The catalog must **span** them so that every sampler's
strength and weakness is exposed by *some* challenge. A challenge that no contender struggles with is
worthless and is rejected.

| axis | span | the sampler it rewards / punishes |
|------|------|-----------------------------------|
| **parameter type** | float · int · categorical (low/high cardinality) · bool · mixed | continuous→CMA family; categorical→NSGA/QMC; discrete→CMAwM; categorical *punishes* native CMA/LM-CMA (they abstain → random) |
| **dimensionality** | 2 · 10 · 50 · 128 · 512+ | low→GP/TPE; high→LM-CMA/SepCMA; high *punishes* full CMA (O(n²)) and GP (one-hot blow-up) |
| **structure** | flat · conditional/tree · combinatorial building-block | conditional→TPE `group=true`; building-block→GA |
| **geometry** | convex · ill-conditioned · coupled/non-separable · multimodal · deceptive | convex/coupled→CMA; ill-conditioned *punishes* SepCMA; multimodal→GA/QMC |
| **separability** | separable ↔ fully coupled | separable→SepCMA; coupled→full CMA |
| **noise** | deterministic ↔ stochastic | rewards averaging/robust methods |
| **objective cost** | cheap (∞ trials) ↔ expensive (~30 trials) | expensive→GP/BoTorch (their only home) |
| **# objectives** | 1 · 2 · 3 · many | 2–3→NSGA-II; many→NSGA-III; scored by hypervolume |
| **constraints** | none ↔ hard | constrained TPE/GP/NSGA |
| **budget** | tiny (30) · small (300) · large (5000) | interacts with hyperparameters (small-population GA won *only* at low budget) |

---

## 3. v1 challenge catalog

Each challenge declares tags (from §2), an optimum (known for synthetic; best-seen proxy for real), a
random baseline, a default budget, and an intended-champion **hypothesis** (a falsifiable prediction —
if the hypothesis is wrong, that itself is a finding worth recording).

| id | challenge | space | axis probed | hypothesized champion |
|----|-----------|-------|-------------|-----------------------|
| **B1** | Sphere | float, D∈{10,128,512} | dimension × convex | CMA (low D) → **LM-CMA (high D)** |
| **B2** | Rosenbrock | float, D∈{10,128} | coupled / non-separable | CMA / LM-CMA |
| **B3** | Rastrigin / Ackley | float, D∈{10,128} | multimodal | GA vs CMA (dimension-dependent) |
| **B4** | Ellipsoid (ill-conditioned) | float, D=40 | conditioning | full CMA; SepCMA should *fail* |
| **B5** | **[1]-[4] reward** ✅ built | 128 categorical × 1024 | high-cardinality combinatorial | tuned NSGA-III / `pycma` |
| **B6** | Conditional tree ✅ built | switch → gated sub-params | dynamic / hierarchical | TPE `group=true` |
| **B7** | Mixed continuous + categorical | float + categorical | mixed types (the tie-breaker) | TPE / CMAwM / **future hybrid** |
| **B8** | Expensive / tiny budget | float D=6, budget=40 | sample efficiency at low budget | **GP / BoTorch** (their home) |
| **B9** | Multi-objective (ZDT / DTLZ) | 2–4 objectives | Pareto-front quality (hypervolume) | NSGA-II (2–3) / NSGA-III (many) |
| **B10** | Noisy sphere | float + Gaussian noise | robustness to noise | TPE / CMA with budget |
| **B11** | Deceptive / needle-in-haystack | flat + narrow optimum | exploration | QMC / random-ish |
| **B12** | Real HPO surrogate | tabulated real sweep (e.g. MNIST/sisyphus) | realistic landscape | *open — the payoff test* |

B5, B6 already exist as scratch experiments from the 2026-09-10 session and are the first to fold in.
B12 is the bridge to real work: a recorded response surface (or a cheap learned surrogate of one) so
the suite is not purely synthetic.

---

## 4. Metrics — two per challenge, never one

1. **Best-at-budget** — final objective value after the challenge's fixed trial budget. "How good does
   it get?"
2. **Trials-to-threshold** — the sisyphus **compute-to-threshold** idea: number of trials to first
   reach {50%, 90%, 99%} of the optimum-above-random band. "How *fast* does it get there?"

Both are reported as a **seed distribution** (median + inter-quartile range), never a bare mean — the
2026-09-10 runs showed high seed variance (a single-seed 13140 validated to a 12277–13140 spread).

---

## 5. Normalization — so challenges are comparable

Raw values are incommensurable (loss ~10⁶ vs reward ~12k vs hypervolume ∈ [0,1]). Each round is
normalized two ways:

- **Normalized score** ∈ [0,1] = `(best − random) / (optimum − random)`. 0 = no better than random;
  1 = optimal. Synthetic challenges carry exact optima; real ones use best-ever-observed as a proxy
  and mark the score *provisional*.
- **Rank** within the contender cohort — robust to scale and outliers.

The volley aggregate is reported **both** as mean/median normalized score **and** as a rank-tournament
score, because they answer different questions ("how close to optimal on average" vs "how often best").

**DNF is a first-class result.** A contender that cannot complete the budget in a sane wall-clock
(BoTorch's 830 s for 11 trials) is recorded `DNF(infeasible)`, not a number. A contender that abstains
on a parameter type (native CMA on categoricals → random) is recorded `≡random(inert)`. Both are
signals, not gaps.

---

## 6. The hyperoptimization protocol

Per the mandate — *each challenge gets to hyperoptimize the sampler* — every Round is a nested study:

1. **Outer study** (TPE by default; CMA optional) searches the contender's declared
   **hyperparameter space** (e.g. NSGA: `population_size, crossover_prob, swapping_prob,
   mutation_prob`; TPE: `n_startup_trials, n_ei_candidates, multivariate, group`; CMA:
   `sigma0, popsize, restart, lr_adapt`).
2. **Inner objective** = the contender runs the challenge at its budget on the **search seeds**;
   returns the aggregated best-at-budget.
3. **Validation** = the outer winner is re-run on **held-out seeds** and reported next to the
   contender's **default** configuration. Validation is mandatory: on 2026-09-10 a single-seed
   inner-best of 13140 validated to 12617 — real, but the gap is exactly the seed-overfit the
   held-out step guards against.
4. **Recipe** = the validated winning hyperparameters are written to the recipe library, keyed by
   `(contender, challenge)`.

Guardrails baked in from experience: tuning is **budget-specific** (small populations win only at low
budget — report per-budget), and the outer search must not be trusted on its raw single-seed best.

---

## 7. Fairness — two leaderboards, not one

Contender per-trial cost spanned **20–800×** in one session (TPE 537 s vs QMC 24 s for the same 500
trials; BoTorch 830 s for 11 trials). A leaderboard that ignores cost is useless in practice, so every
result carries wall-clock and the suite publishes:

- **Per-trial-budget** leaderboard — pure algorithmic sample efficiency (all contenders get N trials).
- **Per-wall-clock** leaderboard — practical efficiency (all contenders get T seconds). A cheap
  sampler that reaches 90% quality in 5% of the time usually wins the real-world question.

---

## 8. Architecture

```
test/OptunaSharp.Benchmark.VolleyEval/            (new project: console CLI + library)
├── IChallenge                 { Name; Tags[]; Direction; Budget; Optimum?; RandomBaseline;
│                                BuildObjective(Trial) -> double | IReadOnlyList<double> }
├── ChallengeRegistry          ← all B1..B12, discoverable/filterable by tag
├── Challenges/                ← Sphere, Rosenbrock, Rastrigin, Ellipsoid, Reward1to4 (B5),
│                                ConditionalTree (B6), Mixed, ExpensiveLowD, MultiObjective, Noisy,
│                                Deceptive, RealSurrogate
├── Contender                  { Name; BuildSampler(hp, seed); HyperparamSpace(Trial) }
├── ContenderRegistry          ← random, qmc, tpe(+group), cma/sepcma/cmawm, lmcma, nsga2, nsga3,
│                                pycma, botorch, gp  (each with its tuning space)
├── Evaluator                  ← runs a Round: inner runs, outer hyperopt, held-out validation,
│                                seed aggregation, normalization, DNF/inert detection
├── Runner                     ← parallel across Rounds (one OS process per job; must throttle
│                                torch-backed contenders — they saturate every core)
├── Scoreboard / ProfileMatrix ← normalized-score + rank + wall-clock, per (contender, challenge)
├── ResultStore                ← persist to SQLite (OptunaSharp.Storage.Sqlite) → over-time tracking
└── Reporter                   ← heatmap + convergence/regret curves (OptunaSharp.Visualization.Plotly)
```

**CLI shape** (a .NET 10 file-based-app-friendly console):

```
volley run   --challenges B1,B5,B8 --contenders nsga3,lmcma,tpe --budget default --seeds 5 --tune 40
volley run   --all                                    # the full volley
volley report --matrix                                # emit the profile-matrix heatmap
volley report --recipes nsga3                         # dump the winning recipe library for a contender
volley diff  --against <run-id>                       # regression: did a code change move any cell?
```

---

## 9. Reuse of existing OptunaSharp infrastructure

VolleyEval is mostly glue over surfaces already ported — a point in its favor:

- **`OptunaSharp.Storage.Sqlite`** — persist every Round; results accumulate across sessions
  (the "over time" requirement) and become the baseline for regression diffs.
- **`OptunaSharp.Importance` (fANOVA / PED-ANOVA)** — after each hyperopt, report *which* of the
  contender's knobs actually explained its variance (e.g. "`population_size` explains 80% of NSGA-III
  variance on B5"). Turns tuning into insight.
- **`WilcoxonPruner`'s signed-rank test** — reuse for **pairwise significance** between contenders
  ("is tuned NSGA-III *significantly* above `pycma` on B5, or within noise?").
- **`OptunaSharp.Visualization.Plotly`** — render the profile matrix and per-challenge regret curves.
- **The nested-study pattern itself** — VolleyEval's outer hyperopt is just an OptunaSharp study whose
  objective runs an OptunaSharp study. "Optimize the optimizer," realized inside the library.

---

## 10. Outputs — what the suite produces

- **Profile matrix** (contenders × challenges heatmap of normalized score) — the one-glance
  strength/weakness map.
- **Recipe library** — best validated hyperparameters per regime, auto-generated.
- **Regression gate** — re-run the volley after a sampler code change; flag any cell that regressed
  (with significance).
- **Recommender** — map a user's problem tags → the contender that owns that regime
  ("high-D continuous → `LmCmaSampler`; categorical building-block → tuned NSGA-III;
  expensive low-D → `GPSampler`").
- **New-algorithm proving ground** — a prospective sampler (e.g. the mixed continuous+categorical
  hybrid) must clear a quantitative, regime-specific bar to justify itself.

---

## 11. Phased rollout

- **v0** — `IChallenge` + `Contender` + `Evaluator` + normalized score; wrap the already-built B5
  (categorical), B6 (conditional), B1/B2 (continuous) and the nested hyperopt loop; one command emits
  a first profile matrix. Proves the apparatus end-to-end.
- **v1** — add B3, B4, B7, B8, and multi-objective B9 (requires a hypervolume indicator); SQLite
  persistence + the Plotly heatmap report; per-trial and per-wall-clock leaderboards.
- **v2** — noisy/constrained/deceptive (B10/B11), the real surrogate (B12), Wilcoxon significance,
  fANOVA per-Round importance, and the regression-diff gate wired into CI.

---

## 12. Open design decisions

1. **Normalization** — normalized-regret (needs optima) vs pure rank. *Recommendation:* both; regret
   primary, rank as the robustness cross-check.
2. **Fairness axis** — per-trial vs per-wall-clock. *Recommendation:* publish both; neither alone is
   honest.
3. **Project home** — a dedicated `test/OptunaSharp.Benchmark.VolleyEval` (a lasting apparatus) vs an
   `experiments/`-style folder. *Recommendation:* dedicated project, sibling to the perf
   `OptunaSharp.Benchmark`.
4. **Hyperopt budget** — outer trials × inner budget × inner seeds is the dominant cost. *Recommendation:*
   default outer=40, inner=challenge-budget, search-seeds=1, validation-seeds=5; expose all as flags.
5. **Multi-objective indicator** — exact hypervolume (expensive in many dims) vs a Monte-Carlo/IGD proxy.
   *Recommendation:* exact for ≤3 objectives, monte-carlo hypervolume beyond.

---

## Appendix A — evidence base (2026-09-10 investigation)

The findings that motivate every choice above; VolleyEval is the apparatus that would produce these
systematically rather than ad hoc.

| observation | on | consequence for the design |
|-------------|-----|----------------------------|
| QMC > TPE at low budget; **tuned NSGA-III (12617) > pycma (11070) > default NSGA (10575)** | 128×1024 categorical (B5) | regime-specific champions; hyperopt matters (+19% over default) |
| **LM-CMA beats NSGA-III 12–102×**, and beats full CMA 3–9× | 128-D continuous (B1/B2/B3) | dimension × type flips the winner; need continuous challenges at high D |
| **CMA-ES ≡ random** (bit-identical) | categorical | "inert" is a first-class result; type-abstention must be recorded |
| **BoTorch: 830 s / 11 trials**, below random | 128×1024 categorical | DNF/infeasible is a first-class result; wall-clock leaderboard essential |
| `pycma` encodes categoricals → wins where native CMA abstains | categorical | native-vs-integration gaps are worth surfacing per challenge |
| single-seed 13140 → validated 12617 across seeds | B5 hyperopt | held-out-seed validation is mandatory |
| small population (6) beat default (50) — but only at the 500-trial budget | B5 hyperopt | tuning is budget-specific; report per-budget |

---

*End of `OptunaSharp.Benchmark.VolleyEval` design.*
