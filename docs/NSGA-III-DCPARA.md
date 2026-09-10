# NSGA-III / NSGA-II with DCPARA — Dynamic Conditional Parameterization

> **Status.** Working design note, kept as the seed of a future paper. It documents the
> OptunaSharp-only changes that let the NSGA-II / NSGA-III genetic samplers optimize *conditional*
> (define-by-run, branch-defined) hyperparameters. Everything here is **opt-in**: with no rules
> declared the samplers stay bit-for-bit identical to Optuna (§8). Numbers in §9 are measurements on
> the deterministic surrogate probes shipped in `examples/` and `test/`, not benchmark claims — the
> benchmark study is future work (§13).

**DCPARA** = *Dynamic Conditional PARAmeterization*: the set of hyperparameters a trial carries is not
fixed but a function of the values of other hyperparameters, and — optionally — a *dynamic* quantity
that flows between an active genome and a dormant side channel without losing what was learned.

---

## 1. Abstract

Modern hyperparameter-optimization APIs are *define-by-run*: the objective function itself decides,
while executing, which parameters exist — `if (optimizer == "adam") beta1 = suggest(...)`. This yields
a **conditional search space** whose active dimensions vary per trial. Evolutionary multi-objective
samplers (NSGA-II, NSGA-III) are a poor fit for such spaces as usually implemented, because they evolve
the *intersection* search space — the parameters present in **every** completed trial. A branch-defined
parameter drops out of that intersection after the first trial that does not take its branch, and is
thereafter resampled uniformly at random on every trial: it receives no crossover, no mutation, and no
inheritance, i.e. no evolutionary pressure at all — precisely on the parameters that make a branch
choice meaningful.

We add a declarative activation layer, **`ConditionalParams`**, that tells the sampler which parameters
are conditional on which activator values. Declared parameters stay in the relative (evolved) search
space and are crossed over, mutated and inherited whenever they are *active*, and are simply absent —
never blended, never counted — whenever they are *inactive*. We integrate this at three seams of the
existing NSGA machinery (search-space construction, crossover, child generation) and prove a
**math-neutrality** property: for a child on a given branch, the generated individual *and the RNG
stream after it* are bit-identical to a counterfactual world in which the deactivated parameter never
existed. Finally we add an opt-in **dynamic** mode (`PreserveDormantValues`) under which a parameter
whose branch switches off parks its last active value in a latent side channel and *resumes* from it
when the branch switches back on, so a fluctuating genome does not repeatedly relearn the same genes.

The design is realized in the .NET port [OptunaSharp](../README.md); the reference behavior of the
untouched path is Optuna 5.x.

---

## 2. Provenance

| Commit | Change |
|--------|--------|
| `fb812ca` | `feat(samplers)` — `ConditionalParams` activation rules for NSGA-II/III (algebra + three seams) |
| `07adb14` | `test(samplers)` — controllable / learns-when-active / dormant-when-inactive proof tests + first example |
| `6b15b88` | `docs(examples)` — three code-reader-oriented recipes (model selection, architecture, schedule) |
| `cbbf660` | `test(samplers)` — null-control: nothing changes unless a rule is explicitly activated |
| `47bd554` | `feat(samplers)` — dormant-value memory (`PreserveDormantValues`), the dynamic mode |
| `71b6610` | `fix(samplers)` — prune-aware conditional crossover: deactivation is math-neutral incl. multi-parent |
| `a6cf625` | `docs(examples)` — enum-array optional-chain example |

Source of record:

- Algebra + dormant carry — `src/OptunaSharp.Core/SearchSpace/ConditionalParams.cs`
- Search-space fold — `src/OptunaSharp.Core/SearchSpace/IntersectionSearchSpace.cs` (`CalculateConditional`)
- Crossover — `src/OptunaSharp.Core/Samplers/Nsgaii/NsgaiiCrossover.cs` (`PerformConditionalCrossover`)
- Child generation — `src/OptunaSharp.Core/Samplers/Nsgaii/NSGAIIChildGenerationStrategy.cs`
- Sampler wiring — `src/OptunaSharp.Core/Samplers/{NSGAIISampler,NSGAIIISampler}.cs`

---

## 3. Problem statement

### 3.1 Two sampling regimes

OptunaSharp (like Optuna) splits each trial's parameters into two groups. **Relative** parameters are
sampled jointly by the sampler's model — for a GA, this is the genetic machinery (selection, crossover,
mutation). **Independent** parameters are sampled one at a time with no cross-parameter model — for the
GA fallback, uniformly at random via `RandomSampler`. `Sampler.InferRelativeSearchSpace` decides which
parameters are relative for a trial; anything not in it is independent.

### 3.2 The intersection search space

For the GA samplers, `InferRelativeSearchSpace` returns the **intersection search space**
(`IntersectionSearchSpace`): the set of parameters that appear, with an identical distribution, in every
completed trial considered so far (single-valued distributions excluded). This is correct and necessary
for a static space — it is exactly the set of genes every individual shares, so crossover and mutation
are well-defined.

### 3.3 The failure on conditional spaces

Consider a define-by-run objective:

```csharp
string optimizer = t.SuggestCategorical("optimizer", new[] { "adam", "sgd" });
if (optimizer == "adam") t.SuggestFloat("beta1", 0.5, 0.999);   // only exists under adam
else                     t.SuggestFloat("momentum", 0.0, 0.99); // only exists under sgd
```

The first `sgd` trial has no `beta1`; the first `adam` trial has no `momentum`. So the intersection is
`{optimizer}` forever after: both `beta1` and `momentum` become **independent** parameters, redrawn
uniformly at random on *every* subsequent trial. The GA can select for `optimizer` but can never tune
`beta1` or `momentum` — it cannot even reliably tell which optimizer is better, because neither
optimizer's own knobs are ever exploited. This is the gap DCPARA closes. (Optuna's answer for the *TPE*
sampler is `group=True`, which models co-occurring parameters as groups; the GA family has no analog.)

---

## 4. Contribution overview

1. **A declarative activation algebra** (`ConditionalParams`, §5): `When(activator, …).Activates(…)`,
   with OR within an activator, AND across activators, predicate activators, chains, Python `==`
   value semantics, and static cycle rejection.
2. **Three-seam integration** (§6) into the existing NSGA machinery with no change to selection,
   ranking, niching or the tournament.
3. **A math-neutrality guarantee** (§7): a deactivated parameter perturbs neither the produced child
   nor the RNG stream — verified against a counterfactual "the parameter never existed" world across
   all six crossovers, 400 seeds each.
4. **Dynamic conditional parameterization** (§8b): the `PreserveDormantValues` mode, under which a
   dormant gene is preserved and resumed rather than relearned — while remaining out of every equation
   whilst dormant.
5. **A strict parity contract** (§8): the whole feature is off by default and the untouched path stays
   bit-identical to Optuna, pinned by null-control tests.

---

## 5. The activation algebra

`ConditionalParams` is a builder of rules. A **rule** binds a *dependent* parameter to an *activator*
parameter plus an admissible-value test (a `Condition`).

```csharp
var rules = new ConditionalParams()
    .When("optimizer", "adam", "adamw").Activates("beta1", "beta2")   // value set
    .When("optimizer", "sgd").Activates("momentum")                    // single value
    .When("n_layers", v => (long)v! >= 3).Activates("width_3");        // predicate
```

**Semantics.**

- **OR within an activator.** Several `When(a, …)` rules for the same dependent and same activator `a`
  are satisfied if *any* admits (value sets are the common case: `a ∈ {v1, v2}`).
- **AND across activators.** A dependent declared under two different activators is active only when
  *every* activator is present, admits, and is itself active.
- **Chains.** An activator may itself be conditional; activity is transitive. Self-activation and
  cycles are rejected at declaration time (`ValueError`), enforced by a dependency walk (`DependsOn`)
  and a topological order (`ComputeOrder`, Kahn's algorithm, ties by first mention → deterministic).
- **Value equality** uses Python `==` via `CategoricalChoice` (`1 == 1.0 == true`, strings ordinal,
  `NaN` matches `NaN` as Optuna's categorical lookup does). A predicate instead receives the value as
  the trial stores it: a boxed `long` for int params, a `double` for float params, the raw choice for
  categorical.

**Two activity queries** distinguish complete vs partial assignments — this distinction is what makes
the integration correct:

```text
IsActive(name, params)      // total: every activator present & admits & itself active  -> bool
ResolveActive(name, assign) // partial: bool, or NULL when an activator is not yet present
```

`ResolveActive` returns `null` (undecidable) when an activator has not been drawn yet — e.g. it was
dropped for mutation and will be sampled independently. The sampler treats `false` as "definitely
inactive" (safe to drop) and `null` as "unknown" (keep; let the objective decide).

Public surface: `When`, `Activates`, `Parameters`, `IsConditional`, `ActivatorsOf`, `IsActive`,
`PreserveDormantValues` (init). Internal: `ResolveActive`, `ActivatorAdmits`, `Prune`,
`ApplyDormantCarry`, `Contradictions`, `Order`.

---

## 6. Integration seams

The genetic core is unchanged. Only three parameter-set-shaping steps become conditional-aware, each
behind a `conditionalParams is null` guard so the default path is untouched.

### 6.1 Search space — `SearchSpaceFunctions.CalculateConditional`

The intersection fold is replaced (only when rules are present) by a fold that treats a conditional
parameter's *absence* as expected:

```text
for each finished trial t, newest→oldest (same incremental cache as the intersection):
    unconditional param p:   kept iff every trial so far had p with the same distribution   (intersection)
    conditional param c:
        absent from t              -> KEEP (c was inactive in t)
        present, same distribution -> keep / first-sighting adds it
        present, different dist.   -> TOMBSTONE c for the study's life (a dynamic range is not evolvable)
    complete trial contradicting the rules (c present while inactive, or absent while active)
                               -> warn once per parameter (rules vs objective drifted apart)
```

The result is order-independent, like the intersection: a conditional parameter is in the space iff
some trial suggested it and every trial that did agreed on its distribution. `IntersectionSearchSpace`
carries the rules, the excluded (tombstoned) set, and the warned set as part of its incremental cache
state.

### 6.2 Crossover — `NsgaiiCrossover.PerformConditionalCrossover`

Parents may lack conditional genes. The operators run over what the selected parents actually share:

- **Categorical** uniform crossover runs over the keys the first and last parent *both* carry (one mask
  draw per shared key, in search-space order — identical draws to the parity path when nothing is
  missing). A key only one parent carries is inherited from it.
- **Numerical** crossover runs over the keys *every* parent carries (a well-formed parents × dims
  matrix in the transformed space, the transform built per parent set). A key only *some* parents carry
  is inherited from one, drawn uniformly (`rng.randint`). A key no parent carries stays absent.
- After the operators, `ConditionalParams.Prune` removes every gene inactive for the child (walking in
  dependency order, so a pruned activator takes its dependents with it), and the box-containment
  redraw loop is unchanged.

### 6.3 Child generation — `NSGAIIChildGenerationStrategy`

- The **crossover branch** dispatches to `PerformConditionalCrossover`.
- The **no-crossover branch** copies exactly the genes the chosen parent carries, then prunes (in case
  the parent's own branches and the declared rules disagree).
- The **mutation pass** is unchanged, and this is the subtle win: because inactive genes were pruned
  *before* it, they neither count toward the default `1/n_params` mutation probability nor get a
  mutation draw.

### 6.4 The objective stays the final word

DCPARA shapes the sampler's *proposal*. The define-by-run contract is preserved end-to-end: a proposed
gene the objective never suggests is never recorded; a gene the objective suggests that the sampler did
not propose is drawn independently. The rules are guidance to the evolver, not a constraint on the
objective.

### 6.5 Wiring

Both samplers take a trailing optional `conditionalParams:` argument and store it; `SampleRelative`
calls the (possibly conditional) child-generation strategy and then `ApplyDormantCarry` (§8b). A custom
`childGenerationStrategy` receives the widened search space and must cope with absent parent genes
itself.

---

## 7. Math-neutrality of deactivation

**Claim.** For a child produced on one branch, every parameter that is *definitely inactive* for that
child influences neither the produced individual nor the RNG state afterwards.

**Falsifiable form (the test).** Generate a child in the full world; generate again in a counterfactual
world where the deactivated parameter was removed from the parents, the search space, and the rules;
require the child (keys, order, bits) **and the next RNG draw** to be bit-identical.
(`ConditionalParamsMathNeutralityTests`, 400 seeds × {uniform, blx, sbx, spx, undx}.)

**What is neutral by construction.** Selection, non-domination ranking, crowding distance /
reference-point niching, and the parent tournament read only objective values and population indices —
parameter-independent. The mutation rate uses the *active*-parameter count and the mutation loop
iterates the active parameters, so a deactivated gene is simply not there.

**What required care — crossover, and the multi-parent fix.** A gene that will be pruned anyway must
not enter the operator (it would change dimensionality or the operator's internal draws for the other
genes) nor trigger the partial-carrier `rng.randint`. The prune-aware fix (`71b6610`) drops a
*definitely-inactive* numerical gene (`ResolveActive(...) == false`) **before** any crossover work.
Before the fix, multi-parent crossovers (SPX, UNDX) diverged on ~93 / ~99 of ~275 deactivating children
per 400-seed probe (a to-be-pruned gene consumed a partial-carrier draw); after it, zero across all
five value-independent crossovers.

**The one documented residual.** The conditional crossover rejects out-of-box children and redraws, so
an operator that can overshoot the box (vSBX) may let a gene that is *active on a rejected attempt*
change the redraw count. This is an **active**-parameter effect on a discarded attempt, not a
deactivated-parameter effect; it is pinned and explained in
`ConditionalParamsMathNeutralityTests.BoxRejectingCrossover_CanCoupleViaActiveRejectedAttempts`.

---

## 8. Parity contract (the null control)

The feature is inert unless rules are declared, and pinned so by `ConditionalParamsGateTests`:

- `WithoutActivation_ReproducesOptunaBaseline_BitExact` — no rules ⇒ the sampler's whole trajectory is
  bit-exact against the plain intersection path.
- `EmptyRules_AreABitExactNoOp_OnABranchyStudy` — an empty `ConditionalParams` on a genuinely branchy
  study changes nothing, over {NSGA-II, NSGA-III} × {uniform, sbx}.
- `Activation_IsPerParameter_UndeclaredSiblingStaysAtBaseline` — declaring one parameter conditional
  does not perturb an undeclared sibling.
- `ConditionalParamsTests.RulesThatNeverFire_KeepTheOptunaSequence_BitExact` — a rule whose activator
  admits every value it ever takes (so it never fires) reproduces the hardcoded Optuna 5.0.0.dev
  NSGA-III sequence bit-for-bit.

### 8b. Dynamic mode — dormant-value memory (`PreserveDormantValues`)

An **opt-in divergence** from Optuna (Optuna always re-samples). When on, a conditional gene inactive
for a child does not vanish: its last active value is parked in a latent side channel — a trial *system
attribute*, `ConditionalParams:dormant` — and a reactivating branch **resumes from that value instead
of a fresh random draw**.

```text
ApplyDormantCarry(study, trial, child, parents, searchSpace):     # runs only if PreserveDormantValues
    latent = GatherLatent(parents)          # newest parent wins; from active params AND parked dormant maps
    if latent empty: return                 # e.g. generation 0 — nothing learned yet
    for name in Order:                      # dependency order
        a = ResolveActive(name, child)
        if a is null:  continue             # undecided activator -> leave to independent sampling
        if a is true and name not in child and latent has name and value in-range:
            child[name] = latent[name]      # RESUME a reactivating branch from its learned value
        if a is false and latent has name:
            dormant[name] = latent[name]    # PARK the value, untouched, out of the active genome
    if dormant nonempty: store dormant in trial system attr DormantAttrKey
```

Key invariants:

- **Out of every equation while dormant.** A parked value is never suggested, never scored, never
  crossed or mutated — it lives only in a system attribute. Dormancy freezes the value; it does not
  reintroduce it into the genome.
- **Non-destructive fluidity.** The hyperparameter list is genuinely dynamic — genes move between the
  active genome and the dormant channel as branches toggle — but a branch turning off and on again does
  not throw away what was learned about it.
- **Consumes no sampler RNG** and writes at most one system attribute; a resumed gene is an ordinary
  active gene from that point (free to be crossed and mutated). A gene that was never active has nothing
  to resume, so its first activation is still a normal draw.
- **Storage requirement.** The chosen storage must round-trip a `{name: value}` map through trial
  system attributes; in-memory and JSON storages do. `CoerceDormant` reads the map back regardless of
  how the backend re-typed it.

Pinned by `ConditionalParamsDormantMemoryTests`: unit-level resume / park / no-op (`ApplyDormantCarry`
driven directly), plus end-to-end studies showing the flag is the *only* behavioral change and that it
reduces random reactivations.

---

## 9. Empirical evidence (on the surrogate probes)

All figures below come from the deterministic surrogate objectives in `test/` and `examples/` (real
search-space shapes, closed-form values). They demonstrate the mechanism; they are **not** a benchmark
of DCPARA on real workloads (§13).

### 9.1 Controllability

`ConditionalParamsProofTests.Controllable_EveryRuleKind_DrivesTheSamplersOwnProposals`: on a study
mixing all rule kinds (value set, single value, numeric predicate, AND across activators, and a 3-level
chain), each trial's *proposal* (captured before the objective suggests) satisfies, for every declared
gene: never proposed next to an activator value that deactivates it; a proposed-and-suggested gene is
bit-for-bit what the trial records; the trial holds a gene iff the rules say it is active. Every rule
kind is exercised both ways (proposed and withheld) many times.

### 9.2 Learning while active vs random under the plain intersection

`ConditionalParamsProofTests.Learns_WhenActive_StaysRandom_UnderThePlainIntersection` (NSGA-II and
NSGA-III; two-task objective where `beta1` lives under `adam` and `momentum` under `sgd`, both branches
kept on the Pareto front; pooled late generations). Distance = |value − optimum|; random-draw medians
are 0.150 (`beta1`) and 0.245 (`momentum`).

| gene | with rules (median / within-0.05) | plain intersection (median / within-0.05) |
|------|-----------------------------------|--------------------------------------------|
| `beta1` | < 0.05 / > 50% | > 0.08 / < 35% |
| `momentum` | < 0.06 / > 50% | > 0.15 / < 25% |
| `lr` (unconditional) | < 0.05 | < 0.05 (unchanged either way) |

with-rules is required to be ≥ 2.5× closer than plain, and generation 0 is bit-identical between the
two arms (same seed) — the divergence is the rules. A 5-seed calibration (`examples/ConditionalParameters.cs`)
put the pooled late medians at roughly `beta1` 0.04→0.01 and `momentum` 0.03→0.02 (rules) vs 0.14–0.16
and 0.23–0.28 (plain), across default and 0.1 mutation rates.

### 9.3 Dormancy is bit-identical to non-existence

`ConditionalParamsProofTests.Dormant_WhenDeactivated_…` and `ConditionalParamsMathNeutralityTests`: 300
/ 400 seeds, child + next RNG draw bit-identical to the counterfactual world without the gene, across
uniform and SBX (proof) and all five value-independent crossovers (neutrality).

### 9.4 The examples

- `ConditionalModelSelection.cs` — picks the best model and tunes only its knobs: the printed best trial
  is `model=mlp`, error 0.0209, `mlp_units=116`, `mlp_lr≈0.00101`, `mlp_beta1≈0.888` (near the surrogate's
  mlp optimum 128 / 1e-3 / 0.90). With the knobs learnable the population concentrates on the best model;
  under the plain intersection it cannot, because no model's own knobs are ever exploited. (An earlier
  diagnostic measured the late-generation mlp share at ≈83% with rules vs ≈35% without; the committed
  example prints the best trial rather than the share.)
- `ConditionalNeuralArchitecture.cs` — numeric-predicate depth + scheduler chain; the Pareto front's
  lowest-error architecture per depth carries exactly its own widths (depth 1→4 errors 0.241, 0.212,
  0.169, 0.096).
- `ConditionalPipelineChain.cs` — enum-array optional chain; the front spans empty → `Pool` →
  `Dense→Dense` → a full 5-stage pipeline (error 0.020).
- `ConditionalTrainingSchedule.cs` — AND/OR/predicate composition; `warmup_epochs` present iff scheduler
  warms up *and* the run is long enough.

---

## 10. Complexity & overhead

Per child, the conditional path adds: an `IsActive`/`ResolveActive` walk (O(rules) with memoized
topological `Order`), a `Prune` pass (O(conditional genes)), and — in dynamic mode — one `GatherLatent`
pass over parents and at most one system-attribute write. No extra sampler RNG draws are introduced on
the neutral path (§7). When no rules are declared none of this runs.

---

## 11. Relationship to prior work

- **Optuna define-by-run & `TPESampler(group=True)`** — Optuna models conditional co-occurrence for
  TPE via groups; DCPARA is the missing analog for the evolutionary (NSGA) family, expressed
  declaratively rather than inferred.
- **NSGA-II / NSGA-III** (Deb et al. 2002; Deb & Jain 2014) — the underlying multi-objective GAs;
  DCPARA leaves their selection/ranking/niching untouched and only reshapes the genome per individual.
- **Hierarchical / conditional genomes in evolutionary computation** — variable-length and
  conditionally-active genes are classical in GP and neuroevolution; DCPARA's contribution is a precise
  *math-neutrality* integration into an existing bit-exact sampler plus the *non-destructive dormant
  memory* variant.

---

## 12. Limitations & threats to validity

- **Evidence is on surrogates.** §9 demonstrates the mechanism on closed-form objectives; no real-task
  benchmark yet.
- **The box-rejection residual** (§7) means DCPARA is not *unconditionally* RNG-neutral for
  box-overshooting operators (vSBX) when the coupling is through an *active* gene on a rejected attempt.
- **Dynamic mode is a deliberate divergence** from Optuna and requires system-attribute round-tripping.
- **Rule/objective drift** is possible: if the declared rules and the objective's branches disagree,
  the sampler warns once per parameter but keeps running (the objective wins).
- **Only the built-in child generation is conditional-aware**; custom strategies must handle absent
  genes themselves.

---

## 13. Toward the paper — open questions & future work

1. **Real benchmarks.** DCPARA vs plain-intersection NSGA-II/III on genuine conditional spaces
   (sklearn/xgboost model selection, PyTorch NAS), measuring convergence and final Pareto quality.
2. **Theory of dormant memory.** When does resuming a dormant gene help vs hurt? Characterize regimes
   (branch-toggle frequency, gene-optimum stability) where non-destructive fluidity beats re-sampling.
3. **Interaction with reference-point niching.** Does a fluctuating active dimensionality interact with
   NSGA-III's normalization / niche assignment? (Selection is value-only, so the hypothesis is "no",
   but this is worth measuring.)
4. **Ablations.** Inheritance-only vs +mutation vs +dormant memory; per-rule-kind contribution.
5. **Beyond GA.** A conditional-aware relative search space for CMA-ES / GP samplers.
6. **Formalization.** State the math-neutrality property as a theorem over the operator algebra and
   prove the box-rejection residual is the unique exception.

---

## 14. Reproducibility

```bash
# unit + proof + neutrality + gate + dormant tests (net10.0), the always-real gate:
dotnet test test/OptunaSharp.Tests/OptunaSharp.Tests.csproj -c Release -f net10.0 \
  --filter "FullyQualifiedName~ConditionalParams"

# NSGA-II/III live-Optuna parity (unchanged by this work) — needs Python+optuna:
OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE=1 \
  dotnet test test/OptunaSharp.Tests.Interop/OptunaSharp.Tests.Interop.csproj -c Release -f net10.0 \
  --filter "FullyQualifiedName~Nsgaii"

# the runnable recipes:
dotnet run examples/ConditionalParameters.cs        # the printed learning A/B (rules vs plain)
dotnet run examples/ConditionalModelSelection.cs    # value-set rules + a chain
dotnet run examples/ConditionalNeuralArchitecture.cs# numeric predicates + a chain, on a front
dotnet run examples/ConditionalTrainingSchedule.cs  # OR / AND / predicate composition
dotnet run examples/ConditionalPipelineChain.cs     # enum-array optional chain + dormant memory
```

Tests: `ConditionalParamsTests` (algebra, prune, search space, sampler smoke, never-fire bit-exact),
`ConditionalParamsProofTests` (controllable / learns / dormant), `ConditionalParamsGateTests`
(null control), `ConditionalParamsMathNeutralityTests` (counterfactual bit-identity),
`ConditionalParamsDormantMemoryTests` (dynamic mode).

---

## 15. Glossary

- **Activator** — a parameter whose value gates whether a dependent parameter is active.
- **Active / inactive / undecidable** — a dependent's tri-state activity against an assignment; the
  last arises when an activator has not been drawn yet.
- **Intersection search space** — parameters present, with identical distribution, in every completed
  trial; what the GA evolves by default.
- **Math-neutral** — a deactivated parameter leaves the produced child and the RNG stream bit-identical
  to a world without it.
- **Dormant / latent value** — a conditional gene's last active value, parked in a trial system
  attribute while its branch is off, out of every equation.
- **DCPARA** — Dynamic Conditional PARAmeterization: conditional activation (static) plus optional
  non-destructive dormant memory (dynamic).

---

*This document is maintained alongside the code; when the behavior above changes, update it in the same
commit. See `docs/website-src/docs/samplers.md` (§ Conditional parameters) for the user-facing guide and
`docs/PORTING_PLAN.md` (entry 2.14-ext) for the port ledger.*
