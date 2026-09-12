# NSGA-III — the complete algorithm sourcebook

This is a verbatim snapshot of the local OptunaSharp NSGA-III implementation and its explicitly listed algorithm/support files. **43 complete files · 8,378 physical lines.** Every included file retains its comments, blank lines, imports, declarations, and implementation. Nothing inside a file is elided or rewritten.

Git HEAD: `07adb147268a4f436b2bd6abfadd428b9d8e77ac`. Snapshot: `2026-09-10T05:22:55.160Z`. Aggregate source SHA-256: `edafc2c2fc63cb5aefdd6bee0217da40c79cbe0d998cdbca7a0e3f722340b3b8`.

The bytes come from the current working tree, not a Git checkout reconstructed at HEAD. All included files match HEAD. Per-file SHA-256 values hash the original bytes, and UTF-8 decoding is checked by byte-for-byte re-encoding. Line endings are preserved in the machine-readable bundle; HTML renders them as numbered physical lines.

## Scope and dependency boundary

Every line of NSGAIIISampler, NSGAIIIElitePopulationSelectionStrategy, BaseGASampler, BaseSampler (including constraint post-processing), RandomSampler, LazyRandomState, the entire shared Nsgaii subtree, nondominated sorting, constraint evaluation, intersection and conditional search spaces, transforms, distributions, ordered dictionaries, and the listed numerical/formatting support files.

This is **not the entire .NET dependency closure** and is not a standalone compilable project. The following boundary remains outside the source bundle:

- Study, Trial, FrozenTrial, Optuna entry points, trial states/directions, objective evaluation, optimization orchestration, storage implementations, logging, project/assembly setup, and other OptunaSharp framework services.
- NumSharp NDArray, NumPyRandom/MT19937 implementation, NumSharp exceptions, NumSharp.Interop.OpenBLAS, native BLAS/LAPACK, .NET runtime/BCL, and their transitive dependencies.
- Other samplers (including NSGAIISampler itself), tests, the upstream Python implementation, and this presentation host/JavaScript; these are not the NSGA-III algorithm source bundle.

- Whole files are included even when only one method is used by NSGA-III. In particular NSGA-II crowding-distance selection is present because the same file supplies RankPopulation; NSGA-III does not invoke that crowding-distance path.
- The default sampler creates UniformCrossover and mutates by independent resampling. The presentation host configures SBXCrossover and PolynomialMutation; both full operator implementations are included.
- ConditionalParams is an opt-in OptunaSharp extension. It preserves eligible conditional parameters for evolution; its disabled path retains the standard intersection-space behavior.
- Source comments are reproduced as written, including historical parity statements. This source archive is not a new claim that every comment has been independently verified.
- The presentation replays a real completed backend generation. Diagnostic stages and highlighted methods explain those results; they are not instruction-level debugger stepping or a live CPU program counter.
- The eight stages form a teaching-level flow graph. Objective evaluation happens in Study/the user objective outside this archive; member ranges link the actual implementation responsible for each step.

## Regenerate and verify

Run from the repository root with Node.js:

```sh
node docs/presentations/build-sourcebook.mjs
node docs/presentations/build-sourcebook.mjs --check
```

The generator uses only Node standard-library modules. It reads full source files, discovers every .cs file beneath the shared Nsgaii directory, computes hashes and symbol ranges, and regenerates this Markdown document, the printable HTML sourcebook, its CSS, the served Markdown copy, and source-data.json. The check mode writes nothing and fails if any source or generated output has drifted. Unchanged source/revision retains its snapshot timestamp; set SOURCE_DATE_EPOCH to pin that timestamp for a reproducible build.

source-data.json is the lossless download: each files[].content string re-encodes to the original UTF-8 bytes and SHA-256. HTML line anchors are `#<file-id>-L<number>`. Symbol IDs are stable semantic names; overloaded members have ordinal suffixes. Stage links are regenerated from actual member boundaries, not hard-coded line numbers.

## Algorithm map

### Initialize a population

Initial parameters are sampled independently within their distributions. Study evaluates each suggested trial; the sampler does not evaluate the objective.

`xⱼ ∼ distributionⱼ; P₀ = {evaluate(x₁), …, evaluate(xₙ)}`

Implementation: `src/OptunaSharp.Core/Samplers/NSGAIIISampler.cs`, lines 193–199 (`sampler.sample-independent`).

- N: population size
- x: decision vector
- P₀: evaluated initial population

### Breed and mutate

Dominance tournaments select parents. The configured crossover produces a child, or one parent is copied. Parameters mutate with the configured probability; absent proposals use independent sampling.

`Qₜ = evaluate(mutate(crossover(Pₜ))); pₘ = configured value or 1 / max(1, n_params)`

Implementation: `src/OptunaSharp.Core/Samplers/Nsgaii/NSGAIIChildGenerationStrategy.cs`, lines 74–137 (`child.generate`).

- Pₜ: selected parents
- Qₜ: evaluated offspring
- pₘ: mutation probability

### Combine parents and offspring

SelectParent gathers the previous generation and its cached parent population, then invokes the NSGA-III elite strategy.

`Rₜ = Qₜ ∪ Pₜ`

Implementation: `src/OptunaSharp.Core/Samplers/NSGAIIISampler.cs`, lines 176–181 (`sampler.select-parent`).

- Rₜ: combined candidate population
- Qₜ: previous completed generation
- Pₜ: previous cached parent population

### Sort by nondomination

Ranking converts maximization objectives to losses and applies nondominated sorting. When constrained, feasible trials precede infeasible trials ranked by total positive violation.

`a ≺ b ⇔ (∀j: aⱼ ≤ bⱼ) ∧ (∃j: aⱼ < bⱼ); Rₜ → F₀, F₁, …`

Implementation: `src/OptunaSharp.Core/Samplers/Nsgaii/NSGAIIElitePopulationSelectionStrategy.cs`, lines 73–108 (`sorting.rank-population`).

- Fᵢ: nondomination front
- a, b: loss vectors
- penalty: Σ max(0, constraint)

### Normalize the selection set

After replacing infinities, subtract the ideal point, find ASF extreme points, and solve the extreme-point hyperplane. A degenerate matrix uses per-axis maxima instead. The selection set contains accepted fronts plus the borderline front.

`zⱼ = minᵢ fᵢⱼ; Eα = 1; yᵢⱼ = (fᵢⱼ − zⱼ) αⱼ`

Implementation: `src/OptunaSharp.Core/Samplers/Nsgaiii/NSGAIIIElitePopulationSelectionStrategy.cs`, lines 197–274 (`selection.normalize-objective-values`).

- z: ideal point
- E: ASF extreme-point matrix
- α: inverse intercepts
- y: normalized objective vector

### Associate with a reference direction

Associate every normalized individual with the reference line having minimum perpendicular distance. Exact ties keep the first reference row.

`πᵣ(y) = ((y · r) / ‖r‖²) r; d(y,r) = ‖y − πᵣ(y)‖; r* = argminᵣ d(y,r)`

Implementation: `src/OptunaSharp.Core/Samplers/Nsgaiii/NSGAIIIElitePopulationSelectionStrategy.cs`, lines 280–326 (`selection.associate-individuals-with-reference-points`).

- r: reference vector
- πᵣ: orthogonal projection onto span(r)
- d: perpendicular distance

### Fill the least occupied niches

Only accepted fronts initially contribute occupancy. Among reference directions with borderline candidates, choose a least occupied direction. An empty niche takes its nearest candidate; a populated niche uses the sampler’s shuffled order.

`r* ∈ argminᵣ ρᵣ; ρᵣ = 0 ⇒ nearest candidate; ρᵣ > 0 ⇒ shuffled candidate; ρᵣ ← ρᵣ + 1`

Implementation: `src/OptunaSharp.Core/Samplers/Nsgaiii/NSGAIIIElitePopulationSelectionStrategy.cs`, lines 332–416 (`selection.preserve-niche-individuals`).

- ρᵣ: selected occupancy of direction r
- K: remaining survivor slots
- borderline: front that reaches or exceeds N

### Keep the next parent population

Append the niche-selected members to all accepted earlier fronts and return the elite population. BaseGASampler caches these parents for subsequent child generation.

`Pₜ₊₁ = accepted fronts ∪ niche-selected members; |Pₜ₊₁| = N`

Implementation: `src/OptunaSharp.Core/Samplers/Nsgaiii/NSGAIIIElitePopulationSelectionStrategy.cs`, lines 65–109 (`selection.select`).

- Pₜ₊₁: next parent population
- N: target size, when enough completed candidates exist

## Source manifest

| File | Lines | SHA-256 of original bytes |
| --- | ---: | --- |
| [src/OptunaSharp.Core/Samplers/NSGAIIISampler.cs](#sampler) | 213 | `6f072670bbc061d724ecaf601750af3f5fdabcf3b46a760ea9db1200c02036a0` |
| [src/OptunaSharp.Core/Samplers/Nsgaiii/NSGAIIIElitePopulationSelectionStrategy.cs](#selection) | 445 | `51bd2f431f2b87e96a88f9fca38e117211756f978fe32dc30074095e6d589d20` |
| [src/OptunaSharp.Core/Samplers/BaseGASampler.cs](#ga) | 124 | `d6c473d3ea512dc507ea728b6abb65a1f9e728203afae29c9c0ddef0646875af` |
| [src/OptunaSharp.Core/Samplers/BaseSampler.cs](#base-sampler) | 134 | `292a35ddd3ab83b73ca1335447cba4d6e424cf03c0123dc0f7d3b10fd9db64c5` |
| [src/OptunaSharp.Core/Samplers/RandomSampler.cs](#random) | 61 | `64a17cfd141b3a4e81cfe8921e460e0dd940396f272a604392653f6d2381a549` |
| [src/OptunaSharp.Core/Samplers/LazyRandomState.cs](#rng) | 41 | `c5265882445ed6686cdd298a1a0ab3f6ec5969750b3da0ef451b669723694502` |
| [src/OptunaSharp.Core/Internal/MultiObjective.cs](#multiobjective) | 346 | `0a8fada3ec121dd4dff0324a960a0eb677771011b250acbfc08685ef69e6950d` |
| [src/OptunaSharp.Core/Internal/ConstrainedOptimization.cs](#constraints) | 63 | `694ee4258cc84be5cab641538033c7ea34e1309f13aa137355e5a2e565626637` |
| [src/OptunaSharp.Core/SearchSpace/IntersectionSearchSpace.cs](#intersection) | 311 | `543993fbde4cae98f8a0eb5b7d291fa5c749cfddde131a6c3b9902d898a15c3b` |
| [src/OptunaSharp.Core/SearchSpace/ConditionalParams.cs](#conditional) | 394 | `3ccfe6bc6600a586fdee9ad8cd7a104db2eedef7187541c410f23d7df5d19bd0` |
| [src/OptunaSharp.Core/Internal/SearchSpaceTransform.cs](#transform) | 343 | `3e77d0a552eee249e5c5b5b32c8e3154abcda412cb93f2f90581329f584c1c32` |
| [src/OptunaSharp.Core/Internal/LinAlg.cs](#linalg) | 177 | `29e3bc0a8c0b235100d75d344436f915b9170286b8ae8ec6b78c1bcb9426e6e6` |
| [src/OptunaSharp.Core/Internal/MathUtil.cs](#math) | 187 | `33e4d1b2f84685a56e5b7ca9422371ec83da27de00cf1912d958a03c69700193` |
| [src/OptunaSharp.Core/Internal/OpenBlasActivation.cs](#openblas) | 32 | `9c2b9b595bf5a2bd9a7d56665fec70268454fe913f8c29b94afb3ed2bdbaf9ce` |
| [src/OptunaSharp.Core/Internal/PythonRepr.cs](#python-repr) | 260 | `886dfa3deace6ee47e5e12e67780224f1cd6e57f08a7eb27f1881b23020cca27` |
| [src/OptunaSharp.Core/Internal/PyJson.cs](#json) | 553 | `1f66d96587518cabba575430f051a7469180153f93680553d1b8a00943bfd1ae` |
| [src/OptunaSharp.Core/Internal/OptunaWarnings.cs](#warnings) | 46 | `24adf5789ff20991622d1bf4a5a542e7b175336edef0cf9ebfd2269ad4aa81bc` |
| [src/OptunaSharp.Core/Samplers/Nsgaii/Crossovers/BLXAlphaCrossover.cs](#blx-alpha-crossover) | 42 | `f5ad867d38ec2af8b80ca02c6f71fd8074324a08993490357a4900cf86739fb7` |
| [src/OptunaSharp.Core/Samplers/Nsgaii/Crossovers/BaseCrossover.cs](#base-crossover) | 39 | `54d9986a4abcbe875daf9f1d31de147b5689b9436766041100f3ad9ef25374ce` |
| [src/OptunaSharp.Core/Samplers/Nsgaii/Crossovers/SBXCrossover.cs](#sbx-crossover) | 78 | `c6398cb98c598d0d193caca26f7933fa336742ea7c825be298a8afa06f68c61b` |
| [src/OptunaSharp.Core/Samplers/Nsgaii/Crossovers/SPXCrossover.cs](#spx-crossover) | 73 | `179bd660088234ddca8bf6aa624cbf6f1bbe63e2e2e2e8112d2753535dd82678` |
| [src/OptunaSharp.Core/Samplers/Nsgaii/Crossovers/SbxLike.cs](#sbx-like) | 53 | `59efd6c04a53f44b8bce5af8a5dbe23105a84297d7fc3ae41ec718bb58fa2aa2` |
| [src/OptunaSharp.Core/Samplers/Nsgaii/Crossovers/UNDXCrossover.cs](#undx-crossover) | 226 | `ebf38e6abf6534870145dee95f425363e939289b9d10fe2cb395d21e476d5301` |
| [src/OptunaSharp.Core/Samplers/Nsgaii/Crossovers/UniformCrossover.cs](#uniform-crossover) | 41 | `f6e378e1ed319d415cba7b8a7549c9e8caf6ae6e1ac97741eb7a14462fd91c80` |
| [src/OptunaSharp.Core/Samplers/Nsgaii/Crossovers/VSBXCrossover.cs](#vsbx-crossover) | 77 | `9b9ae6b622fc8466363d07beec17bbb3067dd21d00963eb741eb2a9562d3d8f8` |
| [src/OptunaSharp.Core/Samplers/Nsgaii/Mutations/BaseMutation.cs](#base-mutation) | 26 | `1560c7a4fe02c68b62fc102edd874a94b06d3423c920549fd2d3ef014c56714f` |
| [src/OptunaSharp.Core/Samplers/Nsgaii/Mutations/PolynomialMutation.cs](#polynomial-mutation) | 55 | `bba077a6296ed13540c861b5caa4851007214660b4707473092f2ab49d26e05a` |
| [src/OptunaSharp.Core/Samplers/Nsgaii/NSGAIIAfterTrialStrategy.cs](#after-trial) | 27 | `5ad8afc1e4c0f894bff605b9698f20d47b7105106415450ca1c946c447d4b877` |
| [src/OptunaSharp.Core/Samplers/Nsgaii/NSGAIIChildGenerationStrategy.cs](#child) | 138 | `a0e6c6a53ba09bf2a7e9bd21b172ed09e9ebd8e1625b435e663bec78bfca658f` |
| [src/OptunaSharp.Core/Samplers/Nsgaii/NSGAIIElitePopulationSelectionStrategy.cs](#sorting) | 187 | `b8d8ffe6567104d530ed8bed2b4911b00865008e52abab27feefc878ba8a7823` |
| [src/OptunaSharp.Core/Samplers/Nsgaii/NsgaiiConstraintsEvaluation.cs](#constraint-evaluation) | 73 | `6165216b22521dfdca27a7594bc4bcda8796df51bb44372fdad04e456286c792` |
| [src/OptunaSharp.Core/Samplers/Nsgaii/NsgaiiCrossover.cs](#crossover) | 294 | `9f76510aae39a49e69999bcb2dc8d4ec46559a9d31b08a72f7aeda9d6fdd1dc4` |
| [src/OptunaSharp.Core/Samplers/Nsgaii/NsgaiiMutation.cs](#mutation) | 31 | `9c8ead1f8ae1b940505c620a3ea158970f41541d0a3ded12d58c96444830387a` |
| [src/OptunaSharp.Core/Distributions/BaseDistribution.cs](#distribution-base-distribution) | 84 | `dd5406b8dbd0719c1620e05b9499706a5794dc2069dd80ed48c445f8f2b4efc2` |
| [src/OptunaSharp.Core/Distributions/CategoricalChoice.cs](#distribution-categorical-choice) | 154 | `658492283aaab3ef846ef104bd244497f220df58f9866aa91e86432f02613a92` |
| [src/OptunaSharp.Core/Distributions/CategoricalDistribution.cs](#distribution-categorical-distribution) | 103 | `a1141b82b8e881c9126c7e04a3d3d2ed8e02ddf69bb85087c1ab2da6e42e9298` |
| [src/OptunaSharp.Core/Distributions/Distribution.cs](#distribution-distribution) | 263 | `b1a2c778db06732120848406286a2734e50ff9849064a8b405991e3ba21bcfd1` |
| [src/OptunaSharp.Core/Distributions/FloatDistribution.cs](#distribution-float-distribution) | 275 | `7a4d5e09ee22a7f946e12c1bca83ddae15a17d73ef341bda6518321c0d723203` |
| [src/OptunaSharp.Core/Distributions/IntDistribution.cs](#distribution-int-distribution) | 156 | `290d8108938a6450431c9d5872fa628c006955af7d74122ade1bc8410745effc` |
| [src/OptunaSharp.Core/Collections/OrderedDict.Custom.cs](#ordered-dict-custom) | 15 | `0d6b7e7787a4953a19da28a9b3bf3500b861c8280eab0afadd4f4c9ed2fed615` |
| [src/OptunaSharp.Core/Collections/OrderedDict.Net8.cs](#ordered-dict-net8) | 1928 | `36c155129e375d509f113b03fbc25f8c8f4e5c22e758167ab4cb38f4a9976d64` |
| [src/OptunaSharp.Core/Collections/OrderedDict.cs](#ordered-dict) | 46 | `8d9282c93437c53f24e9902355e39e8dbb8d697f38c4f13a6b5828c1ef34c7eb` |
| [src/OptunaSharp.Core/Collections/OrderedDictInternal.Net8.cs](#ordered-dict-internal-net8) | 164 | `96fa65d5d8fa90dbd6d42867a0e0ca17e10af89424901838674cb70f5e365a2b` |

<a id="sampler"></a>

## src/OptunaSharp.Core/Samplers/NSGAIIISampler.cs

NSGA-III core · 213 physical lines · SHA-256: `6f072670bbc061d724ecaf601750af3f5fdabcf3b46a760ea9db1200c02036a0`

```csharp
using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.Samplers.Nsgaiii;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     Multi-objective sampler using the NSGA-III algorithm. Port of
///     <c>optuna/samplers/_nsgaiii/_sampler.py: NSGAIIISampler</c> (experimental since Optuna 3.2.0).
///
///     NSGA-III ("Nondominated Sorting Genetic Algorithm III") is a modification of
///     <see cref="NSGAIISampler" /> for many-objective optimization. It reuses the whole NSGA-II
///     genetic machinery (<see cref="BaseGASampler" />, child generation, crossovers, mutations) and
///     only swaps the elite-selection strategy: instead of crowding distance, the borderline front is
///     truncated by <b>reference-point niching</b> (see
///     <see cref="NSGAIIIElitePopulationSelectionStrategy" />). See
///     <a href="https://doi.org/10.1109/TEVC.2013.2281535">Deb &amp; Jain (2014)</a>.
///
///     <para><b>Parity.</b> The default path stays in exact MT19937 lock-step with Optuna (generation
///     attribution, parent cache, tournament/crossover/mutation draws, and the niching
///     <c>rng.shuffle</c>). Objective-value niching is bit-exact except for a ~1-ULP managed-vs-LAPACK
///     drift in the hyperplane-fit solve, which only changes selection when it flips a niche decision
///     within ULPs of a tie — see the <c>[Misaligned]</c> note on
///     <see cref="NSGAIIIElitePopulationSelectionStrategy" />.</para>
///
///     <para><b>Conditional parameters (OptunaSharp extension).</b> Optuna's GA samplers evolve only the
///     intersection search space, so a define-by-run parameter suggested under a branch drops out after
///     the first trial that lacks it and is resampled at random ever after. Passing
///     <see cref="ConditionalParams" /> declares which parameters are conditional on which activator
///     values; they then stay in the relative search space and are crossed over, mutated and inherited
///     whenever active, and are simply absent whenever not. Off by default (parity untouched).</para>
/// </summary>
public class NSGAIIISampler : BaseGASampler
{
    private readonly RandomSampler _randomSampler;
    private readonly LazyRandomState _rng;
    private readonly IntersectionSearchSpace _searchSpace;

    private readonly Func<Study, List<FrozenTrial>, List<FrozenTrial>> _elitePopulationSelectionStrategy;
    private readonly Func<Study, OrderedDict<string, BaseDistribution>, List<FrozenTrial>, OrderedDict<string, object?>>
        _childGenerationStrategy;
    private readonly Action<Study, FrozenTrial, TrialState, IReadOnlyList<double>?> _afterTrialStrategy;

    /// <param name="populationSize">
    ///     Number of individuals (trials) in a generation. Must be &gt;= <c>crossover.NParents</c>
    ///     (3 for <see cref="UNDXCrossover" />/<see cref="SPXCrossover" />, else 2).
    /// </param>
    /// <param name="mutation">
    ///     Mutation applied when creating child individuals. If <c>null</c>, the parameter selected
    ///     for mutation is instead resampled independently using <see cref="RandomSampler" />.
    /// </param>
    /// <param name="mutationProb">
    ///     Probability of mutating each parameter. If <c>null</c>, <c>1 / len(parent.params)</c> is used.
    /// </param>
    /// <param name="crossover">
    ///     Crossover applied when creating child individuals. <see cref="UniformCrossover" /> is
    ///     always applied to categorical parameters, and by default to the others.
    /// </param>
    /// <param name="crossoverProb">Probability that a crossover occurs when creating a new individual.</param>
    /// <param name="swappingProb">Probability of swapping each parameter of the parents during crossover.</param>
    /// <param name="seed">Seed for random number generator.</param>
    /// <param name="constraintsFunc">
    ///     Deprecated (v5.0.0). Optional function computing objective constraints (a value strictly
    ///     larger than 0 means a constraint is violated). Handled by constrained domination.
    /// </param>
    /// <param name="referencePoints">
    ///     A 2-D array with <c>n_objectives</c> columns: the reference points that determine who
    ///     survives from the borderline front (survival favors sparsely populated reference points).
    ///     When <c>null</c> (default), `uniformly` spread points are generated from
    ///     <paramref name="dividingParameter" />.
    /// </param>
    /// <param name="dividingParameter">
    ///     Determines the density of the default reference points — how many divisions are made
    ///     between reference points on each axis. Smaller means fewer reference points. Unused when
    ///     <paramref name="referencePoints" /> is not <c>null</c>.
    /// </param>
    /// <param name="elitePopulationSelectionStrategy">Overrides elite selection.</param>
    /// <param name="childGenerationStrategy">Overrides child generation.</param>
    /// <param name="afterTrialStrategy">Overrides the after-trial procedure.</param>
    /// <param name="conditionalParams">
    ///     OptunaSharp extension (no Optuna analog): activation rules for parameters that only exist under
    ///     some values of other parameters — see <see cref="ConditionalParams" />. Declared parameters are
    ///     evolved (crossed over, mutated, inherited) whenever active instead of falling out of the
    ///     intersection search space and being resampled at random on every trial. <c>null</c> (default)
    ///     keeps the Optuna-parity behavior. Only the built-in child generation honors it; a custom
    ///     <paramref name="childGenerationStrategy" /> receives the widened search space and must cope with
    ///     parents that lack conditional parameters itself.
    /// </param>
    public NSGAIIISampler(
        int populationSize = 50,
        BaseMutation? mutation = null,
        double? mutationProb = null,
        BaseCrossover? crossover = null,
        double crossoverProb = 0.9,
        double swappingProb = 0.5,
        int? seed = null,
        Func<FrozenTrial, IEnumerable<double>>? constraintsFunc = null,
        double[,]? referencePoints = null,
        int dividingParameter = 3,
        Func<Study, List<FrozenTrial>, List<FrozenTrial>>? elitePopulationSelectionStrategy = null,
        Func<Study, OrderedDict<string, BaseDistribution>, List<FrozenTrial>, OrderedDict<string, object?>>?
            childGenerationStrategy = null,
        Action<Study, FrozenTrial, TrialState, IReadOnlyList<double>?>? afterTrialStrategy = null,
        ConditionalParams? conditionalParams = null)
        : base(populationSize)
    {
        // @experimental_class("3.2.0") wraps __init__ and warns before any validation runs.
        OptunaWarnings.Warn(
            "NSGAIIISampler is experimental (supported from v3.2.0). The interface can change in the future.",
            WarningCategory.ExperimentalWarning);

        // TODO(ohta): Reconsider the default value of each parameter.
        if (populationSize < 2)
            throw new ValueError("`population_size` must be greater than or equal to 2.");

        if (constraintsFunc is not null)
        {
            string msg = string.Format(Deprecations.Template, "`constraints_func`", "5.0.0", "7.0.0");
            OptunaWarnings.Warn(
                $"{msg} Use `optuna.trial.Trial.set_constraint` instead.", WarningCategory.FutureWarning);
        }

        crossover ??= new UniformCrossover(swappingProb);

        if (populationSize < crossover.NParents)
            throw new ValueError(
                $"Using {crossover}," +
                $" the population size should be greater than or equal to {crossover.NParents}." +
                $" The specified `population_size` is {populationSize}.");

        _randomSampler = new RandomSampler(seed);
        _rng = new LazyRandomState(seed);
        _searchSpace = new IntersectionSearchSpace(conditionalParams: conditionalParams);

        _elitePopulationSelectionStrategy = elitePopulationSelectionStrategy
            ?? new NSGAIIIElitePopulationSelectionStrategy(
                populationSize, _rng, constraintsFunc, referencePoints, dividingParameter).Select;
        _childGenerationStrategy = childGenerationStrategy
            ?? new NSGAIIChildGenerationStrategy(
                mutation, mutationProb, crossover, crossoverProb, swappingProb, constraintsFunc, _rng,
                conditionalParams).Generate;
        _afterTrialStrategy = afterTrialStrategy
            ?? new NSGAIIAfterTrialStrategy(constraintsFunc).Call;
    }

    public override void ReseedRng()
    {
        _randomSampler.ReseedRng();
        // Python: self._rng.rng.seed() — reseed from OS entropy.
        Span<byte> buffer = stackalloc byte[4];
        RandomNumberGenerator.Fill(buffer);
        _rng.Rng.seed(BitConverter.ToUInt32(buffer));
    }

    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _searchSpace.Calculate(study))
        {
            // The untransform of _SearchSpaceTransform does not assume a single value, so single
            // value objects are sampled via sample_independent, not sample_relative.
            if (distribution.Single())
                continue;
            searchSpace[name] = distribution;
        }
        return searchSpace;
    }

    public override List<FrozenTrial> SelectParent(Study study, int generation)
    {
        var combined = GetPopulation(study, generation - 1);
        combined.AddRange(GetParentPopulation(study, generation - 1));
        return _elitePopulationSelectionStrategy(study, combined);
    }

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        int generation = GetTrialGeneration(study, trial);
        var parentPopulation = GetParentPopulation(study, generation);
        if (parentPopulation.Count == 0)
            return new OrderedDict<string, object?>();
        return _childGenerationStrategy(study, searchSpace, parentPopulation);
    }

    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        // Randomly sampled here: (1) a parameter in the initial/first generation, (2) a parameter to
        // mutate, (3) a parameter excluded from the intersection search space.
        return _randomSampler.SampleIndependent(study, trial, paramName, paramDistribution);
    }

    public override void BeforeTrial(Study study, FrozenTrial trial)
    {
        _randomSampler.BeforeTrial(study, trial);
    }

    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        System.Diagnostics.Debug.Assert(
            state is TrialState.Complete or TrialState.Fail or TrialState.Pruned);
        _afterTrialStrategy(study, trial, state, values);
        _randomSampler.AfterTrial(study, trial, state, values);
    }
}
```


<a id="selection"></a>

## src/OptunaSharp.Core/Samplers/Nsgaiii/NSGAIIIElitePopulationSelectionStrategy.cs

NSGA-III core · 445 physical lines · SHA-256: `51bd2f431f2b87e96a88f9fca38e117211756f978fe32dc30074095e6d589d20`

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Nsgaii;

namespace OptunaSharp.Samplers.Nsgaiii;

/// <summary>
///     The default elite-population selection strategy of <see cref="NSGAIIISampler" />. Port of
///     <c>optuna/samplers/_nsgaiii/_elite_population_selection_strategy.py</c>.
///
///     Ranks the combined population by (constrained) non-domination (shared
///     <see cref="NSGAIIElitePopulationSelectionStrategy.RankPopulation" />), then fills the elite
///     population rank by rank. The rank that overflows <c>population_size</c> — the borderline front —
///     is truncated by <b>reference-point niching</b> rather than crowding distance: objective values
///     are normalized onto a hyperplane (ideal point + achievement-scalarizing extreme points), each
///     individual is associated with its closest reference point, and survivors are drawn to keep the
///     reference points as evenly populated (sparse) as possible.
///
///     <para><b>Bit-exact.</b> The whole pipeline matches Optuna bit-for-bit: <c>_filter_inf</c>,
///     the ideal-point subtraction, the ASF <c>einsum</c>/argmin extreme-point pick, the
///     reference-point association (numpy's 4-operand <c>einsum</c> replicated term-by-term, verified
///     bit-identical over 2000 cases), the niching, and — since 2.15 — the hyperplane fit itself: the
///     degeneracy <see cref="LinAlg.MatrixRank" /> branch and the <see cref="LinAlg.Solve" /> route
///     through NumSharp's <c>np.linalg.matrix_rank</c>/<c>solve</c> over the OpenBLAS LAPACK backend,
///     which is byte-identical to NumPy (the earlier managed reciprocal-multiplier LU drifted ~1 ULP
///     on a few percent of the extreme-point systems; the LAPACK route removes even that). Every RNG
///     draw (<c>rng.shuffle</c>) stays in lock-step. Confirmed byte-for-byte over a 400-config
///     differential (5 crossovers × {2,3,4} objectives × 25 seeds + 25 constrained).</para>
/// </summary>
public class NSGAIIIElitePopulationSelectionStrategy
{
    // Define a coefficient for scaling intervals, used in FilterInf to replace +-inf.
    private const int Coef = 3;

    private readonly int _populationSize;
    private readonly Func<FrozenTrial, IEnumerable<double>>? _constraintsFunc;
    private double[,]? _referencePoints;
    private readonly int _dividingParameter;
    private readonly LazyRandomState _rng;

    public NSGAIIIElitePopulationSelectionStrategy(
        int populationSize,
        LazyRandomState rng,
        Func<FrozenTrial, IEnumerable<double>>? constraintsFunc = null,
        double[,]? referencePoints = null,
        int dividingParameter = 3)
    {
        if (populationSize < 2)
            throw new ValueError("`population_size` must be greater than or equal to 2.");

        _populationSize = populationSize;
        _constraintsFunc = constraintsFunc;
        _referencePoints = referencePoints;
        _dividingParameter = dividingParameter;
        _rng = rng;
    }

    /// <summary>
    ///     Select the elite population from the given trials by the NSGA-III algorithm (Python
    ///     <c>__call__</c>).
    /// </summary>
    public List<FrozenTrial> Select(Study study, List<FrozenTrial> population)
    {
        bool isConstrained = ConstrainedOptimization.IsConstrainedOptimization(population);
        NsgaiiConstraintsEvaluation.ValidateConstraints(population, isConstrained);
        var populationPerRank = NSGAIIElitePopulationSelectionStrategy.RankPopulation(
            population, study.Directions, isConstrained);

        var elitePopulation = new List<FrozenTrial>();
        foreach (var individuals in populationPerRank)
        {
            if (elitePopulation.Count + individuals.Count < _populationSize)
            {
                elitePopulation.AddRange(individuals);
            }
            else
            {
                int nObjectives = study.Directions.Count;
                // Construct reference points in the first run.
                if (_referencePoints is null)
                    _referencePoints = GenerateDefaultReferencePoint(nObjectives, _dividingParameter);
                else if (_referencePoints.GetLength(1) != nObjectives)
                    throw new ValueError(
                        "The dimension of reference points vectors must be the same as the number " +
                        "of objectives of the study.");

                // Normalize objective values after filtering +-inf.
                var combined = new List<FrozenTrial>(elitePopulation);
                combined.AddRange(individuals);
                double[,] objectiveMatrix = NormalizeObjectiveValues(FilterInf(combined));

                var (closestReferencePoints, distanceReferencePoints) =
                    AssociateIndividualsWithReferencePoints(objectiveMatrix, _referencePoints);

                int elitePopulationNum = elitePopulation.Count;
                int targetPopulationSize = _populationSize - elitePopulationNum;
                var additionalElitePopulation = PreserveNicheIndividuals(
                    targetPopulationSize, elitePopulationNum, individuals,
                    closestReferencePoints, distanceReferencePoints);
                elitePopulation.AddRange(additionalElitePopulation);
                break;
            }
        }

        return elitePopulation;
    }

    /// <summary>
    ///     Python <c>_generate_default_reference_point</c>: the reference points `uniformly` spread on
    ///     a hyperplane, one per <c>combinations_with_replacement(range(n_objectives), div)</c>.
    /// </summary>
    internal static double[,] GenerateDefaultReferencePoint(int nObjectives, int dividingParameter = 3)
    {
        var indices = CombinationsWithReplacement(nObjectives, dividingParameter);
        var referencePoints = new double[indices.Count, nObjectives];
        // np.add.at(reference_points, (row, col), 1.0): each combination contributes +1 per index it
        // lists (with multiplicity), so a point can have entries > 1 (e.g. (0,0,0) -> [3, 0, ...]).
        for (int r = 0; r < indices.Count; r++)
            foreach (int col in indices[r])
                referencePoints[r, col] += 1.0;
        return referencePoints;
    }

    /// <summary>
    ///     <c>itertools.combinations_with_replacement(range(n), r)</c> in lexicographic order — the
    ///     same order numpy sees, so the reference-point rows line up byte-for-byte.
    /// </summary>
    private static List<int[]> CombinationsWithReplacement(int n, int r)
    {
        var result = new List<int[]>();
        if (n == 0)
            return result;
        var indices = new int[r];  // starts at all-zeros == (0, 0, ..., 0)
        while (true)
        {
            result.Add((int[])indices.Clone());
            // Advance like CPython's combinations_with_replacement: find rightmost index < n-1.
            int i = r - 1;
            while (i >= 0 && indices[i] == n - 1)
                i--;
            if (i < 0)
                break;
            int next = indices[i] + 1;
            for (int j = i; j < r; j++)
                indices[j] = next;
        }
        return result;
    }

    /// <summary>Python <c>_filter_inf</c>: clip +-inf to a margin beyond the finite range per axis.</summary>
    private static double[,] FilterInf(IReadOnlyList<FrozenTrial> population)
    {
        int p = population.Count;
        int m = population[0].Values!.Count;

        // np.nanmax / np.nanmin over the finite entries of each column (inf -> nan first).
        var maxObjectives = new double[m];
        var minObjectives = new double[m];
        for (int j = 0; j < m; j++)
        {
            double cmax = double.NegativeInfinity, cmin = double.PositiveInfinity;
            bool seen = false;
            for (int i = 0; i < p; i++)
            {
                double v = population[i].Values![j];
                if (double.IsFinite(v))
                {
                    seen = true;
                    if (v > cmax) cmax = v;
                    if (v < cmin) cmin = v;
                }
            }
            maxObjectives[j] = seen ? cmax : double.NaN;
            minObjectives[j] = seen ? cmin : double.NaN;
        }

        var result = new double[p, m];
        for (int j = 0; j < m; j++)
        {
            double margin = Coef * (maxObjectives[j] - minObjectives[j]);
            double lo = minObjectives[j] - margin;
            double hi = maxObjectives[j] + margin;
            for (int i = 0; i < p; i++)
            {
                double v = population[i].Values![j];
                // np.clip(v, lo, hi) == min(max(v, lo), hi).
                result[i, j] = Math.Min(Math.Max(v, lo), hi);
            }
        }
        return result;
    }

    /// <summary>Python <c>_normalize_objective_values</c>. Mutates and returns <paramref name="a" />.</summary>
    private static double[,] NormalizeObjectiveValues(double[,] a)
    {
        int p = a.GetLength(0);
        int m = a.GetLength(1);

        // Subtract ideal point (per-column min) from objective values.
        for (int j = 0; j < m; j++)
        {
            double cmin = a[0, j];
            for (int i = 1; i < p; i++)
                if (a[i, j] < cmin) cmin = a[i, j];
            for (int i = 0; i < p; i++)
                a[i, j] -= cmin;
        }

        // weights = np.eye(m); weights[weights == 0] = 1e6.
        var weights = new double[m, m];
        for (int d = 0; d < m; d++)
            for (int j = 0; j < m; j++)
                weights[d, j] = d == j ? 1.0 : 1e6;

        // asf[d, n] = max_j a[n, j] * weights[d, j]; extreme_points[d] = a[argmin_n asf[d, n]].
        // (einsum("nm,dm->dnm") is an elementwise product; max over m then argmin over n.)
        var extremePoints = new double[m, m];
        for (int d = 0; d < m; d++)
        {
            int bestN = 0;
            double bestAsf = double.NaN;
            bool first = true;
            for (int n = 0; n < p; n++)
            {
                double mx = double.NaN;
                bool firstM = true;
                for (int j = 0; j < m; j++)
                {
                    double val = a[n, j] * weights[d, j];
                    if (firstM || val > mx) { mx = val; firstM = false; }
                }
                if (first || mx < bestAsf) { bestAsf = mx; bestN = n; first = false; }
            }
            for (int j = 0; j < m; j++)
                extremePoints[d, j] = a[bestN, j];
        }

        // Normalize with extreme points, or fall back to per-axis max when they are degenerate.
        var interceptsInv = new double[m];
        bool allFinite = true;
        for (int d = 0; d < m && allFinite; d++)
            for (int j = 0; j < m; j++)
                if (!double.IsFinite(extremePoints[d, j])) { allFinite = false; break; }

        if (allFinite && LinAlg.MatrixRank(extremePoints) == m)
        {
            var ones = new double[m];
            for (int j = 0; j < m; j++) ones[j] = 1.0;
            interceptsInv = LinAlg.Solve(extremePoints, ones);
        }
        else
        {
            for (int j = 0; j < m; j++)
            {
                double intercept = a[0, j];
                for (int i = 1; i < p; i++)
                    if (a[i, j] > intercept) intercept = a[i, j];
                interceptsInv[j] = 1.0 / (intercept == 0.0 ? 1.0 : intercept);
            }
        }

        // objective_matrix *= np.where(np.isfinite(intercepts_inv), intercepts_inv, 1).
        for (int j = 0; j < m; j++)
        {
            double scale = double.IsFinite(interceptsInv[j]) ? interceptsInv[j] : 1.0;
            for (int i = 0; i < p; i++)
                a[i, j] *= scale;
        }

        return a;
    }

    /// <summary>
    ///     Python <c>_associate_individuals_with_reference_points</c>: the closest reference point to
    ///     each individual and that Euclidean (perpendicular) distance.
    /// </summary>
    private static (int[] Closest, double[] Distance) AssociateIndividualsWithReferencePoints(
        double[,] objectiveMatrix, double[,] referencePoints)
    {
        int p = objectiveMatrix.GetLength(0);
        int m = objectiveMatrix.GetLength(1);
        int r = referencePoints.GetLength(0);

        // 1 / (norm(reference_points, axis=1) ** 2) — note norm then square (sqrt(s)**2 != s exactly).
        var inv = new double[r];
        for (int q = 0; q < r; q++)
        {
            double s = 0.0;
            for (int j = 0; j < m; j++) s += referencePoints[q, j] * referencePoints[q, j];
            double t = Math.Sqrt(s);
            inv[q] = 1.0 / (t * t);
        }

        var closest = new int[p];
        var distance = new double[p];
        for (int n = 0; n < p; n++)
        {
            int best = 0;
            double bestDist = double.NaN;
            bool first = true;
            for (int q = 0; q < r; q++)
            {
                // dist = norm(obj_n - perp_{n,q}); perp_{n,q,k} matches numpy's 4-operand
                // einsum("ni,pi,p,pm->npm") accumulated term-by-term (bit-exact, verified).
                double s = 0.0;
                for (int k = 0; k < m; k++)
                {
                    double perp = 0.0;
                    for (int i = 0; i < m; i++)
                        perp += objectiveMatrix[n, i] * referencePoints[q, i] * inv[q] * referencePoints[q, k];
                    double diff = objectiveMatrix[n, k] - perp;
                    s += diff * diff;
                }
                double dist = Math.Sqrt(s);
                // np.argmin / np.min keep the FIRST minimum on ties (strict <).
                if (first || dist < bestDist) { bestDist = dist; best = q; first = false; }
            }
            closest[n] = best;
            distance[n] = bestDist;
        }

        return (closest, distance);
    }

    /// <summary>
    ///     Python <c>_preserve_niche_individuals</c>: draw survivors from the borderline front so the
    ///     reference points stay as evenly (sparsely) populated as possible.
    /// </summary>
    private List<FrozenTrial> PreserveNicheIndividuals(
        int targetPopulationSize,
        int elitePopulationNum,
        List<FrozenTrial> population,
        int[] closestReferencePoints,
        double[] distanceReferencePoints)
    {
        if (population.Count < targetPopulationSize)
            throw new ValueError(
                "The population size must be greater than or equal to the target population size.");

        // reference_point -> list of (distance, borderline-local-index), in first-seen key order.
        var refToBorderline = new Dictionary<int, List<(double Distance, int Index)>>();
        var refInsertionOrder = new List<int>();
        for (int i = 0; i < closestReferencePoints.Length - elitePopulationNum; i++)
        {
            int refIdx = closestReferencePoints[i + elitePopulationNum];
            int populationIdx = i + elitePopulationNum;
            if (!refToBorderline.TryGetValue(refIdx, out var list))
            {
                list = new List<(double, int)>();
                refToBorderline[refIdx] = list;
                refInsertionOrder.Add(refIdx);
            }
            list.Add((distanceReferencePoints[populationIdx], i));
        }

        // reference_point -> how many elite neighbors it has (defaultdict(int)).
        var refToEliteCount = new Dictionary<int, int>();
        for (int i = 0; i < elitePopulationNum; i++)
        {
            int refIdx = closestReferencePoints[i];
            refToEliteCount[refIdx] = refToEliteCount.GetValueOrDefault(refIdx, 0) + 1;
        }

        // elite-neighbor-count -> reference points (only those with a borderline member), in
        // first-seen key order (defaultdict(list)).
        var countToRefs = new Dictionary<int, List<int>>();
        foreach (int refIdx in refInsertionOrder)
        {
            int eliteCount = refToEliteCount.GetValueOrDefault(refIdx, 0);
            GetBucket(countToRefs, eliteCount).Add(refIdx);
        }

        int count = -1;
        var additionalElitePopulation = new List<FrozenTrial>();
        var isShuffled = new HashSet<int>();
        while (additionalElitePopulation.Count < targetPopulationSize)
        {
            var currentBucket = GetBucket(countToRefs, count);
            if (currentBucket.Count == 0)
            {
                count += 1;
                ShuffleInPlace(GetBucket(countToRefs, count));
                continue;
            }

            int refIdx = currentBucket[^1];
            currentBucket.RemoveAt(currentBucket.Count - 1);

            var borderline = refToBorderline[refIdx];
            if (count > 0 && !isShuffled.Contains(refIdx))
            {
                ShuffleInPlace(borderline);
                isShuffled.Add(refIdx);
            }
            else if (count == 0)
            {
                // list.sort(reverse=True) on (distance, index) tuples (index is unique -> total order).
                borderline.Sort((x, y) =>
                {
                    int c = y.Distance.CompareTo(x.Distance);
                    return c != 0 ? c : y.Index.CompareTo(x.Index);
                });
            }

            var selected = borderline[^1];
            borderline.RemoveAt(borderline.Count - 1);
            additionalElitePopulation.Add(population[selected.Index]);
            if (borderline.Count > 0)
                GetBucket(countToRefs, count + 1).Add(refIdx);
        }

        return additionalElitePopulation;
    }

    /// <summary>Python <c>defaultdict(list)</c> access: auto-create the bucket for a missing key.</summary>
    private static List<T> GetBucket<T>(Dictionary<int, List<T>> dict, int key)
    {
        if (!dict.TryGetValue(key, out var list))
        {
            list = new List<T>();
            dict[key] = list;
        }
        return list;
    }

    /// <summary>
    ///     <c>rng.shuffle(list)</c> in place — reproduces numpy's Python-list Fisher-Yates by shuffling
    ///     an index permutation with the same MT19937 j-sequence (bit-exact, verified) and gathering.
    /// </summary>
    private void ShuffleInPlace<T>(List<T> list)
    {
        int n = list.Count;
        if (n <= 1)
            return;  // numpy draws no random numbers for length 0 or 1.
        var permutation = np.arange(n);
        _rng.Rng.shuffle(permutation);
        long[] perm = permutation.ToArray<long>();
        var copy = new List<T>(list);
        for (int k = 0; k < n; k++)
            list[k] = copy[(int)perm[k]];
    }
}
```


<a id="ga"></a>

## src/OptunaSharp.Core/Samplers/BaseGASampler.cs

Generation lifecycle · 124 physical lines · SHA-256: `d6c473d3ea512dc507ea728b6abb65a1f9e728203afae29c9c0ddef0646875af`

```csharp
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace OptunaSharp.Samplers;

/// <summary>
///     Base class for Genetic Algorithm (GA) samplers. Port of
///     <c>optuna/samplers/_ga/_base.py: BaseGASampler</c>.
///
///     GA samplers generate new trials by mimicking natural selection, using generations and
///     populations to iteratively improve solutions. This base class defines the interface and
///     provides utility methods for managing generations and populations. Parent selection is
///     handled by <see cref="SelectParent" />, which subclasses implement.
/// </summary>
public abstract class BaseGASampler : BaseSampler
{
    private int? _populationSize;

    protected BaseGASampler(int? populationSize)
    {
        _populationSize = populationSize;
    }

    // Python __init_subclass__ sets these per subclass to f"{cls.__name__}:...". GetType().Name at
    // runtime is the concrete subclass name, so "NSGAIISampler:generation" etc. — byte-identical.
    internal string GenerationKey => $"{GetType().Name}:generation";
    internal string ParentCacheKeyPrefix => $"{GetType().Name}:parent:";

    /// <summary>Number of individuals (trials) in a generation.</summary>
    public int? PopulationSize
    {
        get => _populationSize;
        set => _populationSize = value;
    }

    /// <summary>
    ///     Select parent trials from the population for the given generation. Called once per
    ///     generation; the result is cached in the study system attributes.
    /// </summary>
    public abstract List<FrozenTrial> SelectParent(Study study, int generation);

    /// <summary>
    ///     Get the generation number of the given trial. If not already set in the trial's system
    ///     attributes, it is calculated (from the max generation of all completed trials) and stored.
    /// </summary>
    public int GetTrialGeneration(Study study, FrozenTrial trial)
    {
        var stored = trial.SystemAttrsDict.GetValueOrDefault(GenerationKey);
        if (stored is not null)
            return Convert.ToInt32(stored, CultureInfo.InvariantCulture);

        var trials = study.GetTrialsInternal(
            deepcopy: false, states: new[] { TrialState.Complete }, useCache: true);

        int maxGeneration = 0, maxGenerationCount = 0;
        for (int i = trials.Count - 1; i >= 0; i--)
        {
            var raw = trials[i].SystemAttrsDict.GetValueOrDefault(GenerationKey);
            int generation = raw is null ? -1 : Convert.ToInt32(raw, CultureInfo.InvariantCulture);

            if (generation < maxGeneration)
                continue;
            if (generation > maxGeneration)
            {
                maxGeneration = generation;
                maxGenerationCount = 1;
            }
            else
            {
                maxGenerationCount += 1;
            }
        }

        System.Diagnostics.Debug.Assert(_populationSize is not null, "Population size must be set.");
        int result = maxGenerationCount < _populationSize ? maxGeneration : maxGeneration + 1;
        study.Storage.SetTrialSystemAttr(trial.TrialId, GenerationKey, result);
        return result;
    }

    /// <summary>Get the population (completed trials) of the given generation.</summary>
    public List<FrozenTrial> GetPopulation(Study study, int generation)
    {
        var result = new List<FrozenTrial>();
        foreach (var trial in study.GetTrialsInternal(
                     deepcopy: false, states: new[] { TrialState.Complete }, useCache: true))
        {
            var raw = trial.SystemAttrsDict.GetValueOrDefault(GenerationKey);
            if (raw is not null && Convert.ToInt32(raw, CultureInfo.InvariantCulture) == generation)
                result.Add(trial);
        }
        return result;
    }

    /// <summary>
    ///     Get the parent population of the given generation (cached in the study system attributes).
    ///     Returns an empty list when <paramref name="generation" /> is 0.
    /// </summary>
    public List<FrozenTrial> GetParentPopulation(Study study, int generation)
    {
        if (generation == 0)
            return new List<FrozenTrial>();

        var studySystemAttrs = study.Storage.GetStudySystemAttrs(study.StudyId);
        var cached = studySystemAttrs.GetValueOrDefault(
            ParentCacheKeyPrefix + generation.ToString(CultureInfo.InvariantCulture));

        if (cached is not null)
        {
            var ids = new HashSet<int>(
                ((IEnumerable<object?>)cached).Select(x => Convert.ToInt32(x, CultureInfo.InvariantCulture)));
            var trials = study.GetTrialsInternal(deepcopy: false);
            return trials.Where(t => ids.Contains(t.TrialId)).ToList();
        }

        var parentPopulation = SelectParent(study, generation);
        study.Storage.SetStudySystemAttr(
            study.StudyId,
            ParentCacheKeyPrefix + generation.ToString(CultureInfo.InvariantCulture),
            parentPopulation.Select(t => (object?)t.TrialId).ToList());
        return parentPopulation;
    }
}
```


<a id="base-sampler"></a>

## src/OptunaSharp.Core/Samplers/BaseSampler.cs

Generation lifecycle · 134 physical lines · SHA-256: `292a35ddd3ab83b73ca1335447cba4d6e424cf03c0123dc0f7d3b10fd9db64c5`

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers;

/// <summary>
///     Base class for samplers. Port of <c>optuna/samplers/_base.py: BaseSampler</c>.
///
///     Optuna combines two types of sampling strategies, which are called <i>relative
///     sampling</i> and <i>independent sampling</i>.
///
///     The relative sampling determines values of multiple parameters simultaneously so that
///     sampling algorithms can use relationship between parameters (e.g., correlation). Target
///     parameters of the relative sampling are described in a relative search space, which is
///     determined by <see cref="InferRelativeSearchSpace" />.
///
///     The independent sampling determines a value of a single parameter without considering any
///     relationship between parameters. Target parameters of the independent sampling are the
///     parameters not described in the relative search space.
/// </summary>
public abstract class BaseSampler
{
    internal const string IndependentSamplingWarningTemplate =
        "The parameter `{0}` in Trial#{1} is sampled independently " +
        "using `{2}` instead of `{3}`, potentially " +
        "degrading the optimization performance. This fallback happened because " +
        "{4}. You can suppress this warning by setting " +
        "`warn_independent_sampling` to `False` in the constructor of `{3}` if " +
        "this independent sampling is intended behavior.";

    public override string ToString() => GetType().Name;

    /// <summary>
    ///     Infer the search space that will be used by relative sampling in the target trial.
    ///
    ///     This method is called right before <see cref="SampleRelative" />, and the search space
    ///     returned by this method is passed to it. The parameters not contained in the search
    ///     space will be sampled by using <see cref="SampleIndependent" />.
    /// </summary>
    public abstract OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial);

    /// <summary>
    ///     Sample parameters in a given search space.
    ///
    ///     This method is called once at the beginning of each trial, i.e., right before the
    ///     evaluation of the objective function. This method is suitable for sampling algorithms
    ///     that use relationship between parameters such as Gaussian Process and CMA-ES.
    /// </summary>
    public abstract OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace);

    /// <summary>
    ///     Sample a parameter for a given distribution.
    ///
    ///     This method is called only for the parameters not contained in the search space
    ///     returned by <see cref="SampleRelative" />. This method is suitable for sampling
    ///     algorithms that do not use relationship between parameters such as random sampling
    ///     and TPE.
    /// </summary>
    public abstract object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution);

    /// <summary>
    ///     Trial pre-processing: called during trial initialization, just before the
    ///     <see cref="InferRelativeSearchSpace" /> call.
    /// </summary>
    public virtual void BeforeTrial(Study study, FrozenTrial trial)
    {
    }

    /// <summary>
    ///     Trial post-processing: called after the objective function returns and right before
    ///     the trial is finished and its state is stored.
    /// </summary>
    /// <param name="values">Resulting trial values. Guaranteed to not be <c>null</c> if trial succeeded.</param>
    public virtual void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
    }

    /// <summary>
    ///     Reseed sampler's random number generator.
    ///
    ///     This method is called by the <see cref="Study" /> instance if trials are executed in
    ///     parallel with the option <c>nJobs &gt; 1</c>. In that case, the sampler instance will
    ///     be replicated including the state of the random number generator, and they may suggest
    ///     the same values. To prevent this issue, this method assigns a different seed to each
    ///     random number generator.
    /// </summary>
    public virtual void ReseedRng()
    {
    }

    /// <summary>Python <c>_raise_error_if_multi_objective</c>.</summary>
    internal void RaiseErrorIfMultiObjective(Study study)
    {
        if (study.IsMultiObjective())
            throw new ValueError(
                "If the study is being used for multi-objective optimization, " +
                $"{GetType().Name} cannot be used.");
    }
}

/// <summary>Module-level helpers of <c>optuna/samplers/_base.py</c>.</summary>
internal static class SamplerHelpers
{
    /// <summary>Python <c>_process_constraints_after_trial</c>.</summary>
    internal static void ProcessConstraintsAfterTrial(
        Func<FrozenTrial, IEnumerable<double>> constraintsFunc, Study study, FrozenTrial trial, TrialState state)
    {
        if (state != TrialState.Complete && state != TrialState.Pruned)
            return;

        List<double>? constraints = null;
        try
        {
            var con = constraintsFunc(trial).ToList();
            if (con.Any(double.IsNaN))
                throw new ValueError("Constraint values cannot be NaN.");
            constraints = con;
        }
        finally
        {
            study.Storage.SetTrialSystemAttr(
                trial.TrialId,
                ConstrainedOptimization.ConstraintsKey,
                constraints?.Cast<object?>().ToList());
        }
    }
}
```


<a id="random"></a>

## src/OptunaSharp.Core/Samplers/RandomSampler.cs

Random sampling · 61 physical lines · SHA-256: `64a17cfd141b3a4e81cfe8921e460e0dd940396f272a604392653f6d2381a549`

```csharp
using System;
using System.Security.Cryptography;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers;

/// <summary>
///     Sampler using random sampling. Port of <c>optuna/samplers/_random.py: RandomSampler</c>.
///
///     This sampler is based on <i>independent sampling</i>.
///     See also <see cref="BaseSampler" /> for more details of 'independent sampling'.
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(sampler: new RandomSampler());
///     study.Optimize(trial => Math.Pow(trial.SuggestFloat("x", -5, 5), 2), nTrials: 10);
///     </code>
/// </example>
public class RandomSampler : BaseSampler
{
    private readonly LazyRandomState _rng;

    /// <param name="seed">Seed for random number generator.</param>
    public RandomSampler(int? seed = null)
    {
        _rng = new LazyRandomState(seed);
    }

    internal LazyRandomState Rng => _rng;

    public override void ReseedRng()
    {
        // Python: self._rng.rng.seed() — reseed from OS entropy.
        Span<byte> buffer = stackalloc byte[4];
        RandomNumberGenerator.Fill(buffer);
        _rng.Rng.seed(BitConverter.ToUInt32(buffer));
    }

    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
        => new();

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
        => new();

    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution> { [paramName] = paramDistribution };
        var trans = new SearchSpaceTransform(searchSpace);
        var bounds = trans.Bounds;
        // uniform(bounds[:, 0], bounds[:, 1]) — one MT19937 draw per bound row, in row order,
        // exactly as NumPy broadcasts.
        var transParams = _rng.Rng.uniform(bounds[":,0"], bounds[":,1"]);

        return trans.Untransform(transParams)[paramName];
    }
}
```


<a id="rng"></a>

## src/OptunaSharp.Core/Samplers/LazyRandomState.cs

Random sampling · 41 physical lines · SHA-256: `c5265882445ed6686cdd298a1a0ab3f6ec5969750b3da0ef451b669723694502`

```csharp
using NumSharp;

namespace OptunaSharp.Samplers;

/// <summary>
///     Lazy Random State class. Port of <c>optuna/samplers/_lazy_random_state.py</c>.
///
///     This is a class to initialize the random state just before use to prevent duplication of
///     the same random state when a sampler instance is copied.
///
///     Every sampler consumes randomness through this wrapper over NumSharp's NumPy-exact
///     MT19937 <c>RandomState</c> — never <see cref="System.Random" /> — which is what makes
///     differential testing against Python bit-exact (plan §2 RNG discipline).
/// </summary>
public sealed class LazyRandomState
{
    private NumPyRandom? _rng;

    public LazyRandomState(int? seed = null)
    {
        _rng = null;
        if (seed is not null)
            Rng.seed(seed.Value);
    }

    private void SetRng()
    {
        _rng = np.random.RandomState();
    }

    /// <summary>The underlying <c>np.random.RandomState</c>, created on first use.</summary>
    public NumPyRandom Rng
    {
        get
        {
            if (_rng is null)
                SetRng();
            return _rng!;
        }
    }
}
```


<a id="multiobjective"></a>

## src/OptunaSharp.Core/Internal/MultiObjective.cs

Ranking and constraints · 346 physical lines · SHA-256: `0a8fada3ec121dd4dff0324a960a0eb677771011b250acbfc08685ef69e6950d`

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Internal;

/// <summary>
///     Port of <c>optuna/study/_multi_objective.py</c>. The NumPy row-wise operations
///     (<c>np.unique(axis=0)</c> lexsort/dedupe) are reproduced in plain C# with identical
///     semantics — everything here is deterministic comparison logic, no RNG.
/// </summary>
internal static class MultiObjective
{
    /// <summary>Python <c>_get_pareto_front_trials_by_trials</c>.</summary>
    internal static List<FrozenTrial> GetParetoFrontTrialsByTrials(
        IEnumerable<FrozenTrial> trials,
        IReadOnlyList<StudyDirection> directions,
        bool considerConstraint = false)
    {
        // NOTE(nabenabe0928): Vectorization relies on all the trials being complete.
        var completeTrials = trials.Where(t => t.State == TrialState.Complete).ToList();
        if (considerConstraint)
            completeTrials = ConstrainedOptimization.GetFeasibleTrials(completeTrials);
        if (completeTrials.Count == 0)
            return new List<FrozenTrial>();

        if (completeTrials.Any(t => t.Values!.Count != directions.Count))
            throw new ValueError("The number of the values and the number of the objectives must be identical.");

        var lossValues = completeTrials
            .Select(t => t.Values!.Select((v, i) => NormalizeValue(v, directions[i])).ToArray())
            .ToArray();
        var onFront = IsParetoFront(lossValues, assumeUniqueLexsorted: false);
        var result = new List<FrozenTrial>();
        for (int i = 0; i < completeTrials.Count; i++)
        {
            if (onFront[i])
                result.Add(completeTrials[i]);
        }
        return result;
    }

    /// <summary>Python <c>_get_pareto_front_trials</c>.</summary>
    internal static List<FrozenTrial> GetParetoFrontTrials(Study study, bool considerConstraint = false)
        => GetParetoFrontTrialsByTrials(study.Trials, study.Directions, considerConstraint);

    /// <summary>
    ///     Python <c>_fast_non_domination_rank</c>: non-domination rank based on the fast
    ///     non-dominated sort algorithm of constrained NSGA-II. Correct only up to the
    ///     top-<paramref name="nBelow" /> solutions.
    /// </summary>
    internal static int[] FastNonDominationRank(double[][] lossValues, double[]? penalty = null, int? nBelow = null)
    {
        if (lossValues.Length == 0)
            return Array.Empty<int>();

        int below = nBelow ?? lossValues.Length;
        System.Diagnostics.Debug.Assert(below > 0, "n_below must be a positive integer.");

        if (penalty is null)
            return CalculateNondominationRank(lossValues, below);

        if (penalty.Length != lossValues.Length)
            throw new ValueError(
                "The length of penalty and loss_values must be same, but got " +
                $"len(penalty)={penalty.Length} and len(loss_values)={lossValues.Length}.");

        var ranks = Enumerable.Repeat(-1, lossValues.Length).ToArray();
        var feasibleIndices = new List<int>();
        var infeasibleIndices = new List<int>();
        for (int i = 0; i < penalty.Length; i++)
        {
            // NaN penalties fall into neither bucket in NumPy (`penalty <= 0` and `penalty > 0`
            // are both false) — but Optuna guards NaN out upstream, and the final assert would
            // fire; mirror by treating NaN as infeasible-last like the Python assert expects.
            if (penalty[i] <= 0)
                feasibleIndices.Add(i);
            else
                infeasibleIndices.Add(i);
        }

        // First, we calculate the domination rank for feasible trials.
        var feasibleLoss = feasibleIndices.Select(i => lossValues[i]).ToArray();
        var feasibleRanks = CalculateNondominationRank(feasibleLoss, below);
        for (int i = 0; i < feasibleIndices.Count; i++)
            ranks[feasibleIndices[i]] = feasibleRanks[i];
        below -= feasibleIndices.Count;

        // Second, we calculate the domination rank for infeasible trials.
        int topRankInfeasible = feasibleRanks.Length == 0 ? 0 : feasibleRanks.Max() + 1;
        var infeasiblePenalty = infeasibleIndices.Select(i => new[] { penalty[i] }).ToArray();
        var infeasibleRanks = CalculateNondominationRank(infeasiblePenalty, below);
        for (int i = 0; i < infeasibleIndices.Count; i++)
            ranks[infeasibleIndices[i]] = topRankInfeasible + infeasibleRanks[i];

        System.Diagnostics.Debug.Assert(ranks.All(r => r != -1), "All the rank must be updated.");
        return ranks;
    }

    /// <summary>Python <c>_is_pareto_front_nd</c>.</summary>
    private static bool[] IsParetoFrontNd(double[][] uniqueLexsortedLossValues)
    {
        // As column 0 is sorted, we do not need it to judge dominance.
        int nTrials = uniqueLexsortedLossValues.Length;
        int nObjectives = nTrials == 0 ? 0 : uniqueLexsortedLossValues[0].Length;
        var onFront = new bool[nTrials];
        var remainingIndices = Enumerable.Range(0, nTrials).ToList();

        while (remainingIndices.Count > 0)
        {
            // NOTE: trials[j] cannot dominate trials[i] for i < j because of lexsort.
            // Therefore, remaining_indices[0] is always non-dominated.
            int newNondominatedIndex = remainingIndices[0];
            onFront[newNondominatedIndex] = true;
            var nondominated = uniqueLexsortedLossValues[newNondominatedIndex];

            var next = new List<int>();
            foreach (int idx in remainingIndices)
            {
                // np.any(loss_values[idx, 1:] < nondominated[1:]).
                bool anyLess = false;
                for (int k = 1; k < nObjectives; k++)
                {
                    if (uniqueLexsortedLossValues[idx][k] < nondominated[k])
                    {
                        anyLess = true;
                        break;
                    }
                }
                if (anyLess)
                    next.Add(idx);
            }
            remainingIndices = next;
        }

        return onFront;
    }

    /// <summary>Python <c>_is_pareto_front_2d</c>.</summary>
    private static bool[] IsParetoFront2d(double[][] uniqueLexsortedLossValues)
    {
        int nTrials = uniqueLexsortedLossValues.Length;
        var onFront = new bool[nTrials];
        if (nTrials == 0)
            return onFront;
        onFront[0] = true;
        double cummin = uniqueLexsortedLossValues[0][1];
        for (int i = 1; i < nTrials; i++)
        {
            double value1 = uniqueLexsortedLossValues[i][1];
            onFront[i] = value1 < cummin;  // True if cummin value1 is new minimum.
            cummin = Math.Min(cummin, value1);
        }
        return onFront;
    }

    /// <summary>Python <c>_is_pareto_front_for_unique_sorted</c>.</summary>
    private static bool[] IsParetoFrontForUniqueSorted(double[][] uniqueLexsortedLossValues)
    {
        int nObjectives = uniqueLexsortedLossValues.Length == 0 ? 0 : uniqueLexsortedLossValues[0].Length;
        if (nObjectives == 1)
        {
            var onFront = new bool[uniqueLexsortedLossValues.Length];
            if (onFront.Length > 0)
                onFront[0] = true;  // Only the first element is Pareto optimal.
            return onFront;
        }
        if (nObjectives == 2)
            return IsParetoFront2d(uniqueLexsortedLossValues);
        return IsParetoFrontNd(uniqueLexsortedLossValues);
    }

    /// <summary>Python <c>_is_pareto_front(loss_values, assume_unique_lexsorted)</c>.</summary>
    internal static bool[] IsParetoFront(double[][] lossValues, bool assumeUniqueLexsorted)
    {
        if (assumeUniqueLexsorted)
            return IsParetoFrontForUniqueSorted(lossValues);

        var (uniqueRows, orderInv) = UniqueRows(lossValues);
        var onFrontUnique = IsParetoFrontForUniqueSorted(uniqueRows);
        var onFront = new bool[lossValues.Length];
        for (int i = 0; i < lossValues.Length; i++)
            onFront[i] = onFrontUnique[orderInv[i]];
        return onFront;
    }

    /// <summary>Python <c>_calculate_nondomination_rank</c>.</summary>
    internal static int[] CalculateNondominationRank(double[][] lossValues, int? nBelow = null)
    {
        if (lossValues.Length == 0 || (nBelow is not null && nBelow <= 0))
            return new int[lossValues.Length];

        int nObjectives = lossValues[0].Length;
        if (nObjectives == 1)
        {
            // np.unique(loss_values[:, 0], return_inverse=True): rank = index into sorted uniques.
            var uniqueSorted = lossValues.Select(r => r[0]).Distinct().OrderBy(v => v).ToArray();
            var ranks1 = new int[lossValues.Length];
            for (int i = 0; i < lossValues.Length; i++)
                ranks1[i] = Array.BinarySearch(uniqueSorted, lossValues[i][0]);
            return ranks1;
        }

        // It ensures that trials[j] will not dominate trials[i] for i < j (np.unique lexsorts).
        var (uniqueRows, orderInv) = UniqueRows(lossValues);
        int nUnique = uniqueRows.Length;
        int below = Math.Min(nBelow ?? nUnique, nUnique);
        var ranks = new int[nUnique];
        int rank = 0;
        var indices = Enumerable.Range(0, nUnique).ToArray();
        var remaining = uniqueRows;
        while (nUnique - indices.Length < below)
        {
            var onFront = IsParetoFront(remaining, assumeUniqueLexsorted: true);
            var nextIndices = new List<int>();
            var nextRemaining = new List<double[]>();
            for (int i = 0; i < indices.Length; i++)
            {
                if (onFront[i])
                    ranks[indices[i]] = rank;
                else
                {
                    nextIndices.Add(indices[i]);
                    nextRemaining.Add(remaining[i]);
                }
            }
            indices = nextIndices.ToArray();
            remaining = nextRemaining.ToArray();
            rank += 1;
        }

        // Rank worse than the top n_below is defined as the worst rank.
        foreach (int idx in indices)
            ranks[idx] = rank;

        var result = new int[lossValues.Length];
        for (int i = 0; i < lossValues.Length; i++)
            result[i] = ranks[orderInv[i]];
        return result;
    }

    /// <summary>Python <c>_dominates(trial0, trial1, directions)</c>.</summary>
    internal static bool Dominates(FrozenTrial trial0, FrozenTrial trial1, IReadOnlyList<StudyDirection> directions)
    {
        var values0 = trial0.Values;
        var values1 = trial1.Values;

        if (trial0.State != TrialState.Complete)
            return false;

        if (trial1.State != TrialState.Complete)
            return true;

        System.Diagnostics.Debug.Assert(values0 is not null);
        System.Diagnostics.Debug.Assert(values1 is not null);

        if (values0!.Count != values1!.Count)
            throw new ValueError("Trials with different numbers of objectives cannot be compared.");

        if (values0.Count != directions.Count)
            throw new ValueError("The number of the values and the number of the objectives are mismatched.");

        var normalized0 = values0.Select((v, i) => NormalizeValue(v, directions[i])).ToArray();
        var normalized1 = values1.Select((v, i) => NormalizeValue(v, directions[i])).ToArray();

        if (normalized0.SequenceEqual(normalized1))
            return false;

        return normalized0.Zip(normalized1, (v0, v1) => v0 <= v1).All(x => x);
    }

    /// <summary>Python <c>_normalize_value</c>.</summary>
    internal static double NormalizeValue(double? value, StudyDirection direction)
    {
        if (value is null)
            return double.PositiveInfinity;
        return direction == StudyDirection.Maximize ? -value.Value : value.Value;
    }

    /// <summary>
    ///     NumPy <c>np.unique(a, axis=0, return_inverse=True)</c>: rows sorted lexicographically
    ///     (leading column first), deduplicated, plus the inverse mapping original-row → unique-row.
    /// </summary>
    internal static (double[][] UniqueRows, int[] OrderInverse) UniqueRows(double[][] rows)
    {
        int n = rows.Length;
        var order = Enumerable.Range(0, n).ToArray();
        Array.Sort(order, (a, b) => CompareRows(rows[a], rows[b]));

        var uniqueRows = new List<double[]>();
        var orderInv = new int[n];
        for (int i = 0; i < n; i++)
        {
            int rowIdx = order[i];
            if (uniqueRows.Count == 0 || CompareRows(uniqueRows[^1], rows[rowIdx]) != 0)
                uniqueRows.Add(rows[rowIdx]);
            orderInv[rowIdx] = uniqueRows.Count - 1;
        }
        return (uniqueRows.ToArray(), orderInv);
    }

    /// <summary>
    ///     NumPy <c>np.unique(a, axis=0, return_index=True)</c>: rows sorted lexicographically and
    ///     deduplicated, plus the index of the FIRST occurrence (smallest original index) of each
    ///     unique row.
    /// </summary>
    internal static (double[][] UniqueRows, int[] FirstIndices) UniqueRowsFirstIndex(double[][] rows)
    {
        int n = rows.Length;
        var order = Enumerable.Range(0, n).ToArray();
        // Tie-break equal rows by original index so the first occurrence is the smallest index.
        Array.Sort(order, (a, b) =>
        {
            int c = CompareRows(rows[a], rows[b]);
            return c != 0 ? c : a.CompareTo(b);
        });

        var uniqueRows = new List<double[]>();
        var firstIndices = new List<int>();
        for (int i = 0; i < n; i++)
        {
            int rowIdx = order[i];
            if (uniqueRows.Count == 0 || CompareRows(uniqueRows[^1], rows[rowIdx]) != 0)
            {
                uniqueRows.Add(rows[rowIdx]);
                firstIndices.Add(rowIdx);
            }
        }
        return (uniqueRows.ToArray(), firstIndices.ToArray());
    }

    private static int CompareRows(double[] a, double[] b)
    {
        for (int i = 0; i < a.Length; i++)
        {
            // Plain comparisons (not CompareTo) so that -0.0 == 0.0, matching NumPy's sort.
            if (a[i] < b[i])
                return -1;
            if (a[i] > b[i])
                return 1;
        }
        return 0;
    }
}
```


<a id="constraints"></a>

## src/OptunaSharp.Core/Internal/ConstrainedOptimization.cs

Ranking and constraints · 63 physical lines · SHA-256: `694ee4258cc84be5cab641538033c7ea34e1309f13aa137355e5a2e565626637`

```csharp
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp.Internal;

/// <summary>Port of <c>optuna/study/_constrained_optimization.py</c>.</summary>
internal static class ConstrainedOptimization
{
    internal const string ConstraintsKey = "constraints";

    /// <summary>Return whether the given trials are created in constrained optimization.</summary>
    internal static bool IsConstrainedOptimization(IEnumerable<FrozenTrial> trials)
        => trials.Any(trial => trial.Constraints.Count > 0);

    /// <summary>Return feasible trials from given trials.</summary>
    internal static List<FrozenTrial> GetFeasibleTrials(IEnumerable<FrozenTrial> trials)
    {
        var feasibleTrials = new List<FrozenTrial>();
        foreach (var trial in trials)
        {
            if (trial.Constraints.Values.All(x => x <= 0.0))
                feasibleTrials.Add(trial);
        }
        return feasibleTrials;
    }

    /// <summary>Python <c>_get_constraints_from_system_attrs</c>.</summary>
    internal static OrderedDict<string, double> GetConstraintsFromSystemAttrs(
        IReadOnlyDictionary<string, object?> systemAttrs)
    {
        var constraintsDict = new OrderedDict<string, double>();

        // Load constraints from old format (list).
        if (systemAttrs.TryGetValue(ConstraintsKey, out var con) && con is not null)
        {
            int i = 0;
            foreach (var c in (IEnumerable<object?>)con)
            {
                constraintsDict[i.ToString(System.Globalization.CultureInfo.InvariantCulture)] = ToDouble(c);
                i++;
            }
        }

        // Load constraints from new format (individual keys).
        string prefix = ConstraintsKey + ":";
        foreach (var (key, value) in systemAttrs)
        {
            if (key.StartsWith(prefix, System.StringComparison.Ordinal))
            {
                string constraintKey = key.Substring(prefix.Length);
                if (constraintsDict.ContainsKey(constraintKey))
                    OptunaWarnings.Warn("Overwrite an old format constraint.");
                constraintsDict[constraintKey] = ToDouble(value);
            }
        }
        return constraintsDict;
    }

    private static double ToDouble(object? value)
        => PyFloat.TryConvert(value, out double d) ? d : double.NaN;
}
```


<a id="intersection"></a>

## src/OptunaSharp.Core/SearchSpace/IntersectionSearchSpace.cs

Search space · 311 physical lines · SHA-256: `543993fbde4cae98f8a0eb5b7d291fa5c749cfddde131a6c3b9902d898a15c3b`

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.SearchSpace;

/// <summary>
///     Port of <c>optuna/search_space/intersection.py</c>.
/// </summary>
public static class SearchSpaceFunctions
{
    /// <summary>Python <c>optuna.search_space.intersection._calculate</c>.</summary>
    internal static (OrderedDict<string, BaseDistribution>? SearchSpace, int NextCachedTrialNumber) Calculate(
        IReadOnlyList<FrozenTrial> trials,
        bool includePruned = false,
        OrderedDict<string, BaseDistribution>? searchSpace = null,
        int cachedTrialNumber = -1)
    {
        var statesOfInterest = new List<TrialState>
        {
            TrialState.Complete,
            TrialState.Waiting,
            TrialState.Running,
        };

        if (includePruned)
            statesOfInterest.Add(TrialState.Pruned);

        int nextCachedTrialNumber = -1;

        for (int i = trials.Count - 1; i >= 0; i--)
        {
            var trial = trials[i];
            if (!statesOfInterest.Contains(trial.State))
                continue;

            if (nextCachedTrialNumber == -1)
                nextCachedTrialNumber = trial.Number + 1;

            if (cachedTrialNumber > trial.Number)
                break;

            if (!trial.State.IsFinished())
            {
                nextCachedTrialNumber = trial.Number;
                continue;
            }

            if (searchSpace is null)
            {
                searchSpace = new OrderedDict<string, BaseDistribution>(trial.DistributionsDict);
                continue;
            }

            var next = new OrderedDict<string, BaseDistribution>();
            foreach (var (name, distribution) in searchSpace)
            {
                if (trial.DistributionsDict.TryGetValue(name, out var trialDist) && trialDist.Equals(distribution))
                    next[name] = distribution;
            }
            searchSpace = next;
        }

        return (searchSpace, nextCachedTrialNumber);
    }

    /// <summary>
    ///     Return the intersection search space of the given trials.
    ///
    ///     Intersection search space contains the intersection of parameter distributions that
    ///     have been suggested in the completed trials of the study so far. If there are multiple
    ///     parameters that have the same name but different distributions, neither is included in
    ///     the resulting search space (i.e., the parameters with dynamic value ranges are
    ///     excluded).
    /// </summary>
    /// <returns>
    ///     A dictionary containing the parameter names and parameter's distributions sorted by
    ///     parameter names.
    /// </returns>
    public static OrderedDict<string, BaseDistribution> IntersectionSearchSpaceOf(
        IReadOnlyList<FrozenTrial> trials, bool includePruned = false)
    {
        var (searchSpace, _) = Calculate(trials, includePruned);
        return Sorted(searchSpace ?? new OrderedDict<string, BaseDistribution>());
    }

    /// <summary>
    ///     OptunaSharp extension (no Optuna analog): the <see cref="Calculate" /> fold for a study whose
    ///     objective has declared <see cref="ConditionalParams" />. Unconditional parameters intersect
    ///     exactly as in <see cref="Calculate" />. A conditional parameter is kept when a trial lacks it
    ///     (it was inactive there), joins the space the first time any trial carries it, and is excluded
    ///     for good (recorded in <paramref name="excluded" />) once two trials carry it with different
    ///     distributions — so, like the intersection, the result is order-independent: a conditional
    ///     parameter is in it iff some trial suggested it and every trial that did agrees on its
    ///     distribution. Completed trials whose parameter set contradicts the rules are reported through
    ///     <see cref="OptunaWarnings" /> once per parameter (tracked in <paramref name="warned" />; pass
    ///     <c>null</c> to stay silent). Same incremental-cache contract as <see cref="Calculate" />.
    /// </summary>
    internal static (OrderedDict<string, BaseDistribution>? SearchSpace, int NextCachedTrialNumber) CalculateConditional(
        IReadOnlyList<FrozenTrial> trials,
        bool includePruned,
        OrderedDict<string, BaseDistribution>? searchSpace,
        int cachedTrialNumber,
        ConditionalParams conditionalParams,
        HashSet<string> excluded,
        HashSet<string>? warned)
    {
        var statesOfInterest = new List<TrialState>
        {
            TrialState.Complete,
            TrialState.Waiting,
            TrialState.Running,
        };

        if (includePruned)
            statesOfInterest.Add(TrialState.Pruned);

        int nextCachedTrialNumber = -1;

        for (int i = trials.Count - 1; i >= 0; i--)
        {
            var trial = trials[i];
            if (!statesOfInterest.Contains(trial.State))
                continue;

            if (nextCachedTrialNumber == -1)
                nextCachedTrialNumber = trial.Number + 1;

            if (cachedTrialNumber > trial.Number)
                break;

            if (!trial.State.IsFinished())
            {
                nextCachedTrialNumber = trial.Number;
                continue;
            }

            // Each finished trial is folded exactly once (the cache skips it afterwards), so this is the
            // one place to check it against the rules. Pruned trials stop mid-objective and legitimately
            // lack parameters they never reached, so only complete trials are judged.
            if (warned is not null && trial.State == TrialState.Complete)
                WarnContradictions(conditionalParams, trial, warned);

            if (searchSpace is null)
            {
                searchSpace = new OrderedDict<string, BaseDistribution>(trial.DistributionsDict);
                continue;
            }

            var next = new OrderedDict<string, BaseDistribution>();
            foreach (var (name, distribution) in searchSpace)
            {
                if (trial.DistributionsDict.TryGetValue(name, out var trialDist))
                {
                    if (trialDist.Equals(distribution))
                        next[name] = distribution;
                    else if (conditionalParams.IsConditional(name))
                        excluded.Add(name);  // A dynamic distribution: never evolve it relatively.
                }
                else if (conditionalParams.IsConditional(name))
                {
                    next[name] = distribution;  // Absent because inactive — the expected case.
                }
            }
            foreach (var (name, distribution) in trial.DistributionsDict)
            {
                if (!conditionalParams.IsConditional(name) || next.ContainsKey(name) || excluded.Contains(name))
                    continue;
                next[name] = distribution;  // First sighting of a conditional parameter.
            }
            searchSpace = next;
        }

        return (searchSpace, nextCachedTrialNumber);
    }

    private static void WarnContradictions(ConditionalParams conditionalParams, FrozenTrial trial, HashSet<string> warned)
    {
        foreach (var (name, message) in conditionalParams.Contradictions(trial))
        {
            if (!warned.Add(name))
                continue;
            OptunaWarnings.Warn(
                $"{message}. The `ConditionalParams` rules and the objective's suggest calls disagree about " +
                $"`{name}`; the sampler keeps working, but where they disagree that parameter is evolved less " +
                "effectively. This warning is shown once per parameter.");
        }
    }

    /// <summary>
    ///     OptunaSharp extension (no Optuna analog): the search space of the given trials with declared
    ///     <see cref="ConditionalParams" /> — the intersection over unconditional parameters, plus every
    ///     conditional parameter some trial suggested whose distribution never changed. See
    ///     <see cref="CalculateConditional" />.
    /// </summary>
    /// <returns>
    ///     A dictionary containing the parameter names and parameter's distributions sorted by
    ///     parameter names.
    /// </returns>
    public static OrderedDict<string, BaseDistribution> IntersectionSearchSpaceOf(
        IReadOnlyList<FrozenTrial> trials, ConditionalParams conditionalParams, bool includePruned = false)
    {
        if (conditionalParams is null)
            throw new ArgumentNullException(nameof(conditionalParams));
        var (searchSpace, _) = CalculateConditional(
            trials, includePruned, searchSpace: null, cachedTrialNumber: -1, conditionalParams,
            excluded: new HashSet<string>(StringComparer.Ordinal), warned: null);
        return Sorted(searchSpace ?? new OrderedDict<string, BaseDistribution>());
    }

    internal static OrderedDict<string, BaseDistribution> Sorted(OrderedDict<string, BaseDistribution> searchSpace)
    {
        var result = new OrderedDict<string, BaseDistribution>(searchSpace.Count);
        foreach (var kv in searchSpace.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            result[kv.Key] = kv.Value;
        return result;
    }
}

/// <summary>
///     A class to calculate the intersection search space of a <see cref="Study" />.
///     Port of <c>optuna/search_space/intersection.py: IntersectionSearchSpace</c>.
///
///     Intersection search space contains the intersection of parameter distributions that have
///     been suggested in the completed trials of the study so far. If there are multiple
///     parameters that have the same name but different distributions, neither is included in the
///     resulting search space (i.e., the parameters with dynamic value ranges are excluded).
///
///     Note that an instance of this class is supposed to be used for only one study. If
///     different studies are passed to <see cref="Calculate(Study, bool)" />, a
///     <see cref="ValueError" /> is raised.
/// </summary>
public class IntersectionSearchSpace
{
    private int _cachedTrialNumber = -1;
    private OrderedDict<string, BaseDistribution>? _searchSpace;
    private int? _studyId;

    private readonly bool _includePruned;
    private readonly ConditionalParams? _conditionalParams;
    // Conditional parameters seen with conflicting distributions (tombstoned for the study's lifetime)
    // and those already reported as contradicting the rules — both part of the incremental cache state.
    private readonly HashSet<string> _excludedConditional = new(StringComparer.Ordinal);
    private readonly HashSet<string> _warnedConditional = new(StringComparer.Ordinal);

    /// <param name="includePruned">Whether pruned trials should be included in the search space.</param>
    /// <param name="conditionalParams">
    ///     OptunaSharp extension: parameters declared conditional stay in the space when a trial lacks
    ///     them (they were inactive there) instead of falling out of the intersection — see
    ///     <see cref="ConditionalParams" /> and <see cref="SearchSpaceFunctions.CalculateConditional" />.
    ///     <c>null</c> (default) is Optuna's plain intersection.
    /// </param>
    public IntersectionSearchSpace(bool includePruned = false, ConditionalParams? conditionalParams = null)
    {
        _includePruned = includePruned;
        _conditionalParams = conditionalParams;
    }

    /// <summary>
    ///     Returns the intersection search space of the <see cref="Study" />.
    /// </summary>
    /// <param name="study">
    ///     A study with completed trials. The same study must be passed for one instance of this
    ///     class through its lifetime.
    /// </param>
    /// <param name="useCache">An option to use cached trials for each trial.</param>
    /// <returns>
    ///     A dictionary containing the parameter names and parameter's distributions sorted by
    ///     parameter names.
    /// </returns>
    public OrderedDict<string, BaseDistribution> Calculate(Study study, bool useCache = false)
    {
        if (_studyId is null)
        {
            _studyId = study.StudyId;
        }
        else
        {
            // Note that the check below is meaningless when InMemoryStorage is used because
            // InMemoryStorage.CreateNewStudy always returns the same study ID.
            if (_studyId != study.StudyId)
                throw new ValueError("`IntersectionSearchSpace` cannot handle multiple studies.");
        }

        if (_conditionalParams is null)
        {
            (_searchSpace, _cachedTrialNumber) = SearchSpaceFunctions.Calculate(
                study.GetTrialsInternal(deepcopy: false, useCache: useCache),
                _includePruned,
                _searchSpace,
                _cachedTrialNumber);
        }
        else
        {
            (_searchSpace, _cachedTrialNumber) = SearchSpaceFunctions.CalculateConditional(
                study.GetTrialsInternal(deepcopy: false, useCache: useCache),
                _includePruned,
                _searchSpace,
                _cachedTrialNumber,
                _conditionalParams,
                _excludedConditional,
                _warnedConditional);
        }
        var searchSpace = _searchSpace ?? new OrderedDict<string, BaseDistribution>();
        return SearchSpaceFunctions.Sorted(searchSpace);
    }
}
```


<a id="conditional"></a>

## src/OptunaSharp.Core/SearchSpace/ConditionalParams.cs

Search space · 394 physical lines · SHA-256: `3ccfe6bc6600a586fdee9ad8cd7a104db2eedef7187541c410f23d7df5d19bd0`

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.SearchSpace;

/// <summary>
///     Declares <b>conditional parameters</b> — parameters that only exist while an <i>activator</i>
///     parameter holds one of a set of values — for the genetic samplers
///     (<see cref="Samplers.NSGAIISampler" /> / <see cref="Samplers.NSGAIIISampler" />). An
///     <b>OptunaSharp extension</b> with no Optuna analog: opt-in, and the samplers' Optuna-parity path is
///     untouched when none is declared.
///
///     <para><b>Why it exists.</b> Optuna's define-by-run API lets an objective suggest a parameter under
///     a branch only (<c>if (optimizer == "adam") t.SuggestFloat("beta1", …)</c>), but the GA samplers
///     evolve the <i>intersection</i> search space — the parameters present in <i>every</i> completed
///     trial. A parameter absent from even one trial leaves the intersection for good and is resampled
///     uniformly at random by the independent sampler on every later trial: no crossover, no mutation,
///     no inheritance — no evolutionary pressure at all. Declaring the parameter here tells the sampler
///     that its absence is <i>expected</i> (it was inactive), so it stays in the relative search space and
///     is evolved like any other parameter whenever it is active, and is simply not present — never
///     blended, never counted — whenever it is not.</para>
///
///     <para><b>Rules.</b> <c>When(activator, values…).Activates(parameters…)</c> declares that each of
///     <c>parameters</c> is active only while <c>activator</c> holds one of <c>values</c> (Python <c>==</c>
///     semantics via <see cref="CategoricalChoice" />: <c>1 == 1.0 == true</c>, strings ordinal, a lone
///     <c>null</c> is <c>None</c>); <c>When(activator, predicate)</c> takes an arbitrary test instead — the
///     value arrives as the trial stores it: a boxed <see cref="long" /> for int parameters, a
///     <see cref="double" /> for float parameters, the raw choice value for categorical ones. Several
///     rules may name the same activator (they OR: any admissible value suffices) and a parameter may be
///     declared under several activators (they AND: every activator must admit). An activator may itself
///     be conditional (chains); self-activation and cycles are rejected at declaration.</para>
///
///     <para><b>What the sampler does with it.</b> (1) Relative search space: a conditional parameter joins
///     it once any completed trial has suggested it, provided every trial that did used the same
///     distribution (a changing distribution excludes it, exactly as the intersection would).
///     (2) Crossover: the categorical uniform crossover and the numerical crossover operator run over the
///     parameters the selected parents all carry; a parameter carried by only some parents is inherited
///     from one of those; one carried by none stays absent. (3) After crossover (or the no-crossover
///     parent copy) every conditional parameter whose activators say "inactive" is removed from the
///     child — it neither counts toward the default <c>1 / n_params</c> mutation probability nor reaches
///     the trial. (4) The objective's own <c>Suggest*</c> calls stay the final word: a proposed parameter
///     the objective never suggests is never recorded, and a suggested parameter the sampler did not
///     propose is drawn independently — the usual define-by-run contract. The sampler warns once per
///     parameter when a completed trial contradicts the rules (present while inactive, or absent while
///     active), which usually means the objective's branches and the declaration drifted apart.</para>
///
///     <example><code>
///     var rules = new ConditionalParams()
///         .When("optimizer", "adam", "adamw").Activates("beta1", "beta2")
///         .When("optimizer", "sgd").Activates("momentum")
///         .When("n_layers", v =&gt; (long)v! &gt;= 3).Activates("width_3");
///     var sampler = new NSGAIIISampler(populationSize: 32, seed: 0, conditionalParams: rules);
///     </code></example>
/// </summary>
public sealed class ConditionalParams
{
    /// <summary>One admissible-value test on an activator plus its display form for diagnostics.</summary>
    internal sealed record Condition(Func<object?, bool> Predicate, string Display);

    // dependent -> activator -> conditions. OR within an activator, AND across activators. Insertion
    // ordered so ToString, the dependency walk and every diagnostic are deterministic functions of the
    // declarations.
    private readonly OrderedDict<string, OrderedDict<string, List<Condition>>> _rules = new();
    private List<string>? _order;

    /// <summary>
    ///     Starts a rule: the parameters named by the following <see cref="ActivationBuilder.Activates" />
    ///     are active only while <paramref name="activator" /> holds one of <paramref name="values" />.
    /// </summary>
    /// <param name="activator">The parameter whose value gates the others.</param>
    /// <param name="values">
    ///     Admissible activator values, compared with Python <c>==</c> semantics (<c>1 == 1.0 == true</c>,
    ///     strings ordinal, NaN matches NaN as Optuna's categorical lookup does). A <c>null</c> element is
    ///     the choice <c>None</c> — spell it <c>(object?)null</c>, since a bare <c>null</c> literal is
    ///     ambiguous with the predicate overload.
    /// </param>
    public ActivationBuilder When(string activator, params object?[]? values)
    {
        ValidateName(activator, nameof(activator));
        // A null array (`When("a", (object?[]?)null)`) means the one admissible value None, not "no values".
        values ??= new object?[] { null };
        if (values.Length == 0)
            throw new ValueError($"`When(\"{activator}\", ...)` needs at least one admissible value.");
        var admissible = (object?[])values.Clone();
        string display = "in {" + string.Join(", ", admissible.Select(PythonRepr.Repr)) + "}";
        return new ActivationBuilder(
            this, activator, new Condition(v => admissible.Any(a => ValuesEqual(a, v)), display));
    }

    /// <summary>
    ///     Starts a rule with an arbitrary test on the activator's value — the value arrives as the trial
    ///     stores it: a boxed <see cref="long" /> for int parameters, a <see cref="double" /> for float
    ///     parameters, the raw choice value (<c>string</c>/<c>long</c>/<c>double</c>/<c>bool</c>/<c>null</c>)
    ///     for categorical ones.
    /// </summary>
    public ActivationBuilder When(string activator, Func<object?, bool> predicate)
    {
        ValidateName(activator, nameof(activator));
        if (predicate is null)
            throw new ArgumentNullException(nameof(predicate));
        return new ActivationBuilder(this, activator, new Condition(predicate, "satisfies predicate"));
    }

    /// <summary>The second half of a <see cref="When(string, object?[])" /> rule.</summary>
    public readonly struct ActivationBuilder
    {
        private readonly ConditionalParams _owner;
        private readonly string _activator;
        private readonly Condition _condition;

        internal ActivationBuilder(ConditionalParams owner, string activator, Condition condition)
        {
            _owner = owner;
            _activator = activator;
            _condition = condition;
        }

        /// <summary>Names the parameters the rule activates (one or more); returns the owner for chaining.</summary>
        public ConditionalParams Activates(params string[] parameters)
        {
            if (parameters is null || parameters.Length == 0)
                throw new ValueError("`Activates(...)` needs at least one parameter name.");
            foreach (var parameter in parameters)
                _owner.Add(parameter, _activator, _condition);
            return _owner;
        }
    }

    /// <summary>The conditional (activated) parameter names, in declaration order.</summary>
    public IReadOnlyCollection<string> Parameters => _rules.Keys;

    /// <summary>Whether <paramref name="name" /> was declared conditional.</summary>
    public bool IsConditional(string name) => _rules.ContainsKey(name);

    /// <summary>The activators <paramref name="name" /> was declared under (empty when unconditional).</summary>
    public IReadOnlyCollection<string> ActivatorsOf(string name)
        => _rules.TryGetValue(name, out var byActivator) ? byActivator.Keys : Array.Empty<string>();

    /// <summary>
    ///     Whether <paramref name="name" /> is active for a complete parameter assignment such as
    ///     <see cref="FrozenTrial.Params" />: unconditional parameters always are; a conditional one is
    ///     active when every activator it was declared under is present, admits the value it holds, and is
    ///     itself active.
    /// </summary>
    public bool IsActive(string name, IReadOnlyDictionary<string, object?> @params)
    {
        if (@params is null)
            throw new ArgumentNullException(nameof(@params));
        if (!_rules.TryGetValue(name, out var byActivator))
            return true;
        foreach (var (activator, conditions) in byActivator)
        {
            if (!@params.TryGetValue(activator, out var value))
                return false;
            if (!Admits(conditions, value))
                return false;
            if (!IsActive(activator, @params))
                return false;
        }
        return true;
    }

    /// <summary>
    ///     Whether <paramref name="value" /> of <paramref name="activator" /> admits <paramref name="name" />
    ///     under the declared rules — <c>true</c> when no rule links the two. Unlike <see cref="IsActive" />
    ///     this judges a single activator, so it can be asked of a partial assignment such as a sampler's
    ///     proposal in which other activators are still to be drawn.
    /// </summary>
    internal bool ActivatorAdmits(string name, string activator, object? value)
        => !_rules.TryGetValue(name, out var byActivator)
           || !byActivator.TryGetValue(activator, out var conditions)
           || Admits(conditions, value);

    /// <summary>Conditional parameter names ordered so that every activator precedes what it activates.</summary>
    internal IReadOnlyList<string> Order => _order ??= ComputeOrder();

    /// <summary>
    ///     Removes from <paramref name="child" /> (a child individual under construction, keyed by the
    ///     relative search space) every conditional parameter that is not active for it. Walks in
    ///     dependency order, so a pruned activator takes its dependents with it. An activator missing from
    ///     the child counts as <i>inactive</i> when the relative search space manages it (it was pruned, or
    ///     no parent carried it) and as <i>unknown</i> otherwise (single-valued, dynamic, or not yet seen —
    ///     it will be sampled independently), in which case the dependent is kept and the objective's own
    ///     suggest calls decide.
    /// </summary>
    internal void Prune(
        OrderedDict<string, object?> child, IReadOnlyDictionary<string, BaseDistribution> relativeSearchSpace)
    {
        foreach (var name in Order)
        {
            if (!child.ContainsKey(name))
                continue;
            if (!ShouldKeep(name, child, relativeSearchSpace))
                child.Remove(name);
        }
    }

    private bool ShouldKeep(
        string name, OrderedDict<string, object?> child, IReadOnlyDictionary<string, BaseDistribution> relativeSearchSpace)
    {
        foreach (var (activator, conditions) in _rules[name])
        {
            if (child.TryGetValue(activator, out var value))
            {
                if (!Admits(conditions, value))
                    return false;
            }
            else if (relativeSearchSpace.ContainsKey(activator))
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>
    ///     Every way a complete trial's parameter set disagrees with these rules — a conditional parameter
    ///     present while inactive, or absent while active — as (parameter, message) pairs. Empty when the
    ///     trial agrees. Used to warn once per parameter that the rules and the objective's suggest calls
    ///     differ.
    /// </summary>
    internal IEnumerable<(string Name, string Message)> Contradictions(FrozenTrial trial)
    {
        foreach (var name in Order)
        {
            bool present = trial.ParamsDict.ContainsKey(name);
            bool active = IsActive(name, trial.ParamsDict);
            if (present == active)
                continue;
            yield return (name, present
                ? $"The conditional parameter `{name}` is present in Trial#{trial.Number} although its rule " +
                  $"({Describe(name)}) says it is inactive there"
                : $"The conditional parameter `{name}` is absent from Trial#{trial.Number} although its rule " +
                  $"({Describe(name)}) says it is active there");
        }
    }

    /// <summary>Renders <paramref name="name" />'s rule: <c>`a` in {1, 2} and `b` satisfies predicate</c>.</summary>
    private string Describe(string name)
        => string.Join(" and ", _rules[name].Select(kv =>
            $"`{kv.Key}` " + string.Join(" or ", kv.Value.Select(c => c.Display))));

    public override string ToString()
        => "ConditionalParams(" + string.Join("; ", _rules.Keys.Select(name => $"`{name}` when {Describe(name)}")) + ")";

    private void Add(string parameter, string activator, Condition condition)
    {
        ValidateName(parameter, nameof(parameter));
        if (string.Equals(parameter, activator, StringComparison.Ordinal))
            throw new ValueError($"Parameter `{parameter}` cannot activate itself.");
        if (DependsOn(activator, parameter))
            throw new ValueError(
                $"Declaring `{parameter}` as activated by `{activator}` would create a cycle: " +
                $"`{activator}` already depends on `{parameter}`.");

        if (!_rules.TryGetValue(parameter, out var byActivator))
            _rules[parameter] = byActivator = new OrderedDict<string, List<Condition>>();
        if (!byActivator.TryGetValue(activator, out var conditions))
            byActivator[activator] = conditions = new List<Condition>();
        conditions.Add(condition);
        _order = null;
    }

    /// <summary>Whether <paramref name="name" /> is (transitively) activated by <paramref name="candidate" />.</summary>
    private bool DependsOn(string name, string candidate)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var stack = new Stack<string>();
        stack.Push(name);
        while (stack.Count > 0)
        {
            var current = stack.Pop();
            if (!_rules.TryGetValue(current, out var byActivator))
                continue;
            foreach (var activator in byActivator.Keys)
            {
                if (string.Equals(activator, candidate, StringComparison.Ordinal))
                    return true;
                if (seen.Add(activator))
                    stack.Push(activator);
            }
        }
        return false;
    }

    private List<string> ComputeOrder()
    {
        // Kahn's algorithm over activator -> dependent edges, ties broken by first mention, so the order
        // is a deterministic function of the declarations. Cycles cannot exist (rejected in Add).
        var nodes = new List<string>();
        var indegree = new Dictionary<string, int>(StringComparer.Ordinal);
        var dependents = new Dictionary<string, List<string>>(StringComparer.Ordinal);

        void Touch(string node)
        {
            if (!indegree.TryAdd(node, 0))
                return;
            nodes.Add(node);
            dependents[node] = new List<string>();
        }

        foreach (var (dependent, byActivator) in _rules)
        {
            foreach (var activator in byActivator.Keys)
                Touch(activator);
            Touch(dependent);
            foreach (var activator in byActivator.Keys)
            {
                dependents[activator].Add(dependent);
                indegree[dependent] += 1;
            }
        }

        var order = new List<string>(_rules.Count);
        var ready = new Queue<string>(nodes.Where(node => indegree[node] == 0));
        while (ready.Count > 0)
        {
            var node = ready.Dequeue();
            if (_rules.ContainsKey(node))
                order.Add(node);
            foreach (var dependent in dependents[node])
            {
                if (--indegree[dependent] == 0)
                    ready.Enqueue(dependent);
            }
        }

        System.Diagnostics.Debug.Assert(order.Count == _rules.Count, "Cycles are rejected at declaration.");
        return order;
    }

    private static bool Admits(List<Condition> conditions, object? value)
    {
        foreach (var condition in conditions)
        {
            if (condition.Predicate(value))
                return true;
        }
        return false;
    }

    private static void ValidateName(string name, string paramName)
    {
        if (name is null)
            throw new ArgumentNullException(paramName);
        if (name.Length == 0)
            throw new ValueError("Parameter names must be non-empty.");
    }

    /// <summary>
    ///     Python <c>==</c> between an admissible value and a stored parameter value (NaN matches NaN, as
    ///     Optuna's categorical lookup does); values outside the categorical-choice domain fall back to
    ///     <see cref="object.Equals(object?, object?)" />.
    /// </summary>
    internal static bool ValuesEqual(object? expected, object? actual)
    {
        if (TryChoice(expected, out var e) && TryChoice(actual, out var a))
            return CategoricalChoice.ChoicesEqual(e, a);
        return Equals(expected, actual);
    }

    private static bool TryChoice(object? value, out CategoricalChoice choice)
    {
        switch (value)
        {
            case null:
            case CategoricalChoice:
            case bool:
            case string:
            case double:
            case float:
            case sbyte:
            case byte:
            case short:
            case ushort:
            case int:
            case uint:
            case long:
                choice = CategoricalChoice.From(value);
                return true;
            case ulong ul when ul <= long.MaxValue:
                choice = CategoricalChoice.From(value);
                return true;
            default:
                choice = default;
                return false;
        }
    }
}
```


<a id="transform"></a>

## src/OptunaSharp.Core/Internal/SearchSpaceTransform.cs

Search space · 343 physical lines · SHA-256: `3e77d0a552eee249e5c5b5b32c8e3154abcda412cb93f2f90581329f584c1c32`

```csharp
using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp.Internal;

/// <summary>
///     Transform a search space and parameter configurations to continuous space.
///     Port of <c>optuna/_transform.py: _SearchSpaceTransform</c> — the numeric boundary of the
///     whole system.
///
///     The search space bounds and parameter configurations are represented as float64 arrays
///     and transformed into continuous space. Bounds and parameters associated with categorical
///     distributions are one-hot encoded. Parameter configurations in this space can additionally
///     be untransformed, or mapped back to the original space.
/// </summary>
/// <remarks>
///     <c>transformLog</c> and <c>transformStep</c> are useful for constructing bounds and
///     parameters without any actual transformations by setting those arguments to <c>false</c>.
///     This is needed for e.g. the hyperparameter importance assessments.
///     Parameter values are not scaled to the unit cube unless <c>transform01</c> is set.
/// </remarks>
internal sealed class SearchSpaceTransform
{
    private readonly double[,] _rawBounds;
    private readonly List<long[]> _columnToEncodedColumns;
    private readonly long[] _encodedColumnToColumn;
    private readonly OrderedDict<string, BaseDistribution> _searchSpace;
    private readonly bool _transformLog;
    private readonly bool _transform01;

    public SearchSpaceTransform(
        OrderedDict<string, BaseDistribution> searchSpace,
        bool transformLog = true,
        bool transformStep = true,
        bool transform01 = false)
    {
        (_rawBounds, _columnToEncodedColumns, _encodedColumnToColumn) =
            TransformSearchSpace(searchSpace, transformLog, transformStep);
        _searchSpace = searchSpace;
        _transformLog = transformLog;
        _transform01 = transform01;
    }

    /// <summary>Constructed bounds from the given search space, shape <c>(n_bounds, 2)</c>, float64.</summary>
    public NDArray Bounds
    {
        get
        {
            int n = _rawBounds.GetLength(0);
            var bounds = new double[n, 2];
            if (_transform01)
            {
                for (int i = 0; i < n; i++)
                {
                    bounds[i, 0] = 0.0;
                    bounds[i, 1] = 1.0;
                }
            }
            else
            {
                Array.Copy(_rawBounds, bounds, _rawBounds.Length);
            }
            return np.array(bounds);
        }
    }

    /// <summary>Raw (untransformed-to-unit-cube) bounds, shape <c>(n_bounds, 2)</c>.</summary>
    internal double[,] RawBounds => _rawBounds;

    /// <summary>Mapping from original parameter column index to encoded column indices.</summary>
    public IReadOnlyList<long[]> ColumnToEncodedColumns => _columnToEncodedColumns;

    /// <summary>Mapping from encoded column index to original parameter column index.</summary>
    public long[] EncodedColumnToColumn => _encodedColumnToColumn;

    /// <summary>
    ///     Transform a parameter configuration from actual values to continuous space.
    /// </summary>
    /// <param name="parameters">A parameter configuration to transform.</param>
    /// <returns>A 1-dimensional float64 array holding the transformed parameters.</returns>
    public NDArray Transform(IReadOnlyDictionary<string, object?> parameters)
    {
        int n = _rawBounds.GetLength(0);
        var transParams = new double[n];

        int boundIdx = 0;
        foreach (var (name, distribution) in _searchSpace)
        {
            System.Diagnostics.Debug.Assert(
                parameters.ContainsKey(name), "Parameter configuration must contain all distributions.");
            var param = parameters[name];

            if (distribution is CategoricalDistribution categorical)
            {
                int choiceIdx = (int)categorical.ToInternalRepr(param);
                transParams[boundIdx + choiceIdx] = 1;
                boundIdx += categorical.Choices.Count;
            }
            else
            {
                transParams[boundIdx] = TransformNumericalParam(
                    PyFloatValue(param), distribution, _transformLog);
                boundIdx += 1;
            }
        }

        if (_transform01)
        {
            for (int i = 0; i < n; i++)
            {
                bool single = _rawBounds[i, 0] == _rawBounds[i, 1];
                if (single)
                    transParams[i] = 0.5;
                else
                    transParams[i] = (transParams[i] - _rawBounds[i, 0]) / (_rawBounds[i, 1] - _rawBounds[i, 0]);
            }
        }

        return np.array(transParams);
    }

    /// <summary>
    ///     Untransform a parameter configuration from continuous space to actual values.
    /// </summary>
    /// <param name="transParams">A 1-dimensional array in the transformed space.</param>
    /// <returns>An untransformed parameter configuration keyed by parameter name.</returns>
    public OrderedDict<string, object?> Untransform(NDArray transParams)
    {
        int n = _rawBounds.GetLength(0);
        System.Diagnostics.Debug.Assert(transParams.shape.Length == 1 && transParams.shape[0] == n);

        var values = transParams.astype(np.float64).ToArray<double>();

        if (_transform01)
        {
            var scaled = new double[n];
            for (int i = 0; i < n; i++)
                scaled[i] = _rawBounds[i, 0] + values[i] * (_rawBounds[i, 1] - _rawBounds[i, 0]);
            values = scaled;
        }

        var parameters = new OrderedDict<string, object?>();

        int column = 0;
        foreach (var (name, distribution) in _searchSpace)
        {
            var encodedColumns = _columnToEncodedColumns[column];
            column++;

            object? param;
            if (distribution is CategoricalDistribution categorical)
            {
                // Select the highest rated one-hot encoding (np.argmax: first max wins).
                int argmax = 0;
                double best = values[encodedColumns[0]];
                for (int i = 1; i < encodedColumns.Length; i++)
                {
                    double v = values[encodedColumns[i]];
                    if (v > best)
                    {
                        best = v;
                        argmax = i;
                    }
                }
                param = categorical.ToExternalRepr(argmax);
            }
            else
            {
                param = UntransformNumericalParam(values[encodedColumns[0]], distribution, _transformLog);
            }

            parameters[name] = param;
        }

        return parameters;
    }

    /// <summary>Python <c>_transform_search_space</c>.</summary>
    private static (double[,], List<long[]>, long[]) TransformSearchSpace(
        OrderedDict<string, BaseDistribution> searchSpace, bool transformLog, bool transformStep)
    {
        System.Diagnostics.Debug.Assert(searchSpace.Count > 0, "Cannot transform if no distributions are given.");

        int nBounds = 0;
        foreach (var d in searchSpace.Values)
            nBounds += d is CategoricalDistribution c ? c.Choices.Count : 1;

        var bounds = new double[nBounds, 2];
        var columnToEncodedColumns = new List<long[]>();
        var encodedColumnToColumn = new long[nBounds];

        int boundIdx = 0;
        foreach (var distribution in searchSpace.Values)
        {
            switch (distribution)
            {
                case CategoricalDistribution categorical:
                {
                    int nChoices = categorical.Choices.Count;
                    var encodedColumns = new long[nChoices];
                    for (int i = 0; i < nChoices; i++)
                    {
                        bounds[boundIdx + i, 0] = 0;
                        bounds[boundIdx + i, 1] = 1;
                        encodedColumns[i] = boundIdx + i;
                        encodedColumnToColumn[boundIdx + i] = columnToEncodedColumns.Count;
                    }
                    columnToEncodedColumns.Add(encodedColumns);
                    boundIdx += nChoices;
                    break;
                }
                case FloatDistribution floatDist:
                {
                    double bdLow, bdHigh;
                    if (floatDist.Step is not null)
                    {
                        double halfStep = transformStep ? 0.5 * floatDist.Step.Value : 0.0;
                        bdLow = TransformNumericalParam(floatDist.Low, floatDist, transformLog) - halfStep;
                        bdHigh = TransformNumericalParam(floatDist.High, floatDist, transformLog) + halfStep;
                    }
                    else
                    {
                        bdLow = TransformNumericalParam(floatDist.Low, floatDist, transformLog);
                        bdHigh = TransformNumericalParam(floatDist.High, floatDist, transformLog);
                    }
                    bounds[boundIdx, 0] = bdLow;
                    bounds[boundIdx, 1] = bdHigh;
                    encodedColumnToColumn[boundIdx] = columnToEncodedColumns.Count;
                    columnToEncodedColumns.Add(new long[] { boundIdx });
                    boundIdx += 1;
                    break;
                }
                case IntDistribution intDist:
                {
                    double halfStep = transformStep ? 0.5 * intDist.Step : 0.0;
                    double bdLow, bdHigh;
                    if (intDist.Log)
                    {
                        bdLow = TransformNumericalParam(intDist.Low - halfStep, intDist, transformLog);
                        bdHigh = TransformNumericalParam(intDist.High + halfStep, intDist, transformLog);
                    }
                    else
                    {
                        bdLow = TransformNumericalParam(intDist.Low, intDist, transformLog) - halfStep;
                        bdHigh = TransformNumericalParam(intDist.High, intDist, transformLog) + halfStep;
                    }
                    bounds[boundIdx, 0] = bdLow;
                    bounds[boundIdx, 1] = bdHigh;
                    encodedColumnToColumn[boundIdx] = columnToEncodedColumns.Count;
                    columnToEncodedColumns.Add(new long[] { boundIdx });
                    boundIdx += 1;
                    break;
                }
                default:
                    throw new InvalidOperationException("Should not reach. Unexpected distribution.");
            }
        }

        System.Diagnostics.Debug.Assert(boundIdx == nBounds);

        return (bounds, columnToEncodedColumns, encodedColumnToColumn);
    }

    /// <summary>Python <c>_transform_numerical_param</c>.</summary>
    internal static double TransformNumericalParam(double param, BaseDistribution distribution, bool transformLog)
        => distribution switch
        {
            CategoricalDistribution => throw new InvalidOperationException("Should not reach. Should be one-hot encoded."),
            FloatDistribution f => f.Log && transformLog ? Math.Log(param) : param,
            IntDistribution i => i.Log && transformLog ? Math.Log(param) : param,
            _ => throw new InvalidOperationException("Should not reach. Unexpected distribution."),
        };

    /// <summary>Python <c>_untransform_numerical_param</c>. Returns a boxed double (float) or long (int).</summary>
    internal static object UntransformNumericalParam(double transParam, BaseDistribution distribution, bool transformLog)
    {
        switch (distribution)
        {
            case CategoricalDistribution:
                throw new InvalidOperationException("Should not reach. Should be one-hot encoded.");
            case FloatDistribution f:
            {
                double param;
                if (f.Log)
                {
                    param = transformLog ? Math.Exp(transParam) : transParam;
                    if (!f.Single())
                        param = Math.Min(param, MathUtil.NextAfter(f.High, f.High - 1));
                }
                else if (f.Step is not null)
                {
                    // np.round: banker's rounding — Math.Round's default mode.
                    param = MathUtil.Clip(
                        Math.Round((transParam - f.Low) / f.Step.Value) * f.Step.Value + f.Low, f.Low, f.High);
                }
                else
                {
                    param = f.Single()
                        ? transParam
                        : Math.Min(transParam, MathUtil.NextAfter(f.High, f.High - 1));
                }
                return param;
            }
            case IntDistribution i:
            {
                long param;
                if (i.Log)
                {
                    if (transformLog)
                    {
                        // Python round(): banker's rounding to int.
                        param = (long)Math.Round(Math.Exp(transParam), MidpointRounding.ToEven);
                        param = Math.Min(Math.Max(param, i.Low), i.High);
                    }
                    else
                    {
                        param = (long)transParam;
                    }
                }
                else
                {
                    // Python: math.floor((t - low) / step + 0.5) — round half up.
                    long stepIndex = (long)Math.Floor((transParam - i.Low) / i.Step + 0.5);
                    param = stepIndex * i.Step + i.Low;
                    param = Math.Min(Math.Max(param, i.Low), i.High);
                }
                return param;
            }
            default:
                throw new InvalidOperationException("Should not reach. Unexpected distribution.");
        }
    }

    private static double PyFloatValue(object? param)
    {
        if (!PyFloat.TryConvert(param, out double value))
            throw new NumSharp.TypeError($"must be real number, not {(param is null ? "NoneType" : param.GetType().Name)}");
        return value;
    }
}
```


<a id="linalg"></a>

## src/OptunaSharp.Core/Internal/LinAlg.cs

Numerical support · 177 physical lines · SHA-256: `29e3bc0a8c0b235100d75d344436f915b9170286b8ae8ec6b78c1bcb9426e6e6`

```csharp
using System;
using NumSharp;

namespace OptunaSharp.Internal;

/// <summary>
///     Dense linear-algebra helpers over plain managed arrays (<c>double[,]</c> / <c>double[]</c>)
///     for the samplers that work outside NumSharp's <c>NDArray</c> world. Every factorisation
///     delegates to NumSharp's <c>np.linalg</c>, which — with <c>NumSharp.Interop.OpenBLAS</c>
///     referenced by OptunaSharp.Core and force-loaded by <see cref="OpenBlasActivation" /> — runs
///     through LAPACK and is <b>byte-identical to NumPy</b> (verified: <c>cholesky</c>,
///     <c>eigh</c> values+vectors, and <c>solve</c> match numpy 2.4.2 bit-for-bit). This is what
///     keeps the linalg-backed samplers byte-exact rather than managed-tolerance: NSGA-III's
///     hyperplane <see cref="Solve" />/<see cref="MatrixRank" /> (2.14, now byte-exact), CMA-ES's
///     <see cref="Eigh" /> (2.16), and the GP stack's <see cref="Cholesky" /> (3.2).
///
///     <para><see cref="SolveTriangular(double[,],double[],bool)" /> is the one primitive still
///     computed in managed code — a plain forward/back substitution, exact to a rounding. NumSharp
///     surfaces no <em>public</em> triangular solve to route it through (no <c>np.linalg</c> entry,
///     no <c>IBlasBackend.TryTrsm</c>); its OpenBLAS interop does now bind <c>cblas_dtrsm</c> at the
///     native layer (<c>OpenBlasNative.Dtrsm</c>), but that binding is <c>internal</c> and
///     unreachable from OptunaSharp — so this awaits a public NumSharp triangular-solve op. It is one
///     of the GP fit's documented tolerance sources (alongside the hand-derived MLL gradient and the
///     managed L-BFGS-B, both a different float-op sequence than torch/scipy), which is why the GP
///     oracle is tolerance-exact, not bit-exact — a boundary no linalg-backend switch can remove.
///     By contrast CMA-ES's <see cref="Eigh" /> and the GP's <see cref="Cholesky" /> already route
///     through OpenBLAS and are byte-identical to NumPy (the "5.5 byte-exactness switch," folded into
///     Core when OpenBLAS became a Core dependency — verified bit-for-bit vs numpy 2.4.2).</para>
/// </summary>
internal static class LinAlg
{
    /// <summary>Solve <c>a x = b</c> for a square <paramref name="a" /> — <c>np.linalg.solve</c> (LAPACK <c>gesv</c>).</summary>
    internal static double[] Solve(double[,] a, double[] b)
        => np.linalg.solve(np.array(a), np.array(b)).ToArray<double>();

    /// <summary>Matrix product <c>a · b</c> — <c>np.dot</c> (CBLAS <c>gemm</c>, byte-identical to NumPy).</summary>
    internal static double[,] Dot(double[,] a, double[,] b)
        => To2D(np.dot(np.array(a), np.array(b)), a.GetLength(0), b.GetLength(1));

    /// <summary>Matrix-vector product <c>a · b</c> — <c>np.dot</c> (byte-identical to NumPy).</summary>
    internal static double[] Dot(double[,] a, double[] b)
        => np.dot(np.array(a), np.array(b)).ToArray<double>();

    /// <summary>Outer product <c>a ⊗ b</c> — <c>np.outer</c> (an elementwise product, exact).</summary>
    internal static double[,] Outer(double[] a, double[] b)
        => To2D(np.outer(np.array(a), np.array(b)), a.Length, b.Length);

    /// <summary>Euclidean norm <c>‖a‖₂</c> — <c>np.linalg.norm</c>.</summary>
    internal static double Norm(double[] a)
        => (double)np.linalg.norm(np.array(a));

    /// <summary>The rank of <paramref name="a" /> — <c>np.linalg.matrix_rank</c> (SVD-based tolerance).</summary>
    internal static int MatrixRank(double[,] a)
        => (int)Convert.ToInt64(np.linalg.matrix_rank(np.array(a)).GetValue(0));

    /// <summary>The determinant of <paramref name="a" /> — <c>np.linalg.det</c> (LAPACK LU).</summary>
    internal static double Det(double[,] a)
        => (double)np.linalg.det(np.array(a));

    /// <summary>
    ///     The lower-triangular Cholesky factor <c>L</c> of a symmetric positive-definite
    ///     <paramref name="a" /> (<c>a = L Lᵀ</c>) — <c>np.linalg.cholesky</c> (LAPACK <c>potrf</c>).
    ///     Raises <see cref="LinAlgError" /> when <paramref name="a" /> is not positive definite.
    /// </summary>
    internal static double[,] Cholesky(double[,] a)
    {
        int n = a.GetLength(0);
        return To2D(np.linalg.cholesky(np.array(a)), n, n);
    }

    /// <summary>
    ///     Eigenvalues (ascending) and orthonormal eigenvectors (as columns) of a real symmetric
    ///     <paramref name="a" /> — <c>np.linalg.eigh</c> (LAPACK <c>syevd</c>, <c>UPLO='L'</c>), so
    ///     <c>a = V · diag(w) · Vᵀ</c>. Byte-identical to NumPy, signs included.
    /// </summary>
    internal static (double[] Eigenvalues, double[,] Eigenvectors) Eigh(double[,] a)
    {
        int n = a.GetLength(0);
        var (w, v) = np.linalg.eigh(np.array(a));
        return (w.ToArray<double>(), To2D(v, n, n));
    }

    /// <summary>
    ///     Solve the triangular system <c>a x = b</c> for a vector. <paramref name="upper" /> selects
    ///     back substitution (upper <paramref name="a" />) vs forward (lower). Managed substitution —
    ///     see the type remarks.
    /// </summary>
    internal static double[] SolveTriangular(double[,] a, double[] b, bool upper)
        => TriSolveVec(a, b, upper, transpose: false);

    /// <summary>
    ///     Triangular solve with a matrix right-hand side — torch's
    ///     <c>solve_triangular(a, b, upper, left)</c>. <paramref name="left" /> <c>true</c> solves
    ///     <c>a X = b</c> (column by column); <c>false</c> solves <c>X a = b</c> (row by row).
    /// </summary>
    internal static double[,] SolveTriangular(double[,] a, double[,] b, bool upper, bool left = true)
    {
        int rows = b.GetLength(0);
        int cols = b.GetLength(1);
        int n = a.GetLength(0);
        var result = new double[rows, cols];

        if (left)
        {
            var column = new double[n];
            for (int c = 0; c < cols; c++)
            {
                for (int i = 0; i < n; i++)
                    column[i] = b[i, c];
                double[] solved = TriSolveVec(a, column, upper, transpose: false);
                for (int i = 0; i < n; i++)
                    result[i, c] = solved[i];
            }
        }
        else
        {
            var rowVec = new double[n];
            for (int r = 0; r < rows; r++)
            {
                for (int j = 0; j < n; j++)
                    rowVec[j] = b[r, j];
                double[] solved = TriSolveVec(a, rowVec, upper, transpose: true);
                for (int j = 0; j < n; j++)
                    result[r, j] = solved[j];
            }
        }
        return result;
    }

    /// <summary>
    ///     Solve <c>op(a) x = b</c> where <c>op(a)</c> is <c>a</c> (<paramref name="transpose" />
    ///     <c>false</c>) or <c>aᵀ</c> (<c>true</c>), <paramref name="a" /> triangular. The effective
    ///     system is upper-triangular (back substitution) iff <paramref name="upper" /> XOR
    ///     <paramref name="transpose" />.
    /// </summary>
    private static double[] TriSolveVec(double[,] a, double[] b, bool upper, bool transpose)
    {
        int n = b.Length;
        var x = new double[n];
        bool effectiveUpper = upper ^ transpose;

        double Get(int i, int j) => transpose ? a[j, i] : a[i, j];

        if (!effectiveUpper)
        {
            for (int i = 0; i < n; i++)
            {
                double s = b[i];
                for (int j = 0; j < i; j++)
                    s -= Get(i, j) * x[j];
                x[i] = s / Get(i, i);
            }
        }
        else
        {
            for (int i = n - 1; i >= 0; i--)
            {
                double s = b[i];
                for (int j = i + 1; j < n; j++)
                    s -= Get(i, j) * x[j];
                x[i] = s / Get(i, i);
            }
        }
        return x;
    }

    /// <summary>Reshape a row-major flattened <c>NDArray</c> back into a <c>double[rows, cols]</c>.</summary>
    private static double[,] To2D(NDArray nd, int rows, int cols)
    {
        double[] flat = nd.ToArray<double>();
        var result = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = flat[i * cols + j];
        return result;
    }
}
```


<a id="math"></a>

## src/OptunaSharp.Core/Internal/MathUtil.cs

Numerical support · 187 physical lines · SHA-256: `33e4d1b2f84685a56e5b7ca9422371ec83da27de00cf1912d958a03c69700193`

```csharp
using System;
using System.Globalization;

namespace OptunaSharp.Internal;

internal static class MathUtil
{
    /// <summary>
    ///     Python <c>f"{x:.3g}"</c> — 3 significant figures, C-style <c>%g</c>. .NET's <c>"G3"</c>
    ///     reproduces C <c>%g</c> exactly (same fixed/scientific threshold, same rounding, same
    ///     trailing-zero stripping, min-2-digit exponent) except for the uppercase <c>E</c>, so
    ///     lowercasing it makes the two byte-identical (verified against CPython's <c>format</c>).
    /// </summary>
    internal static string FormatG3(double x)
        => x.ToString("G3", CultureInfo.InvariantCulture).Replace("E", "e");

    /// <summary>C99 / NumPy <c>nextafter(x, y)</c>: the next representable double after <paramref name="x" /> toward <paramref name="y" />.</summary>
    internal static double NextAfter(double x, double y)
    {
        if (double.IsNaN(x) || double.IsNaN(y))
            return x + y;
        if (x == y)
            return y;
        return y > x ? Math.BitIncrement(x) : Math.BitDecrement(x);
    }

    /// <summary>NumPy <c>np.clip(value, low, high)</c> for scalars (NaN propagates).</summary>
    internal static double Clip(double value, double low, double high)
        => Math.Min(Math.Max(value, low), high);

    /// <summary>
    ///     C99 / NumPy <c>np.log1p(x)</c> — <c>log(1 + x)</c> accurate for small <c>x</c>. .NET has
    ///     no <c>Math.Log1p</c>; this correction-term form (<c>log(u) - ((u-1) - x)/u</c>, u = 1+x)
    ///     is bit-identical to NumPy's <c>log1p</c> because <see cref="Math.Log(double)" /> already
    ///     matches NumPy's <c>np.log</c> bit-for-bit (verified over 220k points, 0 ULP).
    /// </summary>
    internal static double Log1p(double x)
    {
        if (double.IsNaN(x))
            return double.NaN;
        double u = 1.0 + x;
        if (u == 1.0)
            return x;  // |x| below the rounding of 1 + x (incl. x == 0): log1p(x) == x.
        if (u <= 0.0)
            return Math.Log(u);  // x <= -1: -inf at x == -1, NaN below — same as np.log(1 + x).
        if (double.IsPositiveInfinity(u))
            return Math.Log(u);
        return Math.Log(u) - ((u - 1.0) - x) / u;
    }

    private static readonly double Loge2 = Math.Log(2.0);

    /// <summary>
    ///     C99 / NumPy <c>np.expm1(x)</c> — <c>exp(x) - 1</c> accurate for small <c>x</c>. .NET has
    ///     no <c>Math.Expm1</c>; this Kahan form (<c>(u-1)*x/log(u)</c>, u = exp(x)) is bit-identical
    ///     to NumPy's <c>expm1</c> because <see cref="Math.Exp(double)" />/<see cref="Math.Log(double)" />
    ///     already match NumPy's <c>np.exp</c>/<c>np.log</c> bit-for-bit (verified against Python over
    ///     the TPE truncnorm input range).
    /// </summary>
    internal static double Expm1(double x)
    {
        if (double.IsNaN(x))
            return double.NaN;
        double u = Math.Exp(x);
        if (u == 1.0)
            return x;  // |x| below the rounding of exp(x): expm1(x) == x.
        double y = u - 1.0;
        if (y == -1.0)
            return -1.0;  // exp(x) underflowed to 0 (x very negative).
        if (double.IsPositiveInfinity(u))
            return u;  // overflow.
        return y * x / Math.Log(u);
    }

    /// <summary>
    ///     NumPy <c>np.logaddexp(a, b) = log(exp(a) + exp(b))</c>, computed stably. Reproduces
    ///     NumPy's <c>npy_logaddexp</c> exactly (the <c>a == b</c> branch adds <c>log(2)</c> and
    ///     handles the infinities); bit-identical to Python because it rides on the bit-exact
    ///     <see cref="Log1p" /> and <see cref="Math.Exp(double)" />.
    /// </summary>
    internal static double LogAddExp(double a, double b)
    {
        if (a == b)
            return a + Loge2;  // Also covers a == b == ±inf.
        double tmp = a - b;
        if (tmp > 0)
            return a + Log1p(Math.Exp(-tmp));
        if (tmp <= 0)
            return b + Log1p(Math.Exp(tmp));
        return tmp;  // NaN input.
    }

    /// <summary>
    ///     NumPy's <c>np.add.reduce</c> pairwise summation for a contiguous float64 array — the
    ///     exact accumulation order of <c>ndarray.sum()</c>, so reductions stay bit-identical.
    /// </summary>
    internal static double PairwiseSum(double[] a) => PairwiseSum(a, 0, a.Length);

    /// <inheritdoc cref="PairwiseSum(double[])" />
    internal static double PairwiseSum(double[] a, int offset, int n)
    {
        if (n < 8)
        {
            double res = 0.0;
            for (int i = 0; i < n; i++)
                res += a[offset + i];
            return res;
        }

        if (n <= 128)
        {
            double r0 = a[offset], r1 = a[offset + 1], r2 = a[offset + 2], r3 = a[offset + 3];
            double r4 = a[offset + 4], r5 = a[offset + 5], r6 = a[offset + 6], r7 = a[offset + 7];
            int i;
            for (i = 8; i < n - n % 8; i += 8)
            {
                r0 += a[offset + i];
                r1 += a[offset + i + 1];
                r2 += a[offset + i + 2];
                r3 += a[offset + i + 3];
                r4 += a[offset + i + 4];
                r5 += a[offset + i + 5];
                r6 += a[offset + i + 6];
                r7 += a[offset + i + 7];
            }

            double res = (r0 + r1 + (r2 + r3)) + (r4 + r5 + (r6 + r7));
            for (; i < n; i++)
                res += a[offset + i];
            return res;
        }

        // Divide by two but avoid non-multiples of the unroll factor.
        int n2 = n / 2;
        n2 -= n2 % 8;
        return PairwiseSum(a, offset, n2) + PairwiseSum(a, offset + n2, n - n2);
    }

    /// <summary>
    ///     NumPy <c>np.linspace(start, stop, num)</c> (endpoint=True) — bit-identical to NumPy for
    ///     the common case: <c>y[i] = i * step + start</c> with <c>step = (stop - start) / (num - 1)</c>,
    ///     and the final element pinned to <paramref name="stop" /> exactly.
    /// </summary>
    internal static double[] Linspace(double start, double stop, int num)
    {
        if (num <= 0)
            return Array.Empty<double>();
        if (num == 1)
            return new[] { start };

        int div = num - 1;
        double step = (stop - start) / div;
        var y = new double[num];
        for (int i = 0; i < num; i++)
            y[i] = i * step + start;  // arange(num) * step + start.
        y[^1] = stop;  // endpoint override.
        return y;
    }

    /// <summary>
    ///     NumPy <c>np.mean</c> over a float64 array — <c>PairwiseSum(a) / a.Length</c>, so the
    ///     accumulation order matches NumPy's <c>add.reduce</c> bit-for-bit. Empty → NaN (as NumPy).
    /// </summary>
    internal static double Mean(double[] a)
        => a.Length == 0 ? double.NaN : PairwiseSum(a) / a.Length;

    /// <summary>
    ///     NumPy <c>np.std</c> (population, <c>ddof=0</c>) over a float64 array. Reproduces NumPy's
    ///     algorithm exactly: <c>mean = sum/n</c>; residuals squared via <c>x*x</c> (NumPy array
    ///     <c>**2</c> is a multiply, not <see cref="Math.Pow(double,double)" />); variance is the
    ///     pairwise sum of those over <c>n</c>; then <see cref="Math.Sqrt(double)" />. Empty → NaN.
    /// </summary>
    internal static double Std(double[] a)
    {
        int n = a.Length;
        if (n == 0)
            return double.NaN;
        double mean = PairwiseSum(a) / n;
        var sq = new double[n];
        for (int i = 0; i < n; i++)
        {
            double d = a[i] - mean;
            sq[i] = d * d;
        }
        return Math.Sqrt(PairwiseSum(sq) / n);
    }
}
```


<a id="openblas"></a>

## src/OptunaSharp.Core/Internal/OpenBlasActivation.cs

Numerical support · 32 physical lines · SHA-256: `9c2b9b595bf5a2bd9a7d56665fec70268454fe913f8c29b94afb3ed2bdbaf9ce`

```csharp
using System.Runtime.CompilerServices;
using NumSharp.Interop.OpenBLAS;

namespace OptunaSharp.Internal;

/// <summary>
///     Forces <c>NumSharp.Interop.OpenBLAS</c> to load at OptunaSharp startup so its native LAPACK/
///     CBLAS backend is registered on NumSharp's <c>TensorEngine.Blas</c> before any
///     <c>np.linalg</c> call runs. With the backend live, NumSharp's <c>cholesky</c>/<c>eigh</c>/
///     <c>solve</c>/<c>matrix_rank</c>/<c>svd</c>/<c>qr</c> compute through LAPACK and are
///     byte-identical to NumPy — the basis for OptunaSharp's byte-exact linalg-backed samplers
///     (NSGA-III's hyperplane solve, CMA-ES's eigendecomposition, the GP stack's Cholesky).
///
///     <para><b>Why a module initializer and not just the package reference.</b> The OpenBLAS
///     backend self-registers from its own <c>[ModuleInitializer]</c>, but the CLR runs a referenced
///     assembly's module initializer only when code first touches one of its types — a bare
///     project/package reference that nothing in OptunaSharp mentions leaves the assembly unloaded,
///     and <c>np.linalg.cholesky</c> then throws <c>OpenBlasMissingBackendException</c>. Touching
///     <see cref="OpenBlasEngine" /> here, from OptunaSharp.Core's own module initializer, loads the
///     assembly (running its registration) the moment OptunaSharp.Core loads. If the native binary
///     cannot be loaded on the host, the OpenBLAS initializer swallows it and
///     <see cref="OpenBlasEngine.LapackAvailable" /> is simply <c>false</c>.</para>
/// </summary>
internal static class OpenBlasActivation
{
    [ModuleInitializer]
    internal static void Activate()
    {
        // A static read of any OpenBLAS type forces the assembly load + its backend registration.
        _ = OpenBlasEngine.LapackAvailable;
    }
}
```


<a id="python-repr"></a>

## src/OptunaSharp.Core/Internal/PythonRepr.cs

Local support · 260 physical lines · SHA-256: `886dfa3deace6ee47e5e12e67780224f1cd6e57f08a7eb27f1881b23020cca27`

```csharp
using System;
using System.Collections;
using System.Globalization;
using System.Text;
using OptunaSharp.Distributions;

namespace OptunaSharp.Internal;

/// <summary>
///     Renders values the way CPython's <c>repr()</c> / <c>str()</c> do, so that OptunaSharp can
///     reproduce Optuna's error and log messages verbatim (a plan §2 ground rule). The float
///     formatting also backs the Python-compatible JSON writer — CPython's <c>json</c> uses
///     <c>float.__repr__</c> for numbers.
/// </summary>
internal static class PythonRepr
{
    /// <summary>
    ///     CPython <c>repr(float)</c>: the shortest decimal string that round-trips, formatted
    ///     fixed when the decimal point position is in (-4, 16], scientific otherwise
    ///     (see CPython <c>pystrtod.c: format_float_short</c>, mode 'r').
    /// </summary>
    internal static string FloatRepr(double value)
    {
        if (double.IsNaN(value))
            return "nan";
        if (double.IsPositiveInfinity(value))
            return "inf";
        if (double.IsNegativeInfinity(value))
            return "-inf";

        // .NET "R" yields the shortest round-trip decimal, e.g. "1", "-0", "1.5", "1E-05", "1.234E+16".
        string s = value.ToString("R", CultureInfo.InvariantCulture);

        bool negative = s[0] == '-';
        if (negative)
            s = s.Substring(1);

        // Split into digit string and decimal exponent (decpt = position of the decimal point
        // relative to the start of the digit string).
        string digits;
        int decpt;
        int e = s.IndexOfAny(new[] { 'E', 'e' });
        string mantissa = e >= 0 ? s.Substring(0, e) : s;
        int exp = e >= 0 ? int.Parse(s.Substring(e + 1), CultureInfo.InvariantCulture) : 0;
        int dot = mantissa.IndexOf('.');
        if (dot >= 0)
        {
            digits = mantissa.Remove(dot, 1);
            decpt = dot + exp;
        }
        else
        {
            digits = mantissa;
            decpt = mantissa.Length + exp;
        }

        // Strip leading zeros (keep at least one digit), adjusting decpt.
        int lead = 0;
        while (lead < digits.Length - 1 && digits[lead] == '0')
        {
            lead++;
            decpt--;
        }
        digits = digits.Substring(lead);
        // Strip trailing zeros (they carry no round-trip information).
        digits = digits.TrimEnd('0');
        if (digits.Length == 0)
        {
            digits = "0";
            decpt = 1;
        }

        var sb = new StringBuilder();
        if (negative)
            sb.Append('-');

        if (decpt <= -4 || decpt > 16)
        {
            // Scientific: d[.ddd]e±XX with a sign and at least two exponent digits.
            sb.Append(digits[0]);
            if (digits.Length > 1)
            {
                sb.Append('.');
                sb.Append(digits, 1, digits.Length - 1);
            }
            int sciExp = decpt - 1;
            sb.Append('e');
            sb.Append(sciExp < 0 ? '-' : '+');
            int absExp = Math.Abs(sciExp);
            if (absExp < 10)
                sb.Append('0');
            sb.Append(absExp.ToString(CultureInfo.InvariantCulture));
        }
        else if (decpt <= 0)
        {
            // 0.00ddd
            sb.Append("0.");
            sb.Append('0', -decpt);
            sb.Append(digits);
        }
        else if (decpt >= digits.Length)
        {
            // ddd00.0 (ADD_DOT_0: always show a fractional part in fixed notation).
            sb.Append(digits);
            sb.Append('0', decpt - digits.Length);
            sb.Append(".0");
        }
        else
        {
            sb.Append(digits, 0, decpt);
            sb.Append('.');
            sb.Append(digits, decpt, digits.Length - decpt);
        }

        return sb.ToString();
    }

    /// <summary>CPython <c>repr(str)</c>: single-quoted unless the text contains <c>'</c> but no <c>"</c>.</summary>
    internal static string StringRepr(string value)
    {
        char quote = '\'';
        if (value.IndexOf('\'') >= 0 && value.IndexOf('"') < 0)
            quote = '"';

        var sb = new StringBuilder(value.Length + 2);
        sb.Append(quote);
        foreach (char c in value)
        {
            switch (c)
            {
                case '\\':
                    sb.Append("\\\\");
                    break;
                case '\n':
                    sb.Append("\\n");
                    break;
                case '\r':
                    sb.Append("\\r");
                    break;
                case '\t':
                    sb.Append("\\t");
                    break;
                default:
                    if (c == quote)
                    {
                        sb.Append('\\').Append(c);
                    }
                    else if (c < 0x20 || c == 0x7f)
                    {
                        sb.Append("\\x").Append(((int)c).ToString("x2", CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        sb.Append(c);
                    }
                    break;
            }
        }
        sb.Append(quote);
        return sb.ToString();
    }

    /// <summary>CPython <c>repr(value)</c> for the JSON-ish object model OptunaSharp carries around.</summary>
    internal static string Repr(object? value)
    {
        switch (value)
        {
            case null:
                return "None";
            case bool b:
                return b ? "True" : "False";
            case string s:
                return StringRepr(s);
            case double d:
                return FloatRepr(d);
            case float f:
                return FloatRepr(f);
            case sbyte or byte or short or ushort or int or uint or long:
                return Convert.ToInt64(value, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture);
            case ulong ul:
                return ul.ToString(CultureInfo.InvariantCulture);
            case CategoricalChoice choice:
                return Repr(choice.Value);
            case IDictionary dict:
                return DictRepr(dict);
            case IEnumerable seq:
                return ListRepr(seq);
            default:
                return value.ToString() ?? "None";
        }
    }

    /// <summary>CPython <c>str(value)</c>: like <see cref="Repr" /> but raw for strings.</summary>
    internal static string Str(object? value)
        => value is string s ? s : value is CategoricalChoice c && c.Value is string cs ? cs : Repr(value);

    /// <summary>Python list repr: <c>[a, b]</c>.</summary>
    internal static string ListRepr(IEnumerable values)
    {
        var sb = new StringBuilder("[");
        bool first = true;
        foreach (var item in values)
        {
            if (!first)
                sb.Append(", ");
            sb.Append(Repr(item));
            first = false;
        }
        return sb.Append(']').ToString();
    }

    /// <summary>Python tuple repr: <c>(a, b)</c>, with the single-element trailing comma <c>(a,)</c>.</summary>
    internal static string TupleRepr(IEnumerable values)
    {
        var sb = new StringBuilder("(");
        int count = 0;
        foreach (var item in values)
        {
            if (count > 0)
                sb.Append(", ");
            sb.Append(Repr(item));
            count++;
        }
        if (count == 1)
            sb.Append(',');
        return sb.Append(')').ToString();
    }

    /// <summary>Python dict repr: <c>{'a': 1, 'b': 2}</c> in insertion order.</summary>
    internal static string DictRepr(IDictionary dict)
    {
        var sb = new StringBuilder("{");
        bool first = true;
        foreach (DictionaryEntry entry in dict)
        {
            if (!first)
                sb.Append(", ");
            sb.Append(Repr(entry.Key)).Append(": ").Append(Repr(entry.Value));
            first = false;
        }
        return sb.Append('}').ToString();
    }

    /// <summary>Python set repr over string elements: <c>{'a', 'b'}</c> (or <c>set()</c> when empty).</summary>
    internal static string SetRepr(IEnumerable values)
    {
        var sb = new StringBuilder("{");
        bool first = true;
        foreach (var item in values)
        {
            if (!first)
                sb.Append(", ");
            sb.Append(Repr(item));
            first = false;
        }
        if (first)
            return "set()";
        return sb.Append('}').ToString();
    }
}
```


<a id="json"></a>

## src/OptunaSharp.Core/Internal/PyJson.cs

Local support · 553 physical lines · SHA-256: `1f66d96587518cabba575430f051a7469180153f93680553d1b8a00943bfd1ae`

```csharp
using System;
using System.Collections;
using System.Globalization;
using System.Collections.Generic;
using System.Text;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp.Internal;

/// <summary>
///     A self-contained JSON reader/writer matching CPython's <c>json</c> module semantics.
///
///     System.Text.Json cannot express what Optuna's storage formats need — <c>NaN</c> /
///     <c>Infinity</c> / <c>-Infinity</c> literals (Python's <c>allow_nan=True</c> default),
///     Python <c>repr</c> float formatting (<c>1.0</c>, not <c>1</c> — CPython's json uses
///     <c>float.__repr__</c>), <c>ensure_ascii=True</c> escaping, and insertion-ordered objects.
///     The journal-file format is byte-compatible with Python's because both sides serialize
///     through these rules.
///
///     The object model is: <c>null</c>, <see cref="bool" />, <see cref="long" />,
///     <see cref="double" />, <see cref="string" />, <see cref="List{T}" /> of <c>object?</c>,
///     and <see cref="OrderedDict{TKey,TValue}" /> of <c>string → object?</c> — the exact image
///     of Python's <c>json.loads</c> output.
/// </summary>
internal static class PyJson
{
    // ------------------------------------------------------------------ writing

    /// <summary>
    ///     Python <c>json.dumps(value)</c> with the default separators <c>(", ", ": ")</c>.
    /// </summary>
    internal static string Dumps(object? value) => Dumps(value, compact: false);

    /// <summary>
    ///     Python <c>json.dumps(value, separators=(",", ":"))</c> — the journal-file format.
    /// </summary>
    internal static string DumpsCompact(object? value) => Dumps(value, compact: true);

    private static string Dumps(object? value, bool compact)
    {
        var sb = new StringBuilder();
        WriteValue(sb, value, compact);
        return sb.ToString();
    }

    /// <summary>
    ///     Python <c>json.dumps(value, indent=n)</c> — pretty-printed with an <paramref name="indent" />-space
    ///     step, the separators <c>(",", ": ")</c>, and <c>{}</c>/<c>[]</c> for empty containers. Leaves
    ///     reuse the same writers as <see cref="Dumps(object)" />, so floats/strings/NaN stay Python-identical.
    /// </summary>
    internal static string DumpsIndented(object? value, int indent)
    {
        var sb = new StringBuilder();
        WriteValueIndented(sb, value, indent, 0);
        return sb.ToString();
    }

    private static void WriteValueIndented(StringBuilder sb, object? value, int indent, int level)
    {
        switch (value)
        {
            case IDictionary dict:
                WriteObjectIndented(sb, dict, indent, level);
                break;
            case string:
                WriteValue(sb, value, compact: false);
                break;
            case IEnumerable seq:
                WriteArrayIndented(sb, seq, indent, level);
                break;
            default:
                WriteValue(sb, value, compact: false);  // leaf (null/bool/number/enum/CategoricalChoice).
                break;
        }
    }

    private static void WriteObjectIndented(StringBuilder sb, IDictionary dict, int indent, int level)
    {
        if (dict.Count == 0)
        {
            sb.Append("{}");
            return;
        }
        sb.Append("{\n");
        string childPad = new string(' ', indent * (level + 1));
        bool first = true;
        foreach (DictionaryEntry entry in dict)
        {
            if (!first)
                sb.Append(",\n");
            sb.Append(childPad);
            WriteString(sb, KeyToString(entry.Key));
            sb.Append(": ");
            WriteValueIndented(sb, entry.Value, indent, level + 1);
            first = false;
        }
        sb.Append('\n').Append(' ', indent * level).Append('}');
    }

    private static void WriteArrayIndented(StringBuilder sb, IEnumerable seq, int indent, int level)
    {
        var e = seq.GetEnumerator();
        if (!e.MoveNext())
        {
            sb.Append("[]");
            return;
        }
        sb.Append("[\n");
        string childPad = new string(' ', indent * (level + 1));
        bool first = true;
        do
        {
            if (!first)
                sb.Append(",\n");
            sb.Append(childPad);
            WriteValueIndented(sb, e.Current, indent, level + 1);
            first = false;
        } while (e.MoveNext());
        sb.Append('\n').Append(' ', indent * level).Append(']');
    }

    private static void WriteValue(StringBuilder sb, object? value, bool compact)
    {
        switch (value)
        {
            case null:
                sb.Append("null");
                break;
            case bool b:
                sb.Append(b ? "true" : "false");
                break;
            case string s:
                WriteString(sb, s);
                break;
            case double d:
                WriteDouble(sb, d);
                break;
            case float f:
                WriteDouble(sb, f);
                break;
            case sbyte or byte or short or ushort or int or uint or long:
                sb.Append(Convert.ToInt64(value, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture));
                break;
            case ulong ul:
                sb.Append(ul.ToString(CultureInfo.InvariantCulture));
                break;
            case decimal m:
                WriteDouble(sb, (double)m);
                break;
            case Enum e:
                // Python IntEnum members (TrialState, StudyDirection, JournalOperation)
                // serialize as their integer value.
                sb.Append(Convert.ToInt64(e, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture));
                break;
            case CategoricalChoice choice:
                WriteValue(sb, choice.Value, compact);
                break;
            case IDictionary dict:
                WriteObject(sb, dict, compact);
                break;
            case IEnumerable seq:
                WriteArray(sb, seq, compact);
                break;
            default:
                throw new TypeError($"Object of type {value.GetType().Name} is not JSON serializable");
        }
    }

    private static void WriteDouble(StringBuilder sb, double d)
    {
        // Python: allow_nan=True writes the (non-standard) literals NaN / Infinity / -Infinity.
        if (double.IsNaN(d))
            sb.Append("NaN");
        else if (double.IsPositiveInfinity(d))
            sb.Append("Infinity");
        else if (double.IsNegativeInfinity(d))
            sb.Append("-Infinity");
        else
            sb.Append(PythonRepr.FloatRepr(d));
    }

    private static void WriteString(StringBuilder sb, string s)
    {
        // Python default: ensure_ascii=True — non-ASCII escapes to \uXXXX.
        sb.Append('"');
        foreach (char c in s)
        {
            switch (c)
            {
                case '"':
                    sb.Append("\\\"");
                    break;
                case '\\':
                    sb.Append("\\\\");
                    break;
                case '\n':
                    sb.Append("\\n");
                    break;
                case '\r':
                    sb.Append("\\r");
                    break;
                case '\t':
                    sb.Append("\\t");
                    break;
                case '\b':
                    sb.Append("\\b");
                    break;
                case '\f':
                    sb.Append("\\f");
                    break;
                default:
                    if (c < 0x20 || c > 0x7e)
                        sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    else
                        sb.Append(c);
                    break;
            }
        }
        sb.Append('"');
    }

    private static void WriteObject(StringBuilder sb, IDictionary dict, bool compact)
    {
        sb.Append('{');
        bool first = true;
        foreach (DictionaryEntry entry in dict)
        {
            if (!first)
                sb.Append(compact ? "," : ", ");
            WriteString(sb, KeyToString(entry.Key));
            sb.Append(compact ? ":" : ": ");
            WriteValue(sb, entry.Value, compact);
            first = false;
        }
        sb.Append('}');
    }

    private static void WriteArray(StringBuilder sb, IEnumerable seq, bool compact)
    {
        sb.Append('[');
        bool first = true;
        foreach (var item in seq)
        {
            if (!first)
                sb.Append(compact ? "," : ", ");
            WriteValue(sb, item, compact);
            first = false;
        }
        sb.Append(']');
    }

    /// <summary>Python json key coercion: str as-is, int/float/bool/None keys stringified Python-style.</summary>
    private static string KeyToString(object? key) => key switch
    {
        string s => s,
        bool b => b ? "true" : "false",
        null => "null",
        double d => double.IsNaN(d) ? "NaN"
            : double.IsPositiveInfinity(d) ? "Infinity"
            : double.IsNegativeInfinity(d) ? "-Infinity"
            : PythonRepr.FloatRepr(d),
        Enum e => Convert.ToInt64(e, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture),
        _ => Convert.ToInt64(key, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture),
    };

    // ------------------------------------------------------------------ reading

    /// <summary>Python <c>json.loads(text)</c>.</summary>
    internal static object? Loads(string text)
    {
        int pos = 0;
        var value = ParseValue(text, ref pos);
        SkipWhitespace(text, ref pos);
        if (pos != text.Length)
            throw new ValueError($"Extra data: line 1 column {pos + 1} (char {pos})");
        return value;
    }

    private static void SkipWhitespace(string s, ref int pos)
    {
        while (pos < s.Length && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
            pos++;
    }

    private static object? ParseValue(string s, ref int pos)
    {
        SkipWhitespace(s, ref pos);
        if (pos >= s.Length)
            throw new ValueError("Expecting value: line 1 column 1 (char 0)");

        char c = s[pos];
        switch (c)
        {
            case '{':
                return ParseObject(s, ref pos);
            case '[':
                return ParseArray(s, ref pos);
            case '"':
                return ParseString(s, ref pos);
            case 't':
                Expect(s, ref pos, "true");
                return true;
            case 'f':
                Expect(s, ref pos, "false");
                return false;
            case 'n':
                Expect(s, ref pos, "null");
                return null;
            case 'N':
                Expect(s, ref pos, "NaN");
                return double.NaN;
            case 'I':
                Expect(s, ref pos, "Infinity");
                return double.PositiveInfinity;
            default:
                if (c == '-' && pos + 1 < s.Length && s[pos + 1] == 'I')
                {
                    pos++;
                    Expect(s, ref pos, "Infinity");
                    return double.NegativeInfinity;
                }
                return ParseNumber(s, ref pos);
        }
    }

    private static void Expect(string s, ref int pos, string literal)
    {
        if (pos + literal.Length > s.Length || string.CompareOrdinal(s, pos, literal, 0, literal.Length) != 0)
            throw new ValueError($"Expecting value: line 1 column {pos + 1} (char {pos})");
        pos += literal.Length;
    }

    private static OrderedDict<string, object?> ParseObject(string s, ref int pos)
    {
        var result = new OrderedDict<string, object?>();
        pos++;  // '{'
        SkipWhitespace(s, ref pos);
        if (pos < s.Length && s[pos] == '}')
        {
            pos++;
            return result;
        }
        while (true)
        {
            SkipWhitespace(s, ref pos);
            if (pos >= s.Length || s[pos] != '"')
                throw new ValueError($"Expecting property name enclosed in double quotes: line 1 column {pos + 1} (char {pos})");
            string key = ParseString(s, ref pos);
            SkipWhitespace(s, ref pos);
            if (pos >= s.Length || s[pos] != ':')
                throw new ValueError($"Expecting ':' delimiter: line 1 column {pos + 1} (char {pos})");
            pos++;
            result[key] = ParseValue(s, ref pos);  // Python keeps the LAST duplicate key.
            SkipWhitespace(s, ref pos);
            if (pos >= s.Length)
                throw new ValueError($"Expecting ',' delimiter: line 1 column {pos + 1} (char {pos})");
            if (s[pos] == ',')
            {
                pos++;
                continue;
            }
            if (s[pos] == '}')
            {
                pos++;
                return result;
            }
            throw new ValueError($"Expecting ',' delimiter: line 1 column {pos + 1} (char {pos})");
        }
    }

    private static List<object?> ParseArray(string s, ref int pos)
    {
        var result = new List<object?>();
        pos++;  // '['
        SkipWhitespace(s, ref pos);
        if (pos < s.Length && s[pos] == ']')
        {
            pos++;
            return result;
        }
        while (true)
        {
            result.Add(ParseValue(s, ref pos));
            SkipWhitespace(s, ref pos);
            if (pos >= s.Length)
                throw new ValueError($"Expecting ',' delimiter: line 1 column {pos + 1} (char {pos})");
            if (s[pos] == ',')
            {
                pos++;
                continue;
            }
            if (s[pos] == ']')
            {
                pos++;
                return result;
            }
            throw new ValueError($"Expecting ',' delimiter: line 1 column {pos + 1} (char {pos})");
        }
    }

    private static string ParseString(string s, ref int pos)
    {
        pos++;  // opening '"'
        var sb = new StringBuilder();
        while (true)
        {
            if (pos >= s.Length)
                throw new ValueError($"Unterminated string starting at: line 1 column {pos} (char {pos - 1})");
            char c = s[pos++];
            if (c == '"')
                return sb.ToString();
            if (c == '\\')
            {
                if (pos >= s.Length)
                    throw new ValueError($"Unterminated string starting at: line 1 column {pos} (char {pos - 1})");
                char esc = s[pos++];
                switch (esc)
                {
                    case '"': sb.Append('"'); break;
                    case '\\': sb.Append('\\'); break;
                    case '/': sb.Append('/'); break;
                    case 'b': sb.Append('\b'); break;
                    case 'f': sb.Append('\f'); break;
                    case 'n': sb.Append('\n'); break;
                    case 'r': sb.Append('\r'); break;
                    case 't': sb.Append('\t'); break;
                    case 'u':
                        if (pos + 4 > s.Length)
                            throw new ValueError($"Invalid \\uXXXX escape: line 1 column {pos} (char {pos - 1})");
                        sb.Append((char)ushort.Parse(s.Substring(pos, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture));
                        pos += 4;
                        break;
                    default:
                        throw new ValueError($"Invalid \\escape: line 1 column {pos} (char {pos - 1})");
                }
            }
            else
            {
                sb.Append(c);
            }
        }
    }

    private static object ParseNumber(string s, ref int pos)
    {
        int start = pos;
        if (pos < s.Length && s[pos] == '-')
            pos++;
        bool isDouble = false;
        while (pos < s.Length)
        {
            char c = s[pos];
            if (c >= '0' && c <= '9')
            {
                pos++;
            }
            else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')
            {
                isDouble = isDouble || c == '.' || c == 'e' || c == 'E';
                pos++;
            }
            else
            {
                break;
            }
        }
        string token = s.Substring(start, pos - start);
        if (token.Length == 0 || token == "-")
            throw new ValueError($"Expecting value: line 1 column {start + 1} (char {start})");

        if (!isDouble)
        {
            // Python json parses integer literals as int (arbitrary precision); long covers
            // every value Optuna produces. Overflow falls back to double.
            if (long.TryParse(token, NumberStyles.Integer, CultureInfo.InvariantCulture, out long l))
                return l;
        }
        return double.Parse(token, NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    // ------------------------------------------------------------------ deep copy

    /// <summary>
    ///     Deep-copies a JSON-model value (Python <c>copy.deepcopy</c> over json-serializable data).
    /// </summary>
    internal static object? DeepCopy(object? value) => value switch
    {
        List<object?> list => DeepCopyList(list),
        OrderedDict<string, object?> dict => DeepCopyDict(dict),
        _ => value,  // null / bool / long / double / string / immutable values copy by reference.
    };

    private static List<object?> DeepCopyList(List<object?> list)
    {
        var copy = new List<object?>(list.Count);
        foreach (var item in list)
            copy.Add(DeepCopy(item));
        return copy;
    }

    private static OrderedDict<string, object?> DeepCopyDict(OrderedDict<string, object?> dict)
    {
        var copy = new OrderedDict<string, object?>(dict.Count);
        foreach (var kv in dict)
            copy[kv.Key] = DeepCopy(kv.Value);
        return copy;
    }

    /// <summary>
    ///     Python-<c>==</c> equality over the JSON object model (used by attr comparisons in tests
    ///     and by <c>FrozenTrial.__eq__</c>).
    /// </summary>
    internal static bool ValueEquals(object? a, object? b)
    {
        if (a is null || b is null)
            return a is null && b is null;
        if (a is bool ba)
            return b is bool bb ? ba == bb
                : b is long lb2 ? (ba ? 1 : 0) == lb2
                : b is double db2 && (ba ? 1.0 : 0.0) == db2;
        if (b is bool)
            return ValueEquals(b, a);
        if (a is long la)
            return b is long lb ? la == lb : b is double db && la == db;
        if (a is double da)
            return b is double db3 ? da == db3 : b is long lb3 && da == lb3;
        if (a is string sa)
            return b is string sb && sa == sb;
        if (a is List<object?> listA)
        {
            if (b is not List<object?> listB || listA.Count != listB.Count)
                return false;
            for (int i = 0; i < listA.Count; i++)
                if (!ValueEquals(listA[i], listB[i]))
                    return false;
            return true;
        }
        if (a is OrderedDict<string, object?> dictA)
        {
            if (b is not OrderedDict<string, object?> dictB || dictA.Count != dictB.Count)
                return false;
            foreach (var kv in dictA)
            {
                if (!dictB.TryGetValue(kv.Key, out var other) || !ValueEquals(kv.Value, other))
                    return false;
            }
            return true;
        }
        return a.Equals(b);
    }
}
```


<a id="warnings"></a>

## src/OptunaSharp.Core/Internal/OptunaWarnings.cs

Local support · 46 physical lines · SHA-256: `24adf5789ff20991622d1bf4a5a542e7b175336edef0cf9ebfd2269ad4aa81bc`

```csharp
using System;
using OptunaSharp.Logging;

namespace OptunaSharp.Internal;

/// <summary>
///     Categories mirroring the Python warning classes Optuna raises through
///     <c>optuna_warn(message, category)</c>.
/// </summary>
internal enum WarningCategory
{
    /// <summary>Python <c>UserWarning</c> (the <c>optuna_warn</c> default).</summary>
    UserWarning,

    /// <summary>Python <c>RuntimeWarning</c>.</summary>
    RuntimeWarning,

    /// <summary>Python <c>FutureWarning</c>.</summary>
    FutureWarning,

    /// <summary>Python <c>DeprecationWarning</c>.</summary>
    DeprecationWarning,

    /// <summary>Optuna's own <c>optuna.exceptions.ExperimentalWarning</c>.</summary>
    ExperimentalWarning,
}

/// <summary>
///     Port of optuna/_warnings.py — the seam every Optuna <c>warnings.warn</c> call goes
///     through. .NET has no warnings module, so warnings surface on the logging seam at
///     WARNING level; tests can intercept them via <see cref="WarningRaised" />.
/// </summary>
internal static class OptunaWarnings
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna");

    /// <summary>Raised for every warning, regardless of verbosity. Used by tests (`pytest.warns` analog).</summary>
    internal static event Action<string, WarningCategory>? WarningRaised;

    /// <summary>Python: <c>optuna_warn(message, category)</c>.</summary>
    internal static void Warn(string message, WarningCategory category = WarningCategory.UserWarning)
    {
        WarningRaised?.Invoke(message, category);
        _logger.Warning(category == WarningCategory.UserWarning ? message : $"{category}: {message}");
    }
}
```


<a id="blx-alpha-crossover"></a>

## src/OptunaSharp.Core/Samplers/Nsgaii/Crossovers/BLXAlphaCrossover.cs

Crossover operators · 42 physical lines · SHA-256: `f5ad867d38ec2af8b80ca02c6f71fd8074324a08993490357a4900cf86739fb7`

```csharp
using System;
using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Blend Crossover operation used by <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_crossovers/_blxalpha.py: BLXAlphaCrossover</c> (experimental v3.0.0).
///
///     Uniformly samples child individuals from the hyper-rectangles created by the two parent
///     individuals.
/// </summary>
public class BLXAlphaCrossover : BaseCrossover
{
    private readonly double _alpha;

    /// <param name="alpha">Parametrizes blend operation.</param>
    public BLXAlphaCrossover(double alpha = 0.5)
    {
        _alpha = alpha;
    }

    public override int NParents => 2;

    public override double[] Crossover(
        double[][] parentsParams, NumPyRandom rng, Study study, double[,] searchSpaceBounds)
    {
        int n = searchSpaceBounds.GetLength(0);
        var r = rng.rand(n).ToArray<double>();
        var child = new double[n];
        for (int j = 0; j < n; j++)
        {
            double parentsMin = Math.Min(parentsParams[0][j], parentsParams[1][j]);
            double parentsMax = Math.Max(parentsParams[0][j], parentsParams[1][j]);
            double diff = _alpha * (parentsMax - parentsMin);  // Equation (1).
            double low = parentsMin - diff;                    // Equation (1).
            double high = parentsMax + diff;                   // Equation (1).
            child[j] = (high - low) * r[j] + low;
        }
        return child;
    }
}
```


<a id="base-crossover"></a>

## src/OptunaSharp.Core/Samplers/Nsgaii/Crossovers/BaseCrossover.cs

Crossover operators · 39 physical lines · SHA-256: `54d9986a4abcbe875daf9f1d31de147b5689b9436766041100f3ad9ef25374ce`

```csharp
using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Base class for crossovers. Port of
///     <c>optuna/samplers/nsgaii/_crossovers/_base.py: BaseCrossover</c>.
///
///     A crossover operation is used by <see cref="NSGAIISampler" /> to create new parameter
///     combinations from parameters of <c>n</c> parent individuals.
///
///     Concrete implementations of this class are expected to only accept parameters from
///     numerical distributions. Only crossover for categorical parameters (uniform crossover) is
///     built into <see cref="NSGAIISampler" />.
/// </summary>
public abstract class BaseCrossover
{
    public override string ToString() => GetType().Name;

    /// <summary>Number of parent individuals required to perform crossover.</summary>
    public abstract int NParents { get; }

    /// <summary>
    ///     Perform crossover of selected parent individuals.
    /// </summary>
    /// <param name="parentsParams">
    ///     A <c>num_parents x num_parameters</c> array. Represents a parameter space for each
    ///     parent individual (continuous for numerical parameters). Row <c>i</c> is parent <c>i</c>.
    /// </param>
    /// <param name="rng">An instance of <c>numpy.random.RandomState</c>.</param>
    /// <param name="study">Target study object.</param>
    /// <param name="searchSpaceBounds">
    ///     A <c>len_search_space x 2</c> array of numerical distribution bounds constructed from the
    ///     transformed search space. Column 0 is lower, column 1 is upper.
    /// </param>
    /// <returns>A 1-dimensional array containing the new parameter combination.</returns>
    public abstract double[] Crossover(
        double[][] parentsParams, NumPyRandom rng, Study study, double[,] searchSpaceBounds);
}
```


<a id="sbx-crossover"></a>

## src/OptunaSharp.Core/Samplers/Nsgaii/Crossovers/SBXCrossover.cs

Crossover operators · 78 physical lines · SHA-256: `c6398cb98c598d0d193caca26f7933fa336742ea7c825be298a8afa06f68c61b`

```csharp
using System;
using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Simulated Binary Crossover operation used by <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_crossovers/_sbx.py: SBXCrossover</c> (experimental v3.0.0).
///
///     Generates a child from two parent individuals according to the polynomial probability
///     distribution. Optuna returns only one child, so it uses <c>uniform_crossover_prob</c> and
///     <c>use_child_gene_prob</c> to collapse the two SBX children into one.
/// </summary>
public class SBXCrossover : BaseCrossover
{
    private readonly double? _eta;
    private readonly double _uniformCrossoverProb;
    private readonly double _useChildGeneProb;

    /// <param name="eta">Distribution index. Defaults to 2 for single-objective, 20 for multi-objective.</param>
    /// <param name="uniformCrossoverProb">Probability of uniform crossover between the two candidate children.</param>
    /// <param name="useChildGeneProb">Per-variable probability of using the generated child value over the parent value.</param>
    public SBXCrossover(double? eta = null, double uniformCrossoverProb = 0.5, double useChildGeneProb = 0.5)
    {
        if (eta is not null && eta < 0.0)
            throw new ValueError("The value of `eta` must be greater than or equal to 0.0.");
        _eta = eta;

        if (uniformCrossoverProb < 0.0 || uniformCrossoverProb > 1.0)
            throw new ValueError("The value of `uniform_crossover_prob` must be in the range [0.0, 1.0].");
        if (useChildGeneProb <= 0.0 || useChildGeneProb > 1.0)
            throw new ValueError("The value of `use_child_gene_prob` must be in the range (0.0, 1.0].");
        _uniformCrossoverProb = uniformCrossoverProb;
        _useChildGeneProb = useChildGeneProb;
    }

    public override int NParents => 2;

    public override double[] Crossover(
        double[][] parentsParams, NumPyRandom rng, Study study, double[,] searchSpaceBounds)
    {
        int n = searchSpaceBounds.GetLength(0);
        double eta = _eta ?? (study.IsMultiObjective() ? 20.0 : 2.0);

        var us = rng.rand(n).ToArray<double>();

        var c1 = new double[n];
        var c2 = new double[n];
        for (int j = 0; j < n; j++)
        {
            double xls = searchSpaceBounds[j, 0];
            double xus = searchSpaceBounds[j, 1];
            double xsMin = Math.Min(parentsParams[0][j], parentsParams[1][j]);
            double xsMax = Math.Max(parentsParams[0][j], parentsParams[1][j]);

            double xsDiff = Math.Max(xsMax - xsMin, 1e-10);  // np.clip(..., 1e-10, None).
            double beta1 = 1 + 2 * (xsMin - xls) / xsDiff;
            double beta2 = 1 + 2 * (xus - xsMax) / xsDiff;
            double alpha1 = 2 - Math.Pow(beta1, -(eta + 1));
            double alpha2 = 2 - Math.Pow(beta2, -(eta + 1));

            double u = us[j];
            double betaq1 = u > 1 / alpha1  // Equation (3).
                ? Math.Pow(1 / (2 - u * alpha1), 1 / (eta + 1))
                : Math.Pow(u * alpha1, 1 / (eta + 1));
            double betaq2 = u > 1 / alpha2  // Equation (3).
                ? Math.Pow(1 / (2 - u * alpha2), 1 / (eta + 1))
                : Math.Pow(u * alpha2, 1 / (eta + 1));

            c1[j] = 0.5 * ((xsMin + xsMax) - betaq1 * xsDiff);  // Equation (4).
            c2[j] = 0.5 * ((xsMin + xsMax) + betaq2 * xsDiff);  // Equation (5).
        }

        var child = SbxLike.CollapseTwoChildren(
            c1, c2, parentsParams[0], parentsParams[1], rng, _useChildGeneProb, _uniformCrossoverProb);
        return child;
    }
}
```


<a id="spx-crossover"></a>

## src/OptunaSharp.Core/Samplers/Nsgaii/Crossovers/SPXCrossover.cs

Crossover operators · 73 physical lines · SHA-256: `179bd660088234ddca8bf6aa624cbf6f1bbe63e2e2e2e8112d2753535dd82678`

```csharp
using System;
using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Simplex Crossover operation used by <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_crossovers/_spx.py: SPXCrossover</c> (experimental v3.0.0).
///
///     Uniformly samples child individuals from within a single simplex similar to the simplex
///     produced by the (three) parent individuals.
/// </summary>
public class SPXCrossover : BaseCrossover
{
    private readonly double? _epsilon;

    /// <param name="epsilon">Expansion rate. If unset, defaults to <c>sqrt(len(search_space) + 2)</c>.</param>
    public SPXCrossover(double? epsilon = null)
    {
        _epsilon = epsilon;
    }

    public override int NParents => 3;

    public override double[] Crossover(
        double[][] parentsParams, NumPyRandom rng, Study study, double[,] searchSpaceBounds)
    {
        int nParents = NParents;
        int nParams = searchSpaceBounds.GetLength(0);
        int n = nParents - 1;

        // G = np.mean(parents_params, axis=0)  (Equation (1)).
        var g = new double[nParams];
        for (int c = 0; c < nParams; c++)
        {
            double sum = parentsParams[0][c];
            for (int k = 1; k < nParents; k++)
                sum += parentsParams[k][c];
            g[c] = sum / nParents;
        }

        // rs = np.power(rng.rand(n), 1 / (np.arange(n) + 1))  (Equation (2)).
        var rand = rng.rand(n).ToArray<double>();
        var rs = new double[n];
        for (int i = 0; i < n; i++)
            rs[i] = Math.Pow(rand[i], 1.0 / (i + 1));

        double epsilon = _epsilon ?? Math.Sqrt(nParams + 2);

        // xks = [G + epsilon * (pk - G) for pk in parents_params]  (Equation (3)).
        var xks = new double[nParents][];
        for (int k = 0; k < nParents; k++)
        {
            xks[k] = new double[nParams];
            for (int c = 0; c < nParams; c++)
                xks[k][c] = g[c] + epsilon * (parentsParams[k][c] - g[c]);
        }

        // ck = 0; for k in 1..n_parents: ck = rs[k-1] * (xks[k-1] - xks[k] + ck)  (Equation (4)).
        var ck = new double[nParams];
        for (int k = 1; k < nParents; k++)
        {
            for (int c = 0; c < nParams; c++)
                ck[c] = rs[k - 1] * (xks[k - 1][c] - xks[k][c] + ck[c]);
        }

        // child_params = xks[-1] + ck  (Equation (5)).
        var child = new double[nParams];
        for (int c = 0; c < nParams; c++)
            child[c] = xks[nParents - 1][c] + ck[c];
        return child;
    }
}
```


<a id="sbx-like"></a>

## src/OptunaSharp.Core/Samplers/Nsgaii/Crossovers/SbxLike.cs

Crossover operators · 53 physical lines · SHA-256: `59efd6c04a53f44b8bce5af8a5dbe23105a84297d7fc3ae41ec718bb58fa2aa2`

```csharp
using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Shared tail of <see cref="SBXCrossover" /> and <see cref="VSBXCrossover" />: both produce two
///     candidate children <c>c1</c>/<c>c2</c> from the two parents, then collapse them into one via
///     <c>use_child_gene_prob</c> and <c>uniform_crossover_prob</c>. The RNG draw order — for each
///     gene one <c>rand()</c> (use-child decision) then one <c>rand()</c> (swap decision), and a final
///     <c>rand()</c> to pick which candidate to return — is preserved exactly.
/// </summary>
internal static class SbxLike
{
    internal static double[] CollapseTwoChildren(
        double[] c1, double[] c2, double[] parent0, double[] parent1,
        NumPyRandom rng, double useChildGeneProb, double uniformCrossoverProb)
    {
        int n = c1.Length;
        var child1 = new double[n];
        var child2 = new double[n];
        for (int i = 0; i < n; i++)
        {
            if ((double)rng.rand() < useChildGeneProb)
            {
                if ((double)rng.rand() >= uniformCrossoverProb)
                {
                    child1[i] = c1[i];
                    child2[i] = c2[i];
                }
                else
                {
                    child1[i] = c2[i];
                    child2[i] = c1[i];
                }
            }
            else
            {
                if ((double)rng.rand() >= uniformCrossoverProb)
                {
                    child1[i] = parent0[i];
                    child2[i] = parent1[i];
                }
                else
                {
                    child1[i] = parent1[i];
                    child2[i] = parent0[i];
                }
            }
        }

        return (double)rng.rand() < 0.5 ? child1 : child2;
    }
}
```


<a id="undx-crossover"></a>

## src/OptunaSharp.Core/Samplers/Nsgaii/Crossovers/UNDXCrossover.cs

Crossover operators · 226 physical lines · SHA-256: `ebf38e6abf6534870145dee95f425363e939289b9d10fe2cb395d21e476d5301`

```csharp
using System;
using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Unimodal Normal Distribution Crossover used by <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_crossovers/_undx.py: UNDXCrossover</c> (experimental v3.0.0).
///
///     Generates a child from three parents using a multivariate normal distribution.
///
///     <para><b>[Misaligned] managed QR (n &gt; 1 only).</b> The orthonormal basis orthogonal to the
///     primary search line comes from <c>np.linalg.qr</c>, i.e. LAPACK <c>dgeqrf</c>/<c>dorgqr</c>.
///     OptunaSharp uses a managed Householder QR that reproduces numpy's basis vectors to ~1e-14 in
///     magnitude but <b>not their per-column signs</b> (LAPACK's column-sign bookkeeping is internal
///     and not portably reproducible — the same managed-vs-LAPACK class the plan excuses in §10).
///     Because <c>etas ~ N(0, σ²)</c> is symmetric, a sign-flipped basis is still a correct UNDX
///     sampler, but with two or more numerical parameters the child values differ from Optuna by
///     O(1), and that difference cascades through the crowding-distance/non-domination selection —
///     so the study trajectory diverges from Optuna's for such spaces. The Gaussian draw count/order
///     is identical regardless (the study stays in RNG lock-step within OptunaSharp, so it is
///     reproducible), and a single numerical parameter is bit-exact (the basis is unused).</para>
/// </summary>
public class UNDXCrossover : BaseCrossover
{
    private readonly double _sigmaXi;
    private readonly double? _sigmaEta;

    /// <param name="sigmaXi">Parametrizes the normal distribution from which <c>xi</c> is drawn.</param>
    /// <param name="sigmaEta">Parametrizes the normal distribution from which <c>etas</c> are drawn. Defaults to <c>0.35 / sqrt(len(search_space))</c>.</param>
    public UNDXCrossover(double sigmaXi = 0.5, double? sigmaEta = null)
    {
        _sigmaXi = sigmaXi;
        _sigmaEta = sigmaEta;
    }

    public override int NParents => 3;

    public override double[] Crossover(
        double[][] parentsParams, NumPyRandom rng, Study study, double[,] searchSpaceBounds)
    {
        int n = searchSpaceBounds.GetLength(0);
        var p0 = parentsParams[0];
        var p1 = parentsParams[1];

        var xp = new double[n];
        var d = new double[n];
        for (int i = 0; i < n; i++)
        {
            xp[i] = (p0[i] + p1[i]) / 2;  // Section 2 (2).
            d[i] = p0[i] - p1[i];         // Section 2 (3).
        }

        double sigmaEta = _sigmaEta ?? 0.35 / Math.Sqrt(n);

        // Draw order: etas (n gaussians) then xi (1 gaussian) — Optuna passes the *variance* as the
        // normal `scale` argument, replicated verbatim.
        var etas = rng.normal(0.0, Math.Pow(sigmaEta, 2.0), new Shape(n)).ToArray<double>();
        double xi = (double)rng.normal(0.0, Math.Pow(_sigmaXi, 2.0));

        var es = OrthonormalBasisVectorToPsl(parentsParams, n);  // Orthonormal basis orthogonal to psl.

        var child = new double[n];
        for (int i = 0; i < n; i++)
            child[i] = xp[i] + xi * d[i];  // one + two (Section 2 (5)).

        if (n > 1)  // When n == 1 there is no sub-search component.
        {
            double bigD = DistanceFromXToPsl(parentsParams);  // Section 2 (4).
            var three = new double[n];
            for (int i = 0; i < n - 1; i++)
            {
                for (int c = 0; c < n; c++)
                    three[c] += etas[i] * es[i][c];
            }
            for (int c = 0; c < n; c++)
                child[c] += three[c] * bigD;
        }

        return child;
    }

    /// <summary>Python <c>_normalized_x1_to_x2</c>: normalized vector from x1 to x2.</summary>
    private static double[] NormalizedX1ToX2(double[][] parentsParams)
    {
        int n = parentsParams[0].Length;
        var v12 = new double[n];
        double sumSq = 0.0;
        for (int i = 0; i < n; i++)
        {
            v12[i] = parentsParams[1][i] - parentsParams[0][i];
            sumSq += v12[i] * v12[i];
        }
        double m12 = Math.Sqrt(sumSq);
        double denom = Math.Max(m12, 1e-10);  // np.clip(m_12, 1e-10, None).
        var e12 = new double[n];
        for (int i = 0; i < n; i++)
            e12[i] = v12[i] / denom;
        return e12;
    }

    /// <summary>Python <c>_distance_from_x_to_psl</c>.</summary>
    private static double DistanceFromXToPsl(double[][] parentsParams)
    {
        int n = parentsParams[0].Length;
        var e12 = NormalizedX1ToX2(parentsParams);
        var v13 = new double[n];
        double dot = 0.0;
        for (int i = 0; i < n; i++)
        {
            v13[i] = parentsParams[2][i] - parentsParams[0][i];
            dot += v13[i] * e12[i];
        }
        double sumSq = 0.0;
        for (int i = 0; i < n; i++)
        {
            double v = v13[i] - dot * e12[i];  // orthogonal component.
            sumSq += v * v;
        }
        return Math.Sqrt(sumSq);
    }

    /// <summary>Python <c>_orthonormal_basis_vector_to_psl</c> → returns <c>Q.T[1:]</c> (n-1 rows of length n).</summary>
    private static double[][] OrthonormalBasisVectorToPsl(double[][] parentsParams, int n)
    {
        var e12 = NormalizedX1ToX2(parentsParams);
        bool anyNonZero = false;
        for (int i = 0; i < n; i++)
        {
            if (e12[i] != 0.0)
            {
                anyNonZero = true;
                break;
            }
        }

        // basis_matrix = identity(n); if e_12 != 0: basis_matrix[0] = e_12. Then qr(basis_matrix.T).
        var a = new double[n, n];
        for (int i = 0; i < n; i++)
            a[i, i] = 1.0;
        if (anyNonZero)
        {
            for (int c = 0; c < n; c++)
                a[0, c] = e12[c];
        }
        // basis_matrix.T
        var at = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                at[i, j] = a[j, i];

        var q = HouseholderQ(at, n);
        // return Q.T[1:] — rows 1..n-1 of Q^T = columns 1..n-1 of Q.
        var es = new double[n - 1][];
        for (int col = 1; col < n; col++)
        {
            var row = new double[n];
            for (int r = 0; r < n; r++)
                row[r] = q[r, col];
            es[col - 1] = row;
        }
        return es;
    }

    /// <summary>
    ///     Householder QR of the <paramref name="n" />×<paramref name="n" /> matrix
    ///     <paramref name="a" />; returns the orthogonal factor <c>Q</c> such that <c>A = Q R</c>,
    ///     using the LAPACK sign convention (<c>alpha = -sign(x0)·‖x‖</c>).
    /// </summary>
    private static double[,] HouseholderQ(double[,] a, int n)
    {
        var r = (double[,])a.Clone();
        var q = new double[n, n];
        for (int i = 0; i < n; i++)
            q[i, i] = 1.0;

        var v = new double[n];
        for (int k = 0; k < n; k++)
        {
            double normX = 0.0;
            for (int i = k; i < n; i++)
                normX += r[i, k] * r[i, k];
            normX = Math.Sqrt(normX);
            if (normX == 0.0)
                continue;

            double alpha = r[k, k] >= 0 ? -normX : normX;
            for (int i = 0; i < n; i++)
                v[i] = 0.0;
            v[k] = r[k, k] - alpha;
            for (int i = k + 1; i < n; i++)
                v[i] = r[i, k];

            double vNorm2 = 0.0;
            for (int i = k; i < n; i++)
                vNorm2 += v[i] * v[i];
            if (vNorm2 == 0.0)
                continue;
            double beta = 2.0 / vNorm2;

            // R = H_k R : for each column j, R[k:,j] -= beta * v * (v . R[k:,j]).
            for (int j = 0; j < n; j++)
            {
                double s = 0.0;
                for (int i = k; i < n; i++)
                    s += v[i] * r[i, j];
                s *= beta;
                for (int i = k; i < n; i++)
                    r[i, j] -= s * v[i];
            }

            // Q = Q H_k : for each row rr, Q[rr,k:] -= beta * (Q[rr,k:] . v) * v.
            for (int rr = 0; rr < n; rr++)
            {
                double s = 0.0;
                for (int i = k; i < n; i++)
                    s += q[rr, i] * v[i];
                s *= beta;
                for (int i = k; i < n; i++)
                    q[rr, i] -= s * v[i];
            }
        }

        return q;
    }
}
```


<a id="uniform-crossover"></a>

## src/OptunaSharp.Core/Samplers/Nsgaii/Crossovers/UniformCrossover.cs

Crossover operators · 41 physical lines · SHA-256: `f6e378e1ed319d415cba7b8a7549c9e8caf6ae6e1ac97741eb7a14462fd91c80`

```csharp
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Uniform Crossover operation used by <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_crossovers/_uniform.py: UniformCrossover</c>.
///
///     Select each parameter with equal probability from the two parent individuals.
/// </summary>
public class UniformCrossover : BaseCrossover
{
    private readonly double _swappingProb;

    /// <param name="swappingProb">Probability of swapping each parameter of the parents during crossover.</param>
    public UniformCrossover(double swappingProb = 0.5)
    {
        if (!(0.0 <= swappingProb && swappingProb <= 1.0))
            throw new ValueError("`swapping_prob` must be a float value within the range [0.0, 1.0].");
        _swappingProb = swappingProb;
    }

    public override int NParents => 2;

    public override double[] Crossover(
        double[][] parentsParams, NumPyRandom rng, Study study, double[,] searchSpaceBounds)
    {
        int nParams = searchSpaceBounds.GetLength(0);
        // masks = (rng.rand(n_params) >= swapping_prob).astype(int)
        // child_params = parents_params[masks, range(n_params)]
        var r = rng.rand(nParams).ToArray<double>();
        var child = new double[nParams];
        for (int j = 0; j < nParams; j++)
        {
            int mask = r[j] >= _swappingProb ? 1 : 0;
            child[j] = parentsParams[mask][j];
        }
        return child;
    }
}
```


<a id="vsbx-crossover"></a>

## src/OptunaSharp.Core/Samplers/Nsgaii/Crossovers/VSBXCrossover.cs

Crossover operators · 77 physical lines · SHA-256: `9b9ae6b622fc8466363d07beec17bbb3067dd21d00963eb741eb2a9562d3d8f8`

```csharp
using System;
using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Modified Simulated Binary Crossover (vSBX) used by <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_crossovers/_vsbx.py: VSBXCrossover</c> (experimental v3.0.0).
///
///     vSBX generates child individuals without excluding any region of the parameter space while
///     maintaining the properties of SBX.
/// </summary>
public class VSBXCrossover : BaseCrossover
{
    private readonly double? _eta;
    private readonly double _uniformCrossoverProb;
    private readonly double _useChildGeneProb;

    /// <param name="eta">Distribution index. Defaults to 2 for single-objective, 20 for multi-objective.</param>
    /// <param name="uniformCrossoverProb">Probability of uniform crossover between the two candidate children.</param>
    /// <param name="useChildGeneProb">Per-variable probability of using the generated child value over the parent value.</param>
    public VSBXCrossover(double? eta = null, double uniformCrossoverProb = 0.5, double useChildGeneProb = 0.5)
    {
        if (eta is not null && eta < 0.0)
            throw new ValueError("The value of `eta` must be greater than or equal to 0.0.");
        _eta = eta;

        if (uniformCrossoverProb < 0.0 || uniformCrossoverProb > 1.0)
            throw new ValueError("The value of `uniform_crossover_prob` must be in the range [0.0, 1.0].");
        if (useChildGeneProb <= 0.0 || useChildGeneProb > 1.0)
            throw new ValueError("The value of `use_child_gene_prob` must be in the range (0.0, 1.0].");
        _uniformCrossoverProb = uniformCrossoverProb;
        _useChildGeneProb = useChildGeneProb;
    }

    public override int NParents => 2;

    public override double[] Crossover(
        double[][] parentsParams, NumPyRandom rng, Study study, double[,] searchSpaceBounds)
    {
        int n = searchSpaceBounds.GetLength(0);
        double eta = _eta ?? (study.IsMultiObjective() ? 20.0 : 2.0);

        const double eps = 1e-10;
        var us = rng.rand(n).ToArray<double>();
        var beta1 = new double[n];
        var beta2 = new double[n];
        for (int j = 0; j < n; j++)
        {
            beta1[j] = Math.Pow(1 / Math.Max(2 * us[j], eps), 1 / (eta + 1));
            beta2[j] = Math.Pow(1 / Math.Max(2 * (1 - us[j]), eps), 1 / (eta + 1));
        }

        var p0 = parentsParams[0];
        var p1 = parentsParams[1];

        var c1 = new double[n];
        double u1 = (double)rng.rand();
        for (int j = 0; j < n; j++)
        {
            c1[j] = u1 <= 0.5
                ? 0.5 * ((1 + beta1[j]) * p0[j] + (1 - beta2[j]) * p1[j])
                : 0.5 * ((1 - beta1[j]) * p0[j] + (1 + beta2[j]) * p1[j]);
        }

        var c2 = new double[n];
        double u2 = (double)rng.rand();
        for (int j = 0; j < n; j++)
        {
            c2[j] = u2 <= 0.5
                ? 0.5 * ((3 - beta1[j]) * p0[j] - (1 - beta2[j]) * p1[j])
                : 0.5 * (-(1 - beta1[j]) * p0[j] + (3 - beta2[j]) * p1[j]);
        }

        return SbxLike.CollapseTwoChildren(c1, c2, p0, p1, rng, _useChildGeneProb, _uniformCrossoverProb);
    }
}
```


<a id="base-mutation"></a>

## src/OptunaSharp.Core/Samplers/Nsgaii/Mutations/BaseMutation.cs

Mutation operators · 26 physical lines · SHA-256: `1560c7a4fe02c68b62fc102edd874a94b06d3423c920549fd2d3ef014c56714f`

```csharp
using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Base class for mutations. Port of
///     <c>optuna/samplers/nsgaii/_mutations/_base.py: BaseMutation</c> (experimental v5.0.0).
///
///     A mutation operation is used by <see cref="NSGAIISampler" /> to mutate a numerical parameter
///     when creating a new individual.
/// </summary>
public abstract class BaseMutation
{
    public override string ToString() => GetType().Name;

    /// <summary>Mutate the given parameter.</summary>
    /// <param name="param">A parameter value in the transformed numerical search space.</param>
    /// <param name="rng">An instance of <c>numpy.random.RandomState</c>.</param>
    /// <param name="study">Target study object.</param>
    /// <param name="searchSpaceBounds">
    ///     A length-2 array <c>[lower, upper]</c> of numerical distribution bounds from the
    ///     transformed search space.
    /// </param>
    /// <returns>A mutated parameter value in the transformed numerical search space.</returns>
    public abstract double Mutation(double param, NumPyRandom rng, Study study, double[] searchSpaceBounds);
}
```


<a id="polynomial-mutation"></a>

## src/OptunaSharp.Core/Samplers/Nsgaii/Mutations/PolynomialMutation.cs

Mutation operators · 55 physical lines · SHA-256: `bba077a6296ed13540c861b5caa4851007214660b4707473092f2ab49d26e05a`

```csharp
using System;
using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Polynomial mutation operation used by <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_mutations/_polynomial.py: PolynomialMutation</c> (experimental v5.0.0).
///
///     Mutates a real-valued parameter according to the polynomial probability distribution,
///     following revision 1.1.6 of the original NSGA-II C implementation.
/// </summary>
public class PolynomialMutation : BaseMutation
{
    private readonly double _eta;

    /// <param name="eta">Distribution index. Larger values keep mutated values closer to the original.</param>
    public PolynomialMutation(double eta = 20.0)
    {
        if (eta < 0)
            throw new ValueError("`eta` must be a non-negative float value.");
        _eta = eta;
    }

    public override double Mutation(double param, NumPyRandom rng, Study study, double[] searchSpaceBounds)
    {
        double u = (double)rng.rand();
        double lb = searchSpaceBounds[0];
        double ub = searchSpaceBounds[1];
        double width = ub - lb;

        if (width <= 0.0)
            return param;

        double delta1 = (param - lb) / width;
        double delta2 = (ub - param) / width;
        double mutationPower = 1.0 / (_eta + 1.0);

        double deltaQ;
        if (u <= 0.5)
        {
            double xy = 1.0 - delta1;
            double value = 2.0 * u + (1.0 - 2.0 * u) * Math.Pow(xy, _eta + 1.0);
            deltaQ = Math.Pow(value, mutationPower) - 1.0;
        }
        else
        {
            double xy = 1.0 - delta2;
            double value = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * Math.Pow(xy, _eta + 1.0);
            deltaQ = 1.0 - Math.Pow(value, mutationPower);
        }

        return param + deltaQ * width;
    }
}
```


<a id="after-trial"></a>

## src/OptunaSharp.Core/Samplers/Nsgaii/NSGAIIAfterTrialStrategy.cs

Shared genetic machinery · 27 physical lines · SHA-256: `5ad8afc1e4c0f894bff605b9698f20d47b7105106415450ca1c946c447d4b877`

```csharp
using System.Collections.Generic;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     The default after-trial strategy of <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_after_trial_strategy.py: NSGAIIAfterTrialStrategy</c>.
///
///     Examines whether the trial result is valid in terms of constraints and stores the results in
///     the study's system attributes (only when a <c>constraints_func</c> is configured).
/// </summary>
public class NSGAIIAfterTrialStrategy
{
    private readonly System.Func<FrozenTrial, IEnumerable<double>>? _constraintsFunc;

    public NSGAIIAfterTrialStrategy(System.Func<FrozenTrial, IEnumerable<double>>? constraintsFunc = null)
    {
        _constraintsFunc = constraintsFunc;
    }

    /// <summary>Carry out the after-trial process of default NSGA-II (Python <c>__call__</c>).</summary>
    public void Call(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values = null)
    {
        if (_constraintsFunc is not null)
            SamplerHelpers.ProcessConstraintsAfterTrial(_constraintsFunc, study, trial, state);
    }
}
```


<a id="child"></a>

## src/OptunaSharp.Core/Samplers/Nsgaii/NSGAIIChildGenerationStrategy.cs

Shared genetic machinery · 138 physical lines · SHA-256: `a0e6c6a53ba09bf2a7e9bd21b172ed09e9ebd8e1625b435e663bec78bfca658f`

```csharp
using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     The default child-generation strategy of <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_child_generation_strategy.py: NSGAIIChildGenerationStrategy</c>.
///
///     With probability <c>crossover_prob</c> a child is produced by crossover of tournament-selected
///     parents; otherwise a single parent is copied. Each parameter is then mutated with probability
///     <c>mutation_prob</c> (defaulting to <c>1 / len(params)</c>): if a mutation object is set the
///     value is mutated in place, otherwise the parameter is dropped and later resampled independently.
///
///     <para><b>Conditional parameters (OptunaSharp extension).</b> With <see cref="ConditionalParams" />
///     declared, the search space may hold parameters a parent lacks (inactive for it). Crossover then
///     goes through <see cref="NsgaiiCrossover.PerformConditionalCrossover" />, the parent copy takes
///     what the parent carries, and every conditional parameter inactive for the child is pruned before
///     the mutation pass — so it is neither mutated nor counted in the default <c>1 / n_params</c>. The
///     mutation pass itself is unchanged. Without rules the strategy is byte-for-byte the Optuna port.</para>
/// </summary>
public class NSGAIIChildGenerationStrategy
{
    private readonly double _crossoverProb;
    private readonly double? _mutationProb;
    private readonly double _swappingProb;
    private readonly BaseMutation? _mutation;
    private readonly BaseCrossover _crossover;
    private readonly Func<FrozenTrial, IEnumerable<double>>? _constraintsFunc;
    private readonly LazyRandomState _rng;
    private readonly ConditionalParams? _conditionalParams;

    /// <param name="conditionalParams">
    ///     OptunaSharp extension: activation rules for parameters that only exist under some values of
    ///     other parameters. <c>null</c> (default) keeps the Optuna-parity behavior.
    /// </param>
    public NSGAIIChildGenerationStrategy(
        BaseMutation? mutation,
        double? mutationProb,
        BaseCrossover crossover,
        double crossoverProb,
        double swappingProb,
        Func<FrozenTrial, IEnumerable<double>>? constraintsFunc,
        LazyRandomState rng,
        ConditionalParams? conditionalParams = null)
    {
        if (!(mutationProb is null || (0.0 <= mutationProb && mutationProb <= 1.0)))
            throw new ValueError("`mutation_prob` must be None or a float value within the range [0.0, 1.0].");

        if (!(0.0 <= crossoverProb && crossoverProb <= 1.0))
            throw new ValueError("`crossover_prob` must be a float value within the range [0.0, 1.0].");

        if (!(0.0 <= swappingProb && swappingProb <= 1.0))
            throw new ValueError("`swapping_prob` must be a float value within the range [0.0, 1.0].");

        // In C# the crossover/mutation are strongly typed, so the isinstance guards are enforced
        // structurally; Optuna's ValueError for an invalid crossover has no analog to reach here.
        _crossoverProb = crossoverProb;
        _mutationProb = mutationProb;
        _swappingProb = swappingProb;
        _mutation = mutation;
        _crossover = crossover;
        _constraintsFunc = constraintsFunc;
        _rng = rng;
        _conditionalParams = conditionalParams;
    }

    /// <summary>Generate a child parameter dict from the given parent population (Python <c>__call__</c>).</summary>
    public OrderedDict<string, object?> Generate(
        Study study,
        OrderedDict<string, BaseDistribution> searchSpace,
        List<FrozenTrial> parentPopulation)
    {
        Func<FrozenTrial, FrozenTrial, IReadOnlyList<StudyDirection>, bool> dominates =
            ConstrainedOptimization.IsConstrainedOptimization(parentPopulation)
                ? NsgaiiConstraintsEvaluation.ConstrainedDominates
                : MultiObjective.Dominates;

        // We choose a child based on the specified crossover method.
        OrderedDict<string, object?> childParams;
        if ((double)_rng.Rng.rand() < _crossoverProb)
        {
            childParams = _conditionalParams is null
                ? NsgaiiCrossover.PerformCrossover(
                    _crossover, study, parentPopulation, searchSpace, _rng.Rng, _swappingProb, dominates)
                : NsgaiiCrossover.PerformConditionalCrossover(
                    _crossover, study, parentPopulation, searchSpace, _rng.Rng, _swappingProb, dominates,
                    _conditionalParams);
        }
        else
        {
            int parentPopulationSize = parentPopulation.Count;
            var parentParams = parentPopulation[(int)_rng.Rng.randint(0, parentPopulationSize)].ParamsDict;
            childParams = new OrderedDict<string, object?>();
            if (_conditionalParams is null)
            {
                foreach (var name in searchSpace.Keys)
                    childParams[name] = parentParams[name];
            }
            else
            {
                // The parent lacks exactly the conditional parameters that were inactive for it; copy what
                // it carries, then prune in case the declared rules and the parent's own branches disagree.
                foreach (var name in searchSpace.Keys)
                {
                    if (parentParams.TryGetValue(name, out var value))
                        childParams[name] = value;
                }
                _conditionalParams.Prune(childParams, searchSpace);
            }
        }

        int nParams = childParams.Count;
        double mutationProb = _mutationProb ?? 1.0 / Math.Max(1.0, nParams);

        var @params = new OrderedDict<string, object?>();
        foreach (var paramName in childParams.Keys)
        {
            if ((double)_rng.Rng.rand() >= mutationProb)
            {
                @params[paramName] = childParams[paramName];
            }
            else if (_mutation is not null)
            {
                var mutationValue = NsgaiiMutation.PerformMutation(
                    _mutation, _rng.Rng, study, searchSpace[paramName], childParams[paramName]);
                if (mutationValue is not null)
                    @params[paramName] = mutationValue;
            }
        }
        return @params;
    }
}
```


<a id="sorting"></a>

## src/OptunaSharp.Core/Samplers/Nsgaii/NSGAIIElitePopulationSelectionStrategy.cs

Shared genetic machinery · 187 physical lines · SHA-256: `b8d8ffe6567104d530ed8bed2b4911b00865008e52abab27feefc878ba8a7823`

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     The default elite-population selection strategy of <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_elite_population_selection_strategy.py</c>.
///
///     Ranks the combined population by (constrained) non-domination, then fills the elite
///     population rank by rank; the rank that overflows <c>population_size</c> is truncated by
///     crowding distance (descending). The resulting order is load-bearing — it seeds the parent
///     population that later generations index into with <c>rng.choice</c>.
/// </summary>
public class NSGAIIElitePopulationSelectionStrategy
{
    /// <summary>Python <c>&lt;</c> on floats: a &lt; b is the only ordering primitive; -0.0 == 0.0 and NaN compares equal (kept stable).</summary>
    private sealed class PyLessComparer : IComparer<double>
    {
        internal static readonly PyLessComparer Instance = new();

        public int Compare(double a, double b) => a < b ? -1 : (b < a ? 1 : 0);
    }

    private readonly int _populationSize;
    private readonly Func<FrozenTrial, IEnumerable<double>>? _constraintsFunc;

    public NSGAIIElitePopulationSelectionStrategy(
        int populationSize, Func<FrozenTrial, IEnumerable<double>>? constraintsFunc = null)
    {
        if (populationSize < 2)
            throw new ValueError("`population_size` must be greater than or equal to 2.");

        _populationSize = populationSize;
        _constraintsFunc = constraintsFunc;
    }

    /// <summary>Select the elite population from the given trials (Python <c>__call__</c>).</summary>
    public List<FrozenTrial> Select(Study study, List<FrozenTrial> population)
    {
        bool isConstrained = ConstrainedOptimization.IsConstrainedOptimization(population);
        NsgaiiConstraintsEvaluation.ValidateConstraints(population, isConstrained);
        var populationPerRank = RankPopulation(population, study.Directions, isConstrained);

        var elitePopulation = new List<FrozenTrial>();
        foreach (var individuals in populationPerRank)
        {
            if (elitePopulation.Count + individuals.Count < _populationSize)
            {
                elitePopulation.AddRange(individuals);
            }
            else
            {
                int n = _populationSize - elitePopulation.Count;
                CrowdingDistanceSort(individuals);
                elitePopulation.AddRange(individuals.Take(n));
                break;
            }
        }

        return elitePopulation;
    }

    /// <summary>
    ///     Python <c>_rank_population</c>. Module-level in Optuna's
    ///     <c>nsgaii/_elite_population_selection_strategy.py</c> and imported by the NSGA-III strategy,
    ///     so it is <c>internal static</c> here and shared with
    ///     <see cref="OptunaSharp.Samplers.Nsgaiii.NSGAIIIElitePopulationSelectionStrategy" />.
    /// </summary>
    internal static List<List<FrozenTrial>> RankPopulation(
        IReadOnlyList<FrozenTrial> population, IReadOnlyList<StudyDirection> directions, bool isConstrained)
    {
        if (population.Count == 0)
            return new List<List<FrozenTrial>>();

        var objectiveValues = new double[population.Count][];
        for (int i = 0; i < population.Count; i++)
        {
            var values = population[i].Values!;
            var row = new double[values.Count];
            for (int k = 0; k < values.Count; k++)
            {
                double sign = directions[k] == StudyDirection.Maximize ? -1.0 : 1.0;
                row[k] = values[k] * sign;
            }
            objectiveValues[i] = row;
        }

        double[]? penalty = isConstrained ? NsgaiiConstraintsEvaluation.EvaluatePenalty(population) : null;
        var dominationRanks = MultiObjective.FastNonDominationRank(objectiveValues, penalty: penalty);

        int maxRank = dominationRanks.Max();
        var populationPerRank = new List<List<FrozenTrial>>(maxRank + 1);
        for (int r = 0; r <= maxRank; r++)
            populationPerRank.Add(new List<FrozenTrial>());
        for (int i = 0; i < population.Count; i++)
        {
            int rank = dominationRanks[i];
            if (rank == -1)
                continue;
            populationPerRank[rank].Add(population[i]);
        }

        return populationPerRank;
    }

    /// <summary>Python <c>_crowding_distance_sort</c>: sort by crowding distance descending (in place).</summary>
    private static void CrowdingDistanceSort(List<FrozenTrial> population)
    {
        var manhattanDistances = CalcCrowdingDistance(population);
        StableSortInPlace(population, x => manhattanDistances.GetValueOrDefault(x.Number, 0.0));
        population.Reverse();
    }

    /// <summary>Python <c>_calc_crowding_distance</c>.</summary>
    private static Dictionary<int, double> CalcCrowdingDistance(List<FrozenTrial> population)
    {
        var manhattanDistances = new Dictionary<int, double>();
        if (population.Count == 0)
            return manhattanDistances;

        int nObjectives = population[0].Values!.Count;
        for (int i = 0; i < nObjectives; i++)
        {
            int obj = i;
            StableSortInPlace(population, x => x.Values![obj]);

            // If all trials have the same value in this dimension, it makes no difference.
            if (population[0].Values![i] == population[^1].Values![i])
                continue;

            // vs = [-inf] + values + [inf].
            int m = population.Count;
            var vs = new double[m + 2];
            vs[0] = double.NegativeInfinity;
            for (int j = 0; j < m; j++)
                vs[j + 1] = population[j].Values![i];
            vs[m + 1] = double.PositiveInfinity;

            // Smallest / largest finite value: v_min = next(x for x in vs if x != -inf);
            // v_max = next(x for x in reversed(vs) if x != inf).
            double vMin = double.NegativeInfinity;
            for (int k = 0; k < vs.Length; k++)
            {
                if (vs[k] != double.NegativeInfinity)
                {
                    vMin = vs[k];
                    break;
                }
            }
            double vMax = double.PositiveInfinity;
            for (int k = vs.Length - 1; k >= 0; k--)
            {
                if (vs[k] != double.PositiveInfinity)
                {
                    vMax = vs[k];
                    break;
                }
            }

            double width = vMax - vMin;
            if (width <= 0)  // width == 0 or width == -inf.
                width = 1.0;

            for (int j = 0; j < m; j++)
            {
                // inf - inf and (-inf) - (-inf) is considered to be zero.
                double gap = vs[j] == vs[j + 2] ? 0.0 : vs[j + 2] - vs[j];
                int number = population[j].Number;
                manhattanDistances[number] = manhattanDistances.GetValueOrDefault(number, 0.0) + gap / width;
            }
        }

        return manhattanDistances;
    }

    /// <summary>Python <c>list.sort(key=...)</c> — a stable ascending sort under Python's <c>&lt;</c>, in place.</summary>
    private static void StableSortInPlace(List<FrozenTrial> population, Func<FrozenTrial, double> key)
    {
        var sorted = population.OrderBy(key, PyLessComparer.Instance).ToList();
        population.Clear();
        population.AddRange(sorted);
    }
}
```


<a id="constraint-evaluation"></a>

## src/OptunaSharp.Core/Samplers/Nsgaii/NsgaiiConstraintsEvaluation.cs

Shared genetic machinery · 73 physical lines · SHA-256: `6165216b22521dfdca27a7594bc4bcda8796df51bb44372fdad04e456286c792`

```csharp
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Port of <c>optuna/samplers/nsgaii/_constraints_evaluation.py</c>. Deterministic
///     constraint-domination logic (no RNG).
/// </summary>
internal static class NsgaiiConstraintsEvaluation
{
    /// <summary>
    ///     Python <c>_constrained_dominates</c>. A trial x constrained-dominates y iff:
    ///     (1) x is feasible and y is not; (2) both infeasible but x has smaller overall violation;
    ///     or (3) both feasible and x dominates y.
    /// </summary>
    internal static bool ConstrainedDominates(
        FrozenTrial trial0, FrozenTrial trial1, IReadOnlyList<StudyDirection> directions)
    {
        var constraints0 = trial0.Constraints;
        var constraints1 = trial1.Constraints;

        if (trial0.State != TrialState.Complete)
            return false;

        if (trial1.State != TrialState.Complete)
            return true;

        bool satisfyConstraints0 = constraints0.Values.All(v => v <= 0);
        bool satisfyConstraints1 = constraints1.Values.All(v => v <= 0);

        if (satisfyConstraints0 && satisfyConstraints1)
            // Both trials satisfy the constraints.
            return MultiObjective.Dominates(trial0, trial1, directions);

        if (satisfyConstraints0)
            // trial0 satisfies the constraints, but trial1 violates them.
            return true;

        if (satisfyConstraints1)
            // trial1 satisfies the constraints, but trial0 violates them.
            return false;

        // Both trials violate the constraints.
        double violation0 = constraints0.Values.Where(v => v > 0).Sum();
        double violation1 = constraints1.Values.Where(v => v > 0).Sum();
        return violation0 < violation1;
    }

    /// <summary>Python <c>_evaluate_penalty</c>: per-trial sum of positive constraint values.</summary>
    internal static double[] EvaluatePenalty(IReadOnlyList<FrozenTrial> population)
    {
        var penalty = new double[population.Count];
        for (int i = 0; i < population.Count; i++)
            penalty[i] = population[i].Constraints.Values.Where(v => v > 0).Sum();
        return penalty;
    }

    /// <summary>Python <c>_validate_constraints</c>.</summary>
    internal static void ValidateConstraints(IReadOnlyList<FrozenTrial> population, bool isConstrained)
    {
        if (!isConstrained)
            return;

        foreach (var trial in population)
        {
            if (trial.Constraints.Values.Any(double.IsNaN))
                throw new ValueError("NaN is not acceptable as constraint value.");
        }
    }
}
```


<a id="crossover"></a>

## src/OptunaSharp.Core/Samplers/Nsgaii/NsgaiiCrossover.cs

Shared genetic machinery · 294 physical lines · SHA-256: `9f76510aae39a49e69999bcb2dc8d4ec46559a9d31b08a72f7aeda9d6fdd1dc4`

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Port of <c>optuna/samplers/nsgaii/_crossover.py</c> — the crossover dispatcher that splits
///     the search space into numerical (fed to the <see cref="BaseCrossover" />) and categorical
///     (uniform crossover) parts, selects parents by binary tournament, and re-draws until the child
///     lands inside the search space. RNG draw order matches Optuna exactly.
/// </summary>
internal static class NsgaiiCrossover
{
    internal static OrderedDict<string, object?> PerformCrossover(
        BaseCrossover crossover,
        Study study,
        IReadOnlyList<FrozenTrial> parentPopulation,
        OrderedDict<string, BaseDistribution> searchSpace,
        NumPyRandom rng,
        double swappingProb,
        Func<FrozenTrial, FrozenTrial, IReadOnlyList<StudyDirection>, bool> dominates)
    {
        var numericalSearchSpace = new OrderedDict<string, BaseDistribution>();
        var categoricalSearchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (key, value) in searchSpace)
        {
            if (value is FloatDistribution or IntDistribution)
                numericalSearchSpace[key] = value;
            else
                categoricalSearchSpace[key] = value;
        }

        SearchSpaceTransform? numericalTransform = null;
        if (numericalSearchSpace.Count != 0)
            numericalTransform = new SearchSpaceTransform(numericalSearchSpace);

        while (true)  // Repeat while parameters lie outside search space boundaries.
        {
            var parents = SelectParents(crossover, study, parentPopulation, rng, dominates);
            var childParams = TryCrossover(
                parents, crossover, study, rng, swappingProb,
                categoricalSearchSpace, numericalSearchSpace, numericalTransform);

            if (IsContained(childParams, searchSpace))
                return childParams;
        }
    }

    /// <summary>
    ///     The <see cref="ConditionalParams" /> variant of <see cref="PerformCrossover" /> — an OptunaSharp
    ///     extension with no Optuna analog, reached only when rules are declared, so the parity path above
    ///     is untouched. Parents may lack conditional parameters: the operators run over what the selected
    ///     parents share, the rest is inherited from a carrier, and parameters inactive for the child are
    ///     pruned before the containment check (so an inactive value can neither fail it nor reach the
    ///     trial). Parent selection is the same binary tournament as the parity path.
    /// </summary>
    internal static OrderedDict<string, object?> PerformConditionalCrossover(
        BaseCrossover crossover,
        Study study,
        IReadOnlyList<FrozenTrial> parentPopulation,
        OrderedDict<string, BaseDistribution> searchSpace,
        NumPyRandom rng,
        double swappingProb,
        Func<FrozenTrial, FrozenTrial, IReadOnlyList<StudyDirection>, bool> dominates,
        ConditionalParams conditionalParams)
    {
        var numericalSearchSpace = new OrderedDict<string, BaseDistribution>();
        var categoricalSearchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (key, value) in searchSpace)
        {
            if (value is FloatDistribution or IntDistribution)
                numericalSearchSpace[key] = value;
            else
                categoricalSearchSpace[key] = value;
        }

        while (true)  // Repeat while parameters lie outside search space boundaries.
        {
            var parents = SelectParents(crossover, study, parentPopulation, rng, dominates);
            var childParams = TryConditionalCrossover(
                parents, crossover, study, rng, swappingProb, categoricalSearchSpace, numericalSearchSpace);
            conditionalParams.Prune(childParams, searchSpace);

            if (IsContained(childParams, searchSpace))
                return childParams;
        }
    }

    /// <summary>
    ///     <see cref="TryCrossover" /> for parents that may lack conditional parameters. Categorical:
    ///     uniform crossover between the first and last parent over the keys both carry (one mask draw per
    ///     shared key, in search-space order — the same draws as the parity path when nothing is missing);
    ///     a key only one of them carries is inherited from it. Numerical: the crossover operator sees the
    ///     keys every parent carries (a well-formed parents × dims matrix in the transformed space, with
    ///     the transform built per parent set); a key only some parents carry is inherited from one of
    ///     them, drawn uniformly. A key no parent carries stays absent and is sampled independently later
    ///     if the objective asks for it. Child keys keep search-space order, as in the parity path.
    /// </summary>
    private static OrderedDict<string, object?> TryConditionalCrossover(
        List<FrozenTrial> parents,
        BaseCrossover crossover,
        Study study,
        NumPyRandom rng,
        double swappingProb,
        OrderedDict<string, BaseDistribution> categoricalSearchSpace,
        OrderedDict<string, BaseDistribution> numericalSearchSpace)
    {
        var childParams = new OrderedDict<string, object?>();

        if (categoricalSearchSpace.Count > 0)
        {
            var first = parents[0].ParamsDict;
            var last = parents[^1].ParamsDict;
            var shared = categoricalSearchSpace.Keys.Where(k => first.ContainsKey(k) && last.ContainsKey(k)).ToList();
            var crossed = shared.Count == 0
                ? Array.Empty<object?>()
                : InlinedCategoricalUniformCrossover(
                    shared.Select(k => first[k]).ToArray(), shared.Select(k => last[k]).ToArray(),
                    rng, swappingProb, shared.Count);

            int j = 0;
            foreach (var key in categoricalSearchSpace.Keys)
            {
                bool inFirst = first.ContainsKey(key), inLast = last.ContainsKey(key);
                if (inFirst && inLast)
                    childParams[key] = crossed[j++];
                else if (inFirst)
                    childParams[key] = first[key];
                else if (inLast)
                    childParams[key] = last[key];
            }
        }

        if (numericalSearchSpace.Count == 0)
            return childParams;

        var common = new OrderedDict<string, BaseDistribution>();
        var partialCarriers = new Dictionary<string, List<FrozenTrial>>(StringComparer.Ordinal);
        foreach (var (key, distribution) in numericalSearchSpace)
        {
            var carriers = parents.Where(p => p.ParamsDict.ContainsKey(key)).ToList();
            if (carriers.Count == parents.Count)
                common[key] = distribution;
            else if (carriers.Count > 0)
                partialCarriers[key] = carriers;
        }

        var crossedNumerical = new OrderedDict<string, object?>();
        if (common.Count > 0)
        {
            var transform = new SearchSpaceTransform(common);
            var parentsNumerical = new double[parents.Count][];
            for (int i = 0; i < parents.Count; i++)
            {
                var row = new OrderedDict<string, object?>();
                foreach (var key in common.Keys)
                    row[key] = parents[i].ParamsDict[key];
                parentsNumerical[i] = transform.Transform(row).ToArray<double>();
            }

            var childNumericalArray = crossover.Crossover(parentsNumerical, rng, study, transform.RawBounds);
            crossedNumerical = transform.Untransform(np.array(childNumericalArray));
        }

        foreach (var key in numericalSearchSpace.Keys)
        {
            if (crossedNumerical.TryGetValue(key, out var value))
            {
                childParams[key] = value;
            }
            else if (partialCarriers.TryGetValue(key, out var carriers))
            {
                var carrier = carriers.Count == 1 ? carriers[0] : carriers[(int)rng.randint(0, carriers.Count)];
                childParams[key] = carrier.ParamsDict[key];
            }
        }

        return childParams;
    }

    private static OrderedDict<string, object?> TryCrossover(
        List<FrozenTrial> parents,
        BaseCrossover crossover,
        Study study,
        NumPyRandom rng,
        double swappingProb,
        OrderedDict<string, BaseDistribution> categoricalSearchSpace,
        OrderedDict<string, BaseDistribution> numericalSearchSpace,
        SearchSpaceTransform? numericalTransform)
    {
        var childParams = new OrderedDict<string, object?>();

        if (categoricalSearchSpace.Count > 0)
        {
            var catKeys = categoricalSearchSpace.Keys.ToList();
            // parents_categorical_params = [[parents[0][p]...], [parents[-1][p]...]]  (2 rows).
            var parent0Cat = catKeys.Select(p => parents[0].ParamsDict[p]).ToArray();
            var parentLastCat = catKeys.Select(p => parents[^1].ParamsDict[p]).ToArray();

            var childCategorical = InlinedCategoricalUniformCrossover(
                parent0Cat, parentLastCat, rng, swappingProb, catKeys.Count);
            for (int j = 0; j < catKeys.Count; j++)
                childParams[catKeys[j]] = childCategorical[j];
        }

        if (numericalTransform is null)
            return childParams;

        // The following is applied only for numerical parameters.
        var numKeys = numericalSearchSpace.Keys.ToList();
        var parentsNumerical = new double[parents.Count][];
        for (int i = 0; i < parents.Count; i++)
        {
            var row = new OrderedDict<string, object?>();
            foreach (var k in numKeys)
                row[k] = parents[i].ParamsDict[k];
            parentsNumerical[i] = numericalTransform.Transform(row).ToArray<double>();
        }

        var childNumericalArray = crossover.Crossover(
            parentsNumerical, rng, study, numericalTransform.RawBounds);
        var childNumericalParams = numericalTransform.Untransform(np.array(childNumericalArray));
        foreach (var (k, v) in childNumericalParams)
            childParams[k] = v;

        return childParams;
    }

    private static List<FrozenTrial> SelectParents(
        BaseCrossover crossover,
        Study study,
        IReadOnlyList<FrozenTrial> parentPopulation,
        NumPyRandom rng,
        Func<FrozenTrial, FrozenTrial, IReadOnlyList<StudyDirection>, bool> dominates)
    {
        var parents = new List<FrozenTrial>();
        var selectedIds = new HashSet<int>();
        for (int i = 0; i < crossover.NParents; i++)
        {
            var remaining = parentPopulation.Where(t => !selectedIds.Contains(t.TrialId)).ToList();
            var parent = SelectParent(study, remaining, rng, dominates);
            parents.Add(parent);
            selectedIds.Add(parent.TrialId);
        }
        return parents;
    }

    private static FrozenTrial SelectParent(
        Study study,
        IReadOnlyList<FrozenTrial> parentPopulation,
        NumPyRandom rng,
        Func<FrozenTrial, FrozenTrial, IReadOnlyList<StudyDirection>, bool> dominates)
    {
        int populationSize = parentPopulation.Count;
        var candidate0 = parentPopulation[(int)rng.randint(0, populationSize)];
        var candidate1 = parentPopulation[(int)rng.randint(0, populationSize)];

        // TODO(ohta): Consider crowding distance.
        return dominates(candidate0, candidate1, study.Directions) ? candidate0 : candidate1;
    }

    private static bool IsContained(OrderedDict<string, object?> @params, OrderedDict<string, BaseDistribution> searchSpace)
    {
        foreach (var (paramName, param) in @params)
        {
            var distribution = searchSpace[paramName];
            if (!distribution.Contains(distribution.ToInternalRepr(param)))
                return false;
        }
        return true;
    }

    /// <summary>
    ///     Python <c>_inlined_categorical_uniform_crossover</c>. Categorical params are passed
    ///     untransformed, so uniform crossover is done here directly against the two parents'
    ///     external values.
    /// </summary>
    private static object?[] InlinedCategoricalUniformCrossover(
        object?[] parent0, object?[] parentLast, NumPyRandom rng, double swappingProb, int nCategorical)
    {
        // masks = (rng.rand(n) >= swapping_prob).astype(int); child = parent_params[masks, range(n)].
        var r = rng.rand(nCategorical).ToArray<double>();
        var child = new object?[nCategorical];
        for (int j = 0; j < nCategorical; j++)
            child[j] = r[j] >= swappingProb ? parentLast[j] : parent0[j];
        return child;
    }
}
```


<a id="mutation"></a>

## src/OptunaSharp.Core/Samplers/Nsgaii/NsgaiiMutation.cs

Shared genetic machinery · 31 physical lines · SHA-256: `9c8ead1f8ae1b940505c620a3ea158970f41541d0a3ded12d58c96444830387a`

```csharp
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Port of <c>optuna/samplers/nsgaii/_mutation.py: perform_mutation</c>. Applies a mutation to a
///     single numerical parameter in the transformed space and clips it back into range; returns
///     <c>null</c> for categorical parameters (which are then resampled independently).
/// </summary>
internal static class NsgaiiMutation
{
    internal static object? PerformMutation(
        BaseMutation mutation, NumPyRandom rng, Study study, BaseDistribution distribution, object? value)
    {
        if (distribution is not (FloatDistribution or IntDistribution))
            return null;

        var transform = new SearchSpaceTransform(new OrderedDict<string, BaseDistribution> { [""] = distribution });
        double transValue = transform.Transform(new OrderedDict<string, object?> { [""] = value }).ToArray<double>()[0];

        var bounds = transform.RawBounds;
        double transMutatedValue = mutation.Mutation(
            transValue, rng, study, new[] { bounds[0, 0], bounds[0, 1] });
        transMutatedValue = MathUtil.Clip(transMutatedValue, bounds[0, 0], bounds[0, 1]);

        return transform.Untransform(np.array(new[] { transMutatedValue }))[""];
    }
}
```


<a id="distribution-base-distribution"></a>

## src/OptunaSharp.Core/Distributions/BaseDistribution.cs

Parameter distributions · 84 physical lines · SHA-256: `dd5406b8dbd0719c1620e05b9499706a5794dc2069dd80ed48c445f8f2b4efc2`

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Distributions;

/// <summary>
///     Base class for distributions. Port of <c>optuna/distributions.py: BaseDistribution</c>.
///
///     Note that distribution classes are not supposed to be called by library users.
///     They are used by <see cref="Trial" /> and samplers internally.
/// </summary>
public abstract class BaseDistribution : IEquatable<BaseDistribution>
{
    /// <summary>
    ///     Convert internal representation of a parameter value into external representation.
    /// </summary>
    /// <param name="paramValueInInternalRepr">Optuna's internal representation of a parameter value.</param>
    /// <returns>Optuna's external representation of a parameter value.</returns>
    public virtual object? ToExternalRepr(double paramValueInInternalRepr)
        => paramValueInInternalRepr;

    /// <summary>
    ///     Convert external representation of a parameter value into internal representation.
    /// </summary>
    /// <param name="paramValueInExternalRepr">Optuna's external representation of a parameter value.</param>
    /// <returns>Optuna's internal representation of a parameter value.</returns>
    public abstract double ToInternalRepr(object? paramValueInExternalRepr);

    /// <summary>
    ///     Test whether the range of this distribution contains just a single value.
    /// </summary>
    public abstract bool Single();

    /// <summary>
    ///     Test if a parameter value is contained in the range of this distribution.
    ///     (Python: <c>_contains</c>.)
    /// </summary>
    internal abstract bool Contains(double paramValueInInternalRepr);

    /// <summary>
    ///     The attribute view serialized to JSON and shown in <c>repr</c> (Python: <c>_asdict</c>).
    ///     Key insertion order matches the Python class's <c>__dict__</c> exactly — it is
    ///     observable in <c>distribution_to_json</c> output.
    /// </summary>
    internal abstract OrderedDict<string, object?> AsDict();

    /// <summary>Python <c>__eq__</c>: same concrete type and equal attributes.</summary>
    public abstract bool Equals(BaseDistribution? other);

    public override bool Equals(object? obj) => obj is BaseDistribution other && Equals(other);

    public abstract override int GetHashCode();

    public static bool operator ==(BaseDistribution? left, BaseDistribution? right)
        => left is null ? right is null : left.Equals(right);

    public static bool operator !=(BaseDistribution? left, BaseDistribution? right) => !(left == right);

    /// <summary>
    ///     Python <c>__repr__</c>: <c>ClassName(key1=value1, key2=value2)</c> with keys sorted.
    /// </summary>
    public override string ToString()
    {
        var sb = new StringBuilder(GetType().Name).Append('(');
        bool first = true;
        foreach (var kv in AsDict().OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            if (!first)
                sb.Append(", ");
            sb.Append(kv.Key).Append('=').Append(ReprAttr(kv.Value));
            first = false;
        }
        return sb.Append(')').ToString();
    }

    private static string ReprAttr(object? value)
        => value is IReadOnlyList<CategoricalChoice> choices
            ? PythonRepr.TupleRepr(choices.Select(c => c.Value))
            : PythonRepr.Repr(value);
}
```


<a id="distribution-categorical-choice"></a>

## src/OptunaSharp.Core/Distributions/CategoricalChoice.cs

Parameter distributions · 154 physical lines · SHA-256: `658492283aaab3ef846ef104bd244497f220df58f9866aa91e86432f02613a92`

```csharp
using System;
using System.Globalization;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Distributions;

/// <summary>The kind of value a <see cref="CategoricalChoice" /> holds.</summary>
public enum CategoricalChoiceKind
{
    /// <summary>Python <c>None</c>.</summary>
    None = 0,
    Bool,
    Int,
    Float,
    String,
}

/// <summary>
///     A categorical parameter value — Python's
///     <c>CategoricalChoiceType = Union[None, bool, int, float, str]</c> as a value type.
///
///     Equality implements <b>Python <c>==</c> semantics</b>, which is a plan §2 ground rule:
///     <c>True == 1</c>, <c>1.0 == 1</c>, <c>nan != nan</c>, and cross-type number/string
///     comparisons are <c>false</c>. This is the one place C# object equality would silently
///     diverge from Optuna, so it is pinned by dedicated tests.
/// </summary>
public readonly struct CategoricalChoice : IEquatable<CategoricalChoice>
{
    private readonly object? _value;

    private CategoricalChoice(object? normalizedValue)
    {
        _value = normalizedValue;
    }

    /// <summary>Python <c>None</c>.</summary>
    public static CategoricalChoice None => default;

    /// <summary>
    ///     The underlying value: <c>null</c>, <see cref="bool" />, <see cref="long" />,
    ///     <see cref="double" /> or <see cref="string" />.
    /// </summary>
    public object? Value => _value;

    public CategoricalChoiceKind Kind => _value switch
    {
        null => CategoricalChoiceKind.None,
        bool => CategoricalChoiceKind.Bool,
        long => CategoricalChoiceKind.Int,
        double => CategoricalChoiceKind.Float,
        _ => CategoricalChoiceKind.String,
    };

    /// <summary>Wraps an arbitrary value, normalizing numerics (all integrals → <see cref="long" />, <c>float</c> → <see cref="double" />).</summary>
    public static CategoricalChoice From(object? value) => value switch
    {
        null => None,
        CategoricalChoice choice => choice,
        bool b => new CategoricalChoice(b),
        string s => new CategoricalChoice(s),
        double d => new CategoricalChoice(d),
        float f => new CategoricalChoice((double)f),
        sbyte or byte or short or ushort or int or uint or long => new CategoricalChoice(Convert.ToInt64(value, CultureInfo.InvariantCulture)),
        ulong ul => ul <= long.MaxValue
            ? new CategoricalChoice((long)ul)
            : throw new TypeError($"Choices for a categorical distribution should be None, bool, int, float or str, but got {value} of type {value.GetType().Name}."),
        _ => throw new TypeError($"Choices for a categorical distribution should be None, bool, int, float or str, but got {value} of type {value.GetType().Name}."),
    };

    public static implicit operator CategoricalChoice(bool value) => new(value);

    public static implicit operator CategoricalChoice(long value) => new(value);

    public static implicit operator CategoricalChoice(int value) => new((long)value);

    public static implicit operator CategoricalChoice(double value) => new(value);

    public static implicit operator CategoricalChoice(string? value) => value is null ? None : new(value);

    public bool IsNone => _value is null;

    /// <summary>True when the value is a real number whose float value is NaN (Python <c>isinstance(v, Real) and math.isnan(float(v))</c>).</summary>
    public bool IsNaN => _value is double d && double.IsNaN(d);

    /// <summary>The value as a Python <c>float(v)</c> would return it. Throws for None/str.</summary>
    internal double AsDouble() => _value switch
    {
        bool b => b ? 1.0 : 0.0,
        long l => l,
        double d => d,
        _ => throw new TypeError($"float() argument must be a real number, not '{(_value is null ? "NoneType" : "str")}'"),
    };

    /// <summary>Python <c>==</c>.</summary>
    public bool Equals(CategoricalChoice other)
    {
        var a = _value;
        var b = other._value;
        if (a is null || b is null)
            return a is null && b is null;
        if (a is string sa)
            return b is string sb && string.Equals(sa, sb, StringComparison.Ordinal);
        if (b is string)
            return false;
        // Both numeric (bool counts as 0/1 — Python bool is an int subclass). NaN != NaN via IEEE.
        if (a is long la && b is long lb)
            return la == lb;
        return NumericValue(a) == NumericValue(b);
    }

    private static double NumericValue(object value) => value switch
    {
        bool b => b ? 1.0 : 0.0,
        long l => l,
        double d => d,
        _ => throw new InvalidOperationException(),
    };

    public override bool Equals(object? obj) => obj switch
    {
        CategoricalChoice choice => Equals(choice),
        null => IsNone,
        _ => Equals(From(obj)),
    };

    public override int GetHashCode()
    {
        // Values equal under Python == must hash alike: hash all numerics by their double image
        // (Python: hash(True) == hash(1) == hash(1.0)).
        return _value switch
        {
            null => 0,
            string s => s.GetHashCode(),
            _ => NumericValue(_value).GetHashCode(),
        };
    }

    public static bool operator ==(CategoricalChoice left, CategoricalChoice right) => left.Equals(right);

    public static bool operator !=(CategoricalChoice left, CategoricalChoice right) => !left.Equals(right);

    /// <summary>
    ///     Python <c>_categorical_choice_equal</c>: <c>==</c> except that two NaNs compare equal.
    /// </summary>
    internal static bool ChoicesEqual(CategoricalChoice value1, CategoricalChoice value2)
        => value1.Equals(value2) || (value1.IsNaN && value2.IsNaN);

    /// <summary>Python <c>str(value)</c> rendering (<c>None</c> / <c>True</c> / <c>1.0</c> / raw string).</summary>
    public override string ToString() => PythonRepr.Str(_value);

    /// <summary>Python <c>repr(value)</c> rendering (strings quoted).</summary>
    public string Repr() => PythonRepr.Repr(_value);
}
```


<a id="distribution-categorical-distribution"></a>

## src/OptunaSharp.Core/Distributions/CategoricalDistribution.cs

Parameter distributions · 103 physical lines · SHA-256: `a1141b82b8e881c9126c7e04a3d3d2ed8e02ddf69bb85087c1ab2da6e42e9298`

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Distributions;

/// <summary>
///     A categorical distribution. Port of <c>optuna/distributions.py: CategoricalDistribution</c>.
///
///     This object is instantiated by <see cref="Trial.SuggestCategorical(string, System.Collections.Generic.IEnumerable{CategoricalChoice})" />,
///     and passed to samplers in general.
/// </summary>
public class CategoricalDistribution : BaseDistribution
{
    /// <summary>Parameter value candidates (Python: an immutable tuple).</summary>
    public IReadOnlyList<CategoricalChoice> Choices { get; }

    public CategoricalDistribution(IEnumerable<CategoricalChoice> choices)
    {
        var materialized = choices.ToArray();
        if (materialized.Length == 0)
            throw new ValueError("The `choices` must contain one or more elements.");
        // The Python constructor warns when a choice is not None/bool/int/float/str; the
        // CategoricalChoice value type makes that state unrepresentable in C#.
        Choices = materialized;
    }

    public CategoricalDistribution(params CategoricalChoice[] choices)
        : this((IEnumerable<CategoricalChoice>)choices) { }

    /// <summary>External representation is the choice at the internal-repr index.</summary>
    public override object? ToExternalRepr(double paramValueInInternalRepr)
        => Choices[(int)paramValueInInternalRepr].Value;

    /// <summary>The choice at the internal-repr index, typed.</summary>
    public CategoricalChoice ChoiceAt(double paramValueInInternalRepr)
        => Choices[(int)paramValueInInternalRepr];

    public override double ToInternalRepr(object? paramValueInExternalRepr)
    {
        var value = CategoricalChoice.From(paramValueInExternalRepr);
        // NOTE(nabenabe): With this implementation, we cannot distinguish some values
        // such as True and 1, or 1.0 and 1. For example, if choices=[True, 1] and external_repr
        // is 1, this method wrongly returns 0 instead of 1. However, we decided to accept this
        // bug for such exceptional choices for less complexity and faster processing.
        for (int index = 0; index < Choices.Count; index++)
        {
            if (Choices[index] == value)
                return index;
        }
        // Python's fallback loop additionally treats NaN == NaN (and handles float precision).
        for (int index = 0; index < Choices.Count; index++)
        {
            if (CategoricalChoice.ChoicesEqual(value, Choices[index]))
                return index;
        }

        throw new ValueError(
            $"'{PythonRepr.Str(paramValueInExternalRepr is CategoricalChoice c ? c.Value : paramValueInExternalRepr)}' " +
            $"not in {PythonRepr.TupleRepr(Choices.Select(ch => ch.Value))}.");
    }

    public override bool Single() => Choices.Count == 1;

    internal override bool Contains(double paramValueInInternalRepr)
    {
        int index = (int)paramValueInInternalRepr;
        return 0 <= index && index < Choices.Count;
    }

    internal override OrderedDict<string, object?> AsDict() => new()
    {
        ["choices"] = Choices,
    };

    public override bool Equals(BaseDistribution? other)
    {
        // Python: isinstance check (a subclass compares equal to the base by attribute), then
        // choices compared pairwise with _categorical_choice_equal.
        if (other is not CategoricalDistribution o || !GetType().IsInstanceOfType(other) && !other.GetType().IsInstanceOfType(this))
            return false;
        if (Choices.Count != o.Choices.Count)
            return false;
        for (int i = 0; i < Choices.Count; i++)
        {
            if (!CategoricalChoice.ChoicesEqual(Choices[i], o.Choices[i]))
                return false;
        }
        return true;
    }

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(typeof(CategoricalDistribution));
        foreach (var choice in Choices)
            hash.Add(choice);
        return hash.ToHashCode();
    }
}
```


<a id="distribution-distribution"></a>

## src/OptunaSharp.Core/Distributions/Distribution.cs

Parameter distributions · 263 physical lines · SHA-256: `b1a2c778db06732120848406286a2734e50ff9849064a8b405991e3ba21bcfd1`

```csharp
using System;
using System.Collections.Generic;
using System.Globalization;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Distributions;

/// <summary>
///     Module-level functions of <c>optuna/distributions.py</c>: JSON round-trip
///     (<c>distribution_to_json</c> / <c>json_to_distribution</c>) and
///     <c>check_distribution_compatibility</c>. The storage layer depends on the JSON forms.
/// </summary>
public static class Distribution
{
    /// <summary>
    ///     Serialize a distribution to JSON format (Python <c>distribution_to_json</c>).
    /// </summary>
    /// <param name="dist">A distribution to be serialized.</param>
    /// <returns>A JSON string of a given distribution, byte-identical to Python's output.</returns>
    public static string ToJson(BaseDistribution dist)
    {
        var payload = new OrderedDict<string, object?>
        {
            ["name"] = dist.GetType().Name,
            ["attributes"] = dist.AsDict(),
        };
        return PyJson.Dumps(payload);
    }

    /// <summary>
    ///     Deserialize a distribution in JSON format (Python <c>json_to_distribution</c>).
    /// </summary>
    /// <param name="jsonStr">A JSON-serialized distribution.</param>
    /// <returns>A deserialized distribution.</returns>
    public static BaseDistribution FromJson(string jsonStr)
    {
        if (PyJson.Loads(jsonStr) is not OrderedDict<string, object?> jsonDict)
            throw new ValueError("Unknown distribution format.");

        if (jsonDict.ContainsKey("name"))
        {
            string name = (string)jsonDict["name"]!;
            var attributes = (OrderedDict<string, object?>)jsonDict["attributes"]!;

#pragma warning disable CS0618  // Deprecated aliases round-trip by design.
            switch (name)
            {
                case nameof(IntDistribution):
                    return new IntDistribution(
                        low: GetLong(attributes, "low"),
                        high: GetLong(attributes, "high"),
                        log: GetBool(attributes, "log", false),
                        step: GetLong(attributes, "step", 1));
                case nameof(IntLogUniformDistribution):
                    return new IntLogUniformDistribution(
                        low: GetLong(attributes, "low"),
                        high: GetLong(attributes, "high"),
                        step: GetLong(attributes, "step", 1));
                case nameof(IntUniformDistribution):
                    return new IntUniformDistribution(
                        low: GetLong(attributes, "low"),
                        high: GetLong(attributes, "high"),
                        step: GetLong(attributes, "step", 1));
                case nameof(FloatDistribution):
                    return new FloatDistribution(
                        low: GetDouble(attributes, "low"),
                        high: GetDouble(attributes, "high"),
                        log: GetBool(attributes, "log", false),
                        step: GetNullableDouble(attributes, "step"));
                case nameof(UniformDistribution):
                    return new UniformDistribution(
                        low: GetDouble(attributes, "low"),
                        high: GetDouble(attributes, "high"));
                case nameof(LogUniformDistribution):
                    return new LogUniformDistribution(
                        low: GetDouble(attributes, "low"),
                        high: GetDouble(attributes, "high"));
                case nameof(DiscreteUniformDistribution):
                    return new DiscreteUniformDistribution(
                        low: GetDouble(attributes, "low"),
                        high: GetDouble(attributes, "high"),
                        q: GetDouble(attributes, "q"));
                case nameof(CategoricalDistribution):
                    return new CategoricalDistribution(ToChoices(attributes["choices"]));
                default:
                    throw new ValueError($"Unknown distribution class: {name}");
            }
#pragma warning restore CS0618
        }
        else
        {
            // Deserialize a distribution from an abbreviated format.
            string type = (string)jsonDict["type"]!;
            if (type == "categorical")
                return new CategoricalDistribution(ToChoices(jsonDict["choices"]));
            if (type is "float" or "int")
            {
                var step = jsonDict.GetValueOrDefault("step");
                bool log = GetBool(jsonDict, "log", false);

                if (type == "float")
                    return new FloatDistribution(
                        GetDouble(jsonDict, "low"),
                        GetDouble(jsonDict, "high"),
                        log: log,
                        step: step is null ? null : ToDouble(step));

                return new IntDistribution(
                    low: GetLong(jsonDict, "low"),
                    high: GetLong(jsonDict, "high"),
                    log: log,
                    step: step is null ? 1 : ToLong(step));
            }

            throw new ValueError($"Unknown distribution type: {type}");
        }
    }

    /// <summary>
    ///     A function to check compatibility of two distributions
    ///     (Python <c>check_distribution_compatibility</c>).
    ///
    ///     It checks whether <paramref name="distOld" /> and <paramref name="distNew" /> are the
    ///     same kind of distributions. If <paramref name="distOld" /> is
    ///     <see cref="CategoricalDistribution" />, it further checks <c>choices</c> are the same
    ///     between both. Note that this method is not supposed to be called by library users.
    /// </summary>
    /// <param name="distOld">A distribution previously recorded in storage.</param>
    /// <param name="distNew">A distribution newly added to storage.</param>
    public static void CheckDistributionCompatibility(BaseDistribution distOld, BaseDistribution distNew)
    {
        if (distOld.GetType() != distNew.GetType())
            throw new ValueError("Cannot set different distribution kind to the same parameter name.");

        if (distOld is FloatDistribution floatOld && distNew is FloatDistribution floatNew)
        {
            if (floatOld.Log != floatNew.Log)
                throw new ValueError("Cannot set different log configuration to the same parameter name.");
        }
        else if (distOld is IntDistribution intOld && distNew is IntDistribution intNew)
        {
            if (intOld.Log != intNew.Log)
                throw new ValueError("Cannot set different log configuration to the same parameter name.");
        }

        if (distOld is not CategoricalDistribution)
            return;
        if (distNew is not CategoricalDistribution)
            return;
        if (!distOld.Equals(distNew))
            throw new ValueError(nameof(CategoricalDistribution) + " does not support dynamic value space.");
    }

    /// <summary>Python <c>_get_single_value(distribution)</c>.</summary>
    internal static object? GetSingleValue(BaseDistribution distribution)
    {
        System.Diagnostics.Debug.Assert(distribution.Single());

        return distribution switch
        {
            FloatDistribution f => f.Low,
            IntDistribution i => i.Low,
            CategoricalDistribution c => c.Choices[0].Value,
            _ => throw new InvalidOperationException("Should not reach."),
        };
    }

    /// <summary>
    ///     Python <c>_convert_old_distribution_to_new_distribution</c> — maps deprecated aliases
    ///     onto <see cref="FloatDistribution" /> / <see cref="IntDistribution" />.
    /// </summary>
    internal static BaseDistribution ConvertOldDistributionToNewDistribution(
        BaseDistribution distribution, bool suppressWarning = false)
    {
        BaseDistribution newDistribution;

#pragma warning disable CS0618
        switch (distribution)
        {
            // Float distributions.
            case UniformDistribution u:
                newDistribution = new FloatDistribution(u.Low, u.High, log: false, step: null);
                break;
            case LogUniformDistribution lu:
                newDistribution = new FloatDistribution(lu.Low, lu.High, log: true, step: null);
                break;
            case DiscreteUniformDistribution du:
                newDistribution = new FloatDistribution(du.Low, du.High, log: false, step: du.Q);
                break;

            // Integer distributions.
            case IntUniformDistribution iu:
                newDistribution = new IntDistribution(iu.Low, iu.High, log: false, step: iu.Step);
                break;
            case IntLogUniformDistribution ilu:
                newDistribution = new IntDistribution(ilu.Low, ilu.High, log: true, step: ilu.Step);
                break;

            // Categorical distribution (and already-new types).
            default:
                newDistribution = distribution;
                break;
        }
#pragma warning restore CS0618

        if (!newDistribution.Equals(distribution) && !suppressWarning)
        {
            OptunaWarnings.Warn(
                $"{distribution} is deprecated and internally converted to" +
                $" {newDistribution}. See https://github.com/optuna/optuna/issues/2941.",
                WarningCategory.FutureWarning);
        }

        return newDistribution;
    }

    private static IEnumerable<CategoricalChoice> ToChoices(object? value)
    {
        var list = (List<object?>)value!;
        foreach (var item in list)
            yield return CategoricalChoice.From(item);
    }

    private static double ToDouble(object? value) => value switch
    {
        double d => d,
        long l => l,
        bool b => b ? 1.0 : 0.0,
        _ => Convert.ToDouble(value, CultureInfo.InvariantCulture),
    };

    private static long ToLong(object? value) => value switch
    {
        long l => l,
        double d => (long)d,
        bool b => b ? 1L : 0L,
        _ => Convert.ToInt64(value, CultureInfo.InvariantCulture),
    };

    private static double GetDouble(OrderedDict<string, object?> dict, string key) => ToDouble(dict[key]);

    private static double? GetNullableDouble(OrderedDict<string, object?> dict, string key)
    {
        var value = dict.GetValueOrDefault(key);
        return value is null ? null : ToDouble(value);
    }

    private static long GetLong(OrderedDict<string, object?> dict, string key) => ToLong(dict[key]);

    private static long GetLong(OrderedDict<string, object?> dict, string key, long defaultValue)
    {
        var value = dict.GetValueOrDefault(key);
        return value is null ? defaultValue : ToLong(value);
    }

    private static bool GetBool(OrderedDict<string, object?> dict, string key, bool defaultValue)
    {
        var value = dict.GetValueOrDefault(key);
        return value is null ? defaultValue : (bool)value;
    }
}
```


<a id="distribution-float-distribution"></a>

## src/OptunaSharp.Core/Distributions/FloatDistribution.cs

Parameter distributions · 275 physical lines · SHA-256: `7a4d5e09ee22a7f946e12c1bca83ddae15a17d73ef341bda6518321c0d723203`

```csharp
using System;
using System.Globalization;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Distributions;

/// <summary>
///     A distribution on floats. Port of <c>optuna/distributions.py: FloatDistribution</c>.
///
///     This object is instantiated by <see cref="Trial.SuggestFloat(string, double, double)" />, and
///     passed to samplers in general.
/// </summary>
/// <remarks>
///     When <c>step</c> is not <c>null</c>, if the range <c>[low, high]</c> is not divisible by
///     <c>step</c>, <c>high</c> will be replaced with the maximum of
///     <c>k * step + low &lt; high</c>, where <c>k</c> is an integer.
/// </remarks>
public class FloatDistribution : BaseDistribution
{
    /// <summary>A discretization step. <c>null</c> when the distribution is continuous.</summary>
    public double? Step { get; internal set; }

    /// <summary>Lower endpoint of the range of the distribution. <c>Low</c> is included in the range.</summary>
    public double Low { get; }

    /// <summary>Upper endpoint of the range of the distribution. <c>High</c> is included in the range.</summary>
    public double High { get; }

    /// <summary>If <c>true</c>, this distribution is in log-scaled domain.</summary>
    public bool Log { get; }

    public FloatDistribution(double low, double high, bool log = false, double? step = null)
    {
        if (log && step is not null)
            throw new ValueError("The parameter `step` is not supported when `log` is true.");

        if (low > high)
            throw new ValueError(
                $"`low <= high` must hold, but got (low={PythonRepr.FloatRepr(low)}, high={PythonRepr.FloatRepr(high)}).");

        if (log && low <= 0.0)
            throw new ValueError(
                $"`low > 0` must hold for `log=True`, but got (low={PythonRepr.FloatRepr(low)}, high={PythonRepr.FloatRepr(high)}).");

        if (step is not null && step <= 0)
            throw new ValueError($"`step > 0` must hold, but got step={PythonRepr.FloatRepr(step.Value)}.");

        Step = null;
        if (step is not null)
        {
            high = AdjustDiscreteUniformHigh(low, high, step.Value);
            Step = step.Value;
        }

        Low = low;
        High = high;
        Log = log;
    }

    public override bool Single()
    {
        if (Step is null)
            return Low == High;

        if (Low == High)
            return true;
        // Python computes (high - low) < step exactly, via Decimal on the shortest reprs.
        try
        {
            var high = decimal.Parse(PythonRepr.FloatRepr(High), NumberStyles.Float, CultureInfo.InvariantCulture);
            var low = decimal.Parse(PythonRepr.FloatRepr(Low), NumberStyles.Float, CultureInfo.InvariantCulture);
            var step = decimal.Parse(PythonRepr.FloatRepr(Step.Value), NumberStyles.Float, CultureInfo.InvariantCulture);
            return (high - low) < step;
        }
        catch (OverflowException)
        {
            // Magnitudes beyond decimal's range: fall back to double arithmetic.
            return (High - Low) < Step.Value;
        }
    }

    internal override bool Contains(double paramValueInInternalRepr)
    {
        double value = paramValueInInternalRepr;
        if (Step is null)
            return Low <= value && value <= High;

        double k = (value - Low) / Step.Value;
        // Math.Round defaults to banker's rounding — same as Python round().
        return Low <= value && value <= High && Math.Abs(k - Math.Round(k)) < 1.0e-8;
    }

    public override double ToInternalRepr(object? paramValueInExternalRepr)
    {
        if (!PyFloat.TryConvert(paramValueInExternalRepr, out double internalRepr))
            throw new ValueError(
                $"'{PythonRepr.Str(paramValueInExternalRepr)}' is not a valid type. float-castable value is expected.");

        if (double.IsNaN(internalRepr))
            throw new ValueError($"`{PythonRepr.Str(paramValueInExternalRepr)}` is invalid value.");
        if (Log && internalRepr <= 0.0)
            throw new ValueError(
                $"`{PythonRepr.Str(paramValueInExternalRepr)}` is invalid value for the case log=True.");
        return internalRepr;
    }

    internal override OrderedDict<string, object?> AsDict() => new()
    {
        // __init__ assignment order: step, low, high, log.
        ["step"] = Step,
        ["low"] = Low,
        ["high"] = High,
        ["log"] = Log,
    };

    public override bool Equals(BaseDistribution? other)
    {
        if (other is null || other.GetType() != GetType())
            return false;
        var o = (FloatDistribution)other;
        // Python compares __dict__ with ==; None == None and float equality (no NaN in bounds).
        return Nullable.Equals(Step, o.Step) && Low.Equals(o.Low) && High.Equals(o.High) && Log == o.Log;
    }

    public override int GetHashCode() => HashCode.Combine(GetType(), Step, Low, High, Log);

    /// <summary>Python <c>_adjust_discrete_uniform_high</c> (exact Decimal arithmetic on the shortest reprs).</summary>
    internal static double AdjustDiscreteUniformHigh(double low, double high, double step)
    {
        try
        {
            var dHigh = decimal.Parse(PythonRepr.FloatRepr(high), NumberStyles.Float, CultureInfo.InvariantCulture);
            var dLow = decimal.Parse(PythonRepr.FloatRepr(low), NumberStyles.Float, CultureInfo.InvariantCulture);
            var dStep = decimal.Parse(PythonRepr.FloatRepr(step), NumberStyles.Float, CultureInfo.InvariantCulture);

            var dR = dHigh - dLow;

            if (dR % dStep != 0m)
            {
                double oldHigh = high;
                // Python: float((d_r // d_step) * d_step + d_low); Decimal floor division.
                high = (double)(decimal.Floor(dR / dStep) * dStep + dLow);
                OptunaWarnings.Warn(
                    $"The distribution is specified by [{PythonRepr.FloatRepr(low)}, {PythonRepr.FloatRepr(oldHigh)}] " +
                    $"and step={PythonRepr.FloatRepr(step)}, but the range is " +
                    $"not divisible by `step`. It will be replaced with [{PythonRepr.FloatRepr(low)}, {PythonRepr.FloatRepr(high)}].");
            }

            return high;
        }
        catch (OverflowException)
        {
            return high;
        }
    }
}

/// <summary>
///     A uniform distribution in the linear domain.
///     Deprecated since Optuna 3.0.0: use <see cref="FloatDistribution" /> instead.
/// </summary>
[Obsolete("Deprecated in v3.0.0. This feature will be removed in v6.0.0. Use FloatDistribution instead.")]
public class UniformDistribution : FloatDistribution
{
    public UniformDistribution(double low, double high) : base(low, high, log: false, step: null) { }

    internal override OrderedDict<string, object?> AsDict() => new()
    {
        ["low"] = Low,
        ["high"] = High,
    };
}

/// <summary>
///     A uniform distribution in the log domain.
///     Deprecated since Optuna 3.0.0: use <see cref="FloatDistribution" /> instead.
/// </summary>
[Obsolete("Deprecated in v3.0.0. This feature will be removed in v6.0.0. Use FloatDistribution instead.")]
public class LogUniformDistribution : FloatDistribution
{
    public LogUniformDistribution(double low, double high) : base(low, high, log: true, step: null) { }

    internal override OrderedDict<string, object?> AsDict() => new()
    {
        ["low"] = Low,
        ["high"] = High,
    };
}

/// <summary>
///     A discretized uniform distribution in the linear domain.
///     Deprecated since Optuna 3.0.0: use <see cref="FloatDistribution" /> instead.
/// </summary>
[Obsolete("Deprecated in v3.0.0. This feature will be removed in v6.0.0. Use FloatDistribution instead.")]
public class DiscreteUniformDistribution : FloatDistribution
{
    public DiscreteUniformDistribution(double low, double high, double q) : base(low, high, step: q) { }

    /// <summary>
    ///     Discretization step: a proxy for the <see cref="FloatDistribution.Step" /> attribute.
    /// </summary>
    public double Q
    {
        get => Step!.Value;
        set => Step = value;
    }

    internal override OrderedDict<string, object?> AsDict() => new()
    {
        ["low"] = Low,
        ["high"] = High,
        ["q"] = Step,
    };
}

/// <summary>
///     Python <c>float()</c> conversion over the external-repr object model — accepts bools,
///     any numeric, and numeric strings (including <c>"nan"</c>/<c>"inf"</c>).
/// </summary>
internal static class PyFloat
{
    internal static bool TryConvert(object? value, out double result)
    {
        switch (value)
        {
            case bool b:
                result = b ? 1.0 : 0.0;
                return true;
            case double d:
                result = d;
                return true;
            case float f:
                result = f;
                return true;
            case sbyte or byte or short or ushort or int or uint or long:
                result = Convert.ToDouble(value, CultureInfo.InvariantCulture);
                return true;
            case ulong ul:
                result = ul;
                return true;
            case decimal m:
                result = (double)m;
                return true;
            case CategoricalChoice choice:
                return TryConvert(choice.Value, out result);
            case string s:
                return TryParse(s, out result);
            default:
                result = 0;
                return false;
        }
    }

    private static bool TryParse(string s, out double result)
    {
        s = s.Trim();
        // Python float() accepts "inf", "infinity", "nan" (any case, optional sign).
        string lower = s.ToLowerInvariant();
        switch (lower)
        {
            case "inf" or "+inf" or "infinity" or "+infinity":
                result = double.PositiveInfinity;
                return true;
            case "-inf" or "-infinity":
                result = double.NegativeInfinity;
                return true;
            case "nan" or "+nan" or "-nan":
                result = double.NaN;
                return true;
        }
        return double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out result);
    }
}
```


<a id="distribution-int-distribution"></a>

## src/OptunaSharp.Core/Distributions/IntDistribution.cs

Parameter distributions · 156 physical lines · SHA-256: `290d8108938a6450431c9d5872fa628c006955af7d74122ade1bc8410745effc`

```csharp
using System;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Distributions;

/// <summary>
///     A distribution on integers. Port of <c>optuna/distributions.py: IntDistribution</c>.
///
///     This object is instantiated by <see cref="Trial.SuggestInt" />, and passed to samplers
///     in general.
/// </summary>
/// <remarks>
///     When <c>step</c> is not 1, if the range <c>[low, high]</c> is not divisible by
///     <c>step</c>, <c>high</c> will be replaced with the maximum of
///     <c>k * step + low &lt; high</c>, where <c>k</c> is an integer.
/// </remarks>
public class IntDistribution : BaseDistribution
{
    /// <summary>If <c>true</c>, this distribution is in log-scaled domain.</summary>
    public bool Log { get; }

    /// <summary>A discretization step. Must be a positive integer (1 when <c>log</c> is <c>true</c>).</summary>
    public long Step { get; }

    /// <summary>Lower endpoint of the range of the distribution. <c>Low</c> is included in the range.</summary>
    public long Low { get; }

    /// <summary>Upper endpoint of the range of the distribution. <c>High</c> is included in the range.</summary>
    public long High { get; }

    public IntDistribution(long low, long high, bool log = false, long step = 1)
    {
        if (log && step != 1)
            throw new ValueError(
                "Samplers and other components in Optuna only accept step is 1 " +
                "when `log` argument is True.");

        if (low > high)
            throw new ValueError($"`low <= high` must hold, but got (low={low}, high={high}).");

        if (log && low < 1)
            throw new ValueError($"`low >= 1` must hold for `log=True`, but got (low={low}, high={high}).");

        if (step <= 0)
            throw new ValueError($"`step > 0` must hold, but got step={step}.");

        Log = log;
        Step = step;
        Low = low;
        High = AdjustIntUniformHigh(Low, high, Step);
    }

    /// <summary>External representation is a boxed <see cref="long" /> (Python <c>int</c>).</summary>
    public override object? ToExternalRepr(double paramValueInInternalRepr)
        => (long)paramValueInInternalRepr;

    public override double ToInternalRepr(object? paramValueInExternalRepr)
    {
        if (!PyFloat.TryConvert(paramValueInExternalRepr, out double internalRepr))
            throw new ValueError(
                $"'{PythonRepr.Str(paramValueInExternalRepr)}' is not a valid type. float-castable value is expected.");

        if (double.IsNaN(internalRepr))
            throw new ValueError($"`{PythonRepr.Str(paramValueInExternalRepr)}` is invalid value.");
        if (Log && internalRepr <= 0.0)
            throw new ValueError(
                $"`{PythonRepr.Str(paramValueInExternalRepr)}` is invalid value for the case log=True.");
        return internalRepr;
    }

    public override bool Single()
    {
        if (Log)
            return Low == High;

        if (Low == High)
            return true;
        return (High - Low) < Step;
    }

    internal override bool Contains(double paramValueInInternalRepr)
    {
        double value = paramValueInInternalRepr;
        return Low <= value && value <= High && (value - Low) % Step == 0;
    }

    internal override OrderedDict<string, object?> AsDict() => new()
    {
        // __init__ assignment order: log, step, low, high.
        ["log"] = Log,
        ["step"] = Step,
        ["low"] = Low,
        ["high"] = High,
    };

    public override bool Equals(BaseDistribution? other)
    {
        if (other is null || other.GetType() != GetType())
            return false;
        var o = (IntDistribution)other;
        return Log == o.Log && Step == o.Step && Low == o.Low && High == o.High;
    }

    public override int GetHashCode() => HashCode.Combine(GetType(), Log, Step, Low, High);

    /// <summary>Python <c>_adjust_int_uniform_high</c>.</summary>
    internal static long AdjustIntUniformHigh(long low, long high, long step)
    {
        long r = high - low;
        if (r % step != 0)
        {
            long oldHigh = high;
            high = r / step * step + low;
            OptunaWarnings.Warn(
                $"The distribution is specified by [{low}, {oldHigh}] and step={step}, but the range is " +
                $"not divisible by `step`. It will be replaced with [{low}, {high}].");
        }
        return high;
    }
}

/// <summary>
///     A uniform distribution on integers.
///     Deprecated since Optuna 3.0.0: use <see cref="IntDistribution" /> instead.
/// </summary>
[Obsolete("Deprecated in v3.0.0. This feature will be removed in v6.0.0. Use IntDistribution instead.")]
public class IntUniformDistribution : IntDistribution
{
    public IntUniformDistribution(long low, long high, long step = 1) : base(low, high, log: false, step: step) { }

    internal override OrderedDict<string, object?> AsDict() => new()
    {
        ["step"] = Step,
        ["low"] = Low,
        ["high"] = High,
    };
}

/// <summary>
///     A uniform distribution on integers in the log domain.
///     Deprecated since Optuna 3.0.0: use <see cref="IntDistribution" /> instead.
/// </summary>
[Obsolete("Deprecated in v3.0.0. This feature will be removed in v6.0.0. Use IntDistribution instead.")]
public class IntLogUniformDistribution : IntDistribution
{
    public IntLogUniformDistribution(long low, long high, long step = 1) : base(low, high, log: true, step: step) { }

    internal override OrderedDict<string, object?> AsDict() => new()
    {
        ["step"] = Step,
        ["low"] = Low,
        ["high"] = High,
    };
}
```


<a id="ordered-dict-custom"></a>

## src/OptunaSharp.Core/Collections/OrderedDict.Custom.cs

Ordered collections · 15 physical lines · SHA-256: `0d6b7e7787a4953a19da28a9b3bf3500b861c8280eab0afadd4f4c9ed2fed615`

```csharp
using System.Collections.Generic;

namespace OptunaSharp.Collections;

// Members shared across both target frameworks (net8.0 vendored impl + net9.0+ in-box subclass).
// These sit on OrderedDict regardless of which base it has, so OptunaSharp code sees one surface.
public sealed partial class OrderedDict<TKey, TValue>
{
    /// <summary>Python <c>dict.get(key, default)</c>.</summary>
    public TValue? GetValueOrDefault(TKey key, TValue? defaultValue = default)
        => TryGetValue(key, out var value) ? value : defaultValue;

    /// <summary>Shallow copy preserving insertion order (Python <c>copy.copy(dict)</c>).</summary>
    public OrderedDict<TKey, TValue> ShallowClone() => new(this);
}
```


<a id="ordered-dict-net8"></a>

## src/OptunaSharp.Core/Collections/OrderedDict.Net8.cs

Ordered collections · 1928 physical lines · SHA-256: `36c155129e375d509f113b03fbc25f8c8f4e5c22e758167ab4cb38f4a9976d64`

```csharp
// <auto-generated>
// Vendored from dotnet/runtime @ release/10.0:
//   src/libraries/System.Collections/src/System/Collections/Generic/OrderedDictionary.cs
// Licensed by the .NET Foundation under the MIT license (same as OptunaSharp).
// This is the net8.0 polyfill for System.Collections.Generic.OrderedDictionary<TKey,TValue>
// (added to the BCL in .NET 9). On net9.0+ OptunaSharp.Collections.OrderedDict subclasses the
// in-box type instead (see OrderedDict.cs); this file compiles only when the in-box type is absent.
// LOCAL MODIFICATIONS: type renamed OrderedDictionary->OrderedDict; namespace ->
// OptunaSharp.Collections; made 'public sealed partial'; DebuggerTypeProxy removed;
// SYSTEM_COLLECTIONS branches (NonRandomizedStringEqualityComparer) left disabled;
// AddRange switched from Add() to the indexer for Python dict(pairs) last-wins semantics.
// </auto-generated>
#if !NET9_0_OR_GREATER
#nullable enable
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#if NET
using static System.ArgumentOutOfRangeException;
#else
using static System.Collections.ThrowHelper;
#endif

namespace OptunaSharp.Collections
{
    /// <summary>
    /// Represents a collection of key/value pairs that are accessible by the key or index.
    /// </summary>
    /// <typeparam name="TKey">The type of the keys in the dictionary.</typeparam>
    /// <typeparam name="TValue">The type of the values in the dictionary.</typeparam>
    /// <remarks>
    /// Operations on the collection have algorithmic complexities that are similar to that of the <see cref="List{T}"/>
    /// class, except with lookups by key similar in complexity to that of <see cref="Dictionary{TKey, TValue}"/>.
    /// </remarks>
    [DebuggerDisplay("Count = {Count}")]
    public sealed partial class OrderedDict<TKey, TValue> :
        IDictionary<TKey, TValue>, IReadOnlyDictionary<TKey, TValue>, IDictionary,
        IList<KeyValuePair<TKey, TValue>>, IReadOnlyList<KeyValuePair<TKey, TValue>>, IList
        where TKey : notnull
    {
        /// <summary>The comparer used by the collection. May be null if the default comparer is used.</summary>
        private IEqualityComparer<TKey>? _comparer;
        /// <summary>Indexes into <see cref="_entries"/> for the start of chains; indices are 1-based.</summary>
        private int[]? _buckets;
        /// <summary>Ordered entries in the dictionary.</summary>
        /// <remarks>
        /// Unlike <see cref="Dictionary{TKey, TValue}"/>, removed entries are actually removed rather than left as holes
        /// that can be filled in by subsequent additions. This is done to retain ordering.
        /// </remarks>
        private Entry[]? _entries;
        /// <summary>The number of items in the collection.</summary>
        private int _count;
        /// <summary>Version number used to invalidate an enumerator.</summary>
        private int _version;
        /// <summary>Multiplier used on 64-bit to enable faster % operations.</summary>
        private ulong _fastModMultiplier;

        /// <summary>
        /// Initializes a new instance of the <see cref="OrderedDict{TKey, TValue}"/> class that is empty,
        /// has the default initial capacity, and uses the default equality comparer for the key type.
        /// </summary>
        public OrderedDict() : this(0, null)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="OrderedDict{TKey, TValue}"/> class that is empty,
        /// has the specified initial capacity, and uses the default equality comparer for the key type.
        /// </summary>
        /// <param name="capacity">The initial number of elements that the <see cref="OrderedDict{TKey, TValue}"/> can contain.</param>
        /// <exception cref="ArgumentOutOfRangeException">capacity is less than 0.</exception>
        public OrderedDict(int capacity) : this(capacity, null)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="OrderedDict{TKey, TValue}"/> class that is empty,
        /// has the default initial capacity, and uses the specified <see cref="IEqualityComparer{TKey}"/>.
        /// </summary>
        /// <param name="comparer">
        /// The <see cref="IEqualityComparer{TKey}"/> implementation to use when comparing keys,
        /// or null to use the default <see cref="EqualityComparer{TKey}"/> for the type of the key.
        /// </param>
        public OrderedDict(IEqualityComparer<TKey>? comparer) : this(0, comparer)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="OrderedDict{TKey, TValue}"/> class that is empty,
        /// has the specified initial capacity, and uses the specified <see cref="IEqualityComparer{TKey}"/>.
        /// </summary>
        /// <param name="capacity">The initial number of elements that the <see cref="OrderedDict{TKey, TValue}"/> can contain.</param>
        /// <param name="comparer">
        /// The <see cref="IEqualityComparer{TKey}"/> implementation to use when comparing keys,
        /// or null to use the default <see cref="EqualityComparer{TKey}"/> for the type of the key.
        /// </param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="capacity"/> is less than 0.</exception>
        public OrderedDict(int capacity, IEqualityComparer<TKey>? comparer)
        {
            ThrowIfNegative(capacity);

            if (capacity > 0)
            {
                EnsureBucketsAndEntriesInitialized(capacity);
            }

            // Initialize the comparer:
            // - Strings: Special-case EqualityComparer<string>.Default, StringComparer.Ordinal, and
            //   StringComparer.OrdinalIgnoreCase. We start with a non-randomized comparer for improved throughput,
            //   falling back to a randomized comparer if the hash buckets become sufficiently unbalanced to cause
            //   more collisions than a preset threshold.
            // - Other reference types: we always want to store a comparer instance, either the one provided,
            //   or if one wasn't provided, the default (accessing EqualityComparer<TKey>.Default
            //   with shared generics on every dictionary access can add measurable overhead).
            // - Value types: if no comparer is provided, or if the default is provided, we'd prefer to use
            //   EqualityComparer<TKey>.Default.Equals on every use, enabling the JIT to
            //   devirtualize and possibly inline the operation.
            if (!typeof(TKey).IsValueType)
            {
                _comparer = comparer ?? EqualityComparer<TKey>.Default;

#if SYSTEM_COLLECTIONS
                if (typeof(TKey) == typeof(string) &&
                    NonRandomizedStringEqualityComparer.GetStringComparer(_comparer) is IEqualityComparer<string> stringComparer)
                {
                    _comparer = (IEqualityComparer<TKey>)stringComparer;
                }
#endif
            }
            else if (comparer is not null && // first check for null to avoid forcing default comparer instantiation unnecessarily
                     comparer != EqualityComparer<TKey>.Default)
            {
                _comparer = comparer;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="OrderedDict{TKey, TValue}"/> class that contains elements copied from
        /// the specified <see cref="IDictionary{TKey, TValue}"/> and uses the default equality comparer for the key type.
        /// </summary>
        /// <param name="dictionary">
        /// The <see cref="IDictionary{TKey, TValue}"/> whose elements are copied to the new <see cref="OrderedDict{TKey, TValue}"/>.
        /// The initial order of the elements in the new collection is the order the elements are enumerated from the supplied dictionary.
        /// </param>
        /// <exception cref="ArgumentNullException"><paramref name="dictionary"/> is <see langword="null"/>.</exception>
        public OrderedDict(IDictionary<TKey, TValue> dictionary) : this(dictionary, null)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="OrderedDict{TKey, TValue}"/> class that contains elements copied from
        /// the specified <see cref="IDictionary{TKey, TValue}"/> and uses the specified <see cref="IEqualityComparer{TKey}"/>.
        /// </summary>
        /// <param name="dictionary">
        /// The <see cref="IDictionary{TKey, TValue}"/> whose elements are copied to the new <see cref="OrderedDict{TKey, TValue}"/>.
        /// The initial order of the elements in the new collection is the order the elements are enumerated from the supplied dictionary.
        /// </param>
        /// <param name="comparer">
        /// The <see cref="IEqualityComparer{TKey}"/> implementation to use when comparing keys,
        /// or null to use the default <see cref="EqualityComparer{TKey}"/> for the type of the key.
        /// </param>
        /// <exception cref="ArgumentNullException"><paramref name="dictionary"/> is <see langword="null"/>.</exception>
        public OrderedDict(IDictionary<TKey, TValue> dictionary, IEqualityComparer<TKey>? comparer) :
            this(dictionary?.Count ?? 0, comparer)
        {
            ArgumentNullException.ThrowIfNull(dictionary);

            AddRange(dictionary);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="OrderedDict{TKey, TValue}"/> class that contains elements copied
        /// from the specified <see cref="IEnumerable{T}"/> and uses the default equality comparer for the key type.
        /// </summary>
        /// <param name="collection">
        /// The <see cref="IEnumerable{T}"/> whose elements are copied to the new <see cref="OrderedDict{TKey, TValue}"/>.
        /// The initial order of the elements in the new collection is the order the elements are enumerated from the supplied collection.
        /// </param>
        /// <exception cref="ArgumentNullException"><paramref name="collection"/> is <see langword="null"/>.</exception>
        public OrderedDict(IEnumerable<KeyValuePair<TKey, TValue>> collection) : this(collection, null)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="OrderedDict{TKey, TValue}"/> class that contains elements copied
        /// from the specified <see cref="IEnumerable{T}"/> and uses the specified <see cref="IEqualityComparer{TKey}"/>.
        /// </summary>
        /// <param name="collection">
        /// The <see cref="IEnumerable{T}"/> whose elements are copied to the new <see cref="OrderedDict{TKey, TValue}"/>.
        /// The initial order of the elements in the new collection is the order the elements are enumerated from the supplied collection.
        /// </param>
        /// <param name="comparer">
        /// The <see cref="IEqualityComparer{TKey}"/> implementation to use when comparing keys,
        /// or null to use the default <see cref="EqualityComparer{TKey}"/> for the type of the key.
        /// </param>
        /// <exception cref="ArgumentNullException"><paramref name="collection"/> is <see langword="null"/>.</exception>
        public OrderedDict(IEnumerable<KeyValuePair<TKey, TValue>> collection, IEqualityComparer<TKey>? comparer) :
            this((collection as ICollection<KeyValuePair<TKey, TValue>>)?.Count ?? 0, comparer)
        {
            ArgumentNullException.ThrowIfNull(collection);

            AddRange(collection);
        }

        /// <summary>Initializes the <see cref="_buckets"/>/<see cref="_entries"/>.</summary>
        /// <param name="capacity"></param>
        [MemberNotNull(nameof(_buckets))]
        [MemberNotNull(nameof(_entries))]
        private void EnsureBucketsAndEntriesInitialized(int capacity)
        {
            Resize(HashHelpers.GetPrime(capacity));
        }

        /// <summary>Gets the total number of key/value pairs the internal data structure can hold without resizing.</summary>
        public int Capacity => _entries?.Length ?? 0;

        /// <summary>Gets the <see cref="IEqualityComparer{TKey}"/> that is used to determine equality of keys for the dictionary.</summary>
        public IEqualityComparer<TKey> Comparer
        {
            get
            {
                IEqualityComparer<TKey>? comparer = _comparer;

#if SYSTEM_COLLECTIONS
                // If the key is a string, we may have substituted a non-randomized comparer during construction.
                // If we did, fish out and return the actual comparer that had been provided.
                if (typeof(TKey) == typeof(string) &&
                    (comparer as NonRandomizedStringEqualityComparer)?.GetUnderlyingEqualityComparer() is IEqualityComparer<TKey> ec)
                {
                    return ec;
                }
#endif

                // Otherwise, return whatever comparer we have, or the default if none was provided.
                return comparer ?? EqualityComparer<TKey>.Default;
            }
        }

        /// <summary>Gets the number of key/value pairs contained in the <see cref="OrderedDict{TKey, TValue}"/>.</summary>
        public int Count => _count;

        /// <inheritdoc/>
        bool ICollection<KeyValuePair<TKey, TValue>>.IsReadOnly => false;

        /// <inheritdoc/>
        bool IDictionary.IsReadOnly => false;

        /// <inheritdoc/>
        bool IList.IsReadOnly => false;

        /// <inheritdoc/>
        bool IDictionary.IsFixedSize => false;

        /// <inheritdoc/>
        bool IList.IsFixedSize => false;

        /// <summary>Gets a collection containing the keys in the <see cref="OrderedDict{TKey, TValue}"/>.</summary>
        public KeyCollection Keys => field ??= new(this);

        /// <inheritdoc/>
        IEnumerable<TKey> IReadOnlyDictionary<TKey, TValue>.Keys => Keys;

        /// <inheritdoc/>
        ICollection<TKey> IDictionary<TKey, TValue>.Keys => Keys;

        /// <inheritdoc/>
        ICollection IDictionary.Keys => Keys;

        /// <summary>Gets a collection containing the values in the <see cref="OrderedDict{TKey, TValue}"/>.</summary>
        public ValueCollection Values => field ??= new(this);

        /// <inheritdoc/>
        IEnumerable<TValue> IReadOnlyDictionary<TKey, TValue>.Values => Values;

        /// <inheritdoc/>
        ICollection<TValue> IDictionary<TKey, TValue>.Values => Values;

        /// <inheritdoc/>
        ICollection IDictionary.Values => Values;

        /// <inheritdoc/>
        bool ICollection.IsSynchronized => false;

        /// <inheritdoc/>
        object ICollection.SyncRoot => this;

        /// <inheritdoc/>
        object? IList.this[int index]
        {
            get => GetAt(index);
            set
            {
                ArgumentNullException.ThrowIfNull(value);

                if (value is not KeyValuePair<TKey, TValue> tpair)
                {
                    throw new ArgumentException(SR.Format(SR.Arg_WrongType, value, typeof(KeyValuePair<TKey, TValue>)), nameof(value));
                }

                SetAt(index, tpair.Key, tpair.Value);
            }
        }

        /// <inheritdoc/>
        object? IDictionary.this[object key]
        {
            get
            {
                ArgumentNullException.ThrowIfNull(key);

                if (key is TKey tkey && TryGetValue(tkey, out TValue? value))
                {
                    return value;
                }

                return null;
            }
            set
            {
                ArgumentNullException.ThrowIfNull(key);
                if (default(TValue) is not null)
                {
                    ArgumentNullException.ThrowIfNull(value);
                }

                if (key is not TKey tkey)
                {
                    throw new ArgumentException(SR.Format(SR.Arg_WrongType, key, typeof(TKey)), nameof(key));
                }

                TValue tvalue = default!;
                if (value is not null)
                {
                    if (value is not TValue temp)
                    {
                        throw new ArgumentException(SR.Format(SR.Arg_WrongType, value, typeof(TValue)), nameof(value));
                    }

                    tvalue = temp;
                }

                this[tkey] = tvalue;
            }
        }

        /// <inheritdoc/>
        KeyValuePair<TKey, TValue> IList<KeyValuePair<TKey, TValue>>.this[int index]
        {
            get => GetAt(index);
            set => SetAt(index, value.Key, value.Value);
        }

        /// <inheritdoc/>
        KeyValuePair<TKey, TValue> IReadOnlyList<KeyValuePair<TKey, TValue>>.this[int index] => GetAt(index);

        /// <summary>Gets or sets the value associated with the specified key.</summary>
        /// <param name="key">The key of the value to get or set.</param>
        /// <returns>The value associated with the specified key. If the specified key is not found, a get operation throws a <see cref="KeyNotFoundException"/>, and a set operation creates a new element with the specified key.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <see langword="null"/>.</exception>
        /// <exception cref="KeyNotFoundException">The property is retrieved and <paramref name="key"/> does not exist in the collection.</exception>
        /// <remarks>Setting the value of an existing key does not impact its order in the collection.</remarks>
        public TValue this[TKey key]
        {
            get
            {
                if (!TryGetValue(key, out TValue? value))
                {
                    ThrowHelper.ThrowKeyNotFound(key);
                }

                return value;
            }
            set
            {
                ArgumentNullException.ThrowIfNull(key);

                bool modified = TryInsert(index: -1, key, value, InsertionBehavior.OverwriteExisting, out _);
                Debug.Assert(modified);
            }
        }

        /// <summary>Insert the key/value pair at the specified index.</summary>
        /// <param name="index">The index at which to insert the pair, or -1 to append.</param>
        /// <param name="key">The key to insert.</param>
        /// <param name="value">The value to insert.</param>
        /// <param name="behavior">
        /// The behavior controlling insertion behavior with respect to key duplication:
        /// - IgnoreInsertion: Immediately ends the operation, returning false, if the key already exists, e.g. TryAdd(key, value)
        /// - OverwriteExisting: If the key already exists, overwrites its value with the specified value, e.g. this[key] = value
        /// - ThrowOnExisting: If the key already exists, throws an exception, e.g. Add(key, value)
        /// </param>
        /// <param name="keyIndex">The index of the added or existing key. This is always a valid index into the dictionary.</param>
        /// <returns>true if the collection was updated; otherwise, false.</returns>
        private bool TryInsert(int index, TKey key, TValue value, InsertionBehavior behavior, out int keyIndex)
        {
            // Search for the key in the dictionary.
            uint hashCode = 0, collisionCount = 0;
            int i = IndexOf(key, ref hashCode, ref collisionCount);

            // Handle the case where the key already exists, based on the requested behavior.
            if (i >= 0)
            {
                keyIndex = i;
                Debug.Assert(0 <= keyIndex && keyIndex < _count);

                Debug.Assert(_entries is not null);

                switch (behavior)
                {
                    case InsertionBehavior.OverwriteExisting:
                        Debug.Assert(index < 0, "Expected index to be unspecied when overwriting an existing key.");
                        _entries[i].Value = value;
                        return true;

                    case InsertionBehavior.ThrowOnExisting:
                        ThrowHelper.ThrowDuplicateKey(key);
                        break;

                    default:
                        Debug.Assert(behavior is InsertionBehavior.IgnoreInsertion, $"Unknown behavior: {behavior}");
                        Debug.Assert(index < 0, "Expected index to be unspecied when ignoring a duplicate key.");
                        return false;
                }
            }

            // The key doesn't exist. If a non-negative index was provided, that is the desired index at which to insert,
            // which should have already been validated by the caller. If negative, we're appending.
            if (index < 0)
            {
                index = _count;
            }
            Debug.Assert(index <= _count);

            // Ensure the collection has been initialized.
            if (_buckets is null)
            {
                EnsureBucketsAndEntriesInitialized(0);
            }

            // As we just initialized the collection, _entries must be non-null.
            Entry[]? entries = _entries;
            Debug.Assert(entries is not null);

            // Grow capacity if necessary to accomodate the extra entry.
            if (entries.Length == _count)
            {
                Resize(HashHelpers.ExpandPrime(entries.Length));
                entries = _entries;
            }

            // The _entries array is ordered, so we need to insert the new entry at the specified index. That means
            // not only shifting up all elements at that index and higher, but also updating the buckets and chains
            // to record the newly updated indices.
            for (i = _count - 1; i >= index; --i)
            {
                entries[i + 1] = entries[i];
                UpdateBucketIndex(i, shiftAmount: 1);
            }

            // Store the new key/value pair.
            ref Entry entry = ref entries[index];
            entry.HashCode = hashCode;
            entry.Key = key;
            entry.Value = value;
            PushEntryIntoBucket(ref entry, index);
            _count++;
            _version++;

            RehashIfNecessary(collisionCount, entries);

            keyIndex = index;
            Debug.Assert(0 <= keyIndex && keyIndex < _count);

            return true;
        }

        /// <summary>Adds the specified key and value to the dictionary.</summary>
        /// <param name="key">The key of the element to add.</param>
        /// <param name="value">The value of the element to add. The value can be <see langword="null"/> for reference types.</param>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">An element with the same key already exists in the <see cref="OrderedDict{TKey, TValue}"/>.</exception>
        public void Add(TKey key, TValue value)
        {
            ArgumentNullException.ThrowIfNull(key);

            TryInsert(index: -1, key, value, InsertionBehavior.ThrowOnExisting, out _);
        }

        /// <summary>Adds the specified key and value to the dictionary if the key doesn't already exist.</summary>
        /// <param name="key">The key of the element to add.</param>
        /// <param name="value">The value of the element to add. The value can be <see langword="null"/> for reference types.</param>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <see langword="null"/>.</exception>
        /// <returns><see langword="true"/> if the key didn't exist and the key and value were added to the dictionary; otherwise, <see langword="false"/>.</returns>
        public bool TryAdd(TKey key, TValue value) => TryAdd(key, value, out _);

        /// <summary>Adds the specified key and value to the dictionary if the key doesn't already exist.</summary>
        /// <param name="key">The key of the element to add.</param>
        /// <param name="value">The value of the element to add. The value can be <see langword="null"/> for reference types.</param>
        /// <param name="index">The index of the added or existing <paramref name="key"/>. This is always a valid index into the dictionary.</param>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <see langword="null"/>.</exception>
        /// <returns><see langword="true"/> if the key didn't exist and the key and value were added to the dictionary; otherwise, <see langword="false"/>.</returns>
        public bool TryAdd(TKey key, TValue value, out int index)
        {
            ArgumentNullException.ThrowIfNull(key);

            return TryInsert(index: -1, key, value, InsertionBehavior.IgnoreInsertion, out index);
        }

        /// <summary>Adds each element of the enumerable to the dictionary.</summary>
        private void AddRange(IEnumerable<KeyValuePair<TKey, TValue>> collection)
        {
            Debug.Assert(collection is not null);

            if (collection is KeyValuePair<TKey, TValue>[] array)
            {
                foreach (KeyValuePair<TKey, TValue> pair in array)
                {
                    this[pair.Key] = pair.Value; // OPTUNASHARP: last-wins (Python dict(pairs)); upstream used Add() (throw-on-dup)
                }
            }
            else
            {
                foreach (KeyValuePair<TKey, TValue> pair in collection)
                {
                    this[pair.Key] = pair.Value; // OPTUNASHARP: last-wins (Python dict(pairs)); upstream used Add() (throw-on-dup)
                }
            }
        }

        /// <summary>Removes all keys and values from the <see cref="OrderedDict{TKey, TValue}"/>.</summary>
        public void Clear()
        {
            if (_buckets is not null && _count != 0)
            {
                Debug.Assert(_entries is not null);

                Array.Clear(_buckets, 0, _buckets.Length);
                Array.Clear(_entries, 0, _count);
                _count = 0;
                _version++;
            }
        }

        /// <summary>Determines whether the <see cref="OrderedDict{TKey, TValue}"/> contains the specified key.</summary>
        /// <param name="key">The key to locate in the <see cref="OrderedDict{TKey, TValue}"/>.</param>
        /// <returns><see langword="true"/> if the <see cref="OrderedDict{TKey, TValue}"/> contains an element with the specified key; otherwise, <see langword="false"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <see langword="null"/>.</exception>
        public bool ContainsKey(TKey key) => IndexOf(key) >= 0;

        /// <summary>Determines whether the <see cref="OrderedDict{TKey, TValue}"/> contains a specific value.</summary>
        /// <param name="value">The value to locate in the <see cref="OrderedDict{TKey, TValue}"/>. The value can be null for reference types.</param>
        /// <returns><see langword="true"/> if the <see cref="OrderedDict{TKey, TValue}"/> contains an element with the specified value; otherwise, <see langword="false"/>.</returns>
        public bool ContainsValue(TValue value)
        {
            int count = _count;

            Entry[]? entries = _entries;
            if (entries is null)
            {
                return false;
            }

            if (typeof(TValue).IsValueType)
            {
                for (int i = 0; i < count; i++)
                {
                    if (EqualityComparer<TValue>.Default.Equals(value, entries[i].Value))
                    {
                        return true;
                    }
                }
            }
            else
            {
                EqualityComparer<TValue> comparer = EqualityComparer<TValue>.Default;
                for (int i = 0; i < count; i++)
                {
                    if (comparer.Equals(value, entries[i].Value))
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        /// <summary>Gets the key/value pair at the specified index.</summary>
        /// <param name="index">The zero-based index of the pair to get.</param>
        /// <returns>The element at the specified index.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is less than 0 or greater than or equal to <see cref="Count"/>.</exception>
        public KeyValuePair<TKey, TValue> GetAt(int index)
        {
            if ((uint)index >= (uint)_count)
            {
                ThrowHelper.ThrowIndexArgumentOutOfRange();
            }

            Debug.Assert(_entries is not null, "count must be positive, which means we must have entries");

            ref Entry e = ref _entries[index];
            return new(e.Key, e.Value);
        }

        /// <summary>Determines the index of a specific key in the <see cref="OrderedDict{TKey, TValue}"/>.</summary>
        /// <param name="key">The key to locate.</param>
        /// <returns>The index of <paramref name="key"/> if found; otherwise, -1.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <see langword="null"/>.</exception>
        public int IndexOf(TKey key)
        {
            ArgumentNullException.ThrowIfNull(key);

            uint _ = 0;
            return IndexOf(key, ref _, ref _);
        }

        private int IndexOf(TKey key, ref uint outHashCode, ref uint outCollisionCount)
        {
            Debug.Assert(key is not null, "Key nullness should have been validated by caller.");

            uint hashCode;
            uint collisionCount = 0;
            IEqualityComparer<TKey>? comparer = _comparer;

            if (_buckets is null)
            {
                hashCode = (uint)(comparer?.GetHashCode(key) ?? key.GetHashCode());
                collisionCount = 0;
                goto ReturnNotFound;
            }

            int i = -1;
            ref Entry entry = ref Unsafe.NullRef<Entry>();

            Entry[]? entries = _entries;
            Debug.Assert(entries is not null, "expected entries to be is not null");

            if (typeof(TKey).IsValueType && // comparer can only be null for value types; enable JIT to eliminate entire if block for ref types
                comparer is null)
            {
                // ValueType: Devirtualize with EqualityComparer<TKey>.Default intrinsic

                hashCode = (uint)key.GetHashCode();
                i = GetBucket(hashCode) - 1; // Value in _buckets is 1-based; subtract 1 from i. We do it here so it fuses with the following conditional.
                do
                {
                    // Test in if to drop range check for following array access
                    if ((uint)i >= (uint)entries.Length)
                    {
                        goto ReturnNotFound;
                    }

                    entry = ref entries[i];
                    if (entry.HashCode == hashCode && EqualityComparer<TKey>.Default.Equals(entry.Key, key))
                    {
                        goto Return;
                    }

                    i = entry.Next;

                    collisionCount++;
                }
                while (collisionCount <= (uint)entries.Length);

                // The chain of entries forms a loop; which means a concurrent update has happened.
                // Break out of the loop and throw, rather than looping forever.
                goto ConcurrentOperation;
            }
            else
            {
                Debug.Assert(comparer is not null);
                hashCode = (uint)comparer.GetHashCode(key);
                i = GetBucket(hashCode) - 1; // Value in _buckets is 1-based; subtract 1 from i. We do it here so it fuses with the following conditional.
                do
                {
                    // Test in if to drop range check for following array access
                    if ((uint)i >= (uint)entries.Length)
                    {
                        goto ReturnNotFound;
                    }

                    entry = ref entries[i];
                    if (entry.HashCode == hashCode && comparer.Equals(entry.Key, key))
                    {
                        goto Return;
                    }

                    i = entry.Next;

                    collisionCount++;
                }
                while (collisionCount <= (uint)entries.Length);

                // The chain of entries forms a loop; which means a concurrent update has happened.
                // Break out of the loop and throw, rather than looping forever.
                goto ConcurrentOperation;
            }

        ReturnNotFound:
            i = -1;
            outCollisionCount = collisionCount;
            goto Return;

        ConcurrentOperation:
            // We examined more entries than are actually in the list, which means there's a cycle
            // that's caused by erroneous concurrent use.
            ThrowHelper.ThrowConcurrentOperation();

        Return:
            outHashCode = hashCode;
            return i;
        }

        /// <summary>Inserts an item into the collection at the specified index.</summary>
        /// <param name="index">The zero-based index at which item should be inserted.</param>
        /// <param name="key">The key to insert.</param>
        /// <param name="value">The value to insert.</param>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">An element with the same key already exists in the <see cref="OrderedDict{TKey, TValue}"/>.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is less than 0 or greater than <see cref="Count"/>.</exception>
        public void Insert(int index, TKey key, TValue value)
        {
            if ((uint)index > (uint)_count)
            {
                ThrowHelper.ThrowIndexArgumentOutOfRange();
            }

            ArgumentNullException.ThrowIfNull(key);

            TryInsert(index, key, value, InsertionBehavior.ThrowOnExisting, out _);
        }

        /// <summary>Removes the value with the specified key from the <see cref="OrderedDict{TKey, TValue}"/>.</summary>
        /// <param name="key">The key of the element to remove.</param>
        /// <returns><see langword="true"/> if the element is successfully found and removed; otherwise, <see langword="false"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <see langword="null"/>.</exception>
        public bool Remove(TKey key) => Remove(key, out _);

        /// <summary>Removes the value with the specified key from the <see cref="OrderedDict{TKey, TValue}"/> and copies the element to the value parameter.</summary>
        /// <param name="key">The key of the element to remove.</param>
        /// <param name="value">The removed element.</param>
        /// <returns><see langword="true"/> if the element is successfully found and removed; otherwise, <see langword="false"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <see langword="null"/>.</exception>
        public bool Remove(TKey key, [MaybeNullWhen(false)] out TValue value)
        {
            ArgumentNullException.ThrowIfNull(key);

            // Find the key.
            int index = IndexOf(key);
            if (index >= 0)
            {
                // It exists. Remove it.
                Debug.Assert(_entries is not null);

                value = _entries[index].Value;
                RemoveAt(index);

                return true;
            }

            value = default;
            return false;
        }

        /// <summary>Removes the key/value pair at the specified index.</summary>
        /// <param name="index">The zero-based index of the item to remove.</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is less than 0 or greater than or equal to <see cref="Count"/>.</exception>
        public void RemoveAt(int index)
        {
            int count = _count;
            if ((uint)index >= (uint)count)
            {
                ThrowHelper.ThrowIndexArgumentOutOfRange();
            }

            // Remove from the associated bucket chain the entry that lives at the specified index.
            RemoveEntryFromBucket(index);

            // Shift down all entries above this one, and fix up the bucket chains to reflect the new indices.
            Entry[]? entries = _entries;
            Debug.Assert(entries is not null);
            for (int i = index + 1; i < count; i++)
            {
                entries[i - 1] = entries[i];
                UpdateBucketIndex(i, shiftAmount: -1);
            }

            entries[--_count] = default;
            _version++;
        }

        /// <summary>Sets the value for the key at the specified index.</summary>
        /// <param name="index">The zero-based index at which to set the value.</param>
        /// <param name="value">The value to store at the specified index.</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is less than 0 or greater than or equal to <see cref="Count"/>.</exception>
        public void SetAt(int index, TValue value)
        {
            if ((uint)index >= (uint)_count)
            {
                ThrowHelper.ThrowIndexArgumentOutOfRange();
            }

            Debug.Assert(_entries is not null);

            _entries[index].Value = value;
        }

        /// <summary>Sets the key/value pair at the specified index.</summary>
        /// <param name="index">The zero-based index at which to set the key/value pair.</param>
        /// <param name="key">The key to store at the specified index.</param>
        /// <param name="value">The value to store at the specified index.</param>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">An element with the same key already exists at an index different to <paramref name="index"/>.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is less than 0 or greater than or equal to <see cref="Count"/>.</exception>
        public void SetAt(int index, TKey key, TValue value)
        {
            if ((uint)index >= (uint)_count)
            {
                ThrowHelper.ThrowIndexArgumentOutOfRange();
            }

            ArgumentNullException.ThrowIfNull(key);

            Debug.Assert(_entries is not null);
            ref Entry e = ref _entries[index];

            // If the key matches the one that's already in that slot, just update the value.
            if (typeof(TKey).IsValueType && _comparer is null)
            {
                if (EqualityComparer<TKey>.Default.Equals(key, e.Key))
                {
                    e.Value = value;
                    return;
                }
            }
            else
            {
                Debug.Assert(_comparer is not null);
                if (_comparer.Equals(key, e.Key))
                {
                    e.Value = value;
                    return;
                }
            }

            // The key doesn't match that index. If it exists elsewhere in the collection, fail.
            uint hashCode = 0, collisionCount = 0;
            if (IndexOf(key, ref hashCode, ref collisionCount) >= 0)
            {
                ThrowHelper.ThrowDuplicateKey(key);
            }

            // The key doesn't exist in the collection. Update the key and value, but also update
            // the bucket chains, as the new key may not hash to the same bucket as the old key
            // (we could check for this, but in a properly balanced dictionary the chances should
            // be low for a match, so it's not worth it).
            RemoveEntryFromBucket(index);
            e.HashCode = hashCode;
            e.Key = key;
            e.Value = value;
            PushEntryIntoBucket(ref e, index);

            _version++;

            RehashIfNecessary(collisionCount, _entries);
        }

        /// <summary>Ensures that the dictionary can hold up to <paramref name="capacity"/> entries without resizing.</summary>
        /// <param name="capacity">The desired minimum capacity of the dictionary. The actual capacity provided may be larger.</param>
        /// <returns>The new capacity of the dictionary.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="capacity"/> is negative.</exception>
        public int EnsureCapacity(int capacity)
        {
            ThrowIfNegative(capacity);

            if (Capacity < capacity)
            {
                if (_buckets is null)
                {
                    EnsureBucketsAndEntriesInitialized(capacity);
                }
                else
                {
                    Resize(HashHelpers.GetPrime(capacity));
                }

                _version++;
            }

            return Capacity;
        }

        /// <summary>Sets the capacity of this dictionary to what it would be if it had been originally initialized with all its entries.</summary>
        public void TrimExcess() => TrimExcess(_count);

        /// <summary>Sets the capacity of this dictionary to hold up a specified number of entries without resizing.</summary>
        /// <param name="capacity">The desired capacity to which to shrink the dictionary.</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="capacity"/> is less than <see cref="Count"/>.</exception>
        public void TrimExcess(int capacity)
        {
            ThrowIfLessThan(capacity, Count);

            int currentCapacity = _entries?.Length ?? 0;
            capacity = HashHelpers.GetPrime(capacity);
            if (capacity < currentCapacity)
            {
                Resize(capacity);
            }
        }

        /// <summary>Gets the value associated with the specified key.</summary>
        /// <param name="key">The key of the value to get.</param>
        /// <param name="value">
        /// When this method returns, contains the value associated with the specified key, if the key is found;
        /// otherwise, the default value for the type of the value parameter.
        /// </param>
        /// <returns><see langword="true"/> if the <see cref="OrderedDict{TKey, TValue}"/> contains an element with the specified key; otherwise, <see langword="false"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <see langword="null"/>.</exception>
        public bool TryGetValue(TKey key, [MaybeNullWhen(false)] out TValue value) => TryGetValue(key, out value, out _);

        /// <summary>Gets the value associated with the specified key.</summary>
        /// <param name="key">The key of the value to get.</param>
        /// <param name="value">
        /// When this method returns, contains the value associated with the specified key, if the key is found;
        /// otherwise, the default value for the type of the value parameter.
        /// </param>
        /// <param name="index">The index of <paramref name="key"/> if found; otherwise, -1.</param>
        /// <returns><see langword="true"/> if the <see cref="OrderedDict{TKey, TValue}"/> contains an element with the specified key; otherwise, <see langword="false"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <see langword="null"/>.</exception>
        public bool TryGetValue(TKey key, [MaybeNullWhen(false)] out TValue value, out int index)
        {
            ArgumentNullException.ThrowIfNull(key);

            // Find the key.
            index = IndexOf(key);
            if (index >= 0)
            {
                // It exists. Return its value.
                Debug.Assert(_entries is not null);
                value = _entries[index].Value;
                return true;
            }

            value = default;
            return false;
        }

        /// <summary>Pushes the entry into its bucket.</summary>
        /// <remarks>
        /// The bucket is a linked list by index into the <see cref="_entries"/> array.
        /// The new entry's <see cref="Entry.Next"/> is set to the bucket's current
        /// head, and then the new entry is made the new head.
        /// </remarks>
        private void PushEntryIntoBucket(ref Entry entry, int entryIndex)
        {
            ref int bucket = ref GetBucket(entry.HashCode);
            entry.Next = bucket - 1;
            bucket = entryIndex + 1;
        }

        /// <summary>Removes an entry from its bucket.</summary>
        private void RemoveEntryFromBucket(int entryIndex)
        {
            // We're only calling this method if there's an entry to be removed, in which case
            // entries must have been initialized.
            Entry[]? entries = _entries;
            Debug.Assert(entries is not null);

            // Get the entry to be removed and the associated bucket.
            Entry entry = entries[entryIndex];
            ref int bucket = ref GetBucket(entry.HashCode);

            if (bucket == entryIndex + 1)
            {
                // If the entry was at the head of its bucket list, to remove it from the list we
                // simply need to update the next entry in the list to be the new head.
                bucket = entry.Next + 1;
            }
            else
            {
                // The entry wasn't the head of the list. Walk the chain until we find the entry,
                // updating the previous entry's Next to point to this entry's Next.
                int i = bucket - 1;
                int collisionCount = 0;
                while (true)
                {
                    ref Entry e = ref entries[i];
                    if (e.Next == entryIndex)
                    {
                        e.Next = entry.Next;
                        return;
                    }

                    i = e.Next;

                    if (++collisionCount > entries.Length)
                    {
                        // We examined more entries than are actually in the list, which means there's a cycle
                        // that's caused by erroneous concurrent use.
                        ThrowHelper.ThrowConcurrentOperation();
                    }
                }
            }
        }

        /// <summary>
        /// Updates the bucket chain containing the specified entry (by index) to shift indices
        /// by the specified amount.
        /// </summary>
        /// <param name="entryIndex">The index of the target entry.</param>
        /// <param name="shiftAmount">
        /// 1 if this is part of an insert and the values are being shifted one higher.
        /// -1 if this is part of a remove and the values are being shifted one lower.
        /// </param>
        private void UpdateBucketIndex(int entryIndex, int shiftAmount)
        {
            Debug.Assert(shiftAmount is 1 or -1);

            Entry[]? entries = _entries;
            Debug.Assert(entries is not null);

            Entry entry = entries[entryIndex];
            ref int bucket = ref GetBucket(entry.HashCode);

            if (bucket == entryIndex + 1)
            {
                // If the entry was at the head of its bucket list, the only thing that needs to be updated
                // is the bucket head value itself, since no other entries' Next will be referencing this node.
                bucket += shiftAmount;
            }
            else
            {
                // The entry wasn't the head of the list. Walk the chain until we find the entry, updating
                // the previous entry's Next that's pointing to the target entry.
                int i = bucket - 1;
                int collisionCount = 0;
                while (true)
                {
                    ref Entry e = ref entries[i];
                    if (e.Next == entryIndex)
                    {
                        e.Next += shiftAmount;
                        return;
                    }

                    i = e.Next;

                    if (++collisionCount > entries.Length)
                    {
                        // We examined more entries than are actually in the list, which means there's a cycle
                        // that's caused by erroneous concurrent use.
                        ThrowHelper.ThrowConcurrentOperation();
                    }
                }
            }
        }

        /// <summary>
        /// Checks to see whether the collision count that occurred during lookup warrants upgrading to a non-randomized comparer,
        /// and does so if necessary.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [SuppressMessage("Performance", "CA1822:Mark members as static", Justification = "Is no-op in certain targets")]
        private void RehashIfNecessary(uint collisionCount, Entry[] entries)
        {
#if SYSTEM_COLLECTIONS
            // If we exceeded the hash collision threshold and we're using a randomized comparer, rehash.
            // This is only ever done for string keys, so we can optimize it all away for value type keys.
            if (!typeof(TKey).IsValueType &&
                collisionCount > HashHelpers.HashCollisionThreshold &&
                _comparer is NonRandomizedStringEqualityComparer)
            {
                // Switch to a randomized comparer and rehash.
                Resize(entries.Length, forceNewHashCodes: true);
            }
#endif
        }

        /// <summary>Grow or shrink <see cref="_buckets"/> and <see cref="_entries"/> to the specified capacity.</summary>
        [MemberNotNull(nameof(_buckets))]
        [MemberNotNull(nameof(_entries))]
        private void Resize(int newSize, bool forceNewHashCodes = false)
        {
            Debug.Assert(!forceNewHashCodes || !typeof(TKey).IsValueType, "Value types never rehash.");
            Debug.Assert(newSize >= _count, "The requested size must accomodate all of the current elements.");

            // Create the new arrays. We allocate both prior to storing either; in case one of the allocation fails,
            // we want to avoid corrupting the data structure.
            int[] newBuckets = new int[newSize];
            Entry[] newEntries = new Entry[newSize];
            if (IntPtr.Size == 8)
            {
                // Any time the capacity changes, that impacts the divisor of modulo operations,
                // and we need to update our fast modulo multiplier.
                _fastModMultiplier = HashHelpers.GetFastModMultiplier((uint)newSize);
            }

            // Copy the existing entries to the new entries array.
            int count = _count;
            if (_entries is not null)
            {
                Array.Copy(_entries, newEntries, count);
            }

#if SYSTEM_COLLECTIONS
            // If we're being asked to upgrade to a non-randomized comparer due to too many collisions, do so.
            if (!typeof(TKey).IsValueType && forceNewHashCodes)
            {
                // Store the original randomized comparer instead of the non-randomized one.
                Debug.Assert(_comparer is NonRandomizedStringEqualityComparer);
                IEqualityComparer<TKey> comparer = _comparer = (IEqualityComparer<TKey>)((NonRandomizedStringEqualityComparer)_comparer).GetUnderlyingEqualityComparer();
                Debug.Assert(_comparer is not null);
                Debug.Assert(_comparer is not NonRandomizedStringEqualityComparer);

                // Update all of the entries' hash codes based on the new comparer.
                for (int i = 0; i < count; i++)
                {
                    newEntries[i].HashCode = (uint)comparer.GetHashCode(newEntries[i].Key);
                }
            }
#endif

            // Now publish the buckets array. It's necessary to do this prior to the below loop,
            // as PushEntryIntoBucket will be populating _buckets.
            _buckets = newBuckets;

            // Populate the buckets.
            for (int i = 0; i < count; i++)
            {
                PushEntryIntoBucket(ref newEntries[i], i);
            }

            _entries = newEntries;
        }

        /// <summary>Gets the bucket assigned to the specified hash code.</summary>
        /// <remarks>
        /// Buckets are 1-based. This is so that the default initialized value of 0
        /// maps to -1 and is usable as a sentinel.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private ref int GetBucket(uint hashCode)
        {
            int[]? buckets = _buckets;
            Debug.Assert(buckets is not null);

            if (IntPtr.Size == 8)
            {
                return ref buckets[HashHelpers.FastMod(hashCode, (uint)buckets.Length, _fastModMultiplier)];
            }
            else
            {
                return ref buckets[(uint)hashCode % buckets.Length];
            }
        }

        /// <summary>Returns an enumerator that iterates through the <see cref="OrderedDict{TKey, TValue}"/>.</summary>
        /// <returns>A <see cref="OrderedDict{TKey, TValue}.Enumerator"/> structure for the <see cref="OrderedDict{TKey, TValue}"/>.</returns>
        public Enumerator GetEnumerator() => new(this, useDictionaryEntry: false);

        /// <inheritdoc/>
        IEnumerator<KeyValuePair<TKey, TValue>> IEnumerable<KeyValuePair<TKey, TValue>>.GetEnumerator() =>
            Count == 0 ? EnumerableHelpers.GetEmptyEnumerator<KeyValuePair<TKey, TValue>>() :
            GetEnumerator();

        /// <inheritdoc/>
        IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable<KeyValuePair<TKey, TValue>>)this).GetEnumerator();

        /// <inheritdoc/>
        IDictionaryEnumerator IDictionary.GetEnumerator() => new Enumerator(this, useDictionaryEntry: true);

        /// <inheritdoc/>
        int IList<KeyValuePair<TKey, TValue>>.IndexOf(KeyValuePair<TKey, TValue> item)
        {
            ArgumentNullException.ThrowIfNull(item.Key, nameof(item));

            int index = IndexOf(item.Key);
            if (index >= 0)
            {
                Debug.Assert(_entries is not null);
                if (EqualityComparer<TValue>.Default.Equals(item.Value, _entries[index].Value))
                {
                    return index;
                }
            }

            return -1;
        }

        /// <inheritdoc/>
        void IList<KeyValuePair<TKey, TValue>>.Insert(int index, KeyValuePair<TKey, TValue> item) => Insert(index, item.Key, item.Value);

        /// <inheritdoc/>
        void ICollection<KeyValuePair<TKey, TValue>>.Add(KeyValuePair<TKey, TValue> item) => Add(item.Key, item.Value);

        /// <inheritdoc/>
        bool ICollection<KeyValuePair<TKey, TValue>>.Contains(KeyValuePair<TKey, TValue> item)
        {
            ArgumentNullException.ThrowIfNull(item.Key, nameof(item));

            return
                TryGetValue(item.Key, out TValue? value) &&
                EqualityComparer<TValue>.Default.Equals(value, item.Value);
        }

        /// <inheritdoc/>
        void ICollection<KeyValuePair<TKey, TValue>>.CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
        {
            ArgumentNullException.ThrowIfNull(array);
            ThrowIfNegative(arrayIndex);
            if (array.Length - arrayIndex < _count)
            {
                throw new ArgumentException(SR.Arg_ArrayPlusOffTooSmall);
            }

            for (int i = 0; i < _count; i++)
            {
                ref Entry entry = ref _entries![i];
                array[arrayIndex++] = new(entry.Key, entry.Value);
            }
        }

        /// <inheritdoc/>
        bool ICollection<KeyValuePair<TKey, TValue>>.Remove(KeyValuePair<TKey, TValue> item) =>
            TryGetValue(item.Key, out TValue? value) &&
            EqualityComparer<TValue>.Default.Equals(value, item.Value) &&
            Remove(item.Key);

        /// <inheritdoc/>
        void IDictionary.Add(object key, object? value)
        {
            ArgumentNullException.ThrowIfNull(key);
            if (default(TValue) is not null)
            {
                ArgumentNullException.ThrowIfNull(value);
            }

            if (key is not TKey tkey)
            {
                throw new ArgumentException(SR.Format(SR.Arg_WrongType, key, typeof(TKey)), nameof(key));
            }

            if (default(TValue) is not null)
            {
                ArgumentNullException.ThrowIfNull(value);
            }

            TValue tvalue = default!;
            if (value is not null)
            {
                if (value is not TValue temp)
                {
                    throw new ArgumentException(SR.Format(SR.Arg_WrongType, value, typeof(TValue)), nameof(value));
                }

                tvalue = temp;
            }

            Add(tkey, tvalue);
        }

        /// <inheritdoc/>
        bool IDictionary.Contains(object key)
        {
            ArgumentNullException.ThrowIfNull(key);

            return key is TKey tkey && ContainsKey(tkey);
        }

        /// <inheritdoc/>
        void IDictionary.Remove(object key)
        {
            ArgumentNullException.ThrowIfNull(key);

            if (key is TKey tkey)
            {
                Remove(tkey);
            }
        }

        /// <inheritdoc/>
        void ICollection.CopyTo(Array array, int index)
        {
            ArgumentNullException.ThrowIfNull(array);

            if (array.Rank != 1)
            {
                throw new ArgumentException(SR.Arg_RankMultiDimNotSupported, nameof(array));
            }

            if (array.GetLowerBound(0) != 0)
            {
                throw new ArgumentException(SR.Arg_NonZeroLowerBound, nameof(array));
            }

            ThrowIfNegative(index);

            if (array.Length - index < _count)
            {
                throw new ArgumentException(SR.Arg_ArrayPlusOffTooSmall);
            }

            if (array is KeyValuePair<TKey, TValue>[] tarray)
            {
                ((ICollection<KeyValuePair<TKey, TValue>>)this).CopyTo(tarray, index);
            }
            else
            {
                try
                {
                    if (array is not object[] objects)
                    {
                        throw new ArgumentException(SR.Argument_IncompatibleArrayType, nameof(array));
                    }

                    foreach (KeyValuePair<TKey, TValue> pair in this)
                    {
                        objects[index++] = pair;
                    }
                }
                catch (ArrayTypeMismatchException)
                {
                    throw new ArgumentException(SR.Argument_IncompatibleArrayType, nameof(array));
                }
            }
        }

        /// <inheritdoc/>
        int IList.Add(object? value)
        {
            if (value is not KeyValuePair<TKey, TValue> pair)
            {
                throw new ArgumentException(SR.Format(SR.Arg_WrongType, value, typeof(KeyValuePair<TKey, TValue>)), nameof(value));
            }

            Add(pair.Key, pair.Value);
            return Count - 1;
        }

        /// <inheritdoc/>
        bool IList.Contains(object? value) =>
            value is KeyValuePair<TKey, TValue> pair &&
            TryGetValue(pair.Key, out TValue? v) &&
            EqualityComparer<TValue>.Default.Equals(v, pair.Value);

        /// <inheritdoc/>
        int IList.IndexOf(object? value)
        {
            if (value is KeyValuePair<TKey, TValue> pair)
            {
                return ((IList<KeyValuePair<TKey, TValue>>)this).IndexOf(pair);
            }

            return -1;
        }

        /// <inheritdoc/>
        void IList.Insert(int index, object? value)
        {
            if (value is not KeyValuePair<TKey, TValue> pair)
            {
                throw new ArgumentException(SR.Format(SR.Arg_WrongType, value, typeof(KeyValuePair<TKey, TValue>)), nameof(value));
            }

            Insert(index, pair.Key, pair.Value);
        }

        /// <inheritdoc/>
        void IList.Remove(object? value)
        {
            if (value is KeyValuePair<TKey, TValue> pair)
            {
                ((ICollection<KeyValuePair<TKey, TValue>>)this).Remove(pair);
            }
        }

        /// <summary>Represents a key/value pair in the dictionary.</summary>
        private struct Entry
        {
            /// <summary>The index of the next entry in the chain, or -1 if this is the last entry in the chain.</summary>
            public int Next;
            /// <summary>Cached hash code of <see cref="Key"/>.</summary>
            public uint HashCode;
            /// <summary>The key.</summary>
            public TKey Key;
            /// <summary>The value associated with <see cref="Key"/>.</summary>
            public TValue Value;
        }

        /// <summary>Enumerates the elements of a <see cref="OrderedDict{TKey, TValue}"/>.</summary>
        [StructLayout(LayoutKind.Auto)]
        public struct Enumerator : IEnumerator<KeyValuePair<TKey, TValue>>, IDictionaryEnumerator
        {
            /// <summary>The dictionary being enumerated.</summary>
            private readonly OrderedDict<TKey, TValue> _dictionary;
            /// <summary>A snapshot of the dictionary's version when enumeration began.</summary>
            private readonly int _version;
            /// <summary>Whether Current should be a DictionaryEntry.</summary>
            private readonly bool _useDictionaryEntry;
            /// <summary>The current index.</summary>
            private int _index;

            /// <summary>Initialize the enumerator.</summary>
            internal Enumerator(OrderedDict<TKey, TValue> dictionary, bool useDictionaryEntry)
            {
                _dictionary = dictionary;
                _version = _dictionary._version;
                _useDictionaryEntry = useDictionaryEntry;
            }

            /// <inheritdoc/>
            public KeyValuePair<TKey, TValue> Current { get; private set; }

            /// <inheritdoc/>
            readonly object IEnumerator.Current => _useDictionaryEntry ?
                new DictionaryEntry(Current.Key, Current.Value) :
                Current;

            /// <inheritdoc/>
            readonly DictionaryEntry IDictionaryEnumerator.Entry => new(Current.Key, Current.Value);

            /// <inheritdoc/>
            readonly object IDictionaryEnumerator.Key => Current.Key;

            /// <inheritdoc/>
            readonly object? IDictionaryEnumerator.Value => Current.Value;

            /// <inheritdoc/>
            public bool MoveNext()
            {
                OrderedDict<TKey, TValue> dictionary = _dictionary;

                if (_version != dictionary._version)
                {
                    ThrowHelper.ThrowVersionCheckFailed();
                }

                if (_index < dictionary._count)
                {
                    Debug.Assert(dictionary._entries is not null);
                    ref Entry entry = ref dictionary._entries[_index];
                    Current = new KeyValuePair<TKey, TValue>(entry.Key, entry.Value);
                    _index++;
                    return true;
                }

                Current = default;
                return false;
            }

            /// <inheritdoc/>
            void IEnumerator.Reset()
            {
                if (_version != _dictionary._version)
                {
                    ThrowHelper.ThrowVersionCheckFailed();
                }

                _index = 0;
                Current = default;
            }

            /// <inheritdoc/>
            readonly void IDisposable.Dispose() { }
        }

        /// <summary>Represents the collection of keys in a <see cref="OrderedDict{TKey, TValue}"/>.</summary>
        [DebuggerDisplay("Count = {Count}")]
        public sealed class KeyCollection : IList<TKey>, IReadOnlyList<TKey>, IList
        {
            /// <summary>The dictionary whose keys are being exposed.</summary>
            private readonly OrderedDict<TKey, TValue> _dictionary;

            /// <summary>Initialize the collection wrapper.</summary>
            internal KeyCollection(OrderedDict<TKey, TValue> dictionary) => _dictionary = dictionary;

            /// <inheritdoc/>
            public int Count => _dictionary.Count;

            /// <inheritdoc/>
            bool ICollection<TKey>.IsReadOnly => true;

            /// <inheritdoc/>
            bool IList.IsReadOnly => true;

            /// <inheritdoc/>
            bool IList.IsFixedSize => false;

            /// <inheritdoc/>
            bool ICollection.IsSynchronized => false;

            /// <inheritdoc/>
            object ICollection.SyncRoot => ((ICollection)_dictionary).SyncRoot;

            /// <inheritdoc/>
            public bool Contains(TKey key) => _dictionary.ContainsKey(key);

            /// <inheritdoc/>
            bool IList.Contains(object? value) => value is TKey key && Contains(key);

            /// <inheritdoc/>
            public void CopyTo(TKey[] array, int arrayIndex)
            {
                ArgumentNullException.ThrowIfNull(array);
                ThrowIfNegative(arrayIndex);

                OrderedDict<TKey, TValue> dictionary = _dictionary;
                int count = dictionary._count;

                if (array.Length - arrayIndex < count)
                {
                    throw new ArgumentException(SR.Arg_ArrayPlusOffTooSmall, nameof(array));
                }

                Entry[]? entries = dictionary._entries;
                for (int i = 0; i < count; i++)
                {
                    Debug.Assert(entries is not null);
                    array[arrayIndex++] = entries[i].Key;
                }
            }

            /// <inheritdoc/>
            void ICollection.CopyTo(Array array, int index)
            {
                ArgumentNullException.ThrowIfNull(array);

                if (array.Rank != 1)
                {
                    throw new ArgumentException(SR.Arg_RankMultiDimNotSupported, nameof(array));
                }

                if (array.GetLowerBound(0) != 0)
                {
                    throw new ArgumentException(SR.Arg_NonZeroLowerBound, nameof(array));
                }

                ThrowIfNegative(index);

                if (array.Length - index < _dictionary.Count)
                {
                    throw new ArgumentException(SR.Arg_ArrayPlusOffTooSmall);
                }

                if (array is TKey[] keys)
                {
                    CopyTo(keys, index);
                }
                else
                {
                    try
                    {
                        if (array is not object?[] objects)
                        {
                            throw new ArgumentException(SR.Argument_IncompatibleArrayType, nameof(array));
                        }

                        foreach (TKey key in this)
                        {
                            objects[index++] = key;
                        }
                    }
                    catch (ArrayTypeMismatchException)
                    {
                        throw new ArgumentException(SR.Argument_IncompatibleArrayType, nameof(array));
                    }
                }
            }

            /// <inheritdoc/>
            TKey IList<TKey>.this[int index]
            {
                get => _dictionary.GetAt(index).Key;
                set => throw new NotSupportedException();
            }

            /// <inheritdoc/>
            object? IList.this[int index]
            {
                get => _dictionary.GetAt(index).Key;
                set => throw new NotSupportedException();
            }

            /// <inheritdoc/>
            TKey IReadOnlyList<TKey>.this[int index] => _dictionary.GetAt(index).Key;

            /// <summary>Returns an enumerator that iterates through the <see cref="OrderedDict{TKey, TValue}.KeyCollection"/>.</summary>
            /// <returns>A <see cref="OrderedDict{TKey, TValue}.KeyCollection.Enumerator"/> for the <see cref="OrderedDict{TKey, TValue}.KeyCollection"/>.</returns>
            public Enumerator GetEnumerator() => new(_dictionary);

            /// <inheritdoc/>
            IEnumerator<TKey> IEnumerable<TKey>.GetEnumerator() =>
                Count == 0 ? EnumerableHelpers.GetEmptyEnumerator<TKey>() :
                GetEnumerator();

            /// <inheritdoc/>
            IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable<TKey>)this).GetEnumerator();

            /// <inheritdoc/>
            int IList<TKey>.IndexOf(TKey item) => _dictionary.IndexOf(item);

            /// <inheritdoc/>
            void ICollection<TKey>.Add(TKey item) => throw new NotSupportedException();

            /// <inheritdoc/>
            void ICollection<TKey>.Clear() => throw new NotSupportedException();

            /// <inheritdoc/>
            void IList<TKey>.Insert(int index, TKey item) => throw new NotSupportedException();

            /// <inheritdoc/>
            bool ICollection<TKey>.Remove(TKey item) => throw new NotSupportedException();

            /// <inheritdoc/>
            void IList<TKey>.RemoveAt(int index) => throw new NotSupportedException();

            /// <inheritdoc/>
            int IList.Add(object? value) => throw new NotSupportedException();

            /// <inheritdoc/>
            void IList.Clear() => throw new NotSupportedException();

            /// <inheritdoc/>
            int IList.IndexOf(object? value) => value is TKey key ? _dictionary.IndexOf(key) : -1;

            /// <inheritdoc/>
            void IList.Insert(int index, object? value) => throw new NotSupportedException();

            /// <inheritdoc/>
            void IList.Remove(object? value) => throw new NotSupportedException();

            /// <inheritdoc/>
            void IList.RemoveAt(int index) => throw new NotSupportedException();

            /// <summary>Enumerates the elements of a <see cref="OrderedDict{TKey, TValue}.KeyCollection"/>.</summary>
            public struct Enumerator : IEnumerator<TKey>
            {
                /// <summary>The dictionary's enumerator.</summary>
                private OrderedDict<TKey, TValue>.Enumerator _enumerator;

                /// <summary>Initialize the enumerator.</summary>
                internal Enumerator(OrderedDict<TKey, TValue> dictionary) => _enumerator = dictionary.GetEnumerator();

                /// <inheritdoc/>
                public TKey Current => _enumerator.Current.Key;

                /// <inheritdoc/>
                object IEnumerator.Current => Current;

                /// <inheritdoc/>
                public bool MoveNext() => _enumerator.MoveNext();

                /// <inheritdoc/>
                void IEnumerator.Reset() => EnumerableHelpers.Reset(ref _enumerator);

                /// <inheritdoc/>
                readonly void IDisposable.Dispose() { }
            }
        }

        /// <summary>Represents the collection of values in a <see cref="OrderedDict{TKey, TValue}"/>.</summary>
        [DebuggerDisplay("Count = {Count}")]
        public sealed class ValueCollection : IList<TValue>, IReadOnlyList<TValue>, IList
        {
            /// <summary>The dictionary whose values are being exposed.</summary>
            private readonly OrderedDict<TKey, TValue> _dictionary;

            /// <summary>Initialize the collection wrapper.</summary>
            internal ValueCollection(OrderedDict<TKey, TValue> dictionary) => _dictionary = dictionary;

            /// <inheritdoc/>
            public int Count => _dictionary.Count;

            /// <inheritdoc/>
            bool ICollection<TValue>.IsReadOnly => true;

            /// <inheritdoc/>
            bool IList.IsReadOnly => true;

            /// <inheritdoc/>
            bool IList.IsFixedSize => false;

            /// <inheritdoc/>
            bool ICollection.IsSynchronized => false;

            /// <inheritdoc/>
            object ICollection.SyncRoot => ((ICollection)_dictionary).SyncRoot;

            /// <inheritdoc/>
            public void CopyTo(TValue[] array, int arrayIndex)
            {
                ArgumentNullException.ThrowIfNull(array);
                ThrowIfNegative(arrayIndex);

                OrderedDict<TKey, TValue> dictionary = _dictionary;
                int count = dictionary._count;

                if (array.Length - arrayIndex < count)
                {
                    throw new ArgumentException(SR.Arg_ArrayPlusOffTooSmall, nameof(array));
                }

                Entry[]? entries = dictionary._entries;
                for (int i = 0; i < count; i++)
                {
                    Debug.Assert(entries is not null);
                    array[arrayIndex++] = entries[i].Value;
                }
            }

            /// <summary>Returns an enumerator that iterates through the <see cref="OrderedDict{TKey, TValue}.ValueCollection"/>.</summary>
            /// <returns>A <see cref="OrderedDict{TKey, TValue}.ValueCollection.Enumerator"/> for the <see cref="OrderedDict{TKey, TValue}.ValueCollection"/>.</returns>
            public Enumerator GetEnumerator() => new(_dictionary);

            /// <inheritdoc/>
            TValue IList<TValue>.this[int index]
            {
                get => _dictionary.GetAt(index).Value;
                set => throw new NotSupportedException();
            }

            /// <inheritdoc/>
            TValue IReadOnlyList<TValue>.this[int index] => _dictionary.GetAt(index).Value;

            /// <inheritdoc/>
            object? IList.this[int index]
            {
                get => _dictionary.GetAt(index).Value;
                set => throw new NotSupportedException();
            }

            /// <inheritdoc/>
            bool ICollection<TValue>.Contains(TValue item) => _dictionary.ContainsValue(item);

            /// <inheritdoc/>
            IEnumerator<TValue> IEnumerable<TValue>.GetEnumerator() =>
                Count == 0 ? EnumerableHelpers.GetEmptyEnumerator<TValue>() :
                GetEnumerator();

            /// <inheritdoc/>
            IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable<TValue>)this).GetEnumerator();

            /// <inheritdoc/>
            int IList<TValue>.IndexOf(TValue item)
            {
                Entry[]? entries = _dictionary._entries;
                if (entries is not null)
                {
                    int count = _dictionary._count;
                    for (int i = 0; i < count; i++)
                    {
                        if (EqualityComparer<TValue>.Default.Equals(item, entries[i].Value))
                        {
                            return i;
                        }
                    }
                }

                return -1;
            }

            /// <inheritdoc/>
            void ICollection<TValue>.Add(TValue item) => throw new NotSupportedException();

            /// <inheritdoc/>
            void ICollection<TValue>.Clear() => throw new NotSupportedException();

            /// <inheritdoc/>
            void IList<TValue>.Insert(int index, TValue item) => throw new NotSupportedException();

            /// <inheritdoc/>
            bool ICollection<TValue>.Remove(TValue item) => throw new NotSupportedException();

            /// <inheritdoc/>
            void IList<TValue>.RemoveAt(int index) => throw new NotSupportedException();

            /// <inheritdoc/>
            int IList.Add(object? value) => throw new NotSupportedException();

            /// <inheritdoc/>
            void IList.Clear() => throw new NotSupportedException();

            /// <inheritdoc/>
            bool IList.Contains(object? value) =>
                value is null && default(TValue) is null ?
                    _dictionary.ContainsValue(default!) :
                    value is TValue tvalue && _dictionary.ContainsValue(tvalue);

            /// <inheritdoc/>
            int IList.IndexOf(object? value)
            {
                Entry[]? entries = _dictionary._entries;
                if (entries is not null)
                {
                    int count = _dictionary._count;

                    if (value is null && default(TValue) is null)
                    {
                        for (int i = 0; i < count; i++)
                        {
                            if (entries[i].Value is null)
                            {
                                return i;
                            }
                        }
                    }
                    else if (value is TValue tvalue)
                    {
                        for (int i = 0; i < count; i++)
                        {
                            if (EqualityComparer<TValue>.Default.Equals(tvalue, entries[i].Value))
                            {
                                return i;
                            }
                        }
                    }
                }

                return -1;
            }

            /// <inheritdoc/>
            void IList.Insert(int index, object? value) => throw new NotSupportedException();

            /// <inheritdoc/>
            void IList.Remove(object? value) => throw new NotSupportedException();

            /// <inheritdoc/>
            void IList.RemoveAt(int index) => throw new NotSupportedException();

            /// <inheritdoc/>
            void ICollection.CopyTo(Array array, int index)
            {
                ArgumentNullException.ThrowIfNull(array);

                if (array.Rank != 1)
                {
                    throw new ArgumentException(SR.Arg_RankMultiDimNotSupported, nameof(array));
                }

                if (array.GetLowerBound(0) != 0)
                {
                    throw new ArgumentException(SR.Arg_NonZeroLowerBound, nameof(array));
                }

                ThrowIfNegative(index);

                if (array.Length - index < _dictionary.Count)
                {
                    throw new ArgumentException(SR.Arg_ArrayPlusOffTooSmall);
                }

                if (array is TValue[] values)
                {
                    CopyTo(values, index);
                }
                else
                {
                    try
                    {
                        if (array is not object?[] objects)
                        {
                            throw new ArgumentException(SR.Argument_IncompatibleArrayType, nameof(array));
                        }

                        foreach (TValue value in this)
                        {
                            objects[index++] = value;
                        }
                    }
                    catch (ArrayTypeMismatchException)
                    {
                        throw new ArgumentException(SR.Argument_IncompatibleArrayType, nameof(array));
                    }
                }
            }

            /// <summary>Enumerates the elements of a <see cref="OrderedDict{TKey, TValue}.ValueCollection"/>.</summary>
            public struct Enumerator : IEnumerator<TValue>
            {
                /// <summary>The dictionary's enumerator.</summary>
                private OrderedDict<TKey, TValue>.Enumerator _enumerator;

                /// <summary>Initialize the enumerator.</summary>
                internal Enumerator(OrderedDict<TKey, TValue> dictionary) => _enumerator = dictionary.GetEnumerator();

                /// <inheritdoc/>
                public TValue Current => _enumerator.Current.Value;

                /// <inheritdoc/>
                object? IEnumerator.Current => Current;

                /// <inheritdoc/>
                public bool MoveNext() => _enumerator.MoveNext();

                /// <inheritdoc/>
                void IEnumerator.Reset() => EnumerableHelpers.Reset(ref _enumerator);

                /// <inheritdoc/>
                readonly void IDisposable.Dispose() { }
            }
        }
    }

    /// <summary>Used to control behavior of insertion into a <see cref="OrderedDict{TKey, TValue}"/>.</summary>
    /// <remarks>Not nested in <see cref="OrderedDict{TKey, TValue}"/> to avoid multiple generic instantiations.</remarks>
    internal enum InsertionBehavior
    {
        /// <summary>Skip the insertion operation.</summary>
        IgnoreInsertion = 0,

        /// <summary>Specifies that an existing entry with the same key should be overwritten if encountered.</summary>
        OverwriteExisting = 1,

        /// <summary>Specifies that if an existing entry with the same key is encountered, an exception should be thrown.</summary>
        ThrowOnExisting = 2
    }
}

#endif
```


<a id="ordered-dict"></a>

## src/OptunaSharp.Core/Collections/OrderedDict.cs

Ordered collections · 46 physical lines · SHA-256: `8d9282c93437c53f24e9902355e39e8dbb8d697f38c4f13a6b5828c1ef34c7eb`

```csharp
#if NET9_0_OR_GREATER
using System.Collections.Generic;

namespace OptunaSharp.Collections;

/// <summary>
///     An insertion-order-preserving dictionary with Python <c>dict</c> semantics.
///
///     Optuna relies on Python dicts preserving insertion order in many load-bearing places —
///     search-space construction, parameter/distribution maps, transform bounds layout, JSON
///     attribute serialization — so OptunaSharp uses this collection wherever Optuna uses
///     <c>dict</c>. <see cref="System.Collections.Generic.Dictionary{TKey,TValue}" /> does not
///     guarantee enumeration order, which would silently break parity.
///
///     On .NET 9+ this is a thin subclass of the in-box
///     <see cref="System.Collections.Generic.OrderedDictionary{TKey,TValue}" /> (added in .NET 9),
///     inheriting its entries-array + hash-bucket implementation. On net8.0 — where the in-box type
///     does not exist — <c>OrderedDict</c> is instead the vendored dotnet/runtime source of that same
///     type (see <c>OrderedDict.Net8.cs</c>), so the behavior is identical across target frameworks.
///     The shared members (<see cref="GetValueOrDefault" />, <see cref="ShallowClone" />) live
///     alongside on both frameworks (see <c>OrderedDict.Custom.cs</c>).
/// </summary>
public sealed partial class OrderedDict<TKey, TValue> : System.Collections.Generic.OrderedDictionary<TKey, TValue>
    where TKey : notnull
{
    public OrderedDict()
    {
    }

    public OrderedDict(int capacity) : base(capacity)
    {
    }

    /// <summary>Shallow-copies another dictionary preserving its enumeration order (Python <c>copy.copy(dict)</c>).</summary>
    /// <remarks>
    ///     Populates via the indexer so duplicate keys follow Python <c>dict(pairs)</c> semantics
    ///     (last value wins, first-occurrence position); the in-box <c>IEnumerable</c> constructor
    ///     throws on duplicate keys instead.
    /// </remarks>
    public OrderedDict(IEnumerable<KeyValuePair<TKey, TValue>> source)
    {
        foreach (var kv in source)
            this[kv.Key] = kv.Value;
    }
}
#endif
```


<a id="ordered-dict-internal-net8"></a>

## src/OptunaSharp.Core/Collections/OrderedDictInternal.Net8.cs

Ordered collections · 164 physical lines · SHA-256: `96fa65d5d8fa90dbd6d42867a0e0ca17e10af89424901838674cb70f5e365a2b`

```csharp
// <auto-generated>
// net8.0 support types for the vendored OrderedDict (see OrderedDict.Net8.cs).
// Minimal, faithful extracts of the BCL internals that System.Collections.Generic.OrderedDictionary
// depends on, taken from dotnet/runtime @ release/10.0 (MIT, same license as OptunaSharp):
//   - HashHelpers   : src/libraries/System.Private.CoreLib/src/System/Collections/HashHelpers.cs
//   - ThrowHelper   : src/libraries/System.Collections/src/System/Collections/ThrowHelper.cs
//   - EnumerableHelpers.{Reset,GetEmptyEnumerator} : src/libraries/Common/src/System/Collections/Generic/EnumerableHelpers.cs
//   - SR strings    : the corresponding resx entries.
// These are internal and compile only on net8.0 (where the in-box OrderedDictionary is absent).
// </auto-generated>
#if !NET9_0_OR_GREATER
#nullable enable

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace OptunaSharp.Collections
{
    /// <summary>Prime-table + fast-mod helpers (verbatim from BCL System.Collections.HashHelpers).</summary>
    internal static class HashHelpers
    {
        public const uint HashCollisionThreshold = 100;

        // This is the maximum prime smaller than Array.MaxLength.
        public const int MaxPrimeArrayLength = 0x7FFFFFC3;

        public const int HashPrime = 101;

        // Table of prime numbers to use as hash table sizes.
        internal static ReadOnlySpan<int> Primes =>
        [
            3, 7, 11, 17, 23, 29, 37, 47, 59, 71, 89, 107, 131, 163, 197, 239, 293, 353, 431, 521, 631, 761, 919,
            1103, 1327, 1597, 1931, 2333, 2801, 3371, 4049, 4861, 5839, 7013, 8419, 10103, 12143, 14591,
            17519, 21023, 25229, 30293, 36353, 43627, 52361, 62851, 75431, 90523, 108631, 130363, 156437,
            187751, 225307, 270371, 324449, 389357, 467237, 560689, 672827, 807403, 968897, 1162687, 1395263,
            1674319, 2009191, 2411033, 2893249, 3471899, 4166287, 4999559, 5999471, 7199369
        ];

        public static bool IsPrime(int candidate)
        {
            if ((candidate & 1) != 0)
            {
                int limit = (int)Math.Sqrt(candidate);
                for (int divisor = 3; divisor <= limit; divisor += 2)
                {
                    if ((candidate % divisor) == 0)
                        return false;
                }
                return true;
            }
            return candidate == 2;
        }

        public static int GetPrime(int min)
        {
            if (min < 0)
                throw new ArgumentException(SR.Arg_HTCapacityOverflow);

            foreach (int prime in Primes)
            {
                if (prime >= min)
                    return prime;
            }

            // Outside of our predefined table. Compute the hard way.
            for (int i = (min | 1); i < int.MaxValue; i += 2)
            {
                if (IsPrime(i) && ((i - 1) % HashPrime != 0))
                    return i;
            }
            return min;
        }

        // Returns size of hashtable to grow to.
        public static int ExpandPrime(int oldSize)
        {
            int newSize = 2 * oldSize;

            // Allow the hashtables to grow to maximum possible size (~2G elements) before encountering capacity overflow.
            if ((uint)newSize > MaxPrimeArrayLength && MaxPrimeArrayLength > oldSize)
            {
                Debug.Assert(MaxPrimeArrayLength == GetPrime(MaxPrimeArrayLength), "Invalid MaxPrimeArrayLength");
                return MaxPrimeArrayLength;
            }

            return GetPrime(newSize);
        }

        /// <summary>Returns approximate reciprocal of the divisor: ceil(2**64 / divisor).</summary>
        /// <remarks>This should only be used on 64-bit.</remarks>
        public static ulong GetFastModMultiplier(uint divisor) =>
            ulong.MaxValue / divisor + 1;

        /// <summary>Performs a mod operation using the multiplier pre-computed with <see cref="GetFastModMultiplier"/>.</summary>
        /// <remarks>This should only be used on 64-bit.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint FastMod(uint value, uint divisor, ulong multiplier)
        {
            Debug.Assert(divisor <= int.MaxValue);
            uint highbits = (uint)(((((multiplier * value) >> 32) + 1) * divisor) >> 32);
            Debug.Assert(highbits == value % divisor);
            return highbits;
        }
    }

    /// <summary>The throw helpers OrderedDict relies on (from BCL System.Collections.ThrowHelper).</summary>
    internal static class ThrowHelper
    {
        [DoesNotReturn]
        internal static void ThrowKeyNotFound<TKey>(TKey key) =>
            throw new KeyNotFoundException(SR.Format(SR.Arg_KeyNotFoundWithKey, key));

        [DoesNotReturn]
        internal static void ThrowDuplicateKey<TKey>(TKey key) =>
            throw new ArgumentException(SR.Format(SR.Argument_AddingDuplicate, key), nameof(key));

        [DoesNotReturn]
        internal static void ThrowConcurrentOperation() =>
            throw new InvalidOperationException(SR.InvalidOperation_ConcurrentOperationsNotSupported);

        [DoesNotReturn]
        internal static void ThrowIndexArgumentOutOfRange() =>
            throw new ArgumentOutOfRangeException("index");

        [DoesNotReturn]
        internal static void ThrowVersionCheckFailed() =>
            throw new InvalidOperationException(SR.InvalidOperation_EnumFailedVersion);
    }

    /// <summary>The two EnumerableHelpers members OrderedDict's enumerators use (from BCL Common).</summary>
    internal static class EnumerableHelpers
    {
        internal static void Reset<T>(ref T enumerator) where T : IEnumerator => enumerator.Reset();

        internal static IEnumerator<T> GetEmptyEnumerator<T>() =>
            ((IEnumerable<T>)Array.Empty<T>()).GetEnumerator();
    }

    /// <summary>The exact BCL resource strings referenced by the vendored code.</summary>
    internal static class SR
    {
        internal const string Arg_KeyNotFoundWithKey = "The given key '{0}' was not present in the dictionary.";
        internal const string Argument_AddingDuplicate = "An item with the same key has already been added. Key: {0}";
        internal const string InvalidOperation_ConcurrentOperationsNotSupported = "Operations that change non-concurrent collections must have exclusive access. A concurrent update was performed on this collection and corrupted its state. The collection's state is no longer correct.";
        internal const string InvalidOperation_EnumFailedVersion = "Collection was modified; enumeration operation may not execute.";
        internal const string Arg_WrongType = "The value \"{0}\" is not of type \"{1}\" and cannot be used in this generic collection.";
        internal const string Arg_ArrayPlusOffTooSmall = "Destination array is not long enough to copy all the items in the collection. Check array index and length.";
        internal const string Arg_NonZeroLowerBound = "The lower bound of target array must be zero.";
        internal const string Arg_RankMultiDimNotSupported = "Only single dimensional arrays are supported for the requested action.";
        internal const string Argument_IncompatibleArrayType = "Target array type is not compatible with the type of items in the collection.";
        internal const string Arg_HTCapacityOverflow = "Hashtable's capacity overflowed and went negative. Check load factor, capacity and the current size of the table.";

        internal static string Format(string resourceFormat, object? p1) =>
            string.Format(resourceFormat, p1);

        internal static string Format(string resourceFormat, object? p1, object? p2) =>
            string.Format(resourceFormat, p1, p2);
    }
}
#endif
```
