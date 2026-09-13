// AGE-MOEA-2-CPT — AGE-MOEA-2 environmental selection + the "Categorical Parameter Trees" (CPT /
// DCPARA) conditional-parameter machinery. The AGE-MOEA-2 counterpart of NSGAIIICptSampler.
//
// The design fact this file leans on: CPT is entirely a *child-generation / search-space* concern
// (which parameters stay in the relative space, how they are crossed / mutated / inherited when only
// some parents carry them) and is HOST-AGNOSTIC — it knows nothing about survival. The host is only the
// elite-selection strategy. So AGE-MOEA-2-CPT = AGE-MOEA-2 survival (`AgeMoea2ElitePopulationSelectionStrategy`)
// wired to the SAME reused CPT seams NSGA-III-CPT uses (`NsgaiiiCptSearchSpace`,
// `NsgaiiiCptChildGenerationStrategy`, `ConditionalParams`). Nothing about the conditional machinery is
// re-implemented here — this sampler is `NSGAIIICptSampler` with the elite strategy and operator
// defaults swapped, so the two share one implementation of conditional child generation.
//
// Why a standalone class and not `conditionalParams` re-added to `AgeMoea2Sampler`: the 2026-09-13
// consolidation made the CPT feature the sole responsibility of the `NsgaiiiCpt` family and reverted
// every other GA sampler (including `AgeMoea2Sampler`) to a bit-exact Optuna/pymoo-parity port that no
// longer accepts rules. This class keeps that boundary: `AgeMoea2Sampler` stays a pure port; the
// conditional variant lives beside it, exactly as `NSGAIIICptSampler` lives beside `NSGAIIISampler`.

using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers.AgeMoea;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.Samplers.NsgaiiiCpt;

namespace OptunaSharp.Samplers;

/// <summary>
///     AGE-MOEA-2-CPT ("AGE-MOEA-2 with <b>C</b>ategorical <b>P</b>arameter <b>T</b>rees") — AGE-MOEA-2's
///     geometry-based environmental selection (<see cref="AgeMoea2ElitePopulationSelectionStrategy" />)
///     joined to the conditional-parameter machinery <see cref="NSGAIIICptSampler" /> introduced, so the
///     define-by-run, branch-defined hyperparameters Rock's gene trees produce are evolved instead of
///     resampled at random. It is the AGE-MOEA-2 sibling of <see cref="NSGAIIICptSampler" /> and reuses the
///     <i>same</i> CPT seams (<see cref="NsgaiiiCptSearchSpace" />, <see cref="NsgaiiiCptChildGenerationStrategy" />,
///     <see cref="ConditionalParams" />); only the survival rule and the default operators differ.
///
///     <para><b>Why reach for this over <see cref="NSGAIIICptSampler" />.</b> Both handle conditional
///     parameters identically. They differ only in <i>survival</i>: NSGA-III niches against a fixed
///     reference-point simplex whose size grows as <c>C(M+2, 3)</c>, so past ~5 objectives most of the
///     population is mutually non-dominated and the front stops selecting; AGE-MOEA-2 estimates the front's
///     geometry adaptively and keeps selection pressure in that regime. So this sampler is the one to use
///     when the objective count is (or may grow) high; at 2–4 objectives the two are close and NSGA-III-CPT
///     remains the conservative default. This is a genuine tradeoff, not a strict improvement — validate
///     the swap on the actual objective set before adopting it.</para>
///
///     <para><b>Relationship to <see cref="AgeMoea2Sampler" />.</b> Constructed with
///     <c>conditionalParams: null</c> this sampler is behaviorally identical to <see cref="AgeMoea2Sampler" />
///     (the CPT search space and child generation both fall through to the shared Optuna-parity fold, and the
///     elite strategy, operators, and RNG draw order are the same), so it is a strict superset. The
///     conditional behavior is the same deliberate, no-Optuna-analog divergence documented on
///     <see cref="ConditionalParams" />; AGE-MOEA-2 itself (Panichella, GECCO 2022) is a beyond-core,
///     convergence-validated sampler with no Optuna counterpart, so no <c>ExperimentalWarning</c> is emitted.
///     <b>Unconstrained multi-objective only</b> — AGE-MOEA-2's survival does not model constraints, so unlike
///     <see cref="NSGAIIICptSampler" /> this class exposes no <c>constraintsFunc</c>.</para>
/// </summary>
/// <seealso cref="AgeMoea2Sampler" />
/// <seealso cref="NSGAIIICptSampler" />
/// <seealso cref="ConditionalParams" />
public class AgeMoea2CptSampler : BaseGASampler
{
    private readonly RandomSampler _randomSampler;
    private readonly LazyRandomState _rng;
    private readonly NsgaiiiCptSearchSpace _searchSpace;
    private readonly ConditionalParams? _conditionalParams;

    private readonly Func<Study, List<FrozenTrial>, List<FrozenTrial>> _elitePopulationSelectionStrategy;
    private readonly Func<Study, OrderedDict<string, BaseDistribution>, List<FrozenTrial>, OrderedDict<string, object?>>
        _childGenerationStrategy;
    private readonly Action<Study, FrozenTrial, TrialState, IReadOnlyList<double>?> _afterTrialStrategy;

    /// <param name="populationSize">
    ///     Number of individuals (trials) per generation. Must be &gt;= 2 and &gt;= <c>crossover.NParents</c>.
    /// </param>
    /// <param name="mutation">
    ///     Mutation applied when creating children. <c>null</c> selects AGE-MOEA-2's default
    ///     <see cref="Nsgaii.PolynomialMutation" /> with <c>eta = 20</c> (NOT NSGA-III's uniform-resample
    ///     default) — kept so this sampler matches <see cref="AgeMoea2Sampler" /> operator-for-operator.
    /// </param>
    /// <param name="mutationProb">
    ///     Per-parameter mutation probability. <c>null</c> uses <c>1 / n_active_params</c> — adaptive to each
    ///     child's active-gene count, which is what makes it well-behaved for the fluctuating genome sizes
    ///     conditional parameters produce.
    /// </param>
    /// <param name="crossover">
    ///     Crossover applied when creating children. <c>null</c> selects AGE-MOEA-2's default
    ///     <see cref="SBXCrossover" /> with <c>eta = 15</c>. Categorical parameters always cross uniformly.
    /// </param>
    /// <param name="crossoverProb">Probability that a crossover occurs (else a parent is copied).</param>
    /// <param name="swappingProb">Probability of swapping each parameter of the parents during crossover.</param>
    /// <param name="seed">Seed for the random number generator; the same seed replays the search.</param>
    /// <param name="elitePopulationSelectionStrategy">
    ///     Overrides environmental selection. <c>null</c> uses <see cref="AgeMoea2ElitePopulationSelectionStrategy" />
    ///     (the AGE-MOEA-2 survival that is the reason to choose this sampler); an override replaces exactly that
    ///     and leaves conditional child generation intact.
    /// </param>
    /// <param name="childGenerationStrategy">
    ///     Overrides child generation. A custom strategy receives the (conditional-widened) search space and must
    ///     itself cope with parents that lack conditional parameters; the built-in
    ///     <see cref="NsgaiiiCptChildGenerationStrategy" /> is the one that honors <paramref name="conditionalParams" />,
    ///     so an override drops conditional handling unless it reproduces it.
    /// </param>
    /// <param name="afterTrialStrategy">Overrides the after-trial procedure (population bookkeeping).</param>
    /// <param name="conditionalParams">
    ///     The activation rules for parameters that only exist under some values of other parameters — the reason
    ///     to reach for this sampler over <see cref="AgeMoea2Sampler" />. Declared parameters are evolved
    ///     (crossed over, mutated, inherited) whenever active instead of falling out of the intersection search
    ///     space and being resampled at random on every trial. <c>null</c> makes this sampler behave exactly like
    ///     <see cref="AgeMoea2Sampler" />.
    /// </param>
    /// <summary>Configures AGE-MOEA-2 survival, conditional-aware reproduction, and trial-completion hooks.</summary>
    /// <exception cref="NumSharp.ValueError">Population size is below two or the crossover parent count, or a configured reproduction probability is invalid.</exception>
    public AgeMoea2CptSampler(
        int populationSize = 50,
        BaseMutation? mutation = null,
        double? mutationProb = null,
        BaseCrossover? crossover = null,
        double crossoverProb = 0.9,
        double swappingProb = 0.5,
        int? seed = null,
        Func<Study, List<FrozenTrial>, List<FrozenTrial>>? elitePopulationSelectionStrategy = null,
        Func<Study, OrderedDict<string, BaseDistribution>, List<FrozenTrial>, OrderedDict<string, object?>>?
            childGenerationStrategy = null,
        Action<Study, FrozenTrial, TrialState, IReadOnlyList<double>?>? afterTrialStrategy = null,
        ConditionalParams? conditionalParams = null)
        : base(populationSize)
    {
        if (populationSize < 2)
            throw new NumSharp.ValueError("`population_size` must be greater than or equal to 2.");

        // AGE-MOEA-2's own operator defaults (PM eta=20, SBX eta=15) — deliberately NOT NSGA-III-CPT's
        // uniform-resample default, so `conditionalParams: null` is bit-identical to `AgeMoea2Sampler`.
        mutation ??= new Nsgaii.PolynomialMutation(eta: 20.0);
        crossover ??= new SBXCrossover(eta: 15.0);

        if (populationSize < crossover.NParents)
            throw new NumSharp.ValueError(
                $"Using {crossover}," +
                $" the population size should be greater than or equal to {crossover.NParents}." +
                $" The specified `population_size` is {populationSize}.");

        _randomSampler = new RandomSampler(seed);
        _rng = new LazyRandomState(seed);
        _searchSpace = new NsgaiiiCptSearchSpace(conditionalParams);
        _conditionalParams = conditionalParams;

        _elitePopulationSelectionStrategy = elitePopulationSelectionStrategy
            ?? new AgeMoea2ElitePopulationSelectionStrategy(populationSize).Select;
        // The SAME conditional child-generation seam NSGA-III-CPT uses (no constraints — AGE-MOEA-2 is
        // unconstrained). With null rules it delegates to the Optuna-parity crossover/mutation, so the
        // draw order matches `AgeMoea2Sampler`'s `NSGAIIChildGenerationStrategy` exactly.
        _childGenerationStrategy = childGenerationStrategy
            ?? new NsgaiiiCptChildGenerationStrategy(
                mutation, mutationProb, crossover, crossoverProb, swappingProb, null, _rng,
                conditionalParams).Generate;
        _afterTrialStrategy = afterTrialStrategy ?? new NSGAIIAfterTrialStrategy(null).Call;
    }

    /// <inheritdoc />
    public override void ReseedRng()
    {
        _randomSampler.ReseedRng();
        // Python: self._rng.rng.seed() — reseed from OS entropy so parallel workers diverge.
        Span<byte> buffer = stackalloc byte[4];
        RandomNumberGenerator.Fill(buffer);
        _rng.Rng.seed(BitConverter.ToUInt32(buffer));
    }

    /// <inheritdoc />
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _searchSpace.Calculate(study))
        {
            // Single-valued distributions are sampled inside Trial, not relatively, exactly as in
            // NSGAIIICptSampler / AgeMoea2Sampler — so this sampler agrees with them on what is relative.
            if (distribution.Single())
                continue;
            searchSpace[name] = distribution;
        }
        return searchSpace;
    }

    /// <inheritdoc />
    public override List<FrozenTrial> SelectParent(Study study, int generation)
    {
        var combined = GetPopulation(study, generation - 1);
        combined.AddRange(GetParentPopulation(study, generation - 1));
        return _elitePopulationSelectionStrategy(study, combined);
    }

    /// <inheritdoc />
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        int generation = GetTrialGeneration(study, trial);
        var parentPopulation = GetParentPopulation(study, generation);
        if (parentPopulation.Count == 0)
            return new OrderedDict<string, object?>();
        var child = _childGenerationStrategy(study, searchSpace, parentPopulation);
        // Off unless ConditionalParams.PreserveDormantValues is set. Parks inactive conditional values out of
        // the equations and resumes reactivating ones from their latent value instead of a random draw; no-op
        // otherwise. Identical to NSGAIIICptSampler — conditional handling is shared code.
        _conditionalParams?.ApplyDormantCarry(study, trial, child, parentPopulation, searchSpace);
        return child;
    }

    /// <inheritdoc />
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        // Randomly sampled here: (1) a parameter in the initial/first generation, (2) a parameter to
        // mutate, (3) a parameter excluded from the intersection search space.
        return _randomSampler.SampleIndependent(study, trial, paramName, paramDistribution);
    }

    /// <inheritdoc />
    public override void BeforeTrial(Study study, FrozenTrial trial)
    {
        _randomSampler.BeforeTrial(study, trial);
    }

    /// <inheritdoc />
    /// <exception cref="System.Exception">A configured after-trial callback fails; its exception propagates.</exception>
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        System.Diagnostics.Debug.Assert(
            state is TrialState.Complete or TrialState.Fail or TrialState.Pruned);
        _afterTrialStrategy(study, trial, state, values);
        _randomSampler.AfterTrial(study, trial, state, values);
    }
}
