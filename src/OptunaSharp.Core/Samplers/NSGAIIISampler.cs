using System;
using System.Collections.Generic;
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
///     <c>rng.shuffle</c>), and objective-value niching is bit-exact: the hyperplane-fit
///     <c>matrix_rank</c>/<c>solve</c> route through NumSharp's OpenBLAS LAPACK backend (byte-identical
///     to NumPy), so the earlier ~1-ULP managed-LU drift is gone. Confirmed byte-for-byte over a
///     400-config differential plus the live parity seeds — see
///     <see cref="NSGAIIIElitePopulationSelectionStrategy" />.</para>
///
///     <para><b>Conditional parameters.</b> To optimize <i>conditional</i> (define-by-run, branch-defined)
///     hyperparameters — which otherwise fall out of the intersection search space and are resampled at
///     random on every trial — use <see cref="NSGAIIICptSampler" /> (NSGA-III-CPT), the OptunaSharp variant
///     that carries a <see cref="OptunaSharp.Samplers.NsgaiiiCpt.ConditionalParams" /> rule set. This sampler itself is a bit-exact Optuna
///     port with no such extension.</para>
/// </summary>
/// <seealso cref="NSGAIIICptSampler" />
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIIISampler.html">Official Optuna NSGAIIISampler reference.</seealso>
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
    /// <summary>Configures NSGA-III reference-point selection, reproduction, and trial-completion hooks.</summary>
    /// <exception cref="NumSharp.ValueError">Population size is below two or the crossover parent count, or a configured reproduction probability is invalid.</exception>
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
        Action<Study, FrozenTrial, TrialState, IReadOnlyList<double>?>? afterTrialStrategy = null)
        : base(populationSize)
    {
        // Optuna decorates NSGAIIISampler with @experimental_class("3.2.0"), which raises an
        // ExperimentalWarning on construction. OptunaSharp targets the Optuna 5.x surface and treats
        // this sampler as stable, so that warning is intentionally omitted (deliberate divergence).

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
        _searchSpace = new IntersectionSearchSpace();

        _elitePopulationSelectionStrategy = elitePopulationSelectionStrategy
            ?? new NSGAIIIElitePopulationSelectionStrategy(
                populationSize, _rng, constraintsFunc, referencePoints, dividingParameter).Select;
        _childGenerationStrategy = childGenerationStrategy
            ?? new NSGAIIChildGenerationStrategy(
                mutation, mutationProb, crossover, crossoverProb, swappingProb, constraintsFunc, _rng).Generate;
        _afterTrialStrategy = afterTrialStrategy
            ?? new NSGAIIAfterTrialStrategy(constraintsFunc).Call;
    }

    /// <inheritdoc />
    public override void ReseedRng()
    {
        _randomSampler.ReseedRng();
        // Python: self._rng.rng.seed(), fresh OS entropy: 128 bits through numpy's own key construction,
        // so concurrent workers never share a stream (THREADING.md §6.3).
        OsEntropySeed.Reseed(_rng.Rng);
    }

    /// <inheritdoc />
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
        return _childGenerationStrategy(study, searchSpace, parentPopulation);
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
    /// <exception cref="System.Exception">A configured after-trial or constraint callback fails; its exception propagates.</exception>
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        System.Diagnostics.Debug.Assert(
            state is TrialState.Complete or TrialState.Fail or TrialState.Pruned);
        _afterTrialStrategy(study, trial, state, values);
        _randomSampler.AfterTrial(study, trial, state, values);
    }
}
