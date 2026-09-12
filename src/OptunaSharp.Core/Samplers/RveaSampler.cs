// RVEA sampler — Reference Vector Guided Evolutionary Algorithm for many-objective optimization.
//
// Reference: Ran Cheng, Yaochu Jin, Markus Olhofer & Bernhard Sendhoff, "A Reference Vector Guided
// Evolutionary Algorithm for Many-Objective Optimization", IEEE TEVC 20(5), 2016. The structure
// (Das-Dennis reference vectors, angle-penalized-distance survival, periodic reference-vector
// adaptation) follows pymoo's `RVEA` (pymoo/algorithms/moo/rvea.py, Apache-2.0). RVEA has NO
// Optuna/OptunaHub counterpart, so — like `NSGAIISampler`'s cousin `SmsEmoaSampler` — it is a native,
// paper-faithful port validated by convergence / hypervolume, not bit-matched to an upstream. Plan 07
// baseline.

using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.Samplers.Rvea;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     Multi-objective sampler using the RVEA algorithm (Cheng et al., 2016): a reference-vector-guided
///     evolutionary algorithm for many-objective optimization. It reuses the whole NSGA-II genetic
///     machinery (<see cref="BaseGASampler" />, child generation, crossovers, mutations) and swaps the
///     elite-selection strategy for <b>angle-penalized-distance (APD) survival</b>
///     (<see cref="RveaApdSurvival" />): each individual is associated with its nearest reference vector,
///     and within every reference vector the survivor is the one whose distance to the ideal point is
///     least after a generation-growing angle penalty — an explicit diversity→convergence shift that suits
///     three-and-more-objective fronts. The reference vectors are periodically re-scaled to the current
///     objective ranges.
///
///     <para><b>Beyond-core.</b> RVEA has no Optuna / OptunaHub counterpart; like <see cref="SmsEmoaSampler" />
///     it is validated by convergence and achieved hypervolume, not bit-matched to an upstream. The default
///     operators are RVEA's classic <see cref="SBXCrossover" /> (η=30) + <see cref="PolynomialMutation" />
///     (η=20); mating uses the shared NSGA-II child generation, so only the survival differs from the
///     NSGA-II family.</para>
///
///     <para><b>Generation horizon.</b> The APD penalty grows with <c>(generation / nMaxGenerations)^α</c>
///     and the reference vectors adapt every <c>ceil(nMaxGenerations · adaptFreq)</c> generations. Optuna is
///     trial-budget-driven and does not expose a maximum generation to the sampler, so <c>nMaxGenerations</c>
///     sets that schedule's horizon explicitly (the ratio is clamped to 1 beyond it).</para>
///
///     <para><b>Reference-vector count.</b> The structured Das-Dennis lattice has a fixed size for each
///     (objectives, partitions) pair, so — unless explicit reference points are supplied — the effective
///     population size is set to the largest lattice size not exceeding the requested
///     <c>populationSize</c>.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(
///         directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize, StudyDirection.Minimize },
///         sampler: new RveaSampler(populationSize: 91, seed: 0));
///     study.Optimize(t =>
///     {
///         double x = t.SuggestFloat("x", 0, 1), y = t.SuggestFloat("y", 0, 1), z = t.SuggestFloat("z", 0, 1);
///         return new[] { x, y, z };
///     }, nTrials: 900);
///     </code>
/// </example>
public class RveaSampler : BaseGASampler
{
    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers.rvea");

    private readonly RandomSampler _randomSampler;
    private readonly LazyRandomState _rng;
    private readonly IntersectionSearchSpace _searchSpace;
    private readonly ConditionalParams? _conditionalParams;

    private readonly int _populationSizeTarget;
    private readonly double _alpha;
    private readonly double _adaptFreq;
    private readonly int _nMaxGenerations;
    private readonly double[,]? _referencePoints;
    private readonly int? _dividingParameter;

    private readonly Func<Study, OrderedDict<string, BaseDistribution>, List<FrozenTrial>, OrderedDict<string, object?>>
        _childGenerationStrategy;
    private readonly Action<Study, FrozenTrial, TrialState, IReadOnlyList<double>?> _afterTrialStrategy;

    private RveaApdSurvival? _apdSurvival;   // built lazily once the objective count is known
    private bool _warnedSingleObjective;

    /// <param name="populationSize">Target number of reference vectors / generation size. When reference
    ///     points are generated (the default), the effective count is the largest Das-Dennis lattice size
    ///     not exceeding this (see the class remarks).</param>
    /// <param name="mutation">Mutation applied when creating child individuals; defaults to
    ///     <see cref="PolynomialMutation" /> (η=20).</param>
    /// <param name="mutationProb">Probability of mutating each parameter. If <c>null</c>, <c>1 / len(parent.params)</c> is used.</param>
    /// <param name="crossover">Crossover applied when creating child individuals; defaults to
    ///     <see cref="SBXCrossover" /> (η=30, RVEA's classic operator).</param>
    /// <param name="crossoverProb">Probability that a crossover occurs when creating a new individual.</param>
    /// <param name="swappingProb">Probability of swapping each parameter of the parents during crossover.</param>
    /// <param name="seed">Seed for random number generator.</param>
    /// <param name="alpha">The APD penalty exponent α (default 2.0).</param>
    /// <param name="adaptFreq">Fraction of <paramref name="nMaxGenerations" /> at which the reference vectors
    ///     are re-scaled to the current objective ranges (default 0.1; 0 disables adaptation).</param>
    /// <param name="nMaxGenerations">Generation horizon for the APD penalty schedule and the adaptation
    ///     period (default 100).</param>
    /// <param name="referencePoints">Explicit reference vectors (a 2-D array with <c>n_objectives</c> columns).
    ///     When <c>null</c> (default), Das-Dennis reference vectors are generated.</param>
    /// <param name="dividingParameter">Number of Das-Dennis divisions when generating the default reference
    ///     vectors. When <c>null</c> (default), it is chosen so the lattice size is closest to (≤)
    ///     <paramref name="populationSize" />.</param>
    /// <param name="childGenerationStrategy">Experimental. Overrides child generation.</param>
    /// <param name="afterTrialStrategy">Experimental. Overrides the after-trial procedure.</param>
    /// <param name="conditionalParams">OptunaSharp extension (no Optuna analog): activation rules for
    ///     parameters that only exist under some values of other parameters — see <see cref="ConditionalParams" />.
    ///     <c>null</c> (default) keeps the parity behavior.</param>
    public RveaSampler(
        int populationSize = 50,
        BaseMutation? mutation = null,
        double? mutationProb = null,
        BaseCrossover? crossover = null,
        double crossoverProb = 0.9,
        double swappingProb = 0.5,
        int? seed = null,
        double alpha = 2.0,
        double adaptFreq = 0.1,
        int nMaxGenerations = 100,
        double[,]? referencePoints = null,
        int? dividingParameter = null,
        Func<Study, OrderedDict<string, BaseDistribution>, List<FrozenTrial>, OrderedDict<string, object?>>?
            childGenerationStrategy = null,
        Action<Study, FrozenTrial, TrialState, IReadOnlyList<double>?>? afterTrialStrategy = null,
        ConditionalParams? conditionalParams = null)
        : base(populationSize)
    {
        if (populationSize < 2)
            throw new ValueError("`population_size` must be greater than or equal to 2.");
        if (alpha < 0.0)
            throw new ValueError("`alpha` must be greater than or equal to 0.0.");
        if (adaptFreq < 0.0 || adaptFreq > 1.0)
            throw new ValueError("`adapt_freq` must be a float value within the range [0.0, 1.0].");
        if (nMaxGenerations < 1)
            throw new ValueError("`n_max_generations` must be greater than or equal to 1.");

        // Optuna raises ExperimentalWarnings for after_trial_strategy / child_generation_strategy and
        // for RveaSampler itself; OptunaSharp targets the Optuna 5.x surface and treats them as stable,
        // so those warnings are intentionally omitted (deliberate divergence).

        // RVEA's classic operators (pymoo defaults): SBX(eta=30) + polynomial mutation(eta=20).
        crossover ??= new SBXCrossover(eta: 30.0);
        mutation ??= new PolynomialMutation();

        if (populationSize < crossover.NParents)
            throw new ValueError(
                $"Using {crossover}," +
                $" the population size should be greater than or equal to {crossover.NParents}." +
                $" The specified `population_size` is {populationSize}.");

        _randomSampler = new RandomSampler(seed);
        _rng = new LazyRandomState(seed);
        _searchSpace = new IntersectionSearchSpace(conditionalParams: conditionalParams);
        _conditionalParams = conditionalParams;

        _populationSizeTarget = populationSize;
        _alpha = alpha;
        _adaptFreq = adaptFreq;
        _nMaxGenerations = nMaxGenerations;
        _referencePoints = referencePoints;
        _dividingParameter = dividingParameter;

        _childGenerationStrategy = childGenerationStrategy
            ?? new NSGAIIChildGenerationStrategy(
                mutation, mutationProb, crossover, crossoverProb, swappingProb, null, _rng,
                conditionalParams).Generate;
        _afterTrialStrategy = afterTrialStrategy
            ?? new NSGAIIAfterTrialStrategy(null).Call;
    }

    public override void ReseedRng()
    {
        _randomSampler.ReseedRng();
        Span<byte> buffer = stackalloc byte[4];
        RandomNumberGenerator.Fill(buffer);
        _rng.Rng.seed(BitConverter.ToUInt32(buffer));
    }

    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _searchSpace.Calculate(study))
        {
            if (distribution.Single())
                continue;
            searchSpace[name] = distribution;
        }
        return searchSpace;
    }

    public override List<FrozenTrial> SelectParent(Study study, int generation)
    {
        EnsureReferenceVectors(study.Directions.Count);
        var combined = GetPopulation(study, generation - 1);
        combined.AddRange(GetParentPopulation(study, generation - 1));
        return _apdSurvival!.Select(study, combined, generation);
    }

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        int m = study.Directions.Count;
        if (m < 2)
        {
            // RVEA guides search with reference vectors in objective space; on a single objective it
            // degenerates, so fall back to the independent sampler.
            if (!_warnedSingleObjective)
            {
                Logger.Warning(
                    "`RveaSampler` requires a multi-objective study (>= 2 objectives). " +
                    $"`{_randomSampler.GetType().Name}` is used instead.");
                _warnedSingleObjective = true;
            }
            return new OrderedDict<string, object?>();
        }

        // The reference vectors (and therefore the effective population size) must be fixed before the
        // generation is attributed, so do it first — the objective count is known even before any trial
        // completes, so this is safe for the very first trials too.
        EnsureReferenceVectors(m);

        int generation = GetTrialGeneration(study, trial);
        var parentPopulation = GetParentPopulation(study, generation);
        if (parentPopulation.Count == 0)
            return new OrderedDict<string, object?>();

        var child = _childGenerationStrategy(study, searchSpace, parentPopulation);
        _conditionalParams?.ApplyDormantCarry(study, trial, child, parentPopulation, searchSpace);
        return child;
    }

    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
        => _randomSampler.SampleIndependent(study, trial, paramName, paramDistribution);

    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _randomSampler.BeforeTrial(study, trial);

    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        System.Diagnostics.Debug.Assert(
            state is TrialState.Complete or TrialState.Fail or TrialState.Pruned);
        _afterTrialStrategy(study, trial, state, values);
        _randomSampler.AfterTrial(study, trial, state, values);
    }

    /// <summary>
    ///     Build the reference vectors and the APD survival operator on first use, then pin the base
    ///     population size to the reference-vector count (RVEA keeps at most one survivor per reference
    ///     vector, so the generation size is the vector count). The objective count is available from the
    ///     study directions before any trial finishes, so this can run for the very first trial.
    /// </summary>
    private void EnsureReferenceVectors(int m)
    {
        if (_apdSurvival is not null)
            return;

        double[][] referenceVectors;
        if (_referencePoints is not null)
        {
            if (_referencePoints.GetLength(1) != m)
                throw new ValueError(
                    "The dimension of reference points vectors must be the same as the number " +
                    "of objectives of the study.");
            int rows = _referencePoints.GetLength(0);
            referenceVectors = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                var row = new double[m];
                for (int j = 0; j < m; j++) row[j] = _referencePoints[i, j];
                referenceVectors[i] = row;
            }
        }
        else
        {
            int partitions = _dividingParameter
                             ?? ReferenceDirections.PartitionClosestToPoints(_populationSizeTarget, m);
            referenceVectors = ReferenceDirections.DasDennis(partitions, m);
        }

        _apdSurvival = new RveaApdSurvival(referenceVectors, _alpha, _adaptFreq, _nMaxGenerations);
        PopulationSize = _apdSurvival.RefCount;
    }
}
