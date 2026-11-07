// SMS-EMOA sampler — S-Metric Selection Evolutionary Multi-objective Optimization Algorithm.
//
// Reference: Beume, Naujoks & Emmerich, "SMS-EMOA: Multiobjective selection based on dominated
// hypervolume", EJOR 181(3), 2007; survival details follow pymoo's `SMSEMOA`
// (pymoo/algorithms/moo/sms.py, Apache-2.0). SMS-EMOA has NO Optuna/OptunaHub counterpart, so — like
// `LmCmaSampler` / `JanusSampler` — it is a native, paper-faithful port validated by convergence /
// hypervolume, not bit-matched to an upstream trajectory.
//
// Design (steady-state μ+1 over Optuna's per-trial protocol):
//   • The first `populationSize` (μ) completed trials are the random initial population.
//   • Every subsequent trial proposes ONE offspring, generated from the current μ survivors by the
//     shared NSGA-II child generation (binary-tournament SBX + polynomial mutation) — the same
//     operators the NSGA-II sampler uses, so nothing is re-implemented.
//   • When an offspring completes it is folded into the population (μ+1), and the survival operator
//     drops the individual with the least hypervolume contribution in the worst nondomination front
//     (`SmsEmoaSurvival`), restoring μ. The survivor set is reconstructed incrementally in memory —
//     like `JanusSampler`, the algorithm state lives for the study's duration and does not resume
//     across process boundaries (a documented beyond-core concession).

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.Samplers.SmsEmoa;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     Multi-objective sampler using the SMS-EMOA algorithm (Beume, Naujoks &amp; Emmerich, 2007): a
///     steady-state (μ+1) evolutionary algorithm whose survival operator removes the individual with
///     the least <b>hypervolume contribution</b> in its worst nondomination front. Where NSGA-II breaks
///     ties within a front by crowding distance, SMS-EMOA breaks them by the S-metric (dominated
///     hypervolume) — so it steers directly toward a high-hypervolume Pareto approximation.
///
///     <para><b>Beyond-core.</b> SMS-EMOA has no Optuna / OptunaHub counterpart; like
///     <see cref="LmCmaSampler" /> it is validated by convergence and achieved hypervolume, not
///     bit-matched to an upstream. The crossover/mutation are the shared NSGA-II operators
///     (<see cref="SBXCrossover" /> + <see cref="PolynomialMutation" /> by default), so behavior on the
///     variation side matches the NSGA-II family; only the survival rule differs.</para>
///
///     <para><b>In-memory state.</b> The μ-survivor population is reconstructed incrementally and held
///     in memory for the study's duration, so a SMS-EMOA study does not resume its population across
///     process boundaries (same concession as <see cref="JanusSampler" />). Categorical and
///     single-value distributions are handled by the independent sampler.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(
///         directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
///         sampler: new SmsEmoaSampler(populationSize: 40, seed: 0));
///     study.Optimize(t =>
///     {
///         double x = t.SuggestFloat("x", 0, 1), y = t.SuggestFloat("y", 0, 1);
///         return new[] { x, (1 - Math.Sqrt(x)) + y };
///     }, nTrials: 400);
///     </code>
/// </example>
public class SmsEmoaSampler : BaseSampler
{
    private readonly int _populationSize;
    private readonly double _hvReferenceEps;
    private readonly RandomSampler _randomSampler;
    private readonly LazyRandomState _rng;
    private readonly IntersectionSearchSpace _searchSpace;
    private readonly NSGAIIChildGenerationStrategy _childGeneration;

    private readonly object _stateLock = new();
    // The μ-survivor population of each study this instance serves, keyed by storage object and study id: a
    // single pair of fields carried study A's survivors into study B when the sampler was reused (pre-release
    // review item H3).
    private readonly PerStudyState<StudyPopulation> _populations = new();

    /// <summary>The steady-state population of one study, advanced one completed offspring at a time.</summary>
    private sealed class StudyPopulation
    {
        /// <summary>Trial ids of the current μ survivors; <c>null</c> until the initial population is complete.</summary>
        public List<int>? SurvivorIds;

        /// <summary>The next trial number to fold into the population.</summary>
        public int NextNumber;

        /// <summary>
        ///     Trial 0 of the study when the initial population was formed: if the study now stored under this id
        ///     no longer has it, the study was recreated and this population is stale. <c>null</c> while empty.
        /// </summary>
        public TrialAnchor? Anchor;
    }

    /// <param name="populationSize">Number of individuals μ kept in the population.</param>
    /// <param name="mutation">Mutation applied to child individuals; defaults to <see cref="PolynomialMutation" />. If a parameter selected for mutation has no mutation object it is instead resampled independently.</param>
    /// <param name="mutationProb">Probability of mutating each parameter. If <c>null</c>, <c>1 / len(params)</c> is used.</param>
    /// <param name="crossover">Crossover applied to child individuals; defaults to <see cref="SBXCrossover" /> (SMS-EMOA's classic SBX). <see cref="UniformCrossover" /> is always applied to categorical parameters.</param>
    /// <param name="crossoverProb">Probability that a crossover occurs when creating a new individual.</param>
    /// <param name="swappingProb">Probability of swapping each parameter of the parents during crossover.</param>
    /// <param name="seed">Seed for the random number generator.</param>
    /// <param name="hvReferenceEps">Reference-point offset for the survival hypervolume: <c>ref = 1 + eps</c> in the population-normalized objective space (pymoo default <c>10</c>).</param>
    /// <summary>Configures steady-state hypervolume survival and the operators used to produce offspring.</summary>
    /// <exception cref="NumSharp.ValueError">Population size is below two or the crossover parent count, or a configured reproduction probability is invalid.</exception>
    public SmsEmoaSampler(
        int populationSize = 50,
        BaseMutation? mutation = null,
        double? mutationProb = null,
        BaseCrossover? crossover = null,
        double crossoverProb = 0.9,
        double swappingProb = 0.5,
        int? seed = null,
        double hvReferenceEps = SmsEmoaSurvival.DefaultEps)
    {
        if (populationSize < 2)
            throw new ValueError("`population_size` must be greater than or equal to 2.");

        crossover ??= new SBXCrossover();
        mutation ??= new PolynomialMutation();

        if (populationSize < crossover.NParents)
            throw new ValueError(
                $"Using {crossover}," +
                $" the population size should be greater than or equal to {crossover.NParents}." +
                $" The specified `population_size` is {populationSize}.");

        _populationSize = populationSize;
        _hvReferenceEps = hvReferenceEps;
        _randomSampler = new RandomSampler(seed);
        _rng = new LazyRandomState(seed);
        _searchSpace = new IntersectionSearchSpace();
        _childGeneration = new NSGAIIChildGenerationStrategy(
            mutation, mutationProb, crossover, crossoverProb, swappingProb, null, _rng);
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
            // Single-value distributions are sampled in Trial; the GA works on the multi-value space.
            if (distribution.Single())
                continue;
            searchSpace[name] = distribution;
        }
        return searchSpace;
    }

    /// <inheritdoc />
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        if (searchSpace.Count == 0)
            return new OrderedDict<string, object?>();

        var completed = study.GetTrialsInternal(
            deepcopy: false, states: new[] { TrialState.Complete }, useCache: true);
        if (completed.Count < _populationSize)
            return new OrderedDict<string, object?>();   // still filling the random initial population

        List<FrozenTrial> survivors;
        lock (_stateLock)
        {
            survivors = ReconstructPopulation(study, completed);
        }
        if (survivors.Count == 0)
            return new OrderedDict<string, object?>();

        return _childGeneration.Generate(study, searchSpace, survivors);
    }

    /// <inheritdoc />
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
        => _randomSampler.SampleIndependent(study, trial, paramName, paramDistribution);

    /// <inheritdoc />
    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _randomSampler.BeforeTrial(study, trial);

    /// <inheritdoc />
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
        => _randomSampler.AfterTrial(study, trial, state, values);

    // ---- steady-state population reconstruction --------------------------------------------------

    /// <summary>
    ///     Advance the in-memory μ-survivor set of <paramref name="study" /> to cover every completed offspring
    ///     trial, then return it. Trials are folded in ascending trial-number order (one (μ+1)→μ survival step
    ///     each); a gap in that order stalls the advance until it fills (matching the single-writer assumption).
    ///     Each study has its own survivor set; one left by an earlier study under a reused id is discarded.
    /// </summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="completed">Completed trial snapshots available for harvesting, in ascending trial-number order.</param>
    /// <returns>The incumbent trial population reconstructed from persisted replacement history.</returns>
    private List<FrozenTrial> ReconstructPopulation(Study study, List<FrozenTrial> completed)
    {
        // A completed trial never stops being completed, so a population whose anchor is gone belongs to a study
        // that was deleted and recreated under the same id: start this one from its own initial population.
        StudyPopulation population = _populations.Resolve(
            study,
            static () => new StudyPopulation(),
            state => state.Anchor is not { } anchor || anchor.IsIn(completed));

        var byNumber = new Dictionary<int, FrozenTrial>(completed.Count);
        var byId = new Dictionary<int, FrozenTrial>(completed.Count);
        foreach (var t in completed)
        {
            byNumber[t.Number] = t;
            byId[t.TrialId] = t;
        }

        if (population.SurvivorIds is null)
        {
            var initial = new List<int>(_populationSize);
            for (int number = 0; number < _populationSize; number++)
            {
                if (!byNumber.TryGetValue(number, out var t))
                    return new List<FrozenTrial>();   // initial population not contiguously complete yet
                initial.Add(t.TrialId);
            }
            population.SurvivorIds = initial;
            population.NextNumber = _populationSize;
            population.Anchor = TrialAnchor.Of(byNumber[0]);
        }

        double[] signs = DirectionSigns(study);
        while (byNumber.TryGetValue(population.NextNumber, out var offspring))
        {
            var survivorTrials = ResolveIds(population.SurvivorIds, byId);
            var merged = new List<FrozenTrial>(survivorTrials) { offspring };   // μ + 1
            var mergedLoss = merged.Select(t => Loss(t, signs)).ToArray();
            var (ideal, nadir) = IdealNadir(survivorTrials, signs);

            int[] keep = SmsEmoaSurvival.Select(mergedLoss, _populationSize, ideal, nadir, _hvReferenceEps);
            population.SurvivorIds = keep.Select(i => merged[i].TrialId).ToList();
            population.NextNumber++;
        }

        return ResolveIds(population.SurvivorIds, byId);
    }

    /// <summary>Resolves stored trial identifiers against available history in the requested order.</summary>
    /// <param name="ids">Stored trial identifiers in requested order.</param>
    /// <param name="byId">Available trial snapshots indexed by storage identifier.</param>
    /// <returns>Available trials corresponding to the identifiers, in identifier order.</returns>
    private static List<FrozenTrial> ResolveIds(List<int> ids, Dictionary<int, FrozenTrial> byId)
    {
        var result = new List<FrozenTrial>(ids.Count);
        foreach (int id in ids)
            if (byId.TryGetValue(id, out var t))
                result.Add(t);
        return result;
    }

    /// <summary>Per-objective sign so every objective becomes a minimization (Optuna internal convention).</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <returns>One multiplier per objective: plus one for minimization and minus one for maximization.</returns>
    private static double[] DirectionSigns(Study study)
    {
        var directions = study.Directions;
        var signs = new double[directions.Count];
        for (int j = 0; j < directions.Count; j++)
            signs[j] = directions[j] == StudyDirection.Minimize ? 1.0 : -1.0;
        return signs;
    }

    /// <summary>Converts objective values into minimization-oriented losses using one sign per objective.</summary>
    /// <param name="trial">Trial snapshot being examined; copy its contents before modifying them.</param>
    /// <param name="signs">One multiplier per objective: +1 for minimization and -1 for maximization.</param>
    /// <returns>Minimization-oriented objective values in study objective order.</returns>
    private static double[] Loss(FrozenTrial trial, double[] signs)
    {
        var values = trial.Values!;
        var loss = new double[signs.Length];
        for (int j = 0; j < signs.Length; j++)
            loss[j] = signs[j] * values[j];
        return loss;
    }

    /// <summary>ideal/nadir over the pre-merge parent population (pymoo estimates them here, adding 1e-32 to nadir).</summary>
    /// <param name="parents">Selected parent individuals in crossover order.</param>
    /// <param name="signs">One multiplier per objective: +1 for minimization and -1 for maximization.</param>
    /// <returns>Coordinate-wise minimum and maximum objective vectors.</returns>
    private static (double[] Ideal, double[] Nadir) IdealNadir(List<FrozenTrial> parents, double[] signs)
    {
        int m = signs.Length;
        var ideal = new double[m];
        var nadir = new double[m];
        for (int j = 0; j < m; j++)
        {
            ideal[j] = double.PositiveInfinity;
            nadir[j] = double.NegativeInfinity;
        }
        foreach (var t in parents)
        {
            var loss = Loss(t, signs);
            for (int j = 0; j < m; j++)
            {
                if (loss[j] < ideal[j]) ideal[j] = loss[j];
                if (loss[j] > nadir[j]) nadir[j] = loss[j];
            }
        }
        for (int j = 0; j < m; j++)
            nadir[j] += 1e-32;   // pymoo _advance: nadir = F.max(0) + 1e-32 (keeps the denominator positive)
        return (ideal, nadir);
    }
}
