// MOEA/D sampler — Multiobjective Evolutionary Algorithm based on Decomposition.
//
// Reference: Qingfu Zhang & Hui Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on
// Decomposition", IEEE Transactions on Evolutionary Computation 11(6), 2007. The structure (Das-Dennis
// weight vectors, neighborhood mating, ideal-point update, decomposition-based replacement) follows
// pymoo's `MOEAD` (pymoo/algorithms/moo/moead.py, Apache-2.0). MOEA/D has NO Optuna/OptunaHub
// counterpart, so — like `SmsEmoaSampler` / `LmCmaSampler` — it is a native, paper-faithful port
// validated by convergence / hypervolume, not bit-matched to an upstream trajectory. Plan 07 baseline.
//
// Design (steady-state per-subproblem over Optuna's per-trial protocol, mirroring SmsEmoaSampler):
//   • N = the Das-Dennis lattice size closest to (≤) the target `populationSize`; each weight vector is
//     one subproblem, and B[i] is its `nNeighbors` nearest weight vectors (including itself).
//   • The first N completed trials are the random initial population (subproblem i ← the i-th trial),
//     and the ideal point z* is their per-objective minimum.
//   • Every subsequent trial proposes ONE offspring for subproblem k = (#offspring so far) mod N: two
//     parents are drawn (from B[k] with probability `neighborSelectionProb`, else from the whole
//     population) and mated by the shared SBX + polynomial mutation. The subproblem index is stamped on
//     the trial.
//   • When an offspring completes it is folded in (in trial-number order): z* is updated with its
//     objectives, then up to `nReplacements` neighbors j∈B[k] whose scalarized fitness the offspring
//     improves are replaced by it. The population is reconstructed incrementally in memory — like
//     `SmsEmoaSampler` / `JanusSampler`, the algorithm state lives for the study's process lifetime and
//     does not resume across process boundaries (a documented beyond-core concession).

using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers.Moead;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     Multi-objective sampler using the MOEA/D algorithm (Zhang &amp; Li, 2007): a decomposition-based
///     evolutionary algorithm that splits the multi-objective problem into a spread of scalar subproblems
///     (one per Das-Dennis weight vector) and optimizes them cooperatively through neighborhood mating and
///     decomposition-based replacement. The scalarizing function is Tchebycheff by default
///     (<see cref="MoeadScalarization" /> also offers PBI and weighted sum).
///
///     <para><b>Beyond-core.</b> MOEA/D has no Optuna / OptunaHub counterpart; like
///     <see cref="SmsEmoaSampler" /> it is validated by convergence and achieved hypervolume, not
///     bit-matched to an upstream. The crossover/mutation are the shared NSGA operators
///     (<see cref="SBXCrossover" /> + <see cref="PolynomialMutation" /> by default), so only the selection
///     and replacement differ from the NSGA-II family.</para>
///
///     <para><b>In-memory state.</b> The N-solution population, its ideal point and the folded-offspring
///     bookkeeping are reconstructed incrementally and held in memory for the study's process lifetime, so
///     a MOEA/D study does not resume across process boundaries (same concession as
///     <see cref="SmsEmoaSampler" />). Categorical and single-value distributions are handled by the
///     independent sampler.</para>
///
///     <para><b>Actual population size.</b> The structured Das-Dennis lattice has a fixed size for each
///     (objectives, partitions) pair, so the sampler uses the largest lattice whose size does not exceed
///     the requested <c>populationSize</c>; the effective N (and hence the number of subproblems) can
///     therefore be slightly below <c>populationSize</c>.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(
///         directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
///         sampler: new MoeaDSampler(populationSize: 50, nNeighbors: 15, seed: 0));
///     study.Optimize(t =>
///     {
///         double x = t.SuggestFloat("x", 0, 1), y = t.SuggestFloat("y", 0, 1);
///         return new[] { x, (1 - Math.Sqrt(x)) + y };
///     }, nTrials: 600);
///     </code>
/// </example>
public class MoeaDSampler : BaseSampler
{
    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers.moead");

    private const string AttrKeySubproblem = "moead:subproblem";

    private readonly int _populationSizeTarget;
    private readonly int _nNeighbors;
    private readonly double _neighborSelectionProb;
    private readonly int _nReplacements;
    private readonly MoeadScalarization _scalarization;
    private readonly BaseCrossover _crossover;
    private readonly BaseMutation _mutation;
    private readonly double? _mutationProb;
    private readonly double _swappingProb;
    private readonly RandomSampler _randomSampler;
    private readonly LazyRandomState _rng;
    private readonly IntersectionSearchSpace _searchSpace;

    private readonly object _stateLock = new();
    private double[][]? _weights;         // N × m weight vectors (Das-Dennis), built once m is known
    private int[][]? _neighbors;          // N × T nearest-weight-vector indices (including self)
    private int _n;                       // effective population size / number of subproblems
    private List<int>? _populationIds;    // trial-ids of the current per-subproblem solutions (size N)
    private double[]? _ideal;             // running ideal point z* (minimization convention, length m)
    private HashSet<int>? _foldedNumbers; // trial numbers of offspring already folded into the population
    private bool _warnedSingleObjective;

    /// <param name="populationSize">Target number of subproblems / weight vectors. The effective N is the
    ///     largest Das-Dennis lattice size not exceeding this (see the class remarks).</param>
    /// <param name="nNeighbors">Neighborhood size T — how many nearest weight vectors form each
    ///     subproblem's mating/replacement neighborhood (clamped to N). Default 20.</param>
    /// <param name="neighborSelectionProb">Probability δ that a subproblem's parents are drawn from its
    ///     neighborhood rather than the whole population (Zhang &amp; Li's <c>δ</c>). Default 0.9.</param>
    /// <param name="nReplacements">Maximum number of neighbors an offspring may replace (Zhang &amp; Li's
    ///     <c>nr</c>, which preserves diversity). Default 2.</param>
    /// <param name="scalarization">Decomposition / scalarizing function (default
    ///     <see cref="MoeadScalarization.Tchebycheff" />).</param>
    /// <param name="mutation">Mutation applied to child individuals; defaults to <see cref="PolynomialMutation" />.</param>
    /// <param name="mutationProb">Probability of mutating each parameter. If <c>null</c>, <c>1 / len(params)</c> is used.</param>
    /// <param name="crossover">Crossover applied to child individuals; defaults to <see cref="SBXCrossover" />
    ///     (MOEA/D's classic SBX). <see cref="UniformCrossover" />-style crossover is always applied to categorical parameters.</param>
    /// <param name="swappingProb">Probability of swapping each categorical parameter of the parents during crossover.</param>
    /// <param name="seed">Seed for the random number generator.</param>
    public MoeaDSampler(
        int populationSize = 50,
        int nNeighbors = 20,
        double neighborSelectionProb = 0.9,
        int nReplacements = 2,
        MoeadScalarization scalarization = MoeadScalarization.Tchebycheff,
        BaseMutation? mutation = null,
        double? mutationProb = null,
        BaseCrossover? crossover = null,
        double swappingProb = 0.5,
        int? seed = null)
    {
        if (populationSize < 2)
            throw new ValueError("`population_size` must be greater than or equal to 2.");
        if (nNeighbors < 1)
            throw new ValueError("`n_neighbors` must be greater than or equal to 1.");
        if (neighborSelectionProb < 0.0 || neighborSelectionProb > 1.0)
            throw new ValueError("`neighbor_selection_prob` must be a float value within the range [0.0, 1.0].");
        if (nReplacements < 1)
            throw new ValueError("`n_replacements` must be greater than or equal to 1.");

        crossover ??= new SBXCrossover();
        mutation ??= new PolynomialMutation();

        if (populationSize < crossover.NParents)
            throw new ValueError(
                $"Using {crossover}," +
                $" the population size should be greater than or equal to {crossover.NParents}." +
                $" The specified `population_size` is {populationSize}.");

        _populationSizeTarget = populationSize;
        _nNeighbors = nNeighbors;
        _neighborSelectionProb = neighborSelectionProb;
        _nReplacements = nReplacements;
        _scalarization = scalarization;
        _crossover = crossover;
        _mutation = mutation;
        _mutationProb = mutationProb;
        _swappingProb = swappingProb;
        _randomSampler = new RandomSampler(seed);
        _rng = new LazyRandomState(seed);
        _searchSpace = new IntersectionSearchSpace();
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
            // Single-value distributions are sampled in Trial; the GA works on the multi-value space.
            if (distribution.Single())
                continue;
            searchSpace[name] = distribution;
        }
        return searchSpace;
    }

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        if (searchSpace.Count == 0)
            return new OrderedDict<string, object?>();

        int m = study.Directions.Count;
        if (m < 2)
        {
            // MOEA/D decomposes a multi-objective problem; on a single objective it degenerates, so fall
            // back to the independent sampler (matching the "unsupported → random" fallback idiom).
            if (!_warnedSingleObjective)
            {
                Logger.Warning(
                    "`MoeaDSampler` requires a multi-objective study (>= 2 objectives). " +
                    $"`{_randomSampler.GetType().Name}` is used instead.");
                _warnedSingleObjective = true;
            }
            return new OrderedDict<string, object?>();
        }

        var completed = study.GetTrialsInternal(
            deepcopy: false, states: new[] { TrialState.Complete }, useCache: true);
        double[] signs = DirectionSigns(study);

        List<FrozenTrial> population;
        int subproblem;
        int[] neighbors;
        int n;
        lock (_stateLock)
        {
            EnsureWeights(m);
            n = _n;
            if (completed.Count < n)
                return new OrderedDict<string, object?>();   // still filling the random initial population

            population = ReconstructPopulation(study, completed, signs);
            if (population.Count < n)
                return new OrderedDict<string, object?>();

            // Cyclic subproblem assignment: the c-th offspring targets subproblem c mod N. Under sequential
            // execution c equals the number of completed offspring at generation time.
            int offspringCompleted = completed.Count(t => t.SystemAttrs.ContainsKey(AttrKeySubproblem));
            subproblem = offspringCompleted % n;
            neighbors = _neighbors![subproblem];
        }

        // Mating pool: the subproblem's neighborhood with probability δ, else the whole population.
        IReadOnlyList<FrozenTrial> matingPool =
            (double)_rng.Rng.rand() < _neighborSelectionProb
                ? neighbors.Select(j => population[j]).ToList()
                : population;

        var child = MoeadOffspring.Generate(
            study, searchSpace, matingPool, _crossover, _mutation, _mutationProb, _swappingProb, _rng.Rng);

        study.Storage.SetTrialSystemAttr(trial.TrialId, AttrKeySubproblem, subproblem);
        return child;
    }

    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
        => _randomSampler.SampleIndependent(study, trial, paramName, paramDistribution);

    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _randomSampler.BeforeTrial(study, trial);

    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
        => _randomSampler.AfterTrial(study, trial, state, values);

    // ---- weight vectors / neighborhoods (built once m is known) ----------------------------------

    /// <summary>Build the Das-Dennis weight vectors and their neighborhoods on first use, deriving the
    ///     effective population size N from the target and the objective count.</summary>
    private void EnsureWeights(int m)
    {
        if (_weights is not null)
            return;

        int partitions = ReferenceDirections.PartitionClosestToPoints(_populationSizeTarget, m);
        _weights = ReferenceDirections.DasDennis(partitions, m);
        _n = _weights.Length;

        int t = Math.Min(Math.Max(1, _nNeighbors), _n);
        _neighbors = ComputeNeighbors(_weights, t);
    }

    /// <summary>For each weight vector, the indices of its <paramref name="t" /> nearest weight vectors by
    ///     Euclidean distance (including itself, which sorts first at distance 0) — the mating/replacement
    ///     neighborhood B[i], mirroring pymoo's <c>argsort(cdist(ref_dirs, ref_dirs))[:, :n_neighbors]</c>.</summary>
    private static int[][] ComputeNeighbors(double[][] weights, int t)
    {
        int n = weights.Length;
        int m = weights[0].Length;
        var neighbors = new int[n][];
        for (int i = 0; i < n; i++)
        {
            var dist = new (double D, int J)[n];
            for (int j = 0; j < n; j++)
            {
                double s = 0.0;
                for (int d = 0; d < m; d++)
                {
                    double diff = weights[i][d] - weights[j][d];
                    s += diff * diff;
                }
                dist[j] = (s, j);
            }
            // Ascending by distance, ties broken by index (stable, deterministic).
            Array.Sort(dist, (a, b) =>
            {
                int c = a.D.CompareTo(b.D);
                return c != 0 ? c : a.J.CompareTo(b.J);
            });
            var row = new int[t];
            for (int k = 0; k < t; k++) row[k] = dist[k].J;
            neighbors[i] = row;
        }
        return neighbors;
    }

    // ---- steady-state population reconstruction --------------------------------------------------

    /// <summary>
    ///     Advance the in-memory per-subproblem population to cover every completed offspring, then return
    ///     it (index i ↔ subproblem i). The initial population is the first N completed trials without a
    ///     subproblem stamp; each offspring (a completed trial carrying <see cref="AttrKeySubproblem" />) is
    ///     folded once, in trial-number order: it updates the ideal point and replaces up to
    ///     <see cref="_nReplacements" /> neighbors whose scalarized fitness it improves.
    /// </summary>
    private List<FrozenTrial> ReconstructPopulation(Study study, List<FrozenTrial> completed, double[] signs)
    {
        var byId = new Dictionary<int, FrozenTrial>(completed.Count);
        foreach (var t in completed)
            byId[t.TrialId] = t;

        int m = signs.Length;

        if (_populationIds is null)
        {
            var initTrials = completed
                .Where(t => !t.SystemAttrs.ContainsKey(AttrKeySubproblem))
                .OrderBy(t => t.Number)
                .ToList();
            if (initTrials.Count < _n)
                return new List<FrozenTrial>();   // initial population not complete yet

            _populationIds = initTrials.Take(_n).Select(t => t.TrialId).ToList();
            _ideal = new double[m];
            for (int j = 0; j < m; j++) _ideal[j] = double.PositiveInfinity;
            foreach (int id in _populationIds)
            {
                double[] loss = Loss(byId[id], signs);
                for (int j = 0; j < m; j++)
                    if (loss[j] < _ideal[j]) _ideal[j] = loss[j];
            }
            _foldedNumbers = new HashSet<int>();
        }

        // Fold every offspring not yet folded, in ascending trial-number order (steady-state: each fold
        // sees the population and ideal point left by the previous folds).
        var newOffspring = completed
            .Where(t => t.SystemAttrs.ContainsKey(AttrKeySubproblem) && !_foldedNumbers!.Contains(t.Number))
            .OrderBy(t => t.Number)
            .ToList();

        foreach (var off in newOffspring)
        {
            int sc = Convert.ToInt32(off.SystemAttrs[AttrKeySubproblem]);
            if (sc < 0 || sc >= _n)   // guard against a stale/foreign stamp (e.g., a changed population size)
            {
                _foldedNumbers!.Add(off.Number);
                continue;
            }

            double[] offLoss = Loss(off, signs);
            for (int j = 0; j < m; j++)
                if (offLoss[j] < _ideal![j]) _ideal[j] = offLoss[j];

            int replaced = 0;
            foreach (int j in _neighbors![sc])
            {
                var current = byId[_populationIds[j]];
                double gOff = MoeadDecomposition.Scalarize(_scalarization, offLoss, _weights![j], _ideal!);
                double gCur = MoeadDecomposition.Scalarize(_scalarization, Loss(current, signs), _weights[j], _ideal!);
                if (gOff < gCur)
                {
                    _populationIds[j] = off.TrialId;
                    replaced++;
                    if (replaced >= _nReplacements) break;
                }
            }
            _foldedNumbers!.Add(off.Number);
        }

        return _populationIds.Select(id => byId[id]).ToList();
    }

    /// <summary>Per-objective sign so every objective becomes a minimization (Optuna internal convention).</summary>
    private static double[] DirectionSigns(Study study)
    {
        var directions = study.Directions;
        var signs = new double[directions.Count];
        for (int j = 0; j < directions.Count; j++)
            signs[j] = directions[j] == StudyDirection.Minimize ? 1.0 : -1.0;
        return signs;
    }

    private static double[] Loss(FrozenTrial trial, double[] signs)
    {
        var values = trial.Values!;
        var loss = new double[signs.Length];
        for (int j = 0; j < signs.Length; j++)
            loss[j] = signs[j] * values[j];
        return loss;
    }
}
