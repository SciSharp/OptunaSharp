// Sampler for Differential Evolution (DE) — Storn & Price's global black-box optimizer.
//
// Reference:
//   Rainer Storn and Kenneth Price. "Differential Evolution — A Simple and Efficient Heuristic for
//   Global Optimization over Continuous Spaces." Journal of Global Optimization 11:341–359 (1997).
//
// This sampler drives the textbook-faithful `De` optimizer (see De/De.cs) through the Optuna study
// protocol, following the `LmCmaSampler` idiom: the optimizer state is serialized into trial system
// attrs (so a DE study resumes across process boundaries / distributed workers), each trial re-seeds
// the optimizer and asks a single candidate, and every asked candidate is clamped to the transformed
// box (DE is unconstrained, so bounds are the sampler's concession). Optuna itself has no DE sampler
// (and `cmaes` no DE class). OptunaHub has one (`samplers/differential_evolution`, `DESampler`, since
// 2024-12), but this port follows Storn & Price, not that package, so parameters, defaults and seeded
// trajectories differ (the map is in the class remarks; QA-70 of docs/qa/PARITY_QA_2026-11-10.md, where
// the owner chose on 2026-11-12 to keep this port). Like `LmCmaSampler` it is a native implementation
// validated by convergence, with no upstream trajectory to bit-match.

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers.Differential;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     A sampler driving a textbook-faithful <see cref="De" /> Differential Evolution optimizer (Storn
///     &amp; Price 1997) through the Optuna protocol. DE maintains a population of candidate solutions and,
///     for each target member, forms a trial vector by adding a scaled difference of other members
///     (<c>DE/rand/1</c> or <c>DE/best/1</c>) and binomially crossing it with the target; the trial
///     replaces the target only if it is no worse (greedy one-to-one selection). It is a robust,
///     derivative-free global optimizer that complements the Gaussian model of <see cref="CmaEsSampler" />
///     and the limited-memory <see cref="LmCmaSampler" />.
///
///     <para><b>Bootstrapping.</b> DE needs an initial population before it can generate offspring, so the
///     first <c>max(nStartupTrials, populationSize)</c> completed trials are drawn by the independent
///     sampler and become the initial population; DE offspring are proposed from then on.</para>
///
///     <para><b>Two protocol concessions (as <see cref="LmCmaSampler" />):</b> (1) DE is unconstrained, so
///     each asked candidate is <b>clamped</b> to the transformed bounds and the next generation's greedy
///     selection re-derives the parent from the stored (clamped) params — the optimizer learns from the
///     actually-evaluated points. (2) The per-trial protocol asks one candidate per trial from
///     freshly-restored state; each trial is stamped with its <c>generation</c> and its <c>target index</c>
///     so the population's greedy selection pairs each offspring with the right parent once the generation
///     completes.</para>
///
///     <para>Categorical parameters and single-value distributions are handled by the independent sampler,
///     as in <see cref="CmaEsSampler" />.</para>
///
///     <para><b>Not OptunaHub's <c>DESampler</c>.</b> OptunaHub's <c>samplers/differential_evolution</c> is another
///     implementation of DE/rand/1/bin, which this class does not follow, so a study moved over from it gets other
///     defaults and other trials (registry pinned in <c>refs/optunahub-registry</c>):</para>
///     <list type="table">
///         <listheader><term>OptunaHub <c>DESampler</c></term><description>This class</description></listheader>
///         <item><term><c>population_size="auto"</c>: <c>max(20, 10d)</c> below 5 dimensions, <c>max(30, 10d)</c> up to 30,
///             then <c>max(50, 5d)</c></term><description><c>popsize</c>: <c>max(8, 4 + ⌊3 ln d⌋)</c></description></item>
///         <item><term><c>F=0.8</c></term><description><c>f = 0.8</c></description></item>
///         <item><term><c>CR=0.7</c></term><description><c>cr = 0.9</c></description></item>
///         <item><term>DE/rand/1/bin only</term><description><c>strategy</c>: <c>"rand1bin"</c> (default) or <c>"best1bin"</c></description></item>
///         <item><term><c>search_space</c>, <c>debug</c>, <c>seed</c></term><description><c>seed</c>; the search space is
///             inferred; plus <c>nStartupTrials</c>, <c>independentSampler</c>, <c>warnIndependentSampling</c></description></item>
///     </list>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(sampler: new DifferentialEvolutionSampler(seed: 42));
///     study.Optimize(t =>
///     {
///         double s = 0;
///         for (int i = 0; i &lt; 10; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
///         return s;
///     }, nTrials: 400);
///     </code>
/// </example>
public sealed class DifferentialEvolutionSampler : BaseSampler
{
    // system_attrs values must stay under 2046 characters on RDBStorage (same limit as CmaEsSampler).
    private const int SystemAttrMaxLength = 2045;

    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers.differential_evolution");

    private readonly BaseSampler _independentSampler;
    private readonly int _nStartupTrials;
    private bool _warnIndependentSampling;
    private readonly LazyRandomState _deRng;
    private readonly IntersectionSearchSpace _searchSpace;
    private readonly int? _popsize;
    private readonly double _f;
    private readonly double _cr;
    private readonly DeStrategy _strategy;

    // Trials sampled independently on purpose because every target of the generation was held by a running trial, or
    // every redraw met a point a running trial holds; their independent draws are not warned about. Guarded by the
    // sampler's monitor, like every hook.
    private readonly OverrunTrials _overrun = new();

    // Per study: the trial vector each running trial received (generation, clamped point in the optimizer's unit box),
    // so a point a running trial evaluates is not handed out again. Entries leave with AfterTrial. Guarded by the
    // sampler's monitor.
    private readonly HandedOutPoints _handedOut = new();

    /// <summary>
    ///     The last stored optimizer text this instance parsed, and the state parsed from it (plan 15): the text changes
    ///     once per generation, and every ask in between parsed the same JSON again. Guarded by the sampler's monitor,
    ///     like every hook; a per-worker instance has its own.
    /// </summary>
    /// <remarks>
    ///     Safe to share across asks because <see cref="De.FromState" /> never changes the state it reads: the
    ///     <see cref="De" /> constructor clones every population row and the values, and the rest are numbers. A parse
    ///     that throws stores nothing, so the next ask parses (and throws) again, as before.
    /// </remarks>
    private (string Text, DeState State)? _parsedState;

    /// <param name="nStartupTrials">Minimum independent trials before DE begins; DE always waits for at
    ///     least <c>populationSize</c> completed trials to seed the initial population, whichever is larger.</param>
    /// <param name="independentSampler">Sampler for parameters outside the relative search space (default <see cref="RandomSampler" />).</param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently.</param>
    /// <param name="seed">Random seed for DE.</param>
    /// <param name="popsize">Population size NP (default <c>max(8, 4 + ⌊3 ln d⌋)</c>; must be ≥ 4).</param>
    /// <param name="f">Differential weight F, scaling the difference vector (default 0.8).</param>
    /// <param name="cr">Crossover rate CR, the per-coordinate probability of inheriting the mutant (default 0.9).</param>
    /// <param name="strategy">Mutation strategy: <c>rand1bin</c> (default, DE/rand/1/bin) or <c>best1bin</c> (DE/best/1/bin).</param>
    /// <summary>Configures differential-evolution proposals and independent startup sampling.</summary>
    /// <exception cref="System.ArgumentException">The requested population size is below four.</exception>
    public DifferentialEvolutionSampler(
        int nStartupTrials = 1,
        BaseSampler? independentSampler = null,
        bool warnIndependentSampling = true,
        int? seed = null,
        int? popsize = null,
        double f = 0.8,
        double cr = 0.9,
        string strategy = "rand1bin")
    {
        if (popsize is int p && p < 4)
            throw new ArgumentException("populationSize must be at least 4 (target + 3 donors).", nameof(popsize));
        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _nStartupTrials = nStartupTrials;
        _warnIndependentSampling = warnIndependentSampling;
        _deRng = new LazyRandomState(seed);
        _searchSpace = new IntersectionSearchSpace();
        _popsize = popsize;
        _f = f;
        _cr = cr;
        _strategy = ParseStrategy(strategy);
    }

    /// <summary>Maps supported differential-evolution strategy names to the internal strategy enum.</summary>
    /// <param name="strategy">Differential-evolution donor/crossover strategy name or enum.</param>
    /// <returns>The supported strategy, with unrecognized names resolved to the implementation default.</returns>
    private static DeStrategy ParseStrategy(string strategy)
    {
        string s = (strategy ?? "rand1bin").ToLowerInvariant().Replace("/", "").Replace("_", "").Replace("bin", "");
        return s switch
        {
            "best" or "best1" => DeStrategy.Best1Bin,
            "rand" or "rand1" or "" => DeStrategy.Rand1Bin,
            _ => DeStrategy.Rand1Bin,
        };
    }

    /// <inheritdoc />
    public override void ReseedRng()
    {
        // _deRng needs no reseed here: relative sampling reseeds per trial (see SampleRelative).
        _independentSampler.ReseedRng();
    }

    /// <inheritdoc />
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _searchSpace.Calculate(study))
        {
            // A single-valued distribution is sampled inside Trial; a categorical one is unsupported.
            if (distribution.Single()) continue;
            if (distribution is not FloatDistribution && distribution is not IntDistribution) continue;
            searchSpace[name] = distribution;
        }
        return searchSpace;
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">The study has multiple objectives, but this sampler requires a scalar objective.</exception>
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        RaiseErrorIfMultiObjective(study);

        if (searchSpace.Count == 0)
            return new OrderedDict<string, object?>();

        var completedTrials = GetTrials(study);

        var trans = new SearchSpaceTransform(searchSpace, transformStep: true, transform01: true);
        double[] boundsFlat = trans.Bounds.ToArray<double>();
        int nDim = boundsFlat.Length / 2;
        var lower = new double[nDim];
        var upper = new double[nDim];
        for (int i = 0; i < nDim; i++) { lower[i] = boundsFlat[i * 2]; upper[i] = boundsFlat[i * 2 + 1]; }

        int popsize = _popsize ?? De.DefaultPopulationSize(nDim);

        // DE needs a full initial population before it can propose offspring.
        if (completedTrials.Count < Math.Max(_nStartupTrials, popsize))
            return new OrderedDict<string, object?>();

        var optimizer = RestoreOptimizer(completedTrials)
                        ?? BuildInitialOptimizer(completedTrials, trans, nDim, popsize, study.Direction);
        if (optimizer is null)
            return new OrderedDict<string, object?>();

        if (optimizer.Dim != nDim)
        {
            if (_warnIndependentSampling)
            {
                Logger.Warning(
                    "`DifferentialEvolutionSampler` does not support dynamic search space. " +
                    $"`{_independentSampler.GetType().Name}` is used instead of `DifferentialEvolutionSampler`.");
                _warnIndependentSampling = false;
            }
            return new OrderedDict<string, object?>();
        }

        // Complete a generation once all NP target-ordered offspring have finished, then re-serialize.
        var solutionTrials = GetSolutionTrials(completedTrials, optimizer.Generation);
        if (solutionTrials.Count >= optimizer.PopulationSize)
        {
            var solutions = new List<(double[] X, double Value)>(optimizer.PopulationSize);
            foreach (var t in solutionTrials.Take(optimizer.PopulationSize))
            {
                double[] x = trans.Transform(t.Params).ToArray<double>();
                double y = study.Direction == StudyDirection.Minimize ? t.Value!.Value : -t.Value!.Value;
                solutions.Add((x, y));
            }
            optimizer.Tell(solutions);

            string optimizerStr = SerializeOptimizer(optimizer);
            foreach (var (key, value) in SplitOptimizerStr(optimizerStr))
                study.Storage.SetTrialSystemAttr(trial.TrialId, key, value);

            solutionTrials = GetSolutionTrials(completedTrials, optimizer.Generation);
        }

        // Target index within the current generation (0-based): the first target no finished or running trial of the
        // generation holds. Counting only the finished ones gave every worker asking at the same time the same target,
        // and past the population clamped them all onto the last one, so concurrent trials could receive one point.
        // A sequential run holds only its finished targets 0..count-1, so the target is still solutionTrials.Count
        // and seeded runs are unchanged.
        if (FirstFreeTarget(study, trial, optimizer.Generation, optimizer.PopulationSize, solutionTrials) is not int target)
        {
            // Every target is held by a running trial (more workers than the population): rather than hand this trial
            // a target another worker is evaluating, sample it independently. It joins no generation, as its
            // generation and index are never stamped.
            _overrun.Add(study, trial.Number);
            return new OrderedDict<string, object?>();
        }

        // Each trial reseeds so its single ask is reproducible and differs from its siblings' asks.
        int seed = (int)NumPyDraws.RandInt(_deRng.Rng, 1, 65536) + trial.Number;
        optimizer.ReseedRng(seed);

        double[] deParams = optimizer.AskCandidate(target);
        for (int i = 0; i < nDim; i++)
            deParams[i] = Math.Min(Math.Max(deParams[i], lower[i]), upper[i]);   // clamp to the box

        // Distinct targets do not make distinct points: a trial vector that takes every coordinate from the mutant
        // is independent of its target, so two workers drawing the same donors (r1, r2, r3) built the same point, and
        // mutants pushed out of the box clamp onto the same face or corner (2 of 48 trials in one 8-worker run). Hand
        // out the first draw no other running trial of the generation is evaluating. A sequential run has no such
        // trial, so it never redraws and its RNG stream, and seeded results, are unchanged.
        double[]? claimed = _handedOut.Claim(study, trial, optimizer.Generation, deParams, () =>
        {
            double[] next = optimizer.AskCandidate(target);
            for (int i = 0; i < nDim; i++)
                next[i] = Math.Min(Math.Max(next[i], lower[i]), upper[i]);   // clamp to the box
            return next;
        });
        if (claimed is null)
        {
            // Every redraw met a held point: sample this trial independently instead of handing out a point another
            // worker is evaluating. It joins no generation, as its generation and index are never stamped.
            _overrun.Add(study, trial.Number);
            return new OrderedDict<string, object?>();
        }

        deParams = claimed;
        study.Storage.SetTrialSystemAttr(trial.TrialId, AttrKeyGeneration, optimizer.Generation);
        study.Storage.SetTrialSystemAttr(trial.TrialId, AttrKeyIndex, target);

        // View the candidate zero-copy; Untransform copies it into a managed dict, so dispose here.
        using NDArray nd = np.array(deParams, copy: false);
        return trans.Untransform(nd);
    }

    /// <summary>Builds the first population from compatible completed trials when enough individuals are available.</summary>
    /// <param name="completedTrials">Completed trial snapshots used as historical observations.</param>
    /// <param name="trans">Search-space transform between external parameter values and the numerical box.</param>
    /// <param name="nDim">Number of numerical decision coordinates.</param>
    /// <param name="popsize">Number of individuals evaluated per generation.</param>
    /// <param name="direction">Whether objective values are minimized or maximized.</param>
    /// <returns>The initialized population optimizer, or null when compatible startup history is insufficient.</returns>
    private De? BuildInitialOptimizer(
        List<FrozenTrial> completedTrials, SearchSpaceTransform trans, int nDim, int popsize, StudyDirection direction)
    {
        // The first NP completed trials (by number) that carry the full relative param set become the
        // initial population; their objective values (minimization) seed the greedy selection.
        var pop = new List<double[]>(popsize);
        var vals = new List<double>(popsize);
        foreach (var t in completedTrials.OrderBy(t => t.Number))
        {
            if (t.Value is null) continue;
            double[] x;
            try { x = trans.Transform(t.Params).ToArray<double>(); }
            catch (Exception) { continue; }
            if (x.Length != nDim) continue;
            pop.Add(x);
            vals.Add(direction == StudyDirection.Minimize ? t.Value.Value : -t.Value.Value);
            if (pop.Count == popsize) break;
        }
        // The startup gate guarantees ≥ popsize completed trials, but a dynamic space could still starve
        // this — fall back to waiting (independent sampling) rather than a malformed optimizer.
        if (pop.Count < popsize) return null;
        return new De(pop.ToArray(), vals.ToArray(), _f, _cr, _strategy, seed: null);
    }

    // ---- optimizer (de)serialization (mirrors LmCmaSampler) ---------------------------------------

    private const string AttrPrefix = "de:";
    private static string AttrKeyGeneration => AttrPrefix + "generation";
    private static string AttrKeyIndex => AttrPrefix + "index";
    private static string AttrKeyOptimizer => AttrPrefix + "optimizer";

    /// <summary>Reassembles serialized optimizer chunks in numeric chunk order.</summary>
    /// <param name="optimizerAttrs">Stored serialized optimizer chunks keyed by chunk identifier.</param>
    /// <returns>The full serialized optimizer string in chunk-index order.</returns>
    private static string ConcatOptimizerAttrs(IReadOnlyDictionary<string, string> optimizerAttrs)
        => string.Concat(Enumerable.Range(0, optimizerAttrs.Count)
            .Select(i => optimizerAttrs[$"{AttrKeyOptimizer}:{i}"]));

    /// <summary>Splits a serialized optimizer into storage-sized system-attribute chunks.</summary>
    /// <param name="optimizerStr">Tagged serialized optimizer state, after any chunk reassembly.</param>
    /// <returns>Chunk-indexed system attributes within the storage string-length limit.</returns>
    private static Dictionary<string, string> SplitOptimizerStr(string optimizerStr)
    {
        int optimizerLen = optimizerStr.Length;
        var attrs = new Dictionary<string, string>();
        int chunks = (int)Math.Ceiling(optimizerLen / (double)SystemAttrMaxLength);
        for (int i = 0; i < chunks; i++)
        {
            int start = i * SystemAttrMaxLength;
            int end = Math.Min((i + 1) * SystemAttrMaxLength, optimizerLen);
            attrs[$"{AttrKeyOptimizer}:{i}"] = optimizerStr[start..end];
        }
        return attrs;
    }

    /// <summary>Encodes the optimizer type and serializable state for storage on a trial.</summary>
    /// <remarks>
    ///     The state holds every population member's fitness, so an infinite objective value reaches it;
    ///     <see cref="OptimizerStateJson" /> keeps ±inf and NaN as named JSON literals instead of failing.
    /// </remarks>
    /// <param name="optimizer">Optimizer whose proposal and adaptation state are persisted.</param>
    /// <returns>A tagged JSON representation of the optimizer state.</returns>
    private static string SerializeOptimizer(De optimizer)
        => "de\n" + OptimizerStateJson.Serialize(optimizer.ToState());

    /// <summary>Decodes a persisted optimizer tag and reconstructs its matching state type.</summary>
    /// <param name="optimizerStr">Tagged serialized optimizer state, after any chunk reassembly.</param>
    /// <returns>A new optimizer reconstructed from the tagged state; the caller may change it freely.</returns>
    /// <exception cref="System.NotSupportedException">The serialized optimizer tag is unknown.</exception>
    /// <exception cref="System.Text.Json.JsonException">The persisted optimizer payload is not valid JSON for its state type, or is the literal null.</exception>
    /// <exception cref="System.ArgumentException">The decoded state contains invalid optimizer constructor settings.</exception>
    /// <exception cref="System.ArgumentOutOfRangeException">The stored text has no <c>&lt;tag&gt;\n</c> line.</exception>
    /// <remarks>Parses the JSON only when the text differs from the last one parsed (<see cref="_parsedState" />); a new optimizer is built from the state every time.</remarks>
    private De DeserializeOptimizer(string optimizerStr)
    {
        // The same stored text as the last parse: the same state, without parsing the JSON again (see _parsedState).
        // The whole text, tag included, is the key, so a hit has passed the tag check before.
        if (_parsedState is { } parsed && string.Equals(parsed.Text, optimizerStr, StringComparison.Ordinal))
            return De.FromState(parsed.State);

        int nl = optimizerStr.IndexOf('\n');
        string tag = optimizerStr[..nl];
        string json = optimizerStr[(nl + 1)..];
        if (tag != "de")
            throw new NotSupportedException($"Unknown optimizer state tag '{tag}'.");
        DeState state = OptimizerStateJson.Deserialize<DeState>(json);
        _parsedState = (optimizerStr, state);
        return De.FromState(state);
    }

    /// <summary>Finds the newest stored optimizer snapshot in completed trial history.</summary>
    /// <param name="completedTrials">Completed trial snapshots used as historical observations.</param>
    /// <returns>The newest available optimizer, or null when no stored snapshot exists.</returns>
    /// <exception cref="System.NotSupportedException">The newest stored snapshot carries an unknown optimizer tag.</exception>
    /// <exception cref="System.Text.Json.JsonException">The newest stored snapshot's payload is not valid JSON for the state type.</exception>
    /// <exception cref="System.ArgumentException">The decoded state contains invalid optimizer constructor settings.</exception>
    private De? RestoreOptimizer(List<FrozenTrial> completedTrials)
    {
        for (int idx = completedTrials.Count - 1; idx >= 0; idx--)
        {
            var trial = completedTrials[idx];
            var optimizerAttrs = trial.SystemAttrs
                .Where(kv => kv.Key.StartsWith(AttrKeyOptimizer, StringComparison.Ordinal))
                .ToDictionary(kv => kv.Key, kv => (string)kv.Value!);
            if (optimizerAttrs.Count == 0) continue;
            return DeserializeOptimizer(ConcatOptimizerAttrs(optimizerAttrs));
        }
        return null;
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">The study has multiple objectives, but this sampler requires a scalar objective.</exception>
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        RaiseErrorIfMultiObjective(study);

        // An overrun trial is independent on purpose (every target was held), so it is not warned about.
        if (_warnIndependentSampling && !_overrun.Contains(study, trial.Number))
        {
            var completeTrials = GetTrials(study);
            if (completeTrials.Count >= _nStartupTrials)
                LogIndependentSampling(trial, paramName);
        }

        return _independentSampler.SampleIndependent(study, trial, paramName, paramDistribution);
    }

    /// <summary>Explains why the current parameter is delegated to the independent sampler.</summary>
    /// <param name="trial">Trial snapshot being examined; copy its contents before modifying them.</param>
    /// <param name="paramName">Name of the parameter being suggested or looked up.</param>
    private void LogIndependentSampling(FrozenTrial trial, string paramName)
    {
        Logger.Warning(string.Format(
            IndependentSamplingWarningTemplate,
            paramName,
            trial.Number,
            _independentSampler.GetType().Name,
            GetType().Name,
            "dynamic search space and `CategoricalDistribution` are not supported by `DifferentialEvolutionSampler`"));
    }

    /// <summary>Collects historical trials eligible for this sampler, applying its completed/pruned-trial policy.</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <returns>Trial snapshots eligible for the sampler history.</returns>
    private List<FrozenTrial> GetTrials(Study study)
    {
        List<FrozenTrial> all = study.GetTrialsInternal(deepcopy: false, states: null, useCache: true);
        // Sized for the whole history (plan 15): nearly every trial is complete, so the list never regrows on an ask.
        var completeTrials = new List<FrozenTrial>(all.Count);
        foreach (var t in all)
            if (t.State == TrialState.Complete)
                completeTrials.Add(t);
        return completeTrials;
    }

    /// <summary>
    ///     The first target of <paramref name="generation" /> that no finished trial and no other running trial of the
    ///     generation holds, so concurrent asks never build two offspring for one target.
    /// </summary>
    /// <param name="study">
    ///     The study, whose complete and running trials are read from the storage in one read (their stamps are fresh
    ///     there, and a trial told since this trial read its history is still seen).
    /// </param>
    /// <param name="trial">The trial being sampled; it holds no target yet.</param>
    /// <param name="generation">The current generation.</param>
    /// <param name="popsize">The population size.</param>
    /// <param name="finished">The finished trials of the generation in the history snapshot, whose targets are taken.</param>
    /// <returns>The free target's index, or <c>null</c> when every target is held.</returns>
    /// <exception cref="NumSharp.KeyError">The study no longer exists in storage.</exception>
    private static int? FirstFreeTarget(
        Study study, FrozenTrial trial, int generation, int popsize, List<FrozenTrial> finished)
    {
        var held = new HashSet<long>();
        foreach (var t in finished)
            held.Add(t.SystemAttrs.TryGetValue(AttrKeyIndex, out var v) ? Convert.ToInt64(v) : -1);
        foreach (var t in ConcurrentAsks.OtherCompleteOrRunningTrials(study, trial))
        {
            long g = t.SystemAttrs.TryGetValue(AttrKeyGeneration, out var gv) ? Convert.ToInt64(gv) : -1;
            if (g == generation)
                held.Add(t.SystemAttrs.TryGetValue(AttrKeyIndex, out var v) ? Convert.ToInt64(v) : -1);
        }

        for (int i = 0; i < popsize; i++)
            if (!held.Contains(i))
                return i;
        return null;
    }

    /// <summary>The offspring of one generation, ordered by their target index (so <c>result[i]</c> is the
    ///     offspring of parent <c>i</c>) — the pairing DE's greedy selection needs.</summary>
    /// <param name="trials">Trial history available to the operation, in its existing order.</param>
    /// <param name="generation">Zero-based generation whose individuals or optimizer state are requested.</param>
    /// <returns>Trials from the requested optimizer generation.</returns>
    private static List<FrozenTrial> GetSolutionTrials(List<FrozenTrial> trials, int generation)
    {
        var result = new List<(FrozenTrial Trial, long Index)>();
        foreach (var t in trials)
        {
            long g = t.SystemAttrs.TryGetValue(AttrKeyGeneration, out var v) ? Convert.ToInt64(v) : -1;
            if (generation != g) continue;
            long idx = t.SystemAttrs.TryGetValue(AttrKeyIndex, out var iv) ? Convert.ToInt64(iv) : -1;
            result.Add((t, idx));
        }
        return result.OrderBy(e => e.Index).Select(e => e.Trial).ToList();
    }

    /// <inheritdoc />
    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _independentSampler.BeforeTrial(study, trial);

    /// <inheritdoc />
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        // Python `if trial.constraints:` — FrozenTrial.Constraints is never null (an empty dict when
        // unset), so warn only when at least one constraint was actually set on the trial.
        if (trial.Constraints.Count > 0)
            OptunaWarnings.Warn(
                "Constraints have been set, but DifferentialEvolutionSampler will not use them for optimization.",
                WarningCategory.UserWarning);
        _overrun.Remove(study, trial.Number);
        _handedOut.Forget(study, trial.Number);
        _independentSampler.AfterTrial(study, trial, state, values);
    }
}
