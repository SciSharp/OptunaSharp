// Sampler for Particle Swarm Optimization (PSO) — Kennedy & Eberhart's canonical scalar swarm baseline.
//
// References:
//   James Kennedy and Russell Eberhart. "Particle Swarm Optimization." Proc. IEEE Int. Conf. on
//   Neural Networks, 1995.
//   Yuhui Shi and Russell Eberhart. "A Modified Particle Swarm Optimizer." Proc. IEEE Congress on
//   Evolutionary Computation, 1998 (the linearly-decreasing inertia weight).
//
// This sampler drives the textbook-faithful `Pso` optimizer (see Pso/Pso.cs) through the Optuna study
// protocol, following the `DifferentialEvolutionSampler` idiom exactly: the optimizer state is serialized
// into trial system attrs (so a PSO study resumes across process boundaries / distributed workers), each
// trial re-seeds the optimizer and asks a single candidate, and every asked candidate is clamped to the
// transformed unit box. PSO is NOT part of the Optuna ecosystem (no `cmaes` class, no Optuna sampler, no
// OptunaHub package), so — like `DifferentialEvolutionSampler` / `LmCmaSampler` — this is a native
// implementation validated by convergence, with no upstream trajectory to bit-match. Plan 07 baseline.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers.ParticleSwarm;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     A sampler driving a textbook-faithful global-best <see cref="Pso" /> Particle Swarm optimizer
///     (Kennedy &amp; Eberhart 1995; Shi &amp; Eberhart 1998 inertia weight) through the Optuna protocol.
///     PSO maintains a swarm of particles, each carrying a position, a velocity and a personal-best
///     memory, and the swarm shares one global best; each particle's velocity is nudged toward its
///     personal best and the global best (with an inertia term), giving a robust derivative-free scalar
///     optimizer that complements <see cref="CmaEsSampler" /> and <see cref="DifferentialEvolutionSampler" />
///     as a benchmark baseline.
///
///     <para><b>Bootstrapping.</b> PSO needs an initial swarm before it can move particles, so the first
///     <c>max(nStartupTrials, populationSize)</c> completed trials are drawn by the independent sampler and
///     become the initial swarm (velocities start at rest); swarm moves are proposed from then on.</para>
///
///     <para><b>Two protocol concessions (as <see cref="DifferentialEvolutionSampler" />):</b> (1) the
///     canonical PSO velocity/position update is <b>unconstrained</b>, so each asked candidate is
///     <b>clamped</b> to the transformed unit box and the next generation recovers each particle's realized
///     velocity as <c>x_evaluated − x_previous</c> (a velocity reset on bound contact — a standard PSO
///     bound-handling rule). (2) The per-trial protocol asks one candidate per trial from freshly-restored
///     state; each trial is stamped with its <c>generation</c> and its <c>target index</c> so the swarm's
///     update pairs each evaluated position with the right particle once the generation completes.</para>
///
///     <para><b>Inertia schedule.</b> Optuna is trial-budget-driven and does not expose a maximum
///     generation to the sampler, so the linearly-decreasing inertia weight decays from <c>w</c> to
///     <c>wMin</c> over <c>inertiaDecayGenerations</c> generations (constant at <c>wMin</c> thereafter).
///     Categorical parameters and single-value distributions are handled by the independent sampler, as in
///     <see cref="CmaEsSampler" />.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(sampler: new PsoSampler(seed: 42));
///     study.Optimize(t =>
///     {
///         double s = 0;
///         for (int i = 0; i &lt; 10; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
///         return s;
///     }, nTrials: 400);
///     </code>
/// </example>
public sealed class PsoSampler : BaseSampler
{
    // system_attrs values must stay under 2046 characters on RDBStorage (same limit as CmaEsSampler).
    private const int SystemAttrMaxLength = 2045;

    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers.pso");

    private readonly BaseSampler _independentSampler;
    private readonly int _nStartupTrials;
    private bool _warnIndependentSampling;
    private readonly LazyRandomState _psoRng;
    private readonly IntersectionSearchSpace _searchSpace;
    private readonly int? _popsize;
    private readonly double _w0;
    private readonly double _w1;
    private readonly int _inertiaDecayGenerations;
    private readonly double _c1;
    private readonly double _c2;
    private readonly double _maxVelocityRate;

    /// <param name="nStartupTrials">Minimum independent trials before PSO begins; PSO always waits for at
    ///     least <c>populationSize</c> completed trials to seed the initial swarm, whichever is larger.</param>
    /// <param name="independentSampler">Sampler for parameters outside the relative search space (default <see cref="RandomSampler" />).</param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently.</param>
    /// <param name="seed">Random seed for PSO.</param>
    /// <param name="popsize">Swarm size (default <c>max(10, 4 + ⌊3 ln d⌋)</c>; must be ≥ 2).</param>
    /// <param name="w">Initial inertia weight (Shi &amp; Eberhart's <c>w_max</c>, default 0.9).</param>
    /// <param name="wMin">Final inertia weight (Shi &amp; Eberhart's <c>w_min</c>, default 0.4).</param>
    /// <param name="inertiaDecayGenerations">Generations over which the inertia weight decays linearly from
    ///     <paramref name="w" /> to <paramref name="wMin" /> (default 100).</param>
    /// <param name="c1">Cognitive (personal-best) acceleration coefficient (default 1.49445).</param>
    /// <param name="c2">Social (global-best) acceleration coefficient (default 1.49445).</param>
    /// <param name="maxVelocityRate">Per-coordinate velocity clamp as a fraction of the box width (default 0.2).</param>
    /// <summary>Configures particle velocities, inertia decay, and independent startup sampling.</summary>
    /// <exception cref="System.ArgumentException">The requested population size is below two or maximum velocity rate is nonpositive.</exception>
    public PsoSampler(
        int nStartupTrials = 1,
        BaseSampler? independentSampler = null,
        bool warnIndependentSampling = true,
        int? seed = null,
        int? popsize = null,
        double w = 0.9,
        double wMin = 0.4,
        int inertiaDecayGenerations = 100,
        double c1 = 1.49445,
        double c2 = 1.49445,
        double maxVelocityRate = 0.2)
    {
        if (popsize is int p && p < 2)
            throw new ArgumentException("populationSize must be at least 2.", nameof(popsize));
        if (maxVelocityRate <= 0.0)
            throw new ArgumentException("maxVelocityRate must be positive.", nameof(maxVelocityRate));

        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _nStartupTrials = nStartupTrials;
        _warnIndependentSampling = warnIndependentSampling;
        _psoRng = new LazyRandomState(seed);
        _searchSpace = new IntersectionSearchSpace();
        _popsize = popsize;
        _w0 = w;
        _w1 = wMin;
        _inertiaDecayGenerations = inertiaDecayGenerations;
        _c1 = c1;
        _c2 = c2;
        _maxVelocityRate = maxVelocityRate;
    }

    /// <inheritdoc />
    public override void ReseedRng()
    {
        // _psoRng needs no reseed here: relative sampling reseeds per trial (see SampleRelative).
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

        // The optimizer works in the transformed unit box [0,1]^d; clamp to that box (PSO is unconstrained).
        var trans = new SearchSpaceTransform(searchSpace, transformStep: true, transform01: true);
        double[] boundsFlat = trans.Bounds.ToArray<double>();
        int nDim = boundsFlat.Length / 2;
        var lower = new double[nDim];
        var upper = new double[nDim];
        for (int i = 0; i < nDim; i++) { lower[i] = boundsFlat[i * 2]; upper[i] = boundsFlat[i * 2 + 1]; }

        int popsize = _popsize ?? Pso.DefaultPopulationSize(nDim);

        // PSO needs a full initial swarm before it can move particles.
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
                    "`PsoSampler` does not support dynamic search space. " +
                    $"`{_independentSampler.GetType().Name}` is used instead of `PsoSampler`.");
                _warnIndependentSampling = false;
            }
            return new OrderedDict<string, object?>();
        }

        // Complete a generation once all NP target-ordered particles have finished, then re-serialize.
        var solutionTrials = GetSolutionTrials(completedTrials, optimizer.Generation);
        if (solutionTrials.Count >= optimizer.PopulationSize)
        {
            var solutions = new List<(double[] X, double Value)>(optimizer.PopulationSize);
            foreach (var t in solutionTrials.Take(optimizer.PopulationSize))
            {
                double[] x = trans.Transform(t.ParamsDict).ToArray<double>();
                double y = study.Direction == StudyDirection.Minimize ? t.Value!.Value : -t.Value!.Value;
                solutions.Add((x, y));
            }
            optimizer.Tell(solutions);

            string optimizerStr = SerializeOptimizer(optimizer);
            foreach (var (key, value) in SplitOptimizerStr(optimizerStr))
                study.Storage.SetTrialSystemAttr(trial.TrialId, key, value);

            solutionTrials = GetSolutionTrials(completedTrials, optimizer.Generation);
        }

        // Target index (particle) within the current generation (0-based). Clamp guards a mild parallel overrun.
        int target = Math.Min(solutionTrials.Count, optimizer.PopulationSize - 1);

        // Each trial reseeds so its single ask is reproducible and differs from its siblings' asks.
        int seed = (int)_psoRng.Rng.randint(1, 65536) + trial.Number;
        optimizer.ReseedRng(seed);

        double[] psoParams = optimizer.AskCandidate(target);
        for (int i = 0; i < nDim; i++)
            psoParams[i] = Math.Min(Math.Max(psoParams[i], lower[i]), upper[i]);   // clamp to the box

        study.Storage.SetTrialSystemAttr(trial.TrialId, AttrKeyGeneration, optimizer.Generation);
        study.Storage.SetTrialSystemAttr(trial.TrialId, AttrKeyIndex, target);

        return trans.Untransform(np.array(psoParams));
    }

    /// <summary>Builds the first population from compatible completed trials when enough individuals are available.</summary>
    /// <param name="completedTrials">Completed trial snapshots used as historical observations.</param>
    /// <param name="trans">Search-space transform between external parameter values and the numerical box.</param>
    /// <param name="nDim">Number of numerical decision coordinates.</param>
    /// <param name="popsize">Number of individuals evaluated per generation.</param>
    /// <param name="direction">Whether objective values are minimized or maximized.</param>
    /// <returns>The initialized population optimizer, or null when compatible startup history is insufficient.</returns>
    private Pso? BuildInitialOptimizer(
        List<FrozenTrial> completedTrials, SearchSpaceTransform trans, int nDim, int popsize, StudyDirection direction)
    {
        // The first NP completed trials (by number) that carry the full relative param set become the
        // initial swarm; their objective values (minimization) seed the personal/global bests.
        var pop = new List<double[]>(popsize);
        var vals = new List<double>(popsize);
        foreach (var t in completedTrials.OrderBy(t => t.Number))
        {
            if (t.Value is null) continue;
            double[] x;
            try { x = trans.Transform(t.ParamsDict).ToArray<double>(); }
            catch (Exception) { continue; }
            if (x.Length != nDim) continue;
            pop.Add(x);
            vals.Add(direction == StudyDirection.Minimize ? t.Value.Value : -t.Value.Value);
            if (pop.Count == popsize) break;
        }
        // The startup gate guarantees ≥ popsize completed trials, but a dynamic space could still starve
        // this — fall back to waiting (independent sampling) rather than a malformed optimizer.
        if (pop.Count < popsize) return null;
        return new Pso(pop.ToArray(), vals.ToArray(),
            _w0, _w1, _inertiaDecayGenerations, _c1, _c2, _maxVelocityRate, seed: null);
    }

    // ---- optimizer (de)serialization (mirrors DifferentialEvolutionSampler) -----------------------

    private const string AttrPrefix = "pso:";
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
    /// <param name="optimizer">Optimizer whose proposal and adaptation state are persisted.</param>
    /// <returns>A tagged JSON representation of the optimizer state.</returns>
    private static string SerializeOptimizer(Pso optimizer)
        => "pso\n" + JsonSerializer.Serialize(optimizer.ToState());

    /// <summary>Decodes a persisted optimizer tag and reconstructs its matching state type.</summary>
    /// <param name="optimizerStr">Tagged serialized optimizer state, after any chunk reassembly.</param>
    /// <returns>The optimizer reconstructed from the tagged state.</returns>
    /// <exception cref="System.NotSupportedException">The serialized optimizer tag is unknown.</exception>
    /// <exception cref="System.Text.Json.JsonException">The persisted optimizer payload is not valid JSON for its state type.</exception>
    /// <exception cref="System.ArgumentException">The decoded state contains invalid optimizer constructor settings.</exception>
    private static Pso DeserializeOptimizer(string optimizerStr)
    {
        int nl = optimizerStr.IndexOf('\n');
        string tag = optimizerStr[..nl];
        string json = optimizerStr[(nl + 1)..];
        if (tag != "pso")
            throw new NotSupportedException($"Unknown optimizer state tag '{tag}'.");
        return Pso.FromState(JsonSerializer.Deserialize<PsoState>(json)!);
    }

    /// <summary>Finds the newest stored optimizer snapshot in completed trial history.</summary>
    /// <param name="completedTrials">Completed trial snapshots used as historical observations.</param>
    /// <returns>The newest available optimizer, or null when no stored snapshot exists.</returns>
    private static Pso? RestoreOptimizer(List<FrozenTrial> completedTrials)
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

        if (_warnIndependentSampling)
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
            "dynamic search space and `CategoricalDistribution` are not supported by `PsoSampler`"));
    }

    /// <summary>Collects historical trials eligible for this sampler, applying its completed/pruned-trial policy.</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <returns>Trial snapshots eligible for the sampler history.</returns>
    private List<FrozenTrial> GetTrials(Study study)
    {
        var completeTrials = new List<FrozenTrial>();
        foreach (var t in study.GetTrialsInternal(deepcopy: false, states: null, useCache: true))
            if (t.State == TrialState.Complete)
                completeTrials.Add(t);
        return completeTrials;
    }

    /// <summary>The particles of one generation, ordered by their target index (so <c>result[i]</c> is
    ///     particle <c>i</c>) — the pairing PSO's swarm update needs.</summary>
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
        // FrozenTrial.Constraints is an (empty-when-unset) dictionary, never null, so warn only when at
        // least one constraint was actually recorded.
        if (trial.Constraints.Count > 0)
            OptunaWarnings.Warn(
                "Constraints have been set, but PsoSampler will not use them for optimization.",
                WarningCategory.UserWarning);
        _independentSampler.AfterTrial(study, trial, state, values);
    }
}
