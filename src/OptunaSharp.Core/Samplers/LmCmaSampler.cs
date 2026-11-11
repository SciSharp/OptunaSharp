// Sampler for LM-CMA — Limited-Memory Covariance Matrix Adaptation Evolution Strategy.
//
// Reference paper:
//   Ilya Loshchilov. "LM-CMA: an Alternative to L-BFGS for Large Scale Black-box Optimization."
//   Evolutionary Computation (MIT Press). arXiv:1511.00221 (2015).  https://arxiv.org/abs/1511.00221
//
// This sampler drives the paper-identical `LmCma` optimizer (see LmCma.cs) through the Optuna
// study protocol. The two protocol-forced concessions to the paper (box-clamping and non-mirrored
// per-trial driving) are documented on the type <summary> below.

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers.CmaEs;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     A sampler driving the paper-identical <see cref="LmCma" /> optimizer — Limited-Memory CMA-ES
///     (Ilya Loshchilov, <i>"LM-CMA: an Alternative to L-BFGS for Large Scale Black-box Optimization"</i>,
///     Evolutionary Computation 2015; arXiv:1511.00221). LM-CMA stores only <c>m ≪ n</c> direction vectors
///     and reconstructs the Cholesky factor on the fly (<c>O(mn)</c> memory and per-sample time, no
///     eigendecomposition), which targets very high-dimensional search spaces where full <see cref="Cma" />'s
///     <c>O(n²)</c> covariance is impractical. On low-dimensional / ill-conditioned problems it converges
///     more slowly than full CMA-ES — that is the documented large-scale trade-off, not a defect.
///
///     <para><b>Distinct from <see cref="CmaEsSampler" />.</b> This is a separate, standalone sampler, not a
///     flag on <see cref="CmaEsSampler" />: LM-CMA shares none of that sampler's <c>with_margin</c> /
///     <c>use_separable_cma</c> / warm-start / restart machinery, and reusing it would force-fit features the
///     paper does not define. LM-CMA is <b>not</b> part of the Optuna ecosystem (no <c>cmaes</c> class, no
///     Optuna sampler, no OptunaHub package), so — unlike the other CMA variants — this is a native
///     paper-faithful implementation validated by convergence, with no upstream to bit-match.</para>
///
///     <para><b>Two paper concessions the per-trial protocol forces (both documented on the optimizer):</b>
///     (1) LM-CMA (the paper) is <b>unconstrained</b>; Optuna needs in-box params, so the sampler <b>clamps</b>
///     each asked candidate to the transformed bounds and the next <c>tell</c> re-derives x from the stored
///     (clamped) params — the optimizer learns from the actually-evaluated points. (2) Optuna asks one
///     candidate per trial from freshly-restored state, which cannot carry the intra-generation mirror
///     pairing (Alg. 7 l.9–10), so the sampler drives the <b>non-mirrored</b> generation via
///     <see cref="LmCma.AskCandidate" />, passing the true generation position so SelectSubset's <c>k==1</c>
///     boost still fires once per generation.</para>
///
///     <para>Categorical parameters and single-value distributions are handled by the independent sampler,
///     as in <see cref="CmaEsSampler" />. The optimizer needs <c>n &gt; 1</c>; a single-dimensional relative
///     space falls back to independent sampling.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(sampler: new LmCmaSampler(seed: 42));
///     study.Optimize(t =>
///     {
///         double s = 0;
///         for (int i = 0; i &lt; 20; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
///         return s;
///     }, nTrials: 200);
///     </code>
/// </example>
public sealed class LmCmaSampler : BaseSampler
{
    // system_attrs values must stay under 2046 characters on RDBStorage (same limit as CmaEsSampler).
    private const int SystemAttrMaxLength = 2045;
    private const double Eps = 1e-10;

    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers.lm_cmaes");

    private readonly BaseSampler _independentSampler;
    private readonly int _nStartupTrials;
    private bool _warnIndependentSampling;
    private readonly LazyRandomState _cmaRng;
    private readonly IntersectionSearchSpace _searchSpace;
    private readonly bool _considerPrunedTrials;
    private readonly int? _popsize;
    private readonly int? _nVectors;

    // Per study: the candidate each running trial received (generation, clamped point in the optimizer's unit box), so
    // a point a running trial evaluates is not handed out again. Entries leave with AfterTrial. Guarded by the
    // sampler's monitor, like every hook.
    private readonly HandedOutPoints _handedOut = new();

    // Trials sampled independently on purpose because every redraw met a point a running trial holds; their
    // independent draws are not warned about. Guarded by the sampler's monitor.
    private readonly OverrunTrials _overrun = new();

    /// <param name="nStartupTrials">Independent sampling is used until this many trials finish.</param>
    /// <param name="independentSampler">Sampler for parameters outside the relative search space (default <see cref="RandomSampler" />).</param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently.</param>
    /// <param name="seed">Random seed for LM-CMA.</param>
    /// <param name="considerPrunedTrials">If <c>true</c>, PRUNED trials are considered for sampling (experimental).</param>
    /// <param name="popsize">Population size λ (default <c>4 + ⌊3 ln n⌋</c>, the paper's value).</param>
    /// <param name="nVectors">Number of stored direction vectors m (default <c>4 + ⌊3 ln n⌋</c>; the paper notes ≈√n can do better when memory allows).</param>
    /// <summary>Configures limited-memory covariance adaptation for a fixed numerical search space.</summary>
    public LmCmaSampler(
        int nStartupTrials = 1,
        BaseSampler? independentSampler = null,
        bool warnIndependentSampling = true,
        int? seed = null,
        bool considerPrunedTrials = false,
        int? popsize = null,
        int? nVectors = null)
    {
        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _nStartupTrials = nStartupTrials;
        _warnIndependentSampling = warnIndependentSampling;
        _cmaRng = new LazyRandomState(seed);
        _searchSpace = new IntersectionSearchSpace();
        _considerPrunedTrials = considerPrunedTrials;
        _popsize = popsize;
        _nVectors = nVectors;
    }

    /// <inheritdoc />
    public override void ReseedRng()
    {
        // _cmaRng needs no reseed here: relative sampling reseeds per trial (see SampleRelative).
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
        if (completedTrials.Count < _nStartupTrials)
            return new OrderedDict<string, object?>();

        var trans = new SearchSpaceTransform(searchSpace, transformStep: true, transform01: true);

        // Transformed box bounds — the optimizer works in this space; we clamp to it (LM-CMA is unconstrained).
        double[] boundsFlat = trans.Bounds.ToArray<double>();
        int nDim = boundsFlat.Length / 2;
        var lower = new double[nDim];
        var upper = new double[nDim];
        for (int i = 0; i < nDim; i++) { lower[i] = boundsFlat[i * 2]; upper[i] = boundsFlat[i * 2 + 1]; }

        // LM-CMA requires n > 1 (a rank-one direction store is meaningless in 1-D) → independent sampling.
        if (nDim < 2)
        {
            if (_warnIndependentSampling)
            {
                Logger.Warning(
                    "`LmCmaSampler` requires a search space of dimension > 1. " +
                    $"`{_independentSampler.GetType().Name}` is used instead.");
                _warnIndependentSampling = false;
            }
            return new OrderedDict<string, object?>();
        }

        var optimizer = RestoreOptimizer(completedTrials) ?? InitOptimizer(lower, upper);

        if (optimizer.Dim != nDim)
        {
            if (_warnIndependentSampling)
            {
                Logger.Warning(
                    "`LmCmaSampler` does not support dynamic search space. " +
                    $"`{_independentSampler.GetType().Name}` is used instead of `LmCmaSampler`.");
                _warnIndependentSampling = false;
            }
            return new OrderedDict<string, object?>();
        }

        var solutionTrials = GetSolutionTrials(completedTrials, optimizer.Generation);

        if (solutionTrials.Count >= optimizer.PopulationSize)
        {
            var solutions = new List<(double[] X, double Value)>();
            foreach (var t in solutionTrials.Take(optimizer.PopulationSize))
            {
                // Re-derive x in optimizer space from the stored (clamped) params, so the optimizer learns
                // from the actually-evaluated points. Flip sign for maximization.
                double[] x = trans.Transform(t.Params).ToArray<double>();
                double y = study.Direction == StudyDirection.Minimize ? t.Value!.Value : -t.Value!.Value;
                solutions.Add((x, y));
            }

            optimizer.Tell(solutions);

            string optimizerStr = SerializeOptimizer(optimizer);
            foreach (var (key, value) in SplitOptimizerStr(optimizerStr))
                study.Storage.SetTrialSystemAttr(trial.TrialId, key, value);
        }

        // Each trial reseeds so its single ask is reproducible and differs from its siblings' asks.
        int seed = (int)NumPyDraws.RandInt(_cmaRng.Rng, 1, 65536) + trial.Number;
        optimizer.ReseedRng(seed);

        // Non-mirrored generation: k is this candidate's 1-based position within the current generation.
        int k = solutionTrials.Count + 1;
        double[] cmaParams = optimizer.AskCandidate(k);
        for (int i = 0; i < nDim; i++)
            cmaParams[i] = Math.Min(Math.Max(cmaParams[i], lower[i]), upper[i]);   // clamp to the box

        // A Rademacher candidate is one of few points per generation (mean ± σ along each stored direction), so
        // workers asking at the same time drew the same point: 8-9 of 64 trials at 8 workers. Hand out the first draw
        // no other running trial of this generation is evaluating. A sequential run has no such trial, so it never
        // redraws and its RNG stream, and seeded results, are unchanged; its own repeats of finished points (as many
        // sequentially) are the algorithm's and stay.
        double[]? claimed = _handedOut.Claim(study, trial, optimizer.Generation, cmaParams, () =>
        {
            double[] next = optimizer.AskCandidate(k);
            for (int i = 0; i < nDim; i++)
                next[i] = Math.Min(Math.Max(next[i], lower[i]), upper[i]);   // clamp to the box
            return next;
        });
        if (claimed is null)
        {
            // No free point within the redraws: sample this trial independently instead of handing out a point
            // another worker is evaluating. It joins no generation, as its generation is never stamped.
            _overrun.Add(study, trial.Number);
            return new OrderedDict<string, object?>();
        }

        cmaParams = claimed;
        study.Storage.SetTrialSystemAttr(trial.TrialId, AttrKeyGeneration, optimizer.Generation);

        // View the candidate zero-copy; Untransform copies it into a managed dict, so dispose here.
        using NDArray nd = np.array(cmaParams, copy: false);
        return trans.Untransform(nd);
    }

    private const string AttrPrefix = "lmcma:";
    private static string AttrKeyGeneration => AttrPrefix + "generation";
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
    ///     The state carries the previous generation's fitness values, so an infinite objective value reaches
    ///     it; <see cref="OptimizerStateJson" /> keeps ±inf and NaN as named JSON literals instead of failing.
    /// </remarks>
    /// <param name="optimizer">Optimizer whose proposal and adaptation state are persisted.</param>
    /// <returns>A tagged JSON representation of the optimizer state.</returns>
    private static string SerializeOptimizer(LmCma optimizer)
        => "lmcma\n" + OptimizerStateJson.Serialize(optimizer.ToState());

    /// <summary>
    ///     The last stored optimizer text this instance parsed, and the state parsed from it (plan 15, as
    ///     <c>DifferentialEvolutionSampler</c> keeps it): the text changes once per generation, and every ask in between
    ///     parsed the same JSON again. Guarded by the sampler's monitor, like every hook; a per-worker instance has its
    ///     own.
    /// </summary>
    /// <remarks>
    ///     Safe to share across asks because FromState never changes the state it reads: the <see cref="LmCma" /> rebuilt from it copies every array out of the state (Clone, Array.Copy and row copies) and reads the rest as numbers. A parse that throws
    ///     stores nothing, so the next ask parses (and throws) again, as before.
    /// </remarks>
    private (string Text, LmCmaState State)? _parsedState;

    /// <summary>Decodes a persisted optimizer tag and reconstructs its matching state type.</summary>
    /// <param name="optimizerStr">Tagged serialized optimizer state, after any chunk reassembly.</param>
    /// <returns>A new optimizer reconstructed from the tagged state; the caller may change it freely.</returns>
    /// <exception cref="System.NotSupportedException">The serialized optimizer tag is unknown.</exception>
    /// <exception cref="System.Text.Json.JsonException">The persisted optimizer payload is not valid JSON for its state type, or is the literal null.</exception>
    /// <exception cref="System.ArgumentException">The decoded state contains invalid optimizer constructor settings.</exception>
    /// <exception cref="System.ArgumentOutOfRangeException">The stored text has no <c>&lt;tag&gt;\n</c> line.</exception>
    /// <remarks>Parses the JSON only when the text differs from the last one parsed (<see cref="_parsedState" />); a new optimizer is built from the state every time.</remarks>
    private LmCma DeserializeOptimizer(string optimizerStr)
    {
        // The same stored text as the last parse: the same state, without parsing the JSON again (see _parsedState).
        // The whole text, tag included, is the key, so a hit has passed the tag check before.
        if (_parsedState is { } parsed && string.Equals(parsed.Text, optimizerStr, StringComparison.Ordinal))
            return LmCma.FromState(parsed.State);

        int nl = optimizerStr.IndexOf('\n');
        string tag = optimizerStr[..nl];
        string json = optimizerStr[(nl + 1)..];
        if (tag != "lmcma")
            throw new NotSupportedException($"Unknown optimizer state tag '{tag}'.");
        LmCmaState state = OptimizerStateJson.Deserialize<LmCmaState>(json);
        _parsedState = (optimizerStr, state);
        return LmCma.FromState(state);
    }

    /// <summary>Finds the newest stored optimizer snapshot in completed trial history.</summary>
    /// <param name="completedTrials">Completed trial snapshots used as historical observations.</param>
    /// <returns>The newest available optimizer, or null when no stored snapshot exists.</returns>
    /// <exception cref="System.NotSupportedException">The newest stored snapshot carries an unknown optimizer tag.</exception>
    /// <exception cref="System.Text.Json.JsonException">The newest stored snapshot's payload is not valid JSON for the state type.</exception>
    /// <exception cref="System.ArgumentException">The decoded state contains invalid optimizer constructor settings.</exception>
    private LmCma? RestoreOptimizer(List<FrozenTrial> completedTrials)
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

    /// <summary>Creates the initial optimizer in the transformed numerical box.</summary>
    /// <param name="lower">Lower bound of each transformed numerical coordinate.</param>
    /// <param name="upper">Upper bound of each transformed numerical coordinate.</param>
    /// <returns>The initialized proposal optimizer.</returns>
    private LmCma InitOptimizer(double[] lower, double[] upper)
    {
        int nDim = lower.Length;
        var mean = new double[nDim];
        for (int i = 0; i < nDim; i++) mean[i] = lower[i] + (upper[i] - lower[i]) / 2;   // box midpoint

        double sigma0 = double.PositiveInfinity;                                          // min((hi-lo)/6)
        for (int i = 0; i < nDim; i++) sigma0 = Math.Min(sigma0, (upper[i] - lower[i]) / 6);
        sigma0 = Math.Max(sigma0, Eps);

        int seed = (int)NumPyDraws.RandInt(_cmaRng.Rng, 1, 2147483646);
        return new LmCma(mean, sigma0, seed: seed, populationSize: _popsize, nVectors: _nVectors);
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">The study has multiple objectives, but this sampler requires a scalar objective.</exception>
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        RaiseErrorIfMultiObjective(study);

        // An overrun trial is independent on purpose (no free point was found), so it is not warned about.
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
            "dynamic search space and `CategoricalDistribution` are not supported by `LmCmaSampler`"));
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
        {
            if (t.State == TrialState.Complete)
            {
                completeTrials.Add(t);
            }
            else if (t.State == TrialState.Pruned
                     && t.IntermediateValuesForRead.Count > 0
                     && _considerPrunedTrials)
            {
                long maxStep = t.IntermediateValuesForRead.Keys.Max();
                double value = t.IntermediateValuesForRead[maxStep];
                var copied = t.DeepCopy();
                copied.Value = value;
                completeTrials.Add(copied);
            }
        }
        return completeTrials;
    }

    /// <summary>Selects trials belonging to one optimizer generation so adaptation uses a coherent population.</summary>
    /// <param name="trials">Trial history available to the operation, in its existing order.</param>
    /// <param name="generation">Zero-based generation whose individuals or optimizer state are requested.</param>
    /// <returns>Trials from the requested optimizer generation.</returns>
    private static List<FrozenTrial> GetSolutionTrials(List<FrozenTrial> trials, int generation)
    {
        var result = new List<FrozenTrial>();
        foreach (var t in trials)
        {
            long g = t.SystemAttrs.TryGetValue(AttrKeyGeneration, out var v) ? Convert.ToInt64(v) : -1;
            if (generation == g) result.Add(t);
        }
        return result;
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
                "Constraints have been set, but LmCmaSampler will not use them for optimization.",
                WarningCategory.UserWarning);
        _overrun.Remove(study, trial.Number);
        _handedOut.Forget(study, trial.Number);
        _independentSampler.AfterTrial(study, trial, state, values);
    }
}
