// Sampler for VkD-CMA — Projection-Based Restricted Covariance Matrix Adaptation for high dimension.
//
// Reference papers:
//   Youhei Akimoto and Nikolaus Hansen. "Projection-Based Restricted Covariance Matrix Adaptation for
//   High Dimension." Proc. of GECCO 2016, pp. 197–204.
//   Youhei Akimoto and Nikolaus Hansen. "Online Model Selection for Restricted Covariance Matrix
//   Adaptation." Proc. of PPSN 2016, pp. 3–13.
//
// This sampler drives the `VkdCma` optimizer (a port of Akimoto's `GaussVkDSampler`; see VkdCma.cs)
// through the Optuna study protocol. The two protocol-forced concessions (box-clamping and single-ask
// per-trial TPA driving) are documented on the type <summary> below and on the optimizer.

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
///     A sampler driving the <see cref="VkdCma" /> optimizer — VkD-CMA-ES (Akimoto &amp; Hansen, GECCO/PPSN
///     2016), a port of Akimoto's own <c>GaussVkDSampler</c>. VkD-CMA restricts the covariance to a
///     <b>rank-<c>k</c> term plus a diagonal</b> and adapts the diagonal, the low-rank basis, AND the rank
///     <c>k</c> online — cost <c>O(N k² + k³)</c> per generation, the middle point between a diagonal-only
///     model (<see cref="CmaEsSampler" /> with <c>useSeparableCma</c>) and full <see cref="CmaEsSampler" />'s
///     <c>O(N²)</c> covariance. It targets high-dimensional problems whose covariance is well-approximated
///     by a few dominant directions, and learns how many directions (<c>k</c>) that is as it runs.
///
///     <para><b>Distinct from <see cref="CmaEsSampler" /> and <see cref="LmCmaSampler" />.</b> Like
///     <see cref="LmCmaSampler" />, this is a separate standalone sampler, not a flag on
///     <see cref="CmaEsSampler" />, and it shares none of that sampler's <c>with_margin</c> /
///     <c>use_separable_cma</c> / warm-start / restart machinery. VkD-CMA is <b>not</b> part of the Optuna
///     ecosystem (no <c>cmaes</c> class, no Optuna sampler, no OptunaHub package), so — like
///     <see cref="LmCmaSampler" /> — it is a native, reference-faithful implementation validated by
///     convergence, with no upstream to bit-match. It sits beside <see cref="LmCmaSampler" /> in the
///     restricted-covariance family (both approximate the covariance; LM-CMA with stored direction
///     vectors, VkD-CMA with an adapting rank-<c>k</c>+diagonal factorization).</para>
///
///     <para><b>Two concessions the per-trial protocol forces (both documented on the optimizer):</b>
///     (1) The optimizer is <b>unconstrained</b>; Optuna needs in-box params, so the sampler <b>clamps</b>
///     each asked candidate to the transformed bounds and the next <c>tell</c> re-derives x from the stored
///     (clamped) params — the optimizer learns from the actually-evaluated points. (2) VkD-CMA adapts its
///     step-size by <b>TPA</b>, which injects a mirrored pair as a generation's first two samples and
///     recovers them <b>by direction</b>; Optuna asks one candidate per trial from freshly-restored state,
///     so the sampler drives the generation through <see cref="VkdCma.AskCandidate" />, passing the true
///     generation position — positions 1 and 2 emit the exact <c>±dx</c> directions (with independent
///     lengths, which TPA's direction recovery does not depend on).</para>
///
///     <para>Categorical parameters and single-value distributions are handled by the independent sampler,
///     as in <see cref="CmaEsSampler" />. The optimizer needs <c>n &gt; 1</c>; a single-dimensional relative
///     space falls back to independent sampling.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(sampler: new VkdCmaSampler(seed: 42));
///     study.Optimize(t =>
///     {
///         double s = 0;
///         for (int i = 0; i &lt; 20; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
///         return s;
///     }, nTrials: 200);
///     </code>
/// </example>
public sealed class VkdCmaSampler : BaseSampler
{
    // system_attrs values must stay under 2046 characters on RDBStorage (same limit as CmaEsSampler).
    private const int SystemAttrMaxLength = 2045;
    private const double Eps = 1e-10;

    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers.vkd_cmaes");

    private readonly BaseSampler _independentSampler;
    private readonly int _nStartupTrials;
    private bool _warnIndependentSampling;
    private readonly LazyRandomState _cmaRng;
    private readonly IntersectionSearchSpace _searchSpace;
    private readonly bool _considerPrunedTrials;
    private readonly int? _popsize;
    private readonly int? _kInit;
    private readonly bool _kAdapt;

    /// <param name="nStartupTrials">Independent sampling is used until this many trials finish.</param>
    /// <param name="independentSampler">Sampler for parameters outside the relative search space (default <see cref="RandomSampler" />).</param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently.</param>
    /// <param name="seed">Random seed for VkD-CMA.</param>
    /// <param name="considerPrunedTrials">If <c>true</c>, PRUNED trials are considered for sampling (experimental).</param>
    /// <param name="popsize">Population size λ (default <c>4 + ⌊3 ln n⌋</c>).</param>
    /// <param name="kInit">Initial rank k of the low-rank covariance term (default 0 — starts diagonal and grows).</param>
    /// <param name="kAdapt">Whether the rank k adapts online (PPSN-2016 model selection); default <c>true</c>.</param>
    /// <summary>Configures covariance adaptation with a diagonal-plus-low-rank model of the numerical search space.</summary>
    public VkdCmaSampler(
        int nStartupTrials = 1,
        BaseSampler? independentSampler = null,
        bool warnIndependentSampling = true,
        int? seed = null,
        bool considerPrunedTrials = false,
        int? popsize = null,
        int? kInit = null,
        bool kAdapt = true)
    {
        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _nStartupTrials = nStartupTrials;
        _warnIndependentSampling = warnIndependentSampling;
        _cmaRng = new LazyRandomState(seed);
        _searchSpace = new IntersectionSearchSpace();
        _considerPrunedTrials = considerPrunedTrials;
        _popsize = popsize;
        _kInit = kInit;
        _kAdapt = kAdapt;
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

        // Transformed box bounds — the optimizer works in this space; we clamp to it (VkD-CMA is unconstrained).
        double[] boundsFlat = trans.Bounds.ToArray<double>();
        int nDim = boundsFlat.Length / 2;
        var lower = new double[nDim];
        var upper = new double[nDim];
        for (int i = 0; i < nDim; i++) { lower[i] = boundsFlat[i * 2]; upper[i] = boundsFlat[i * 2 + 1]; }

        // VkD-CMA requires n > 1 (a rank-k+diagonal model is meaningless in 1-D) → independent sampling.
        if (nDim < 2)
        {
            if (_warnIndependentSampling)
            {
                Logger.Warning(
                    "`VkdCmaSampler` requires a search space of dimension > 1. " +
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
                    "`VkdCmaSampler` does not support dynamic search space. " +
                    $"`{_independentSampler.GetType().Name}` is used instead of `VkdCmaSampler`.");
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
                double[] x = trans.Transform(t.ParamsDict).ToArray<double>();
                double y = study.Direction == StudyDirection.Minimize ? t.Value!.Value : -t.Value!.Value;
                solutions.Add((x, y));
            }

            optimizer.Tell(solutions);

            string optimizerStr = SerializeOptimizer(optimizer);
            foreach (var (key, value) in SplitOptimizerStr(optimizerStr))
                study.Storage.SetTrialSystemAttr(trial.TrialId, key, value);
        }

        // Each trial reseeds so its single ask is reproducible and differs from its siblings' asks.
        int seed = (int)_cmaRng.Rng.randint(1, 65536) + trial.Number;
        optimizer.ReseedRng(seed);

        // TPA generation: k is this candidate's 1-based position within the current generation.
        int k = solutionTrials.Count + 1;
        double[] cmaParams = optimizer.AskCandidate(k);
        for (int i = 0; i < nDim; i++)
            cmaParams[i] = Math.Min(Math.Max(cmaParams[i], lower[i]), upper[i]);   // clamp to the box

        study.Storage.SetTrialSystemAttr(trial.TrialId, AttrKeyGeneration, optimizer.Generation);

        // View the candidate zero-copy; Untransform copies it into a managed dict, so dispose here.
        using NDArray nd = np.array(cmaParams, copy: false);
        return trans.Untransform(nd);
    }

    private const string AttrPrefix = "vkdcma:";
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
    ///     Written through <see cref="OptimizerStateJson" />, so a state that picks up a non-finite value (an
    ///     overflowing log-scale tracker, for example) is stored as named literals instead of failing the ask.
    /// </remarks>
    /// <param name="optimizer">Optimizer whose proposal and adaptation state are persisted.</param>
    /// <returns>A tagged JSON representation of the optimizer state.</returns>
    private static string SerializeOptimizer(VkdCma optimizer)
        => "vkdcma\n" + OptimizerStateJson.Serialize(optimizer.ToState());

    /// <summary>Decodes a persisted optimizer tag and reconstructs its matching state type.</summary>
    /// <param name="optimizerStr">Tagged serialized optimizer state, after any chunk reassembly.</param>
    /// <returns>The optimizer reconstructed from the tagged state.</returns>
    /// <exception cref="System.NotSupportedException">The serialized optimizer tag is unknown.</exception>
    /// <exception cref="System.Text.Json.JsonException">The persisted optimizer payload is not valid JSON for its state type, or is the literal null.</exception>
    /// <exception cref="System.ArgumentException">The decoded state contains invalid optimizer constructor settings.</exception>
    /// <exception cref="System.ArgumentOutOfRangeException">The stored text has no <c>&lt;tag&gt;\n</c> line.</exception>
    private static VkdCma DeserializeOptimizer(string optimizerStr)
    {
        int nl = optimizerStr.IndexOf('\n');
        string tag = optimizerStr[..nl];
        string json = optimizerStr[(nl + 1)..];
        if (tag != "vkdcma")
            throw new NotSupportedException($"Unknown optimizer state tag '{tag}'.");
        return VkdCma.FromState(OptimizerStateJson.Deserialize<VkdCmaState>(json));
    }

    /// <summary>Finds the newest stored optimizer snapshot in completed trial history.</summary>
    /// <param name="completedTrials">Completed trial snapshots used as historical observations.</param>
    /// <returns>The newest available optimizer, or null when no stored snapshot exists.</returns>
    private static VkdCma? RestoreOptimizer(List<FrozenTrial> completedTrials)
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
    private VkdCma InitOptimizer(double[] lower, double[] upper)
    {
        int nDim = lower.Length;
        var mean = new double[nDim];
        for (int i = 0; i < nDim; i++) mean[i] = lower[i] + (upper[i] - lower[i]) / 2;   // box midpoint

        double sigma0 = double.PositiveInfinity;                                          // min((hi-lo)/6)
        for (int i = 0; i < nDim; i++) sigma0 = Math.Min(sigma0, (upper[i] - lower[i]) / 6);
        sigma0 = Math.Max(sigma0, Eps);

        int seed = (int)_cmaRng.Rng.randint(1, 2147483646);
        return new VkdCma(mean, sigma0, seed: seed, populationSize: _popsize, kInit: _kInit, kAdapt: _kAdapt);
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
            "dynamic search space and `CategoricalDistribution` are not supported by `VkdCmaSampler`"));
    }

    /// <summary>Collects historical trials eligible for this sampler, applying its completed/pruned-trial policy.</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <returns>Trial snapshots eligible for the sampler history.</returns>
    private List<FrozenTrial> GetTrials(Study study)
    {
        var completeTrials = new List<FrozenTrial>();
        foreach (var t in study.GetTrialsInternal(deepcopy: false, states: null, useCache: true))
        {
            if (t.State == TrialState.Complete)
            {
                completeTrials.Add(t);
            }
            else if (t.State == TrialState.Pruned
                     && t.IntermediateValues.Count > 0
                     && _considerPrunedTrials)
            {
                int maxStep = t.IntermediateValues.Keys.Max();
                double value = t.IntermediateValues[maxStep];
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
                "Constraints have been set, but VkdCmaSampler will not use them for optimization.",
                WarningCategory.UserWarning);
        _independentSampler.AfterTrial(study, trial, state, values);
    }
}
