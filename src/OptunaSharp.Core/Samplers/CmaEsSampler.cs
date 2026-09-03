using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers.CmaEs;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     A sampler using <a href="https://github.com/CyberAgentAILab/cmaes">cmaes</a> as the backend.
///     Port of <c>optuna/samplers/_cmaes.py: CmaEsSampler</c> driving the <see cref="Cma" /> optimizer.
///
///     <para>This sampler does not support <see cref="CategoricalDistribution" />. If the search
///     space includes categorical parameters they are sampled by the independent sampler.</para>
///
///     <para><b>[Misaligned] optimizer state is our own JSON, not Python <c>pickle</c>.</b> Optuna
///     stores <c>pickle.dumps(optimizer).hex()</c> in trial system attrs; OptunaSharp serializes an
///     equivalent <see cref="CmaState" /> JSON under the same chunked <c>cma:optimizer:{i}</c> keys.
///     A study therefore can NOT resume CMA-ES mid-optimization across the Python/C# boundary — but a
///     pure-OptunaSharp study is bit-exact with the Python <c>cmaes</c> package for a shared seed.</para>
/// </summary>
/// <example>
///     <code>
///     var sampler = new CmaEsSampler();
///     var study = Optuna.CreateStudy(sampler: sampler);
///     study.Optimize(t => Math.Pow(t.SuggestFloat("x", -1, 1), 2) + t.SuggestInt("y", -1, 1), nTrials: 20);
///     </code>
/// </example>
public class CmaEsSampler : BaseSampler
{
    // The value of system_attrs must be less than 2046 characters on RDBStorage.
    private const int SystemAttrMaxLength = 2045;
    private const double Eps = 1e-10;

    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers._cmaes");

    private readonly IReadOnlyDictionary<string, object?>? _x0;
    private readonly double? _sigma0;
    private readonly BaseSampler _independentSampler;
    private readonly int _nStartupTrials;
    private bool _warnIndependentSampling;
    private readonly LazyRandomState _cmaRng;
    private readonly IntersectionSearchSpace _searchSpace;
    private readonly bool _considerPrunedTrials;
    private readonly int? _popsize;
    private readonly bool _useSeparableCma;
    private readonly bool _withMargin;
    private readonly bool _lrAdapt;
    private readonly List<FrozenTrial>? _sourceTrials;
    private readonly string _attrPrefix;

    /// <param name="x0">
    ///     Initial parameter values for CMA-ES (external representation). By default the mean of the
    ///     low and high of each distribution. <b>Deprecated in v4.9.0.</b>
    /// </param>
    /// <param name="sigma0">
    ///     Initial standard deviation of CMA-ES. By default <c>min_range / 6</c>. <b>Deprecated in v4.9.0.</b>
    /// </param>
    /// <param name="nStartupTrials">Independent sampling is used until this many trials finish.</param>
    /// <param name="independentSampler">Sampler for parameters outside the relative search space (default <see cref="RandomSampler" />).</param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently.</param>
    /// <param name="seed">Random seed for CMA-ES.</param>
    /// <param name="considerPrunedTrials">If <c>true</c>, PRUNED trials are considered for sampling (experimental).</param>
    /// <param name="restartStrategy"><b>Deprecated in v4.4.0</b> — falls back to <c>null</c>.</param>
    /// <param name="popsize">Population size of CMA-ES.</param>
    /// <param name="incPopsize"><b>Deprecated in v4.4.0</b> — no longer used.</param>
    /// <param name="useSeparableCma">Constrain the covariance matrix to be diagonal (experimental).</param>
    /// <param name="withMargin">Use CMA-ES with margin for discrete distributions (experimental).</param>
    /// <param name="lrAdapt">Use CMA-ES with learning-rate adaptation (experimental).</param>
    /// <param name="sourceTrials">Warm Starting CMA-ES source trials (experimental).</param>
    public CmaEsSampler(
        IReadOnlyDictionary<string, object?>? x0 = null,
        double? sigma0 = null,
        int nStartupTrials = 1,
        BaseSampler? independentSampler = null,
        bool warnIndependentSampling = true,
        int? seed = null,
        bool considerPrunedTrials = false,
        string? restartStrategy = null,
        int? popsize = null,
        int incPopsize = -1,
        bool useSeparableCma = false,
        bool withMargin = false,
        bool lrAdapt = false,
        List<FrozenTrial>? sourceTrials = null)
    {
        if (restartStrategy != null || incPopsize != -1)
        {
            string msg = string.Format(Deprecations.Template, "`restart_strategy`", "4.4.0", "6.0.0");
            OptunaWarnings.Warn(
                msg + " From v4.4.0 onward, `restart_strategy` automatically falls back to " +
                "`None`. `restart_strategy` will be supported in OptunaHub.",
                WarningCategory.FutureWarning);
        }
        if (x0 != null)
            OptunaWarnings.Warn(string.Format(Deprecations.Template, "`x0`", "4.9.0", "6.0.0"),
                WarningCategory.FutureWarning);
        if (sigma0 != null)
            OptunaWarnings.Warn(string.Format(Deprecations.Template, "`sigma0`", "4.9.0", "6.0.0"),
                WarningCategory.FutureWarning);

        _x0 = x0;
        _sigma0 = sigma0;
        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _nStartupTrials = nStartupTrials;
        _warnIndependentSampling = warnIndependentSampling;
        _cmaRng = new LazyRandomState(seed);
        _searchSpace = new IntersectionSearchSpace();
        _considerPrunedTrials = considerPrunedTrials;
        _popsize = popsize;
        _useSeparableCma = useSeparableCma;
        _withMargin = withMargin;
        _lrAdapt = lrAdapt;
        _sourceTrials = sourceTrials;

        _attrPrefix = useSeparableCma ? "sepcma:" : withMargin ? "cmawm:" : "cma:";

        if (considerPrunedTrials) WarnExperimentalArgument("consider_pruned_trials");
        if (useSeparableCma) WarnExperimentalArgument("use_separable_cma");
        if (sourceTrials != null) WarnExperimentalArgument("source_trials");
        if (withMargin) WarnExperimentalArgument("with_margin");
        if (lrAdapt) WarnExperimentalArgument("lr_adapt");

        if (sourceTrials != null && (x0 != null || sigma0 != null))
            throw new ValueError(
                "It is prohibited to pass `source_trials` argument when x0 or sigma0 is specified.");
        if (sourceTrials != null && useSeparableCma)
            throw new ValueError(
                "It is prohibited to pass `source_trials` argument when using separable CMA-ES.");
        if (lrAdapt && (useSeparableCma || withMargin))
            throw new ValueError(
                "It is prohibited to pass `use_separable_cma` or `with_margin` argument when " +
                "using `lr_adapt`.");
        if (useSeparableCma && withMargin)
            throw new ValueError(
                "Currently, we do not support `use_separable_cma=True` and `with_margin=True`.");
    }

    public override void ReseedRng()
    {
        // _cmaRng doesn't require reseeding because the relative sampling reseeds in each trial.
        _independentSampler.ReseedRng();
    }

    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _searchSpace.Calculate(study))
        {
            if (distribution.Single())
                // `cma` cannot handle distributions that contain just a single value, so we skip
                // them. Note that the parameter values for such distributions are sampled in `Trial`.
                continue;

            if (distribution is not FloatDistribution && distribution is not IntDistribution)
                // Categorical distribution is unsupported.
                continue;

            searchSpace[name] = distribution;
        }

        return searchSpace;
    }

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        RaiseErrorIfMultiObjective(study);

        if (searchSpace.Count == 0)
            return new OrderedDict<string, object?>();

        var completedTrials = GetTrials(study);
        if (completedTrials.Count < _nStartupTrials)
            return new OrderedDict<string, object?>();

        // When `with_margin=true`, bounds in discrete dimensions are handled inside `CMAwM`.
        var trans = new SearchSpaceTransform(searchSpace, transformStep: !_withMargin, transform01: true);

        var optimizer = RestoreOptimizer(completedTrials) ?? InitOptimizer(trans, study.Direction);

        int nDim = (int)trans.Bounds.shape[0];
        if (optimizer.Dim != nDim)
        {
            if (_warnIndependentSampling)
            {
                string indName = _independentSampler.GetType().Name;
                Logger.Warning(
                    "`CmaEsSampler` does not support dynamic search space. " +
                    $"`{indName}` is used instead of `CmaEsSampler`.");
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
                double[] x = trans.Transform(t.ParamsDict).ToArray<double>();
                double y = study.Direction == StudyDirection.Minimize ? t.Value!.Value : -t.Value!.Value;
                solutions.Add((x, y));
            }

            optimizer.Tell(solutions);

            // Store optimizer.
            string optimizerStr = SerializeOptimizer(optimizer);
            foreach (var (key, value) in SplitOptimizerStr(optimizerStr))
                study.Storage.SetTrialSystemAttr(trial.TrialId, key, value);
        }

        // Caution: optimizer should update its seed value.
        int seed = (int)_cmaRng.Rng.randint(1, 65536) + trial.Number;
        optimizer.ReseedRng(seed);
        double[] cmaParams = optimizer.Ask();

        study.Storage.SetTrialSystemAttr(trial.TrialId, AttrKeyGeneration, optimizer.Generation);

        return trans.Untransform(np.array(cmaParams));
    }

    private string AttrKeyGeneration => _attrPrefix + "generation";
    private string AttrKeyOptimizer => _attrPrefix + "optimizer";

    private string ConcatOptimizerAttrs(IReadOnlyDictionary<string, string> optimizerAttrs)
        => string.Concat(Enumerable.Range(0, optimizerAttrs.Count)
            .Select(i => optimizerAttrs[$"{AttrKeyOptimizer}:{i}"]));

    private Dictionary<string, string> SplitOptimizerStr(string optimizerStr)
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

    // Our optimizer-state JSON is self-describing (like Optuna's pickle): a `<tag>\n` prefix names
    // the concrete optimizer so restore reconstructs the right type regardless of the sampler's
    // current config — e.g. a single-dimensional `use_separable_cma=True` study stores a fallback
    // `Cma` under the `sepcma:` prefix and must restore it as `Cma`, not `SepCma`. System.Text.Json
    // emits no newlines, so the `\n` split is unambiguous.

    /// <summary>Serialize the optimizer's evolving state to our own JSON (the pickle analogue).</summary>
    private static string SerializeOptimizer(ICmaOptimizer optimizer) => optimizer switch
    {
        Cma c => "cma\n" + JsonSerializer.Serialize(c.ToState()),
        SepCma s => "sepcma\n" + JsonSerializer.Serialize(s.ToState()),
        _ => throw new NotSupportedException($"Cannot serialize optimizer of type {optimizer.GetType().Name}."),
    };

    private static ICmaOptimizer DeserializeOptimizer(string optimizerStr)
    {
        int nl = optimizerStr.IndexOf('\n');
        string tag = optimizerStr[..nl];
        string json = optimizerStr[(nl + 1)..];
        return tag switch
        {
            "cma" => Cma.FromState(JsonSerializer.Deserialize<CmaState>(json)!),
            "sepcma" => SepCma.FromState(JsonSerializer.Deserialize<SepCmaState>(json)!),
            _ => throw new NotSupportedException($"Unknown optimizer state tag '{tag}'."),
        };
    }

    private ICmaOptimizer? RestoreOptimizer(List<FrozenTrial> completedTrials)
    {
        // Restore a previous CMA object.
        for (int idx = completedTrials.Count - 1; idx >= 0; idx--)
        {
            var trial = completedTrials[idx];
            var optimizerAttrs = trial.SystemAttrs
                .Where(kv => kv.Key.StartsWith(AttrKeyOptimizer, StringComparison.Ordinal))
                .ToDictionary(kv => kv.Key, kv => (string)kv.Value!);
            if (optimizerAttrs.Count == 0)
                continue;

            string optimizerStr = ConcatOptimizerAttrs(optimizerAttrs);
            return DeserializeOptimizer(optimizerStr);
        }
        return null;
    }

    private ICmaOptimizer InitOptimizer(SearchSpaceTransform trans, StudyDirection direction)
    {
        double[] boundsFlat = trans.Bounds.ToArray<double>();
        int nDim = boundsFlat.Length / 2;
        var lower = new double[nDim];
        var upper = new double[nDim];
        var boundsArr = new double[nDim, 2];
        for (int i = 0; i < nDim; i++)
        {
            lower[i] = boundsFlat[i * 2];
            upper[i] = boundsFlat[i * 2 + 1];
            boundsArr[i, 0] = lower[i];
            boundsArr[i, 1] = upper[i];
        }

        double[,]? cov = null;
        double[] mean;
        double sigma0;

        if (_sourceTrials == null)
        {
            if (_x0 == null)
            {
                mean = new double[nDim];
                for (int i = 0; i < nDim; i++) mean[i] = lower[i] + (upper[i] - lower[i]) / 2;
            }
            else
            {
                // `_x0` is external representations.
                mean = trans.Transform(_x0).ToArray<double>();
            }

            if (_sigma0 == null)
            {
                // np.min((upper_bounds - lower_bounds) / 6)
                sigma0 = double.PositiveInfinity;
                for (int i = 0; i < nDim; i++)
                {
                    double v = (upper[i] - lower[i]) / 6;
                    if (v < sigma0) sigma0 = v;
                }
            }
            else
            {
                sigma0 = _sigma0.Value;
            }
        }
        else
        {
            throw new NotSupportedException(
                "Warm Starting CMA-ES (`source_trials`) is not yet ported to OptunaSharp; " +
                "it lands in a follow-up increment of Phase 2.16.");
        }

        // Avoid ZeroDivisionError in cmaes.
        sigma0 = Math.Max(sigma0, Eps);

        if (_useSeparableCma)
        {
            if (boundsArr.GetLength(0) == 1)
            {
                OptunaWarnings.Warn(
                    "Separable CMA-ES does not operate meaningfully on single-dimensional " +
                    "search spaces. The setting `use_separable_cma=True` will be ignored.",
                    WarningCategory.UserWarning);
            }
            else
            {
                return new SepCma(
                    mean: mean,
                    sigma: sigma0,
                    bounds: boundsArr,
                    nMaxResampling: 10 * nDim,
                    seed: (int)_cmaRng.Rng.randint(1, 2147483646),
                    populationSize: _popsize);
            }
        }

        if (_withMargin)
            throw new NotSupportedException(
                "CMA-ES with margin (`with_margin`) is not yet ported to OptunaSharp; " +
                "it lands in a follow-up increment of Phase 2.16.");

        int seed = (int)_cmaRng.Rng.randint(1, 2147483646);
        return new Cma(
            mean: mean,
            sigma: sigma0,
            bounds: boundsArr,
            nMaxResampling: 10 * nDim,
            seed: seed,
            populationSize: _popsize,
            cov: cov,
            lrAdapt: _lrAdapt);
    }

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

    private void LogIndependentSampling(FrozenTrial trial, string paramName)
    {
        Logger.Warning(string.Format(
            IndependentSamplingWarningTemplate,
            paramName,
            trial.Number,
            _independentSampler.GetType().Name,
            GetType().Name,
            "dynamic search space and `CategoricalDistribution` are not supported by `CmaEsSampler`"));
    }

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
                // max(t.intermediate_values.items()) — the value at the largest step key.
                int maxStep = t.IntermediateValues.Keys.Max();
                double value = t.IntermediateValues[maxStep];
                // We rewrite the value of the trial `t` for sampling, so we need a copy.
                var copied = t.DeepCopy();
                copied.Value = value;
                completeTrials.Add(copied);
            }
        }
        return completeTrials;
    }

    private List<FrozenTrial> GetSolutionTrials(List<FrozenTrial> trials, int generation)
    {
        var result = new List<FrozenTrial>();
        foreach (var t in trials)
        {
            long g = t.SystemAttrs.TryGetValue(AttrKeyGeneration, out var v) ? Convert.ToInt64(v) : -1;
            if (generation == g)
                result.Add(t);
        }
        return result;
    }

    public override void BeforeTrial(Study study, FrozenTrial trial)
    {
        _independentSampler.BeforeTrial(study, trial);
    }

    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        if (trial.Constraints != null)
            OptunaWarnings.Warn(
                "Constraints have been set, but CmaEsSampler will not use them for optimization.",
                WarningCategory.UserWarning);
        _independentSampler.AfterTrial(study, trial, state, values);
    }

    /// <summary>Python <c>optuna._experimental.warn_experimental_argument</c>.</summary>
    private static void WarnExperimentalArgument(string name)
        => OptunaWarnings.Warn(
            $"Argument ``{name}`` is an experimental feature. The interface can change in the future.",
            WarningCategory.ExperimentalWarning);
}
