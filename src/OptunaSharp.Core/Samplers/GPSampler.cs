using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Gp;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     Sampler using Gaussian-process-based Bayesian optimization — port of
///     <c>optuna/samplers/_gp/sampler.py: GPSampler</c>. It fits a Matérn-5/2 ARD Gaussian process to
///     the past trials (kernel hyperparameters by marginal-likelihood maximization with the Gamma /
///     hand-crafted priors) and suggests the next parameters by maximizing an acquisition function
///     with the mixed continuous/discrete optimizer (<see cref="OptimMixed" />).
///
///     <para><b>GP stays native (Tier N):</b> no TorchSharp — the fit rides the hand-derived
///     Matérn-5/2 marginal-likelihood gradient and the managed L-BFGS-B (<see cref="LBfgsB" />), and
///     the acquisition optimization the managed acqf gradient. The path is tolerance-exact vs Optuna
///     (ULP drift from scipy's OpenBLAS BLAS + the erf/ucrt class in LogEI), the documented GP boundary
///     — <b>not</b> bit-for-bit like the MT19937 samplers.</para>
///
///     <para><b>Scope (PORTING_PLAN Phase 3.2 part 3d):</b> the single-objective, unconstrained,
///     sequential path (acquisition = LogEI) is ported. Multi-objective (LogEHVI), constrained
///     (LogCEI/LogCEHVI), and running-trial q-batch (qLogEI/qLogEHVI) acquisition functions — which
///     ride <c>box_decomposition</c> and the QMC normal Sobol' sampler — are deferred to a later part
///     and raise <see cref="NotImplementedException" /> for now.</para>
/// </summary>
/// <example>
///     <code>
///     var sampler = new GPSampler(seed: 0);
///     var study = Optuna.CreateStudy(sampler: sampler);
///     study.Optimize(t => Math.Pow(t.SuggestFloat("x", -10, 10) - 2, 2), nTrials: 30);
///     </code>
/// </example>
public class GPSampler : BaseSampler
{
    private const double Eps = 1e-10;

    // Multi-objective and constrained optimization may derive QMC seeds by adding small (~30) offsets
    // to a base seed. Use `1 << 30` as a conservative upper bound. (Unused on the single-objective path.)
    private const int MaxQmcSeedValue = 1 << 30;
    private const string RelativeParamsKey = "gp:relative_params";
    // The value of system_attrs must be less than 2046 characters on RDBStorage.
    private const int SystemAttrMaxLength = 2045;

    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers._gp.sampler");

    private readonly LazyRandomState _rng;
    private readonly BaseSampler _independentSampler;
    private readonly IntersectionSearchSpace _intersectionSearchSpace;
    private readonly int _nStartupTrials;
    private readonly double _minimumNoise;
    private readonly bool _deterministic;
    private readonly Func<FrozenTrial, IEnumerable<double>>? _constraintsFunc;
    private bool _warnIndependentSampling;

    // We cache the kernel parameters for initial values of fitting the next time.
    private List<GPRegressor>? _gprsCacheList;

    // Control parameters of the acquisition function optimization.
    private readonly int _nPreliminarySamples = 2048;
    private readonly int _nLocalSearch = 10;
    private readonly double _tol = 1e-4;

    /// <param name="seed">Random seed to initialize the internal RNG. Defaults to <c>null</c> (picked randomly).</param>
    /// <param name="independentSampler">
    ///     Sampler used for initial sampling (the first <paramref name="nStartupTrials" /> trials) and
    ///     for conditional parameters. Defaults to a <see cref="RandomSampler" /> with the same seed.
    /// </param>
    /// <param name="nStartupTrials">Number of initial trials sampled independently. Defaults to 10.</param>
    /// <param name="deterministicObjective">
    ///     Whether the objective is deterministic. If <c>true</c>, the noise variance is fixed at the
    ///     minimum. Defaults to <c>false</c> (experimental).
    /// </param>
    /// <param name="constraintsFunc">
    ///     Optional constraints function evaluated after each successful trial. <b>Deprecated in
    ///     v5.0.0</b> — use <c>Trial.SetConstraint</c> instead.
    /// </param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently.</param>
    public GPSampler(
        int? seed = null,
        BaseSampler? independentSampler = null,
        int nStartupTrials = 10,
        bool deterministicObjective = false,
        Func<FrozenTrial, IEnumerable<double>>? constraintsFunc = null,
        bool warnIndependentSampling = true)
    {
        _rng = new LazyRandomState(seed);
        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _intersectionSearchSpace = new IntersectionSearchSpace();
        _nStartupTrials = nStartupTrials;
        _minimumNoise = Prior.DefaultMinimumNoiseVar;
        _deterministic = deterministicObjective;
        _constraintsFunc = constraintsFunc;
        _warnIndependentSampling = warnIndependentSampling;

        if (constraintsFunc != null)
            OptunaWarnings.Warn(
                string.Format(Deprecations.Template, "`constraints_func`", "5.0.0", "7.0.0")
                + " Use `optuna.trial.Trial.set_constraint` instead.",
                WarningCategory.FutureWarning);
        if (deterministicObjective)
            WarnExperimentalArgument("deterministic_objective");
    }

    public override void ReseedRng()
    {
        // Python: self._rng.rng.seed() — reseed from OS entropy.
        Span<byte> buffer = stackalloc byte[4];
        System.Security.Cryptography.RandomNumberGenerator.Fill(buffer);
        _rng.Rng.seed(BitConverter.ToUInt32(buffer));
        _independentSampler.ReseedRng();
    }

    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _intersectionSearchSpace.Calculate(study))
        {
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

        var states = new[] { TrialState.Complete, TrialState.Running };
        var trials = study.GetTrialsInternal(deepcopy: false, states: states, useCache: false);
        var completedTrials = trials.Where(t => t.State == TrialState.Complete).ToList();
        var runningTrials = trials
            .Where(t => t.State == TrialState.Running
                        && searchSpace.Keys.All(k => GetParams(t).ContainsKey(k)))
            .ToList();

        if (completedTrials.Count < _nStartupTrials)
            return new OrderedDict<string, object?>();

        var sampled = SampleRelativeImpl(study, completedTrials, runningTrials, searchSpace);

        if (sampled.Count > 0)
        {
            // Share the params obtained by the relative sampling with the other processes.
            // OrderedDict preserves the search-space key order, so this matches json.dumps(params).
            string paramsStr = PyJson.Dumps(sampled);
            for (int i = 0; i < paramsStr.Length; i += SystemAttrMaxLength)
            {
                int len = Math.Min(SystemAttrMaxLength, paramsStr.Length - i);
                study.Storage.SetTrialSystemAttr(
                    trial.TrialId, $"{RelativeParamsKey}:{i / SystemAttrMaxLength}",
                    paramsStr.Substring(i, len));
            }
        }
        return sampled;
    }

    private OrderedDict<string, object?> SampleRelativeImpl(
        Study study, List<FrozenTrial> completedTrials, List<FrozenTrial> runningTrials,
        OrderedDict<string, BaseDistribution> searchSpace)
    {
        var internalSearchSpace = new GpSearchSpace(searchSpace);
        var completedParams = completedTrials
            .Select(t => (IReadOnlyDictionary<string, object?>)t.ParamsDict)
            .ToList();
        double[,] normalizedParams = internalSearchSpace.GetNormalizedParams(completedParams);

        if (runningTrials.Count > 0)
            throw new NotImplementedException(
                "GPSampler with running trials (q-batch acquisition qLogEI/qLogEHVI) is not yet "
                + "ported (PORTING_PLAN Phase 3.2 part 3d covers the single-objective sequential path).");

        var directions = study.Directions;
        int nObjectives = directions.Count;
        var sign = directions.Select(d => d == StudyDirection.Minimize ? -1.0 : 1.0).ToArray();

        int n = completedTrials.Count;
        var values = new double[n, nObjectives];
        for (int t = 0; t < n; t++)
        {
            IReadOnlyList<double> tv = completedTrials[t].Values!;
            for (int j = 0; j < nObjectives; j++)
                values[t, j] = sign[j] * tv[j];
        }
        double[,] standardizedScoreVals = StandardizeValues(values);

        if (_gprsCacheList != null
            && _gprsCacheList[0].InverseSquaredLengthscales.Length != internalSearchSpace.Dim)
            _gprsCacheList = null;  // Clear cache if the search space changes.

        if (ConstrainedOptimization.IsConstrainedOptimization(completedTrials))
            throw new NotImplementedException(
                "GPSampler constrained optimization (LogCEI/LogCEHVI) is not yet ported "
                + "(PORTING_PLAN Phase 3.2 part 3d covers the single-objective unconstrained path).");
        if (nObjectives != 1)
            throw new NotImplementedException(
                "GPSampler multi-objective optimization (LogEHVI) is not yet ported "
                + "(PORTING_PLAN Phase 3.2 part 3d covers the single-objective path).");

        var y = new double[n];
        for (int t = 0; t < n; t++)
            y[t] = standardizedScoreVals[t, 0];

        GPRegressor? cache = _gprsCacheList != null ? _gprsCacheList[0] : null;
        GPRegressor gpr = GPRegressor.FitKernelParams(
            normalizedParams, y, internalSearchSpace.IsCategorical,
            _minimumNoise, _deterministic, cache);
        _gprsCacheList = new List<GPRegressor> { gpr };

        double threshold = y.Max();
        var acqf = new LogEI(gpr, internalSearchSpace, threshold);

        // best_params = normalized_params[argmax(standardized_score_vals[:, 0])][None, :].
        int argmaxIdx = 0;
        for (int t = 1; t < n; t++)
            if (y[t] > y[argmaxIdx]) argmaxIdx = t;
        int dim = internalSearchSpace.Dim;
        var bestParams = new double[1, dim];
        for (int d = 0; d < dim; d++)
            bestParams[0, d] = normalizedParams[argmaxIdx, d];

        (double[] normalizedParam, _) = OptimMixed.OptimizeAcqfMixed(
            acqf, bestParams, _nPreliminarySamples, _nLocalSearch, _tol, _rng.Rng);

        Dictionary<string, object?> unnormalized = internalSearchSpace.GetUnnormalizedParam(normalizedParam);
        var result = new OrderedDict<string, object?>();
        foreach (var name in searchSpace.Keys)
            result[name] = unnormalized[name];
        return result;
    }

    /// <summary>
    ///     <c>_standardize_values</c> — clip non-finite values (<c>warn_and_convert_inf</c>) then
    ///     per-objective standardize <c>(v − mean) / max(EPS, std)</c> (population std, matching
    ///     <c>np.std</c>). Returns only the standardized matrix (the means/stds are used for the
    ///     constraint path, deferred).
    /// </summary>
    private static double[,] StandardizeValues(double[,] values)
    {
        double[,] clipped = WarnAndConvertInf(values);
        int n = clipped.GetLength(0);
        int m = clipped.GetLength(1);
        var result = new double[n, m];
        for (int j = 0; j < m; j++)
        {
            var col = new double[n];
            for (int t = 0; t < n; t++)
                col[t] = clipped[t, j];
            double mean = MathUtil.Mean(col);
            double std = MathUtil.Std(col);
            double denom = Math.Max(Eps, std);
            for (int t = 0; t < n; t++)
                result[t, j] = (col[t] - mean) / denom;
        }
        return result;
    }

    /// <summary>
    ///     <c>warn_and_convert_inf</c> — clip each column's non-finite entries to that column's finite
    ///     min/max (or 0 when a whole column is non-finite). Fast-paths the all-finite case (the common
    ///     one) to return the input unchanged.
    /// </summary>
    private static double[,] WarnAndConvertInf(double[,] values)
    {
        int n = values.GetLength(0);
        int m = values.GetLength(1);
        bool allFinite = true;
        for (int i = 0; i < n && allFinite; i++)
            for (int j = 0; j < m; j++)
                if (!double.IsFinite(values[i, j])) { allFinite = false; break; }
        if (allFinite)
            return values;

        OptunaWarnings.Warn(
            "Clip non-finite values to the min/max finite values for GP fittings.",
            WarningCategory.UserWarning);
        var result = new double[n, m];
        for (int j = 0; j < m; j++)
        {
            double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
            bool anyFinite = false;
            for (int i = 0; i < n; i++)
                if (double.IsFinite(values[i, j]))
                {
                    anyFinite = true;
                    lo = Math.Min(lo, values[i, j]);
                    hi = Math.Max(hi, values[i, j]);
                }
            if (!anyFinite) { lo = 0.0; hi = 0.0; }
            for (int i = 0; i < n; i++)
            {
                double v = values[i, j];
                result[i, j] = double.IsFinite(v) ? v : Math.Min(Math.Max(v, lo), hi);
            }
        }
        return result;
    }

    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        if (_warnIndependentSampling)
        {
            var completedTrials = study.GetTrialsInternal(
                deepcopy: false, states: new[] { TrialState.Complete }, useCache: true);
            if (completedTrials.Count >= _nStartupTrials)
                LogIndependentSampling(trial, paramName);
        }
        return _independentSampler.SampleIndependent(study, trial, paramName, paramDistribution);
    }

    private void LogIndependentSampling(FrozenTrial trial, string paramName)
        => Logger.Warning(string.Format(
            IndependentSamplingWarningTemplate,
            paramName,
            trial.Number,
            _independentSampler.GetType().Name,
            GetType().Name,
            "dynamic search space is not supported by GPSampler"));

    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _independentSampler.BeforeTrial(study, trial);

    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        if (_constraintsFunc != null)
            SamplerHelpers.ProcessConstraintsAfterTrial(_constraintsFunc, study, trial, state);
        _independentSampler.AfterTrial(study, trial, state, values);
    }

    /// <summary>
    ///     <c>_get_params</c> — a finished trial's params, or (for a running trial) the params it shared
    ///     via the chunked <c>gp:relative_params:{i}</c> system attrs, merged with any concrete params.
    /// </summary>
    private static IReadOnlyDictionary<string, object?> GetParams(FrozenTrial trial)
    {
        if (trial.State.IsFinished())
            return trial.ParamsDict;

        var parts = new List<string>();
        int i = 0;
        while (trial.SystemAttrs.TryGetValue($"{RelativeParamsKey}:{i}", out object? v)
               && v is string s && s.Length > 0)
        {
            parts.Add(s);
            i++;
        }
        if (parts.Count == 0)
            return trial.ParamsDict;

        var merged = ParseJsonObject(string.Concat(parts));
        foreach (var kv in trial.ParamsDict)
            merged[kv.Key] = kv.Value;
        return merged;
    }

    private static Dictionary<string, object?> ParseJsonObject(string json)
    {
        var dict = new Dictionary<string, object?>();
        using var doc = JsonDocument.Parse(json);
        foreach (JsonProperty prop in doc.RootElement.EnumerateObject())
            dict[prop.Name] = prop.Value.ValueKind switch
            {
                JsonValueKind.Number => prop.Value.TryGetInt64(out long l) ? l : prop.Value.GetDouble(),
                JsonValueKind.String => prop.Value.GetString(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                _ => null,
            };
        return dict;
    }

    /// <summary>Python <c>optuna._experimental.warn_experimental_argument</c>.</summary>
    private static void WarnExperimentalArgument(string name)
        => OptunaWarnings.Warn(
            $"Argument ``{name}`` is an experimental feature. The interface can change in the future.",
            WarningCategory.ExperimentalWarning);
}
