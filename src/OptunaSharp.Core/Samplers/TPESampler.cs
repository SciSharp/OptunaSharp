using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.SearchSpace;
using OptunaSharp.Samplers.Tpe;

namespace OptunaSharp.Samplers;

/// <summary>
///     Sampler using the TPE (Tree-structured Parzen Estimator) algorithm — Optuna's default
///     sampler. Port of <c>optuna/samplers/_tpe/sampler.py: TPESampler</c>.
///
///     <para>On each trial, for each parameter, TPE fits one Gaussian Mixture Model <c>l(x)</c> to the
///     parameter values of the best trials, and another GMM <c>g(x)</c> to the rest, then picks the
///     value maximizing <c>l(x)/g(x)</c>.</para>
///
///     <para><b>Scope (PORTING_PLAN Phase 2.8/2.9).</b> The univariate (independent), the
///     non-group multivariate, and (Phase 2.9) the group-decomposed multivariate (<c>group=true</c>)
///     paths are implemented — so a default <see cref="TPESampler" /> works for single- and
///     multi-objective studies. Multi-objective splitting inherits the Phase 2.10 HSSP tie-break gap
///     (an explicit target avoids it).</para>
///
///     <para><b>[Misaligned] erf/ucrt tolerance.</b> The sampled values and acquisition densities
///     route through <see cref="TruncNorm" /> → <see cref="Erf" /> (FreeBSD <c>s_erf</c>), ~1-3 ULP
///     from Python's <c>math.erf</c> (ucrt). So TPE agrees with Optuna to a few ULP, not bit-for-bit —
///     the same bound documented for PED-ANOVA. RNG draw order and count are identical, so the study
///     trajectory stays aligned. See also the argsort tie-break note on <see cref="Tpe.ParzenEstimator" />.</para>
/// </summary>
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html">Official Optuna TPESampler reference.</seealso>
public class TPESampler : BaseSampler
{
    private const string RelativeParamsKey = "tpe:relative_params";
    // The value of system_attrs must be less than 2046 characters on RDBStorage.
    private const int SystemAttrMaxLength = 2045;

    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.samplers._tpe.sampler");

    private readonly ParzenEstimatorParameters _parzenEstimatorParameters;
    private readonly int _nStartupTrials;
    private readonly int _nEiCandidates;
    private Func<int, int>? _gamma;
    private readonly bool _warnIndependentSampling;
    private readonly LazyRandomState _rng;
    private readonly RandomSampler _randomSampler;
    private readonly bool? _multivariate;
    private readonly bool _group;
    private readonly GroupDecomposedSearchSpace? _groupDecomposedSearchSpace;
    private SearchSpaceGroup? _searchSpaceGroup;
    private readonly IntersectionSearchSpace _searchSpace;
    private readonly bool _constantLiar;
    private readonly Func<FrozenTrial, IEnumerable<double>>? _constraintsFunc;

    /// <param name="considerPrior">Deprecated (v4.3.0). Falls back to <c>true</c>.</param>
    /// <param name="priorWeight">Deprecated (v4.9.0). Falls back to <c>1.0</c>.</param>
    /// <param name="considerMagicClip">Deprecated (v4.9.0). Falls back to <c>true</c>.</param>
    /// <param name="considerEndpoints">Deprecated (v4.9.0). Falls back to <c>false</c>.</param>
    /// <param name="nStartupTrials">Random sampling is used until this many trials finish.</param>
    /// <param name="nEiCandidates">Number of candidates for the expected-improvement computation.</param>
    /// <param name="gamma">Deprecated (v4.9.0). Splits finished trials into below/above.</param>
    /// <param name="weights">Deprecated (v4.9.0). Weight of each finished trial.</param>
    /// <param name="seed">Seed for the random number generator.</param>
    /// <param name="multivariate">
    ///     If <c>true</c>, multivariate TPE is used. If <c>null</c> (default), it is <c>true</c> for
    ///     single-objective and <c>false</c> for multi-objective optimization.
    /// </param>
    /// <param name="group">
    ///     If this and <paramref name="multivariate" /> are <c>true</c>, the multivariate TPE with a
    ///     group-decomposed search space is used: the search space is partitioned into maximal
    ///     subspaces from the past trials and each subspace is sampled jointly by multivariate TPE.
    ///     Requires <paramref name="multivariate" /> to not be <c>false</c> (experimental since v2.8.0).
    /// </param>
    /// <param name="warnIndependentSampling">Deprecated (v4.9.0). Falls back to <c>false</c>.</param>
    /// <param name="constantLiar">Penalize running trials to avoid nearby suggestions. Defaults to <c>true</c>.</param>
    /// <param name="constraintsFunc">Deprecated (v5.0.0). Optional constraint function evaluated after each trial.</param>
    /// <summary>Configures density estimation, startup sampling, and optional joint search-space groups.</summary>
    /// <exception cref="NumSharp.ValueError">Grouping is enabled while multivariate estimation is explicitly disabled.</exception>
    public TPESampler(
        bool? considerPrior = null,
        double? priorWeight = null,
        bool? considerMagicClip = null,
        bool? considerEndpoints = null,
        int nStartupTrials = 10,
        int nEiCandidates = 24,
        Func<int, int>? gamma = null,
        Func<int, double[]>? weights = null,
        int? seed = null,
        bool? multivariate = null,
        bool group = false,
        bool? warnIndependentSampling = null,
        bool constantLiar = true,
        Func<FrozenTrial, IEnumerable<double>>? constraintsFunc = null)
    {
        _ = WarnDeprecatedVal("`consider_prior`", considerPrior, true, "4.3.0", "6.0.0");
        double resolvedPriorWeight = WarnDeprecatedVal("`prior_weight`", priorWeight, 1.0, "4.9.0", "6.0.0");
        bool resolvedMagicClip = WarnDeprecatedVal("`consider_magic_clip`", considerMagicClip, true, "4.9.0", "6.0.0");
        bool resolvedEndpoints = WarnDeprecatedVal("`consider_endpoints`", considerEndpoints, false, "4.9.0", "6.0.0");
        Func<int, int>? resolvedGamma = WarnDeprecatedRef<Func<int, int>>("`gamma`", gamma, null, "4.9.0", "6.0.0");
        Func<int, double[]> resolvedWeights = WarnDeprecatedRef(
            "`weights`", weights, TpeTrialSplit.DefaultWeights, "4.9.0", "6.0.0")!;
        bool resolvedWarnIndependent = WarnDeprecatedVal("`warn_independent_sampling`", warnIndependentSampling, false, "4.9.0", "6.0.0");

        _parzenEstimatorParameters = new ParzenEstimatorParameters(
            priorWeight: resolvedPriorWeight,
            considerMagicClip: resolvedMagicClip,
            considerEndpoints: resolvedEndpoints,
            weights: resolvedWeights,
            // The `multivariate` field remains only for historical reasons and is unused.
            multivariate: true);

        _nStartupTrials = nStartupTrials;
        _nEiCandidates = nEiCandidates;
        _gamma = resolvedGamma;

        _warnIndependentSampling = resolvedWarnIndependent;
        _rng = new LazyRandomState(seed);
        _randomSampler = new RandomSampler(seed);

        _multivariate = multivariate;
        _group = group;
        _searchSpace = new IntersectionSearchSpace(includePruned: true);
        _constantLiar = constantLiar;
        _constraintsFunc = constraintsFunc;

        if (group)
        {
            if (multivariate == false)
                throw new ValueError(
                    "``group`` option can only be enabled when ``multivariate`` is enabled.");
            // Optuna raises an ExperimentalWarning (warn_experimental_argument) for the group option.
            // OptunaSharp targets the Optuna 5.x surface and treats it as stable, so it is omitted.
            _groupDecomposedSearchSpace = new GroupDecomposedSearchSpace(includePruned: true);
        }

        if (constraintsFunc is not null)
        {
            string msg = string.Format(
                Deprecations.Template, "`constraints_func`", "5.0.0", "7.0.0");
            OptunaWarnings.Warn(
                $"{msg} Use `optuna.trial.Trial.set_constraint` instead.", WarningCategory.FutureWarning);
        }
    }

    /// <summary>
    ///     A worker instance for <see cref="SamplerConcurrency.PerWorker" />: <paramref name="prototype" />'s configuration
    ///     (shared, all immutable or user functions) with a generator, a startup sampler and search-space caches of its own.
    /// </summary>
    /// <param name="prototype">The study's sampler whose configuration the worker instance takes.</param>
    /// <remarks>
    ///     Not the public constructor: that one re-validates and re-emits the deprecation warnings the user saw once for
    ///     the prototype, which every worker would otherwise print again.
    /// </remarks>
    private TPESampler(TPESampler prototype)
    {
        _parzenEstimatorParameters = prototype._parzenEstimatorParameters;
        _nStartupTrials = prototype._nStartupTrials;
        _nEiCandidates = prototype._nEiCandidates;
        _gamma = prototype._gamma;
        _warnIndependentSampling = prototype._warnIndependentSampling;
        // Fresh generators; the executor reseeds both from OS entropy before each trial (ReseedRng).
        _rng = new LazyRandomState(null);
        _randomSampler = new RandomSampler();
        _multivariate = prototype._multivariate;
        _group = prototype._group;
        // Search-space caches are per instance: IntersectionSearchSpace and its group variant remember the study they
        // last computed for, which a shared cache would let concurrent workers overwrite under each other.
        _searchSpace = new IntersectionSearchSpace(includePruned: true);
        _constantLiar = prototype._constantLiar;
        _constraintsFunc = prototype._constraintsFunc;
        if (_group)
            _groupDecomposedSearchSpace = new GroupDecomposedSearchSpace(includePruned: true);
    }

    /// <summary>
    ///     A new <see cref="TPESampler" /> with this one's configuration, for a <see cref="SamplerConcurrency.PerWorker" />
    ///     worker. TPE draws its candidates from Parzen estimators fitted on the stored history with its own generator, so
    ///     separate instances never correlate (0 duplicate samples in the measured eight-instance runs, multivariate or
    ///     not).
    /// </summary>
    /// <returns>A fresh instance, or <see langword="null" /> for a subclass, which keeps shared access.</returns>
    /// <remarks>
    ///     Footgun: the user functions it shares (<c>gamma</c>, <c>weights</c>, <c>constraintsFunc</c>) then run on several
    ///     workers at once, so they must be thread-safe.
    /// </remarks>
    internal override BaseSampler? CreateWorkerInstance()
        => GetType() == typeof(TPESampler) ? new TPESampler(this) : null;

    /// <inheritdoc />
    public override void ReseedRng()
    {
        // Python: self._rng.rng.seed(), fresh OS entropy: 128 bits through numpy's own key construction,
        // so concurrent workers never share a stream (THREADING.md §6.3).
        OsEntropySeed.Reseed(_rng.Rng);
        _randomSampler.ReseedRng();
    }

    /// <summary>Python <c>_is_multivariate</c>.</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <returns>True if this study uses joint TPE estimation under the configured policy.</returns>
    private bool IsMultivariate(Study study)
    {
        if (_multivariate is not null)
            return _multivariate.Value;
        if (_group)
        {
            // `group` can only be enabled with the multivariate TPE.
            if (study.IsMultiObjective())
                OptunaWarnings.Warn(
                    "``multivariate`` defaults to False for multi-objective optimization, but"
                    + " it is treated as True because ``group`` can only be enabled with the"
                    + " multivariate TPE. Set ``multivariate=True`` explicitly to suppress this"
                    + " warning.");
            return true;
        }
        // By default, the multivariate TPE is used for single-objective optimization and
        // the independent TPE is used for multi-objective optimization.
        return !study.IsMultiObjective();
    }

    /// <inheritdoc />
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        bool multivariate = IsMultivariate(study);
        if (!multivariate)
            return new OrderedDict<string, BaseDistribution>();

        var searchSpace = new OrderedDict<string, BaseDistribution>();
        bool useTrialCache = multivariate || !_constantLiar;

        if (_group)
        {
            _searchSpaceGroup = _groupDecomposedSearchSpace!.Calculate(study, useTrialCache);
            foreach (var subSpace in _searchSpaceGroup.SearchSpaces)
            {
                // Sort keys because Python's string hashing is nondeterministic.
                foreach (var name in subSpace.Keys.OrderBy(k => k, StringComparer.Ordinal))
                {
                    var distribution = subSpace[name];
                    if (distribution.Single())
                        continue;
                    searchSpace[name] = distribution;
                }
            }
            return searchSpace;
        }

        foreach (var (name, distribution) in _searchSpace.Calculate(study, useTrialCache))
        {
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
        OrderedDict<string, object?> @params;
        if (_group)
        {
            @params = new OrderedDict<string, object?>();
            foreach (var subSpace in _searchSpaceGroup!.SearchSpaces)
            {
                var localSearchSpace = new OrderedDict<string, BaseDistribution>();
                // Sort keys because Python's string hashing is nondeterministic.
                foreach (var name in subSpace.Keys.OrderBy(k => k, StringComparer.Ordinal))
                {
                    var distribution = subSpace[name];
                    if (distribution.Single())
                        continue;
                    if (!searchSpace.ContainsKey(name))
                        // When used together with PartialFixedSampler, the search space may be
                        // smaller than what is inferred from the study.
                        continue;
                    localSearchSpace[name] = distribution;
                }

                foreach (var (name, value) in SampleRelativeCore(study, trial, localSearchSpace))
                    @params[name] = value;
            }
        }
        else
        {
            @params = SampleRelativeCore(study, trial, searchSpace);
        }

        if (@params.Count != 0 && _constantLiar)
        {
            // Share the params obtained by the relative sampling with the other processes.
            string paramsStr = PyJson.Dumps(@params);
            for (int i = 0; i < paramsStr.Length; i += SystemAttrMaxLength)
            {
                int len = Math.Min(SystemAttrMaxLength, paramsStr.Length - i);
                study.Storage.SetTrialSystemAttr(
                    trial.TrialId,
                    $"{RelativeParamsKey}:{i / SystemAttrMaxLength}",
                    paramsStr.Substring(i, len));
            }
        }
        return @params;
    }

    /// <summary>Python <c>_sample_relative</c> (called once per subspace in the <c>group</c> path).</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="trial">Trial snapshot being examined; copy its contents before modifying them.</param>
    /// <param name="searchSpace">Ordered parameter distributions defining the coordinates used by this operation.</param>
    /// <returns>Joint TPE values, or an empty map when startup history or the subspace is insufficient.</returns>
    private OrderedDict<string, object?> SampleRelativeCore(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        if (searchSpace.Count == 0)
            return new OrderedDict<string, object?>();

        var states = new[] { TrialState.Complete, TrialState.Pruned };
        var trials = study.GetTrialsInternal(deepcopy: false, states: states, useCache: true);
        // If the number of samples is insufficient, we run random trial.
        if (trials.Count < _nStartupTrials)
            return new OrderedDict<string, object?>();

        return Sample(study, trial, searchSpace);
    }

    /// <inheritdoc />
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        var states = new[] { TrialState.Complete, TrialState.Pruned };
        var trials = study.GetTrialsInternal(deepcopy: false, states: states, useCache: true);

        // If the number of samples is insufficient, we run random trial.
        if (trials.Count < _nStartupTrials)
            return _randomSampler.SampleIndependent(study, trial, paramName, paramDistribution);

        if (_warnIndependentSampling && IsMultivariate(study))
        {
            // Avoid independent warning at the first sampling of `param_name`.
            if (trials.Any(t => t.ParamsDict.ContainsKey(paramName)))
                _logger.Warning(string.Format(
                    IndependentSamplingWarningTemplate,
                    paramName,
                    trial.Number,
                    _randomSampler.GetType().Name,
                    GetType().Name,
                    "`multivariate=True,group=False` does not support dynamic search space" +
                    " (but `multivariate=True,group=True` works)"));
        }

        var singleSpace = new OrderedDict<string, BaseDistribution> { [paramName] = paramDistribution };
        return Sample(study, trial, singleSpace)[paramName];
    }

    /// <summary>Python <c>_get_params</c>.</summary>
    /// <param name="trial">Trial snapshot being examined; copy its contents before modifying them.</param>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <returns>Recorded external parameters supplemented with readable shared relative-sampling attributes.</returns>
    private OrderedDict<string, object?> GetParams(FrozenTrial trial, Study study)
    {
        if (trial.State.IsFinished() || !IsMultivariate(study))
            return trial.ParamsDict;

        var paramsStrs = new List<string>();
        int i = 0;
        while (trial.SystemAttrsDict.TryGetValue($"{RelativeParamsKey}:{i}", out var chunk)
               && chunk is string s && s.Length > 0)
        {
            paramsStrs.Add(s);
            i++;
        }

        if (paramsStrs.Count == 0)
            return trial.ParamsDict;

        OrderedDict<string, object?> @params;
        try
        {
            @params = (OrderedDict<string, object?>)PyJson.Loads(string.Concat(paramsStrs))!;
        }
        catch
        {
            // Incomplete JSON from a concurrent write: fall back to trial.params.
            return trial.ParamsDict;
        }
        foreach (var (k, v) in trial.ParamsDict)
            @params[k] = v;
        return @params;
    }

    /// <summary>Python <c>_get_internal_repr</c>.</summary>
    /// <param name="trials">Trial history available to the operation, in its existing order.</param>
    /// <param name="searchSpace">Ordered parameter distributions defining the coordinates used by this operation.</param>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <returns>Encoded observation columns from trials carrying every search-space parameter.</returns>
    private OrderedDict<string, double[]> GetInternalRepr(
        IReadOnlyList<FrozenTrial> trials, OrderedDict<string, BaseDistribution> searchSpace, Study study)
    {
        var values = new Dictionary<string, List<double>>();
        foreach (var name in searchSpace.Keys)
            values[name] = new List<double>();

        foreach (var trial in trials)
        {
            var @params = GetParams(trial, study);
            // search_space.keys() <= params.keys(): the trial must carry every search-space param.
            if (searchSpace.Keys.All(k => @params.ContainsKey(k)))
            {
                foreach (var (name, distribution) in searchSpace)
                    values[name].Add(distribution.ToInternalRepr(@params[name]));
            }
        }

        var result = new OrderedDict<string, double[]>(searchSpace.Count);
        foreach (var name in searchSpace.Keys)
            result[name] = values[name].ToArray();
        return result;
    }

    /// <summary>Python <c>_sample</c>.</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="trial">Trial snapshot being examined; copy its contents before modifying them.</param>
    /// <param name="searchSpace">Ordered parameter distributions defining the coordinates used by this operation.</param>
    /// <returns>External values of the candidate maximizing the preferred-versus-remaining density ratio.</returns>
    private OrderedDict<string, object?> Sample(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        var states = _constantLiar
            ? new[] { TrialState.Complete, TrialState.Pruned, TrialState.Running }
            : new[] { TrialState.Complete, TrialState.Pruned };
        bool useCache = !_constantLiar;
        var trials = study.GetTrialsInternal(deepcopy: false, states: states, useCache: useCache);

        if (_constantLiar)
            // For constant_liar, filter out the current trial.
            trials = trials.Where(t => trial.Number != t.Number).ToList();

        // Divide data into below and above.
        if (_gamma is null)
            _gamma = study.Directions.Count <= 1
                ? TpeTrialSplit.DefaultGamma
                : TpeTrialSplit.DefaultGammaMultiObjective;

        int n = trials.Count(t => t.State != TrialState.Running);  // Ignore running trials.

        var (belowTrials, aboveTrials) = TpeTrialSplit.SplitTrials(study, trials, _gamma(n));

        var mpeBelow = BuildParzenEstimator(study, searchSpace, belowTrials, handleBelow: true);
        var mpeAbove = BuildParzenEstimator(study, searchSpace, aboveTrials, handleBelow: false);

        var samplesBelow = mpeBelow.Sample(_rng, _nEiCandidates);
        var acqFuncVals = ComputeAcquisitionFunc(samplesBelow, mpeBelow, mpeAbove);
        var ret = Compare(samplesBelow, acqFuncVals);

        var result = new OrderedDict<string, object?>(searchSpace.Count);
        foreach (var (name, dist) in searchSpace)
            result[name] = dist.ToExternalRepr(ret[name]);

        return result;
    }

    /// <summary>Python <c>_build_parzen_estimator</c>.</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="searchSpace">Ordered parameter distributions defining the coordinates used by this operation.</param>
    /// <param name="trials">Trial history available to the operation, in its existing order.</param>
    /// <param name="handleBelow">Whether this estimator represents the preferred trial group.</param>
    /// <returns>The mixture estimator fitted to the selected trial group.</returns>
    private ParzenEstimator BuildParzenEstimator(
        Study study, OrderedDict<string, BaseDistribution> searchSpace,
        IReadOnlyList<FrozenTrial> trials, bool handleBelow)
    {
        var observations = GetInternalRepr(trials, searchSpace, study);
        if (handleBelow && study.IsMultiObjective())
        {
            int nBelow = observations.Count > 0 ? observations[searchSpace.Keys.First()].Length : 0;
            var weightsBelow = new double[nBelow];
            for (int i = 0; i < nBelow; i++)
                weightsBelow[i] = 1.0;
            return new ParzenEstimator(observations, searchSpace, _parzenEstimatorParameters, weightsBelow);
        }
        return new ParzenEstimator(observations, searchSpace, _parzenEstimatorParameters);
    }

    /// <summary>Python <c>_compute_acquisition_func</c>.</summary>
    /// <param name="samples">Candidate values whose density or acquisition scores are evaluated.</param>
    /// <param name="mpeBelow">Density estimator fitted to the preferred trial group.</param>
    /// <param name="mpeAbove">Density estimator fitted to the remaining trial group.</param>
    /// <returns>Log-density ratios aligned with the candidate samples.</returns>
    private static double[] ComputeAcquisitionFunc(
        OrderedDict<string, double[]> samples, ParzenEstimator mpeBelow, ParzenEstimator mpeAbove)
    {
        var logLikelihoodsBelow = mpeBelow.LogPdf(samples);
        var logLikelihoodsAbove = mpeAbove.LogPdf(samples);
        var acqFuncVals = new double[logLikelihoodsBelow.Length];
        for (int i = 0; i < acqFuncVals.Length; i++)
            acqFuncVals[i] = logLikelihoodsBelow[i] - logLikelihoodsAbove[i];
        return acqFuncVals;
    }

    /// <summary>Python <c>_compare</c>: pick the candidate maximizing the acquisition function.</summary>
    /// <param name="samples">Candidate values whose density or acquisition scores are evaluated.</param>
    /// <param name="acquisitionFuncVals">Acquisition values aligned with the candidate sample rows.</param>
    /// <returns>The candidate with maximum acquisition value, keyed by parameter name.</returns>
    /// <exception cref="NumSharp.ValueError">There are no candidates or the candidate and acquisition arrays have different lengths.</exception>
    private static OrderedDict<string, double> Compare(
        OrderedDict<string, double[]> samples, double[] acquisitionFuncVals)
    {
        int sampleSize = samples.Count == 0 ? 0 : samples.Values.First().Length;
        if (sampleSize == 0)
            throw new ValueError($"The size of `samples` must be positive, but got {sampleSize}.");

        if (sampleSize != acquisitionFuncVals.Length)
            throw new ValueError(
                "The sizes of `samples` and `acquisition_func_vals` must be same, but got " +
                "(samples.size, acquisition_func_vals.size) = " +
                $"({sampleSize}, {acquisitionFuncVals.Length}).");

        int bestIdx = ArgMax(acquisitionFuncVals);
        var result = new OrderedDict<string, double>(samples.Count);
        foreach (var (k, v) in samples)
            result[k] = v[bestIdx];
        return result;
    }

    /// <summary>NumPy <c>np.argmax</c>: the first index of the maximum (NaN-max semantics excluded — acq values are finite).</summary>
    /// <param name="values">Scores compared to choose the first maximum.</param>
    /// <returns>The first maximum index.</returns>
    private static int ArgMax(double[] values)
    {
        int best = 0;
        double bestVal = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > bestVal)
            {
                bestVal = values[i];
                best = i;
            }
        }
        return best;
    }

    /// <inheritdoc />
    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _randomSampler.BeforeTrial(study, trial);

    /// <inheritdoc />
    /// <exception cref="System.Exception">A configured after-trial or constraint callback fails; its exception propagates.</exception>
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        System.Diagnostics.Debug.Assert(
            state is TrialState.Complete or TrialState.Fail or TrialState.Pruned);
        if (_constraintsFunc is not null)
            SamplerHelpers.ProcessConstraintsAfterTrial(_constraintsFunc, study, trial, state);
        _randomSampler.AfterTrial(study, trial, state, values);
    }

    /// <summary>Python <c>_warn_if_deprecated_argument</c> — reference-typed arguments (<c>gamma</c>/<c>weights</c>).</summary>
    /// <param name="name">Argument name included in deprecation diagnostics.</param>
    /// <param name="value">Input value used by the comparison or compatibility fallback.</param>
    /// <param name="defaultValue">Fallback used when the deprecated argument was omitted.</param>
    /// <param name="dVer">Version in which the argument was deprecated.</param>
    /// <param name="rVer">Version scheduled to remove the deprecated argument.</param>
    /// <typeparam name="T">Type of the compatibility argument whose supplied value or default is returned.</typeparam>
    /// <returns>The supplied compatibility value, or its default when omitted.</returns>
    private static T? WarnDeprecatedRef<T>(string name, T? value, T? defaultValue, string dVer, string rVer)
        where T : class
    {
        if (value is not null)
        {
            OptunaWarnings.Warn(string.Format(Deprecations.Template, name, dVer, rVer), WarningCategory.FutureWarning);
            return value;
        }
        return defaultValue;
    }

    /// <summary>Python <c>_warn_if_deprecated_argument</c> — value-typed arguments (bool / double).</summary>
    /// <param name="name">Argument name included in deprecation diagnostics.</param>
    /// <param name="value">Input value used by the comparison or compatibility fallback.</param>
    /// <param name="defaultValue">Fallback used when the deprecated argument was omitted.</param>
    /// <param name="dVer">Version in which the argument was deprecated.</param>
    /// <param name="rVer">Version scheduled to remove the deprecated argument.</param>
    /// <typeparam name="T">Type of the compatibility argument whose supplied value or default is returned.</typeparam>
    /// <returns>The supplied compatibility value, or its default when omitted.</returns>
    private static T WarnDeprecatedVal<T>(string name, T? value, T defaultValue, string dVer, string rVer)
        where T : struct
    {
        if (value.HasValue)
        {
            OptunaWarnings.Warn(string.Format(Deprecations.Template, name, dVer, rVer), WarningCategory.FutureWarning);
            return value.Value;
        }
        return defaultValue;
    }
}

/// <summary>Shared deprecation message template (<c>optuna/_deprecated.py</c>).</summary>
internal static class Deprecations
{
    /// <summary>Versioned deprecation message shared by compatibility arguments.</summary>
    internal const string Template =
        "{0} has been deprecated in v{1}. " +
        "This feature will be removed in v{2}. " +
        "See https://github.com/optuna/optuna/releases/tag/v{1}.";
}
