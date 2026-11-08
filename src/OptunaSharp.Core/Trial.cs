using System;
using System.Collections.Generic;
using System.Threading;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp;

/// <summary>
///     A trial is a process of evaluating an objective function.
///     Port of <c>optuna/trial/_trial.py: Trial</c>.
///
///     This object is passed to an objective function and provides interfaces to get parameter
///     suggestion, manage the trial's state, and set/get user-defined attributes of the trial.
///
///     Note that the direct use of this constructor is not recommended. This object is
///     seamlessly instantiated and passed to the objective function behind the
///     <see cref="Study.Optimize(Func{Trial, double}, int?, double?, int, Type[], IEnumerable{Action{Study, FrozenTrial}}, bool, bool, TrialExecutorOptions, CancellationToken)" />
///     method; hence library users do not care about instantiation of this object.
/// </summary>
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html">Official Optuna active-trial contract.</seealso>
public class Trial : BaseTrial
{
    /// <summary>The parent <see cref="Study" />.</summary>
    public Study Study { get; }

    /// <summary>
    ///     The cancellation token of the <c>Optimize</c> call running this trial (its <c>ct</c> argument), or
    ///     <see cref="CancellationToken.None" /> for a trial from <see cref="Study.Ask(OrderedDict{string, BaseDistribution})" />
    ///     or one built directly. OptunaSharp only: Python interrupts a running objective with <c>KeyboardInterrupt</c>
    ///     instead.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         An objective may poll it (<see cref="CancellationToken.ThrowIfCancellationRequested" />) or hand it to
    ///         cancellable work, so that a long evaluation stops when the caller cancels the run. An
    ///         <see cref="OperationCanceledException" /> that escapes the objective while the token is cancelled stores
    ///         this trial as FAIL, as Python stores the trial its <c>KeyboardInterrupt</c> lands in, skips the callbacks,
    ///         and ends the <c>Optimize</c> call whatever its <c>catch</c> argument holds.
    ///     </para>
    ///     <para>
    ///         An objective that ignores the token runs to its end and is stored as it returns; the run stops before the
    ///         next trial. Observing the token is the objective's choice, as in any cooperative .NET cancellation.
    ///     </para>
    /// </remarks>
    public CancellationToken CancellationToken { get; internal set; }

    /// <summary>Storage identifier for this trial, separate from its per-study number.</summary>
    internal readonly int TrialId;
    /// <summary>Storage backend shared with the parent study.</summary>
    internal readonly BaseStorage Storage;

    private readonly FrozenTrial _cachedFrozenTrial;
    private OrderedDict<string, BaseDistribution>? _relativeSearchSpace;
    private OrderedDict<string, object?>? _relativeParams;
    private readonly OrderedDict<string, object?> _fixedParams;

    // Serializes this trial's operations (suggest, report, prune check, attributes, constraint) when several
    // threads drive one Trial. Optuna's Trial is not thread-safe: its "already reported?" check and the write
    // after it are separate steps, so racing reporters stored the LAST value of a step instead of the first
    // (5-23% of contended steps here, 63% in Python 5.0), and the cache dictionaries were written concurrently.
    // Uncontended in the usual one-thread-per-trial use, so single-threaded behavior is unchanged.
    // Lock order: this monitor, then the pruner's or sampler's SyncRoot, then the storage's own lock.
    // Always entered through MonitorGate.Scope, never a plain lock: it comes before the Python GIL in that order, so a
    // thread holding the GIL must release it while it waits here, or a second thread driving this Trial could hold
    // this monitor while it waits for a sampler whose holder needs the GIL (P3-1).
    private readonly object _syncRoot = new();

    // The worker's own sampler instance when this trial was asked by a worker of an Optimize(nJobs) call with
    // SamplerConcurrency.PerWorker, bound here so a trial handed to another thread still samples with the instance that
    // ran its BeforeTrial; null otherwise, and then every call reads Study.Sampler afresh, as before.
    private readonly BaseSampler? _workerSampler;

    /// <summary>
    ///     The sampler for this trial's next call: its worker's instance (<see cref="SamplerConcurrency.PerWorker" />), or
    ///     <see cref="Study.Sampler" /> read now, so a sampler swapped on the study takes effect as it always has.
    /// </summary>
    private BaseSampler CurrentSampler => _workerSampler ?? Study.Sampler;

    // The record the current thread's study trial cache may hold for this trial, for MirrorIntoThreadCache: first
    // the storage's own object as of construction, then this trial's private copy once it replaced that object in
    // the cache. Python's Trial mutates the storage object it was created from; the port mutates a detached copy
    // (see the constructor) and mirrors the visibility through this alias instead.
    private FrozenTrial _cacheAlias;

    // True once _cacheAlias is this trial's private copy in the current thread's cache (written only here and safe
    // to mutate in place); false while it is still the shared storage object, which must never be mutated.
    private bool _cacheAliasIsOwnCopy;

    /// <param name="study">Study whose history and storage are used by the operation.</param>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <summary>Binds to an existing trial record and invokes the sampler's BeforeTrial hook.</summary>
    /// <exception cref="NumSharp.KeyError">The trial identifier does not exist in storage.</exception>
    /// <exception cref="System.Exception">The configured storage or sampler hook fails; its original exception propagates.</exception>
    /// <remarks>
    ///     The trial keeps a private copy of the stored record. Python's <c>Trial</c> keeps the storage's own
    ///     object and later mutates it, after the storage has already replaced it with a copy-on-write update.
    ///     Other threads may still hold that object from <c>GetAllTrials(deepcopy: false)</c>, so the mutation
    ///     was a data race on its dictionaries.
    /// </remarks>
    public Trial(Study study, int trialId)
    {
        Study = study;
        TrialId = trialId;

        Storage = study.Storage;

        FrozenTrial stored = Storage.GetTrial(TrialId);
        _cachedFrozenTrial = DetachedCopy(stored);
        _cacheAlias = stored;
        _workerSampler = study.ThreadLocal.SamplerOverride;
        var filteredStudy = PrunerHelpers.FilterStudy(Study, _cachedFrozenTrial);

        // Sampler monitors are entered through the sampler's EnterMonitor, so a Python-bridged sampler can release the
        // GIL while it waits: an objective that holds Py.GIL() must never block on the monitor (P3-1).
        var sampler = CurrentSampler;
        sampler.Locked(() => sampler.BeforeTrial(filteredStudy, _cachedFrozenTrial));

        // NOTE(not522): Evaluate it lazily to get as latest search as possible.
        _relativeSearchSpace = null;
        _relativeParams = null;
        _fixedParams = _cachedFrozenTrial.SystemAttrsDict.GetValueOrDefault("fixed_params")
            as OrderedDict<string, object?> ?? new OrderedDict<string, object?>();
    }

    /// <summary>Gets the inferred relative search space, or null before lazy sampling begins.</summary>
    internal OrderedDict<string, BaseDistribution>? RelativeSearchSpace => _relativeSearchSpace;

    /// <summary>Gets and lazily computes the jointly sampled parameter values.</summary>
    /// <exception cref="System.Exception">The configured sampler fails while inferring or sampling its relative search space.</exception>
    internal OrderedDict<string, object?> RelativeParams
    {
        get
        {
            using (MonitorGate.Scope(_syncRoot))
            {
                if (_relativeParams is null)
                {
                    var study = PrunerHelpers.FilterStudy(Study, _cachedFrozenTrial);
                    // Inference and sampling form one sampler call, as in Python: a concurrent trial must not
                    // change the sampler's state between the two.
                    var sampler = CurrentSampler;
                    sampler.Locked(() =>
                    {
                        _relativeSearchSpace = sampler.InferRelativeSearchSpace(study, _cachedFrozenTrial);
                        _relativeParams = sampler.SampleRelative(study, _cachedFrozenTrial, _relativeSearchSpace);
                    });
                }
                return _relativeParams;
            }
        }
    }

    /// <summary>
    ///     A private copy of a stored trial record: the scalars are shared, the five dictionaries this trial
    ///     mutates (parameters, distributions, user and system attributes, intermediate values) are cloned.
    /// </summary>
    /// <param name="stored">The record returned by the storage, possibly the storage's live object.</param>
    /// <returns>A record whose dictionaries no other object references.</returns>
    private static FrozenTrial DetachedCopy(FrozenTrial stored)
    {
        var copy = stored.ShallowCopy();
        copy.ParamsDict = stored.ParamsDict.ShallowClone();
        copy.DistributionsDict = stored.DistributionsDict.ShallowClone();
        copy.UserAttrsDict = stored.UserAttrsDict.ShallowClone();
        copy.SystemAttrsDict = stored.SystemAttrsDict.ShallowClone();
        copy.IntermediateValues = stored.IntermediateValues.ShallowClone();
        return copy;
    }

    /// <summary>
    ///     Suggest a value for the floating point parameter.
    /// </summary>
    /// <param name="name">A parameter name.</param>
    /// <param name="low">
    ///     Lower endpoint of the range of suggested values. <c>low</c> is included in the range.
    ///     <c>low</c> must be less than or equal to <c>high</c>. If <c>log</c> is <c>true</c>,
    ///     <c>low</c> must be larger than 0.
    /// </param>
    /// <param name="high">
    ///     Upper endpoint of the range of suggested values. <c>high</c> is included in the range.
    /// </param>
    /// <param name="step">A step of discretization. Cannot be combined with <c>log</c>.</param>
    /// <param name="log">A flag to sample the value from the log domain or not.</param>
    /// <returns>A suggested float value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    /// <exception cref="OptunaSharp.UpdateFinishedTrialError">The storage record is already finished.</exception>
    /// <exception cref="NumSharp.KeyError">The trial was removed from storage.</exception>
    /// <exception cref="System.Exception">A configured storage or sampler implementation fails; its exception propagates.</exception>
    /// <exception cref="NumSharp.TypeError">The stored or sampled value cannot be converted to a float.</exception>
    public override double SuggestFloat(string name, double low, double high, double? step = null, bool log = false)
    {
        var distribution = new FloatDistribution(low, high, log: log, step: step);
        var suggestedValue = Suggest(name, distribution);
        CheckDistribution(name, distribution);
        return PyFloat.TryConvert(suggestedValue, out double v)
            ? v
            : throw new TypeError("float-castable value is expected.");
    }

    /// <summary>
    ///     Suggest a value for the integer parameter. The value is sampled from the integers in
    ///     <c>[low, high]</c>.
    /// </summary>
    /// <param name="name">A parameter name.</param>
    /// <param name="low">Lower endpoint of the range of suggested values (included).</param>
    /// <param name="high">Upper endpoint of the range of suggested values (included).</param>
    /// <param name="step">Positive integer spacing; logarithmic sampling requires a step of one.</param>
    /// <param name="log">A flag to sample the value from the log domain or not.</param>
    /// <returns>The suggested or previously recorded integer value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    /// <exception cref="OptunaSharp.UpdateFinishedTrialError">The storage record is already finished.</exception>
    /// <exception cref="NumSharp.KeyError">The trial was removed from storage.</exception>
    /// <exception cref="System.Exception">A configured storage or sampler implementation fails; its exception propagates.</exception>
    /// <exception cref="System.FormatException">A stored or sampled string is not an integer.</exception>
    /// <exception cref="System.OverflowException">An integer conversion exceeds Int64.</exception>
    /// <exception cref="System.InvalidCastException">The value does not support integer conversion.</exception>
    public override long SuggestInt(string name, long low, long high, long step = 1, bool log = false)
    {
        var distribution = new IntDistribution(low: low, high: high, log: log, step: step);
        var suggested = Suggest(name, distribution);
        long suggestedValue = suggested switch
        {
            long l => l,
            double d => (long)d,
            bool b => b ? 1L : 0L,
            _ => Convert.ToInt64(suggested, System.Globalization.CultureInfo.InvariantCulture),
        };
        CheckDistribution(name, distribution);
        return suggestedValue;
    }

    /// <summary>
    ///     Suggest a value for the categorical parameter. The value is sampled from
    ///     <paramref name="choices" />.
    /// </summary>
    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="choices">Nonempty sequence of categorical candidates; order is retained and determines stored indices.</param>
    /// <returns>The suggested or previously recorded categorical value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    /// <exception cref="OptunaSharp.UpdateFinishedTrialError">The storage record is already finished.</exception>
    /// <exception cref="NumSharp.KeyError">The trial was removed from storage.</exception>
    /// <exception cref="System.Exception">A configured storage or sampler implementation fails; its exception propagates.</exception>
    /// <exception cref="NumSharp.TypeError">A stored or sampled value is not a supported categorical scalar.</exception>
    /// <exception cref="System.ArgumentNullException">The choices sequence is null.</exception>
    public override CategoricalChoice SuggestCategorical(string name, IEnumerable<CategoricalChoice> choices)
    {
        // There is no need to call CheckDistribution because CategoricalDistribution does not
        // support dynamic value space.
        return CategoricalChoice.From(Suggest(name, new CategoricalDistribution(choices)));
    }

    /// <inheritdoc />
    /// <remarks>
    ///     Reuses <paramref name="distribution" /> as-is — no per-call rebuild, and because the same instance
    ///     reaches the study's parameter registry on every trial, the per-suggest compatibility check
    ///     short-circuits by reference. The instance is stored into the trial record; it must stay immutable.
    /// </remarks>
    public override CategoricalChoice SuggestCategorical(string name, CategoricalDistribution distribution)
    {
        if (distribution is null)
            throw new ArgumentNullException(nameof(distribution));
        return CategoricalChoice.From(Suggest(name, distribution));
    }

    /// <summary>
    ///     Report an objective function value for a given step.
    ///
    ///     The reported values are used by the pruners to determine whether this trial should be
    ///     pruned. If this method is called multiple times at the same <paramref name="step" />
    ///     in a trial, the reported value only the first time is stored and the reported values
    ///     from the second time are ignored.
    /// </summary>
    /// <param name="value">A value returned from the objective function.</param>
    /// <param name="step">
    ///     Step of the trial (e.g., Epoch of neural network training). Note that pruners assume
    ///     that <c>step</c> starts at zero. <c>step</c> must be a non-negative integer. It is 64-bit, as
    ///     Python's unbounded <c>int</c> is in journal, SQLite and in-memory storages; the PostgreSQL and
    ///     MySQL schemas keep Optuna's 32-bit column, which refuses a larger step.
    /// </param>
    /// <exception cref="OptunaSharp.UpdateFinishedTrialError">The storage record is already finished.</exception>
    /// <exception cref="NumSharp.KeyError">The trial was removed from storage.</exception>
    /// <exception cref="System.Exception">A configured storage or sampler implementation fails; its exception propagates.</exception>
    /// <exception cref="System.NotImplementedException">Reporting intermediate values is unsupported for multi-objective studies.</exception>
    /// <exception cref="NumSharp.ValueError">The reporting step is negative.</exception>
    public override void Report(double value, long step)
    {
        if (Study.Directions.Count > 1)
            throw new NotImplementedException(
                "Trial.report is not supported for multi-objective optimization.");

        if (step < 0)
            throw new ValueError($"`step={step}` must be non-negative.");

        // The check and the write form one step under this trial's monitor, so the first report of a step wins
        // even when several threads report it at once (Optuna documents first-wins but races here).
        using (MonitorGate.Scope(_syncRoot))
        {
            if (_cachedFrozenTrial.IntermediateValues.ContainsKey(step))
            {
                // Do nothing if already reported.
                OptunaWarnings.Warn(
                    $"The reported value is ignored because this `step={step}` is already reported.");
                return;
            }

            // Publish to storage before changing the cache so a rejected write cannot appear successful
            // to the objective or to a later pruning decision on this Trial instance.
            Storage.SetTrialIntermediateValue(TrialId, step, value);
            _cachedFrozenTrial.IntermediateValues[step] = value;
            MirrorIntoThreadCache(t => t.IntermediateValues[step] = value);
        }
    }

    /// <summary>
    ///     Suggest whether the trial should be pruned or not.
    ///
    ///     The suggestion is made by a pruning algorithm associated with the trial and is based
    ///     on previously reported values. The algorithm can be specified when constructing a
    ///     <see cref="Study" />.
    /// </summary>
    /// <returns>True if the configured pruner recommends stopping the trial.</returns>
    /// <exception cref="System.NotImplementedException">Pruning through Trial is unsupported for multi-objective studies.</exception>
    /// <exception cref="System.Exception">The configured pruner or storage provider fails.</exception>
    public override bool ShouldPrune()
    {
        if (Study.Directions.Count > 1)
            throw new NotImplementedException(
                "Trial.should_prune is not supported for multi-objective optimization.");

        // The latest-trial view shares this trial's dictionaries, so a concurrent Report on this trial waits until
        // the pruner is done reading them. The pruner's monitor keeps its own state whole across trials.
        using (MonitorGate.Scope(_syncRoot))
        {
            var trial = GetLatestTrial();
            var pruner = Study.Pruner;
            return pruner.Locked(() => pruner.Prune(Study, trial));
        }
    }

    /// <summary>
    ///     Set user attributes to the trial, accessible via <see cref="UserAttrs" />.
    /// </summary>
    /// <param name="key">A key string of the attribute.</param>
    /// <param name="value">A value of the attribute. The value should be JSON serializable.</param>
    /// <exception cref="OptunaSharp.UpdateFinishedTrialError">The storage record is already finished.</exception>
    /// <exception cref="NumSharp.KeyError">The trial was removed from storage.</exception>
    /// <exception cref="System.Exception">A configured storage or sampler implementation fails; its exception propagates.</exception>
    public override void SetUserAttr(string key, object? value)
    {
        using (MonitorGate.Scope(_syncRoot))
        {
            Storage.SetTrialUserAttr(TrialId, key, value);
            _cachedFrozenTrial.UserAttrsDict[key] = value;
            MirrorIntoThreadCache(t => t.UserAttrsDict[key] = value);
        }
    }

    // Trial.set_system_attr (deprecated 3.1.0) was removed in Optuna 5.0.0 (#6834). Optuna-internal
    // writers call Storage.SetTrialSystemAttr(TrialId, key, value) directly; users keep SetUserAttr.

    /// <summary>Python <c>Trial._suggest</c>.</summary>
    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="distribution">Distribution defining the parameter representation and allowed values.</param>
    /// <returns>The external parameter value; an out-of-range fixed numeric value may be returned with a warning.</returns>
    /// <exception cref="NumSharp.ValueError">A required parameter is absent, its representation is invalid, or distributions are incompatible.</exception>
    /// <exception cref="System.Exception">A sampler hook or storage operation fails; its original exception propagates.</exception>
    internal object? Suggest(string name, BaseDistribution distribution)
    {
        var storage = Storage;
        int trialId = TrialId;

        // One suggestion is one step: two threads suggesting the same name on this trial must not both sample it.
        using (MonitorGate.Scope(_syncRoot))
        {
            var trial = GetLatestTrial();

            object? paramValue;
            if (trial.DistributionsDict.ContainsKey(name))
            {
                // No need to sample if already suggested.
                Distribution.CheckDistributionCompatibility(trial.DistributionsDict[name], distribution);
                paramValue = trial.ParamsDict[name];
            }
            else
            {
                // This precedence preserves queued values and singleton distributions before invoking
                // stochastic samplers, so replay and degenerate ranges do not consume random draws.
                if (IsFixedParam(name, distribution))
                {
                    paramValue = _fixedParams[name];
                }
                else if (distribution.Single())
                {
                    paramValue = Distribution.GetSingleValue(distribution);
                }
                else if (IsRelativeParam(name, distribution))
                {
                    paramValue = RelativeParams[name];
                }
                else
                {
                    var study = PrunerHelpers.FilterStudy(Study, trial);
                    // Concurrent trials share this sampler and its random generator (see BaseSampler.SyncRoot),
                    // unless each worker has its own (SamplerConcurrency.PerWorker).
                    var sampler = CurrentSampler;
                    paramValue = sampler.Locked(() => sampler.SampleIndependent(study, trial, name, distribution));
                }

                // `paramValue` is validated here (invalid value like NaN raises ValueError).
                double paramValueInInternalRepr = distribution.ToInternalRepr(paramValue);
                storage.SetTrialParam(trialId, name, paramValueInInternalRepr, distribution);

                object? externalValue = paramValue is CategoricalChoice choice ? choice.Value : paramValue;
                _cachedFrozenTrial.DistributionsDict[name] = distribution;
                _cachedFrozenTrial.ParamsDict[name] = externalValue;
                MirrorIntoThreadCache(t =>
                {
                    t.DistributionsDict[name] = distribution;
                    t.ParamsDict[name] = externalValue;
                });
            }
            return paramValue is CategoricalChoice c ? c.Value : paramValue;
        }
    }

    /// <summary>Python <c>Trial._is_fixed_param</c>.</summary>
    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="distribution">Distribution defining the parameter representation and allowed values.</param>
    /// <returns>True when the queued fixed-parameter dictionary contains the name, including out-of-range values after warning.</returns>
    /// <exception cref="NumSharp.ValueError">The fixed value cannot be represented by the requested distribution.</exception>
    internal bool IsFixedParam(string name, BaseDistribution distribution)
    {
        if (!_fixedParams.ContainsKey(name))
            return false;

        var paramValue = _fixedParams[name];
        double paramValueInInternalRepr = distribution.ToInternalRepr(paramValue);

        bool contained = distribution.Contains(paramValueInInternalRepr);
        if (!contained)
        {
            OptunaWarnings.Warn(
                $"Fixed parameter {name} with value {PythonRepr.Str(paramValue)} is out of range " +
                $"for distribution {distribution}.");
        }
        return true;
    }

    /// <summary>Python <c>Trial._is_relative_param</c>.</summary>
    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="distribution">Distribution defining the parameter representation and allowed values.</param>
    /// <returns>True when relative sampling supplied an in-range value for this parameter.</returns>
    /// <exception cref="NumSharp.ValueError">The sampler supplied a name outside its relative search space, distributions disagree, or the value cannot be represented.</exception>
    /// <exception cref="System.Exception">Lazy relative sampling or its storage access fails.</exception>
    internal bool IsRelativeParam(string name, BaseDistribution distribution)
    {
        if (!RelativeParams.ContainsKey(name))
            return false;

        System.Diagnostics.Debug.Assert(_relativeSearchSpace is not null);

        if (!_relativeSearchSpace!.ContainsKey(name))
            throw new ValueError(
                $"The parameter {name} was sampled by `sample_relative` method " +
                "but it is not contained in the relative search space.");

        var relativeDistribution = _relativeSearchSpace[name];
        Distribution.CheckDistributionCompatibility(relativeDistribution, distribution);

        var paramValue = RelativeParams[name];
        double paramValueInInternalRepr = distribution.ToInternalRepr(paramValue);
        return distribution.Contains(paramValueInInternalRepr);
    }

    /// <summary>Python <c>Trial._check_distribution</c>.</summary>
    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="distribution">Distribution defining the parameter representation and allowed values.</param>
    private void CheckDistribution(string name, BaseDistribution distribution)
    {
        BaseDistribution oldDistribution;
        using (MonitorGate.Scope(_syncRoot))
            oldDistribution = _cachedFrozenTrial.DistributionsDict.GetValueOrDefault(name, distribution)!;
        if (!oldDistribution.Equals(distribution))
        {
            OptunaWarnings.Warn(
                $"Inconsistent parameter values for distribution with name \"{name}\"! " +
                "This might be a configuration mistake. " +
                "Optuna allows to call the same distribution with the same " +
                "name more than once in a trial. " +
                "When the parameter values are inconsistent optuna only " +
                "uses the values of the first call and ignores all following. " +
                $"Using these values: {PythonRepr.DictRepr(oldDistribution.AsDict())}",
                WarningCategory.RuntimeWarning);
        }
    }

    /// <summary>Python <c>Trial._get_latest_trial</c>.</summary>
    /// <returns>A shallow view of the local trial with system attributes loaded from storage on first access.</returns>
    /// <remarks>
    ///     The view shares this trial's dictionaries, as Python's <c>copy.copy</c> does. Read it while holding this
    ///     trial's monitor (every caller here does), or a concurrent report or suggestion may change it under you.
    /// </remarks>
    internal FrozenTrial GetLatestTrial()
    {
        using (MonitorGate.Scope(_syncRoot))
        {
            var latestTrial = _cachedFrozenTrial.ShallowCopy();
            latestTrial.SetLazySystemAttrs(() => Storage.GetTrialSystemAttrs(TrialId) is OrderedDict<string, object?> d
                ? d
                : new OrderedDict<string, object?>(Storage.GetTrialSystemAttrs(TrialId)));
            return latestTrial;
        }
    }

    /// <summary>Return parameters to be optimized.</summary>
    public override IReadOnlyDictionary<string, object?> Params
    {
        get
        {
            using (MonitorGate.Scope(_syncRoot))
                return _cachedFrozenTrial.DeepCopy().ParamsDict;
        }
    }

    /// <summary>Return distributions of parameters to be optimized.</summary>
    public override IReadOnlyDictionary<string, BaseDistribution> Distributions
    {
        get
        {
            using (MonitorGate.Scope(_syncRoot))
                return _cachedFrozenTrial.DistributionsDict.ShallowClone();
        }
    }

    /// <summary>Return user attributes.</summary>
    public override IReadOnlyDictionary<string, object?> UserAttrs
    {
        get
        {
            using (MonitorGate.Scope(_syncRoot))
                return _cachedFrozenTrial.DeepCopy().UserAttrsDict;
        }
    }

    // Trial.system_attrs (deprecated 3.1.0) was removed in Optuna 5.0.0 (#6834); FrozenTrial and
    // FixedTrial keep theirs. A live trial's system attributes are optuna-internal storage state:
    // read them with Storage.GetTrialSystemAttrs(TrialId), or through GetLatestTrial().

    /// <summary>Return start datetime.</summary>
    public override DateTime? DatetimeStart => _cachedFrozenTrial.DatetimeStart;

    /// <summary>Return trial's number which is consecutive and unique in a study.</summary>
    public override int Number => _cachedFrozenTrial.Number;

    /// <summary>Gets the mutable local trial cache used by the active trial facade.</summary>
    internal FrozenTrial CachedFrozenTrial => _cachedFrozenTrial;

    /// <summary>
    ///     Returns constraint values. The trial is considered feasible when all constraint values
    ///     are zero or less.
    /// </summary>
    public override OrderedDict<string, double> Constraints
    {
        get
        {
            var systemAttrs = Storage.GetTrialSystemAttrs(TrialId);
            return ConstrainedOptimization.GetConstraintsFromSystemAttrs(systemAttrs);
        }
    }

    /// <summary>
    ///     Set a constraint value for the trial. The trial is considered feasible when all
    ///     constraint values are zero or less.
    /// </summary>
    /// <param name="key">Constraint name; later writes of the same name are ignored with a warning.</param>
    /// <param name="value">Constraint value; zero or negative is feasible and NaN is forbidden.</param>
    /// <exception cref="NumSharp.ValueError">The constraint value is NaN.</exception>
    /// <exception cref="OptunaSharp.UpdateFinishedTrialError">The storage record is already finished.</exception>
    /// <exception cref="NumSharp.KeyError">The trial was removed from storage.</exception>
    /// <exception cref="System.Exception">A configured storage or sampler implementation fails; its exception propagates.</exception>
    public override void SetConstraint(string key, double value)
    {
        if (double.IsNaN(value))
            throw new ValueError($"Attempted to set a constraint for {PythonRepr.StringRepr(key)}, but NaN is not allowed.");

        string constraintKey = $"{ConstrainedOptimization.ConstraintsKey}:{key}";

        // First-set-wins, like Report: the existence check and the write must not interleave with another thread's.
        using (MonitorGate.Scope(_syncRoot))
        {
            var systemAttrs = Storage.GetTrialSystemAttrs(TrialId);
            if (systemAttrs.ContainsKey(constraintKey))
            {
                // Do nothing if already set.
                OptunaWarnings.Warn(
                    $"The constraint value is ignored because this constraint `key={PythonRepr.Repr(key)}` is already set.");
                return;
            }

            Storage.SetTrialSystemAttr(TrialId, constraintKey, value);
            _cachedFrozenTrial.SystemAttrsDict[constraintKey] = value;
            MirrorIntoThreadCache(t => t.SystemAttrsDict[constraintKey] = value);
        }
    }

    /// <summary>
    ///     Makes the current thread's study trial cache show this trial's latest write whenever Python's cache would,
    ///     because it holds the very object Python's <c>Trial</c> mutates. Called after every write that Python also
    ///     applies to <c>_cached_frozen_trial</c>: suggestions, reports, user attributes and constraints.
    /// </summary>
    /// <remarks>
    ///     <para>Python: <c>Study._thread_local.cached_all_trials</c> (reset by every <c>ask</c>) holds the storage's
    ///     objects as of its fetch. <c>Trial</c> mutates the object the in-memory storage held when the trial was
    ///     created, while the storage itself replaces it copy-on-write. So the cache sees this trial's writes exactly
    ///     when it was fetched before the trial's first storage write. <c>QMCSampler.infer_relative_search_space</c>
    ///     reads RUNNING trials' distributions from that cache. In a batch of asks, trial 1's first suggestion
    ///     therefore sees trial 0's parameter and takes Sobol point 0. Without this, the port sampled trials
    ///     1…k-1 of the first batch at random and shifted the whole QMC sequence (QA-31).</para>
    ///     <para>Thread-safety: the shared storage object is never mutated (that was the data race the detached
    ///     <c>_cachedFrozenTrial</c> fixed). The cache list belongs to the calling thread alone (<c>GetAllTrials</c>
    ///     returns a fresh list), so the first matching write puts a private copy of this trial in place of the
    ///     storage object, and later writes mutate that copy, as Python mutates its shared one. Caches of other
    ///     threads are not touched: in Python their view depends on fetch timing, which no test can pin. Only the
    ///     slot of this trial's number is examined, since <c>GetAllTrials</c> lists trials in number order; a storage
    ///     that hands out copies never matches, which keeps its behavior as it was.</para>
    /// </remarks>
    /// <param name="apply">The write just applied to <see cref="_cachedFrozenTrial" />, re-applied to the private copy.</param>
    private void MirrorIntoThreadCache(Action<FrozenTrial> apply)
    {
        var cache = Study.ThreadLocal.CachedAllTrials;
        int number = _cachedFrozenTrial.Number;
        if (cache is null || number < 0 || number >= cache.Count || !ReferenceEquals(cache[number], _cacheAlias))
            return;

        if (_cacheAliasIsOwnCopy)
        {
            apply(_cacheAlias);
            return;
        }

        // First write the cache can see: replace the shared storage object with a private copy that already
        // carries every write so far (this one included), then keep mutating that copy.
        var copy = DetachedCopy(_cachedFrozenTrial);
        cache[number] = copy;
        _cacheAlias = copy;
        _cacheAliasIsOwnCopy = true;
    }
}
