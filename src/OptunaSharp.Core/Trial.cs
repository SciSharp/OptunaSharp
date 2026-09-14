using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Pruners;
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
///     <see cref="Study.Optimize(Func{Trial, double}, int?, double?, int, Type[], IEnumerable{Action{Study, FrozenTrial}}, bool, bool)" />
///     method; hence library users do not care about instantiation of this object.
/// </summary>
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html">Official Optuna active-trial contract.</seealso>
public class Trial : BaseTrial
{
    /// <summary>The parent <see cref="Study" />.</summary>
    public Study Study { get; }

    /// <summary>Storage identifier for this trial, separate from its per-study number.</summary>
    internal readonly int TrialId;
    /// <summary>Storage backend shared with the parent study.</summary>
    internal readonly BaseStorage Storage;

    private readonly FrozenTrial _cachedFrozenTrial;
    private OrderedDict<string, BaseDistribution>? _relativeSearchSpace;
    private OrderedDict<string, object?>? _relativeParams;
    private readonly OrderedDict<string, object?> _fixedParams;

    /// <param name="study">Study whose history and storage are used by the operation.</param>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <summary>Binds to an existing trial record and invokes the sampler's BeforeTrial hook.</summary>
    /// <exception cref="NumSharp.KeyError">The trial identifier does not exist in storage.</exception>
    /// <exception cref="System.Exception">The configured storage or sampler hook fails; its original exception propagates.</exception>
    public Trial(Study study, int trialId)
    {
        Study = study;
        TrialId = trialId;

        Storage = study.Storage;

        _cachedFrozenTrial = Storage.GetTrial(TrialId);
        var filteredStudy = PrunerHelpers.FilterStudy(Study, _cachedFrozenTrial);

        Study.Sampler.BeforeTrial(filteredStudy, _cachedFrozenTrial);

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
            if (_relativeParams is null)
            {
                var study = PrunerHelpers.FilterStudy(Study, _cachedFrozenTrial);
                _relativeSearchSpace = Study.Sampler.InferRelativeSearchSpace(study, _cachedFrozenTrial);
                _relativeParams = Study.Sampler.SampleRelative(study, _cachedFrozenTrial, _relativeSearchSpace);
            }
            return _relativeParams;
        }
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
    ///     that <c>step</c> starts at zero. <c>step</c> must be a non-negative integer.
    /// </param>
    /// <exception cref="OptunaSharp.UpdateFinishedTrialError">The storage record is already finished.</exception>
    /// <exception cref="NumSharp.KeyError">The trial was removed from storage.</exception>
    /// <exception cref="System.Exception">A configured storage or sampler implementation fails; its exception propagates.</exception>
    /// <exception cref="System.NotImplementedException">Reporting intermediate values is unsupported for multi-objective studies.</exception>
    /// <exception cref="NumSharp.ValueError">The reporting step is negative.</exception>
    public override void Report(double value, int step)
    {
        if (Study.Directions.Count > 1)
            throw new NotImplementedException(
                "Trial.report is not supported for multi-objective optimization.");

        if (step < 0)
            throw new ValueError($"`step={step}` must be non-negative.");

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

        var trial = GetLatestTrial();
        return Study.Pruner.Prune(Study, trial);
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
        Storage.SetTrialUserAttr(TrialId, key, value);
        _cachedFrozenTrial.UserAttrsDict[key] = value;
    }

    /// <summary>
    ///     Set system attributes to the trial. Note that Optuna internally uses this method to
    ///     save system messages such as failure reason of trials. Please use
    ///     <see cref="SetUserAttr" /> to set users' attributes.
    /// </summary>
    /// <param name="key">Attribute name; an existing attribute is overwritten.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    /// <exception cref="OptunaSharp.UpdateFinishedTrialError">The storage record is already finished.</exception>
    /// <exception cref="NumSharp.KeyError">The trial was removed from storage.</exception>
    /// <exception cref="System.Exception">A configured storage or sampler implementation fails; its exception propagates.</exception>
    [Obsolete(DeprecationMessages.SetSystemAttr)]
    public override void SetSystemAttr(string key, object? value)
    {
        OptunaWarnings.Warn(DeprecationMessages.SetSystemAttr, WarningCategory.FutureWarning);
        Storage.SetTrialSystemAttr(TrialId, key, value);
        _cachedFrozenTrial.SystemAttrsDict[key] = value;
    }

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
                paramValue = Study.Sampler.SampleIndependent(study, trial, name, distribution);
            }

            // `paramValue` is validated here (invalid value like NaN raises ValueError).
            double paramValueInInternalRepr = distribution.ToInternalRepr(paramValue);
            storage.SetTrialParam(trialId, name, paramValueInInternalRepr, distribution);

            _cachedFrozenTrial.DistributionsDict[name] = distribution;
            _cachedFrozenTrial.ParamsDict[name] = paramValue is CategoricalChoice choice ? choice.Value : paramValue;
        }
        return paramValue is CategoricalChoice c ? c.Value : paramValue;
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
        var oldDistribution = _cachedFrozenTrial.DistributionsDict.GetValueOrDefault(name, distribution)!;
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
    internal FrozenTrial GetLatestTrial()
    {
        // TODO(eukaryo): Remove this method after `system_attrs` property is removed.
        var latestTrial = _cachedFrozenTrial.ShallowCopy();
        latestTrial.SetLazySystemAttrs(() => Storage.GetTrialSystemAttrs(TrialId) is OrderedDict<string, object?> d
            ? d
            : new OrderedDict<string, object?>(Storage.GetTrialSystemAttrs(TrialId)));
        return latestTrial;
    }

    /// <summary>Return parameters to be optimized.</summary>
    public override IReadOnlyDictionary<string, object?> Params
        => _cachedFrozenTrial.DeepCopy().ParamsDict;

    /// <summary>Return distributions of parameters to be optimized.</summary>
    public override IReadOnlyDictionary<string, BaseDistribution> Distributions
        => _cachedFrozenTrial.DistributionsDict.ShallowClone();

    /// <summary>Return user attributes.</summary>
    public override IReadOnlyDictionary<string, object?> UserAttrs
        => _cachedFrozenTrial.DeepCopy().UserAttrsDict;

    /// <summary>Return system attributes.</summary>
    // Optuna deprecates `Trial.system_attrs` specifically (FrozenTrial / FixedTrial keep theirs), so
    // this override is obsolete over a non-obsolete base member on purpose — hence the CS0809 mute.
#pragma warning disable CS0809
    [Obsolete(DeprecationMessages.SystemAttrs)]
    public override IReadOnlyDictionary<string, object?> SystemAttrs
    {
        get
        {
            OptunaWarnings.Warn(DeprecationMessages.SystemAttrs, WarningCategory.FutureWarning);
            return new OrderedDict<string, object?>(Storage.GetTrialSystemAttrs(TrialId));
        }
    }
#pragma warning restore CS0809

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
    }
}
