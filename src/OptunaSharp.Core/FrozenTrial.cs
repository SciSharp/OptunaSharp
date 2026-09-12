using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp;

/// <summary>
///     Status and results of a <see cref="Trial" />. Port of <c>optuna/trial/_frozen.py</c>.
///
///     An object of this class has the same methods as <see cref="Trial" />, but is not
///     associated with, nor has any references to a <see cref="Study" />.
///
///     It will suggest the parameter values stored in <see cref="Params" /> and will not sample
///     values from any distributions. It can be passed to objective functions and is useful for
///     deploying optimization results.
/// </summary>
/// <remarks>Instances are mutable, despite the name.</remarks>
public class FrozenTrial : BaseTrial
{
    private int _number;
    private List<double>? _values;
    private DateTime? _datetimeStart;
    private OrderedDict<string, object?> _params;
    private OrderedDict<string, object?> _userAttrs;
    private OrderedDict<string, object?> _systemAttrs;
    private OrderedDict<string, BaseDistribution> _distributions;
    internal int TrialId;

    // Lazy system-attrs loading used by Trial._get_latest_trial (Python: _LazyTrialSystemAttrs).
    private Func<OrderedDict<string, object?>>? _systemAttrsLoader;

    public FrozenTrial(
        int number,
        TrialState state,
        double? value,
        DateTime? datetimeStart,
        DateTime? datetimeComplete,
        OrderedDict<string, object?> @params,
        OrderedDict<string, BaseDistribution> distributions,
        OrderedDict<string, object?> userAttrs,
        OrderedDict<string, object?> systemAttrs,
        OrderedDict<int, double> intermediateValues,
        int trialId,
        IEnumerable<double>? values = null)
    {
        _number = number;
        State = state;
        _values = null;
        if (value is not null && values is not null)
            throw new ValueError("Specify only one of `value` and `values`.");
        if (value is not null)
            _values = new List<double> { value.Value };
        else if (values is not null)
            _values = values.ToList();
        _datetimeStart = datetimeStart;
        DatetimeComplete = datetimeComplete;
        _params = @params;
        _userAttrs = userAttrs;
        _systemAttrs = systemAttrs;
        IntermediateValues = intermediateValues;
        _distributions = distributions;
        TrialId = trialId;
    }

    /// <summary><see cref="TrialState" /> of the trial.</summary>
    public TrialState State { get; set; }

    /// <summary>Datetime where the trial finished.</summary>
    public DateTime? DatetimeComplete { get; set; }

    /// <summary>Intermediate objective values set with <see cref="Trial.Report" />.</summary>
    public OrderedDict<int, double> IntermediateValues { get; set; }

    // ------------------------------------------------------------------ suggest API

    public override double SuggestFloat(string name, double low, double high, double? step = null, bool log = false)
        => PyFloat.TryConvert(Suggest(name, new FloatDistribution(low, high, log: log, step: step)), out double v)
            ? v
            : throw new TypeError("float-castable value is expected.");

    public override long SuggestInt(string name, long low, long high, long step = 1, bool log = false)
    {
        var value = Suggest(name, new IntDistribution(low, high, log: log, step: step));
        return value switch
        {
            long l => l,
            double d => (long)d,
            bool b => b ? 1L : 0L,
            _ => Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture),
        };
    }

    public override CategoricalChoice SuggestCategorical(string name, IEnumerable<CategoricalChoice> choices)
        => CategoricalChoice.From(Suggest(name, new CategoricalDistribution(choices)));

    /// <summary>
    ///     Interface of report function.
    ///
    ///     Since <see cref="FrozenTrial" /> is not pruned, this report function does nothing.
    /// </summary>
    public override void Report(double value, int step)
    {
    }

    /// <summary>
    ///     Suggest whether the trial should be pruned or not.
    ///
    ///     The suggestion is always <c>false</c> regardless of a pruning algorithm.
    /// </summary>
    public override bool ShouldPrune() => false;

    public override void SetUserAttr(string key, object? value) => _userAttrs[key] = value;

    /// <summary>
    ///     Read this trial's observed-components vector (the residual terms written with
    ///     <see cref="BaseTrial.SetComponentLosses" />) from its user attributes. Returns <c>false</c>
    ///     when the trial carries no valid component vector — see <see cref="ComponentLosses" /> for the
    ///     contract (list-or-array tolerant, non-finite / empty rejected). On success
    ///     <paramref name="components" /> is a fresh, caller-owned copy.
    /// </summary>
    public bool TryGetComponentLosses(out double[] components)
        => ComponentLosses.TryRead(UserAttrs, out components);

    [Obsolete(DeprecationMessages.SetSystemAttr)]
    public override void SetSystemAttr(string key, object? value)
    {
        OptunaWarnings.Warn(DeprecationMessages.SetSystemAttr, WarningCategory.FutureWarning);
        SystemAttrsDict[key] = value;
    }

    // ------------------------------------------------------------------ validation

    /// <summary>Python <c>FrozenTrial._validate</c>.</summary>
    internal void Validate()
    {
        if (State != TrialState.Waiting && DatetimeStart is null)
            throw new ValueError("`datetime_start` is supposed to be set when the trial state is not waiting.");

        if (State.IsFinished())
        {
            if (DatetimeComplete is null)
                throw new ValueError("`datetime_complete` is supposed to be set for a finished trial.");
        }
        else
        {
            if (DatetimeComplete is not null)
                throw new ValueError("`datetime_complete` is supposed to be None for an unfinished trial.");
        }

        if (State == TrialState.Fail && _values is not null)
            throw new ValueError($"values should be None for a failed trial, but got {PythonRepr.ListRepr(_values)}.");
        if (State == TrialState.Complete)
        {
            if (_values is null)
                throw new ValueError("values should be set for a complete trial.");
            if (_values.Any(double.IsNaN))
                throw new ValueError("values should not contain NaN.");
        }

        var paramKeys = new HashSet<string>(_params.Keys);
        var distKeys = new HashSet<string>(_distributions.Keys);
        if (!paramKeys.SetEquals(distKeys))
            throw new ValueError(
                $"Inconsistent parameters {PythonRepr.SetRepr(_params.Keys)} and " +
                $"distributions {PythonRepr.SetRepr(_distributions.Keys)}.");

        foreach (var (paramName, paramValue) in _params)
        {
            var distribution = _distributions[paramName];

            double paramValueInInternalRepr = distribution.ToInternalRepr(paramValue);
            if (!distribution.Contains(paramValueInInternalRepr))
                throw new ValueError(
                    $"The value {PythonRepr.Str(paramValue)} of parameter '{paramName}' isn't contained in " +
                    $"the distribution {distribution}.");
        }
    }

    /// <summary>Python <c>FrozenTrial._suggest</c>.</summary>
    internal object? Suggest(string name, BaseDistribution distribution)
    {
        if (!_params.ContainsKey(name))
            throw new ValueError(
                $"The value of the parameter '{name}' is not found. " +
                $"Please set it at the construction of the FrozenTrial object.");

        var value = _params[name];
        double paramValueInInternalRepr = distribution.ToInternalRepr(value);
        if (!distribution.Contains(paramValueInInternalRepr))
            OptunaWarnings.Warn(
                $"The value {PythonRepr.Str(value)} of the parameter '{name}' is out of " +
                $"the range of the distribution {distribution}.");

        if (_distributions.TryGetValue(name, out var existing))
            Distribution.CheckDistributionCompatibility(existing, distribution);

        _distributions[name] = distribution;

        return value;
    }

    // ------------------------------------------------------------------ properties

    /// <summary>
    ///     Unique and consecutive number of <see cref="Trial" /> for each <see cref="Study" />.
    ///     Note that this field uses zero-based numbering.
    /// </summary>
    public override int Number => _number;

    /// <summary>Internal setter mirroring Python's <c>number</c> property setter (used by storages).</summary>
    public void SetNumber(int value) => _number = value;

    /// <summary>Objective value of the trial. Throws for multi-objective trials.</summary>
    public double? Value
    {
        get
        {
            if (_values is not null)
            {
                if (_values.Count > 1)
                    throw new RuntimeError("This attribute is not available during multi-objective optimization.");
                return _values[0];
            }
            return null;
        }
        set
        {
            if (_values is not null && _values.Count > 1)
                throw new RuntimeError("This attribute is not available during multi-objective optimization.");
            _values = value is null ? null : new List<double> { value.Value };
        }
    }

    /// <summary>Sequence of objective values of the trial (length &gt; 1 for multi-objective).</summary>
    public List<double>? Values
    {
        get => _values;
        set => _values = value;
    }

    internal void SetValues(IEnumerable<double>? values) => _values = values?.ToList();

    /// <summary>Datetime where the trial started.</summary>
    public override DateTime? DatetimeStart => _datetimeStart;

    public void SetDatetimeStart(DateTime? value) => _datetimeStart = value;

    /// <summary>Dictionary that contains suggested parameters (external representations).</summary>
    public override IReadOnlyDictionary<string, object?> Params => _params;

    /// <summary>The mutable parameter dictionary (Python's <c>params</c> setter).</summary>
    public OrderedDict<string, object?> ParamsDict
    {
        get => _params;
        set => _params = value;
    }

    /// <summary>Dictionary that contains the distributions of <see cref="Params" />.</summary>
    public override IReadOnlyDictionary<string, BaseDistribution> Distributions => _distributions;

    public OrderedDict<string, BaseDistribution> DistributionsDict
    {
        get => _distributions;
        set => _distributions = value;
    }

    /// <summary>Dictionary that contains the user attributes.</summary>
    public override IReadOnlyDictionary<string, object?> UserAttrs => _userAttrs;

    public OrderedDict<string, object?> UserAttrsDict
    {
        get => _userAttrs;
        set => _userAttrs = value;
    }

    /// <summary>Dictionary that contains the optuna-internal attributes.</summary>
    public override IReadOnlyDictionary<string, object?> SystemAttrs => SystemAttrsDict;

    public OrderedDict<string, object?> SystemAttrsDict
    {
        get
        {
            if (_systemAttrsLoader is not null)
            {
                _systemAttrs = _systemAttrsLoader();
                _systemAttrsLoader = null;
            }
            return _systemAttrs;
        }
        set
        {
            _systemAttrsLoader = null;
            _systemAttrs = value;
        }
    }

    /// <summary>Python <c>Trial._get_latest_trial</c> support: defer system-attr loading to the storage.</summary>
    internal void SetLazySystemAttrs(Func<OrderedDict<string, object?>> loader) => _systemAttrsLoader = loader;

    /// <summary>Return the maximum step of <see cref="IntermediateValues" /> in the trial.</summary>
    public int? LastStep
    {
        get
        {
            if (IntermediateValues.Count == 0)
                return null;
            return IntermediateValues.Keys.Max();
        }
    }

    /// <summary>Return the elapsed time taken to complete the trial.</summary>
    public TimeSpan? Duration
    {
        get
        {
            if (DatetimeStart is not null && DatetimeComplete is not null)
                return DatetimeComplete.Value - DatetimeStart.Value;
            return null;
        }
    }

    /// <summary>
    ///     Returns constraint values. The trial is considered feasible when all constraint values
    ///     are zero or less.
    /// </summary>
    public override OrderedDict<string, double> Constraints
        => ConstrainedOptimization.GetConstraintsFromSystemAttrs(SystemAttrsDict);

    /// <summary>
    ///     Set a constraint value for the trial. The trial is considered feasible when all
    ///     constraint values are zero or less.
    /// </summary>
    public override void SetConstraint(string key, double value)
    {
        if (double.IsNaN(value))
            throw new ValueError($"Attempted to set a constraint for {PythonRepr.StringRepr(key)}, but NaN is not allowed.");

        string constraintKey = $"{ConstrainedOptimization.ConstraintsKey}:{key}";

        if (SystemAttrsDict.ContainsKey(constraintKey))
        {
            // Do nothing if already set.
            OptunaWarnings.Warn(
                $"The constraint value is ignored because this constraint `key={PythonRepr.Repr(key)}` is already set.");
            return;
        }

        SystemAttrsDict[constraintKey] = value;
    }

    // ------------------------------------------------------------------ equality / copying

    /// <summary>Python <c>__eq__</c>: full state comparison.</summary>
    public override bool Equals(object? obj)
    {
        if (obj is not FrozenTrial other)
            return false;
        return _number == other._number
            && State == other.State
            && ValuesEqual(_values, other._values)
            && Nullable.Equals(_datetimeStart, other._datetimeStart)
            && Nullable.Equals(DatetimeComplete, other.DatetimeComplete)
            && DictValueEquals(_params, other._params)
            && DistributionsEqual(_distributions, other._distributions)
            && DictValueEquals(_userAttrs, other._userAttrs)
            && DictValueEquals(SystemAttrsDict, other.SystemAttrsDict)
            && IntermediateValuesEqual(IntermediateValues, other.IntermediateValues)
            && TrialId == other.TrialId;
    }

    public override int GetHashCode() => HashCode.Combine(TrialId, _number, State);

    public static bool operator <(FrozenTrial left, FrozenTrial right) => left.Number < right.Number;

    public static bool operator >(FrozenTrial left, FrozenTrial right) => left.Number > right.Number;

    public static bool operator <=(FrozenTrial left, FrozenTrial right) => left.Number <= right.Number;

    public static bool operator >=(FrozenTrial left, FrozenTrial right) => left.Number >= right.Number;

    private static bool ValuesEqual(List<double>? a, List<double>? b)
    {
        if (a is null || b is null)
            return a is null && b is null;
        return a.SequenceEqual(b);
    }

    private static bool DictValueEquals(OrderedDict<string, object?> a, OrderedDict<string, object?> b)
    {
        if (a.Count != b.Count)
            return false;
        foreach (var (key, value) in a)
        {
            if (!b.TryGetValue(key, out var otherValue))
                return false;
            if (!PyJson.ValueEquals(value is CategoricalChoice c ? c.Value : value,
                                    otherValue is CategoricalChoice oc ? oc.Value : otherValue))
                return false;
        }
        return true;
    }

    private static bool DistributionsEqual(
        OrderedDict<string, BaseDistribution> a, OrderedDict<string, BaseDistribution> b)
    {
        if (a.Count != b.Count)
            return false;
        foreach (var (key, value) in a)
        {
            if (!b.TryGetValue(key, out var otherValue) || !value.Equals(otherValue))
                return false;
        }
        return true;
    }

    private static bool IntermediateValuesEqual(OrderedDict<int, double> a, OrderedDict<int, double> b)
    {
        if (a.Count != b.Count)
            return false;
        foreach (var (key, value) in a)
        {
            if (!b.TryGetValue(key, out var otherValue) || !value.Equals(otherValue))
                return false;
        }
        return true;
    }

    /// <summary>Python <c>copy.copy(trial)</c>: a shallow copy (dict references shared).</summary>
    internal FrozenTrial ShallowCopy()
    {
        var copy = new FrozenTrial(
            number: _number,
            state: State,
            value: null,
            datetimeStart: _datetimeStart,
            datetimeComplete: DatetimeComplete,
            @params: _params,
            distributions: _distributions,
            userAttrs: _userAttrs,
            systemAttrs: _systemAttrs,
            intermediateValues: IntermediateValues,
            trialId: TrialId,
            values: null);
        copy._values = _values;
        copy._systemAttrsLoader = _systemAttrsLoader;
        return copy;
    }

    /// <summary>Python <c>copy.deepcopy(trial)</c>.</summary>
    public FrozenTrial DeepCopy()
    {
        var copy = new FrozenTrial(
            number: _number,
            state: State,
            value: null,
            datetimeStart: _datetimeStart,
            datetimeComplete: DatetimeComplete,
            @params: DeepCopyAttrs(_params),
            distributions: _distributions.ShallowClone(),  // Distributions are treated as immutable.
            userAttrs: DeepCopyAttrs(_userAttrs),
            systemAttrs: DeepCopyAttrs(SystemAttrsDict),
            intermediateValues: IntermediateValues.ShallowClone(),
            trialId: TrialId,
            values: null);
        copy._values = _values is null ? null : new List<double>(_values);
        return copy;
    }

    private static OrderedDict<string, object?> DeepCopyAttrs(OrderedDict<string, object?> attrs)
    {
        var copy = new OrderedDict<string, object?>(attrs.Count);
        foreach (var (key, value) in attrs)
            copy[key] = PyJson.DeepCopy(value);
        return copy;
    }

    /// <summary>Python <c>__repr__</c> (used in logs and debugging).</summary>
    public override string ToString()
    {
        var parts = new List<string>
        {
            $"number={_number}",
            $"state={State.PythonStr()}",
            $"values={(_values is null ? "None" : PythonRepr.ListRepr(_values))}",
            $"datetime_start={PythonRepr.Repr(_datetimeStart?.ToString("yyyy-MM-dd HH:mm:ss.ffffff"))}",
            $"datetime_complete={PythonRepr.Repr(DatetimeComplete?.ToString("yyyy-MM-dd HH:mm:ss.ffffff"))}",
            $"params={PythonRepr.DictRepr(_params)}",
            $"user_attrs={PythonRepr.DictRepr(_userAttrs)}",
            $"system_attrs={PythonRepr.DictRepr(_systemAttrs)}",
            $"intermediate_values={PythonRepr.DictRepr(IntermediateValues)}",
            $"distributions={PythonRepr.DictRepr(_distributions)}",
            $"trial_id={TrialId}",
            "value=None",
        };
        return $"FrozenTrial({string.Join(", ", parts)})";
    }

    // ------------------------------------------------------------------ factory

    /// <summary>
    ///     Create a new <see cref="FrozenTrial" />. Port of <c>optuna/trial/_frozen.py: create_trial</c>.
    ///
    ///     When <paramref name="state" /> is <see cref="TrialState.Complete" />, <c>params</c>,
    ///     <c>distributions</c> and <c>value</c> (or <c>values</c>) are required.
    /// </summary>
    public static FrozenTrial CreateTrial(
        TrialState state = TrialState.Complete,
        double? value = null,
        IEnumerable<double>? values = null,
        OrderedDict<string, object?>? @params = null,
        OrderedDict<string, BaseDistribution>? distributions = null,
        OrderedDict<string, object?>? userAttrs = null,
        OrderedDict<string, object?>? systemAttrs = null,
        OrderedDict<int, double>? intermediateValues = null,
        OrderedDict<string, double>? constraints = null)
    {
        @params ??= new OrderedDict<string, object?>();
        distributions ??= new OrderedDict<string, BaseDistribution>();
        var convertedDistributions = new OrderedDict<string, BaseDistribution>(distributions.Count);
        foreach (var (key, dist) in distributions)
            convertedDistributions[key] = Distribution.ConvertOldDistributionToNewDistribution(dist);
        userAttrs ??= new OrderedDict<string, object?>();
        systemAttrs ??= new OrderedDict<string, object?>();
        intermediateValues ??= new OrderedDict<int, double>();

        DateTime? datetimeStart = state == TrialState.Waiting ? null : DateTime.Now;
        DateTime? datetimeComplete = state.IsFinished() ? datetimeStart : null;

        var trial = new FrozenTrial(
            number: -1,
            trialId: -1,
            state: state,
            value: value,
            values: values,
            datetimeStart: datetimeStart,
            datetimeComplete: datetimeComplete,
            @params: @params,
            distributions: convertedDistributions,
            userAttrs: userAttrs,
            systemAttrs: systemAttrs,
            intermediateValues: intermediateValues);

        if (constraints is not null)
        {
            foreach (var (key, constraintValue) in constraints)
                trial.SetConstraint(key, constraintValue);
        }

        trial.Validate();

        return trial;
    }
}
