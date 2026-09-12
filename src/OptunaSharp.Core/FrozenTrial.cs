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
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.FrozenTrial.html">Official Optuna trial-record contract.</seealso>
public class FrozenTrial : BaseTrial
{
    private int _number;
    private List<double>? _values;
    private DateTime? _datetimeStart;
    private OrderedDict<string, object?> _params;
    private OrderedDict<string, object?> _userAttrs;
    private OrderedDict<string, object?> _systemAttrs;
    private OrderedDict<string, BaseDistribution> _distributions;
    /// <summary>Storage identifier for this trial, separate from its per-study number.</summary>
    internal int TrialId;

    // Lazy system-attrs loading used by Trial._get_latest_trial (Python: _LazyTrialSystemAttrs).
    private Func<OrderedDict<string, object?>>? _systemAttrsLoader;

    /// <param name="state">Trial state to record or inspect.</param>
    /// <param name="value">Single objective value; mutually exclusive with values.</param>
    /// <param name="datetimeStart">Local start time, or null when the trial or study has not started.</param>
    /// <param name="datetimeComplete">Local completion time, or null for an unfinished trial.</param>
    /// <param name="params">Parameter names mapped to their external values.</param>
    /// <param name="distributions">Distribution for each parameter, keyed by the same names as the parameter values.</param>
    /// <param name="userAttrs">User metadata; values should be JSON serializable for persistent storage.</param>
    /// <param name="systemAttrs">Internal metadata used by samplers and storage; preserve reserved keys.</param>
    /// <param name="intermediateValues">Objective observations keyed by reporting step.</param>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="values">Objective values in study direction order, or null when no objective values are recorded.</param>
    /// <summary>Constructs a mutable trial record, retaining the supplied dictionaries by reference.</summary>
    /// <param name="number">Per-study trial number; use -1 for a record not yet assigned by storage.</param>
    /// <exception cref="NumSharp.ValueError">Both value and values are supplied.</exception>
    /// <remarks>Construction does not run full state validation. CreateTrial validates timestamps, state, objective values, and parameter distributions before returning.</remarks>
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
        // The constructor is also used to rebuild partially populated records. Full lifecycle
        // validation is deferred to Validate/CreateTrial rather than imposed on storage hydration.
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

    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="low">Inclusive lower endpoint of the parameter range.</param>
    /// <param name="high">Inclusive upper endpoint; a stepped range may lower it to the last reachable grid point.</param>
    /// <param name="step">Positive float grid spacing, or null for a continuous range; cannot be combined with log.</param>
    /// <param name="log">Whether to sample in the logarithmic domain, which requires a strictly positive lower endpoint.</param>
    /// <summary>Replays a stored floating-point parameter value for the objective.</summary>
    /// <returns>The stored floating-point value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    /// <exception cref="NumSharp.TypeError">The stored or sampled value cannot be converted to a float.</exception>
    /// <remarks>Numeric values outside the requested range are returned with a warning when replaying stored parameters.</remarks>
    /// <exception cref="System.FormatException">A stepped bound or step is nonfinite and cannot be parsed as a decimal.</exception>
    /// <exception cref="System.DivideByZeroException">The positive step rounds to zero in decimal arithmetic while aligning the upper endpoint.</exception>
    public override double SuggestFloat(string name, double low, double high, double? step = null, bool log = false)
        => PyFloat.TryConvert(Suggest(name, new FloatDistribution(low, high, log: log, step: step)), out double v)
            ? v
            : throw new TypeError("float-castable value is expected.");

    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="low">Inclusive lower endpoint of the parameter range.</param>
    /// <param name="high">Inclusive upper endpoint; a stepped range may lower it to the last reachable grid point.</param>
    /// <param name="step">Positive integer grid spacing; must be one for logarithmic sampling.</param>
    /// <param name="log">Whether to sample in the logarithmic domain, which requires a strictly positive lower endpoint.</param>
    /// <summary>Replays a stored integer parameter value for the objective.</summary>
    /// <returns>The stored integer value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    /// <exception cref="System.FormatException">A stored or sampled string is not an integer.</exception>
    /// <exception cref="System.OverflowException">An integer conversion exceeds Int64.</exception>
    /// <exception cref="System.InvalidCastException">The value does not support integer conversion.</exception>
    /// <remarks>Numeric values outside the requested range are returned with a warning when replaying stored parameters.</remarks>
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

    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="choices">Nonempty sequence of categorical candidates; order is retained and determines stored indices.</param>
    /// <summary>Replays a stored categorical parameter value for the objective.</summary>
    /// <returns>The stored categorical value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    /// <exception cref="NumSharp.TypeError">A stored or sampled value is not a supported categorical scalar.</exception>
    /// <exception cref="System.ArgumentNullException">The choices sequence is null.</exception>
    public override CategoricalChoice SuggestCategorical(string name, IEnumerable<CategoricalChoice> choices)
        => CategoricalChoice.From(Suggest(name, new CategoricalDistribution(choices)));

    /// <summary>
    ///     Interface of report function.
    ///
    ///     Since <see cref="FrozenTrial" /> is not pruned, this report function does nothing.
    /// </summary>
    /// <param name="value">Intermediate objective value; ignored by replay trials.</param>
    /// <param name="step">Zero-based reporting step; ignored by replay trials.</param>
    public override void Report(double value, int step)
    {
    }

    /// <summary>
    ///     Suggest whether the trial should be pruned or not.
    ///
    ///     The suggestion is always <c>false</c> regardless of a pruning algorithm.
    /// </summary>
    /// <returns>Always false.</returns>
    public override bool ShouldPrune() => false;

    /// <param name="key">Attribute name; an existing attribute is overwritten.</param>
    /// <param name="value">Value retained in this replay record's local attribute dictionary.</param>
    /// <summary>Updates the local user attribute by name.</summary>
    public override void SetUserAttr(string key, object? value) => _userAttrs[key] = value;

    /// <summary>
    ///     Read this trial's observed-components vector (the residual terms written with
    ///     <see cref="BaseTrial.SetComponentLosses" />) from its user attributes. Returns <c>false</c>
    ///     when the trial carries no valid component vector — see <see cref="ComponentLosses" /> for the
    ///     contract (list-or-array tolerant, non-finite / empty rejected). On success
    ///     <paramref name="components" /> is a fresh, caller-owned copy.
    /// </summary>
    /// <param name="components">Caller-owned finite component vector on success; otherwise an empty array.</param>
    /// <returns>True when valid component losses were found; false leaves an empty array.</returns>
    public bool TryGetComponentLosses(out double[] components)
        => ComponentLosses.TryRead(UserAttrs, out components);

    /// <param name="key">Attribute name; an existing attribute is overwritten.</param>
    /// <param name="value">Value retained in this record's local internal-attribute dictionary.</param>
    /// <summary>Updates the local internal attribute by name.</summary>
    [Obsolete(DeprecationMessages.SetSystemAttr)]
    public override void SetSystemAttr(string key, object? value)
    {
        OptunaWarnings.Warn(DeprecationMessages.SetSystemAttr, WarningCategory.FutureWarning);
        SystemAttrsDict[key] = value;
    }

    // ------------------------------------------------------------------ validation

    /// <summary>Python <c>FrozenTrial._validate</c>.</summary>
    /// <exception cref="NumSharp.ValueError">State and timestamps disagree, a complete trial lacks valid values, parameter keys and distribution keys differ, or a parameter lies outside its distribution.</exception>
    /// <exception cref="NumSharp.TypeError">A categorical parameter has an unsupported scalar representation.</exception>
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
    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="distribution">Distribution defining the parameter representation and allowed values.</param>
    /// <returns>The external parameter value; an out-of-range fixed numeric value may be returned with a warning.</returns>
    /// <exception cref="NumSharp.ValueError">A required parameter is absent, its representation is invalid, or distributions are incompatible.</exception>
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
    /// <param name="value">New per-study trial number assigned by storage.</param>
    public void SetNumber(int value) => _number = value;

    /// <summary>Objective value of the trial. Throws for multi-objective trials.</summary>
    /// <exception cref="NumSharp.RuntimeError">The trial holds more than one objective value.</exception>
    /// <exception cref="System.ArgumentOutOfRangeException">The values list is non-null but empty when read.</exception>
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

    /// <param name="values">Objective values in study direction order, or null when no objective values are recorded.</param>
    /// <summary>Replaces objective values with a materialized list, or clears them for null.</summary>
    internal void SetValues(IEnumerable<double>? values) => _values = values?.ToList();

    /// <summary>Datetime where the trial started.</summary>
    public override DateTime? DatetimeStart => _datetimeStart;

    /// <param name="value">New start timestamp, or null for a waiting trial.</param>
    /// <summary>Replaces the recorded trial start time.</summary>
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

    /// <summary>Gets or replaces the mutable distribution dictionary; entries are shared distribution objects.</summary>
    public OrderedDict<string, BaseDistribution> DistributionsDict
    {
        get => _distributions;
        set => _distributions = value;
    }

    /// <summary>Dictionary that contains the user attributes.</summary>
    public override IReadOnlyDictionary<string, object?> UserAttrs => _userAttrs;

    /// <summary>Gets or replaces the mutable user-attribute dictionary.</summary>
    public OrderedDict<string, object?> UserAttrsDict
    {
        get => _userAttrs;
        set => _userAttrs = value;
    }

    /// <summary>Dictionary that contains the optuna-internal attributes.</summary>
    public override IReadOnlyDictionary<string, object?> SystemAttrs => SystemAttrsDict;

    /// <summary>Gets or replaces internal attributes, resolving any deferred storage lookup once.</summary>
    /// <exception cref="System.Exception">The deferred storage loader fails; the loader remains available for a later attempt.</exception>
    public OrderedDict<string, object?> SystemAttrsDict
    {
        get
        {
            if (_systemAttrsLoader is not null)
            {
                // Clear the loader only after success, allowing a transient backend failure to be
                // retried instead of permanently exposing an obsolete attribute snapshot.
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
    /// <param name="loader">Deferred storage read invoked on first SystemAttrsDict access; failures are not swallowed.</param>
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
    /// <param name="key">Constraint name; later writes of the same name are ignored with a warning.</param>
    /// <param name="value">Constraint value; zero or negative is feasible and NaN is forbidden.</param>
    /// <exception cref="NumSharp.ValueError">The constraint value is NaN.</exception>
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
    /// <param name="obj">Object to compare with this instance.</param>
    /// <returns>True when the other value has the same type and equal recorded fields.</returns>
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

    /// <summary>Computes a hash from the fields participating in equality.</summary>
    /// <returns>Hash code consistent with this type's equality contract.</returns>
    public override int GetHashCode() => HashCode.Combine(TrialId, _number, State);

    /// <summary>Applies the <c>&lt;</c> comparison by trial number.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator <(FrozenTrial left, FrozenTrial right) => left.Number < right.Number;

    /// <summary>Applies the <c>&gt;</c> comparison by trial number.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator >(FrozenTrial left, FrozenTrial right) => left.Number > right.Number;

    /// <summary>Applies the <c>&lt;=</c> comparison by trial number.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator <=(FrozenTrial left, FrozenTrial right) => left.Number <= right.Number;

    /// <summary>Applies the <c>&gt;=</c> comparison by trial number.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator >=(FrozenTrial left, FrozenTrial right) => left.Number >= right.Number;

    /// <summary>Compares nullable objective lists in objective order.</summary>
    /// <param name="a">First collection to compare.</param>
    /// <param name="b">Second collection to compare.</param>
    /// <returns>True when both collections contain equal corresponding values.</returns>
    private static bool ValuesEqual(List<double>? a, List<double>? b)
    {
        if (a is null || b is null)
            return a is null && b is null;
        return a.SequenceEqual(b);
    }

    /// <summary>Compares parameter or attribute dictionaries using Python-compatible nested value equality.</summary>
    /// <param name="a">First collection to compare.</param>
    /// <param name="b">Second collection to compare.</param>
    /// <returns>True when both collections contain equal corresponding values.</returns>
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

    /// <summary>Compares distribution dictionaries by parameter name and distribution value.</summary>
    /// <param name="a">First collection to compare.</param>
    /// <param name="b">Second collection to compare.</param>
    /// <returns>True when both collections contain equal corresponding values.</returns>
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

    /// <summary>Compares intermediate observations by step and numeric value.</summary>
    /// <param name="a">First collection to compare.</param>
    /// <param name="b">Second collection to compare.</param>
    /// <returns>True when both collections contain equal corresponding values.</returns>
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
    /// <returns>A new trial wrapper sharing dictionaries, values, and the deferred loader with this instance.</returns>
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
    /// <returns>A new trial with copied value lists and supported nested attribute containers; distribution instances remain shared.</returns>
    public FrozenTrial DeepCopy()
    {
        // Nested JSON-compatible values need separate containers because users commonly mutate
        // copied trial metadata; distributions remain shared under the port's immutable-use convention.
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

    /// <summary>Copies an attribute dictionary through the Python-compatible nested-value copier.</summary>
    /// <param name="attrs">Attribute dictionary to copy.</param>
    /// <returns>A new dictionary with supported nested containers copied.</returns>
    private static OrderedDict<string, object?> DeepCopyAttrs(OrderedDict<string, object?> attrs)
    {
        var copy = new OrderedDict<string, object?>(attrs.Count);
        foreach (var (key, value) in attrs)
            copy[key] = PyJson.DeepCopy(value);
        return copy;
    }

    /// <summary>Python <c>__repr__</c> (used in logs and debugging).</summary>
    /// <returns>Python-style representation of the recorded trial fields.</returns>
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
    /// <param name="state">Trial state to record or inspect.</param>
    /// <param name="value">Single objective value, mutually exclusive with values.</param>
    /// <param name="values">Objective values in study direction order, or null when no objective values are recorded.</param>
    /// <param name="params">Parameter names mapped to their external values.</param>
    /// <param name="distributions">Distribution for each parameter, keyed by the same names as the parameter values.</param>
    /// <param name="userAttrs">User metadata; values should be JSON serializable for persistent storage.</param>
    /// <param name="systemAttrs">Internal metadata used by samplers and storage; preserve reserved keys.</param>
    /// <param name="intermediateValues">Objective observations keyed by reporting step.</param>
    /// <param name="constraints">Optional named constraints; zero or negative values are feasible.</param>
    /// <returns>A validated trial with placeholder number and storage identifier of -1.</returns>
    /// <exception cref="NumSharp.ValueError">Both value forms are supplied, the state has invalid values, parameter/distribution keys or bounds disagree, or a constraint is NaN.</exception>
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
