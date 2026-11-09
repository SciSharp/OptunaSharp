using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
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

    // Each dictionary field holds the OrderedDict the trial was built with, or, once a storage built with
    // compactFinishedTrials compacted the finished record (Rustuna plan 13, item S12), an immutable compact form
    // (ICompactDictionary). The read-only accessors return either as is; the mutable …Dict accessors swap a compact form
    // for a real dictionary first (Decompact*), so a write through them is never lost.
    private IReadOnlyDictionary<string, object?> _params;
    private IReadOnlyDictionary<string, object?> _userAttrs;
    private IReadOnlyDictionary<string, object?> _systemAttrs;
    private IReadOnlyDictionary<string, BaseDistribution> _distributions;
    private OrderedDict<long, double> _intermediateValues;

    /// <summary>
    ///     What a compact record holds in place of an empty <see cref="IntermediateValues" /> dictionary (plan 13, item
    ///     S12): shared by every such record and never handed out, because the public getter returns a mutable dictionary.
    ///     Internal reads may use it as the empty dictionary it is; <see cref="IntermediateValues" /> replaces it with a
    ///     record's own dictionary before returning.
    /// </summary>
    private static readonly OrderedDict<long, double> CompactEmptyIntermediateValues = new();
    /// <summary>Storage identifier for this trial, separate from its per-study number.</summary>
    internal int TrialId;

    /// <summary>
    ///     Read and written only by <see cref="Storages.InMemoryStorage" />, under its lock: true while that storage
    ///     holds the only reference to this record and the record owns every one of its dictionaries, so a write may
    ///     change it in place instead of copying it (Rustuna plan 13, item S4).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         False for every record created any other way, which is the safe default: a write then copies, as
    ///         Optuna's copy-on-write storage always does. The storage clears it the moment a reference leaves (any
    ///         read that returns this object), so a caller that holds a record never sees it change. That is the
    ///         invariant Optuna's copying protected, kept at a fraction of the cost: a trial's suggestions write its
    ///         record repeatedly while nobody reads it, which made every suggestion copy both parameter dictionaries.
    ///     </para>
    ///     <para>
    ///         Not copied by <see cref="ShallowCopy" /> or <see cref="DeepCopy" />: a shallow copy shares the
    ///         dictionaries, and only the storage knows when it alone holds a record.
    ///     </para>
    /// </remarks>
    internal bool StorageExclusive;

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
        OrderedDict<long, double> intermediateValues,
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
        _intermediateValues = intermediateValues;
        _distributions = distributions;
        TrialId = trialId;
    }

    /// <summary><see cref="TrialState" /> of the trial.</summary>
    public TrialState State { get; set; }

    /// <summary>Datetime where the trial finished.</summary>
    public DateTime? DatetimeComplete { get; set; }

    /// <summary>
    ///     Intermediate objective values set with <see cref="Trial.Report" />, keyed by step. Steps are 64-bit, like
    ///     Python's unbounded <c>int</c> steps in journal, SQLite and in-memory storages.
    /// </summary>
    /// <remarks>
    ///     On a record an <see cref="Storages.InMemoryStorage" /> built with <c>compactFinishedTrials: true</c> compacted
    ///     while it had no intermediate values, the first read gives the record its own empty dictionary and keeps it, so
    ///     the dictionary returned here is always the record's and a change to it stays (plan 13, item S12).
    /// </remarks>
    public OrderedDict<long, double> IntermediateValues
    {
        get
        {
            OrderedDict<long, double> current = _intermediateValues;
            if (!ReferenceEquals(current, CompactEmptyIntermediateValues))
                return current;
            // Only one dictionary may become the record's: a reader that loses the race returns the winner's.
            var own = new OrderedDict<long, double>();
            OrderedDict<long, double> witnessed = Interlocked.CompareExchange(ref _intermediateValues, own, current);
            return ReferenceEquals(witnessed, current) ? own : witnessed;
        }
        set => _intermediateValues = value;
    }

    /// <summary>
    ///     The intermediate values for internal reads that change nothing: the record's dictionary, or the shared empty
    ///     one of a compact record, which <see cref="IntermediateValues" /> would replace with a dictionary of its own.
    /// </summary>
    internal OrderedDict<long, double> IntermediateValuesForRead => _intermediateValues;

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
    /// <param name="step">Zero-based reporting step (64-bit, like every trial's); ignored by replay trials.</param>
    public override void Report(double value, long step)
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
    /// <remarks>Writes through <see cref="UserAttrsDict" />, so a compact record gets a dictionary of its own first.</remarks>
    public override void SetUserAttr(string key, object? value) => UserAttrsDict[key] = value;

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

    // FrozenTrial.set_system_attr (deprecated 3.1.0) was removed in Optuna 5.0.0 (#6834). The record's
    // optuna-internal attributes stay readable through SystemAttrs and replaceable through
    // SystemAttrsDict, mirroring the system_attrs property/setter Optuna keeps on FrozenTrial.

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

        // Through the mutable accessor: a compact record gets a dictionary of its own before the write.
        DistributionsDict[name] = distribution;

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
    /// <remarks>
    ///     On a finished record of an <see cref="Storages.InMemoryStorage" /> built with <c>compactFinishedTrials: true</c>
    ///     this is a read-only view over the trial's internal values (plan 13, item S12): the same entries in the same
    ///     order, each numeric value a new box with the stored bits on every read, and not an
    ///     <see cref="OrderedDict{TKey,TValue}" />. <see cref="ParamsDict" /> returns a real dictionary in every case.
    /// </remarks>
    public override IReadOnlyDictionary<string, object?> Params => _params;

    /// <summary>The mutable parameter dictionary (Python's <c>params</c> setter).</summary>
    /// <remarks>
    ///     A compact record (see <see cref="Params" />) gets a dictionary of its own on the first read, which it keeps:
    ///     every later read returns that dictionary, and a change to it is the trial's change. It costs the memory the
    ///     compact form saved, so internal readers that only read use <see cref="Params" />.
    /// </remarks>
    public OrderedDict<string, object?> ParamsDict
    {
        get => _params as OrderedDict<string, object?> ?? Decompact(ref _params);
        set => _params = value;
    }

    /// <summary>Dictionary that contains the distributions of <see cref="Params" />.</summary>
    /// <remarks>
    ///     On a compact record (see <see cref="Params" />), a read-only view that every trial of the same parameter
    ///     schema shares; <see cref="DistributionsDict" /> returns a real dictionary in every case.
    /// </remarks>
    public override IReadOnlyDictionary<string, BaseDistribution> Distributions => _distributions;

    /// <summary>Gets or replaces the mutable distribution dictionary; entries are shared distribution objects.</summary>
    /// <remarks>A compact record gets a dictionary of its own on the first read, as <see cref="ParamsDict" /> does.</remarks>
    public OrderedDict<string, BaseDistribution> DistributionsDict
    {
        get => _distributions as OrderedDict<string, BaseDistribution> ?? Decompact(ref _distributions);
        set => _distributions = value;
    }

    /// <summary>Dictionary that contains the user attributes.</summary>
    /// <remarks>
    ///     On a compact record (see <see cref="Params" />), a read-only view: long ASCII strings come back as new string
    ///     instances with the stored characters, every other value as the stored object.
    /// </remarks>
    public override IReadOnlyDictionary<string, object?> UserAttrs => _userAttrs;

    /// <summary>Gets or replaces the mutable user-attribute dictionary.</summary>
    /// <remarks>A compact record gets a dictionary of its own on the first read, as <see cref="ParamsDict" /> does.</remarks>
    public OrderedDict<string, object?> UserAttrsDict
    {
        get => _userAttrs as OrderedDict<string, object?> ?? Decompact(ref _userAttrs);
        set => _userAttrs = value;
    }

    /// <summary>
    ///     Dictionary that contains the optuna-internal attributes. Still part of Optuna 5.0.0's
    ///     <c>FrozenTrial</c> (only <c>BaseTrial</c>/<c>Trial</c>/<c>Study</c> lost theirs in #6834), so samplers
    ///     read system attributes of finished trials through it.
    /// </summary>
    /// <remarks>On a compact record, a read-only view as <see cref="UserAttrs" /> describes.</remarks>
    /// <exception cref="System.Exception">The deferred storage loader fails; the loader remains available for a later attempt.</exception>
    public IReadOnlyDictionary<string, object?> SystemAttrs
    {
        get
        {
            ResolveLazySystemAttrs();
            return _systemAttrs;
        }
    }

    /// <summary>Gets or replaces internal attributes, resolving any deferred storage lookup once.</summary>
    /// <remarks>A compact record gets a dictionary of its own on the first read, as <see cref="ParamsDict" /> does.</remarks>
    /// <exception cref="System.Exception">The deferred storage loader fails; the loader remains available for a later attempt.</exception>
    public OrderedDict<string, object?> SystemAttrsDict
    {
        get
        {
            ResolveLazySystemAttrs();
            return _systemAttrs as OrderedDict<string, object?> ?? Decompact(ref _systemAttrs);
        }
        set
        {
            _systemAttrsLoader = null;
            _systemAttrs = value;
        }
    }

    /// <summary>Loads deferred system attributes, if any are pending, before either accessor returns them.</summary>
    /// <exception cref="System.Exception">The deferred storage loader fails; the loader remains available for a later attempt.</exception>
    private void ResolveLazySystemAttrs()
    {
        if (_systemAttrsLoader is not null)
        {
            // Clear the loader only after success, allowing a transient backend failure to be
            // retried instead of permanently exposing an obsolete attribute snapshot.
            _systemAttrs = _systemAttrsLoader();
            _systemAttrsLoader = null;
        }
    }

    /// <summary>Python <c>Trial._get_latest_trial</c> support: defer system-attr loading to the storage.</summary>
    /// <param name="loader">Deferred storage read invoked on first SystemAttrsDict access; failures are not swallowed.</param>
    internal void SetLazySystemAttrs(Func<OrderedDict<string, object?>> loader) => _systemAttrsLoader = loader;

    // ------------------------------------------------------------------ compact records (plan 13, item S12)

    /// <summary>
    ///     Replaces the compact form in <paramref name="field" /> with a real dictionary of the same entries and returns
    ///     it, so the caller of a mutable accessor gets the dictionary the record keeps.
    /// </summary>
    /// <typeparam name="TValue">The dictionary's value type.</typeparam>
    /// <param name="field">One of the record's dictionary fields, holding a compact form or null.</param>
    /// <returns>The record's dictionary from now on; null when the field held null (a record built with a null dictionary keeps returning null).</returns>
    /// <remarks>
    ///     Readers on other threads may call this at once on a shared finished record: the compare-exchange lets exactly
    ///     one materialized dictionary become the record's, and every caller returns that one, so no caller can write into
    ///     a dictionary the record does not keep. A reader that loaded the compact form a moment earlier keeps reading it,
    ///     which is safe because a compact form never changes and holds the same entries.
    /// </remarks>
    private static OrderedDict<string, TValue> Decompact<TValue>(ref IReadOnlyDictionary<string, TValue> field)
    {
        IReadOnlyDictionary<string, TValue> observed = field;
        while (true)
        {
            switch (observed)
            {
                case null:
                    return null!;
                case OrderedDict<string, TValue> dictionary:
                    return dictionary;
            }
            OrderedDict<string, TValue> own = ((ICompactDictionary<string, TValue>)observed).Materialize();
            IReadOnlyDictionary<string, TValue> witnessed = Interlocked.CompareExchange(ref field, own, observed);
            if (ReferenceEquals(witnessed, observed))
                return own;
            // Another thread (or a setter) replaced the form first: return whatever the record holds now.
            observed = witnessed;
        }
    }

    /// <summary>
    ///     True while any of the record's dictionaries is held in compact form (plan 13, item S12); a diagnostic the tests
    ///     read to prove which records a storage compacted and which a mutable accessor turned back.
    /// </summary>
    internal bool IsCompact
        => _params is ICompactDictionary<string, object?>
           || _distributions is ICompactDictionary<string, BaseDistribution>
           || _userAttrs is ICompactDictionary<string, object?>
           || _systemAttrs is ICompactDictionary<string, object?>
           || ReferenceEquals(_intermediateValues, CompactEmptyIntermediateValues);

    /// <summary>
    ///     Stores a finished record compactly (plan 13, item S12): parameters as internal values against a schema shared
    ///     by the study's trials, attributes as arrays with long ASCII strings as bytes, and nothing for empty
    ///     dictionaries. Every accessor returns the same entries afterwards.
    /// </summary>
    /// <param name="schemas">The study's schema cache; the caller (the storage) holds the lock that guards it.</param>
    /// <remarks>
    ///     <para>
    ///         The caller must hold the only reference to this record object: the storage calls it on the copy it is about
    ///         to publish, under its lock, so no reader can see the fields change. The record's dictionaries are only
    ///         read, never changed, because the record it was copied from (and a reader holding that one) may share
    ///         them; after this call the published record no longer references them.
    ///     </para>
    ///     <para>
    ///         Parameters stay a dictionary when they cannot be compacted exactly: names in another order than the
    ///         distributions, more parameter shapes than <see cref="ParamSchemaCache.MaxSchemas" />, or a value that
    ///         would not read back identical (see <see cref="CompactParams.TryCreate" />). A record with deferred
    ///         system attributes keeps them deferred.
    ///     </para>
    /// </remarks>
    internal void CompactFinished(ParamSchemaCache schemas)
    {
        if (_params is OrderedDict<string, object?> @params
            && _distributions is OrderedDict<string, BaseDistribution> distributions)
        {
            ParamSchema? schema = schemas.FindOrAdd(@params, distributions);
            CompactParams? compact = schema is null ? null : CompactParams.TryCreate(schema, @params);
            if (compact is not null)
            {
                _params = compact;
                _distributions = compact.Schema.View;
            }
        }

        if (_userAttrs is OrderedDict<string, object?> userAttrs)
            _userAttrs = CompactAttrs.Create(userAttrs);
        if (_systemAttrsLoader is null && _systemAttrs is OrderedDict<string, object?> systemAttrs)
            _systemAttrs = CompactAttrs.Create(systemAttrs);

        OrderedDict<long, double> intermediate = _intermediateValues;
        if (intermediate is { Count: 0 })
        {
            _intermediateValues = CompactEmptyIntermediateValues;
        }
        else if (intermediate is not null)
        {
            // Kept as a dictionary (the public getter returns the mutable type), but as a copy sized to its entries:
            // reports grow it by prime steps, and the source may be shared with the record this one was copied from.
            var trimmed = new OrderedDict<long, double>(intermediate.Count);
            foreach (var (step, value) in intermediate)
                trimmed.Add(step, value);
            _intermediateValues = trimmed;
        }
    }

    /// <summary>Return the maximum step of <see cref="IntermediateValues" /> in the trial (64-bit, like the steps).</summary>
    public long? LastStep
    {
        get
        {
            // The read-only view, so a compact record without reports does not get a dictionary just to say so.
            OrderedDict<long, double> intermediateValues = IntermediateValuesForRead;
            if (intermediateValues.Count == 0)
                return null;
            return intermediateValues.Keys.Max();
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
        => ConstrainedOptimization.GetConstraintsFromSystemAttrs(SystemAttrs);

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
            // The read-only accessors: comparing must not give a compact record dictionaries of its own.
            && DictValueEquals(SystemAttrs, other.SystemAttrs)
            && IntermediateValuesEqual(IntermediateValuesForRead, other.IntermediateValuesForRead)
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
    /// <param name="a">First collection to compare; a dictionary or a compact record's read-only view.</param>
    /// <param name="b">Second collection to compare; a dictionary or a compact record's read-only view.</param>
    /// <returns>True when both collections contain equal corresponding values.</returns>
    private static bool DictValueEquals(IReadOnlyDictionary<string, object?> a, IReadOnlyDictionary<string, object?> b)
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
    /// <param name="a">First collection to compare; a dictionary or a compact record's read-only view.</param>
    /// <param name="b">Second collection to compare; a dictionary or a compact record's read-only view.</param>
    /// <returns>True when both collections contain equal corresponding values.</returns>
    private static bool DistributionsEqual(
        IReadOnlyDictionary<string, BaseDistribution> a, IReadOnlyDictionary<string, BaseDistribution> b)
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
    private static bool IntermediateValuesEqual(OrderedDict<long, double> a, OrderedDict<long, double> b)
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
    /// <remarks>
    ///     A compact record (plan 13, item S12) first gets real dictionaries of its own, which the copy then shares:
    ///     sharing the compact forms instead would let each object later take a different dictionary, and a change
    ///     through one would no longer show in the other as it does for Python's shallow copy. The storages copy only
    ///     unfinished records, which are never compact, so this costs nothing on their paths.
    /// </remarks>
    internal FrozenTrial ShallowCopy()
    {
        // Field by field rather than through the accessors: SystemAttrsDict would run a pending deferred loader, which
        // a shallow copy hands over unrun instead.
        var copy = new FrozenTrial(
            number: _number,
            state: State,
            value: null,
            datetimeStart: _datetimeStart,
            datetimeComplete: DatetimeComplete,
            @params: _params as OrderedDict<string, object?> ?? Decompact(ref _params),
            distributions: _distributions as OrderedDict<string, BaseDistribution> ?? Decompact(ref _distributions),
            userAttrs: _userAttrs as OrderedDict<string, object?> ?? Decompact(ref _userAttrs),
            systemAttrs: _systemAttrs as OrderedDict<string, object?> ?? Decompact(ref _systemAttrs),
            intermediateValues: IntermediateValues,
            trialId: TrialId,
            values: null);
        copy._values = _values;
        copy._systemAttrsLoader = _systemAttrsLoader;
        return copy;
    }

    /// <summary>Python <c>copy.deepcopy(trial)</c>.</summary>
    /// <returns>A new trial with copied value lists and supported nested attribute containers; distribution instances remain shared.</returns>
    /// <remarks>
    ///     The copy always holds real dictionaries, including the copy of a compact record (plan 13, item S12), which is
    ///     read through its views and left compact itself.
    /// </remarks>
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
            distributions: CloneDistributions(_distributions),  // Distributions are treated as immutable.
            userAttrs: DeepCopyAttrs(_userAttrs),
            systemAttrs: DeepCopyAttrs(SystemAttrs),
            intermediateValues: IntermediateValuesForRead.ShallowClone(),
            trialId: TrialId,
            values: null);
        copy._values = _values is null ? null : new List<double>(_values);
        return copy;
    }

    /// <summary>A new distributions dictionary sharing the distribution instances, as <c>ShallowClone</c> builds one.</summary>
    /// <param name="distributions">The record's dictionary, or a compact record's shared view.</param>
    /// <returns>A dictionary the copy owns, in the same order.</returns>
    private static OrderedDict<string, BaseDistribution> CloneDistributions(
        IReadOnlyDictionary<string, BaseDistribution> distributions)
        => distributions is OrderedDict<string, BaseDistribution> dictionary
            ? dictionary.ShallowClone()
            : new OrderedDict<string, BaseDistribution>(distributions);

    /// <summary>Copies an attribute dictionary through the Python-compatible nested-value copier.</summary>
    /// <param name="attrs">Attribute dictionary to copy; a dictionary or a compact record's read-only view.</param>
    /// <returns>A new dictionary with supported nested containers copied.</returns>
    private static OrderedDict<string, object?> DeepCopyAttrs(IReadOnlyDictionary<string, object?> attrs)
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
            // Every field is an IDictionary: an OrderedDict, or a compact record's view, which prints the same entries.
            $"params={PythonRepr.DictRepr((IDictionary)_params)}",
            $"user_attrs={PythonRepr.DictRepr((IDictionary)_userAttrs)}",
            $"system_attrs={PythonRepr.DictRepr((IDictionary)_systemAttrs)}",
            $"intermediate_values={PythonRepr.DictRepr(IntermediateValuesForRead)}",
            $"distributions={PythonRepr.DictRepr((IDictionary)_distributions)}",
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
        OrderedDict<long, double>? intermediateValues = null,
        OrderedDict<string, double>? constraints = null)
    {
        @params ??= new OrderedDict<string, object?>();
        distributions ??= new OrderedDict<string, BaseDistribution>();
        var convertedDistributions = new OrderedDict<string, BaseDistribution>(distributions.Count);
        foreach (var (key, dist) in distributions)
            convertedDistributions[key] = Distribution.ConvertOldDistributionToNewDistribution(dist);
        userAttrs ??= new OrderedDict<string, object?>();
        systemAttrs ??= new OrderedDict<string, object?>();
        intermediateValues ??= new OrderedDict<long, double>();

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
