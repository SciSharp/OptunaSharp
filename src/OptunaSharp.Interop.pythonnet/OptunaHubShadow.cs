using System;
using System.Collections;
using System.Collections.Generic;
using Python.Runtime;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Storages;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     The persistent Python <b>shadow study</b> the OptunaHub bridges (<see cref="OptunaHubSampler" />,
///     <see cref="OptunaHubPruner" />) hand to the Python sampler or pruner: an in-memory <c>optuna.Study</c> kept in
///     lockstep with one OptunaSharp study, trial for trial, so the Python object reads and writes what it would in a
///     native Optuna run.
/// </summary>
/// <remarks>
///     <para>
///         <b>Mirroring in.</b> Before every call into Python, <see cref="Sync" /> brings every OptunaSharp trial, in
///         number order, into the shadow: a trial it has not seen is created with its state, values, parameters,
///         distributions, user and system attributes and intermediate values; a mirrored trial that is not finished
///         receives the parameters, reports and attributes written since, then its new state. The study's user and
///         system attributes follow. Finished mirrors are skipped, as finished trials no longer change on either side.
///         The bridges used to rebuild a study per trial from <c>create_trial(state, values, params, distributions)</c>,
///         which lost every system attribute a hub object wrote and never forwarded <c>before_trial</c> /
///         <c>after_trial</c>, so PSO never built its swarm, MO-CMA failed with <c>KeyError: 'mocma:g'</c> and successive
///         halving never pruned (QA-72, QA-73).
///     </para>
///     <para>
///         <b>The trial objects Python receives</b> are the ones native Optuna passes. Each running trial keeps the
///         record its storage held when the trial was first seen (<see cref="CachedTrial" />, Optuna's
///         <c>Trial._cached_frozen_trial</c>), and the mirrored parameters, reports and user attributes update that
///         record in place, as <c>Trial._suggest</c>, <c>report</c> and <c>set_user_attr</c> do; system attributes go
///         to the storage only. <see cref="LatestTrial" /> is <c>Trial._get_latest_trial()</c> (a shallow copy whose
///         system attributes are read from the storage), what <c>sample_independent</c> and <c>prune</c> receive;
///         <see cref="StoredTrial" /> is <c>storage.get_trial</c>, what <c>after_trial</c> receives.
///     </para>
///     <para>
///         <b>Mirroring out.</b> <see cref="Call{T}" /> snapshots the study's and the trial's user and system attributes
///         around the Python call and writes every key the call set to the OptunaSharp storage, so the C# study holds
///         what a native run would store (PSO's swarm, MO-CMA's <c>mocma:g</c>, successive halving's
///         <c>completed_rung_N</c>). A value with no JSON shape (a NumPy array, an arbitrary object) stays in the shadow
///         only; native Optuna could not store it outside memory either.
///     </para>
///     <para>
///         Not thread-safe: the owning bridge serializes calls under its gate, on the Python thread
///         (<see cref="PythonRuntime" />), holding the GIL. One shadow serves one OptunaSharp study; the bridges keep one
///         per study they meet. Trials of other workers are mirrored with the state they have when a call syncs, as a
///         native sampler sees them in the storage.
///     </para>
/// </remarks>
internal sealed class OptunaHubShadow : IDisposable
{
    /// <summary>The shadow <c>optuna.Study</c> (owned).</summary>
    private readonly PyObject _study;

    /// <summary>One mirror per OptunaSharp trial, indexed by trial number (the shadow numbers its trials alike).</summary>
    private readonly List<Mirror> _mirrors = new();

    /// <summary>The study user attributes as last synchronized, in OptunaSharp values, by key.</summary>
    private readonly Dictionary<string, object?> _studyUserAttrs = new();

    /// <summary>The study system attributes as last synchronized, in OptunaSharp values, by key.</summary>
    private readonly Dictionary<string, object?> _studySystemAttrs = new();

    /// <summary>Wraps a new shadow study.</summary>
    /// <param name="study">The owned <c>optuna.Study</c>.</param>
    private OptunaHubShadow(PyObject study) => _study = study;

    /// <summary>The shadow <c>optuna.Study</c>, borrowed: pass it to the Python object, never dispose it.</summary>
    internal PyObject Study => _study;

    /// <summary>
    ///     Creates the shadow of <paramref name="study" />: an empty in-memory study with its directions and its name (a
    ///     hub object may derive state from the name: <c>HyperbandPruner</c>'s brackets do).
    /// </summary>
    /// <param name="study">The OptunaSharp study to shadow.</param>
    /// <returns>The shadow, empty until the first <see cref="Sync" />.</returns>
    /// <exception cref="PythonException">Optuna's <c>create_study</c> fails.</exception>
    internal static OptunaHubShadow Create(Study study)
    {
        OptunaHubRuntime.EnsureQuiet();  // re-assert right before create_study (Optuna re-configures logging lazily)
        return new OptunaHubShadow(Python.optuna.create_study(study.Directions, study.StudyName));
    }

    /// <summary>
    ///     Mirrors every trial of <paramref name="study" /> and its attributes into the shadow (see the class remarks).
    ///     Reads the storage afresh, so values the bridge wrote back are recognized and not echoed.
    /// </summary>
    /// <param name="study">The OptunaSharp study this shadow mirrors.</param>
    /// <exception cref="OptunaHubInteropException">The study's trial numbers are not contiguous from 0.</exception>
    /// <exception cref="PythonException">A shadow storage write fails.</exception>
    internal void Sync(Study study)
    {
        BaseStorage storage = study.Storage;
        foreach (FrozenTrial trial in storage.GetAllTrials(study.StudyId, deepcopy: false))
        {
            if (trial.Number < _mirrors.Count)
            {
                Mirror mirror = _mirrors[trial.Number];
                if (!mirror.State.IsFinished())
                    Update(mirror, trial);
                continue;
            }
            if (trial.Number != _mirrors.Count)
                throw new OptunaHubInteropException(
                    $"The OptunaHub shadow study expected trial number {_mirrors.Count} next but read {trial.Number}.");
            _mirrors.Add(Add(trial));
        }

        foreach (var (key, value) in storage.GetStudyUserAttrs(study.StudyId))
        {
            if (Unchanged(_studyUserAttrs, key, value) || !TryToPy(value, out PyObject? py))
                continue;
            using (py)
            using (var k = key.ToPython())
                Python.optunasharp_shadow.set_study_user_attr(_study, k, py!);
            _studyUserAttrs[key] = value;
        }
        foreach (var (key, value) in storage.GetStudySystemAttrs(study.StudyId))
        {
            if (Unchanged(_studySystemAttrs, key, value) || !TryToPy(value, out PyObject? py))
                continue;
            using (py)
            using (var k = key.ToPython())
                Python.optunasharp_shadow.set_study_system_attr(_study, k, py!);
            _studySystemAttrs[key] = value;
        }
    }

    /// <summary>
    ///     Starts a trial the way <c>Study.ask</c> does: clears the shadow study's per-thread trial cache, so the samplers
    ///     that read the history through <c>study._get_trials(use_cache=True)</c> see it as of this trial. Without it the
    ///     shadow kept its first read for good: restart CMA-ES diverged from trial 1, TPE from its first model-based
    ///     trial, the NSGA-II variants from their first generation boundary.
    /// </summary>
    /// <exception cref="PythonException">The shadow study has no per-thread cache (an incompatible Optuna).</exception>
    internal void BeginTrial() => Python.optunasharp_shadow.begin_trial(_study);

    /// <summary>The mirror of <paramref name="trial" />, which <see cref="Sync" /> has created.</summary>
    /// <param name="trial">An OptunaSharp trial of the shadowed study.</param>
    /// <returns>Its mirror.</returns>
    /// <exception cref="OptunaHubInteropException">The trial is not mirrored (no <see cref="Sync" /> since it was created).</exception>
    internal Mirror MirrorOf(FrozenTrial trial)
    {
        if (trial.Number < 0 || trial.Number >= _mirrors.Count)
            throw new OptunaHubInteropException(
                $"Trial {trial.Number} is not mirrored in the OptunaHub shadow study ({_mirrors.Count} trials).");
        return _mirrors[trial.Number];
    }

    /// <summary>
    ///     The running trial's cached record, Optuna's <c>Trial._cached_frozen_trial</c>: the storage record when the
    ///     trial is first seen (or, with <paramref name="refresh" />, now), updated in place as the trial writes.
    /// </summary>
    /// <param name="mirror">A running trial's mirror.</param>
    /// <param name="refresh">
    ///     Take the record anew, as <c>Trial.__init__</c> does: <c>before_trial</c> is the start of the trial.
    /// </param>
    /// <returns>The record, borrowed from the mirror: never dispose it.</returns>
    /// <exception cref="PythonException">The shadow storage read fails.</exception>
    internal PyObject CachedTrial(Mirror mirror, bool refresh = false)
    {
        if (refresh)
        {
            mirror.Cached?.Dispose();
            mirror.Cached = null;
        }
        return mirror.Cached ??= Python.optunasharp_shadow.get_trial(_study, mirror.TrialId);
    }

    /// <summary>
    ///     Optuna's <c>Trial._get_latest_trial()</c>: a shallow copy of the cached record whose system attributes are read
    ///     from the storage on first use, what <c>sample_independent</c> and <c>prune</c> receive.
    /// </summary>
    /// <param name="mirror">A running trial's mirror.</param>
    /// <returns>A new owned reference; dispose it under the GIL.</returns>
    /// <exception cref="PythonException">The copy fails.</exception>
    internal PyObject LatestTrial(Mirror mirror)
        => Python.optunasharp_shadow.latest(_study, CachedTrial(mirror));

    /// <summary>The trial's current storage record, <c>storage.get_trial(trial_id)</c>, what <c>after_trial</c> receives.</summary>
    /// <param name="mirror">The trial's mirror.</param>
    /// <returns>A new owned reference; dispose it under the GIL.</returns>
    /// <exception cref="PythonException">The shadow storage read fails.</exception>
    internal PyObject StoredTrial(Mirror mirror) => Python.optunasharp_shadow.get_trial(_study, mirror.TrialId);

    /// <summary>
    ///     Runs a call into the Python sampler or pruner and writes every study or trial attribute it set to the
    ///     OptunaSharp storage (see the class remarks).
    /// </summary>
    /// <param name="study">The OptunaSharp study the attributes are written to.</param>
    /// <param name="trial">The OptunaSharp trial being sampled, told or pruned.</param>
    /// <param name="mirror">Its mirror.</param>
    /// <param name="call">The Python call; it runs under the GIL.</param>
    /// <typeparam name="T">The call's result.</typeparam>
    /// <returns>The call's result.</returns>
    /// <exception cref="PythonException">The Python call, or reading the attributes back, fails.</exception>
    internal T Call<T>(Study study, FrozenTrial trial, Mirror mirror, Func<T> call)
    {
        using PyObject before = Python.optunasharp_shadow.attrs(_study, mirror.TrialId);
        T result = call();
        using PyObject after = Python.optunasharp_shadow.attrs(_study, mirror.TrialId);

        // The four dictionaries, in attrs()'s order: study user, study system, trial user, trial system.
        BaseStorage storage = study.Storage;
        WriteBack(before, after, 0, (key, value) => storage.SetStudyUserAttr(study.StudyId, key, value), _studyUserAttrs);
        WriteBack(before, after, 1, (key, value) => storage.SetStudySystemAttr(study.StudyId, key, value), _studySystemAttrs);
        WriteBack(before, after, 2, (key, value) => storage.SetTrialUserAttr(trial.TrialId, key, value), mirror.UserAttrs);
        WriteBack(before, after, 3, (key, value) => storage.SetTrialSystemAttr(trial.TrialId, key, value), mirror.SystemAttrs);
        return result;
    }

    /// <summary>
    ///     Finishes the trial's mirror with the state and values OptunaSharp stores, as <c>_tell_with_warning</c> does after
    ///     <c>after_trial</c>, and drops its cached record.
    /// </summary>
    /// <param name="mirror">The trial's mirror.</param>
    /// <param name="state">The finished state.</param>
    /// <param name="values">The objective values, or null.</param>
    /// <exception cref="PythonException">The shadow storage write fails.</exception>
    internal void Finish(Mirror mirror, TrialState state, IReadOnlyList<double>? values)
    {
        if (mirror.State.IsFinished())
            return;
        using (PyObject pyState = Python.optuna.trial.TrialState(state))
        using (PyObject pyValues = ToPyValues(values))
            Python.optunasharp_shadow.set_state(_study, mirror.TrialId, pyState, pyValues);
        mirror.State = state;
        mirror.Cached?.Dispose();
        mirror.Cached = null;
    }

    /// <summary>Releases the shadow study and every cached record; call under the GIL.</summary>
    public void Dispose()
    {
        foreach (Mirror mirror in _mirrors)
        {
            mirror.Cached?.Dispose();
            mirror.Cached = null;
        }
        _study.Dispose();
    }

    // ---- mirroring ---------------------------------------------------------------------------------------------------

    /// <summary>Creates the next shadow trial from an OptunaSharp trial, whatever its state.</summary>
    /// <param name="trial">The OptunaSharp trial; its number is the shadow's next.</param>
    /// <returns>The new mirror.</returns>
    /// <exception cref="PythonException">Optuna rejects the trial or the storage write fails.</exception>
    private Mirror Add(FrozenTrial trial)
    {
        var mirror = new Mirror { State = trial.State };
        using var pyParams = new PyDict();
        using var pyDistributions = new PyDict();
        foreach (var (name, value) in trial.Params)
        {
            using (PyObject v = PyConvert.ToPy(value))
                pyParams[name] = v;
            using (PyObject d = PyConvert.ToPy(trial.Distributions[name]))
                pyDistributions[name] = d;
            mirror.Params.Add(name);
        }
        using PyDict pyUserAttrs = ToPyAttrs(trial.UserAttrs, mirror.UserAttrs);
        using PyDict pySystemAttrs = ToPyAttrs(trial.SystemAttrs, mirror.SystemAttrs);
        using var pyIntermediate = new PyDict();
        foreach (var (step, value) in trial.IntermediateValues)
        {
            using var s = new PyInt(step);
            using var v = value.ToPython();
            pyIntermediate.SetItem(s, v);
            mirror.Steps.Add(step);
        }
        using PyObject pyState = Python.optuna.trial.TrialState(trial.State);
        using PyObject pyValues = ToPyValues(trial.Values);
        mirror.TrialId = Python.optunasharp_shadow.add_trial(
            _study, pyState, pyValues, pyParams, pyDistributions, pyUserAttrs, pySystemAttrs, pyIntermediate);
        return mirror;
    }

    /// <summary>
    ///     Brings an unfinished mirror up to its OptunaSharp trial: new parameters, reports and attributes in the order
    ///     a native trial writes them, then the trial's new state.
    /// </summary>
    /// <param name="mirror">The unfinished mirror.</param>
    /// <param name="trial">The trial as the OptunaSharp storage holds it now.</param>
    /// <exception cref="PythonException">A shadow storage write fails.</exception>
    private void Update(Mirror mirror, FrozenTrial trial)
    {
        // WAITING -> RUNNING first: a popped enqueued trial is written to only once it runs.
        if (mirror.State == TrialState.Waiting && trial.State != TrialState.Waiting && !trial.State.IsFinished())
        {
            using (PyObject running = Python.optuna.trial.TrialState(TrialState.Running))
            using (PyObject none = ToPyValues(null))
                Python.optunasharp_shadow.set_state(_study, mirror.TrialId, running, none);
            mirror.State = TrialState.Running;
        }

        foreach (var (name, value) in trial.Params)
        {
            if (!mirror.Params.Add(name))
                continue;
            using PyObject v = PyConvert.ToPy(value);
            using PyObject d = PyConvert.ToPy(trial.Distributions[name]);
            using var n = name.ToPython();
            Python.optunasharp_shadow.set_param(_study, mirror.Cached, mirror.TrialId, n, v, d);
        }
        foreach (var (step, value) in trial.IntermediateValues)
        {
            if (!mirror.Steps.Add(step))
                continue;
            using var s = new PyInt(step);
            using var v = value.ToPython();
            Python.optunasharp_shadow.report(_study, mirror.Cached, mirror.TrialId, s, v);
        }
        foreach (var (key, value) in trial.UserAttrs)
        {
            if (Unchanged(mirror.UserAttrs, key, value) || !TryToPy(value, out PyObject? py))
                continue;
            using (py)
            using (var k = key.ToPython())
                Python.optunasharp_shadow.set_user_attr(_study, mirror.Cached, mirror.TrialId, k, py!);
            mirror.UserAttrs[key] = value;
        }
        foreach (var (key, value) in trial.SystemAttrs)
        {
            if (Unchanged(mirror.SystemAttrs, key, value) || !TryToPy(value, out PyObject? py))
                continue;
            using (py)
            using (var k = key.ToPython())
                Python.optunasharp_shadow.set_system_attr(_study, mirror.TrialId, k, py!);
            mirror.SystemAttrs[key] = value;
        }

        // A trial finished without the bridge's after_trial (failed as stale, told through another sampler) finishes here.
        if (trial.State.IsFinished())
            Finish(mirror, trial.State, trial.Values);
    }

    /// <summary>
    ///     Writes back the keys one of <c>attrs()</c>'s dictionaries gained or replaced during a Python call, and records
    ///     them as synchronized so the next <see cref="Sync" /> does not echo them.
    /// </summary>
    /// <param name="before">The <c>attrs()</c> tuple before the call.</param>
    /// <param name="after">The <c>attrs()</c> tuple after the call.</param>
    /// <param name="index">Which dictionary: 0 study user, 1 study system, 2 trial user, 3 trial system.</param>
    /// <param name="write">Writes one key to the OptunaSharp storage.</param>
    /// <param name="synced">The values last synchronized for this dictionary.</param>
    /// <exception cref="PythonException">Reading the dictionaries fails.</exception>
    private static void WriteBack(
        PyObject before, PyObject after, int index, Action<string, object?> write, Dictionary<string, object?> synced)
    {
        using PyObject b = before[index];
        using PyObject a = after[index];
        using PyObject changed = Python.optunasharp_shadow.changed(b, a);
        using PyObject items = changed.InvokeMethod("items");
        using PyIter iter = PyIter.GetIter(items);
        while (iter.MoveNext())
        {
            using PyObject kv = iter.Current;
            using PyObject k = kv[0];
            using PyObject v = kv[1];
            // A value without a JSON shape stays in the shadow, where the Python object keeps reading it.
            if (!TryToClr(v, out object? value))
                continue;
            string key = k.As<string>();
            write(key, value);
            synced[key] = value;
        }
    }

    // ---- values ------------------------------------------------------------------------------------------------------

    /// <summary>Objective values as a Python list, or <c>None</c>, as <c>after_trial</c> and the storages take them.</summary>
    /// <param name="values">The values, or null.</param>
    /// <returns>A new owned reference; dispose it under the GIL.</returns>
    internal static PyObject ToPyValues(IReadOnlyList<double>? values)
    {
        if (values is null)
            return ((object?)null).ToPython();
        var list = new PyList();
        foreach (double v in values)
        {
            using var item = v.ToPython();
            list.Append(item);
        }
        return list;
    }

    /// <summary>Converts an attribute dictionary for <c>create_trial</c>, recording the converted keys as synchronized.</summary>
    /// <param name="attrs">The OptunaSharp attributes.</param>
    /// <param name="synced">Receives every converted key's value.</param>
    /// <returns>A new owned Python dict; keys whose values have no JSON shape are left out.</returns>
    private static PyDict ToPyAttrs(IReadOnlyDictionary<string, object?> attrs, Dictionary<string, object?> synced)
    {
        var dict = new PyDict();
        foreach (var (key, value) in attrs)
        {
            if (!TryToPy(value, out PyObject? py))
                continue;
            using (py)
                dict[key] = py!;
            synced[key] = value;
        }
        return dict;
    }

    /// <summary>Whether <paramref name="key" />'s value is the one last synchronized (see <see cref="SameValue" />).</summary>
    /// <param name="synced">The values last synchronized.</param>
    /// <param name="key">The attribute key.</param>
    /// <param name="value">The value OptunaSharp holds now.</param>
    /// <returns><c>true</c> to skip the key.</returns>
    private static bool Unchanged(Dictionary<string, object?> synced, string key, object? value)
        => synced.TryGetValue(key, out object? last) && SameValue(last, value);

    /// <summary>
    ///     A JSON-shaped OptunaSharp value as a Python object: <c>None</c>, bool, int, float, str, list (any non-string
    ///     sequence) or dict (string keys), recursively.
    /// </summary>
    /// <param name="value">The value.</param>
    /// <param name="result">A new owned reference when convertible.</param>
    /// <returns>Whether the value has a JSON shape.</returns>
    internal static bool TryToPy(object? value, out PyObject? result)
    {
        switch (value)
        {
            case null or bool or string or double or float or sbyte or byte or short or ushort or int or uint or long
                or CategoricalChoice:
                result = PyConvert.ToPy(value);
                return true;
            case IEnumerable<KeyValuePair<string, object?>> map:
            {
                var dict = new PyDict();
                foreach (var (k, v) in map)
                {
                    if (!TryToPy(v, out PyObject? item))
                    {
                        dict.Dispose();
                        result = null;
                        return false;
                    }
                    using (item)
                        dict[k] = item!;
                }
                result = dict;
                return true;
            }
            case IEnumerable sequence:
            {
                var list = new PyList();
                foreach (object? element in sequence)
                {
                    if (!TryToPy(element, out PyObject? item))
                    {
                        list.Dispose();
                        result = null;
                        return false;
                    }
                    using (item)
                        list.Append(item!);
                }
                result = list;
                return true;
            }
            default:
                result = null;
                return false;
        }
    }

    /// <summary>
    ///     A JSON-shaped Python value as an OptunaSharp one: <c>None</c>, bool, int (within <see cref="long" />), float
    ///     and str (NumPy scalars included), list or tuple as <see cref="List{T}" />, dict with str keys as
    ///     <see cref="OrderedDict{TKey, TValue}" />, recursively.
    /// </summary>
    /// <param name="value">The Python value, borrowed.</param>
    /// <param name="result">The converted value.</param>
    /// <returns>Whether the value has a JSON shape.</returns>
    internal static bool TryToClr(PyObject value, out object? result)
    {
        result = null;
        if (value.IsNone())
            return true;
        string kind;
        using (PyType type = value.GetPythonType())
        using (PyObject name = type.GetAttr("__name__"))
            kind = name.As<string>() ?? "";
        switch (kind)
        {
            case "dict":
            {
                var dict = new OrderedDict<string, object?>();
                using PyObject items = value.InvokeMethod("items");
                using PyIter iter = PyIter.GetIter(items);
                while (iter.MoveNext())
                {
                    using PyObject kv = iter.Current;
                    using PyObject k = kv[0];
                    using PyObject v = kv[1];
                    if (!PyString.IsStringType(k) || !TryToClr(v, out object? item))
                        return false;
                    dict[k.As<string>()] = item;
                }
                result = dict;
                return true;
            }
            case "list" or "tuple":
            {
                var list = new List<object?>();
                using PyIter iter = PyIter.GetIter(value);
                while (iter.MoveNext())
                {
                    using PyObject element = iter.Current;
                    if (!TryToClr(element, out object? item))
                        return false;
                    list.Add(item);
                }
                result = list;
                return true;
            }
            case "bool" or "bool_" or "int" or "int64" or "int32" or "int16" or "int8" or "uint64" or "uint32" or "uint16"
                or "uint8" or "float" or "float64" or "float32" or "float16" or "str" or "str_":
                try
                {
                    result = PyConvert.ToClr(value);
                    return true;
                }
                catch (Exception)
                {
                    return false;  // an int beyond 64 bits has no OptunaSharp value
                }
            default:
                return false;
        }
    }

    /// <summary>
    ///     Value equality across the shapes the storages hand back: numbers by value (an int and a float that are equal
    ///     match, as in Python; two NaNs match, being the same stored value), strings and bools exactly, sequences and
    ///     string-keyed dictionaries element by element, anything else by <see cref="object.Equals(object, object)" />.
    /// </summary>
    /// <param name="a">One value.</param>
    /// <param name="b">The other.</param>
    /// <returns>Whether a storage round trip could turn one into the other.</returns>
    internal static bool SameValue(object? a, object? b)
    {
        if (ReferenceEquals(a, b))
            return true;
        if (a is null || b is null)
            return false;
        if (a is CategoricalChoice ca)
            return SameValue(ca.Value, b);
        if (b is CategoricalChoice cb)
            return SameValue(a, cb.Value);
        if (a is string sa)
            return b is string sb && string.Equals(sa, sb, StringComparison.Ordinal);
        if (a is bool ba)
            return b is bool bb && ba == bb;
        if (IsNumber(a) && IsNumber(b))
        {
            if (IsIntegral(a) && IsIntegral(b))
                return Convert.ToDecimal(a) == Convert.ToDecimal(b);
            double da = Convert.ToDouble(a), db = Convert.ToDouble(b);
            return da.Equals(db);  // Equals: NaN matches NaN
        }
        if (a is IEnumerable<KeyValuePair<string, object?>> ma && b is IEnumerable<KeyValuePair<string, object?>> mb)
        {
            var left = new Dictionary<string, object?>();
            foreach (var (k, v) in ma)
                left[k] = v;
            int count = 0;
            foreach (var (k, v) in mb)
            {
                if (!left.TryGetValue(k, out object? lv) || !SameValue(lv, v))
                    return false;
                count++;
            }
            return count == left.Count;
        }
        if (a is IEnumerable ea && b is IEnumerable eb && a is not string && b is not string)
        {
            IEnumerator ia = ea.GetEnumerator(), ib = eb.GetEnumerator();
            while (true)
            {
                bool hasA = ia.MoveNext(), hasB = ib.MoveNext();
                if (hasA != hasB)
                    return false;
                if (!hasA)
                    return true;
                if (!SameValue(ia.Current, ib.Current))
                    return false;
            }
        }
        return Equals(a, b);
    }

    /// <summary>Whether a value is a CLR number (not a bool).</summary>
    /// <param name="value">The value.</param>
    /// <returns><c>true</c> for the integral and floating-point types.</returns>
    private static bool IsNumber(object value)
        => value is sbyte or byte or short or ushort or int or uint or long or ulong or float or double or decimal;

    /// <summary>Whether a value is a CLR integer.</summary>
    /// <param name="value">The value.</param>
    /// <returns><c>true</c> for the integral types.</returns>
    private static bool IsIntegral(object value)
        => value is sbyte or byte or short or ushort or int or uint or long or ulong;

    /// <summary>One OptunaSharp trial's shadow trial and what has been mirrored of it.</summary>
    internal sealed class Mirror
    {
        /// <summary>The shadow storage's id of the trial (<c>trial._trial_id</c>).</summary>
        internal int TrialId;

        /// <summary>The state last mirrored; a finished mirror never changes again.</summary>
        internal TrialState State;

        /// <summary>The running trial's cached record (<see cref="CachedTrial" />), owned; null until first needed.</summary>
        internal PyObject? Cached;

        /// <summary>The parameter names mirrored (a parameter is written once).</summary>
        internal readonly HashSet<string> Params = new();

        /// <summary>The intermediate steps mirrored (a step is reported once).</summary>
        internal readonly HashSet<long> Steps = new();

        /// <summary>The trial user attributes as last synchronized, in OptunaSharp values, by key.</summary>
        internal readonly Dictionary<string, object?> UserAttrs = new();

        /// <summary>The trial system attributes as last synchronized, in OptunaSharp values, by key.</summary>
        internal readonly Dictionary<string, object?> SystemAttrs = new();
    }
}
