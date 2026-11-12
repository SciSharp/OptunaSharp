using System;
using System.Collections.Generic;
using Python.Runtime;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     Wraps a live OptunaHub / Optuna sampler (an <c>optuna.samplers.BaseSampler</c> or
///     <c>optunahub.samplers.SimpleBaseSampler</c> instance) as an OptunaSharp
///     <see cref="BaseSampler" />, so an OptunaSharp <see cref="Study" /> can be driven by any of the
///     ~85 samplers on https://hub.optuna.org/ — without porting them.
///
///     <para><b>How it works (the shadow study).</b> A Python sampler reads a real <c>optuna.Study</c> and
///     <c>FrozenTrial</c>s. For each OptunaSharp study it meets, this bridge keeps one in-memory Python study in
///     lockstep with it (<see cref="OptunaHubShadow" />): every trial with its state, values, parameters, user and
///     system attributes and intermediate values, and the study's attributes. It forwards all five sampler hooks with
///     the trial objects native Optuna passes: <c>before_trial</c> and <c>infer_relative_search_space</c> /
///     <c>sample_relative</c> with the trial's cached record, <c>sample_independent</c> with the latest-trial copy,
///     and <c>after_trial</c> with the stored record, before the trial is finished. The attributes the Python sampler
///     writes (a generation counter, a swarm, constraints) are written to the OptunaSharp trial or study as well, so
///     the C# storage holds what a native run stores. The suggestions are the Python sampler's own output, marshaled
///     back to CLR values.</para>
///
///     <para><b>Limits.</b> The Python sampler's internal state lives in its Python object, not in OptunaSharp's
///     storage, so a restarted process does not resume it. Attribute values with no JSON shape (NumPy arrays,
///     arbitrary objects) stay in the shadow only. A study filtered for <c>HyperbandPruner</c>'s brackets is mirrored
///     whole. Calls are serialized on the Python thread; trials of parallel workers are mirrored with the state the
///     storage holds when a call starts, as a native sampler reads them. Reproducibility follows the Python sampler's
///     own RNG (seed it through the constructor kwargs).</para>
/// </summary>
public sealed class OptunaHubSampler : BaseSampler, IDisposable
{
    private readonly string _name;
    private PyObject? _sampler;

    // The shadow of each study this sampler drives, by (storage, study id). Guarded by _gate; every use runs on the
    // Python thread under the GIL.
    private readonly object _gate = new();
    private readonly Dictionary<StudyKey, OptunaHubShadow> _shadows = new();

    /// <summary>Takes ownership of the Python algorithm instance and records its diagnostic name.</summary>
    /// <param name="name">The package and class name displayed in diagnostics.</param>
    /// <param name="sampler">The Python sampler handle owned by the wrapper, or borrowed by a low-level invocation.</param>
    internal OptunaHubSampler(string name, PyObject sampler)
    {
        _name = name;
        _sampler = sampler;
    }

    /// <inheritdoc />
    public override string ToString() => _name;

    /// <summary>The Python sampler, borrowed.</summary>
    /// <exception cref="ObjectDisposedException">The wrapper has been disposed.</exception>
    private PyObject Sampler => _sampler ?? throw new ObjectDisposedException(nameof(OptunaHubSampler));

    /// <inheritdoc />
    /// <remarks>
    ///     Optuna's <c>Study.ask</c> and <c>Trial.__init__</c>: the study's per-thread trial cache is cleared, the trial's
    ///     cached record is taken, then <c>before_trial</c> runs with it.
    /// </remarks>
    /// <exception cref="System.ObjectDisposedException">The algorithm wrapper has already been disposed.</exception>
    /// <exception cref="global::Python.Runtime.PythonException">The Python algorithm or shadow-study operation fails.</exception>
    /// <exception cref="OptunaSharp.Interop.PythonNet.PythonInteropException">The interpreter or a required Python module cannot be initialized.</exception>
    public override void BeforeTrial(Study study, FrozenTrial trial)
        => PythonRuntime.Run(() =>
        {
            lock (_gate)
            {
                OptunaHubShadow shadow = SyncedShadow(study);
                OptunaHubShadow.Mirror mirror = shadow.MirrorOf(trial);
                shadow.BeginTrial();
                PyObject cached = shadow.CachedTrial(mirror, refresh: true);
                if (!Sampler.HasAttr("before_trial"))
                    return;
                shadow.Call(study, trial, mirror, () =>
                {
                    Python.optuna.samplers.BaseSampler.before_trial(Sampler, shadow.Study, cached);
                    return 0;
                });
            }
        });

    /// <inheritdoc />
    /// <exception cref="System.ObjectDisposedException">The algorithm wrapper has already been disposed.</exception>
    /// <exception cref="global::Python.Runtime.PythonException">The Python algorithm or shadow-study operation fails.</exception>
    /// <exception cref="OptunaSharp.Interop.PythonNet.PythonInteropException">The interpreter or a required Python module cannot be initialized.</exception>
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
        => PythonRuntime.Run(() =>
        {
            lock (_gate)
            {
                OptunaHubShadow shadow = SyncedShadow(study);
                OptunaHubShadow.Mirror mirror = shadow.MirrorOf(trial);
                PyObject cached = shadow.CachedTrial(mirror);
                if (!Sampler.HasAttr("infer_relative_search_space"))
                    return new OrderedDict<string, BaseDistribution>();

                return shadow.Call(study, trial, mirror, () =>
                {
                    using PyObject result =
                        Python.optuna.samplers.BaseSampler.infer_relative_search_space(Sampler, shadow.Study, cached);
                    return ToClrDistributionDict(result);
                });
            }
        });

    /// <inheritdoc />
    /// <exception cref="System.ObjectDisposedException">The algorithm wrapper has already been disposed.</exception>
    /// <exception cref="global::Python.Runtime.PythonException">The Python algorithm or shadow-study operation fails.</exception>
    /// <exception cref="OptunaSharp.Interop.PythonNet.PythonInteropException">The interpreter or a required Python module cannot be initialized.</exception>
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
        => PythonRuntime.Run(() =>
        {
            lock (_gate)
            {
                OptunaHubShadow shadow = SyncedShadow(study);
                OptunaHubShadow.Mirror mirror = shadow.MirrorOf(trial);
                PyObject cached = shadow.CachedTrial(mirror);
                using var pySearchSpace = new PyDict();
                foreach (var (name, dist) in searchSpace)
                {
                    using PyObject pyDist = PyConvert.ToPy(dist);
                    pySearchSpace[name] = pyDist;
                }

                return shadow.Call(study, trial, mirror, () =>
                {
                    using PyObject result =
                        Python.optuna.samplers.BaseSampler.sample_relative(Sampler, shadow.Study, cached, pySearchSpace);
                    return ToClrValueDict(result);
                });
            }
        });

    /// <inheritdoc />
    /// <remarks>
    ///     Optuna's <c>Trial._suggest</c> passes <c>_get_latest_trial()</c>: the cached record's shallow copy whose system
    ///     attributes are read from the storage.
    /// </remarks>
    /// <exception cref="System.ObjectDisposedException">The algorithm wrapper has already been disposed.</exception>
    /// <exception cref="global::Python.Runtime.PythonException">The Python algorithm or shadow-study operation fails.</exception>
    /// <exception cref="OptunaSharp.Interop.PythonNet.PythonInteropException">The interpreter or a required Python module cannot be initialized.</exception>
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
        => PythonRuntime.Run(() =>
        {
            lock (_gate)
            {
                OptunaHubShadow shadow = SyncedShadow(study);
                OptunaHubShadow.Mirror mirror = shadow.MirrorOf(trial);
                using PyObject latest = shadow.LatestTrial(mirror);
                using var name = paramName.ToPython();
                using PyObject dist = PyConvert.ToPy(paramDistribution);

                return shadow.Call(study, trial, mirror, () =>
                {
                    using PyObject result =
                        Python.optuna.samplers.BaseSampler.sample_independent(Sampler, shadow.Study, latest, name, dist);
                    return PyConvert.ToClr(result);
                });
            }
        });

    /// <inheritdoc />
    /// <remarks>
    ///     Optuna's <c>_tell_with_warning</c>: <c>after_trial</c> receives the stored record, and the trial's state and
    ///     values are written in a <c>finally</c>, so the shadow trial finishes even when the Python sampler raises.
    /// </remarks>
    /// <exception cref="System.ObjectDisposedException">The algorithm wrapper has already been disposed.</exception>
    /// <exception cref="global::Python.Runtime.PythonException">The Python algorithm or shadow-study operation fails.</exception>
    /// <exception cref="OptunaSharp.Interop.PythonNet.PythonInteropException">The interpreter or a required Python module cannot be initialized.</exception>
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
        => PythonRuntime.Run(() =>
        {
            lock (_gate)
            {
                OptunaHubShadow shadow = SyncedShadow(study);
                OptunaHubShadow.Mirror mirror = shadow.MirrorOf(trial);
                try
                {
                    if (!Sampler.HasAttr("after_trial"))
                        return;
                    using PyObject stored = shadow.StoredTrial(mirror);
                    using PyObject pyState = Python.optuna.trial.TrialState(state);
                    using PyObject pyValues = OptunaHubShadow.ToPyValues(values);
                    shadow.Call(study, trial, mirror, () =>
                    {
                        Python.optuna.samplers.BaseSampler.after_trial(Sampler, shadow.Study, stored, pyState, pyValues);
                        return 0;
                    });
                }
                finally
                {
                    shadow.Finish(mirror, state, values);
                }
            }
        });

    /// <inheritdoc />
    public override void ReseedRng()
    {
        if (_sampler is null)
            return;
        PythonRuntime.Run(() =>
        {
            if (Sampler.HasAttr("reseed_rng"))
                Python.optuna.samplers.BaseSampler.reseed_rng(Sampler);
        });
    }

    // ---- shadow study ---------------------------------------------------------------------------

    /// <summary>
    ///     The shadow of <paramref name="study" />, created on first use and synchronized with it now. Under the GIL and
    ///     <see cref="_gate" />.
    /// </summary>
    /// <param name="study">The OptunaSharp study being sampled.</param>
    /// <returns>The synchronized shadow, owned by this sampler.</returns>
    /// <exception cref="global::Python.Runtime.PythonException">Creating or synchronizing the shadow fails.</exception>
    /// <exception cref="OptunaHubInteropException">The study's trial numbers are not contiguous.</exception>
    private OptunaHubShadow SyncedShadow(Study study)
    {
        var key = new StudyKey(study.Storage, study.StudyId);
        if (!_shadows.TryGetValue(key, out OptunaHubShadow? shadow))
        {
            shadow = OptunaHubShadow.Create(study);
            _shadows[key] = shadow;
        }
        shadow.Sync(study);
        return shadow;
    }

    // ---- result conversion ----------------------------------------------------------------------

    /// <summary>Copies a Python parameter-distribution mapping into an ordered managed dictionary.</summary>
    /// <param name="dict">The borrowed Python mapping converted in its iteration order.</param>
    /// <returns>The copied managed distributions in Python mapping order.</returns>
    private static OrderedDict<string, BaseDistribution> ToClrDistributionDict(PyObject dict)
    {
        var result = new OrderedDict<string, BaseDistribution>();
        using var items = dict.GetAttr("items").Invoke();
        using var iter = PyIter.GetIter(items);
        while (iter.MoveNext())
        {
            using PyObject kv = iter.Current;
            using var key = kv[new PyInt(0)];
            using var val = kv[new PyInt(1)];
            result[key.As<string>()!] = PyConvert.ToClrDistribution(val);
        }
        return result;
    }

    /// <summary>Copies a Python parameter-value mapping into an ordered managed dictionary.</summary>
    /// <param name="dict">The borrowed Python mapping converted in its iteration order.</param>
    /// <returns>The copied managed parameter values in Python mapping order.</returns>
    private static OrderedDict<string, object?> ToClrValueDict(PyObject dict)
    {
        var result = new OrderedDict<string, object?>();
        using var items = dict.GetAttr("items").Invoke();
        using var iter = PyIter.GetIter(items);
        while (iter.MoveNext())
        {
            using PyObject kv = iter.Current;
            using var key = kv[new PyInt(0)];
            using var val = kv[new PyInt(1)];
            result[key.As<string>()!] = PyConvert.ToClr(val);
        }
        return result;
    }

    /// <summary>Releases owned Python references (the sampler and every shadow study) while holding the GIL.</summary>
    public void Dispose()
    {
        var sampler = _sampler;
        _sampler = null;
        PythonRuntime.Run(() =>
        {
            lock (_gate)
            {
                foreach (OptunaHubShadow shadow in _shadows.Values)
                    shadow.Dispose();
                _shadows.Clear();
            }
            sampler?.Dispose();
        });
    }

    /// <summary>A study's identity across <see cref="Study" /> objects: its storage instance and id.</summary>
    /// <param name="Storage">The storage instance, compared by reference.</param>
    /// <param name="StudyId">The study's id in that storage.</param>
    private readonly record struct StudyKey(BaseStorage Storage, int StudyId);
}
