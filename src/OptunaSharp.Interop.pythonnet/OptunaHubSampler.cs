using System;
using System.Collections.Generic;
using Python.Runtime;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     Wraps a live OptunaHub / Optuna sampler (an <c>optuna.samplers.BaseSampler</c> or
///     <c>optunahub.samplers.SimpleBaseSampler</c> instance) as an OptunaSharp
///     <see cref="BaseSampler" />, so an OptunaSharp <see cref="Study" /> can be driven by any of the
///     ~85 samplers on https://hub.optuna.org/ — without porting them.
///
///     <para><b>How it works (the shadow study).</b> A Python sampler reads a real
///     <c>optuna.Study</c> + <c>FrozenTrial</c>. On each ask, this bridge rebuilds an in-memory Python
///     study mirroring the OptunaSharp study's finished trials (via <c>optuna.trial.create_trial</c> +
///     <c>study.add_trial</c>), opens a running trial on it, and delegates
///     <c>infer_relative_search_space</c> / <c>sample_relative</c> / <c>sample_independent</c> to the
///     Python sampler against that mirror. The suggestions it returns are the real Optuna sampler's
///     output, marshaled back to CLR values.</para>
///
///     <para><b>Scope of the first cut.</b> Designed for single-worker <c>Study.Optimize</c>
///     (<c>nJobs=1</c>): the mirror is rebuilt per trial from finished trials, so trial numbers align.
///     Parallel trials (concurrent RUNNING trials) are not mirrored. Reproducibility follows the Python
///     sampler's own RNG (seed it via the constructor kwargs).</para>
/// </summary>
public sealed class OptunaHubSampler : BaseSampler, IDisposable
{
    private readonly string _name;
    private PyObject? _sampler;

    // Per-trial shadow cache (guarded by _gate). Rebuilt when the C# trial number changes.
    private readonly object _gate = new();
    private int _cachedNumber = -1;
    private PyObject? _cachedStudy;
    private PyObject? _cachedFrozen;

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

    private PyObject Sampler => _sampler ?? throw new ObjectDisposedException(nameof(OptunaHubSampler));

    /// <inheritdoc />
    /// <exception cref="System.ObjectDisposedException">The algorithm wrapper has already been disposed.</exception>
    /// <exception cref="global::Python.Runtime.PythonException">The Python algorithm or shadow-study operation fails.</exception>
    /// <exception cref="OptunaSharp.Interop.PythonNet.PythonInteropException">The interpreter or a required Python module cannot be initialized.</exception>
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
        => PythonRuntime.Run(() =>
        {
            var (pystudy, pyfrozen) = EnsureShadow(study, trial);
            if (!Sampler.HasAttr("infer_relative_search_space"))
                return new OrderedDict<string, BaseDistribution>();

            using PyObject result = Python.optuna.samplers.BaseSampler.infer_relative_search_space(Sampler, pystudy, pyfrozen);
            return ToClrDistributionDict(result);
        });

    /// <inheritdoc />
    /// <exception cref="System.ObjectDisposedException">The algorithm wrapper has already been disposed.</exception>
    /// <exception cref="global::Python.Runtime.PythonException">The Python algorithm or shadow-study operation fails.</exception>
    /// <exception cref="OptunaSharp.Interop.PythonNet.PythonInteropException">The interpreter or a required Python module cannot be initialized.</exception>
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
        => PythonRuntime.Run(() =>
        {
            var (pystudy, pyfrozen) = EnsureShadow(study, trial);
            using var pySearchSpace = new PyDict();
            foreach (var (name, dist) in searchSpace)
                pySearchSpace[name] = PyConvert.ToPy(dist);

            using PyObject result = Python.optuna.samplers.BaseSampler.sample_relative(Sampler, pystudy, pyfrozen, pySearchSpace);
            return ToClrValueDict(result);
        });

    /// <inheritdoc />
    /// <exception cref="System.ObjectDisposedException">The algorithm wrapper has already been disposed.</exception>
    /// <exception cref="global::Python.Runtime.PythonException">The Python algorithm or shadow-study operation fails.</exception>
    /// <exception cref="OptunaSharp.Interop.PythonNet.PythonInteropException">The interpreter or a required Python module cannot be initialized.</exception>
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
        => PythonRuntime.Run(() =>
        {
            var (pystudy, pyfrozen) = EnsureShadow(study, trial);
            using var name = paramName.ToPython();
            using var dist = PyConvert.ToPy(paramDistribution);
            using PyObject result = Python.optuna.samplers.BaseSampler.sample_independent(Sampler, pystudy, pyfrozen, name, dist);
            return PyConvert.ToClr(result);
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

    /// <summary>Ensure the Python shadow study + running frozen trial mirror the C# state for this trial. Under the GIL.</summary>
    /// <param name="study">The study whose trial history is mirrored, or its borrowed Python shadow-study handle.</param>
    /// <param name="trial">The trial being mirrored, sampled, or checked for optimizer compatibility.</param>
    /// <returns>Borrowed cached study and frozen-trial handles; the cache retains ownership and callers must not dispose them.</returns>
    private (PyObject study, PyObject frozen) EnsureShadow(Study study, FrozenTrial trial)
    {
        lock (_gate)
        {
            if (_cachedNumber == trial.Number && _cachedStudy is not null && _cachedFrozen is not null)
                return (_cachedStudy, _cachedFrozen);

            DisposeShadowNoLock();

            var pystudy = BuildStudy(study, trial.Number);
            var pyfrozen = OpenRunningTrial(pystudy, trial.Number);

            _cachedStudy = pystudy;
            _cachedFrozen = pyfrozen;
            _cachedNumber = trial.Number;
            return (pystudy, pyfrozen);
        }
    }

    /// <summary>Builds an in-memory Python study containing earlier finished trials for sampler or pruner replay.</summary>
    /// <param name="study">The study whose trial history is mirrored, or its borrowed Python shadow-study handle.</param>
    /// <param name="currentNumber">The current trial number; only earlier finished trials are included in the shadow study.</param>
    /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
    private static PyObject BuildStudy(Study study, int currentNumber)
    {
        OptunaHubRuntime.EnsureQuiet(); // re-assert right before create_study (Optuna re-configures logging lazily)
        PyObject pystudy = Python.optuna.create_study(study.Directions);

        foreach (var ft in study.GetTrials(deepcopy: false, states: null))
        {
            if (ft.Number >= currentNumber || !ft.State.IsFinished())
                continue;
            using var pyTrial = Python.optuna.trial.create_trial(ft.State, ft.Values, ft.Params, ft.Distributions);
            Python.optuna.Study.add_trial(pystudy, pyTrial);
        }

        return pystudy;
    }

    /// <summary>Adds a running trial to the Python shadow study and returns its frozen snapshot.</summary>
    /// <param name="pystudy">The borrowed Python shadow study that receives a new running trial.</param>
    /// <param name="number">The managed trial number associated with the shadow; the returned Python ask number controls indexing.</param>
    /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
    private static PyObject OpenRunningTrial(PyObject pystudy, int number)
    {
        // ask() adds a RUNNING trial (no params sampled without fixed distributions); fetch its frozen form.
        using PyObject running = Python.optuna.Study.ask(pystudy);
        int askedNumber = Python.optuna.Trial.number_t(running);
        using PyObject trials = Python.optuna.Study.get_trials(pystudy, deepcopy: false);
        // Index by the asked number so it lines up even if numbering diverged.
        return trials[new PyInt(askedNumber)];
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

    /// <summary>Releases the cached Python study and frozen-trial references while the caller holds the GIL and cache lock.</summary>
    private void DisposeShadowNoLock()
    {
        _cachedFrozen?.Dispose();
        _cachedStudy?.Dispose();
        _cachedFrozen = null;
        _cachedStudy = null;
        _cachedNumber = -1;
    }

    /// <summary>Releases owned Python references while holding the GIL.</summary>
    public void Dispose()
    {
        var sampler = _sampler;
        _sampler = null;
        PythonRuntime.Run(() =>
        {
            lock (_gate)
                DisposeShadowNoLock();
            sampler?.Dispose();
        });
    }
}
