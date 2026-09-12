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

    internal OptunaHubSampler(string name, PyObject sampler)
    {
        _name = name;
        _sampler = sampler;
    }

    /// <inheritdoc />
    public override string ToString() => _name;

    private PyObject Sampler => _sampler ?? throw new ObjectDisposedException(nameof(OptunaHubSampler));

    /// <inheritdoc />
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

    private void DisposeShadowNoLock()
    {
        _cachedFrozen?.Dispose();
        _cachedStudy?.Dispose();
        _cachedFrozen = null;
        _cachedStudy = null;
        _cachedNumber = -1;
    }

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
