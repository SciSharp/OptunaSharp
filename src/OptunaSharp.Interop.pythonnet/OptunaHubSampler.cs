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

    public override string ToString() => _name;

    private PyObject Sampler => _sampler ?? throw new ObjectDisposedException(nameof(OptunaHubSampler));

    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
        => PythonRuntime.Run(() =>
        {
            var (pystudy, pyfrozen) = EnsureShadow(study, trial);
            if (!Sampler.HasAttr("infer_relative_search_space"))
                return new OrderedDict<string, BaseDistribution>();

            using var infer = Sampler.GetAttr("infer_relative_search_space");
            using PyObject result = infer.Invoke(pystudy, pyfrozen);
            return ToClrDistributionDict(result);
        });

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
        => PythonRuntime.Run(() =>
        {
            var (pystudy, pyfrozen) = EnsureShadow(study, trial);
            using var pySearchSpace = new PyDict();
            foreach (var (name, dist) in searchSpace)
                pySearchSpace[name] = PyConvert.ToPy(dist);

            using var sampleRelative = Sampler.GetAttr("sample_relative");
            using PyObject result = sampleRelative.Invoke(new[] { pystudy, pyfrozen, (PyObject)pySearchSpace });
            return ToClrValueDict(result);
        });

    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
        => PythonRuntime.Run(() =>
        {
            var (pystudy, pyfrozen) = EnsureShadow(study, trial);
            using var sampleIndependent = Sampler.GetAttr("sample_independent");
            using var name = paramName.ToPython();
            using var dist = PyConvert.ToPy(paramDistribution);
            using PyObject result = sampleIndependent.Invoke(new[] { pystudy, pyfrozen, name, dist });
            return PyConvert.ToClr(result);
        });

    public override void ReseedRng()
    {
        if (_sampler is null)
            return;
        PythonRuntime.Run(() =>
        {
            if (Sampler.HasAttr("reseed_rng"))
                using (var reseed = Sampler.GetAttr("reseed_rng"))
                    reseed.Invoke();
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
        PythonRuntime.EnsureQuiet(); // re-assert right before create_study (Optuna re-configures logging lazily)
        using var createStudy = PythonRuntime.Optuna.GetAttr("create_study");
        using var kwargs = new PyDict();
        var dirItems = new List<PyObject>();
        foreach (var d in study.Directions)
            dirItems.Add((d == StudyDirection.Maximize ? "maximize" : "minimize").ToPython());
        using (var directions = new PyList(dirItems.ToArray()))
            kwargs["directions"] = directions;
        foreach (var it in dirItems) it.Dispose();

        var pystudy = createStudy.Invoke(Array.Empty<PyObject>(), kwargs);

        using var addTrial = pystudy.GetAttr("add_trial");
        foreach (var ft in study.GetTrials(deepcopy: false, states: null))
        {
            if (ft.Number >= currentNumber || !ft.State.IsFinished())
                continue;
            using var pyTrial = CreateFrozenTrial(ft);
            addTrial.Invoke(pyTrial);
        }

        return pystudy;
    }

    private static PyObject OpenRunningTrial(PyObject pystudy, int number)
    {
        // ask() adds a RUNNING trial (no params sampled without fixed distributions); fetch its frozen form.
        using var ask = pystudy.GetAttr("ask");
        using PyObject running = ask.Invoke();
        int askedNumber = running.GetAttr("number").As<int>();
        using var getTrials = pystudy.GetAttr("get_trials");
        using var kwargs = new PyDict();
        kwargs["deepcopy"] = false.ToPython();
        using PyObject trials = getTrials.Invoke(Array.Empty<PyObject>(), kwargs);
        // Index by the asked number so it lines up even if numbering diverged.
        return trials[new PyInt(askedNumber)];
    }

    private static PyObject CreateFrozenTrial(FrozenTrial ft)
    {
        using var trialModule = PythonRuntime.Optuna.GetAttr("trial");
        using var createTrial = trialModule.GetAttr("create_trial");
        using var kwargs = new PyDict();

        kwargs["state"] = PyTrialState(ft.State);

        if (ft.Values is { Count: > 0 } vals)
        {
            var items = new PyObject[vals.Count];
            for (int k = 0; k < vals.Count; k++)
                items[k] = vals[k].ToPython();
            using (var pyValues = new PyList(items))
                kwargs["values"] = pyValues;
            foreach (var it in items) it.Dispose();
        }

        using (var pyParams = new PyDict())
        {
            foreach (var (name, value) in ft.Params)
                pyParams[name] = PyConvert.ToPy(value);
            kwargs["params"] = pyParams;

            using var pyDists = new PyDict();
            foreach (var (name, dist) in ft.Distributions)
                pyDists[name] = PyConvert.ToPy(dist);
            kwargs["distributions"] = pyDists;

            return createTrial.Invoke(Array.Empty<PyObject>(), kwargs);
        }
    }

    private static PyObject PyTrialState(TrialState state)
    {
        string name = state switch
        {
            TrialState.Complete => "COMPLETE",
            TrialState.Pruned => "PRUNED",
            TrialState.Fail => "FAIL",
            TrialState.Waiting => "WAITING",
            _ => "RUNNING",
        };
        using var trialModule = PythonRuntime.Optuna.GetAttr("trial");
        using var stateEnum = trialModule.GetAttr("TrialState");
        return stateEnum.GetAttr(name);
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
