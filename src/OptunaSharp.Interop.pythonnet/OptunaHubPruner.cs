using System;
using Python.Runtime;
using OptunaSharp;
using OptunaSharp.Pruners;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     Wraps a live OptunaHub / Optuna pruner (an <c>optuna.pruners.BasePruner</c> instance) as an
///     OptunaSharp <see cref="BasePruner" />, so an OptunaSharp study's <c>report</c> /
///     <c>should_prune</c> loop can be driven by any pruner on https://hub.optuna.org/.
///
///     <para>Like the sampler bridge, each <see cref="Prune" /> rebuilds a Python <b>shadow study</b>
///     mirroring the OptunaSharp study's finished trials (with their intermediate values) and opens a
///     running trial carrying the current trial's reported intermediate values, then delegates
///     <c>prune(study, trial)</c> to the real Python pruner. Optuna pruning is single-objective; this
///     targets single-worker <c>Study.Optimize</c> (<c>nJobs=1</c>).</para>
/// </summary>
public sealed class OptunaHubPruner : BasePruner, IDisposable
{
    private readonly string _name;
    private PyObject? _pruner;

    internal OptunaHubPruner(string name, PyObject pruner)
    {
        _name = name;
        _pruner = pruner;
    }

    public override string ToString() => _name;

    private PyObject Pruner => _pruner ?? throw new ObjectDisposedException(nameof(OptunaHubPruner));

    public override bool Prune(Study study, FrozenTrial trial) => PythonRuntime.Run(() =>
    {
        using PyObject pystudy = BuildStudy(study, trial.Number);
        using PyObject frozen = OpenRunningTrial(pystudy, trial);
        using var prune = Pruner.GetAttr("prune");
        using PyObject result = prune.Invoke(pystudy, frozen);
        // IsTrue() (Python truthiness) not As<bool>(): a pruner may return a numpy bool_ from its
        // comparison, which As<bool>() rejects with "cannot convert object to target type".
        return result.IsTrue();
    });

    // ---- shadow study (finished trials + their intermediate values) ------------------------------

    private static PyObject BuildStudy(Study study, int currentNumber)
    {
        PythonRuntime.EnsureQuiet();
        using var createStudy = PythonRuntime.Optuna.GetAttr("create_study");
        using var kwargs = new PyDict();
        string dir = study.Directions.Count > 0 && study.Directions[0] == StudyDirection.Maximize
            ? "maximize" : "minimize";
        using (var pyDir = dir.ToPython())
        using (var directions = new PyList(new[] { pyDir }))
            kwargs["directions"] = directions;

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

    // Open a running trial and replay this trial's reported intermediate values onto it — that is what
    // the pruner reads (trial.intermediate_values / last_step) to decide.
    private static PyObject OpenRunningTrial(PyObject pystudy, FrozenTrial trial)
    {
        using var ask = pystudy.GetAttr("ask");
        using PyObject running = ask.Invoke();
        using var report = running.GetAttr("report");
        foreach (var (step, value) in trial.IntermediateValues)
        {
            using var v = value.ToPython();
            using var s = step.ToPython();
            report.Invoke(v, s);
        }
        int number = running.GetAttr("number").As<int>();
        using var getTrials = pystudy.GetAttr("get_trials");
        using var kw = new PyDict();
        kw["deepcopy"] = false.ToPython();
        using PyObject trials = getTrials.Invoke(Array.Empty<PyObject>(), kw);
        return trials[new PyInt(number)];
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
        using (var pyDists = new PyDict())
        using (var pyInter = new PyDict())
        {
            foreach (var (name, value) in ft.Params)
                pyParams[name] = PyConvert.ToPy(value);
            kwargs["params"] = pyParams;

            foreach (var (name, dist) in ft.Distributions)
                pyDists[name] = PyConvert.ToPy(dist);
            kwargs["distributions"] = pyDists;

            foreach (var (step, value) in ft.IntermediateValues)
                pyInter.SetItem(step, value.ToPython());   // int step key -> float value
            kwargs["intermediate_values"] = pyInter;

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

    public void Dispose()
    {
        var pruner = _pruner;
        _pruner = null;
        PythonRuntime.Run(() => pruner?.Dispose());
    }
}
