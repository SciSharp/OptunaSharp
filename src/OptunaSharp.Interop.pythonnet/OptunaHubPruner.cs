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

    /// <inheritdoc />
    public override string ToString() => _name;

    private PyObject Pruner => _pruner ?? throw new ObjectDisposedException(nameof(OptunaHubPruner));

    /// <inheritdoc />
    public override bool Prune(Study study, FrozenTrial trial) => PythonRuntime.Run(() =>
    {
        using PyObject pystudy = BuildStudy(study, trial.Number);
        using PyObject frozen = OpenRunningTrial(pystudy, trial);
        using PyObject result = Python.optuna.pruners.BasePruner.prune(Pruner, pystudy, frozen);
        // IsTrue() (Python truthiness) not As<bool>(): a pruner may return a numpy bool_ from its
        // comparison, which As<bool>() rejects with "cannot convert object to target type".
        return result.IsTrue();
    });

    // ---- shadow study (finished trials + their intermediate values) ------------------------------

    private static PyObject BuildStudy(Study study, int currentNumber)
    {
        OptunaHubRuntime.EnsureQuiet();
        // Optuna pruning is single-objective — mirror the shadow study on the first direction.
        StudyDirection dir = study.Directions.Count > 0 ? study.Directions[0] : StudyDirection.Minimize;
        PyObject pystudy = Python.optuna.create_study(new[] { dir });

        foreach (var ft in study.GetTrials(deepcopy: false, states: null))
        {
            if (ft.Number >= currentNumber || !ft.State.IsFinished())
                continue;
            using var pyTrial = Python.optuna.trial.create_trial(
                ft.State, ft.Values, ft.Params, ft.Distributions, ft.IntermediateValues);
            Python.optuna.Study.add_trial(pystudy, pyTrial);
        }
        return pystudy;
    }

    // Open a running trial and replay this trial's reported intermediate values onto it — that is what
    // the pruner reads (trial.intermediate_values / last_step) to decide.
    private static PyObject OpenRunningTrial(PyObject pystudy, FrozenTrial trial)
    {
        using PyObject running = Python.optuna.Study.ask(pystudy);
        foreach (var (step, value) in trial.IntermediateValues)
            Python.optuna.Trial.report(running, value, step);
        int number = Python.optuna.Trial.number_t(running);
        using PyObject trials = Python.optuna.Study.get_trials(pystudy, deepcopy: false);
        return trials[new PyInt(number)];
    }

    public void Dispose()
    {
        var pruner = _pruner;
        _pruner = null;
        PythonRuntime.Run(() => pruner?.Dispose());
    }
}
