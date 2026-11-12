using System;
using System.Collections.Generic;
using Python.Runtime;
using OptunaSharp;
using OptunaSharp.Pruners;
using OptunaSharp.Storages;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     Wraps a live OptunaHub / Optuna pruner (an <c>optuna.pruners.BasePruner</c> instance) as an
///     OptunaSharp <see cref="BasePruner" />, so an OptunaSharp study's <c>report</c> /
///     <c>should_prune</c> loop can be driven by any pruner on https://hub.optuna.org/.
///
///     <para>Like the sampler bridge, the pruner keeps one persistent Python <b>shadow study</b> per OptunaSharp study
///     (<see cref="OptunaHubShadow" />), mirrored trial for trial before every decision: states, values, parameters,
///     user and system attributes and every intermediate value, the current trial's included. <see cref="Prune" />
///     calls <c>prune(study, trial)</c> with the trial object native <c>Trial.should_prune</c> passes (the
///     latest-trial copy), and the attributes the pruner writes reach the OptunaSharp trial too: successive halving
///     and Hyperband record each rung a trial completes (<c>completed_rung_N</c>) and compare it with the other
///     trials' records, which a shadow rebuilt per call used to lose, so they never pruned (QA-73). Optuna pruning is
///     single-objective; calls are serialized on the Python thread.</para>
/// </summary>
public sealed class OptunaHubPruner : BasePruner, IDisposable
{
    private readonly string _name;
    private PyObject? _pruner;

    // The shadow of each study this pruner decides for, by (storage, study id). Guarded by _gate; every use runs on the
    // Python thread under the GIL.
    private readonly object _gate = new();
    private readonly Dictionary<StudyKey, OptunaHubShadow> _shadows = new();

    /// <summary>Takes ownership of the Python algorithm instance and records its diagnostic name.</summary>
    /// <param name="name">The package and class name displayed in diagnostics.</param>
    /// <param name="pruner">The Python pruner handle owned by the wrapper, or borrowed by a low-level invocation.</param>
    internal OptunaHubPruner(string name, PyObject pruner)
    {
        _name = name;
        _pruner = pruner;
    }

    /// <inheritdoc />
    public override string ToString() => _name;

    /// <summary>The Python pruner, borrowed.</summary>
    /// <exception cref="ObjectDisposedException">The wrapper has been disposed.</exception>
    private PyObject Pruner => _pruner ?? throw new ObjectDisposedException(nameof(OptunaHubPruner));

    /// <inheritdoc />
    /// <exception cref="System.ObjectDisposedException">The algorithm wrapper has already been disposed.</exception>
    /// <exception cref="global::Python.Runtime.PythonException">The Python algorithm or shadow-study operation fails.</exception>
    /// <exception cref="OptunaSharp.Interop.PythonNet.PythonInteropException">The interpreter or a required Python module cannot be initialized.</exception>
    public override bool Prune(Study study, FrozenTrial trial) => PythonRuntime.Run(() =>
    {
        lock (_gate)
        {
            var key = new StudyKey(study.Storage, study.StudyId);
            if (!_shadows.TryGetValue(key, out OptunaHubShadow? shadow))
            {
                shadow = OptunaHubShadow.Create(study);
                _shadows[key] = shadow;
            }
            shadow.Sync(study);
            OptunaHubShadow.Mirror mirror = shadow.MirrorOf(trial);

            // Trial.should_prune: `self.study.pruner.prune(self.study, self._get_latest_trial())`.
            using PyObject latest = shadow.LatestTrial(mirror);
            return shadow.Call(study, trial, mirror, () =>
            {
                using PyObject result = Python.optuna.pruners.BasePruner.prune(Pruner, shadow.Study, latest);
                // IsTrue() (Python truthiness) not As<bool>(): a pruner may return a numpy bool_ from its
                // comparison, which As<bool>() rejects with "cannot convert object to target type".
                return result.IsTrue();
            });
        }
    });

    /// <summary>Releases owned Python references (the pruner and every shadow study) while holding the GIL.</summary>
    public void Dispose()
    {
        var pruner = _pruner;
        _pruner = null;
        PythonRuntime.Run(() =>
        {
            lock (_gate)
            {
                foreach (OptunaHubShadow shadow in _shadows.Values)
                    shadow.Dispose();
                _shadows.Clear();
            }
            pruner?.Dispose();
        });
    }

    /// <summary>A study's identity across <see cref="Study" /> objects: its storage instance and id.</summary>
    /// <param name="Storage">The storage instance, compared by reference.</param>
    /// <param name="StudyId">The study's id in that storage.</param>
    private readonly record struct StudyKey(BaseStorage Storage, int StudyId);
}
