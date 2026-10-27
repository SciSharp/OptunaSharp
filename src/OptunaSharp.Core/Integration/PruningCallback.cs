using System;

namespace OptunaSharp.Integration;

/// <summary>
///     The framework-agnostic Optuna pruning callback for iterative / epoch-based .NET training loops —
///     the canonical .NET pruning callback, and the base that framework-specific integration packages
///     extend.
///
///     <para>
///     It is the shared essence of Optuna's per-framework pruning callbacks
///     (<c>optuna_integration</c>'s <c>XGBoostPruningCallback</c> / <c>LightGBMPruningCallback</c> /
///     <c>CatBoostPruningCallback</c> / <c>KerasPruningCallback</c> / …): once per boosting round or
///     training epoch, report the current validation metric to the <see cref="Trial" /> and abort the
///     trial when the study's pruner says it is unpromising.
///     </para>
///
///     <para>
///     <b>Why this is the .NET replacement for the named per-framework callbacks.</b> In Python each
///     callback is invoked <i>by the training framework</i> — CatBoost/XGBoost/Keras/… hand it their
///     per-iteration evaluation data — which is the callback's whole reason to exist. Those frameworks
///     are Python-only: there is no .NET CatBoost/XGBoost/Ignite/Lightning/skorch/fastai to call a
///     named callback, and .NET's own trainers (ML.NET's <c>Microsoft.ML.LightGbm</c> /
///     <c>Microsoft.ML.FastTree</c>, TorchSharp) expose <b>no managed per-iteration hook</b>. So instead
///     of hollow per-framework shells, you drive this one callback yourself from whatever iterative loop
///     you own — a TorchSharp epoch loop, an online / incremental ML.NET learner, or any hand-written
///     training loop where a scalar validation metric becomes available each step.
///     </para>
///
///     <para>
///     <b>Extending it.</b> A framework that <i>does</i> expose a managed callback in .NET can ship a
///     satellite package whose callback subclasses this one and implements that framework's callback
///     interface — for example a real Keras callback over SciSharp's TensorFlow.NET would be
///     <c>class KerasPruningCallback : PruningCallback, Tensorflow.Keras.Engine.ICallback</c>, reading
///     the metric in <c>on_epoch_end(epoch, logs)</c> and calling the inherited <see cref="Report" />.
///     Subclasses get the <see cref="Trial" /> via the protected <see cref="Trial" /> property and can
///     override <see cref="FormatPrunedMessage" /> to word the pruning message their way (e.g. "epoch"
///     rather than "iteration").
///     </para>
///
///     <para>
///     Two usage styles are provided, matching the two Optuna callback conventions:
///     </para>
///     <list type="bullet">
///         <item>
///             <description>
///             <b>Immediate</b> (the XGBoost / LightGBM convention) — call <see cref="Report" /> each step;
///             it reports the value and throws <see cref="TrialPruned" /> as soon as the pruner decides to
///             prune, unwinding your loop the way an objective function expects.
///             </description>
///         </item>
///         <item>
///             <description>
///             <b>Deferred</b> (the CatBoost convention, for engines that must not have an exception thrown
///             from inside their callback) — call <see cref="AfterIteration" /> each step; it returns
///             <c>false</c> when the trial should stop, without throwing. After training, call
///             <see cref="CheckPruned" /> once to raise <see cref="TrialPruned" /> if pruning occurred.
///             </description>
///         </item>
///     </list>
/// </summary>
/// <example>
///     <code>
///     double Objective(Trial trial)
///     {
///         var pruning = new PruningCallback(trial);
///         var model = CreateModel(trial);   // e.g. a TorchSharp module
///         for (int epoch = 0; epoch &lt; nEpochs; epoch++)
///         {
///             model.TrainOneEpoch();
///             double validationError = model.Evaluate(validationSet);
///             pruning.Report(validationError, epoch);   // reports + throws TrialPruned when unpromising
///         }
///         return model.Evaluate(validationSet);
///     }
///     </code>
/// </example>
public class PruningCallback
{
    private readonly Trial _trial;
    private bool _pruned;
    private string _message = "";

    /// <param name="trial">
    ///     The <see cref="Trial" /> corresponding to the current evaluation of the objective function.
    /// </param>
    /// <summary>Associates an iterative training callback with the trial whose intermediate values it reports.</summary>
    /// <exception cref="System.ArgumentNullException">The trial is null.</exception>
    public PruningCallback(Trial trial)
    {
        _trial = trial ?? throw new ArgumentNullException(nameof(trial));
    }

    /// <summary><c>true</c> once <see cref="AfterIteration" /> has decided the trial should be pruned.</summary>
    public bool Pruned => _pruned;

    /// <summary>
    ///     The <see cref="OptunaSharp.Trial" /> this callback drives — exposed for subclasses (framework
    ///     integration packages) that need to read the trial or its study directly.
    /// </summary>
    protected Trial Trial => _trial;

    /// <summary>
    ///     The message carried by the <see cref="TrialPruned" /> this callback raises. The default is
    ///     <c>"Trial was pruned at iteration {step}."</c>; a subclass may override it to word the step
    ///     differently (e.g. <c>"Trial was pruned at epoch {step}."</c>).
    /// </summary>
    /// <param name="step">The step at which pruning was decided (64-bit, like every trial step).</param>
    /// <returns>The diagnostic message used when the callback raises TrialPruned.</returns>
    protected virtual string FormatPrunedMessage(long step) => $"Trial was pruned at iteration {step}.";

    /// <summary>
    ///     Report an intermediate <paramref name="value" /> for the given <paramref name="step" /> and, if
    ///     the study's pruner decides the trial is unpromising, immediately raise
    ///     <see cref="TrialPruned" /> (the XGBoost / LightGBM convention).
    /// </summary>
    /// <param name="value">The intermediate objective value observed at this step (e.g. a validation metric).</param>
    /// <param name="step">
    ///     The step of the trial (e.g. the boosting round or training epoch). Must be a non-negative integer;
    ///     pruners assume <paramref name="step" /> starts at zero. 64-bit, like <see cref="OptunaSharp.Trial.Report" />;
    ///     an <see cref="int" /> iteration counter converts implicitly.
    /// </param>
    /// <exception cref="TrialPruned">Thrown when the pruner decides the trial should be pruned.</exception>
    /// <exception cref="NumSharp.ValueError">The reported step is negative.</exception>
    /// <exception cref="System.NotImplementedException">Intermediate reporting or pruning is requested for a multi-objective study.</exception>
    /// <exception cref="OptunaSharp.UpdateFinishedTrialError">The trial has already finished when its intermediate value is stored.</exception>
    /// <exception cref="NumSharp.KeyError">The trial no longer exists in storage.</exception>
    /// <exception cref="System.Exception">A custom pruner, storage provider, or message formatter throws.</exception>
    public void Report(double value, long step)
    {
        _trial.Report(value, step);
        if (_trial.ShouldPrune())
        {
            _message = FormatPrunedMessage(step);
            _pruned = true;
            throw new TrialPruned(_message);
        }
    }

    /// <summary>
    ///     Report an intermediate <paramref name="value" /> for the given <paramref name="step" /> and
    ///     return whether training should continue, <b>without</b> raising (the CatBoost convention, for
    ///     engines whose callback cannot propagate an exception). Returns <c>false</c> if the trial should
    ///     be pruned — stop your loop and then call <see cref="CheckPruned" />.
    /// </summary>
    /// <param name="value">The intermediate objective value observed at this step.</param>
    /// <param name="step">The step of the trial (64-bit). Must be a non-negative integer starting at zero.</param>
    /// <returns><c>true</c> to keep training; <c>false</c> if the trial should be pruned.</returns>
    /// <exception cref="NumSharp.ValueError">The reported step is negative.</exception>
    /// <exception cref="System.NotImplementedException">Intermediate reporting or pruning is requested for a multi-objective study.</exception>
    /// <exception cref="OptunaSharp.UpdateFinishedTrialError">The trial has already finished when its intermediate value is stored.</exception>
    /// <exception cref="NumSharp.KeyError">The trial no longer exists in storage.</exception>
    /// <exception cref="System.Exception">A custom pruner, storage provider, or message formatter throws.</exception>
    /// <remarks>The deferred convention suppresses the pruning exception only. Reporting validation, storage, and custom-pruner failures can still propagate.</remarks>
    public bool AfterIteration(double value, long step)
    {
        _trial.Report(value, step);
        if (_trial.ShouldPrune())
        {
            // Remember the decision so training engines that cannot unwind inside their callback can
            // stop normally and translate the decision into TrialPruned after returning to managed code.
            _message = FormatPrunedMessage(step);
            _pruned = true;
            return false;
        }

        return true;
    }

    /// <summary>
    ///     Raise <see cref="TrialPruned" /> if a previous <see cref="AfterIteration" /> call decided the
    ///     trial should be pruned. Call this once after the (deferred-style) training loop, as with
    ///     Optuna's <c>CatBoostPruningCallback.check_pruned</c>.
    /// </summary>
    /// <exception cref="TrialPruned">Thrown when the trial was pruned.</exception>
    public void CheckPruned()
    {
        if (_pruned)
            throw new TrialPruned(_message);
    }
}
