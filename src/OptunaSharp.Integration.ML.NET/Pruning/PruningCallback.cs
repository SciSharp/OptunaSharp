using System;

namespace OptunaSharp.Integration;

/// <summary>
///     A framework-agnostic Optuna pruning callback for iterative / epoch-based .NET training loops.
///
///     <para>
///     This is the shared essence of Optuna's per-framework pruning callbacks
///     (<c>optuna_integration</c>'s <c>XGBoostPruningCallback</c> / <c>LightGBMPruningCallback</c> /
///     <c>CatBoostPruningCallback</c>): once per boosting round or training epoch, report the current
///     validation metric to the <see cref="Trial" /> and abort the trial when the study's pruner says it
///     is unpromising.
///     </para>
///
///     <para>
///     <b>.NET note (this is a Phase-6 adapter, not a behavior-parity port).</b> In Python those
///     callbacks are invoked <i>by the training framework</i> (XGBoost/LightGBM/CatBoost hand the
///     callback their per-iteration evaluation log). ML.NET's gradient-boosting trainers
///     (<c>Microsoft.ML.LightGbm</c> / <c>Microsoft.ML.FastTree</c>) expose <b>no managed per-iteration
///     hook</b>, so — unlike the Python callbacks — you drive this yourself from whatever iterative loop
///     you own: a TorchSharp epoch loop, an online / incremental ML.NET learner, or any hand-written
///     training loop where a scalar validation metric becomes available each step.
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
    public PruningCallback(Trial trial)
    {
        _trial = trial ?? throw new ArgumentNullException(nameof(trial));
    }

    /// <summary><c>true</c> once <see cref="AfterIteration" /> has decided the trial should be pruned.</summary>
    public bool Pruned => _pruned;

    /// <summary>
    ///     Report an intermediate <paramref name="value" /> for the given <paramref name="step" /> and, if
    ///     the study's pruner decides the trial is unpromising, immediately raise
    ///     <see cref="TrialPruned" /> (the XGBoost / LightGBM convention).
    /// </summary>
    /// <param name="value">The intermediate objective value observed at this step (e.g. a validation metric).</param>
    /// <param name="step">
    ///     The step of the trial (e.g. the boosting round or training epoch). Must be a non-negative integer;
    ///     pruners assume <paramref name="step" /> starts at zero.
    /// </param>
    /// <exception cref="TrialPruned">Thrown when the pruner decides the trial should be pruned.</exception>
    public void Report(double value, int step)
    {
        _trial.Report(value, step);
        if (_trial.ShouldPrune())
        {
            _message = $"Trial was pruned at iteration {step}.";
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
    /// <param name="step">The step of the trial. Must be a non-negative integer starting at zero.</param>
    /// <returns><c>true</c> to keep training; <c>false</c> if the trial should be pruned.</returns>
    public bool AfterIteration(double value, int step)
    {
        _trial.Report(value, step);
        if (_trial.ShouldPrune())
        {
            _message = $"Trial was pruned at iteration {step}.";
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
