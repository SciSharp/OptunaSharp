using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using OptunaSharp.Logging;
using OptunaSharp.Storages;

namespace OptunaSharp.Internal;

/// <summary>Port of <c>optuna/study/_optimize.py</c>.</summary>
internal static class OptimizeLoop
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.study._optimize");

    /// <summary>Python <c>_optimize</c>.</summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="func">The objective callback returning one value per study objective for each running trial.</param>
    /// <param name="nTrials">The trial-count limit; null leaves the count unlimited.</param>
    /// <param name="timeout">The optimization time limit in seconds; null leaves time unlimited.</param>
    /// <param name="nJobs">The requested worker count; minus one uses the processor count.</param>
    /// <param name="catchExceptions">Exception types that mark a failed objective trial without stopping optimization.</param>
    /// <param name="callbacks">Delegates invoked after each trial has been recorded in its final state.</param>
    /// <param name="gcAfterTrial">Whether to request garbage collection after each objective invocation.</param>
    /// <param name="showProgressBar">Whether optimization should render progress to the console.</param>
    /// <exception cref="NumSharp.RuntimeError">Optimization is invoked recursively for a study already being optimized on the current execution context.</exception>
    /// <exception cref="System.Exception">The supplied objective or evaluation callback throws an exception that is not handled by this operation.</exception>
    internal static void Optimize(
        Study study,
        Func<Trial, IReadOnlyList<double>> func,
        int? nTrials = null,
        double? timeout = null,
        int nJobs = 1,
        Type[]? catchExceptions = null,
        IReadOnlyList<Action<Study, FrozenTrial>>? callbacks = null,
        bool gcAfterTrial = false,
        bool showProgressBar = false)
    {
        var catchTypes = catchExceptions ?? Array.Empty<Type>();

        if (study.ThreadLocal.InOptimizeLoop)
            throw new NumSharp.RuntimeError("Nested invocation of `Study.optimize` method isn't allowed.");

        if (showProgressBar && nTrials is null && timeout is not null && nJobs != 1)
        {
            OptunaWarnings.Warn("The timeout-based progress bar is not supported with n_jobs != 1.");
            showProgressBar = false;
        }

        var progressBar = new ProgressBar(showProgressBar, nTrials, timeout);

        study.StopFlag = false;

        try
        {
            if (nJobs == 1)
            {
                OptimizeSequential(
                    study, func, nTrials, timeout, catchTypes, callbacks, gcAfterTrial,
                    reseedSamplerRng: false, timeStart: null, progressBar: progressBar);
            }
            else
            {
                if (nJobs == -1)
                    nJobs = Environment.ProcessorCount;

                var timeStart = DateTime.Now;
                var futures = new HashSet<Task>();

                for (int nSubmittedTrials = 0; ; nSubmittedTrials++)
                {
                    if (study.StopFlag)
                        break;

                    if (timeout is not null && (DateTime.Now - timeStart).TotalSeconds > timeout.Value)
                        break;

                    if (nTrials is not null && nSubmittedTrials >= nTrials.Value)
                        break;

                    if (futures.Count >= nJobs)
                    {
                        var completed = Task.WhenAny(futures).GetAwaiter().GetResult();
                        futures.Remove(completed);
                        // Raise if exception occurred in executing the completed futures.
                        completed.GetAwaiter().GetResult();
                    }

                    futures.Add(Task.Run(() => OptimizeSequential(
                        study, func, 1, timeout, catchTypes, callbacks, gcAfterTrial,
                        reseedSamplerRng: true, timeStart: timeStart, progressBar: progressBar)));
                }

                Task.WaitAll(futures.ToArray());
            }
        }
        finally
        {
            study.ThreadLocal.InOptimizeLoop = false;
            progressBar.Close();
        }
    }

    /// <summary>Python <c>_optimize_sequential</c>.</summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="func">The objective callback returning one value per study objective for each running trial.</param>
    /// <param name="nTrials">The trial-count limit; null leaves the count unlimited.</param>
    /// <param name="timeout">The optimization time limit in seconds; null leaves time unlimited.</param>
    /// <param name="catchTypes">Exception types permitted to fail a trial while optimization continues.</param>
    /// <param name="callbacks">Delegates invoked after each trial has been recorded in its final state.</param>
    /// <param name="gcAfterTrial">Whether to request garbage collection after each objective invocation.</param>
    /// <param name="reseedSamplerRng">Whether the worker should reseed its sampler before starting sequential trials.</param>
    /// <param name="timeStart">The start time shared by workers when enforcing the overall timeout.</param>
    /// <param name="progressBar">The progress renderer shared with this optimization loop, or null when no progress is displayed.</param>
    /// <exception cref="System.Exception">The supplied objective or evaluation callback throws an exception that is not handled by this operation.</exception>
    private static void OptimizeSequential(
        Study study,
        Func<Trial, IReadOnlyList<double>> func,
        int? nTrials,
        double? timeout,
        Type[] catchTypes,
        IReadOnlyList<Action<Study, FrozenTrial>>? callbacks,
        bool gcAfterTrial,
        bool reseedSamplerRng,
        DateTime? timeStart,
        ProgressBar? progressBar)
    {
        // Here we set `InOptimizeLoop = true`, not at the beginning of the `Optimize()` function.
        // Because it is a thread-local value and the n_jobs option spawns new threads.
        study.ThreadLocal.InOptimizeLoop = true;
        if (reseedSamplerRng)
            study.Sampler.ReseedRng();

        int iTrial = 0;

        timeStart ??= DateTime.Now;

        while (true)
        {
            if (study.StopFlag)
                break;

            if (nTrials is not null)
            {
                if (iTrial >= nTrials.Value)
                    break;
                iTrial++;
            }

            if (timeout is not null)
            {
                double elapsedSeconds = (DateTime.Now - timeStart.Value).TotalSeconds;
                if (elapsedSeconds >= timeout.Value)
                    break;
            }

            int frozenTrialId;
            try
            {
                frozenTrialId = RunTrial(study, func, catchTypes);
            }
            finally
            {
                if (gcAfterTrial)
                    GC.Collect();
            }

            if (callbacks is not null)
            {
                var frozenTrial = study.Storage.GetTrial(frozenTrialId);
                foreach (var callback in callbacks)
                    callback(study, frozenTrial.DeepCopy());
            }

            if (progressBar is not null)
            {
                double elapsedSeconds = (DateTime.Now - timeStart.Value).TotalSeconds;
                progressBar.Update(elapsedSeconds, study);
            }
        }

        study.Storage.RemoveSession();
    }

    /// <summary>Python <c>_run_trial</c>.</summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="func">The objective callback returning one value per study objective for the new trial.</param>
    /// <param name="catchTypes">Exception types permitted to fail a trial while optimization continues.</param>
    /// <returns>The storage identifier of the executed trial, suitable for a subsequent storage GetTrial call.</returns>
    /// <exception cref="System.Exception">The supplied objective or evaluation callback throws an exception that is not handled by this operation.</exception>
    internal static int RunTrial(Study study, Func<Trial, IReadOnlyList<double>> func, Type[] catchTypes)
    {
        if (HeartbeatHelpers.IsHeartbeatEnabled(study.Storage))
            HeartbeatHelpers.FailStaleTrials(study);

        var trial = study.Ask();

        TrialState? state = null;
        IReadOnlyList<double>? valueOrValues = null;
        Exception? funcErr = null;

        var heartbeatThread = HeartbeatHelpers.GetHeartbeatThread(trial.TrialId, study.Storage);
        heartbeatThread.Start();
        try
        {
            try
            {
                valueOrValues = func(trial);
            }
            catch (TrialPruned e)
            {
                // TODO(mamu): Handle multi-objective cases.
                state = TrialState.Pruned;
                funcErr = e;
            }
            catch (Exception e)
            {
                state = TrialState.Fail;
                funcErr = e;
            }
        }
        finally
        {
            // The heartbeat must finish before Tell changes the trial state, including when the
            // objective throws, so the worker cannot keep advertising a finalized trial as running.
            heartbeatThread.Join();
        }

        // TellWithWarning may raise during trial post-processing.
        TrialState updatedState;
        List<double>? values;
        string? warningMessage;
        try
        {
            var frozen = TellHelpers.GetFrozenTrial(study, trial);
            (updatedState, values, warningMessage) = TellHelpers.TellWithWarning(
                study: study,
                frozenTrial: frozen,
                valueOrValues: valueOrValues,
                state: state,
                suppressWarning: true);
        }
        catch (Exception)
        {
            var frozenTrial = study.Storage.GetTrial(trial.TrialId);
            LogTrialOutcome(study, trial, frozenTrial.State, frozenTrial.Values, funcErr, null, valueOrValues);
            throw;
        }

        LogTrialOutcome(study, trial, updatedState, values, funcErr, warningMessage, valueOrValues);

        if (updatedState == TrialState.Fail && funcErr is not null && !IsCaught(funcErr, catchTypes))
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(funcErr).Throw();
        return trial.TrialId;
    }

    /// <summary>Routes a finalized trial to the completed, pruned, or failed diagnostic path.</summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="trial">The trial whose recorded state and values are used.</param>
    /// <param name="updatedState">The final state recorded for the trial.</param>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <param name="funcErr">The exception captured from the objective, or null when it returned normally.</param>
    /// <param name="warningMessage">The diagnostic produced while validating objective values, or null when none was needed.</param>
    /// <param name="valueOrValues">The values returned by the objective before completion-state validation.</param>
    private static void LogTrialOutcome(
        Study study,
        Trial trial,
        TrialState updatedState,
        IReadOnlyList<double>? values,
        Exception? funcErr,
        string? warningMessage,
        IReadOnlyList<double>? valueOrValues)
    {
        if (updatedState == TrialState.Complete)
        {
            System.Diagnostics.Debug.Assert(values is not null);
            // Build the log arguments only when the INFO line can actually be emitted: `trial.Params` deep-copies
            // the whole cached trial (attrs included — a conditional genome's dormant map runs to hundreds of
            // entries), which measured as a dominant PER-TRIAL cost of a cheap-objective study even with logging
            // silenced. LogCompletedTrial re-checks the level itself, so behavior at INFO is unchanged.
            if (_logger.IsEnabledFor(LogLevel.Info))
                study.LogCompletedTrial(values!.ToList(), trial.Number, new Collections.OrderedDict<string, object?>(trial.Params));
        }
        else if (updatedState == TrialState.Pruned)
        {
            _logger.Info($"Trial {trial.Number} pruned. {funcErr?.Message ?? string.Empty}");
        }
        else if (updatedState == TrialState.Fail)
        {
            if (funcErr is not null)
            {
                LogFailedTrial(
                    trial.Number, trial.Params,
                    $"{funcErr.GetType().Name}({PythonRepr.Repr(funcErr.Message)})",
                    excInfo: funcErr, valueOrValues: valueOrValues);
            }
            else if (warningMessage is not null)
            {
                LogFailedTrial(trial.Number, trial.Params, warningMessage, valueOrValues: valueOrValues);
            }
        }
    }

    /// <summary>Returns true when the exception is assignable to one of the allowed failure types.</summary>
    /// <param name="error">The exception being tested against the allowed failure types.</param>
    /// <param name="catchTypes">Exception types permitted to fail a trial while optimization continues.</param>
    /// <returns>True when the exception is assignable to one of the allowed failure types.</returns>
    private static bool IsCaught(Exception error, Type[] catchTypes)
        => catchTypes.Any(t => t.IsInstanceOfType(error));

    /// <summary>Python <c>_log_failed_trial</c>.</summary>
    /// <param name="trialNumber">The study-local trial number, rather than the storage's internal trial identifier.</param>
    /// <param name="trialParams">The recorded external parameter values of the failed trial, keyed by parameter name.</param>
    /// <param name="message">The diagnostic text to display or emit.</param>
    /// <param name="excInfo">The exception whose details should accompany the failed-trial message.</param>
    /// <param name="valueOrValues">The values returned by the objective before completion-state validation.</param>
    private static void LogFailedTrial(
        int trialNumber,
        IReadOnlyDictionary<string, object?> trialParams,
        string message,
        Exception? excInfo = null,
        IReadOnlyList<double>? valueOrValues = null)
    {
        _logger.Warning(
            $"Trial {trialNumber} failed with parameters: {PythonRepr.DictRepr(new Collections.OrderedDict<string, object?>(trialParams))} " +
            $"because of the following error: {message}.",
            excInfo);

        _logger.Warning(
            $"Trial {trialNumber} failed with value {(valueOrValues is null ? "None" : PythonRepr.ListRepr(valueOrValues))}.");
    }
}
