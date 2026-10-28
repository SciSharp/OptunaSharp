using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
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
    /// <exception cref="NumSharp.ValueError"><paramref name="nJobs" /> is 0 or below -1 (Python's <c>ThreadPoolExecutor</c> refusal).</exception>
    /// <exception cref="System.Exception">The supplied objective or evaluation callback throws an exception that is not handled by this operation.</exception>
    /// <remarks>
    ///     With <paramref name="nJobs" /> other than 1 this follows Python's executor block exactly: a failure is
    ///     raised (as the original exception, not an <see cref="AggregateException" />) only after every trial
    ///     still running has finished, and a failure among the trials that finish after the last submission is
    ///     not raised at all; such a trial is still recorded as FAIL and logged.
    /// </remarks>
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
                // Python passes n_jobs to ThreadPoolExecutor(max_workers=n_jobs), which refuses anything below one.
                // Without this check 0 or -2 reached Task.WhenAny on an empty set, an ArgumentException Python never raises.
                if (nJobs <= 0)
                    throw new NumSharp.ValueError("max_workers must be greater than 0");

                var timeStart = DateTime.Now;
                var futures = new HashSet<Task>();

                try
                {
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
                            Task.WhenAny(futures).GetAwaiter().GetResult();
                            // Python's wait(FIRST_COMPLETED) returns every future finished by then, removes them
                            // all, and raises the first failure among them; GetResult rethrows that original
                            // exception (not an AggregateException).
                            var completed = futures.Where(f => f.IsCompleted).ToList();
                            foreach (var f in completed)
                                futures.Remove(f);
                            // Reading Exception marks every failure observed: Python drops the ones after the first,
                            // and here they must not resurface later through TaskScheduler.UnobservedTaskException.
                            foreach (var f in completed)
                                _ = f.Exception;
                            completed.FirstOrDefault(f => !f.IsCompletedSuccessfully)?.GetAwaiter().GetResult();
                        }

                        futures.Add(StartWorker(() => OptimizeSequential(
                            study, func, 1, timeout, catchTypes, callbacks, gcAfterTrial,
                            reseedSamplerRng: true, timeStart: timeStart, progressBar: progressBar,
                            restoreLoopFlag: true)));
                    }
                }
                finally
                {
                    // Python's `with ThreadPoolExecutor(...)` exit: wait for every trial still running, on success
                    // and on failure alike, so Optimize never returns while its trials are in flight. Their
                    // exceptions are not raised, as Python never calls result() on them; RunTrial already logged
                    // each failure and the storage holds it as FAIL.
                    WaitForAllWithoutRaising(futures);
                }
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
    /// <param name="restoreLoopFlag">
    ///     True for an n_jobs worker: put this thread's <c>InOptimizeLoop</c> back to its previous value on exit.
    ///     False on the caller's own thread, where <see cref="Optimize" /> clears it, as Python does.
    /// </param>
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
        ProgressBar? progressBar,
        bool restoreLoopFlag = false)
    {
        // Here we set `InOptimizeLoop = true`, not at the beginning of the `Optimize()` function.
        // Because it is a thread-local value and the n_jobs option spawns new threads.
        bool wasInOptimizeLoop = study.ThreadLocal.InOptimizeLoop;
        study.ThreadLocal.InOptimizeLoop = true;
        try
        {
            if (reseedSamplerRng)
            {
                // Workers reseed the one shared sampler concurrently with each other's draws; the sampler's
                // monitor keeps the generator state whole (numpy's RandomState locks every call).
                var sampler = study.Sampler;
                lock (sampler.SyncRoot)
                    sampler.ReseedRng();
            }

            RunTrials(study, func, nTrials, timeout, catchTypes, callbacks, gcAfterTrial, timeStart, progressBar);
        }
        finally
        {
            // Python's executor threads end with the executor, taking their flag with them. Workers used to run on
            // thread-pool threads, which outlive this call: a stale `true` made 61 of 64 later Optimize calls of
            // this study on pool threads fail with a false "Nested invocation" error. Workers now get dedicated
            // threads (StartWorker); restoring the flag keeps any thread that runs a worker clean regardless.
            if (restoreLoopFlag)
                study.ThreadLocal.InOptimizeLoop = wasInOptimizeLoop;
        }
    }

    /// <summary>The trial loop of Python's <c>_optimize_sequential</c>, after its thread-local setup.</summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="func">The objective callback returning one value per study objective for each running trial.</param>
    /// <param name="nTrials">The trial-count limit; null leaves the count unlimited.</param>
    /// <param name="timeout">The optimization time limit in seconds; null leaves time unlimited.</param>
    /// <param name="catchTypes">Exception types permitted to fail a trial while optimization continues.</param>
    /// <param name="callbacks">Delegates invoked after each trial has been recorded in its final state.</param>
    /// <param name="gcAfterTrial">Whether to request garbage collection after each objective invocation.</param>
    /// <param name="timeStart">The start time shared by workers when enforcing the overall timeout.</param>
    /// <param name="progressBar">The progress renderer shared with this optimization loop, or null when no progress is displayed.</param>
    /// <exception cref="System.Exception">The supplied objective or evaluation callback throws an exception that is not handled by this operation.</exception>
    private static void RunTrials(
        Study study,
        Func<Trial, IReadOnlyList<double>> func,
        int? nTrials,
        double? timeout,
        Type[] catchTypes,
        IReadOnlyList<Action<Study, FrozenTrial>>? callbacks,
        bool gcAfterTrial,
        DateTime? timeStart,
        ProgressBar? progressBar)
    {
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

    /// <summary>
    ///     Starts one n_jobs worker on a dedicated thread, as Python's <c>ThreadPoolExecutor</c> runs its work on
    ///     threads of its own.
    /// </summary>
    /// <param name="work">One <c>_optimize_sequential(n_trials=1)</c> call.</param>
    /// <returns>The worker's task; it completes when the trial and its callbacks are done.</returns>
    /// <remarks>
    ///     A worker spends most of its time blocked: on the storage's lock, on file or network I/O, on the
    ///     objective. Run on the .NET thread pool, n_jobs above the pool's minimum thread count blocked every pool
    ///     thread, and StackExchange.Redis (which completes even synchronous calls on the pool) then timed out:
    ///     <c>Optimize(nJobs: 96)</c> on a Redis journal died after 88 trials. A dedicated thread per trial costs
    ///     microseconds against a trial's milliseconds, and it starts with fresh thread-local state, as a new
    ///     executor thread does.
    /// </remarks>
    private static Task StartWorker(Action work)
        => Task.Factory.StartNew(work, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);

    /// <summary>
    ///     Blocks until every worker task has finished, observing (not raising) their failures: the exit of
    ///     Python's <c>with ThreadPoolExecutor(...)</c> block, which calls <c>shutdown(wait=True)</c> only.
    /// </summary>
    /// <param name="futures">Worker tasks that may still be running; failed ones are observed and dropped.</param>
    /// <remarks>
    ///     Workers run on dedicated threads (see <see cref="StartWorker" />), which are never inlined into this
    ///     wait. Catching the <see cref="AggregateException" /> marks every failure observed, so none resurfaces
    ///     later through <see cref="TaskScheduler.UnobservedTaskException" />.
    /// </remarks>
    private static void WaitForAllWithoutRaising(IReadOnlyCollection<Task> futures)
    {
        if (futures.Count == 0)
            return;
        try
        {
            Task.WaitAll(futures.ToArray());
        }
        catch (AggregateException)
        {
            // Not raised, as in Python (see the summary); each failure was already logged by RunTrial.
        }
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
            // Under the trial's tell monitor, as Study.Tell: an application thread telling this trial at the same
            // time (a reaper failing slow trials, say) acts wholly before or after this tell, so exactly one of the
            // two passes the "still running" check and reaches the sampler's AfterTrial (see BaseStorage.TellLockFor).
            lock (study.Storage.TellLockFor(trial.TrialId))
            {
                var frozen = TellHelpers.GetFrozenTrial(study, trial);
                (updatedState, values, warningMessage) = TellHelpers.TellWithWarning(
                    study: study,
                    frozenTrial: frozen,
                    valueOrValues: valueOrValues,
                    state: state,
                    suppressWarning: true);
            }
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
