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

    private static bool IsCaught(Exception error, Type[] catchTypes)
        => catchTypes.Any(t => t.IsInstanceOfType(error));

    /// <summary>Python <c>_log_failed_trial</c>.</summary>
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
