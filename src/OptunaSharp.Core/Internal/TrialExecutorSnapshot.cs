using System;
using System.Threading;
using System.Threading.Tasks;

namespace OptunaSharp.Internal;

/// <summary>
///     The executor one <c>Optimize(nJobs)</c> call runs its trials on: the <see cref="TrialExecutorOptions" /> that
///     won the resolution, read once when the call starts, plus the scheduler those values select.
/// </summary>
/// <remarks>
///     Reading the options once makes the call immune to later changes of the options object, and lets a derived
///     options class compute its values: each virtual getter is called exactly once per call (THREADING.md §3.5).
/// </remarks>
internal readonly struct TrialExecutorSnapshot
{
    private readonly TaskScheduler _scheduler;
    private readonly TaskCreationOptions _creationOptions;

    /// <summary>Captures resolved values and the scheduler they select.</summary>
    /// <param name="kind">The executor kind the call uses.</param>
    /// <param name="threadIdleTimeout">The dedicated threads' idle timeout; kept for diagnostics on the pool.</param>
    private TrialExecutorSnapshot(TrialExecutorKind kind, TimeSpan threadIdleTimeout)
    {
        Kind = kind;
        ThreadIdleTimeout = threadIdleTimeout;
        if (kind == TrialExecutorKind.ThreadPool)
        {
            _scheduler = TaskScheduler.Default;
            // PreferFairness queues each worker on the pool's global queue, not on the submitting thread's local
            // one. Optimize called from a pool thread then cannot inline its own trials: Task.WaitAll only inlines a
            // task it can pop from the current thread's local queue. Queued trials also take turns with the host's
            // own work instead of jumping ahead of it.
            _creationOptions = TaskCreationOptions.DenyChildAttach | TaskCreationOptions.PreferFairness;
        }
        else
        {
            // One reused thread set per distinct idle timeout; the default timeout is TrialWorkerScheduler.Shared.
            _scheduler = TrialWorkerScheduler.For(threadIdleTimeout);
            _creationOptions = TaskCreationOptions.DenyChildAttach;
        }
    }

    /// <summary>The executor kind this call runs its trials on.</summary>
    internal TrialExecutorKind Kind { get; }

    /// <summary>The idle timeout of the dedicated threads; the pool ignores it.</summary>
    internal TimeSpan ThreadIdleTimeout { get; }

    /// <summary>
    ///     Resolves the four levels of THREADING.md §3.5 and reads the winner once. Three are visible here: the call's
    ///     argument, the study's setting, and the process-wide default. The fourth, the DI default, arrives through the
    ///     study's setting, which the DI study factory writes when it builds the study.
    /// </summary>
    /// <param name="callOptions">The <c>executorOptions</c> argument of the <c>Optimize</c> call, or null.</param>
    /// <param name="study">The study being optimized; its <see cref="Study.ExecutorOptions" /> is the second level.</param>
    /// <returns>The values the call runs with.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    ///     The winning options return an undefined executor kind or an invalid idle timeout; only a derived options
    ///     class can, because the setters validate.
    /// </exception>
    internal static TrialExecutorSnapshot Resolve(TrialExecutorOptions? callOptions, Study study)
    {
        TrialExecutorOptions options = callOptions ?? study.ExecutorOptions ?? TrialExecutorOptions.Default;

        // Each getter exactly once: a derived class may compute its values, and the call must run with one
        // consistent pair even if another thread changes the instance meanwhile.
        TrialExecutorKind kind = options.Executor;
        TimeSpan threadIdleTimeout = options.ThreadIdleTimeout;
        TrialExecutorOptions.ValidateExecutor(kind, nameof(TrialExecutorOptions.Executor));
        TrialExecutorOptions.ValidateThreadIdleTimeout(threadIdleTimeout, nameof(TrialExecutorOptions.ThreadIdleTimeout));
        return new TrialExecutorSnapshot(kind, threadIdleTimeout);
    }

    /// <summary>Starts one n_jobs worker on this call's executor.</summary>
    /// <param name="worker">One <c>_optimize_sequential(n_trials=1)</c> call.</param>
    /// <returns>The worker's task; it completes when the trial and its callbacks are done.</returns>
    internal Task Start(Action worker)
        => Task.Factory.StartNew(worker, CancellationToken.None, _creationOptions, _scheduler);
}
