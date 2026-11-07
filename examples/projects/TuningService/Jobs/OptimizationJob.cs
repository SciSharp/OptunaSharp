using OptunaSharp;
using TuningService.Contracts;

namespace TuningService.Jobs;

/// <summary>Where an <see cref="OptimizationJob" /> is in its life.</summary>
public enum OptimizationJobState
{
    /// <summary>Waiting in the queue for the runner.</summary>
    Queued,

    /// <summary>Its study is optimizing.</summary>
    Running,

    /// <summary>It ran all its trials.</summary>
    Completed,

    /// <summary>It was canceled: before it started, or while it ran (the trials in flight then finished).</summary>
    Canceled,

    /// <summary>An exception ended it; <see cref="JobDto.Error" /> says which.</summary>
    Failed,
}

/// <summary>
///     One request to optimize a study with the built-in objective in the background: its parameters, its
///     progress, and the cancellation switch the <c>DELETE /api/jobs/{id}</c> endpoint flips.
/// </summary>
/// <remarks>
///     <para>
///         <b>Thread-safety.</b> HTTP requests read a job while the runner updates it, and under <c>nJobs</c>
///         the per-trial callback that counts finished trials runs on several worker threads at once. The trial
///         counters are therefore updated with <see cref="Interlocked" />, and the state and timestamps change
///         together under one lock, so <see cref="ToDto" /> never returns a half-updated job.
///     </para>
///     <para>
///         The <see cref="CancellationTokenSource" /> is never disposed: it has no timer and no registrations
///         that outlive the job, so disposing it would only add a race with a late <see cref="Cancel" />.
///     </para>
/// </remarks>
public sealed class OptimizationJob
{
    private readonly object _gate = new();
    private readonly CancellationTokenSource _cancellation = new();

    private OptimizationJobState _state = OptimizationJobState.Queued;
    private DateTimeOffset? _startedAt;
    private DateTimeOffset? _finishedAt;
    private string? _error;

    private int _complete;
    private int _pruned;
    private int _failed;

    /// <summary>Creates a queued job.</summary>
    /// <param name="studyName">The study to optimize; it is created if it does not exist yet.</param>
    /// <param name="nTrials">How many trials to run.</param>
    /// <param name="nJobs">How many of them to run in parallel.</param>
    /// <exception cref="ArgumentException"><paramref name="studyName" /> is <c>null</c>, empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="nTrials" /> or <paramref name="nJobs" /> is below 1.</exception>
    public OptimizationJob(string studyName, int nTrials, int nJobs)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(studyName);
        ArgumentOutOfRangeException.ThrowIfLessThan(nTrials, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(nJobs, 1);
        Id = Guid.NewGuid();
        StudyName = studyName;
        NTrials = nTrials;
        NJobs = nJobs;
        QueuedAt = DateTimeOffset.UtcNow;
    }

    /// <summary>The job's identifier, as used in <c>/api/jobs/{id}</c>.</summary>
    public Guid Id { get; }

    /// <summary>The study the job optimizes.</summary>
    public string StudyName { get; }

    /// <summary>How many trials the job runs.</summary>
    public int NTrials { get; }

    /// <summary>How many trials it runs in parallel.</summary>
    public int NJobs { get; }

    /// <summary>When the job was queued, in UTC.</summary>
    public DateTimeOffset QueuedAt { get; }

    /// <summary>Signaled by <see cref="Cancel" />; the runner stops scheduling trials once it is.</summary>
    public CancellationToken CancellationToken => _cancellation.Token;

    /// <summary>Moves a queued job to <see cref="OptimizationJobState.Running" />.</summary>
    /// <returns><c>false</c> if the job was canceled while it waited, so the runner must skip it.</returns>
    public bool TryStart()
    {
        lock (_gate)
        {
            if (_state != OptimizationJobState.Queued)
                return false;
            _state = OptimizationJobState.Running;
            _startedAt = DateTimeOffset.UtcNow;
            return true;
        }
    }

    /// <summary>Counts one finished trial; called from the study's per-trial callback.</summary>
    /// <param name="state">The state the trial finished in.</param>
    public void RecordTrial(TrialState state)
    {
        // the callback runs on the nJobs worker threads at the same time, hence Interlocked
        switch (state)
        {
            case TrialState.Complete:
                Interlocked.Increment(ref _complete);
                break;
            case TrialState.Pruned:
                Interlocked.Increment(ref _pruned);
                break;
            case TrialState.Fail:
                Interlocked.Increment(ref _failed);
                break;
        }
    }

    /// <summary>Marks a running job as finished.</summary>
    /// <param name="canceled">Whether it stopped because of a cancellation rather than by running out of trials.</param>
    public void Finish(bool canceled)
    {
        lock (_gate)
        {
            _state = canceled ? OptimizationJobState.Canceled : OptimizationJobState.Completed;
            _finishedAt = DateTimeOffset.UtcNow;
        }
    }

    /// <summary>Marks the job as failed.</summary>
    /// <param name="error">The exception that ended it; only its message is kept, for the API.</param>
    /// <exception cref="ArgumentNullException"><paramref name="error" /> is <c>null</c>.</exception>
    public void Fail(Exception error)
    {
        ArgumentNullException.ThrowIfNull(error);
        lock (_gate)
        {
            _state = OptimizationJobState.Failed;
            _error = error.Message;
            _finishedAt = DateTimeOffset.UtcNow;
        }
    }

    /// <summary>
    ///     Cancels the job. A queued job is canceled at once and never starts. A running job stops scheduling
    ///     trials, and the trials already running finish, so a running job reaches
    ///     <see cref="OptimizationJobState.Canceled" /> shortly afterwards.
    /// </summary>
    /// <returns><c>false</c> if the job had already finished, so there was nothing to cancel.</returns>
    public bool Cancel()
    {
        lock (_gate)
        {
            switch (_state)
            {
                case OptimizationJobState.Queued:
                    _state = OptimizationJobState.Canceled;
                    _finishedAt = DateTimeOffset.UtcNow;
                    break;
                case OptimizationJobState.Running:
                    break;
                default:
                    return false;
            }
        }

        // outside the lock: Cancel runs the token's registrations synchronously, and none of them may wait
        // for a caller that holds _gate
        _cancellation.Cancel();
        return true;
    }

    /// <summary>Takes a consistent snapshot of the job for the API.</summary>
    /// <returns>The job's current state, progress and timestamps.</returns>
    public JobDto ToDto()
    {
        lock (_gate)
        {
            return new JobDto(
                Id,
                StudyName,
                _state,
                NTrials,
                NJobs,
                Volatile.Read(ref _complete),
                Volatile.Read(ref _pruned),
                Volatile.Read(ref _failed),
                QueuedAt,
                _startedAt,
                _finishedAt,
                _error);
        }
    }
}
