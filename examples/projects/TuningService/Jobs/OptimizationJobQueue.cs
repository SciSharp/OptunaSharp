using System.Collections.Concurrent;
using System.Threading.Channels;
using Microsoft.Extensions.Options;

namespace TuningService.Jobs;

/// <summary>
///     The hand-off between the HTTP endpoints, which queue optimization jobs, and
///     <see cref="OptimizationJobRunner" />, which runs them one at a time. It also remembers every job, so its
///     status stays readable after it finished.
/// </summary>
/// <remarks>
///     <para>
///         Registered as a singleton: the endpoints and the runner must share the one channel. The channel is
///         bounded by <see cref="TuningServiceOptions.MaxQueuedJobs" />, so a burst of requests is refused with
///         429 instead of growing an unbounded backlog.
///     </para>
///     <para>
///         Finished jobs stay in memory for the life of the process: a few hundred bytes each, which is fine for
///         an example. A long-running service would evict old ones.
///     </para>
/// </remarks>
public sealed class OptimizationJobQueue
{
    private readonly Channel<OptimizationJob> _channel;
    private readonly ConcurrentDictionary<Guid, OptimizationJob> _jobs = new();

    /// <summary>Creates the queue with the capacity the options configure.</summary>
    /// <param name="options">The service options; <see cref="TuningServiceOptions.MaxQueuedJobs" /> is the capacity.</param>
    /// <exception cref="ArgumentNullException"><paramref name="options" /> is <c>null</c>.</exception>
    public OptimizationJobQueue(IOptions<TuningServiceOptions> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        _channel = Channel.CreateBounded<OptimizationJob>(new BoundedChannelOptions(options.Value.MaxQueuedJobs)
        {
            SingleReader = true,   // only the runner reads
            FullMode = BoundedChannelFullMode.Wait,   // makes TryWrite return false when full, never drop a job
        });
    }

    /// <summary>The jobs in queue order, for the runner.</summary>
    public ChannelReader<OptimizationJob> Reader => _channel.Reader;

    /// <summary>Queues a job and remembers it.</summary>
    /// <param name="job">The job to run.</param>
    /// <returns><c>false</c> if the queue is full; the job is then neither queued nor remembered.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="job" /> is <c>null</c>.</exception>
    public bool TryEnqueue(OptimizationJob job)
    {
        ArgumentNullException.ThrowIfNull(job);

        // remember it first, so a status request made the moment the runner picks it up finds it
        _jobs[job.Id] = job;
        if (_channel.Writer.TryWrite(job))
            return true;

        _jobs.TryRemove(job.Id, out _);
        return false;
    }

    /// <summary>Looks a job up by its identifier.</summary>
    /// <param name="id">The job identifier.</param>
    /// <param name="job">The job, when found.</param>
    /// <returns>Whether a job with that identifier was ever queued.</returns>
    public bool TryGet(Guid id, [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out OptimizationJob? job)
        => _jobs.TryGetValue(id, out job);

    /// <summary>Every job ever queued, oldest first.</summary>
    /// <returns>A snapshot of the jobs; later jobs are not added to it.</returns>
    public IReadOnlyList<OptimizationJob> Snapshot()
        => _jobs.Values.OrderBy(j => j.QueuedAt).ToList();
}
