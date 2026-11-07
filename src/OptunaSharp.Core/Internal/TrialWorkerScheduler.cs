using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace OptunaSharp.Internal;

/// <summary>
///     The threads that run <c>Optimize(nJobs)</c> trials under <see cref="TrialExecutorKind.DedicatedThreads" />, the
///     default executor: OptunaSharp's own dedicated threads, never .NET thread-pool threads. They are created on
///     demand, reused by every later trial and <c>Optimize</c> call, and retired after <see cref="IdleTimeout" />
///     without work. This is the port of Python's <c>ThreadPoolExecutor</c> thread reuse. The opt-in
///     <see cref="TrialExecutorKind.ThreadPool" /> executor does not use this class.
/// </summary>
/// <remarks>
///     <para>Why not a new thread per trial: creating a thread costs about 100 µs, serialized on the submit loop, which
///     capped <c>nJobs</c> near 10,000 trials/s whatever the trials did. Cheap trials at <c>nJobs: 8</c> ran slower than
///     sequential ones (review item P3-4).</para>
///     <para>Why not the .NET thread pool: a worker blocks for most of a trial, on the storage lock, I/O or the
///     objective. With <c>nJobs</c> above the pool's minimum, every pool thread blocked, and StackExchange.Redis, which
///     completes even synchronous calls on the pool, timed out. <c>Optimize(nJobs: 96)</c> on a Redis journal died after
///     88 trials. The pool also delayed the host's own work by up to 58 s.</para>
///     <para>THREADING.md §3.6 measured this design against the alternatives (2026-10-29). It meets all five
///     properties a replacement had to keep:</para>
///     <list type="number">
///         <item>Python's loop semantics: only scheduling changes, <c>Optimize</c>'s submit/wait loop is untouched;</item>
///         <item>clean per-thread state: each worker restores <c>InOptimizeLoop</c> and drops its trial cache;</item>
///         <item><c>nJobs</c> trials in flight: a queued trial never waits while fewer workers than queued trials are
///         awake;</item>
///         <item>no parked pool threads;</item>
///         <item>the pool stays free for the host.</item>
///     </list>
///     <para>Trials are never inlined onto a waiting thread. <c>Optimize</c> waits on its workers' tasks, and the TPL
///     would otherwise run a not-yet-started trial on the caller's thread.</para>
/// </remarks>
internal sealed class TrialWorkerScheduler : TaskScheduler
{
    // One scheduler per distinct idle timeout, for the life of the process. Declared before Shared, which reads it
    // during the same static initialization. An entry costs one small object once its threads have retired.
    private static readonly ConcurrentDictionary<TimeSpan, TrialWorkerScheduler> _byIdleTimeout = new();

    /// <summary>
    ///     The process-wide scheduler of every <c>Optimize(nJobs)</c> call on dedicated threads with the default idle
    ///     timeout (30 s). Its threads outlive one call, so a series of small calls does not pay for thread creation
    ///     each time.
    /// </summary>
    internal static readonly TrialWorkerScheduler Shared = For(TrialExecutorOptions.DefaultThreadIdleTimeout);

    /// <summary>
    ///     The process-wide scheduler for <paramref name="idleTimeout" />: calls with equal timeouts share one set of
    ///     worker threads, so reuse across studies and calls keeps working when an application sets its own
    ///     <see cref="TrialExecutorOptions.ThreadIdleTimeout" />.
    /// </summary>
    /// <param name="idleTimeout">An idle timeout already validated by <see cref="TrialExecutorOptions" />.</param>
    /// <returns>The scheduler of that timeout, created on first use.</returns>
    internal static TrialWorkerScheduler For(TimeSpan idleTimeout)
        // A racing first use may build two schedulers; GetOrAdd keeps one, and the loser has started no thread yet.
        => _byIdleTimeout.GetOrAdd(idleTimeout, static timeout => new TrialWorkerScheduler(timeout));

    private readonly object _gate = new();
    private readonly Queue<Task> _queue = new();
    private int _threads;
    private int _waiting;
    private long _threadsStarted;

    /// <summary>Creates a scheduler whose idle threads retire after <paramref name="idleTimeout" />.</summary>
    /// <param name="idleTimeout">How long a worker waits for a new trial before its thread ends.</param>
    internal TrialWorkerScheduler(TimeSpan idleTimeout)
    {
        IdleTimeout = idleTimeout;
    }

    /// <summary>How long an idle worker thread waits for the next trial before it ends.</summary>
    internal TimeSpan IdleTimeout { get; }

    /// <summary>The number of worker threads alive now, busy or idle; for tests and diagnostics.</summary>
    internal int ThreadCount
    {
        get
        {
            lock (_gate)
                return _threads;
        }
    }

    /// <summary>
    ///     How many worker threads this scheduler has started since it was created; only grows. Tests read its change
    ///     across a call to check that threads are reused (a call starts at most about <c>nJobs</c> of them, not one per
    ///     trial).
    /// </summary>
    internal long ThreadsStarted
    {
        get
        {
            lock (_gate)
                return _threadsStarted;
        }
    }

    /// <summary>Workers run as many trials at once as are queued; <c>Optimize</c> itself bounds that by <c>nJobs</c>.</summary>
    public override int MaximumConcurrencyLevel => int.MaxValue;

    /// <summary>
    ///     Queues a trial and makes sure a worker will take it: an idle worker is woken, or, when every awake worker is
    ///     already claimed by an earlier queued trial, a new thread starts.
    /// </summary>
    /// <param name="task">The worker task to run; it is executed exactly once by one worker thread.</param>
    protected override void QueueTask(Task task)
    {
        bool startThread;
        lock (_gate)
        {
            _queue.Enqueue(task);
            // `_waiting` counts workers asleep in Monitor.Wait, or woken but not yet back under the lock: each of them
            // will take one queued trial. Fewer of them than queued trials means this trial would wait for a busy
            // worker, which breaks "nJobs trials in flight", so it gets a thread of its own. Over-starting (a busy
            // worker finishes first and takes the trial) only leaves an idle thread that retires after IdleTimeout.
            startThread = _waiting < _queue.Count;
            if (startThread)
            {
                _threads++;
                _threadsStarted++;
            }
            else
            {
                Monitor.Pulse(_gate);
            }
        }

        if (startThread)
        {
            // Started outside the lock: thread creation takes about 100 µs and other submitters must not wait on it.
            var thread = new Thread(WorkerLoop)
            {
                IsBackground = true,   // an idle worker must never keep the process alive
                Name = "OptunaSharp trial worker",
            };
            thread.Start();
        }
    }

    /// <summary>
    ///     A worker thread: runs queued trials one after another, and ends after <see cref="IdleTimeout" /> without one.
    /// </summary>
    private void WorkerLoop()
    {
        while (true)
        {
            Task task;
            lock (_gate)
            {
                while (_queue.Count == 0)
                {
                    _waiting++;
                    bool signalled = Monitor.Wait(_gate, IdleTimeout);
                    _waiting--;
                    // A Pulse can race the timeout. A timed-out worker that finds work still takes it, because the
                    // submitter counted it as awake and started no thread for that trial; only an empty queue retires
                    // it.
                    if (!signalled && _queue.Count == 0)
                    {
                        _threads--;
                        return;
                    }
                }
                task = _queue.Dequeue();
            }

            // The task captures its own exception; nothing a trial throws can end the worker.
            TryExecuteTask(task);
        }
    }

    /// <summary>
    ///     Refuses to run a trial on the calling thread. <c>Optimize</c> waits on its workers' tasks, and inlining would
    ///     run a trial on the caller's thread, outside the worker accounting and with the caller's thread-local state.
    /// </summary>
    /// <param name="task">The task the TPL offers to run inline.</param>
    /// <param name="taskWasPreviouslyQueued">Whether the task is already in this scheduler's queue.</param>
    /// <returns>Always false: every trial runs on a worker thread.</returns>
    protected override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued) => false;

    /// <summary>The trials queued but not yet taken by a worker, for the debugger's task views.</summary>
    /// <returns>A snapshot of the queue.</returns>
    protected override IEnumerable<Task> GetScheduledTasks()
    {
        lock (_gate)
            return _queue.ToArray();
    }
}
