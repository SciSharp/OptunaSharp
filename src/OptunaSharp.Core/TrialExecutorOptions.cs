using System;
using System.Globalization;
using System.Threading;

namespace OptunaSharp;

/// <summary>
///     Which threads run the trials of <c>Study.Optimize(nJobs: …)</c> when <c>nJobs</c> is not 1. With
///     <c>nJobs: 1</c> every trial runs on the calling thread, whatever is chosen here.
/// </summary>
/// <remarks>
///     Optuna has no counterpart: Python always runs <c>n_jobs</c> on a <c>ThreadPoolExecutor</c> of its own threads,
///     which is what <see cref="DedicatedThreads" /> ports. <c>nJobs</c> keeps Python's meaning on both kinds.
/// </remarks>
public enum TrialExecutorKind
{
    /// <summary>
    ///     OptunaSharp's own dedicated background threads, never .NET thread-pool threads: created on demand, reused
    ///     by later trials and later <c>Optimize</c> calls, and retired after
    ///     <see cref="TrialExecutorOptions.ThreadIdleTimeout" /> without work. The default, and the safe choice for
    ///     any <c>nJobs</c>: <c>nJobs</c> trials really run at once, and the pool stays free for the host and for the
    ///     storage clients that complete their calls on it.
    /// </summary>
    DedicatedThreads = 0,

    /// <summary>
    ///     The .NET thread pool, shared with the rest of the process.
    ///     <para>
    ///         <b>Footgun.</b> A trial blocks its thread for the whole trial (the storage's lock, network I/O, the
    ///         objective). Once <c>nJobs</c> reaches the pool's minimum worker count (<c>ThreadPool.GetMinThreads</c>,
    ///         the processor count by default), the pool adds threads only about once a second, so fewer than
    ///         <c>nJobs</c> trials run at once. StackExchange.Redis and gRPC calls, which complete on the pool, then
    ///         stall or time out, and the host's own work waits behind queued trials. Measured on 32 cores: Redis
    ///         journals timed out from about 40 workers, a gRPC study took 54.5 s instead of 0.6 s at 96, and at most
    ///         34 trials ran at once (THREADING.md §3.6, §7.2). OptunaSharp does not guard against this.
    ///     </para>
    ///     <para>
    ///         <b>If you hit thread-pool starvation</b>, use <see cref="DedicatedThreads" />, which reuses OptunaSharp's
    ///         own threads, or raise the pool's minimum with <c>ThreadPool.SetMinThreads</c> to at least <c>nJobs</c>
    ///         plus what the rest of the process needs.
    ///     </para>
    /// </summary>
    ThreadPool = 1,
}

/// <summary>
///     Options for the executor that runs the trials of <c>Study.Optimize(nJobs: …)</c>. Optuna has no counterpart;
///     with no options set anywhere, trials run on OptunaSharp's reused dedicated threads, as before these options
///     existed.
/// </summary>
/// <remarks>
///     <para>
///         <b>Four levels, the first that is set wins</b>, resolved when each <c>Optimize</c> call starts:
///     </para>
///     <list type="number">
///         <item>the <c>executorOptions</c> argument of that <c>Optimize</c> call;</item>
///         <item><see cref="Study.ExecutorOptions" />, a value set on the study;</item>
///         <item>the DI default, <see cref="Extensions.OptunaStudyOptions.DefaultExecutorOptions" />: the DI
///         study factory writes it into <see cref="Study.ExecutorOptions" /> of every study it builds, so a value set
///         on the study afterwards replaces it;</item>
///         <item><see cref="Default" />, the process-wide default.</item>
///     </list>
///     <para>
///         <b>Snapshot.</b> A call reads <see cref="Executor" /> and <see cref="ThreadIdleTimeout" /> of the resolved
///         instance once, when it starts, and runs with those values to the end. Changing an instance, or
///         <see cref="Default" />, affects only calls that start afterwards.
///     </para>
///     <para>
///         <b>Inheritable.</b> The class is not sealed and its properties are virtual. A derived class may compute
///         them; each getter is called once per <c>Optimize</c> call, and the values it returns are validated then,
///         as the setters validate theirs.
///     </para>
///     <para>
///         <b>Sharing.</b> One instance may serve any number of studies and threads. Its properties are read and
///         written atomically, but a change made while another thread starts an <c>Optimize</c> call can reach that
///         call or not.
///     </para>
/// </remarks>
public class TrialExecutorOptions
{
    /// <summary>The idle time after which a dedicated worker thread ends unless set otherwise: 30 seconds.</summary>
    internal static readonly TimeSpan DefaultThreadIdleTimeout = TimeSpan.FromSeconds(30);

    private static TrialExecutorOptions _default = new();

    // Stored as ticks so a 32-bit process reads and writes the 64-bit value atomically (Volatile.Read/Write on a
    // long are atomic everywhere; a TimeSpan field is not).
    private long _threadIdleTimeoutTicks = DefaultThreadIdleTimeout.Ticks;
    private int _executor = (int)TrialExecutorKind.DedicatedThreads;

    /// <summary>
    ///     Creates options with the defaults: <see cref="TrialExecutorKind.DedicatedThreads" /> and a
    ///     <see cref="ThreadIdleTimeout" /> of 30 seconds, the behavior of a study with no options set anywhere.
    /// </summary>
    public TrialExecutorOptions()
    {
    }

    /// <summary>
    ///     The threads that run the trials: <see cref="TrialExecutorKind.DedicatedThreads" /> (the default) or
    ///     <see cref="TrialExecutorKind.ThreadPool" />. Read the footgun on <see cref="TrialExecutorKind.ThreadPool" />
    ///     before choosing the pool for a large <c>nJobs</c>.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The value is not a defined <see cref="TrialExecutorKind" />.</exception>
    public virtual TrialExecutorKind Executor
    {
        get => (TrialExecutorKind)Volatile.Read(ref _executor);
        set
        {
            ValidateExecutor(value, nameof(value));
            Volatile.Write(ref _executor, (int)value);
        }
    }

    /// <summary>
    ///     With <see cref="TrialExecutorKind.DedicatedThreads" />, how long an idle worker thread waits for the next
    ///     trial before it ends; default 30 seconds. A longer value keeps threads warm for an application that calls
    ///     <c>Optimize</c> often; a shorter one frees them sooner. <see cref="Timeout.InfiniteTimeSpan" /> keeps idle
    ///     workers for the life of the process (they are background threads, so they never keep it alive). The
    ///     thread pool ignores this value: it manages its own threads.
    /// </summary>
    /// <remarks>
    ///     Calls with the same timeout share one set of worker threads, so reuse across studies and calls keeps
    ///     working; a call with a different timeout gets a set of its own.
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    ///     The value is zero or negative (other than <see cref="Timeout.InfiniteTimeSpan" />), or longer than
    ///     <see cref="int.MaxValue" /> milliseconds, the longest wait <see cref="Monitor.Wait(object, TimeSpan)" />
    ///     accepts.
    /// </exception>
    public virtual TimeSpan ThreadIdleTimeout
    {
        get => new(Volatile.Read(ref _threadIdleTimeoutTicks));
        set
        {
            ValidateThreadIdleTimeout(value, nameof(value));
            Volatile.Write(ref _threadIdleTimeoutTicks, value.Ticks);
        }
    }

    /// <summary>
    ///     The process-wide default, used by every <c>Optimize</c> call whose argument and study leave the executor
    ///     unset. Starts as a new <see cref="TrialExecutorOptions" />: <see cref="TrialExecutorKind.DedicatedThreads" />.
    /// </summary>
    /// <remarks>
    ///     Read at the start of each <c>Optimize</c> call, so a new value reaches every study that never chose its own,
    ///     including the study <c>OptunaSearchCV</c> creates inside itself. Prefer assigning a new instance over
    ///     changing the shared one: a library or test that also uses this process sees either change.
    /// </remarks>
    /// <exception cref="ArgumentNullException">The value assigned is <see langword="null" />.</exception>
    public static TrialExecutorOptions Default
    {
        get => Volatile.Read(ref _default);
        set => Volatile.Write(ref _default, value ?? throw new ArgumentNullException(nameof(value)));
    }

    /// <summary>Rejects a value that is not one of the defined executor kinds.</summary>
    /// <param name="executor">The executor kind to check.</param>
    /// <param name="paramName">The parameter or property name reported in the exception.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="executor" /> is not a defined <see cref="TrialExecutorKind" />.</exception>
    internal static void ValidateExecutor(TrialExecutorKind executor, string paramName)
    {
        if (executor is not (TrialExecutorKind.DedicatedThreads or TrialExecutorKind.ThreadPool))
            throw new ArgumentOutOfRangeException(paramName, executor,
                "Executor must be TrialExecutorKind.DedicatedThreads or TrialExecutorKind.ThreadPool.");
    }

    /// <summary>Rejects an idle timeout a worker thread cannot wait for.</summary>
    /// <param name="timeout">The idle timeout to check.</param>
    /// <param name="paramName">The parameter or property name reported in the exception.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    ///     <paramref name="timeout" /> is zero or negative (other than <see cref="Timeout.InfiniteTimeSpan" />), or
    ///     longer than <see cref="int.MaxValue" /> milliseconds.
    /// </exception>
    internal static void ValidateThreadIdleTimeout(TimeSpan timeout, string paramName)
    {
        if (timeout == Timeout.InfiniteTimeSpan)
            return;
        // Zero would retire every worker the moment it finished a trial, so each trial would pay for a new thread
        // again (review item P3-4); the upper bound is Monitor.Wait's own limit.
        if (timeout <= TimeSpan.Zero || timeout.TotalMilliseconds > int.MaxValue)
            throw new ArgumentOutOfRangeException(paramName, timeout, string.Format(CultureInfo.InvariantCulture,
                "ThreadIdleTimeout must be positive and at most {0} ms, or Timeout.InfiniteTimeSpan to keep idle " +
                "trial threads.", int.MaxValue));
    }
}
