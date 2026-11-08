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
///     How the workers of one <c>Study.Optimize(nJobs: …)</c> call reach the study's sampler. With <c>nJobs: 1</c>, and
///     for every <c>Ask</c>/<c>Tell</c> outside <c>Optimize</c>, the study's own sampler serves every trial, whatever is
///     chosen here.
/// </summary>
/// <remarks>
///     Optuna has no counterpart: its <c>n_jobs</c> threads share the study's one sampler object, which is what
///     <see cref="Shared" /> ports. Several samplers sampling in parallel exist in Optuna only as several processes on
///     one storage, and that is what <see cref="PerWorker" /> reproduces inside one process.
/// </remarks>
public enum SamplerConcurrency
{
    /// <summary>
    ///     Every worker samples through the study's one sampler instance, one call at a time (the sampler's monitor,
    ///     THREADING.md §6.3). The default, and Optuna's model: a seeded sampler draws from one generator, and samplers
    ///     that propose from shared state (QMC's sequence, the evolution strategies' generations) never hand two trials
    ///     the same point because of the parallelism. The cost: sampling is serial, so a sampler whose proposals are
    ///     slow caps the throughput at one proposal per sampling time, whatever <c>nJobs</c> is (measured 2026-11-08:
    ///     GPSampler ran 120 trials of a 20 ms objective in 18.7 s at <c>nJobs: 8</c> against 12.1 s at
    ///     <c>nJobs: 1</c>).
    /// </summary>
    Shared = 0,

    /// <summary>
    ///     Each worker samples with an instance of its own: made from the study's sampler (same configuration), reseeded
    ///     from fresh OS entropy before each of its trials as Optuna reseeds a worker, and reused by that worker for
    ///     later trials of the call. The instances share only the storage, so they sample in parallel, exactly like
    ///     separate processes on one storage.
    ///     <para>
    ///         <b>When it pays.</b> Only when sampling, not the objective, limits the run. Measured 2026-11-08 at
    ///         <c>nJobs: 8</c> on 32 cores: a multivariate TPE over 10 parameters (600 trials, 5 ms objective) took
    ///         1.20 s against 1.95 s shared, and a 40-dimensional CMA-ES (2,000 trials, no sleep) 1.46 s against
    ///         3.68 s; a univariate TPE over 3 parameters with a 20 ms objective gained nothing (1.54 s against 1.56 s).
    ///     </para>
    ///     <para>
    ///         <b>Only samplers that produce no duplicate samples this way take part</b>: <c>RandomSampler</c>,
    ///         <c>TPESampler</c> (multivariate or not), <c>CmaEsSampler</c>, <c>VkdCmaSampler</c>, <c>GridSampler</c>,
    ///         and <c>PartialFixedSampler</c> over one of them. Every other sampler keeps <see cref="Shared" /> access,
    ///         one call at a time, because separate instances propose the same point: measured with eight instances on
    ///         one study (three rounds of 64 trials, fewer for the slow model-based samplers), QMC repeated 4–20 points
    ///         a round, the NES samplers 21–29, JANUS and JANUS-DE 49–54, PSO, TuRBO, REMBO, SAASBO and BruteForce 2–32,
    ///         and GPSampler one point in eleven rounds of 24 trials (none shared): instances that see the same
    ///         history optimize the same acquisition function. A sampler written outside OptunaSharp also keeps
    ///         <see cref="Shared" /> access.
    ///     </para>
    ///     <para>
    ///         <b>Footguns.</b> Functions passed to a taking-part sampler (TPE's <c>gamma</c>, <c>weights</c> and
    ///         <c>constraintsFunc</c>, for example) now run on several workers at once, so they must be thread-safe.
    ///         Trials of one call no longer draw from one generator, so a seeded sampler's parallel run is no more
    ///         reproducible than before (it never was, THREADING.md §3.2). The study's own sampler instance is not used
    ///         by the workers of such a call, so state it keeps in memory (TPE's caches, CMA-ES's) is not updated by
    ///         them.
    ///     </para>
    /// </summary>
    PerWorker = 1,
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
///         <b>Snapshot.</b> A call reads <see cref="Executor" />, <see cref="ThreadIdleTimeout" />,
///         <see cref="SamplerConcurrency" /> and <see cref="BlasThreads" /> of the resolved instance once, when it
///         starts, and runs with those values to the end. Changing an instance, or <see cref="Default" />, affects only
///         calls that start afterwards.
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
    private int _samplerConcurrency = (int)OptunaSharp.SamplerConcurrency.Shared;

    // 0 encodes null (the automatic policy): every valid explicit count is at least 1, and an int is read and written
    // atomically, where an int? field is not.
    private int _blasThreads;

    /// <summary>
    ///     Creates options with the defaults: <see cref="TrialExecutorKind.DedicatedThreads" />, a
    ///     <see cref="ThreadIdleTimeout" /> of 30 seconds, <see cref="OptunaSharp.SamplerConcurrency.Shared" /> and
    ///     <see cref="BlasThreads" /> <see langword="null" /> (the automatic policy), the behavior of a study with no
    ///     options set anywhere.
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
    ///     How the call's <c>nJobs</c> workers reach the study's sampler: <see cref="OptunaSharp.SamplerConcurrency.Shared" />
    ///     (the default, Optuna's model: one instance, one call at a time) or
    ///     <see cref="OptunaSharp.SamplerConcurrency.PerWorker" /> (an instance per worker, for the samplers that produce
    ///     no duplicate samples that way). Ignored with <c>nJobs: 1</c>. Read the footguns on
    ///     <see cref="OptunaSharp.SamplerConcurrency.PerWorker" /> before choosing it.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The value is not a defined <see cref="OptunaSharp.SamplerConcurrency" />.</exception>
    public virtual SamplerConcurrency SamplerConcurrency
    {
        get => (SamplerConcurrency)Volatile.Read(ref _samplerConcurrency);
        set
        {
            ValidateSamplerConcurrency(value, nameof(value));
            Volatile.Write(ref _samplerConcurrency, (int)value);
        }
    }

    /// <summary>
    ///     The OpenBLAS thread count for the duration of the call. <see langword="null" /> (the default) is the automatic
    ///     policy: with <c>nJobs: 1</c> OpenBLAS keeps its own setting (one thread per CPU unless configured otherwise,
    ///     as NumPy's), and with any other <c>nJobs</c> it runs single-threaded while the call lasts, so parallel workers
    ///     do not oversubscribe the cores. A positive value is applied for the duration of any call, <c>nJobs: 1</c>
    ///     included. When the call ends the count in effect before it is restored.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         <b>Why.</b> OpenBLAS's threads are one pool for the whole process, sized for one caller. With several
    ///         workers in linear algebra at once (CMA-ES's eigendecomposition, the GP stack's Cholesky) they fight for
    ///         the same cores. Measured 2026-11-08 on 32 cores: 2,000 trials of a 40-dimensional CMA-ES at
    ///         <c>nJobs: 8</c> took 5.22 s with OpenBLAS's own 24 threads and 3.58 s pinned to one, against 4.50 s at
    ///         <c>nJobs: 1</c>; unpinned, the parallel run was the slowest of the three. A GP study showed no difference
    ///         (18.66 s against 18.68 s): its matrices are too small for OpenBLAS to thread. Optuna's own GPSampler
    ///         limits its BLAS and torch threads to one for the same reason (<c>optuna/_gp/thread_limiting.py</c>). A
    ///         parallel run is not reproducible anyway (its workers reseed from OS entropy), so pinning there costs no
    ///         reproducibility; <c>nJobs: 1</c> keeps byte parity with a default NumPy, whose results depend on the
    ///         thread count too.
    ///     </para>
    ///     <para>
    ///         <b>Footguns.</b> The count is process-wide: while a call holds it, every OpenBLAS call in the process runs
    ///         with it, the host's own included, and a call that holds no count (<c>nJobs: 1</c> on the automatic
    ///         policy) runs with whatever another call holds. Concurrent calls take the smallest count any of them asks
    ///         for, and the last to end restores the original. Changing the count goes through NumSharp's OpenBLAS
    ///         threading knob, which also writes <c>OPENBLAS_NUM_THREADS</c> into the process environment and the C
    ///         runtime's copy of it; OptunaSharp removes the process-environment variable again on restore when it was
    ///         not set before (the C runtime's copy keeps the restored count, the one OpenBLAS was already using). The
    ///         memory OpenBLAS commits for its threads at load (measured: 788 MB of private bytes with 24 threads, 50 MB
    ///         with one; the working set stays near 44 MB either way) is not returned: only
    ///         <c>OPENBLAS_NUM_THREADS=1</c> set before the process starts avoids it. Without a loaded OpenBLAS the
    ///         setting does nothing.
    ///     </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">The value is zero or negative; use <see langword="null" /> for the automatic policy.</exception>
    public virtual int? BlasThreads
    {
        get
        {
            int stored = Volatile.Read(ref _blasThreads);
            return stored == 0 ? null : stored;
        }
        set
        {
            ValidateBlasThreads(value, nameof(value));
            Volatile.Write(ref _blasThreads, value ?? 0);
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

    /// <summary>Rejects a value that is not one of the defined sampler concurrency modes.</summary>
    /// <param name="concurrency">The mode to check.</param>
    /// <param name="paramName">The parameter or property name reported in the exception.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="concurrency" /> is not a defined <see cref="OptunaSharp.SamplerConcurrency" />.</exception>
    internal static void ValidateSamplerConcurrency(SamplerConcurrency concurrency, string paramName)
    {
        if (concurrency is not (OptunaSharp.SamplerConcurrency.Shared or OptunaSharp.SamplerConcurrency.PerWorker))
            throw new ArgumentOutOfRangeException(paramName, concurrency,
                "SamplerConcurrency must be SamplerConcurrency.Shared or SamplerConcurrency.PerWorker.");
    }

    /// <summary>Rejects an OpenBLAS thread count that is not <see langword="null" /> or positive.</summary>
    /// <param name="threads">The count to check; <see langword="null" /> is the automatic policy.</param>
    /// <param name="paramName">The parameter or property name reported in the exception.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="threads" /> is zero or negative.</exception>
    internal static void ValidateBlasThreads(int? threads, string paramName)
    {
        if (threads is <= 0)
            throw new ArgumentOutOfRangeException(paramName, threads,
                "BlasThreads must be null (the automatic policy) or a positive thread count.");
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
