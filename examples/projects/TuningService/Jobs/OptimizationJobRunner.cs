using Microsoft.Extensions.Options;
using OptunaSharp;
using OptunaSharp.Extensions;
using TuningCommon;

namespace TuningService.Jobs;

/// <summary>
///     The hosted service that runs queued <see cref="OptimizationJob" />s, one at a time: it builds each job's
///     study through the container's <see cref="IOptunaStudyFactory" />, so the job uses the configured
///     storage, sampler and pruner, and optimizes it with the built-in <see cref="SimulatedTraining" />
///     objective.
/// </summary>
/// <remarks>
///     <para>
///         <b>A scope per job.</b> The factory resolves the sampler and pruner from the provider that resolved
///         it, and they are transient, so each job's study gets its own instances. The scope is disposed only
///         when <c>Optimize</c> has returned, because anything the scope owns must live as long as the study
///         that uses it.
///     </para>
///     <para>
///         <b>A dedicated thread per job.</b> <c>Optimize</c> is synchronous, as in Optuna, and blocks until the
///         job ends. Running it with <see cref="TaskCreationOptions.LongRunning" /> keeps it off the thread pool,
///         which serves the HTTP requests. With <c>nJobs &gt; 1</c> the trials themselves run on OptunaSharp's
///         own worker threads, never on the pool either.
///     </para>
///     <para>
///         <b>Stopping.</b> OptunaSharp has no <see cref="CancellationToken" /> overload, and <c>Study.Stop()</c>
///         works only inside an objective or a callback. So a callback checks the job's token, linked to the
///         host's stopping token, and calls <c>Stop()</c>. The trials in flight then finish, which also lets the
///         app shut down cleanly in the middle of a job.
///     </para>
/// </remarks>
public sealed class OptimizationJobRunner : BackgroundService
{
    private readonly OptimizationJobQueue _queue;
    private readonly IServiceScopeFactory _scopes;
    private readonly IOptions<TuningServiceOptions> _options;
    private readonly ILogger<OptimizationJobRunner> _logger;

    /// <summary>Creates the runner.</summary>
    /// <param name="queue">The queue the job endpoints write to.</param>
    /// <param name="scopes">Creates the per-job scope the study's services are resolved from.</param>
    /// <param name="options">The service options; the runner reads the simulated epoch delay.</param>
    /// <param name="logger">Logs each job's start and end.</param>
    /// <exception cref="ArgumentNullException">Any argument is <c>null</c>.</exception>
    public OptimizationJobRunner(
        OptimizationJobQueue queue,
        IServiceScopeFactory scopes,
        IOptions<TuningServiceOptions> options,
        ILogger<OptimizationJobRunner> logger)
    {
        _queue = queue ?? throw new ArgumentNullException(nameof(queue));
        _scopes = scopes ?? throw new ArgumentNullException(nameof(scopes));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>Reads jobs from the queue until the host stops, and runs each one to its end.</summary>
    /// <param name="stoppingToken">Signaled when the host stops; it also stops the running job (see remarks).</param>
    /// <returns>A task that completes when the host stops.</returns>
    /// <exception cref="OperationCanceledException">
    ///     The host is stopping while the runner waits for a job. <see cref="BackgroundService" /> treats this as a
    ///     normal shutdown.
    /// </exception>
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        await foreach (OptimizationJob job in _queue.Reader.ReadAllAsync(stoppingToken))
        {
            // a job canceled while it waited is already marked Canceled; skip it
            if (!job.TryStart())
                continue;

            // CancellationToken.None: the job must run to its own end (Run stops it through the callback), not
            // be abandoned unobserved when the host stops while it is being scheduled
            await Task.Factory.StartNew(
                () => Run(job, stoppingToken),
                CancellationToken.None,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);
        }
    }

    /// <summary>Optimizes one job's study, on the dedicated thread <see cref="ExecuteAsync" /> started.</summary>
    /// <param name="job">The job, already in the running state.</param>
    /// <param name="stoppingToken">The host's stopping token.</param>
    /// <remarks>
    ///     Never throws: a failure is recorded on the job, where <c>GET /api/jobs/{id}</c> shows it, and logged.
    ///     An exception escaping here would stop the whole host.
    /// </remarks>
    private void Run(OptimizationJob job, CancellationToken stoppingToken)
    {
        using CancellationTokenSource cancellation =
            CancellationTokenSource.CreateLinkedTokenSource(job.CancellationToken, stoppingToken);
        try
        {
            using IServiceScope scope = _scopes.CreateScope();
            IOptunaStudyFactory factory = scope.ServiceProvider.GetRequiredService<IOptunaStudyFactory>();

            // loadIfExists: true attaches to a study made by an earlier job, a REST client, a TuningWorker or
            // Python, and creates it (minimize, the default direction) on first use
            Study study = factory.CreateStudy(job.StudyName, loadIfExists: true);

            Func<Trial, double> objective = SimulatedTraining.CreateObjective(
                workerName: $"service:job-{job.Id.ToString("N")[..8]}",
                epochDelay: TimeSpan.FromMilliseconds(_options.Value.EpochDelayMilliseconds));

            _logger.LogInformation(
                "Job {JobId} started: {Trials} trials on study {Study}, {Parallel} in parallel.",
                job.Id, job.NTrials, job.StudyName, job.NJobs);

            study.Optimize(
                objective,
                nTrials: job.NTrials,
                nJobs: job.NJobs,
                callbacks:
                [
                    // runs after each trial is stored, on the worker thread that ran it
                    (s, trial) =>
                    {
                        job.RecordTrial(trial.State);
                        if (cancellation.IsCancellationRequested)
                            s.Stop();   // stops scheduling; the trials in flight finish
                    },
                ]);

            job.Finish(canceled: cancellation.IsCancellationRequested);
            _logger.LogInformation("Job {JobId} on study {Study} ended: {State}.", job.Id, job.StudyName, job.ToDto().State);
        }
        catch (Exception e)
        {
            job.Fail(e);
            _logger.LogError(e, "Job {JobId} on study {Study} failed.", job.Id, job.StudyName);
        }
    }
}
