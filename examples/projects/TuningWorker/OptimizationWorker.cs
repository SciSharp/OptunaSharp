using Microsoft.Extensions.Options;
using OptunaSharp;
using OptunaSharp.Extensions;
using OptunaSharp.Storages.Grpc;
using TuningCommon;

namespace TuningWorker;

/// <summary>
///     The worker's one job: wait for the TuningService's storage proxy, attach to the shared study, run
///     <see cref="WorkerOptions.Trials" /> trials of <see cref="SimulatedTraining" /> on it, report, and stop the
///     host, so the process exits like a batch job.
/// </summary>
/// <remarks>
///     <para>
///         <b>Distributed optimization.</b> The study lives in the TuningService's storage, and this process only
///         reaches it through gRPC. Every worker runs its own sampler, which reads all workers' trials from the
///         shared storage before each suggestion, so several processes (and Python workers) cooperate without
///         talking to each other. TPE's constant-liar default treats the other workers' running trials as bad
///         results, so the workers spread out instead of evaluating the same point.
///     </para>
///     <para>
///         <b>Stopping.</b> The Generic Host turns Ctrl+C into a graceful stop and signals the stopping token. A
///         callback then calls <c>Study.Stop()</c>, the trials in flight finish and are stored, and the host
///         exits. A process killed outright leaves its running trials RUNNING in the storage.
///     </para>
///     <para>
///         <b>Exit code.</b> The process exits with 1 when the worker fails, for example when the proxy cannot
///         be reached, so scripts and orchestrators can tell a failure from a finished batch.
///     </para>
/// </remarks>
public sealed class OptimizationWorker : BackgroundService
{
    private readonly IServiceScopeFactory _scopes;
    private readonly IOptions<WorkerOptions> _options;
    private readonly IHostApplicationLifetime _lifetime;
    private readonly ILogger<OptimizationWorker> _logger;

    /// <summary>Creates the worker.</summary>
    /// <param name="scopes">Creates the scope the study's services are resolved from.</param>
    /// <param name="options">The worker settings.</param>
    /// <param name="lifetime">Stops the host when the batch is done.</param>
    /// <param name="logger">Logs the summary and any failure.</param>
    /// <exception cref="ArgumentNullException">Any argument is <c>null</c>.</exception>
    public OptimizationWorker(
        IServiceScopeFactory scopes,
        IOptions<WorkerOptions> options,
        IHostApplicationLifetime lifetime,
        ILogger<OptimizationWorker> logger)
    {
        _scopes = scopes ?? throw new ArgumentNullException(nameof(scopes));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _lifetime = lifetime ?? throw new ArgumentNullException(nameof(lifetime));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>Runs the batch on a dedicated thread, then stops the host.</summary>
    /// <param name="stoppingToken">Signaled by Ctrl+C or by the host stopping.</param>
    /// <returns>A task that completes when the batch is done.</returns>
    /// <remarks>Never throws: a failure is logged and sets the process exit code to 1.</remarks>
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        try
        {
            // Optimize blocks until the batch ends, so it gets a thread of its own rather than a pool thread;
            // CancellationToken.None because the batch is stopped through Study.Stop(), not abandoned
            await Task.Factory.StartNew(
                () => Optimize(stoppingToken),
                CancellationToken.None,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);
        }
        catch (TimeoutException e)
        {
            WorkerOptions options = _options.Value;
            _logger.LogError(
                e,
                "No storage proxy answered at {Host}:{Port} within {Seconds} s. Start the TuningService first.",
                options.Host, options.Port, options.ConnectTimeoutSeconds);
            Environment.ExitCode = 1;
        }
        catch (Exception e)
        {
            _logger.LogError(e, "The worker failed.");
            Environment.ExitCode = 1;
        }
        finally
        {
            _lifetime.StopApplication();
        }
    }

    /// <summary>Attaches to the shared study and runs this process's trials on it.</summary>
    /// <param name="stoppingToken">Checked after every trial; once signaled, the run stops scheduling trials.</param>
    /// <exception cref="TimeoutException">The storage proxy did not answer within the configured timeout.</exception>
    /// <exception cref="Exception">The storage or the objective failed; the exception propagates unchanged.</exception>
    private void Optimize(CancellationToken stoppingToken)
    {
        WorkerOptions options = _options.Value;

        // the scope outlives the study: everything the study resolved from it stays alive until Optimize returns
        using IServiceScope scope = _scopes.CreateScope();
        IServiceProvider services = scope.ServiceProvider;

        // the client registration exposes the concrete GrpcStorageProxy next to BaseStorage. Wait for the
        // server first (Optuna's wait_server_ready), so a worker started before the service gets a clear error
        services.GetRequiredService<GrpcStorageProxy>()
            .WaitServerReady(TimeSpan.FromSeconds(options.ConnectTimeoutSeconds));

        // no arguments: the factory's defaults name the shared study and attach to it if it exists
        Study study = services.GetRequiredService<IOptunaStudyFactory>().CreateStudy();

        string workerName = $"worker:{Environment.MachineName}:{Environment.ProcessId}";
        int finished = 0;
        study.Optimize(
            SimulatedTraining.CreateObjective(workerName, TimeSpan.FromMilliseconds(options.EpochDelayMilliseconds)),
            nTrials: options.Trials,
            nJobs: options.Jobs,
            callbacks:
            [
                // runs after each trial is stored, on the thread that ran it; nJobs threads run it concurrently
                (s, _) =>
                {
                    Interlocked.Increment(ref finished);
                    if (stoppingToken.IsCancellationRequested)
                        s.Stop();
                },
            ]);

        // the best trial may well be another worker's: the study holds every worker's trials
        if (study.GetTrials(deepcopy: false, states: [TrialState.Complete]).Count == 0)
        {
            _logger.LogInformation("{Worker} ran {Finished} trials of study {Study}; none completed yet.",
                workerName, finished, study.StudyName);
            return;
        }

        FrozenTrial best = study.BestTrial;
        _logger.LogInformation(
            "{Worker} ran {Finished} trials of study {Study}, which now has {Total}. Best: trial {Number}, loss {Value}, from {BestWorker}.",
            workerName,
            finished,
            study.StudyName,
            study.GetTrials(deepcopy: false).Count,
            best.Number,
            best.Value,
            best.UserAttrs.GetValueOrDefault(SimulatedTraining.WorkerAttribute));
    }
}
