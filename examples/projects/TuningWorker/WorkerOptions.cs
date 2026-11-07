using System.ComponentModel.DataAnnotations;

namespace TuningWorker;

/// <summary>
///     The worker's settings, bound from the <c>Worker</c> section of <c>appsettings.json</c>. Override any of
///     them per process on the command line, for example
///     <c>dotnet run --project examples/projects/TuningWorker -- --Worker:Trials=50 --Worker:Jobs=4</c>.
/// </summary>
public sealed class WorkerOptions
{
    /// <summary>The configuration section the options bind from.</summary>
    public const string SectionName = "Worker";

    /// <summary>The host of the TuningService's gRPC storage proxy.</summary>
    [Required]
    public string Host { get; set; } = "localhost";

    /// <summary>The proxy's port; 13000 is Optuna's default and the TuningService's.</summary>
    [Range(1, 65_535)]
    public int Port { get; set; } = 13_000;

    /// <summary>
    ///     The study to join. Every worker that names the same study, and the TuningService's own jobs, work on
    ///     it together; the first one to arrive creates it.
    /// </summary>
    [Required]
    public string StudyName { get; set; } = "distributed";

    /// <summary>How many trials this process runs before it exits.</summary>
    [Range(1, 100_000)]
    public int Trials { get; set; } = 30;

    /// <summary>How many of them it runs at the same time (<c>Optimize(nJobs: …)</c>).</summary>
    [Range(1, 256)]
    public int Jobs { get; set; } = 2;

    /// <summary>How long one simulated training epoch sleeps, in milliseconds.</summary>
    [Range(0, 10_000)]
    public int EpochDelayMilliseconds { get; set; } = 25;

    /// <summary>How long to wait for the proxy to answer before giving up, in seconds.</summary>
    [Range(1, 3_600)]
    public int ConnectTimeoutSeconds { get; set; } = 15;
}
