using System.ComponentModel.DataAnnotations;

namespace TuningService;

/// <summary>
///     The service's OptunaSharp settings, bound from the <c>Optuna</c> section of <c>appsettings.json</c>.
///     Like any ASP.NET Core setting, each one can be overridden by an environment variable
///     (<c>Optuna__Sampler=GP</c>) or a command-line argument (<c>--Optuna:Sampler=GP</c>).
/// </summary>
/// <remarks>
///     The values are validated when the app starts (<c>ValidateOnStart</c>), so a typo such as
///     <c>"Sampler": "TPF"</c> stops the app with a clear message instead of failing the first request.
/// </remarks>
public sealed class TuningServiceOptions
{
    /// <summary>The configuration section the options bind from.</summary>
    public const string SectionName = "Optuna";

    /// <summary>
    ///     The Optuna storage URL every study lives in, in the same form <c>optuna.create_study(storage=…)</c>
    ///     takes. The default, <c>sqlite:///tuning.db</c>, is a SQLite file in the current directory that Python
    ///     Optuna (and optuna-dashboard) can open too.
    /// </summary>
    /// <remarks>
    ///     The URL's scheme decides which package serves it, and Core loads that package by name:
    ///     <c>sqlite</c> needs <c>OptunaSharp.Storage.Sqlite</c>, which this project references;
    ///     <c>postgresql</c> and <c>mysql</c> need <c>OptunaSharp.Storage.EFCore</c>, which it does not. A URL
    ///     that no referenced package serves stops the app at startup with Optuna's <c>ImportError</c> text.
    /// </remarks>
    [Required]
    public string Storage { get; set; } = "sqlite:///tuning.db";

    /// <summary>
    ///     The sampler every study uses: <c>TPE</c> (Optuna's default), <c>Random</c>, <c>QMC</c>, <c>GP</c>,
    ///     <c>CmaEs</c> or <c>NSGAII</c>. Changing it changes the algorithm of every study without a code change.
    /// </summary>
    /// <remarks>
    ///     CMA-ES and QMC handle only numbers; they sample the categorical parameters independently and log a
    ///     warning for each, as in Optuna.
    /// </remarks>
    [Required]
    [AllowedValues("TPE", "Random", "QMC", "GP", "CmaEs", "NSGAII")]
    public string Sampler { get; set; } = "TPE";

    /// <summary>
    ///     The sampler seed, or <c>null</c> (the default) to seed each sampler from OS entropy.
    /// </summary>
    /// <remarks>
    ///     <b>Footgun.</b> The container builds a new sampler for every study it serves, and a request-handling
    ///     endpoint builds one per request. With a fixed seed, every one of those samplers draws the same random
    ///     startup points, so the first trials asked over HTTP would all get identical parameters. Set a seed
    ///     only to reproduce one background job.
    /// </remarks>
    public int? SamplerSeed { get; set; }

    /// <summary>How long one simulated training epoch of the built-in objective sleeps, in milliseconds.</summary>
    [Range(0, 10_000)]
    public int EpochDelayMilliseconds { get; set; } = 25;

    /// <summary>The largest <c>nTrials</c> one optimization job may request.</summary>
    [Range(1, 100_000)]
    public int MaxTrialsPerJob { get; set; } = 1_000;

    /// <summary>The largest <c>nJobs</c> (trials run in parallel) one optimization job may request.</summary>
    [Range(1, 256)]
    public int MaxParallelTrials { get; set; } = 16;

    /// <summary>How many optimization jobs may wait in the queue; one more is refused with 429 Too Many Requests.</summary>
    [Range(1, 10_000)]
    public int MaxQueuedJobs { get; set; } = 32;
}
