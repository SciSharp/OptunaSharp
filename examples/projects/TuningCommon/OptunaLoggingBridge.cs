using System.Collections.Concurrent;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using OptunaSharp.Logging;
// both libraries name their level type LogLevel; the aliases keep every use unambiguous
using MicrosoftLogLevel = Microsoft.Extensions.Logging.LogLevel;
using OptunaLogLevel = OptunaSharp.Logging.LogLevel;

namespace TuningCommon;

/// <summary>
///     Sends OptunaSharp's log records (<c>Trial 3 finished with value: …</c>, sampler warnings, …) to
///     Microsoft.Extensions.Logging, so they reach the host's console, files, OpenTelemetry or any other
///     configured provider instead of OptunaSharp's own stderr handler.
/// </summary>
/// <remarks>
///     <para>
///         OptunaSharp's logging is a port of <c>optuna.logging</c>: one process-wide logger with its own
///         verbosity and handler list, and no dependency on Microsoft.Extensions.Logging. This bridge is an
///         <see cref="ILogHandler" />: while the host runs, it replaces the default stderr handler and forwards
///         every record to an <see cref="ILogger" /> whose category is the record's logger name, for example
///         <c>optuna.study.study</c>. A <c>Logging:LogLevel:optuna</c> entry in configuration therefore filters
///         all of OptunaSharp's output.
///     </para>
///     <para>
///         <b>Process-wide.</b> <see cref="OptunaLogging" /> is static, so the bridge affects the whole process.
///         Register it once per host. Records below OptunaSharp's own verbosity (Info by default; change it with
///         <see cref="OptunaLogging.SetVerbosity" />) never reach the bridge, whatever the
///         Microsoft.Extensions.Logging filters say.
///     </para>
///     <para>
///         <b>Thread-safety.</b> OptunaSharp calls <see cref="Emit" /> from whichever thread logs, including the
///         <c>nJobs</c> worker threads, so the bridge only uses thread-safe types and never throws.
///     </para>
/// </remarks>
public sealed class OptunaLoggingBridge : ILogHandler, IHostedService
{
    private readonly ILoggerFactory _loggerFactory;

    // one ILogger per OptunaSharp logger name; ConcurrentDictionary because Emit runs on many threads
    private readonly ConcurrentDictionary<string, ILogger> _loggers = new(StringComparer.Ordinal);

    /// <summary>Creates the bridge over the host's logger factory.</summary>
    /// <param name="loggerFactory">The factory that creates one <see cref="ILogger" /> per OptunaSharp logger name.</param>
    /// <exception cref="ArgumentNullException"><paramref name="loggerFactory" /> is <c>null</c>.</exception>
    public OptunaLoggingBridge(ILoggerFactory loggerFactory)
    {
        ArgumentNullException.ThrowIfNull(loggerFactory);
        _loggerFactory = loggerFactory;
    }

    /// <summary>
    ///     Starts forwarding: removes OptunaSharp's default stderr handler and adds this bridge, so every record
    ///     is written once, by the host's logging providers.
    /// </summary>
    /// <param name="cancellationToken">Unused; installing the handler cannot block.</param>
    /// <returns>A completed task.</returns>
    public Task StartAsync(CancellationToken cancellationToken)
    {
        OptunaLogging.DisableDefaultHandler();
        OptunaLogging.AddHandler(this);
        return Task.CompletedTask;
    }

    /// <summary>
    ///     Stops forwarding and restores the default stderr handler, so OptunaSharp's messages stay visible
    ///     while the host's providers shut down.
    /// </summary>
    /// <param name="cancellationToken">Unused; removing the handler cannot block.</param>
    /// <returns>A completed task.</returns>
    public Task StopAsync(CancellationToken cancellationToken)
    {
        OptunaLogging.RemoveHandler(this);
        OptunaLogging.EnableDefaultHandler();
        return Task.CompletedTask;
    }

    /// <summary>Forwards one OptunaSharp record to the <see cref="ILogger" /> named after its logger.</summary>
    /// <param name="record">The record OptunaSharp emitted.</param>
    /// <remarks>
    ///     Never throws. OptunaSharp does not catch handler exceptions, so a failing logging provider would
    ///     otherwise fail the trial that happened to log. Python's <c>logging</c> makes the same choice: a
    ///     handler error is reported, never raised into the caller.
    /// </remarks>
    public void Emit(in LogRecord record)
    {
        try
        {
            ILogger logger = _loggers.GetOrAdd(record.LoggerName, _loggerFactory.CreateLogger);

            // the message goes in as a template ARGUMENT, never as the template: OptunaSharp messages contain
            // Python dict reprs such as {'x': 1.5}, whose braces a message template would try to parse
            logger.Log(ToMicrosoftLevel(record.Level), record.Exception, "{OptunaMessage}", record.Message);
        }
        catch (Exception)
        {
            // dropping one record is better than failing the trial that logged it (see remarks)
        }
    }

    /// <summary>
    ///     Maps OptunaSharp's numeric levels, which are Python's <c>logging</c> constants, onto
    ///     <see cref="MicrosoftLogLevel" />.
    /// </summary>
    /// <param name="level">The numeric level, such as <see cref="OptunaLogLevel.Info" /> (20).</param>
    /// <returns>The matching Microsoft.Extensions.Logging level; anything below Debug maps to Trace.</returns>
    internal static MicrosoftLogLevel ToMicrosoftLevel(int level) => level switch
    {
        >= OptunaLogLevel.Critical => MicrosoftLogLevel.Critical,
        >= OptunaLogLevel.Error => MicrosoftLogLevel.Error,
        >= OptunaLogLevel.Warning => MicrosoftLogLevel.Warning,
        >= OptunaLogLevel.Info => MicrosoftLogLevel.Information,
        >= OptunaLogLevel.Debug => MicrosoftLogLevel.Debug,
        _ => MicrosoftLogLevel.Trace,
    };
}

/// <summary>Registers <see cref="OptunaLoggingBridge" /> with a host.</summary>
public static class OptunaLoggingBridgeServiceCollectionExtensions
{
    /// <summary>
    ///     Adds <see cref="OptunaLoggingBridge" /> as a hosted service, so OptunaSharp's log records go to the
    ///     host's logging providers from host start to host stop.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> is <c>null</c>.</exception>
    /// <remarks>
    ///     Hosted services start in registration order. Register the bridge before the services that optimize,
    ///     so that their first trials are already logged through it.
    /// </remarks>
    public static IServiceCollection AddOptunaLoggingBridge(this IServiceCollection services)
    {
        ArgumentNullException.ThrowIfNull(services);
        services.AddHostedService<OptunaLoggingBridge>();
        return services;
    }
}
