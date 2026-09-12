using System;
using System.Collections.Generic;
using System.IO;

namespace OptunaSharp.Logging;

// Port of optuna/logging.py — Optuna's own, self-contained verbosity API. This is deliberately
// NOT built on Microsoft.Extensions.Logging: the Core package stays dependency-pure (BCL +
// NumSharp only); the M.E.L adapter is a separate Tier X package that plugs in via AddHandler.

/// <summary>
///     Log levels, numerically identical to Python's <c>logging</c> module constants so that
///     <c>optuna.logging.set_verbosity(optuna.logging.WARNING)</c> ports 1:1.
/// </summary>
public static class LogLevel
{
    /// <summary>Disables emission in this logging implementation.</summary>
    public const int NotSet = 0;
    /// <summary>Detailed diagnostic logging level.</summary>
    public const int Debug = 10;
    /// <summary>Normal study and trial progress logging level.</summary>
    public const int Info = 20;
    /// <summary>Recoverable condition or suspicious configuration logging level.</summary>
    public const int Warning = 30;
    /// <summary>Alias for Warning.</summary>
    public const int Warn = Warning;
    /// <summary>Operation failure logging level.</summary>
    public const int Error = 40;
    /// <summary>Severe failure logging level.</summary>
    public const int Critical = 50;
    /// <summary>Alias for Critical.</summary>
    public const int Fatal = Critical;
}

/// <summary>A single log record passed to <see cref="ILogHandler" /> implementations.</summary>
public readonly struct LogRecord
{
    /// <param name="message">Text to include in the log record.</param>
    /// <summary>Captures a log event and its optional exception.</summary>
    /// <param name="loggerName">Name of the component that emitted the record.</param>
    /// <param name="level">Numeric logging severity, using the LogLevel constants.</param>
    /// <param name="exception">Optional exception retained for handler formatting.</param>
    /// <param name="timestamp">Local timestamp assigned when the record is created.</param>
    public LogRecord(string loggerName, int level, string message, Exception? exception, DateTime timestamp)
    {
        LoggerName = loggerName;
        Level = level;
        Message = message;
        Exception = exception;
        Timestamp = timestamp;
    }

    /// <summary>Gets the name of the emitting logger.</summary>
    public string LoggerName { get; }
    /// <summary>Gets the numeric severity.</summary>
    public int Level { get; }
    /// <summary>Gets the event text.</summary>
    public string Message { get; }
    /// <summary>Gets the associated exception, or null.</summary>
    public Exception? Exception { get; }
    /// <summary>Gets the local event timestamp.</summary>
    public DateTime Timestamp { get; }

    /// <summary>Single-letter level tag, as in Optuna's default formatter (<c>%(levelname)1.1s</c>).</summary>
    public char LevelChar => Level switch
    {
        >= LogLevel.Critical => 'C',
        >= LogLevel.Error => 'E',
        >= LogLevel.Warning => 'W',
        >= LogLevel.Info => 'I',
        _ => 'D',
    };
}

/// <summary>Receives log records emitted through the Optuna logging seam.</summary>
public interface ILogHandler
{
    /// <summary>Receives a log record synchronously.</summary>
    /// <param name="record">Log record to deliver to the handler.</param>
    /// <exception cref="System.Exception">The handler implementation fails; callers do not swallow handler exceptions.</exception>
    void Emit(in LogRecord record);
}

/// <summary>
///     The default handler: writes <c>[I 2021-10-31 05:35:17,232] message</c> lines to stderr,
///     mirroring Optuna's <c>create_default_formatter()</c> (sans colors).
/// </summary>
public sealed class StreamLogHandler : ILogHandler
{
    private readonly TextWriter _writer;

    /// <summary>Creates an Optuna-style text handler using stderr or a supplied writer.</summary>
    public StreamLogHandler() : this(Console.Error) { }

    /// <summary>Creates an Optuna-style text handler using stderr or a supplied writer.</summary>
    /// <param name="writer">Caller-owned text writer; the handler does not dispose it.</param>
    public StreamLogHandler(TextWriter writer)
    {
        _writer = writer;
    }

    /// <summary>Writes the formatted event and optional exception to the configured writer.</summary>
    /// <param name="record">Log record to deliver to the handler.</param>
    /// <exception cref="System.IO.IOException">The writer cannot write the record.</exception>
    /// <exception cref="System.ObjectDisposedException">The configured writer has been disposed.</exception>
    public void Emit(in LogRecord record)
    {
        // Python asctime default format: "2021-10-31 05:35:17,232" (comma before milliseconds).
        var ts = record.Timestamp.ToString("yyyy-MM-dd HH:mm:ss,fff", System.Globalization.CultureInfo.InvariantCulture);
        _writer.WriteLine($"[{record.LevelChar} {ts}] {record.Message}");
        if (record.Exception is not null)
            _writer.WriteLine(record.Exception.ToString());
    }
}

/// <summary>
///     Port of optuna/logging.py — controls the verbosity of OptunaSharp's library-root logger.
/// </summary>
public static class OptunaLogging
{
    private static readonly object _lock = new();
    private static ILogHandler? _defaultHandler;
    private static readonly List<ILogHandler> _handlers = new();
    private static int _level = LogLevel.Info;
    private static bool _configured;

    // Python: optuna's library root logger name.
    /// <summary>Python-compatible root logger name used by OptunaSharp components.</summary>
    internal const string LibraryName = "optuna";

    /// <summary>Initializes the default handler and Info threshold once under the shared lock.</summary>
    private static void ConfigureLibraryRootLogger()
    {
        lock (_lock)
        {
            if (_configured)
                return;
            _defaultHandler = new StreamLogHandler();
            _handlers.Add(_defaultHandler);
            _level = LogLevel.Info;
            _configured = true;
        }
    }

    /// <summary>Clears handler registrations and initialization state for a fresh logging configuration.</summary>
    internal static void ResetLibraryRootLogger()
    {
        lock (_lock)
        {
            if (!_configured)
                return;
            _handlers.Clear();
            _defaultHandler = null;
            _level = LogLevel.NotSet;
            _configured = false;
        }
    }

    /// <summary>
    ///     Return a logger with the specified name.
    ///
    ///     This function is not supposed to be directly accessed by library users.
    /// </summary>
    /// <param name="name">Component logger name; this does not create an independent threshold.</param>
    /// <returns>A named logger sharing the library-wide threshold and handlers.</returns>
    public static Logger GetLogger(string name)
    {
        ConfigureLibraryRootLogger();
        return new Logger(name);
    }

    /// <summary>Return the current level for the OptunaSharp's root logger.</summary>
    /// <returns>Current numeric library-wide severity threshold.</returns>
    public static int GetVerbosity()
    {
        ConfigureLibraryRootLogger();
        return _level;
    }

    /// <summary>Set the level for the OptunaSharp's root logger.</summary>
    /// <param name="verbosity">Library-wide threshold; NotSet disables emission in this implementation.</param>
    public static void SetVerbosity(int verbosity)
    {
        ConfigureLibraryRootLogger();
        lock (_lock)
            _level = verbosity;
    }

    /// <summary>Disable the default handler of the OptunaSharp's root logger.</summary>
    public static void DisableDefaultHandler()
    {
        ConfigureLibraryRootLogger();
        lock (_lock)
        {
            if (_defaultHandler is not null)
                _handlers.Remove(_defaultHandler);
        }
    }

    /// <summary>Enable the default handler of the OptunaSharp's root logger.</summary>
    public static void EnableDefaultHandler()
    {
        ConfigureLibraryRootLogger();
        lock (_lock)
        {
            if (_defaultHandler is not null && !_handlers.Contains(_defaultHandler))
                _handlers.Add(_defaultHandler);
        }
    }

    /// <summary>
    ///     Add a custom handler (the seam external logging adapters — e.g. the Tier X
    ///     Microsoft.Extensions.Logging bridge — plug into).
    /// </summary>
    /// <param name="handler">Handler instance to register or unregister; its lifetime remains caller-owned.</param>
    public static void AddHandler(ILogHandler handler)
    {
        ConfigureLibraryRootLogger();
        lock (_lock)
            _handlers.Add(handler);
    }

    /// <summary>Remove a previously added handler.</summary>
    /// <param name="handler">Handler instance to register or unregister; its lifetime remains caller-owned.</param>
    public static void RemoveHandler(ILogHandler handler)
    {
        ConfigureLibraryRootLogger();
        lock (_lock)
            _handlers.Remove(handler);
    }

    /// <summary>Checks whether the shared verbosity threshold permits this severity.</summary>
    /// <param name="level">Numeric logging severity, using the LogLevel constants.</param>
    /// <returns>True when the severity meets the threshold and logging is not disabled.</returns>
    internal static bool IsEnabledFor(int level)
    {
        ConfigureLibraryRootLogger();
        return level >= _level && _level != LogLevel.NotSet;
    }

    /// <param name="message">Text to include in the log record.</param>
    /// <summary>Dispatches an enabled record to a snapshot of the registered handlers.</summary>
    /// <param name="loggerName">Name of the component that emitted the record.</param>
    /// <param name="level">Numeric logging severity, using the LogLevel constants.</param>
    /// <param name="exception">Optional exception retained for handler formatting.</param>
    /// <exception cref="System.Exception">A handler fails; subsequent handlers are not invoked for this record.</exception>
    internal static void Emit(string loggerName, int level, string message, Exception? exception)
    {
        if (!IsEnabledFor(level))
            return;

        var record = new LogRecord(loggerName, level, message, exception, DateTime.Now);
        ILogHandler[] handlers;
        lock (_lock)
            handlers = _handlers.ToArray();
        // Invoke user handlers outside the configuration lock so handlers can change registration
        // without blocking unrelated configuration access during potentially slow output.
        foreach (var handler in handlers)
            handler.Emit(record);
    }
}

/// <summary>
///     A named logger routed through the shared OptunaSharp root logger, mirroring the subset of
///     Python's <c>logging.Logger</c> that Optuna uses.
/// </summary>
public sealed class Logger
{
    /// <summary>Gets this logger's component name.</summary>
    public string Name { get; }

    /// <param name="name">Component name attached to emitted records.</param>
    /// <summary>Creates a named facade over shared logging configuration.</summary>
    internal Logger(string name)
    {
        Name = name;
    }

    /// <summary>Checks whether the shared verbosity threshold permits this severity.</summary>
    /// <param name="level">Numeric logging severity, using the LogLevel constants.</param>
    /// <returns>True when the severity meets the threshold and logging is not disabled.</returns>
    public bool IsEnabledFor(int level) => OptunaLogging.IsEnabledFor(level);

    /// <param name="message">Text to include in the log record.</param>
    /// <summary>Emits a debug record when permitted by the current verbosity.</summary>
    /// <exception cref="System.Exception">A registered log handler fails while processing the record.</exception>
    public void Debug(string message) => OptunaLogging.Emit(Name, LogLevel.Debug, message, null);

    /// <param name="message">Text to include in the log record.</param>
    /// <summary>Emits a info record when permitted by the current verbosity.</summary>
    /// <exception cref="System.Exception">A registered log handler fails while processing the record.</exception>
    public void Info(string message) => OptunaLogging.Emit(Name, LogLevel.Info, message, null);

    /// <param name="message">Text to include in the log record.</param>
    /// <summary>Emits a warning record when permitted by the current verbosity.</summary>
    /// <exception cref="System.Exception">A registered log handler fails while processing the record.</exception>
    public void Warning(string message) => OptunaLogging.Emit(Name, LogLevel.Warning, message, null);

    /// <param name="message">Text to include in the log record.</param>
    /// <summary>Emits a warning record when permitted by the current verbosity.</summary>
    /// <param name="exception">Optional exception retained for handler formatting.</param>
    /// <exception cref="System.Exception">A registered log handler fails while processing the record.</exception>
    public void Warning(string message, Exception? exception) => OptunaLogging.Emit(Name, LogLevel.Warning, message, exception);

    /// <param name="message">Text to include in the log record.</param>
    /// <summary>Emits a error record when permitted by the current verbosity.</summary>
    /// <exception cref="System.Exception">A registered log handler fails while processing the record.</exception>
    public void Error(string message) => OptunaLogging.Emit(Name, LogLevel.Error, message, null);
}
