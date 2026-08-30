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
    public const int NotSet = 0;
    public const int Debug = 10;
    public const int Info = 20;
    public const int Warning = 30;
    public const int Warn = Warning;
    public const int Error = 40;
    public const int Critical = 50;
    public const int Fatal = Critical;
}

/// <summary>A single log record passed to <see cref="ILogHandler" /> implementations.</summary>
public readonly struct LogRecord
{
    public LogRecord(string loggerName, int level, string message, Exception? exception, DateTime timestamp)
    {
        LoggerName = loggerName;
        Level = level;
        Message = message;
        Exception = exception;
        Timestamp = timestamp;
    }

    public string LoggerName { get; }
    public int Level { get; }
    public string Message { get; }
    public Exception? Exception { get; }
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
    void Emit(in LogRecord record);
}

/// <summary>
///     The default handler: writes <c>[I 2021-10-31 05:35:17,232] message</c> lines to stderr,
///     mirroring Optuna's <c>create_default_formatter()</c> (sans colors).
/// </summary>
public sealed class StreamLogHandler : ILogHandler
{
    private readonly TextWriter _writer;

    public StreamLogHandler() : this(Console.Error) { }

    public StreamLogHandler(TextWriter writer)
    {
        _writer = writer;
    }

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
    internal const string LibraryName = "optuna";

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
    public static Logger GetLogger(string name)
    {
        ConfigureLibraryRootLogger();
        return new Logger(name);
    }

    /// <summary>Return the current level for the OptunaSharp's root logger.</summary>
    public static int GetVerbosity()
    {
        ConfigureLibraryRootLogger();
        return _level;
    }

    /// <summary>Set the level for the OptunaSharp's root logger.</summary>
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
    public static void AddHandler(ILogHandler handler)
    {
        ConfigureLibraryRootLogger();
        lock (_lock)
            _handlers.Add(handler);
    }

    /// <summary>Remove a previously added handler.</summary>
    public static void RemoveHandler(ILogHandler handler)
    {
        ConfigureLibraryRootLogger();
        lock (_lock)
            _handlers.Remove(handler);
    }

    internal static bool IsEnabledFor(int level)
    {
        ConfigureLibraryRootLogger();
        return level >= _level && _level != LogLevel.NotSet;
    }

    internal static void Emit(string loggerName, int level, string message, Exception? exception)
    {
        if (!IsEnabledFor(level))
            return;

        var record = new LogRecord(loggerName, level, message, exception, DateTime.Now);
        ILogHandler[] handlers;
        lock (_lock)
            handlers = _handlers.ToArray();
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
    public string Name { get; }

    internal Logger(string name)
    {
        Name = name;
    }

    public bool IsEnabledFor(int level) => OptunaLogging.IsEnabledFor(level);

    public void Debug(string message) => OptunaLogging.Emit(Name, LogLevel.Debug, message, null);

    public void Info(string message) => OptunaLogging.Emit(Name, LogLevel.Info, message, null);

    public void Warning(string message) => OptunaLogging.Emit(Name, LogLevel.Warning, message, null);

    public void Warning(string message, Exception? exception) => OptunaLogging.Emit(Name, LogLevel.Warning, message, exception);

    public void Error(string message) => OptunaLogging.Emit(Name, LogLevel.Error, message, null);
}
