using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;

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
/// <remarks>
///     <para>
///         <b>One record at a time per handler.</b> OptunaSharp calls <see cref="Emit" /> of one handler instance for
///         one record at a time, under a lock of that instance's own, as Python's <c>logging.Handler.handle</c> holds
///         the handler's <c>lock</c> around <c>emit</c>. A handler therefore needs no lock of its own for what only
///         <see cref="Emit" /> touches; state it shares with other code still needs one. Different handlers run at the
///         same time, on whichever threads log.
///     </para>
///     <para>
///         <b>Footguns.</b> <see cref="Emit" /> runs on the thread that logged, which may hold any OptunaSharp lock,
///         storage monitors included, and other threads wait for the handler meanwhile. So it must be quick, must not
///         throw (the exception surfaces in the operation that logged), must not call OptunaSharp (a record it logs
///         would re-enter this handler on the same thread), and must not take the Python GIL or wait for another
///         thread that does (THREADING.md §5.3).
///     </para>
/// </remarks>
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
/// <remarks>
///     Like Python's <c>logging.StreamHandler</c>, the parameterless constructor binds to the standard
///     error stream <i>current at construction</i> (<see cref="Console.Error" />, Python's
///     <c>sys.stderr</c>), not at each write. Footgun: if the library's default handler is created while
///     <see cref="Console.Error" /> is temporarily redirected (a test capturing stderr, a host swapping it),
///     it keeps writing to that redirect afterwards. If the redirect is later disposed, writes fail — and,
///     exactly as in Python, the failure is reported on the current stderr ("--- Logging error ---")
///     instead of propagating, so a broken log stream never fails the optimization that was logging.
///     <para>
///         <b>Threads.</b> Each record is one write of its line and, when it has one, its exception, followed by a
///         flush, as Python's <c>StreamHandler.emit</c> does since issue 35046; OptunaSharp emits one record at a time
///         per handler (<see cref="ILogHandler" />). So records from parallel trials never interleave, even over a
///         <see cref="TextWriter" /> that is not synchronized, as long as only this handler writes to it. A writer
///         shared with other code still needs <see cref="TextWriter.Synchronized" />.
///     </para>
/// </remarks>
public sealed class StreamLogHandler : ILogHandler
{
    private readonly TextWriter _writer;

    /// <summary>
    ///     Creates the Optuna-style text handler over the standard error stream as it is <i>now</i>
    ///     (Python's <c>StreamHandler()</c> capturing <c>sys.stderr</c>); a later
    ///     <see cref="Console.SetError" /> does not move it.
    /// </summary>
    public StreamLogHandler() : this(Console.Error) { }

    /// <summary>Creates an Optuna-style text handler over a supplied writer.</summary>
    /// <param name="writer">Caller-owned text writer; the handler does not dispose it.</param>
    public StreamLogHandler(TextWriter writer)
    {
        _writer = writer;
    }

    /// <summary>
    ///     Writes the formatted event and optional exception to the configured writer in one write, then flushes it.
    ///     Never throws: a write failure (a disposed or broken writer) is reported on the current standard error the
    ///     way Python's <c>logging.Handler.handleError</c> reports it, and the record is dropped.
    /// </summary>
    /// <param name="record">Log record to deliver to the handler.</param>
    public void Emit(in LogRecord record)
    {
        try
        {
            // Python asctime default format: "2021-10-31 05:35:17,232" (comma before milliseconds).
            var ts = record.Timestamp.ToString("yyyy-MM-dd HH:mm:ss,fff", System.Globalization.CultureInfo.InvariantCulture);
            // Python's StreamHandler.emit writes `msg + terminator` in one stream.write (issue 35046 merged its two
            // writes), the formatted traceback included. One Write call matters here too: Console.Error locks per
            // call, so a record written in two calls let another thread's line land between its line and its
            // exception. The text is unchanged: the line and the exception each end with the writer's NewLine.
            string newLine = _writer.NewLine;
            string text = record.Exception is null
                ? $"[{record.LevelChar} {ts}] {record.Message}{newLine}"
                : $"[{record.LevelChar} {ts}] {record.Message}{newLine}{record.Exception}{newLine}";
            _writer.Write(text);
            // Python's emit flushes after every record (StreamHandler.flush), so a record reaches a file or a pipe
            // before the next trial starts; a failed flush is a write failure like any other.
            _writer.Flush();
        }
        catch (Exception error)
        {
            // Python's StreamHandler.emit: `except RecursionError: raise; except Exception:
            // self.handleError(record)`. .NET's stack overflow cannot be caught at all, so every
            // catchable failure takes the handleError path — the caller (Study.Optimize, Ask, …) keeps
            // running exactly as it does under Optuna.
            HandleError(record, error);
        }
    }

    /// <summary>
    ///     Port of <c>logging.Handler.handleError</c> (with <c>logging.raiseExceptions</c> at its default,
    ///     <c>True</c>): writes <c>--- Logging error ---</c>, the failure, the calling context
    ///     (<c>Call stack:</c>, logging frames skipped) and <c>Message: …</c> / <c>Arguments: ()</c> to the
    ///     <i>current</i> standard error, then swallows the failure.
    /// </summary>
    /// <remarks>
    ///     The report is best-effort. Python ignores an <c>OSError</c> while writing it; this port also
    ///     ignores <see cref="ObjectDisposedException" /> (the .NET form of writing to a closed
    ///     <c>sys.stderr</c>), so a disposed standard error can never turn a logging hiccup back into an
    ///     exception for the caller.
    /// </remarks>
    /// <param name="record">The record whose emission failed; its message is echoed as Python's <c>%r</c>.</param>
    /// <param name="error">The exception the write raised.</param>
    private static void HandleError(in LogRecord record, Exception error)
    {
        try
        {
            var report = new System.Text.StringBuilder();
            report.Append("--- Logging error ---\n");
            report.Append(error).Append('\n');                 // traceback.print_exception(exc)
            report.Append("Call stack:\n");
            report.Append(CallingContext()).Append('\n');      // traceback.print_stack(first non-logging frame)
            report.Append("Message: ").Append(Internal.PythonRepr.StringRepr(record.Message)).Append('\n');
            // OptunaSharp records carry pre-formatted messages, i.e. Python's record.args == ().
            report.Append("Arguments: ()\n");
            Console.Error.Write(report.ToString());
        }
        catch (Exception e) when (e is IOException or ObjectDisposedException)
        {
            // Python: `except OSError: pass` (issue 5971) — reporting a logging error must not fail.
        }
    }

    /// <summary>
    ///     The stack of the code that logged, with the logging machinery's own frames skipped — Python's
    ///     <c>handleError</c> walks <c>f_back</c> "until we're out of logging, so as to print the calling
    ///     context".
    /// </summary>
    /// <returns>The calling context in <see cref="System.Diagnostics.StackTrace" /> format, or the full stack when every frame is a logging frame.</returns>
    private static string CallingContext()
    {
        var full = new System.Diagnostics.StackTrace(fNeedFileInfo: true);
        System.Diagnostics.StackFrame[] frames = full.GetFrames();
        string loggingNamespace = typeof(StreamLogHandler).Namespace!;
        // Skip the handler/dispatcher/Logger frames (all in OptunaSharp.Logging) to land on the caller.
        int skip = 0;
        while (skip < frames.Length && frames[skip].GetMethod()?.DeclaringType?.Namespace == loggingNamespace)
            skip++;
        return skip < frames.Length
            ? new System.Diagnostics.StackTrace(skip, fNeedFileInfo: true).ToString().TrimEnd()
            : full.ToString().TrimEnd();
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

    // Each handler's own lock, Python's Handler.lock (created with the handler; Handler.handle holds it around emit).
    // Keyed by the handler instance and created on its first record, so the lock lives exactly as long as the handler:
    // a handler removed and added again while another thread is still in its Emit keeps the same lock, and one
    // registered handler never runs two records at once. Locks after every other OptunaSharp lock (THREADING.md §5.2).
    private static readonly ConditionalWeakTable<ILogHandler, object> _handlerLocks = new();

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
    ///     Microsoft.Extensions.Logging bridge — plug into). A handler already registered is not added again, so it
    ///     receives each record once, as Python's <c>Logger.addHandler</c> ensures.
    /// </summary>
    /// <param name="handler">Handler instance to register or unregister; its lifetime remains caller-owned.</param>
    public static void AddHandler(ILogHandler handler)
    {
        ConfigureLibraryRootLogger();
        lock (_lock)
        {
            // Python's addHandler: `if not (hdlr in self.handlers): self.handlers.append(hdlr)`. Contains is the
            // port of `in` (Equals, which is identity unless the handler overrides it, as `==` is in Python).
            if (!_handlers.Contains(handler))
                _handlers.Add(handler);
        }
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
    /// <summary>
    ///     Dispatches an enabled record to a snapshot of the registered handlers, each under its own lock, so a handler
    ///     receives one record at a time (Python's <c>Handler.handle</c>).
    /// </summary>
    /// <param name="loggerName">Name of the component that emitted the record.</param>
    /// <param name="level">Numeric logging severity, using the LogLevel constants.</param>
    /// <param name="exception">Optional exception retained for handler formatting.</param>
    /// <exception cref="System.Exception">A handler fails; subsequent handlers are not invoked for this record.</exception>
    /// <remarks>
    ///     Footgun: the calling thread may hold any OptunaSharp lock, and it waits here while the handler serves another
    ///     thread's record. The handler lock is a plain monitor, not a <c>MonitorGate</c> entry, on purpose: a storage
    ///     monitor's holder logs (a new study, a journal warning), and such a holder must never wait for the Python GIL,
    ///     which a GIL-releasing entry would make it do to take the GIL back. A handler that takes no lock and not the GIL
    ///     (the rule for handlers) always finishes, so the wait is one record long.
    /// </remarks>
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
        {
            // Python's Handler.handle: `self.acquire(); try: self.emit(record) finally: self.release()`. Without it,
            // parallel trials called one handler at once: the default handler's line and exception interleaved with
            // other records, and a StreamLogHandler over an unsynchronized writer corrupted it (measured, 2026-11-08).
            lock (_handlerLocks.GetOrCreateValue(handler))
                handler.Emit(record);
        }
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
