using System;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Logging;

namespace OptunaSharp.Tests;

/// <summary>
///     The default stream handler's failure path — the port of Python's <c>StreamHandler.emit</c> →
///     <c>Handler.handleError</c>. A write to a broken stream must be reported on the current standard error
///     ("--- Logging error ---") and swallowed, never propagated: under Optuna a closed log stream cannot fail
///     the optimization that is logging, and before this port it did (an <see cref="ObjectDisposedException" />
///     escaped <see cref="Study.Optimize(Func{Trial, double}, int?, double?, int, Type[], System.Collections.Generic.IEnumerable{Action{Study, FrozenTrial}}, bool, bool, TrialExecutorOptions, System.Threading.CancellationToken)" />).
///     <para>Also the handler contract Python's <c>logging</c> keeps: one record at a time per handler (the lock
///     <c>Handler.handle</c> holds), one write and one flush per record, and a handler registered twice still receiving
///     each record once.</para>
///     <para>These tests redirect <see cref="Console.Error" /> and rebuild the process-wide logging
///     configuration, so every test restores both in a <c>finally</c> (the project runs tests sequentially).</para>
/// </summary>
[TestClass]
public class LoggingTests
{
    /// <summary>
    ///     A handler whose writer was disposed reports the failure on the current stderr in Python's
    ///     <c>handleError</c> layout — header, exception, <c>Call stack:</c>, <c>Message: %r</c>,
    ///     <c>Arguments: ()</c> — and does not throw.
    /// </summary>
    [TestMethod]
    public void StreamLogHandler_DisposedWriter_ReportsLikePythonHandleError_AndDoesNotThrow()
    {
        var dead = new StringWriter();
        var handler = new StreamLogHandler(dead);
        dead.Dispose();
        var record = new LogRecord("optuna.test", LogLevel.Info, "it's done", null, DateTime.Now);

        string report = CaptureStderr(() =>
        {
            Action emit = () => handler.Emit(record);
            emit.Should().NotThrow("Python's StreamHandler.emit routes write failures to handleError");
        });

        report.Should().StartWith("--- Logging error ---\n");
        report.Should().Contain(nameof(ObjectDisposedException));
        report.Should().Contain("Call stack:\n");
        // The calling context starts outside OptunaSharp.Logging (Python walks f_back out of logging).
        report.Should().Contain(nameof(StreamLogHandler_DisposedWriter_ReportsLikePythonHandleError_AndDoesNotThrow));
        // repr("it's done") picks double quotes because the text holds a single quote.
        report.Should().Contain("Message: \"it's done\"\nArguments: ()\n");
    }

    /// <summary>
    ///     The failure that made the suite order-dependent: the library's default handler is created while
    ///     stderr is redirected to a writer that is then disposed. Optimizing afterwards must still complete
    ///     (every INFO trial log hits the dead writer) and must report each failure on the real stderr.
    /// </summary>
    [TestMethod]
    public void Optimize_Completes_WhenDefaultHandlerWasBoundToADisposedStderr()
    {
        int verbosity = OptunaLogging.GetVerbosity();
        TextWriter original = Console.Error;
        var redirect = new StringWriter();
        try
        {
            // Bind a fresh default handler to the redirect, then restore stderr and kill the redirect —
            // the exact sequence a stderr-capturing test (or host) produces when it is the first to log.
            OptunaLogging.ResetLibraryRootLogger();
            Console.SetError(redirect);
            OptunaLogging.SetVerbosity(LogLevel.Info);
            Console.SetError(original);
            redirect.Dispose();

            Study? study = null;
            string report = CaptureStderr(() =>
            {
                study = Optuna.CreateStudy();
                study.Optimize(t => t.SuggestFloat("x", -1, 1), nTrials: 3);
            });

            study!.Trials.Should().HaveCount(3).And.OnlyContain(t => t.State == TrialState.Complete);
            report.Should().Contain("--- Logging error ---");
            // repr() picks double quotes: the message holds the params dict's single-quoted keys.
            report.Should().Contain("Message: \"Trial 0 finished with value:");
        }
        finally
        {
            // Rebuild the default handler over the real stderr so later tests log normally.
            Console.SetError(original);
            OptunaLogging.ResetLibraryRootLogger();
            OptunaLogging.SetVerbosity(verbosity);
        }
    }

    /// <summary>
    ///     A healthy writer is untouched by the error path: the record is written once in Optuna's default
    ///     format and nothing is reported on stderr.
    /// </summary>
    [TestMethod]
    public void StreamLogHandler_HealthyWriter_WritesTheRecord_AndReportsNothing()
    {
        var sink = new StringWriter();
        var handler = new StreamLogHandler(sink);
        var record = new LogRecord("optuna.test", LogLevel.Warning, "careful", null, new DateTime(2021, 10, 31, 5, 35, 17, 232));

        string report = CaptureStderr(() => handler.Emit(record));

        sink.ToString().Should().Be("[W 2021-10-31 05:35:17,232] careful" + Environment.NewLine);
        report.Should().BeEmpty();
    }

    /// <summary>
    ///     The INFO line of a pruned trial matches Optuna's <c>f"Trial {number} pruned. {str(func_err)}"</c>. Live
    ///     Optuna 5.0.0 logs <c>"Trial 0 pruned. "</c> for <c>raise TrialPruned()</c> (an empty <c>str()</c>) and
    ///     <c>"Trial 1 pruned. custom"</c> for <c>raise TrialPruned("custom")</c>. Before the fix the bare
    ///     constructor logged the BCL default <c>"Exception of type 'OptunaSharp.TrialPruned' was thrown."</c>.
    /// </summary>
    /// <exception cref="TrialPruned">
    ///     Escapes only if <c>Optimize</c> stopped catching the objectives' prune signal, which fails the test;
    ///     normally <c>Optimize</c> catches it and records the trial as pruned.
    /// </exception>
    [TestMethod]
    public void PrunedTrialLogLine_MatchesOptuna_ForBareAndMessagedTrialPruned()
    {
        new TrialPruned().Message.Should().BeEmpty("Python's str(TrialPruned()) is ''");

        int verbosity = OptunaLogging.GetVerbosity();
        var messages = new System.Collections.Generic.List<string>();
        var handler = new CollectingHandler(messages);
        try
        {
            OptunaLogging.SetVerbosity(LogLevel.Info);
            OptunaLogging.AddHandler(handler);

            // Typed as single-objective explicitly: a throw-only lambda has no return type to pick an overload by.
            Func<Trial, double> bare = _ => throw new TrialPruned();
            Func<Trial, double> messaged = _ => throw new TrialPruned("custom");
            var study = Optuna.CreateStudy();
            study.Optimize(bare, nTrials: 1);
            study.Optimize(messaged, nTrials: 1);
        }
        finally
        {
            OptunaLogging.RemoveHandler(handler);
            OptunaLogging.SetVerbosity(verbosity);
        }

        messages.Should().Contain("Trial 0 pruned. ").And.Contain("Trial 1 pruned. custom");
        messages.Should().NotContain(m => m.Contains("was thrown"));
    }

    /// <summary>
    ///     A record is one write and one flush, as Python's <c>StreamHandler.emit</c> since issue 35046: the line and the
    ///     exception arrive together, so nothing another thread writes can land between them, and the text is the same
    ///     as the two writes produced before.
    /// </summary>
    [TestMethod]
    public void StreamLogHandler_WritesEachRecordInOneWrite_ThenFlushes()
    {
        var sink = new CountingWriter();
        var handler = new StreamLogHandler(sink);
        var error = new InvalidOperationException("boom");
        var stamp = new DateTime(2021, 10, 31, 5, 35, 17, 232);

        handler.Emit(new LogRecord("optuna.test", LogLevel.Warning, "failed", error, stamp));
        handler.Emit(new LogRecord("optuna.test", LogLevel.Info, "plain", null, stamp));

        sink.Writes.Should().Be(2, "one write per record, exception included");
        sink.Flushes.Should().Be(2, "Python flushes after every record");
        sink.ToString().Should().Be(
            "[W 2021-10-31 05:35:17,232] failed" + Environment.NewLine + error + Environment.NewLine +
            "[I 2021-10-31 05:35:17,232] plain" + Environment.NewLine);
    }

    /// <summary>
    ///     Eight threads log failures with exceptions through one <see cref="StreamLogHandler" /> over a plain, unsynchronized
    ///     writer, as the CLI's <c>--log-file</c> builds it. OptunaSharp calls a handler for one record at a time (Python's
    ///     <c>Handler.handle</c> lock), so every record arrives whole: its line, then its own exception. Measured before the
    ///     lock, over a plain <c>StreamWriter</c>: 412 of 40,000 records whole and 48,247 mangled lines.
    /// </summary>
    [TestMethod]
    public void StreamLogHandler_OverAnUnsynchronizedWriter_KeepsEveryRecordWhole_WhenThreadsLogAtOnce()
    {
        const int threads = 8, perThread = 1_000;
        int verbosity = OptunaLogging.GetVerbosity();
        var sink = new StringWriter();
        var handler = new StreamLogHandler(sink);
        Logger logger = OptunaLogging.GetLogger("optuna.tests.handler_lock");
        var failures = new System.Collections.Concurrent.ConcurrentQueue<Exception>();
        try
        {
            OptunaLogging.DisableDefaultHandler();
            OptunaLogging.SetVerbosity(LogLevel.Warning);
            OptunaLogging.AddHandler(handler);
            using var start = new System.Threading.ManualResetEventSlim();
            var workers = new System.Threading.Thread[threads];
            for (int t = 0; t < threads; t++)
            {
                int thread = t;
                workers[t] = new System.Threading.Thread(() =>
                {
                    try
                    {
                        start.Wait();
                        for (int i = 0; i < perThread; i++)
                            logger.Warning($"record {thread}-{i}", new InvalidOperationException($"boom {thread}-{i}"));
                    }
                    catch (Exception e)
                    {
                        failures.Enqueue(e);
                    }
                });
                workers[t].Start();
            }

            start.Set();
            foreach (var worker in workers)
                worker.Join();
        }
        finally
        {
            OptunaLogging.RemoveHandler(handler);
            OptunaLogging.EnableDefaultHandler();
            OptunaLogging.SetVerbosity(verbosity);
        }

        failures.Should().BeEmpty("an unsynchronized StringBuilder appended to by two threads at once can throw");
        string[] lines = sink.ToString().Split('\n').Select(l => l.TrimEnd('\r')).Where(l => l.Length > 0).ToArray();
        lines.Should().HaveCount(2 * threads * perThread, "a header line and an exception line per record, none mangled");
        for (int i = 0; i < lines.Length; i += 2)
        {
            // "[W <stamp>] record T-I" followed at once by "System.InvalidOperationException: boom T-I".
            string id = lines[i][(lines[i].LastIndexOf(' ') + 1)..];
            lines[i].Should().StartWith("[W ").And.EndWith($"] record {id}");
            lines[i + 1].Should().Be($"System.InvalidOperationException: boom {id}");
        }
    }

    /// <summary>
    ///     Adding a handler that is already registered does nothing, so it still receives each record once, as Python's
    ///     <c>Logger.addHandler</c> ensures (live Python: two <c>addHandler(h)</c> calls leave one <c>h</c> and one line);
    ///     one removal then removes it.
    /// </summary>
    [TestMethod]
    public void AddHandler_TheSameHandlerTwice_ReceivesEachRecordOnce_AsPython()
    {
        int verbosity = OptunaLogging.GetVerbosity();
        var messages = new System.Collections.Generic.List<string>();
        var handler = new CollectingHandler(messages);
        Logger logger = OptunaLogging.GetLogger("optuna.tests.add_twice");
        try
        {
            OptunaLogging.SetVerbosity(LogLevel.Warning);
            OptunaLogging.AddHandler(handler);
            OptunaLogging.AddHandler(handler);
            logger.Warning("once?");
            OptunaLogging.RemoveHandler(handler);
            logger.Warning("removed");
        }
        finally
        {
            OptunaLogging.RemoveHandler(handler);
            OptunaLogging.SetVerbosity(verbosity);
        }

        messages.Should().Equal("once?");
    }

    /// <summary>A string writer that counts the write and flush calls a handler makes.</summary>
    private sealed class CountingWriter : StringWriter
    {
        /// <summary>How many times a string was written.</summary>
        public int Writes { get; private set; }

        /// <summary>How many times the writer was flushed.</summary>
        public int Flushes { get; private set; }

        /// <summary>Counts the write and appends the text.</summary>
        /// <param name="value">The text written.</param>
        public override void Write(string? value)
        {
            Writes++;
            base.Write(value);
        }

        /// <summary>Counts the flush.</summary>
        public override void Flush()
        {
            Flushes++;
            base.Flush();
        }
    }

    /// <summary>Collects the message of every record it receives, so a test can assert exact log text.</summary>
    /// <param name="sink">List that receives each record's message, in emission order.</param>
    private sealed class CollectingHandler(System.Collections.Generic.List<string> sink) : ILogHandler
    {
        /// <summary>Appends the record's message under a lock, because Optimize may log from worker threads.</summary>
        /// <param name="record">Log record whose message is collected.</param>
        public void Emit(in LogRecord record)
        {
            lock (sink)
                sink.Add(record.Message);
        }
    }

    /// <summary>
    ///     Runs <paramref name="action" /> with <see cref="Console.Error" /> redirected and returns what it
    ///     wrote. The library logger is configured first so this redirect can never become its default stream.
    /// </summary>
    /// <param name="action">The code whose standard-error output is captured.</param>
    /// <returns>Everything written to standard error while <paramref name="action" /> ran.</returns>
    private static string CaptureStderr(Action action)
    {
        _ = OptunaLogging.GetVerbosity();
        TextWriter original = Console.Error;
        using var buffer = new StringWriter();
        try
        {
            Console.SetError(buffer);
            action();
        }
        finally
        {
            Console.SetError(original);
        }
        return buffer.ToString();
    }
}
