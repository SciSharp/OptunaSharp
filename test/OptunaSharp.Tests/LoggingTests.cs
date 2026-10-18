using System;
using System.IO;
using AwesomeAssertions;
using OptunaSharp.Logging;

namespace OptunaSharp.Tests;

/// <summary>
///     The default stream handler's failure path — the port of Python's <c>StreamHandler.emit</c> →
///     <c>Handler.handleError</c>. A write to a broken stream must be reported on the current standard error
///     ("--- Logging error ---") and swallowed, never propagated: under Optuna a closed log stream cannot fail
///     the optimization that is logging, and before this port it did (an <see cref="ObjectDisposedException" />
///     escaped <see cref="Study.Optimize(Func{Trial, double}, int?, double?, int, Type[], System.Collections.Generic.IEnumerable{Action{Study, FrozenTrial}}, bool, bool)" />).
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
