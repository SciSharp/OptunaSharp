using System;
using System.IO;
using AwesomeAssertions;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs found while writing the real-world <c>examples/0_*.cs</c> programs (2026-11-08).
///     Expected values are Optuna 5.0.0's output on CPython 3.12; each test fails today and passes once the
///     divergence is fixed, at which point its <c>[OpenBugs]</c> attribute should be removed.
/// </summary>
[TestClass]
public class RealWorldExampleBugTests
{
    /// <summary>Runs an action with standard error redirected and returns what it wrote there.</summary>
    /// <param name="action">The code whose standard-error output is captured (the progress bar writes there).</param>
    /// <returns>Everything written to standard error while <paramref name="action" /> ran.</returns>
    private static string CaptureStderr(Action action)
    {
        // Touch the logger first: its default handler binds to the stderr current at creation, and binding it
        // to the temporary writer below would leave every later test logging into a disposed writer.
        _ = OptunaSharp.Logging.OptunaLogging.GetVerbosity();

        var original = Console.Error;
        using var writer = new StringWriter();
        try
        {
            Console.SetError(writer);
            action();
        }
        finally
        {
            // Always restore: stderr is process-wide state shared with every other test.
            Console.SetError(original);
        }

        return writer.ToString();
    }

    /// <summary>Log handler that keeps the text of every record it receives, in order.</summary>
    /// <remarks>Registered for one test at a time; records from other threads' studies can interleave.</remarks>
    private sealed class CollectingHandler : OptunaSharp.Logging.ILogHandler
    {
        /// <summary>The messages received so far, oldest first.</summary>
        public System.Collections.Generic.List<string> Messages { get; } = new();

        /// <summary>Stores the record's message; never throws, so it cannot disturb the code under test.</summary>
        /// <param name="record">The log record delivered by <see cref="OptunaSharp.Logging.OptunaLogging" />.</param>
        public void Emit(in OptunaSharp.Logging.LogRecord record)
        {
            lock (Messages)
                Messages.Add(record.Message);
        }
    }

    /// <summary>
    ///     A single-objective objective that returns NaN must be logged with the value it returned, as
    ///     Optuna's <c>repr(value_or_values)</c> does: <c>Trial 0 failed with value nan.</c>; the port logs
    ///     <c>[nan]</c>.
    /// </summary>
    /// <remarks>
    ///     <c>optuna/study/_optimize.py</c> (<c>_log_failed_trial</c>) warns
    ///     <c>f"Trial {trial_number} failed with value {repr(value_or_values)}."</c> with the objective's own
    ///     return value — a float for a single-objective study. The single-objective <c>Study.Optimize</c>
    ///     overload wraps the scalar into a one-element array before the loop runs, and
    ///     <c>Internal/Optimize.cs</c> (<c>LogFailedTrial</c>) prints that array with <c>ListRepr</c>. The same
    ///     scalar-wrapping root cause as QA-12 (the skipped-tell message), on a different message. Found by
    ///     diffing the logs of a real-world run (<c>examples/0_GpuFineTunePlanner.cs</c> returns NaN for
    ///     diverged fp16 runs) against Optuna 5.0.0.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void Optimize_NaNScalarObjective_LogsScalarValueLikePython()
    {
        var handler = new CollectingHandler();
        int verbosity = OptunaSharp.Logging.OptunaLogging.GetVerbosity();
        OptunaSharp.Logging.OptunaLogging.SetVerbosity(OptunaSharp.Logging.LogLevel.Info);
        OptunaSharp.Logging.OptunaLogging.AddHandler(handler);
        try
        {
            var study = Optuna.CreateStudy();
            study.Optimize(_ => double.NaN, nTrials: 1);
        }
        finally
        {
            // Global logging state: always restore it, or later tests log at the wrong level.
            OptunaSharp.Logging.OptunaLogging.RemoveHandler(handler);
            OptunaSharp.Logging.OptunaLogging.SetVerbosity(verbosity);
        }

        handler.Messages.Should().Contain("Trial 0 failed with value nan.");
    }

    /// <summary>
    ///     The progress bar's running best must be formatted like Optuna's <c>f"{study.best_value:.6g}"</c>
    ///     (six significant digits: <c>0.930382</c>, <c>1.23457e+06</c>); the port prints the full float repr.
    /// </summary>
    /// <remarks>
    ///     Optuna's <c>_ProgressBar.update</c> (<c>optuna/progress_bar.py</c>) sets the description to
    ///     <c>f"Best trial: {study.best_trial.number}. Best value: {study.best_value:.6g}"</c>.
    ///     <c>Internal/ProgressBar.cs</c> builds the same text with <c>PythonRepr.FloatRepr</c>, i.e. Python's
    ///     <c>repr()</c>, so <c>0.9303817601391925</c> appears where Optuna shows <c>0.930382</c>. Found running
    ///     <c>examples/0_SpamFilterTimeBudget.cs</c> (<c>show_progress_bar=True</c>).
    /// </remarks>
    /// <param name="value">The objective value every trial returns.</param>
    /// <param name="expected">Python's <c>format(value, ".6g")</c>.</param>
    [TestMethod]
    [OpenBugs]
    [DataRow(0.9303817601391925, "0.930382")]
    [DataRow(1234567.0, "1.23457e+06")]
    public void ProgressBar_BestValue_UsesPythonSixSignificantDigits(double value, string expected)
    {
        string output = CaptureStderr(() =>
        {
            var study = Optuna.CreateStudy();
            study.Optimize(_ => value, nTrials: 2, showProgressBar: true);
        });

        output.Should().Contain($"Best trial: 0. Best value: {expected}",
            "Optuna formats the progress bar's best value with :.6g");
    }
}
