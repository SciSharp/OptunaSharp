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
