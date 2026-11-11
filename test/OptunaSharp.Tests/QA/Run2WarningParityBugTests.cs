using System;
using System.IO;
using System.Text.RegularExpressions;
using AwesomeAssertions;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs in how Optuna's warnings reach the user, found by the 2026-11-10 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-10.md</c>, QA-77). Optuna raises them with Python's <c>warnings.warn</c>, which
///     prints to stderr through <c>warnings.showwarning</c>, outside <c>optuna.logging</c>, once per distinct message and
///     call site under the default filters. The port sends every warning through the <c>optuna</c> logger at WARNING
///     level (<c>OptunaWarnings.Warn</c>). A test marked <c>[OpenBugs]</c> fails until its divergence is fixed.
/// </summary>
/// <remarks>
///     Both tests redirect stderr and rebuild the library logger around it, the way <c>LoggingTests</c> does, and put
///     both back afterwards; they do not run in parallel with other tests.
/// </remarks>
[TestClass]
[DoNotParallelize]
public class Run2WarningParityBugTests
{
    /// <summary>The step-divisibility warning's fixed part, the same in Optuna's and the port's text.</summary>
    private const string StepWarning = "but the range is not divisible by `step`";

    /// <summary>
    ///     QA-77: <c>optuna.logging.set_verbosity(optuna.logging.ERROR)</c>, the usual way to silence the per-trial log,
    ///     does not touch Python warnings, so Optuna still prints
    ///     <c>UserWarning: The distribution is specified by [0, 1] and step=0.3, but the range is not divisible by `step`.
    ///     It will be replaced with [0, 0.9].</c> The port routes the warning through the logger, whose level then drops
    ///     it: nothing reaches stderr.
    /// </summary>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void Warnings_StillReachStderr_WhenVerbosityIsError()
    {
        string stderr = CaptureStderr(LogLevel.Error, () =>
        {
            var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 0));
            study.Optimize(t => t.SuggestFloat("x", 0, 1, step: 0.3), nTrials: 3);
        });

        stderr.Should().Contain(StepWarning, "Python's warnings.warn ignores optuna.logging's verbosity");
    }

    /// <summary>
    ///     QA-77, repetition: under Python's default warning filters a warning with the same text from the same call site
    ///     is shown once, so five trials suggesting the same non-divisible float show it once (measured: two distinct
    ///     warnings, two lines, for a float and an int parameter over five trials). The port logs it on every trial. This
    ///     half was already listed in the 2026-10-26 release review's LOW notes ("OptunaWarnings logs every
    ///     occurrence"); this test pins it.
    /// </summary>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void RepeatedWarning_IsShownOnceLikePythonsDefaultFilter()
    {
        string stderr = CaptureStderr(LogLevel.Info, () =>
        {
            var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 0));
            study.Optimize(t => t.SuggestFloat("x", 0, 1, step: 0.3), nTrials: 5);
        });

        Regex.Matches(stderr, Regex.Escape(StepWarning)).Count.Should().Be(1);
    }

    /// <summary>
    ///     Runs <paramref name="body" /> with stderr redirected to a buffer and the library logger rebuilt over it at
    ///     <paramref name="verbosity" />, then restores stderr, the logger and the previous verbosity.
    /// </summary>
    /// <param name="verbosity">The <c>optuna</c> logger level to run the body at.</param>
    /// <param name="body">The code whose stderr output is wanted.</param>
    /// <returns>Everything written to stderr while <paramref name="body" /> ran.</returns>
    private static string CaptureStderr(int verbosity, Action body)
    {
        int previous = OptunaLogging.GetVerbosity();
        TextWriter original = Console.Error;
        var buffer = new StringWriter();
        try
        {
            // The default handler binds Console.Error when it is created, so rebuild it after redirecting.
            OptunaLogging.ResetLibraryRootLogger();
            Console.SetError(buffer);
            OptunaLogging.SetVerbosity(verbosity);
            body();
        }
        finally
        {
            // Rebuild the default handler over the real stderr so later tests log normally.
            Console.SetError(original);
            OptunaLogging.ResetLibraryRootLogger();
            OptunaLogging.SetVerbosity(previous);
        }
        return buffer.ToString();
    }
}
