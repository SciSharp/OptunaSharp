using System;
using System.IO;
using AwesomeAssertions;

namespace OptunaSharp.Tests;

/// <summary>
///     The <c>show_progress_bar</c> path (plan 4.9.2) — <c>Internal/ProgressBar.cs</c>, the
///     dependency-free tqdm stand-in wired through <c>Study.Optimize(showProgressBar:)</c>. The
///     renderer writes to stderr, so these tests capture <see cref="Console.Error" /> (the test
///     project runs sequentially, so the swap is safe) and assert the trial-count progress + the
///     best-value postfix appear when enabled, and that nothing is written when it is off.
/// </summary>
[TestClass]
public class ProgressBarTests
{
    private static double Sphere(Trial t)
    {
        double x = t.SuggestFloat("x", -1, 1);
        return x * x;
    }

    private static string CaptureStderr(Action action)
    {
        var original = Console.Error;
        using var writer = new StringWriter();
        try
        {
            Console.SetError(writer);
            action();
        }
        finally
        {
            Console.SetError(original);
        }

        return writer.ToString();
    }

    [TestMethod]
    public void ProgressBar_Enabled_WritesTrialProgressAndBestValueToStderr()
    {
        string output = CaptureStderr(() =>
        {
            var study = Optuna.CreateStudy();
            study.Optimize(Sphere, nTrials: 3, showProgressBar: true);
        });

        output.Should().Contain("3/3");          // the n_trials-mode "<count>/<n_trials>" progress
        output.Should().Contain("Best value:");  // the postfix carrying the running best
    }

    [TestMethod]
    public void ProgressBar_Disabled_WritesNothing()
    {
        string output = CaptureStderr(() =>
        {
            var study = Optuna.CreateStudy();
            study.Optimize(Sphere, nTrials: 3, showProgressBar: false);
        });

        output.Should().BeEmpty();
    }
}
