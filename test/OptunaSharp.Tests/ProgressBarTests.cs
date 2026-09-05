using System;
using System.Collections.Generic;
using System.IO;
using AwesomeAssertions;
using OptunaSharp.Internal;

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

    // optuna/progress_bar.py: _ProgressBar.__init__ warns when asked to display a bar with neither a
    // trial count nor a timeout. Driven on the ProgressBar ctor directly — routing it through
    // Study.Optimize would need n_trials=null AND timeout=null, i.e. an unbounded (never-ending) run.
    [TestMethod]
    public void ProgressBar_EnabledWithoutTrialsOrTimeout_Warns()
    {
        string? captured = null;
        void Handler(string msg, WarningCategory cat)
        {
            if (msg.Contains("Progress bar won't be displayed"))
                captured = msg;
        }

        OptunaWarnings.WarningRaised += Handler;
        try
        {
            _ = new ProgressBar(enabled: true, nTrials: null, timeout: null);
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }

        captured.Should().Be("Progress bar won't be displayed because n_trials and timeout are None.");
    }

    // The warning fires only on the Optuna condition `is_valid and n_trials is None and timeout is
    // None`: disabled, or sized by either n_trials or timeout, must stay silent.
    [TestMethod]
    public void ProgressBar_DisabledOrSized_DoesNotWarnAboutDisplay()
    {
        var messages = new List<string>();
        void Handler(string msg, WarningCategory cat) => messages.Add(msg);

        OptunaWarnings.WarningRaised += Handler;
        try
        {
            _ = new ProgressBar(enabled: false, nTrials: null, timeout: null);  // not enabled
            _ = new ProgressBar(enabled: true, nTrials: 5, timeout: null);      // sized by n_trials
            _ = new ProgressBar(enabled: true, nTrials: null, timeout: 1.0);    // sized by timeout
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }

        messages.Should().NotContain(m => m.Contains("Progress bar won't be displayed"));
    }
}
