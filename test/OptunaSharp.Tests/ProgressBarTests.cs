using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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

    /// <summary>
    ///     Runs <paramref name="action" /> with <see cref="Console.Error" /> redirected to a private buffer and
    ///     returns what it wrote (the progress bar renders to stderr).
    /// </summary>
    /// <param name="action">The code whose standard-error output is captured.</param>
    /// <returns>Everything written to standard error while <paramref name="action" /> ran.</returns>
    private static string CaptureStderr(Action action)
    {
        // Configure the library logger BEFORE swapping stderr. Its default handler binds to the stderr that
        // is current when it is created (Python's StreamHandler capturing sys.stderr), so if this capture
        // were the first thing to log, the process-wide handler would keep the buffer disposed below, and
        // every later log line of every later test would hit a closed writer. It was: run alone (or first),
        // this class poisoned logging for the rest of the run.
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

    /// <summary>
    ///     With nJobs every worker updates the bar: each finished trial must count once, in order, and the bar must
    ///     end at n_trials, as tqdm's locked update does. The unlocked counter lost increments and could write a line
    ///     built from an older count after a newer one.
    /// </summary>
    [TestMethod]
    public void ProgressBar_NJobs_CountsEveryTrialOnceAndInOrder()
    {
        const int nTrials = 2000;
        string output = CaptureStderr(() =>
        {
            var study = Optuna.CreateStudy(sampler: new OptunaSharp.Samplers.RandomSampler(seed: 1));
            study.Optimize(Sphere, nTrials: nTrials, nJobs: 16, showProgressBar: true);
        });

        var counts = System.Text.RegularExpressions.Regex.Matches(output, @"\| (\d+)/" + nTrials + @"\]")
            .Select(m => int.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture))
            .ToList();
        counts.Should().Equal(Enumerable.Range(1, nTrials), "every trial counts once, and lines are written in count order");
    }
}
