#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// .NET runtime knobs — ask-and-tell TPE tunes the CLR's own configuration switches (tiered compilation,
// dynamic PGO, ReadyToRun, server/concurrent GC, gen-0 budget) for a short-lived batch job, timing real child
// processes: the classic "the evaluation happens somewhere else" use of Optuna's ask-and-tell interface.
//
// source: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html
//         (ask-and-tell: `study.ask()`, run the evaluation yourself, `study.tell(trial, value)` — or
//         `state=TrialState.FAIL` when it breaks)
//         https://learn.microsoft.com/en-us/dotnet/core/runtime-config/compilation
//         https://learn.microsoft.com/en-us/dotnet/core/runtime-config/garbage-collector
//
// Real objective: each trial starts this same program as a child process with `--workload` and the trial's
// DOTNET_* environment variables, and measures the wall time from process start to exit (a CLI tool or a
// cron job pays exactly that: runtime startup + JIT + the work + GC). The workload builds a word-frequency
// index over 400k generated tokens and sorts it — allocation-heavy, with hot loops that tiering and PGO
// treat differently. The score is the median of 3 launches; a child that crashes is told as FAIL so the
// sampler stops proposing that corner. Timings are machine-dependent, so the winner varies by machine.
//
//   Run:  dotnet run -c Release examples/0_DotnetRuntimeKnobs.cs

using System.Diagnostics;
using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

if (args.Contains("--workload"))
{
    Workload.Run();                              // child mode: do the job and exit
    return;
}

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

string dll = typeof(Workload).Assembly.Location;   // this program, re-launched as the child
var study = Optuna.CreateStudy(
    sampler: new TPESampler(seed: 8, nStartupTrials: 8, multivariate: true),
    direction: StudyDirection.Minimize,
    studyName: "clr-knobs");

// the runtime's defaults first, so "tuned" is measured against what everyone ships.
study.EnqueueTrial(new OptunaSharp.Collections.OrderedDict<string, object?>
{
    ["TieredCompilation"] = "1", ["TieredPGO"] = "1", ["TC_QuickJitForLoops"] = "1", ["ReadyToRun"] = "1",
    ["gcServer"] = "0", ["gcConcurrent"] = "1", ["GCgen0size"] = "default",
});

for (int i = 0; i < 28; i++)
{
    Trial trial = study.Ask();                   // 1. ask: the sampler proposes a configuration
    var env = new Dictionary<string, string>();
    foreach (var knob in new[] { "TieredCompilation", "TieredPGO", "TC_QuickJitForLoops", "ReadyToRun", "gcServer", "gcConcurrent" })
        env["DOTNET_" + knob] = trial.SuggestCategorical(knob, new[] { "0", "1" });
    string gen0 = trial.SuggestCategorical("GCgen0size", new[] { "default", "0x1000000", "0x4000000" });  // 16/64 MiB
    if (gen0 != "default")
        env["DOTNET_GCgen0size"] = gen0;

    double? ms = MedianLaunchMillis(dll, env, launches: 3);   // 2. evaluate it ourselves
    if (ms is null)
        study.Tell(trial, state: TrialState.Fail);            // 3a. tell: it broke
    else
        study.Tell(trial, ms.Value);                          // 3b. tell: it took this long
}

var defaults = study.Trials[0];
var best = study.BestTrial;
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "runtime defaults : {0,7:F1} ms", defaults.Value));
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "best (trial #{0,-2}): {1,7:F1} ms  ({2:P0} of the default time)",
    best.Number, best.Value, best.Value / defaults.Value));
foreach (var kv in best.Params)
    Console.WriteLine($"    DOTNET_{kv.Key,-20} {kv.Value}");
Console.WriteLine($"failed launches told as FAIL: {study.Trials.Count(t => t.State == TrialState.Fail)}");

/// <summary>Launches the workload child several times and returns the median wall time.</summary>
/// <param name="dll">This program's assembly, run with <c>dotnet exec</c>.</param>
/// <param name="env">The DOTNET_* variables of the configuration under test.</param>
/// <param name="launches">How many launches to take the median of.</param>
/// <returns>The median milliseconds from start to exit, or <c>null</c> when any launch fails.</returns>
static double? MedianLaunchMillis(string dll, Dictionary<string, string> env, int launches)
{
    var times = new List<double>();
    for (int k = 0; k < launches; k++)
    {
        var psi = new ProcessStartInfo("dotnet") { RedirectStandardOutput = true, RedirectStandardError = true };
        psi.ArgumentList.Add("exec");
        psi.ArgumentList.Add(dll);
        psi.ArgumentList.Add("--workload");
        // the child must not inherit a knob from the parent's own environment.
        foreach (var name in psi.Environment.Keys.Where(n => n.StartsWith("DOTNET_", StringComparison.Ordinal)
                                                            && n != "DOTNET_ROOT").ToList())
            psi.Environment.Remove(name);
        foreach (var (name, value) in env)
            psi.Environment[name] = value;
        var sw = Stopwatch.StartNew();
        using var p = Process.Start(psi)!;
        p.StandardOutput.ReadToEnd();
        p.StandardError.ReadToEnd();
        p.WaitForExit();
        sw.Stop();
        if (p.ExitCode != 0)
            return null;
        times.Add(sw.Elapsed.TotalMilliseconds);
    }
    times.Sort();
    return times[times.Count / 2];
}

/// <summary>The batch job being tuned: a word-frequency index, built and sorted.</summary>
static class Workload
{
    /// <summary>Builds the index over generated text and prints a checksum (so nothing is optimized away).</summary>
    public static void Run()
    {
        var rng = new Random(42);
        string[] syllables = { "ka", "lo", "mi", "ra", "te", "su", "no", "vi", "da", "ze", "po", "qua" };
        var counts = new Dictionary<string, int>();
        for (int i = 0; i < 400_000; i++)
        {
            int len = 1 + rng.Next(3);
            string word = string.Concat(Enumerable.Range(0, len).Select(_ => syllables[rng.Next(syllables.Length)]));
            counts[word] = counts.GetValueOrDefault(word) + 1;
        }
        var top = counts.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key, StringComparer.Ordinal).Take(10).ToList();
        long checksum = top.Aggregate(0L, (acc, kv) => acc * 31 + kv.Value + kv.Key.Length);
        Console.WriteLine(checksum);
    }
}
