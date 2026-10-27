#:project ../../src/OptunaSharp.Interop.pythonnet/OptunaSharp.Interop.pythonnet.csproj
#:property PublishAot=false

// OptunaHub pruner — drive an OptunaSharp report/should_prune loop with a real hub PRUNER. this is the
// third bridge direction (after samplers and benchmarks): OptunaHub.LoadModule("pruners/median") is
// wrapped as an OptunaSharp BasePruner, and each should_prune() delegates to the real Python
// MedianPruner over a shadow study carrying every trial's intermediate values. laggards get cut short.
//
// source (paper):  https://arxiv.org/abs/2510.02798
// source (pruner): https://hub.optuna.org/pruners/median/
//
// needs `pip install optuna optunahub` (MedianPruner is pure Optuna — no extra deps). the objective is
// a deterministic surrogate training curve, so the point is the PRUNER, not the objective.
//
//   Run:  examples/optunahub/run-optunahub.sh
//   or:   dotnet run examples/optunahub/OptunaHubPruning.cs

using System;
using System.Linq;
using System.Runtime.InteropServices;
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Logging;
using OptunaSharp.Interop.PythonNet;

Native.EnableVirtualTerminal();
try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
OptunaLogging.DisableDefaultHandler();

bool color = Environment.GetEnvironmentVariable("FORCE_COLOR") is { Length: > 0 }
    || (Environment.GetEnvironmentVariable("NO_COLOR") is null && !Console.IsOutputRedirected);
string C(string s, string code) => color ? $"\x1b[{code}m{s}\x1b[0m" : s;

const int epochs = 20;

if (!TryProbe(out string why))
{
    Console.WriteLine(C("OptunaHub pruner demo — prerequisites missing", "1;33"));
    Console.WriteLine($"  {why}");
    Console.WriteLine(C("  pip install optuna optunahub", "36"));
    Console.WriteLine(C("  or:  examples/optunahub/run-optunahub.sh", "36"));
    return;
}

Console.WriteLine();
Console.WriteLine(C("  OptunaHub pruner  —  a hub MedianPruner cutting laggards short", "1;36"));
Console.WriteLine($"  optuna {C(OptunaHubRuntime.OptunaVersion ?? "?", "32")}   optunahub {C(OptunaHubRuntime.OptunaHubVersion ?? "?", "32")}");
Console.WriteLine();

try
{
    Console.Write("  loading " + C("pruners/median", "1;35") + " … ");
    using var mod = Load("pruners/median");
    using var pruner = mod.CreatePruner("MedianPruner", new() { ["n_startup_trials"] = 4, ["n_warmup_steps"] = 2 });
    Console.WriteLine(C("ok", "1;32"));
    Console.WriteLine(C($"  \"training\" {epochs} epochs per trial; the hub pruner stops trials that fall behind the median.", "2"));
    Console.WriteLine();

    var study = Optuna.CreateStudy(
        sampler: new TPESampler(seed: 0),
        pruner: pruner,                              // the HUB pruner, wrapped as an OptunaSharp BasePruner
        direction: StudyDirection.Minimize,
        studyName: "optunahub-median-pruning");

    void Render(Study s, FrozenTrial t)
    {
        if (t.State == TrialState.Pruned)
        {
            long lastEpoch = t.IntermediateValues.Count > 0 ? t.IntermediateValues.Keys.Max() + 1 : 0;
            Console.WriteLine($"  {C($"#{t.Number,3}", "90")}  {C("✂ pruned", "33")}  @ epoch {lastEpoch,2}/{epochs}");
        }
        else
        {
            Console.WriteLine($"  {C($"#{t.Number,3}", "90")}  {C("✔ done  ", "32")}  {epochs} epochs   loss {t.Value,8:F4}");
        }
    }
    study.Optimize(Objective, nTrials: 40, callbacks: new[] { (Action<Study, FrozenTrial>)Render });

    int completed = study.Trials.Count(t => t.State == TrialState.Complete);
    int prunedN = study.Trials.Count(t => t.State == TrialState.Pruned);
    long savedEpochs = study.Trials.Where(t => t.State == TrialState.Pruned)
                                  .Sum(t => epochs - (t.IntermediateValues.Count > 0 ? t.IntermediateValues.Keys.Max() + 1 : 0));
    Console.WriteLine();
    Console.WriteLine(C($"  {study.Trials.Count} trials  ·  {completed} completed  ·  {prunedN} pruned early", "1;37")
        + C($"  (~{savedEpochs} epochs of \"training\" skipped)", "2"));
    Console.WriteLine(C($"  ✔ best loss {study.BestValue:F4}  at  " +
        string.Join(", ", study.BestParams.Select(kv => $"{kv.Key}={kv.Value}")), "1;32"));
    Console.WriteLine();
}
catch (Exception ex)
{
    Console.WriteLine(C("failed", "1;31"));
    Console.WriteLine(C("  " + ex.Message.Split('\n')[0], "1;33"));
}

// a deterministic stand-in for training: 20 epochs of validation loss decaying toward a floor set by the
// hyperparameters. laggards (worse floor / slower decay) drop below the running median and get pruned.
static double Objective(Trial t)
{
    double lr = t.SuggestFloat("learning_rate", 1e-4, 1e-1, log: true);
    long units = t.SuggestInt("n_units", 16, 256, log: true);
    double dropout = t.SuggestFloat("dropout", 0.0, 0.6);

    double offLr = Math.Log10(lr) + 2.0;                                 // 0 at the sweet spot lr = 1e-2
    double floor = 0.08 + 0.12 * offLr * offLr + 0.40 * Math.Abs(dropout - 0.15) + 0.30 / Math.Log2(units);
    double rate = 0.5 / (1.0 + Math.Abs(offLr));

    double loss = 0.0;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        loss = floor + (1.2 - floor) * Math.Exp(-rate * (epoch + 1));    // decreasing toward the floor
        t.Report(loss, epoch);                                          // intermediate value -> the pruner reads it
        if (t.ShouldPrune())                                            // worse than the hub pruner's median?
            throw new TrialPruned();                                    // -> stop early; recorded as PRUNED
    }
    return loss;
}

// ── helpers ──────────────────────────────────────────────────────────────────────────────────────

static OptunaHubModule Load(string package)
{
    try { return OptunaHub.LoadModule(package); }
    catch (OptunaHubInteropException)
    {
        string local = System.IO.Path.Combine(RepoRoot(), "refs", "optunahub-registry", "package");
        return OptunaHub.LoadLocalModule(package, local);
    }
}

static bool TryProbe(out string reason)
{
    try
    {
        PythonRuntime.Run(() => { _ = OptunaHubRuntime.Optuna; _ = OptunaHubRuntime.OptunaHub; });
        reason = "";
        return true;
    }
    catch (Exception ex) { reason = ex.Message.Split('\n')[0]; return false; }
}

static string RepoRoot([System.Runtime.CompilerServices.CallerFilePath] string? thisFile = null)
    => System.IO.Path.GetFullPath(System.IO.Path.Combine(System.IO.Path.GetDirectoryName(thisFile)!, "..", ".."));

static class Native
{
    const int STD_OUTPUT_HANDLE = -11;
    const uint ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004;
    [DllImport("kernel32.dll", SetLastError = true)] static extern IntPtr GetStdHandle(int n);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool GetConsoleMode(IntPtr h, out uint m);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool SetConsoleMode(IntPtr h, uint m);
    public static void EnableVirtualTerminal()
    {
        if (!OperatingSystem.IsWindows()) return;
        try { IntPtr h = GetStdHandle(STD_OUTPUT_HANDLE); if (GetConsoleMode(h, out uint m)) SetConsoleMode(h, m | ENABLE_VIRTUAL_TERMINAL_PROCESSING); }
        catch { }
    }
}
