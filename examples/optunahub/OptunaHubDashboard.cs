#:project ../../src/OptunaSharp.Interop.pythonnet/OptunaSharp.Interop.pythonnet.csproj
#:project ../../src/OptunaSharp.Storage.Sqlite/OptunaSharp.Storage.Sqlite.csproj
#:property PublishAot=false
#:property SkipNDScopeWeave=true
#:property GeneratePackageOnBuild=false

// OptunaHub × optuna-dashboard — an OptunaSharp study driven by a REAL hub sampler, PERSISTED by
// OptunaSharp's own SQLite storage, then explored live in the real Python optuna-dashboard. This ties
// the whole folder together: the search strategy is the hub's AutoSampler (via the pythonnet bridge),
// the objective is a deterministic "training" curve reporting intermediate values, a native MedianPruner
// cuts laggards short — and every trial is written to a `.db` file with Optuna's EXACT schema by
// OptunaSharp.Storage.Sqlite, so `optuna-dashboard sqlite:///…` opens it with no conversion at all.
//
// source (paper):     https://arxiv.org/abs/2510.02798        (OptunaHub: A Platform for Black-Box Optimization)
// source (sampler):   https://hub.optuna.org/samplers/auto_sampler/
// source (dashboard): https://github.com/optuna/optuna-dashboard   ·   https://optuna-dashboard.readthedocs.io/en/latest/
// source (storage):   src/OptunaSharp.Storage.Sqlite   (a faithful port of optuna.storages.RDBStorage)
//
// needs `pip install optuna optunahub optuna-dashboard` (AutoSampler also uses `cmaes`; if the hub
// sampler can't load it falls back to OptunaSharp's native TPE, so the study is always written). the
// objective is a self-contained closed-form curve — the point here is the STORAGE + the dashboard, not
// the objective, so no surrogate ML is needed.
//
//   Run:  examples/optunahub/run-optunahub.sh     (Linux/macOS — installs deps, runs the suite, then
//         examples\optunahub\run-optunahub.bat    (Windows)         opens the dashboard on this study)
//   or:   dotnet run examples/optunahub/OptunaHubDashboard.cs
//         optuna-dashboard sqlite:///examples/optunahub/optunahub-dashboard.db   # then browse to it
//
//   colors: honors NO_COLOR / FORCE_COLOR.

using System;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Pruners;
using OptunaSharp.Storages;
using OptunaSharp.Logging;
using OptunaSharp.Interop.PythonNet;

Native.EnableVirtualTerminal();
try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { /* redirected handle — leave as-is */ }
OptunaLogging.DisableDefaultHandler();   // silence OptunaSharp's own INFO logs — our colored view replaces them

bool color = Environment.GetEnvironmentVariable("FORCE_COLOR") is { Length: > 0 }
    || (Environment.GetEnvironmentVariable("NO_COLOR") is null && !Console.IsOutputRedirected);
string C(string s, string code) => color ? $"\x1b[{code}m{s}\x1b[0m" : s;

const int epochs = 20;      // "training" epochs per trial — each reports one intermediate value
const int nTrials = 50;     // enough trials for the dashboard's history / importance / contour plots

// the engine + optunahub must be present (the dashboard is Python too). probe first, guide if missing.
if (!TryProbe(out string why))
{
    Console.WriteLine(C("OptunaHub dashboard demo — prerequisites missing", "1;33"));
    Console.WriteLine($"  {why}");
    Console.WriteLine(C("  pip install optuna optunahub optuna-dashboard cmaes", "36"));
    Console.WriteLine(C("  or:  examples/optunahub/run-optunahub.sh", "36"));
    return;
}

Console.WriteLine();
Console.WriteLine(C("  OptunaHub  ×  optuna-dashboard  —  a hub-sampled study, persisted for the dashboard", "1;36"));
Console.WriteLine($"  optuna {C(PythonRuntime.OptunaVersion ?? "?", "32")}   optunahub {C(PythonRuntime.OptunaHubVersion ?? "?", "32")}");
Console.WriteLine();

// where the study is written. an absolute path derived from THIS file, so it lands beside the example
// no matter the working directory — and the launcher's dashboard.py resolves the very same file.
string dbPath = Path.Combine(ThisDir(), "optunahub-dashboard.db");
DeleteDbFiles(dbPath);                                             // start fresh so the dashboard shows just this run
string url = "sqlite:///" + dbPath.Replace('\\', '/');            // sqlite:///C:/…/x.db — Optuna's own URL form

// pick the sampler: the hub's AutoSampler if it loads, else OptunaSharp's native TPE. either way the
// study is persisted, so the dashboard always has something to show.
var (sampler, owned, samplerLabel) = BuildSampler();
Console.WriteLine($"  sampler   : {C(samplerLabel, "1;35")}");
Console.WriteLine($"  storage   : {C("OptunaSharp.Storage.Sqlite (RDBStorage → Optuna schema)", "37")}");
Console.WriteLine($"  file      : {C(dbPath, "90")}");
Console.WriteLine();

try
{
    // RDBStorage writes Optuna's EXACT schema; disposing it at the end frees the file for the dashboard.
    using var storage = new RDBStorage(url);
    var study = Optuna.CreateStudy(
        storage: storage,
        sampler: sampler,
        pruner: new MedianPruner(nStartupTrials: 5, nWarmupSteps: 3),   // native pruner → pruned trials in the dashboard
        direction: StudyDirection.Minimize,
        studyName: "optunahub-dashboard",
        loadIfExists: true);

    double? best = null;
    void Render(Study s, FrozenTrial t)
    {
        if (t.State == TrialState.Pruned)
        {
            int lastEpoch = t.IntermediateValues.Count > 0 ? t.IntermediateValues.Keys.Max() + 1 : 0;
            Console.WriteLine($"  {C($"#{t.Number,3}", "90")}  {C("✂ pruned", "33")}  @ epoch {lastEpoch,2}/{epochs}");
        }
        else
        {
            if (t.Value is { } v && (best is null || v < best)) best = v;
            bool improved = t.Value is { } vv && vv <= best;
            string mark = improved ? C("★", "1;32") : " ";
            Console.WriteLine($"  {C($"#{t.Number,3}", "90")}  {C("✔ done  ", "32")}  loss {t.Value,8:F4}   best {C($"{best,8:F4}", "1;32")} {mark}");
        }
    }

    try
    {
        study.Optimize(Objective, nTrials: nTrials, callbacks: new[] { (Action<Study, FrozenTrial>)Render });
    }
    catch (Exception ex)
    {
        // a bridged hub sampler evaluates in Python; if it fails mid-run, keep whatever is already
        // persisted (the dashboard can still show it) and report instead of crashing.
        Console.WriteLine();
        Console.WriteLine(C("  optimization stopped early: " + ex.Message.Split('\n')[0], "1;33"));
    }

    int completed = study.Trials.Count(t => t.State == TrialState.Complete);
    int prunedN = study.Trials.Count(t => t.State == TrialState.Pruned);
    Console.WriteLine();
    Console.WriteLine(C($"  {study.Trials.Count} trials  ·  {completed} completed  ·  {prunedN} pruned", "1;37"));
    if (study.Trials.Any(t => t.State == TrialState.Complete))
    {
        Console.WriteLine(C($"  ✔ best loss {study.BestValue:F4}  at  " +
            string.Join(", ", study.BestParams.Select(kv => $"{kv.Key}={kv.Value}")), "1;32"));
    }
}
finally
{
    foreach (var d in owned) d.Dispose();     // dispose the hub sampler + module (no-op for native TPE)
}

Console.WriteLine();
Console.WriteLine(C($"  ✔ study persisted to  {url}", "1;32"));
Console.WriteLine(C("    explore it live in the real Python optuna-dashboard:", "2"));
Console.WriteLine(C($"      optuna-dashboard {url}", "36"));
Console.WriteLine(C("    (the run-optunahub launcher opens this for you automatically; it also runs", "2"));
Console.WriteLine(C("     examples/optunahub/dashboard.py, which starts the server + opens your browser.)", "2"));
Console.WriteLine();

// ── objective ──────────────────────────────────────────────────────────────────────────────────────

// a deterministic stand-in for training: `epochs` of validation loss decaying toward a floor set by the
// hyperparameters. mixed Float/Int/Categorical space + intermediate values → a rich dashboard (history,
// contour, parallel-coordinate, hyperparameter importance, intermediate-values). the MedianPruner reads
// the reported values and cuts trials that fall behind the running median.
static double Objective(Trial t)
{
    double lr = t.SuggestFloat("learning_rate", 1e-4, 1e-1, log: true);          // log-scale float
    long units = t.SuggestInt("n_units", 16, 256, log: true);                    // log-scale int
    double dropout = t.SuggestFloat("dropout", 0.0, 0.6);                         // linear float
    string optimizer = t.SuggestCategorical("optimizer", new[] { "adam", "sgd", "rmsprop" });  // categorical

    double offLr = Math.Log10(lr) + 2.0;                                          // 0 at the sweet spot lr = 1e-2
    double optPenalty = optimizer switch { "adam" => 0.0, "rmsprop" => 0.08, _ => 0.20 };  // sgd is worst
    double floor = 0.08
        + 0.12 * Math.Pow(offLr, 2.0)                                            // Math.Pow, not offLr*offLr, for x**2 parity
        + 0.40 * Math.Abs(dropout - 0.15)
        + 0.30 / Math.Log2(units)
        + optPenalty;
    double rate = 0.5 / (1.0 + Math.Abs(offLr));

    double loss = 0.0;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        loss = floor + (1.2 - floor) * Math.Exp(-rate * (epoch + 1));            // decreasing toward the floor
        t.Report(loss, epoch);                                                   // intermediate value -> pruner + dashboard
        if (t.ShouldPrune())                                                     // worse than the running median?
            throw new TrialPruned();                                             // -> recorded as PRUNED
    }
    return loss;
}

// ── helpers ──────────────────────────────────────────────────────────────────────────────────────

// the hub AutoSampler if it loads (network hub first, else the vendored submodule), else native TPE.
static (BaseSampler sampler, IDisposable[] owned, string label) BuildSampler()
{
    try
    {
        OptunaHubModule module;
        try { module = OptunaHub.LoadModule("samplers/auto_sampler"); }
        catch (OptunaHubInteropException)
        {
            string local = Path.Combine(RepoRoot(), "refs", "optunahub-registry", "package");
            module = OptunaHub.LoadLocalModule("samplers/auto_sampler", local);
        }
        var sampler = module.CreateSampler("AutoSampler", new() { ["seed"] = 0 });
        return (sampler, new IDisposable[] { sampler, module }, "AutoSampler (hub)");
    }
    catch (Exception ex)
    {
        Console.WriteLine("  " + (Environment.GetEnvironmentVariable("NO_COLOR") is null ? "\x1b[90m" : "")
            + "hub AutoSampler unavailable (" + ex.Message.Split('\n')[0] + ") — using native TPE"
            + (Environment.GetEnvironmentVariable("NO_COLOR") is null ? "\x1b[0m" : ""));
        return (new TPESampler(seed: 0), Array.Empty<IDisposable>(), "TPESampler (native)");
    }
}

static void DeleteDbFiles(string dbPath)
{
    foreach (var p in new[] { dbPath, dbPath + "-wal", dbPath + "-shm", dbPath + "-journal" })
        try { if (File.Exists(p)) File.Delete(p); } catch { /* locked by a prior dashboard? leave it — CreateStudy loadIfExists reuses */ }
}

static bool TryProbe(out string reason)
{
    try
    {
        PythonRuntime.Run(() => { _ = PythonRuntime.Optuna; _ = PythonRuntime.OptunaHubModuleRef; });
        reason = "";
        return true;
    }
    catch (Exception ex) { reason = ex.Message.Split('\n')[0]; return false; }
}

static string ThisDir([CallerFilePath] string? thisFile = null)
    => Path.GetDirectoryName(thisFile!)!;

static string RepoRoot([CallerFilePath] string? thisFile = null)
    => Path.GetFullPath(Path.Combine(Path.GetDirectoryName(thisFile!)!, "..", ".."));

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
