#:project ../../src/OptunaSharp.Interop.pythonnet/OptunaSharp.Interop.pythonnet.csproj
#:property PublishAot=false

// OptunaHub benchmark, live and in color — load a REAL OptunaHub benchmark (BBOB Sphere) and optimize
// it with an OptunaSharp sampler. this is the paper's "Code 1" run, in .NET: optunahub.load_module(
// "benchmarks/bbob") bridged to a C# Study by OptunaSharp.Interop.pythonnet. the objective is the real
// BBOB function evaluated in Python — no surrogate; OptunaSharp's native TPE sampler drives the search.
//
// source (paper):  https://arxiv.org/abs/2510.02798        (OptunaHub: A Platform for Black-Box Optimization)
// source (hub):    https://hub.optuna.org/benchmarks/bbob/
// source (bridge): src/OptunaSharp.Interop.pythonnet
//
// needs a Python with `pip install optuna optunahub coco-experiment` (BBOB wraps the COCO suite). the
// launcher installs these for you; this probes first and prints friendly guidance if anything is missing.
//
//   Run:  examples/optunahub/run-optunahub.sh     (Linux/macOS)   — installs deps + runs
//         examples\optunahub\run-optunahub.bat    (Windows)
//   or:   dotnet run examples/optunahub/OptunaHubBenchmark.cs
//
//   colors: honors NO_COLOR / FORCE_COLOR.

using System;
using System.Runtime.InteropServices;
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Logging;
using OptunaSharp.Interop.PythonNet;

Native.EnableVirtualTerminal();   // make ANSI colors work on legacy Windows consoles too
try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { /* redirected handle — leave as-is */ }
OptunaLogging.DisableDefaultHandler();   // silence OptunaSharp's own INFO logs — our colored view replaces them

// color helpers (ANSI). off when NO_COLOR is set or output is piped, unless FORCE_COLOR overrides.
bool color = Environment.GetEnvironmentVariable("FORCE_COLOR") is { Length: > 0 }
    || (Environment.GetEnvironmentVariable("NO_COLOR") is null && !Console.IsOutputRedirected);
string C(string s, string code) => color ? $"\x1b[{code}m{s}\x1b[0m" : s;

// this experiment needs Python + optunahub + the BBOB (COCO) suite. probe first, guide if missing.
if (!TryProbe(out string why))
{
    Console.WriteLine(C("OptunaHub benchmark demo — prerequisites missing", "1;33"));
    Console.WriteLine($"  {why}");
    Console.WriteLine("  install and re-run:");
    Console.WriteLine(C("    pip install optuna optunahub coco-experiment", "36"));
    Console.WriteLine("  or just use the launcher:");
    Console.WriteLine(C("    examples/optunahub/run-optunahub.sh   (examples\\optunahub\\run-optunahub.bat on Windows)", "36"));
    Console.WriteLine("  point pythonnet at a specific interpreter with  PYTHONNET_PYDLL=<libpython>.");
    return;
}

Console.WriteLine();
Console.WriteLine(C("  ╔══════════════════════════════════════════════════════════╗", "36"));
Console.WriteLine(C("  ║           OptunaHub  ×  OptunaSharp   —   live run        ║", "1;36"));
Console.WriteLine(C("  ╚══════════════════════════════════════════════════════════╝", "36"));
Console.WriteLine($"  optuna {C(PythonRuntime.OptunaVersion ?? "?", "32")}   optunahub {C(PythonRuntime.OptunaHubVersion ?? "?", "32")}   (embedded CPython via pythonnet)");
Console.WriteLine();

// ── load a real OptunaHub benchmark ──────────────────────────────────────────────────────────────
Console.Write("  loading " + C("benchmarks/bbob", "1;35") + " from OptunaHub … ");
OptunaHubModule bbob;
OptunaHubProblem problem;
try
{
    bbob = LoadBbob();                                        // network hub first, else the vendored submodule
    problem = bbob.CreateProblem("Problem",                  // BBOB f1 = the Sphere, in 2 dimensions
        new() { ["function_id"] = 1, ["dimension"] = 2 });
}
catch (Exception ex)
{
    // loading can still fail after the probe (no `git` on PATH, the submodule missing, a bbob/cocoex
    // load error, a network hiccup) — report it instead of crashing.
    Console.WriteLine(C("failed", "1;31"));
    Console.WriteLine(C("  could not load benchmarks/bbob: " + ex.Message.Split('\n')[0], "1;33"));
    Console.WriteLine("  ensure `pip install optunahub coco-experiment` succeeded, and either `git` is on");
    Console.WriteLine("  PATH (the default hub download) or the registry submodule is initialized:");
    Console.WriteLine(C("      git submodule update --init refs/optunahub-registry", "36"));
    return;
}
using var _bbob = bbob;                                        // dispose the loaded module on exit
Console.WriteLine(C("ok", "1;32"));

Console.WriteLine($"  problem     : {C("BBOB f1 (Sphere)", "1;37")}, {problem.SearchSpace.Count}-D");
Console.WriteLine($"  direction   : {C(problem.Directions[0].ToString(), "37")}");
Console.WriteLine($"  search space: {C(string.Join(", ", problem.SearchSpace.Keys) + " ∈ [-5, 5]", "37")}");
Console.WriteLine();
Console.WriteLine(C("  optimizing with OptunaSharp's native TPESampler …", "2"));
Console.WriteLine();

// ── optimize the hub problem with an OptunaSharp sampler, rendering each trial in color ───────────
double? runMin = null, runMax = null, best = null;
var study = Optuna.CreateStudy(
    sampler: new TPESampler(seed: 0),                          // OptunaSharp's own TPE drives the search
    directions: problem.Directions);                          // directions come straight from the hub problem

// a callback runs after each finished trial — we use it to draw a live, colored convergence bar
void Render(Study s, FrozenTrial t)
{
    double cur = t.Value ?? double.NaN;
    runMin = runMin is null ? cur : Math.Min(runMin.Value, cur);
    runMax = runMax is null ? cur : Math.Max(runMax.Value, cur);
    bool improved = best is null || cur < best;
    if (improved) best = cur;

    // bar length ∝ how good this trial is within the range seen so far (longer, greener = better)
    double span = runMax!.Value - runMin!.Value;
    double frac = span <= 0 ? 1.0 : (runMax.Value - cur) / span;    // 1 = best seen, 0 = worst seen
    int width = 34, fill = (int)Math.Round(frac * width);
    string barCode = frac > 0.66 ? "32" : frac > 0.33 ? "33" : "31";  // green / yellow / red
    string bar = C(new string('█', fill), barCode) + C(new string('░', width - fill), "90");

    string bestCell = improved ? C($"{best,11:F4} ★", "1;32") : C($"{best,11:F4}  ", "2");
    Console.WriteLine($"  {C($"#{t.Number,3}", "90")}  cur {cur,11:F4}  {bar}  best {bestCell}");
}

try
{
    study.Optimize(problem.SingleObjective, nTrials: 45, callbacks: new[] { (Action<Study, FrozenTrial>)Render });
}
catch (Exception ex)
{
    // a bridged hub problem evaluates in Python; if that fails mid-run, report it instead of crashing.
    Console.WriteLine();
    Console.WriteLine(C("  optimization stopped early: " + ex.Message.Split('\n')[0], "1;33"));
    return;
}

// ── result ───────────────────────────────────────────────────────────────────────────────────────
Console.WriteLine();
Console.WriteLine(C($"  ✔ best value  {study.BestValue:F6}", "1;32"));
Console.Write("    at ");
Console.WriteLine(C(string.Join(", ", System.Linq.Enumerable.Select(study.BestParams, kv => $"{kv.Key}={Convert.ToDouble(kv.Value):F4}")), "37"));
Console.WriteLine($"    over {study.Trials.Count} trials");
Console.WriteLine();
Console.WriteLine(C("  the same run in Python (the paper's Code 1):", "2"));
Console.WriteLine(C("    import optuna, optunahub", "90"));
Console.WriteLine(C("    bbob = optunahub.load_module(\"benchmarks/bbob\")", "90"));
Console.WriteLine(C("    problem = bbob.Problem(function_id=1, dimension=2)", "90"));
Console.WriteLine(C("    study = optuna.create_study(directions=problem.directions)", "90"));
Console.WriteLine(C("    study.optimize(problem, n_trials=45)", "90"));
Console.WriteLine();

// ── helpers ──────────────────────────────────────────────────────────────────────────────────────

// try the real hub over the network first (the genuine load_module experience, cached after first run);
// fall back to the pinned refs/optunahub-registry submodule so the example still runs offline in-repo.
OptunaHubModule LoadBbob()
{
    try { return OptunaHub.LoadModule("benchmarks/bbob"); }
    catch (OptunaHubInteropException)
    {
        string local = System.IO.Path.Combine(RepoRoot(), "refs", "optunahub-registry", "package");
        return OptunaHub.LoadLocalModule("benchmarks/bbob", local);
    }
}

// probe the whole chain (engine + optunahub + the COCO suite bbob needs) without throwing.
static bool TryProbe(out string reason)
{
    try
    {
        PythonRuntime.Run(() =>
        {
            _ = PythonRuntime.Optuna;                              // touch cached modules → populates versions
            _ = PythonRuntime.OptunaHubModuleRef;
            using var __ = Python.Runtime.Py.Import("cocoex");     // coco-experiment — BBOB's backend
        });
        reason = "";
        return true;
    }
    catch (Exception ex)
    {
        reason = ex.Message.Split('\n')[0];
        return false;
    }
}

static string RepoRoot([System.Runtime.CompilerServices.CallerFilePath] string? thisFile = null)
    => System.IO.Path.GetFullPath(System.IO.Path.Combine(System.IO.Path.GetDirectoryName(thisFile)!, "..", ".."));

// enable ANSI escape processing on Windows consoles (no-op elsewhere / on already-capable terminals).
static class Native
{
    const int STD_OUTPUT_HANDLE = -11;
    const uint ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004;

    [DllImport("kernel32.dll", SetLastError = true)] static extern IntPtr GetStdHandle(int nStdHandle);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool GetConsoleMode(IntPtr h, out uint mode);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool SetConsoleMode(IntPtr h, uint mode);

    public static void EnableVirtualTerminal()
    {
        if (!OperatingSystem.IsWindows())
            return;
        try
        {
            IntPtr h = GetStdHandle(STD_OUTPUT_HANDLE);
            if (GetConsoleMode(h, out uint mode))
                SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
        catch { /* best effort — colors just won't show on a console that refuses VT */ }
    }
}
