#:project ../../src/OptunaSharp.Interop.pythonnet/OptunaSharp.Interop.pythonnet.csproj
#:property PublishAot=false
#:property SkipNDScopeWeave=true
#:property GeneratePackageOnBuild=false

// OptunaHub multi-objective — optimize a real BI-objective hub benchmark (bbob_biobj) and recover the
// Pareto front. this exercises the bridge's MULTI-objective path: OptunaHubProblem.Objective returns a
// vector (one value per direction), the study carries two directions from the hub problem, and
// OptunaSharp's native NSGA-II sampler drives the search. the objective is the real COCO bi-objective
// BBOB function evaluated in Python.
//
// source (paper):     https://arxiv.org/abs/2510.02798
// source (benchmark): https://hub.optuna.org/benchmarks/bbob_biobj/
//
// needs `pip install optuna optunahub coco-experiment`. probes first and guides if anything is missing.
//
//   Run:  examples/optunahub/run-optunahub.sh
//   or:   dotnet run examples/optunahub/OptunaHubMultiObjective.cs

using System;
using System.Collections.Generic;
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

if (!TryProbe(out string why))
{
    Console.WriteLine(C("OptunaHub multi-objective demo — prerequisites missing", "1;33"));
    Console.WriteLine($"  {why}");
    Console.WriteLine(C("  pip install optuna optunahub coco-experiment", "36"));
    Console.WriteLine(C("  or:  examples/optunahub/run-optunahub.sh", "36"));
    return;
}

Console.WriteLine();
Console.WriteLine(C("  OptunaHub multi-objective  —  bbob_biobj (2 objectives)  ×  native NSGA-II", "1;36"));
Console.WriteLine($"  optuna {C(PythonRuntime.OptunaVersion ?? "?", "32")}   optunahub {C(PythonRuntime.OptunaHubVersion ?? "?", "32")}");
Console.WriteLine();

try
{
    Console.Write("  loading " + C("benchmarks/bbob_biobj", "1;35") + " … ");
    using var mod = Load("benchmarks/bbob_biobj");
    var problem = mod.CreateProblem("Problem", new() { ["function_id"] = 1, ["dimension"] = 3 });
    Console.WriteLine(C("ok", "1;32"));
    Console.WriteLine($"  directions : {C(string.Join(", ", problem.Directions), "37")}   ({problem.Directions.Count} objectives)");
    Console.WriteLine($"  variables  : {C(string.Join(", ", problem.SearchSpace.Keys), "37")}");
    Console.WriteLine(C("  optimizing with OptunaSharp's native NSGAIISampler …", "2"));
    Console.WriteLine();

    var study = Optuna.CreateStudy(
        sampler: new NSGAIISampler(populationSize: 16, seed: 0),   // a genetic sampler built for many objectives
        directions: problem.Directions);                          // two directions, straight from the hub problem
    study.Optimize(problem.Objective, nTrials: 100);              // Objective returns one value per direction

    // BestTrials is the Pareto front — trials no other trial beats on BOTH objectives at once
    var front = study.BestTrials.OrderBy(t => t.Values![0]).ToList();
    double f0min = front.Min(t => t.Values![0]), f0max = front.Max(t => t.Values![0]);
    Console.WriteLine(C($"  Pareto front: {front.Count} non-dominated trade-offs (of {study.Trials.Count} trials)", "1;37"));
    Console.WriteLine($"  {C($"  {"f0",-12}{"f1",-12}", "90")}");
    foreach (var t in front)
    {
        double f0 = t.Values![0], f1 = t.Values![1];
        double frac = f0max > f0min ? (f0 - f0min) / (f0max - f0min) : 0.0;   // position along the front
        int pos = (int)Math.Round(frac * 24);
        string track = C(new string('·', pos), "90") + C("●", "36") + C(new string('·', 24 - pos), "90");
        Console.WriteLine($"    {f0,-12:F3}{f1,-12:F3}{track}");
    }
    Console.WriteLine();
    Console.WriteLine(C("  each ● is one non-dominated trade-off: lower f0 usually costs higher f1, and vice-versa.", "2"));
    Console.WriteLine();
}
catch (Exception ex)
{
    Console.WriteLine(C("failed", "1;31"));
    Console.WriteLine(C("  " + ex.Message.Split('\n')[0], "1;33"));
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
        PythonRuntime.Run(() =>
        {
            _ = PythonRuntime.Optuna;
            _ = PythonRuntime.OptunaHubModuleRef;
            using var _c = Python.Runtime.Py.Import("cocoex");
        });
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
