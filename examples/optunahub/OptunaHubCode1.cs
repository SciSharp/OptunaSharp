#:project ../../src/OptunaSharp.Interop.pythonnet/OptunaSharp.Interop.pythonnet.csproj
#:property PublishAot=false

// OptunaHub "Code 1" — a hub SAMPLER and a hub BENCHMARK together, the paper's headline example, in
// .NET. LoadModule("samplers/auto_sampler") + LoadModule("benchmarks/bbob"), both bridged so an
// OptunaSharp Study drives the real Python AutoSampler over the real BBOB Sphere. Nothing native here
// except the study loop — both the search strategy and the objective come from OptunaHub.
//
// source (paper Code 1): https://arxiv.org/abs/2510.02798
// source (sampler):      https://hub.optuna.org/samplers/auto_sampler/
// source (benchmark):    https://hub.optuna.org/benchmarks/bbob/
//
// needs `pip install optuna optunahub coco-experiment cmaes` (AutoSampler picks CMA-ES/TPE/GP; BBOB
// wraps COCO). probes first and guides if anything is missing.
//
//   Run:  examples/optunahub/run-optunahub.sh   (installs deps + runs the whole suite)
//   or:   dotnet run examples/optunahub/OptunaHubCode1.cs

using System;
using System.Linq;
using System.Runtime.InteropServices;
using OptunaSharp;
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
    Console.WriteLine(C("OptunaHub Code 1 demo — prerequisites missing", "1;33"));
    Console.WriteLine($"  {why}");
    Console.WriteLine(C("  pip install optuna optunahub coco-experiment cmaes", "36"));
    Console.WriteLine(C("  or:  examples/optunahub/run-optunahub.sh", "36"));
    return;
}

Console.WriteLine();
Console.WriteLine(C("  OptunaHub \"Code 1\"  —  a hub sampler + a hub benchmark, together", "1;36"));
Console.WriteLine($"  optuna {C(OptunaHubRuntime.OptunaVersion ?? "?", "32")}   optunahub {C(OptunaHubRuntime.OptunaHubVersion ?? "?", "32")}");
Console.WriteLine();

try
{
    Console.Write("  loading " + C("samplers/auto_sampler", "1;35") + " + " + C("benchmarks/bbob", "1;35") + " … ");
    using var autoMod = Load("samplers/auto_sampler");
    using var bbobMod = Load("benchmarks/bbob");
    using var sampler = autoMod.CreateSampler("AutoSampler", new() { ["seed"] = 0 });   // the hub SAMPLER
    var problem = bbobMod.CreateProblem("Problem",                                       // the hub BENCHMARK
        new() { ["function_id"] = 1, ["dimension"] = 2 });
    Console.WriteLine(C("ok", "1;32"));
    Console.WriteLine($"  sampler = {C("AutoSampler (hub)", "1;35")}   objective = {C("BBOB f1 Sphere (hub)", "1;35")}, 2-D");
    Console.WriteLine();

    var study = Optuna.CreateStudy(sampler: sampler, directions: problem.Directions);
    double? best = null;
    void Render(Study s, FrozenTrial t)
    {
        bool improved = best is null || (t.Value is { } v && v < best);
        if (t.Value is { } val && improved) best = val;
        string mark = improved ? C("★", "1;32") : " ";
        Console.WriteLine($"  {C($"#{t.Number,3}", "90")}  value {t.Value,11:F4}   best {C($"{best,11:F4}", improved ? "1;32" : "2")} {mark}");
    }
    study.Optimize(problem.SingleObjective, nTrials: 40, callbacks: new[] { (Action<Study, FrozenTrial>)Render });

    Console.WriteLine();
    Console.WriteLine(C($"  ✔ best {study.BestValue:F6}  at  " +
        string.Join(", ", study.BestParams.Select(kv => $"{kv.Key}={Convert.ToDouble(kv.Value):F4}")), "1;32"));
    Console.WriteLine();
    Console.WriteLine(C("  the same run in Python (the paper's Code 1):", "2"));
    Console.WriteLine(C("    import optuna, optunahub", "90"));
    Console.WriteLine(C("    auto = optunahub.load_module(\"samplers/auto_sampler\")", "90"));
    Console.WriteLine(C("    bbob = optunahub.load_module(\"benchmarks/bbob\")", "90"));
    Console.WriteLine(C("    problem = bbob.Problem(function_id=1, dimension=2)", "90"));
    Console.WriteLine(C("    study = optuna.create_study(sampler=auto.AutoSampler(), directions=problem.directions)", "90"));
    Console.WriteLine(C("    study.optimize(problem, n_trials=40)", "90"));
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
            _ = OptunaHubRuntime.Optuna;
            _ = OptunaHubRuntime.OptunaHub;
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
