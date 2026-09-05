#:project ../../src/OptunaSharp.Interop.pythonnet/OptunaSharp.Interop.pythonnet.csproj
#:property PublishAot=false
#:property SkipNDScopeWeave=true
#:property GeneratePackageOnBuild=false

// OptunaHub sampler shootout, in color — race several REAL OptunaHub samplers (loaded through the
// bridge) against OptunaSharp's own native TPE on the 2-D Rastrigin function, and draw each one's
// convergence as a colored sparkline + a final leaderboard. every "(hub)" sampler is the genuine
// Python implementation from https://hub.optuna.org/, driven by an OptunaSharp Study via
// OptunaSharp.Interop.pythonnet's shadow-study sampler bridge.
//
// source (paper):       https://arxiv.org/abs/2510.02798
// source (hub sampler): https://hub.optuna.org/samplers/nelder_mead/  ·  /samplers/cmaes/  ·  /samplers/random_search/
// source (bridge):      src/OptunaSharp.Interop.pythonnet
//
// the objective is the real Rastrigin (a genuine closed-form multimodal function — not a surrogate).
// needs a Python with `pip install optuna optunahub`; the NelderMead/CMA-ES hub samplers add `pandas` /
// `cmaes` — any hub sampler whose deps are missing is simply skipped, so the shootout always runs.
//
//   Run:  examples/optunahub/run-optunahub.sh     (Linux/macOS — installs deps + runs)
//         examples\optunahub\run-optunahub.bat    (Windows)
//   or:   dotnet run examples/optunahub/OptunaHubSamplerShootout.cs
//
//   colors: honors NO_COLOR / FORCE_COLOR.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Logging;
using OptunaSharp.Interop.PythonNet;

Native.EnableVirtualTerminal();
try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { /* redirected handle — leave as-is */ }
OptunaLogging.DisableDefaultHandler();   // silence OptunaSharp's own INFO logs — our colored view replaces them

bool color = Environment.GetEnvironmentVariable("FORCE_COLOR") is { Length: > 0 }
    || (Environment.GetEnvironmentVariable("NO_COLOR") is null && !Console.IsOutputRedirected);
string C(string s, string code) => color ? $"\x1b[{code}m{s}\x1b[0m" : s;

const int nTrials = 50;

// the engine + optunahub must be present; individual sampler deps are handled per-entry (skipped).
if (!TryProbe(out string why))
{
    Console.WriteLine(C("OptunaHub sampler shootout — prerequisites missing", "1;33"));
    Console.WriteLine($"  {why}");
    Console.WriteLine(C("  pip install optuna optunahub pandas cmaes", "36"));
    Console.WriteLine(C("  or:  examples/optunahub/run-optunahub.sh   (examples\\optunahub\\run-optunahub.bat on Windows)", "36"));
    return;
}

Console.WriteLine();
Console.WriteLine(C("  ╔══════════════════════════════════════════════════════════╗", "35"));
Console.WriteLine(C("  ║        OptunaHub sampler shootout  —  2-D Rastrigin       ║", "1;35"));
Console.WriteLine(C("  ╚══════════════════════════════════════════════════════════╝", "35"));
Console.WriteLine($"  optuna {C(PythonRuntime.OptunaVersion ?? "?", "32")}   optunahub {C(PythonRuntime.OptunaHubVersion ?? "?", "32")}   ·   {nTrials} trials each   ·   lower is better");
Console.WriteLine();

// the roster: native OptunaSharp samplers + real OptunaHub samplers (via the bridge). each factory
// returns the sampler plus anything to dispose after the run; a hub sampler with missing deps throws
// and is reported as skipped, so the shootout always produces a result.
var roster = new (string Label, string Code, bool Hub, Func<(BaseSampler, IDisposable[])> Make)[]
{
    ("TPE (native)",      "36", false, () => (new TPESampler(seed: 0), Array.Empty<IDisposable>())),
    ("Random (hub)",      "34", true,  () => MakeHub("samplers/random_search", "RandomSampler")),
    ("NelderMead (hub)",  "33", true,  () => MakeHub("samplers/nelder_mead",  "NelderMeadSampler")),
    ("CMA-ES (hub)",      "35", true,  () => MakeHub("samplers/cmaes",        "CmaEsSampler")),
};

var results = new List<(string Label, string Code, double Best, List<double> Curve)>();
var skipped = new List<(string Label, string Why)>();

foreach (var entry in roster)
{
    Console.Write($"  running {C(entry.Label.PadRight(18), entry.Code)} ");
    IDisposable[] owned = Array.Empty<IDisposable>();
    try
    {
        // load may fail (package/deps missing). the whole run is inside the try because a bridged
        // sampler can ALSO fail during optimization, not just at load — e.g. optuna's CmaEsSampler
        // imports its `cmaes` backend lazily on the first sample_relative — so any failure just skips
        // this sampler instead of crashing the shootout.
        BaseSampler sampler;
        (sampler, owned) = entry.Make();

        var curve = new List<double>(nTrials);
        double? best = null;
        var study = Optuna.CreateStudy(sampler: sampler, direction: StudyDirection.Minimize);
        study.Optimize(Rastrigin, nTrials: nTrials, callbacks: new[] { (Action<Study, FrozenTrial>)((s, t) =>
        {
            double v = t.Value ?? double.NaN;
            best = best is null ? v : Math.Min(best.Value, v);   // track best-so-far for the convergence curve
            curve.Add(best.Value);
        }) });

        results.Add((entry.Label, entry.Code, study.BestValue, curve));
        Console.WriteLine(C($"done  best={study.BestValue:F4}", entry.Code));
    }
    catch (Exception ex)
    {
        string reason = FirstLine(ex.Message);
        skipped.Add((entry.Label, reason));
        Console.WriteLine(C("skipped — " + reason, "90"));
    }
    finally
    {
        foreach (var d in owned) d.Dispose();
    }
}

// ── leaderboard with colored convergence sparklines ──────────────────────────────────────────────
Console.WriteLine();
Console.WriteLine(C("  leaderboard (best value, and how each converged over the run):", "1;37"));
Console.WriteLine();

// normalize sparklines across ALL curves so the heights are comparable between samplers
double gMin = results.SelectMany(r => r.Curve).DefaultIfEmpty(0).Min();
double gMax = results.SelectMany(r => r.Curve).DefaultIfEmpty(1).Max();

int rank = 1;
foreach (var r in results.OrderBy(r => r.Best))
{
    string medal = rank == 1 ? "🏆" : $"{rank}.";
    string label = rank == 1 ? C(r.Label.PadRight(18), "1;32") : C(r.Label.PadRight(18), r.Code);
    string bestCell = rank == 1 ? C($"{r.Best,10:F4}", "1;32") : $"{r.Best,10:F4}";
    Console.WriteLine($"  {medal,-3} {label} best {bestCell}   {C(Spark(r.Curve, gMin, gMax), r.Code)}");
    rank++;
}

foreach (var s in skipped)
    Console.WriteLine(C($"   –  {s.Label,-18} skipped: {s.Why}", "90"));

Console.WriteLine();
Console.WriteLine(C("  every \"(hub)\" row ran the real Python sampler from hub.optuna.org, driving a C# study.", "2"));
Console.WriteLine();

// ── objective + helpers ──────────────────────────────────────────────────────────────────────────

// the 2-D Rastrigin function: highly multimodal, global minimum 0 at the origin. a real, deterministic
// objective (no surrogate) — a classic stress test that separates good samplers from random search.
static double Rastrigin(Trial t)
{
    double x = t.SuggestFloat("x", -5.12, 5.12);
    double y = t.SuggestFloat("y", -5.12, 5.12);
    // 10*n + Σ(xi² - 10·cos(2π·xi)); Math.Pow, not x*x, to match Python x**2 bit-for-bit.
    return 20.0
        + (Math.Pow(x, 2.0) - 10.0 * Math.Cos(2.0 * Math.PI * x))
        + (Math.Pow(y, 2.0) - 10.0 * Math.Cos(2.0 * Math.PI * y));
}

// load a hub sampler (network hub first, else the pinned refs/optunahub-registry submodule) and wrap it.
static (BaseSampler, IDisposable[]) MakeHub(string package, string className)
{
    OptunaHubModule module;
    try { module = OptunaHub.LoadModule(package); }
    catch (OptunaHubInteropException)
    {
        string local = System.IO.Path.Combine(RepoRoot(), "refs", "optunahub-registry", "package");
        module = OptunaHub.LoadLocalModule(package, local);
    }
    var sampler = module.CreateSampler(className, new() { ["seed"] = 0 });
    return (sampler, new IDisposable[] { sampler, module });
}

// an 8-level unicode sparkline of a best-so-far curve; taller = worse, so convergence reads as a drop.
static string Spark(List<double> curve, double lo, double hi)
{
    const string ramp = "▁▂▃▄▅▆▇█";
    double span = hi - lo;
    var chars = new char[curve.Count];
    for (int i = 0; i < curve.Count; i++)
    {
        double norm = span <= 0 ? 0 : (curve[i] - lo) / span;         // 0..1, high value = tall bar
        chars[i] = ramp[Math.Clamp((int)Math.Round(norm * (ramp.Length - 1)), 0, ramp.Length - 1)];
    }
    return new string(chars);
}

static string FirstLine(string s) => (s.Split('\n')[0]).Trim();

static bool TryProbe(out string reason)
{
    try
    {
        PythonRuntime.Run(() => { _ = PythonRuntime.Optuna; _ = PythonRuntime.OptunaHubModuleRef; });
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
        catch { /* best effort */ }
    }
}
