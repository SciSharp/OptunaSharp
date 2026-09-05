#:project ../../src/OptunaSharp.Interop.pythonnet/OptunaSharp.Interop.pythonnet.csproj
#:property PublishAot=false
#:property SkipNDScopeWeave=true
#:property GeneratePackageOnBuild=false

// OptunaHub local registry + mixed search space — two things at once:
//   1. LoadLocalModule: load packages from a LOCAL registry checkout (no network) — both a tiny problem
//      we write to a temp folder AND a real package from the vendored refs/optunahub-registry submodule.
//   2. all three distribution types across the bridge: the toy benchmark's search space mixes Float,
//      Int and Categorical, exercising PyConvert both directions (Python distributions -> OptunaSharp,
//      and suggested values -> Python for evaluate).
//
// source (paper):   https://arxiv.org/abs/2510.02798
// source (loader):  https://optuna.github.io/optunahub/   (load_module / load_local_module)
//
// needs `pip install optuna optunahub` (no COCO/cmaes here — the objective is a self-contained toy).
//
//   Run:  examples/optunahub/run-optunahub.sh
//   or:   dotnet run examples/optunahub/OptunaHubLocalRegistry.cs

using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Distributions;
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
    Console.WriteLine(C("OptunaHub local-registry demo — prerequisites missing", "1;33"));
    Console.WriteLine($"  {why}");
    Console.WriteLine(C("  pip install optuna optunahub", "36"));
    Console.WriteLine(C("  or:  examples/optunahub/run-optunahub.sh", "36"));
    return;
}

Console.WriteLine();
Console.WriteLine(C("  OptunaHub local registry  —  LoadLocalModule + a mixed Float/Int/Categorical space", "1;36"));
Console.WriteLine($"  optuna {C(PythonRuntime.OptunaVersion ?? "?", "32")}   optunahub {C(PythonRuntime.OptunaHubVersion ?? "?", "32")}");
Console.WriteLine();

string root = WriteToyRegistry();
try
{
    // 1) load a benchmark from a LOCAL registry folder we just wrote — no network, no GitHub.
    Console.WriteLine("  loading " + C("benchmarks/toy", "1;35") + " via " + C("LoadLocalModule", "1;34") + $" from {C(root, "90")}");
    using var mod = OptunaHub.LoadLocalModule("benchmarks/toy", root);
    var problem = mod.CreateProblem("Problem");

    // the search space marshals to OptunaSharp types — one of EACH kind
    Console.WriteLine("  search space (all three distribution kinds crossed the bridge):");
    foreach (var (name, dist) in problem.SearchSpace)
        Console.WriteLine($"    {C(name.PadRight(12), "37")} {C(DistLabel(dist), "36")}");
    Console.WriteLine();

    var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 0), directions: problem.Directions);
    study.Optimize(problem.SingleObjective, nTrials: 60);

    Console.WriteLine(C($"  ✔ best {study.BestValue:F5}", "1;32"));
    foreach (var kv in study.BestParams)
        Console.WriteLine($"      {C(kv.Key.PadRight(12), "37")} = {C(kv.Value?.ToString() ?? "None", "1;37")}  ({kv.Value?.GetType().Name ?? "null"})");
    Console.WriteLine();

    // 2) LoadLocalModule also works against the vendored refs/optunahub-registry submodule (offline).
    string submodule = Path.Combine(RepoRoot(), "refs", "optunahub-registry", "package");
    if (Directory.Exists(Path.Combine(submodule, "samplers", "random_search")))
    {
        using var rs = OptunaHub.LoadLocalModule("samplers/random_search", submodule);
        Console.WriteLine("  also loaded " + C("samplers/random_search", "1;35") + " from the vendored submodule — " +
            C($"HasAttr(RandomSampler)={rs.HasAttr("RandomSampler")}", "32"));
        Console.WriteLine();
    }
}
finally
{
    try { Directory.Delete(root, recursive: true); } catch { }
}

// ── loading OptunaHub packages from anywhere (the knobs) ─────────────────────────────────────────
Console.WriteLine(C("  where packages come from — the LoadModule knobs:", "2"));
Console.WriteLine(C("    OptunaHub.LoadModule(\"samplers/cmaes\")                                  // official registry (network)", "90"));
Console.WriteLine(C("    OptunaHub.LoadModule(\"samplers/cmaes\", gitRef: \"v0.4.1\")                 // pin a registry tag/branch/SHA", "90"));
Console.WriteLine(C("    OptunaHub.LoadModule(\"samplers/x\", repoOwner: \"you\", repoName: \"my-hub\")  // your own / a fork's registry", "90"));
Console.WriteLine(C("    OptunaHub.LoadModule(\"samplers/x\", baseUrl: \"https://ghe.example.com/api/v3\")  // GitHub Enterprise / GitLab", "90"));
Console.WriteLine(C("    OptunaHub.LoadLocalModule(\"benchmarks/toy\", \"/path/to/registry/package\")  // a local checkout, no network", "90"));
Console.WriteLine();

// ── helpers ──────────────────────────────────────────────────────────────────────────────────────

static string DistLabel(BaseDistribution d) => d switch
{
    FloatDistribution f => $"FloatDistribution(low={f.Low}, high={f.High}, log={f.Log})",
    IntDistribution i => $"IntDistribution(low={i.Low}, high={i.High}, step={i.Step})",
    CategoricalDistribution c => $"CategoricalDistribution(choices=[{string.Join(", ", c.Choices.Select(ch => ch.Value))}])",
    _ => d.GetType().Name,
};

// a dependency-free BaseProblem with one Float, one Int and one Categorical parameter.
static string WriteToyRegistry()
{
    string root = Path.Combine(Path.GetTempPath(), "optunasharp_localreg_" + Guid.NewGuid().ToString("N"));
    string dir = Path.Combine(root, "benchmarks", "toy");
    Directory.CreateDirectory(dir);
    File.WriteAllText(Path.Combine(dir, "__init__.py"),
        "from __future__ import annotations\n" +
        "from typing import Any\n" +
        "import math, optuna, optunahub\n" +
        "\n" +
        "class Problem(optunahub.benchmarks.BaseProblem):\n" +
        "    @property\n" +
        "    def search_space(self):\n" +
        "        return {\n" +
        "            'lr': optuna.distributions.FloatDistribution(1e-4, 1e-1, log=True),\n" +
        "            'layers': optuna.distributions.IntDistribution(1, 8),\n" +
        "            'activation': optuna.distributions.CategoricalDistribution(['relu', 'tanh', 'gelu']),\n" +
        "        }\n" +
        "    @property\n" +
        "    def directions(self):\n" +
        "        return [optuna.study.StudyDirection.MINIMIZE]\n" +
        "    def evaluate(self, params: dict[str, Any]) -> float:\n" +
        "        # a toy 'validation loss', minimized near lr=1e-2, layers=4, activation='relu'\n" +
        "        lr_term = (math.log10(params['lr']) + 2.0) ** 2\n" +
        "        layer_term = 0.1 * (params['layers'] - 4) ** 2\n" +
        "        act_term = {'relu': 0.0, 'tanh': 0.3, 'gelu': 0.15}[params['activation']]\n" +
        "        return lr_term + layer_term + act_term\n");
    return root;
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
