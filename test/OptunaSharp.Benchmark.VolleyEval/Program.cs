using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using OptunaSharp.Benchmark.VolleyEval;
using OptunaSharp.Logging;

// ======================================================================================
// VolleyEval v0 CLI — `volley run` prints the three-radical profile matrix.
//
//   dotnet run -c Release -- run
//   dotnet run -c Release -- run --seeds 5 --assert
//   dotnet run -c Release -- run --cheap --seeds 3           (fast: excludes the cubic GP)
//   dotnet run -c Release -- run --challenges rosenbrock,zdt1 --contenders cmaes,nsga2
//
// Flags: --seeds N  --instance N  --challenges a,b,c  --contenders x,y,z  --cheap
//        --assert (exit non-zero if a DoD gate fails)  --max-seconds S  --verbose
// ======================================================================================

var argv = args.ToList();
string verb = argv.Count > 0 && !argv[0].StartsWith("--", StringComparison.Ordinal) ? argv[0] : "run";

if (verb is "-h" or "--help" or "help")
{
    Console.WriteLine("volley run [--seeds N] [--instance N] [--challenges a,b,c] [--contenders x,y,z] [--cheap] [--assert] [--max-seconds S] [--verbose]");
    return 0;
}
if (verb != "run")
{
    Console.Error.WriteLine($"Unknown verb '{verb}'. Try: volley run");
    return 2;
}

int seeds = GetInt("--seeds", 5);
int instance = GetInt("--instance", 0);
bool doAssert = HasFlag("--assert");
bool cheap = HasFlag("--cheap");
bool verbose = HasFlag("--verbose");
double? maxSeconds = HasOption("--max-seconds") ? GetDouble("--max-seconds", 0) : null;

// Optuna's INFO/WARNING lines (independent-sampling fallbacks, experimental notices) would bury the
// matrix; silence the default handler unless the user asks for it.
if (!verbose) OptunaLogging.DisableDefaultHandler();

IReadOnlyList<IChallenge> challenges = HasOption("--challenges")
    ? ChallengeRegistry.Select(GetList("--challenges"), instance)
    : ChallengeRegistry.Default(instance);

IReadOnlyList<Contender> contenders =
    HasOption("--contenders") ? ContenderRegistry.Select(GetList("--contenders"))
    : cheap ? ContenderRegistry.Cheap
    : ContenderRegistry.All;

var seedArray = Enumerable.Range(0, seeds).ToArray();

Console.WriteLine($"VolleyEval v0 — {challenges.Count} challenges × {contenders.Count} contenders × {seeds} seeds"
                  + (maxSeconds is double s ? $"  (≤{s}s/run)" : ""));
Console.WriteLine($"challenges: {string.Join(", ", challenges.Select(c => $"{c.Name}(D={c.Dimension},b={c.Budget})"))}");
Console.WriteLine($"contenders: {string.Join(", ", contenders.Select(c => c.Name))}");
Console.WriteLine();

var evaluator = new Evaluator(seedArray, maxSeconds);
var matrix = evaluator.Run(challenges, contenders, onCell: cell =>
{
    string status = cell.Dnf ? $"DNF({cell.Error})" : cell.Median.ToString("F4", CultureInfo.InvariantCulture);
    Console.WriteLine($"  - {cell.Challenge,-14} {cell.Contender,-8} {status,-10} " +
                      $"({cell.WallSeconds,5:F1}s, {cell.EffectiveBudget} trials)");
});

Console.WriteLine();
Console.WriteLine(matrix.Render());
Console.WriteLine(DodGate.Render(matrix));

if (doAssert && !DodGate.AllPassed(matrix))
{
    Console.Error.WriteLine();
    Console.Error.WriteLine("ASSERT FAILED: one or more DoD gates did not pass.");
    return 1;
}
return 0;

// ---- tiny arg helpers ----
bool HasFlag(string name) => argv.Contains(name);
bool HasOption(string name) => argv.IndexOf(name) is int i && i >= 0 && i + 1 < argv.Count;
string? GetOption(string name)
{
    int i = argv.IndexOf(name);
    return i >= 0 && i + 1 < argv.Count ? argv[i + 1] : null;
}
int GetInt(string name, int fallback) => GetOption(name) is string v ? int.Parse(v, CultureInfo.InvariantCulture) : fallback;
double GetDouble(string name, double fallback) => GetOption(name) is string v ? double.Parse(v, CultureInfo.InvariantCulture) : fallback;
List<string> GetList(string name) => (GetOption(name) ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
