#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Conditional pipeline chain — a network built as an ARRAY of stages whose length the search discovers.
// Each stage is an ENUM (`Op`: Conv / Pool / Dense / Stop), and each stage gates the NEXT one: `op_1`
// exists only if `op_0` did not Stop, `op_2` only if `op_1` did not Stop, and so on. That "one parameter
// activates the next" is the optional chain — the pipeline grows stage by stage until an `Op.Stop`
// (or the cap) ends it. A Conv/Dense stage additionally activates its own `width_i`; Pool and Stop have none.
//
// This is the example to read for chains built PROGRAMMATICALLY over an array, and for enum-typed rules
// (the activator values are `nameof(Op.Conv)` &c., so the rules refactor with the enum). It also turns on
// `PreserveDormantValues`: as evolution reshapes the chain, a stage that briefly switches to Stop parks
// its learned width and resumes it when it switches back on — the gene list is fluid, but non-destructively.
//
// source (define-by-run, a stage per loop iteration):  https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html
// source (layers-as-a-list NAS example):                https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
//
// Deterministic surrogate (a known-good pipeline shape), multi-objective (quality vs cost) so chains of
// every length share the Pareto front and you can watch the chain grow. Runs anywhere.
//
//   Run:  dotnet run examples/ConditionalPipelineChain.cs

using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.NsgaiiiCpt;
using OptunaSharp.SearchSpace;
using OptunaSharp.Storages;

// build the chain rules over the stage array. read this next to Objective's loop — the two mirror each other.
var rules = new ConditionalParams { PreserveDormantValues = true };   // dormant stages keep (and resume) their width
string[] keepGoing = { nameof(Op.Conv), nameof(Op.Pool), nameof(Op.Dense) };   // any non-Stop op continues the chain
for (int i = 1; i < Net.MaxStages; i++)
    rules.When($"op_{i - 1}", keepGoing).Activates($"op_{i}");        // <- the link: stage i exists only if stage i-1 didn't Stop
for (int i = 0; i < Net.MaxStages; i++)
    rules.When($"op_{i}", nameof(Op.Conv), nameof(Op.Dense)).Activates($"width_{i}");   // only Conv/Dense stages have a width

OptunaLogging.SetVerbosity(LogLevel.Warning);
var study = Optuna.CreateStudy(
    storage: new InMemoryStorage(),
    sampler: new NSGAIIICptSampler(populationSize: 24, seed: 0, conditionalParams: rules),   // <- wires the chain in
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });             // minimize (error, cost)
study.Optimize(Objective, nTrials: 480);

// one line per chain length on the Pareto front — the shortest-error pipeline at each length. reading down
// the rows you watch the optional chain extend: Conv, then Conv -> Pool, then Conv -> Pool -> Dense, ...
Console.WriteLine("best pipeline at each chain length (Pareto front of error vs cost):");
foreach (var t in study.BestTrials
             .GroupBy(t => Pipeline(t).Count)
             .OrderBy(g => g.Key)
             .Select(g => g.MinBy(t => t.Values![0])!))
{
    var stages = Pipeline(t);
    string chain = stages.Count == 0 ? "(empty)"
        : string.Join(" -> ", stages.Select(s => s.Width is { } w ? $"{s.Op}({w})" : s.Op.ToString()));
    Console.WriteLine($"  {stages.Count} stage(s)   err {t.Values![0]:F3}  cost {t.Values![1],5:F0}   {chain}");
}

// walk the chain exactly as the rules gate it: op_i, its optional width, and Stop terminates. every active
// `SuggestInt("width_i")` here is the width rule above; the `if (op == Op.Stop) break` is the chain link.
static IReadOnlyList<double> Objective(Trial t)
{
    double quality = 0.0;   // higher is better (we minimize 0.6 - quality)
    double cost = 0.0;
    for (int i = 0; i < Net.MaxStages; i++)
    {
        Op op = Enum.Parse<Op>(t.SuggestCategorical($"op_{i}", Net.OpNames));   // the enum stage — an array element
        if (op == Op.Stop)
            break;                                                              // Stop ends the chain: no op_{i+1}
        if (op is Op.Conv or Op.Dense)
        {
            long width = t.SuggestInt($"width_{i}", 16, 256, log: true);        // this stage's width (optimum 96)
            quality += (op == Op.Conv ? 0.16 : 0.13) - Math.Pow((width - 96) / 96.0, 2.0) * 0.06;
            cost += width * (op == Op.Conv ? 4 : 1);                            // conv is the pricier op
        }
        else // Pool — cheap, a small steady gain, no width of its own
        {
            quality += 0.06;
            cost += 8;
        }
    }
    return new[] { Math.Max(0.6 - quality, 0.02), cost };   // (error, cost), both minimized
}

// reconstruct a trial's pipeline from its recorded params — present op_i keys, in order, up to Stop. this
// is possible precisely because the rules recorded exactly the stages that ran (no more, no fewer).
static List<(Op Op, long? Width)> Pipeline(FrozenTrial t)
{
    var stages = new List<(Op, long?)>();
    for (int i = 0; i < Net.MaxStages; i++)
    {
        if (!t.Params.TryGetValue($"op_{i}", out var raw))   // op_i absent -> the previous stage Stopped (chain ended)
            break;
        var op = Enum.Parse<Op>((string)raw!);
        if (op == Op.Stop)
            break;
        long? width = t.Params.TryGetValue($"width_{i}", out var w) ? Convert.ToInt64(w) : null;
        stages.Add((op, width));
    }
    return stages;
}

// the stage vocabulary. Stop is the sentinel that ends the chain; Conv/Dense carry a width, Pool does not.
enum Op { Conv, Pool, Dense, Stop }

// shared config, so both the top-level rule loop and the static methods can see it (a static local function
// cannot capture top-level locals, but it can read a type's members).
static class Net
{
    public const int MaxStages = 5;                          // the chain cap — the array length
    public static readonly string[] OpNames = Enum.GetNames<Op>();
}
