#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false
#:property SkipNDScopeWeave=true
#:property GeneratePackageOnBuild=false

// Multi-objective optimization — optimize two competing goals at once and recover the Pareto front
// (the set of non-dominated trade-offs) instead of a single best trial. here we tune a model for both
// low validation error AND small size with the NSGA-II genetic sampler, then print study.BestTrials.
//
// source (tutorial): https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html
// source (example):  https://github.com/optuna/optuna-examples/blob/main/multi_objective/pytorch_simple.py
//
// the two objectives are a deterministic surrogate (error falls with capacity; size grows with it), so
// the example runs anywhere; a real "accuracy vs. #features" sklearn objective is the PackageParity track.
//
//   Run:  dotnet run examples/MultiObjectivePareto.cs

using System.Collections.Generic;
using System.Linq;
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

var study = Optuna.CreateStudy(
    storage: new InMemoryStorage(),
    sampler: new NSGAIISampler(populationSize: 20, seed: 0),          // NSGA-II: a genetic sampler built for many objectives
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },   // minimize both (error, size)
    studyName: "multi-objective");

study.Optimize(Objective, nTrials: 60);

// BestTrials is the Pareto front — every trial no other trial beats on BOTH objectives at once
var front = study.BestTrials.OrderBy(t => t.Values![0]).ToList();     // sort by error for a readable trade-off curve
Console.WriteLine($"Pareto front: {front.Count} non-dominated trade-offs (of {study.Trials.Count} trials)");
Console.WriteLine($"  {"error",-10}{"size",-10}params");
foreach (var t in front)
    // each row is one trade-off: less error usually costs more size, and vice-versa
    Console.WriteLine($"  {t.Values![0],-10:F4}{t.Values![1],-10:F0}" +
                      $"n_layers={t.Params["n_layers"]}, n_units={t.Params["n_units"]}, dropout={t.Params["dropout"]:F2}");

// objective 1: validation error — falls as capacity (layers × log2 units) grows, best near dropout 0.2.
// objective 2: model size — grows with layers × units. the two pull against each other → a real front.
static IReadOnlyList<double> Objective(Trial t)
{
    long nLayers = t.SuggestInt("n_layers", 1, 6);
    long nUnits = t.SuggestInt("n_units", 16, 512, log: true);        // log scale over the width
    double dropout = t.SuggestFloat("dropout", 0.0, 0.6);

    double capacity = nLayers * Math.Log2(nUnits);                    // bigger net → more capacity
    double error = 0.5 * Math.Exp(-0.15 * capacity) + 0.05 * Math.Abs(dropout - 0.2);
    double size = nLayers * nUnits;                                   // the cost that competes with error
    return new[] { error, size };                                     // return one value per direction, in order
}
