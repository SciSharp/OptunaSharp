#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false
#:property SkipNDScopeWeave=true
#:property GeneratePackageOnBuild=false

// Multi-objective optimization — Optuna's flagship feature: optimize two competing goals at once and
// recover the Pareto front (the set of non-dominated trade-offs) instead of a single best trial. Here
// we tune a model for both low validation error AND small size, with the NSGA-II genetic sampler, and
// print study.BestTrials — the Pareto-optimal set.
//
// The two objectives are a deterministic surrogate (error falls with capacity; size grows with it), so
// the example runs anywhere; a real "accuracy vs. #features" sklearn objective is the
// OptunaSharp.Tests.PackageParity track.
//
//   Run:  dotnet run examples/MultiObjectivePareto.cs

using System.Collections.Generic;
using System.Linq;
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

var study = Optuna.CreateStudy(
    storage: new InMemoryStorage(),
    sampler: new NSGAIISampler(populationSize: 20, seed: 0),
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },   // (error, size)
    studyName: "multi-objective");

study.Optimize(Objective, nTrials: 60);

var front = study.BestTrials.OrderBy(t => t.Values![0]).ToList();
Console.WriteLine($"Pareto front: {front.Count} non-dominated trade-offs (of {study.Trials.Count} trials)");
Console.WriteLine($"  {"error",-10}{"size",-10}params");
foreach (var t in front)
    Console.WriteLine($"  {t.Values![0],-10:F4}{t.Values![1],-10:F0}" +
                      $"n_layers={t.Params["n_layers"]}, n_units={t.Params["n_units"]}, dropout={t.Params["dropout"]:F2}");

// Objective 1: validation error — falls as capacity (layers × log2 units) grows, best near dropout 0.2.
// Objective 2: model size — grows with layers × units. The two pull against each other → a real front.
static IReadOnlyList<double> Objective(Trial t)
{
    long nLayers = t.SuggestInt("n_layers", 1, 6);
    long nUnits = t.SuggestInt("n_units", 16, 512, log: true);
    double dropout = t.SuggestFloat("dropout", 0.0, 0.6);

    double capacity = nLayers * Math.Log2(nUnits);
    double error = 0.5 * Math.Exp(-0.15 * capacity) + 0.05 * Math.Abs(dropout - 0.2);
    double size = nLayers * nUnits;
    return new[] { error, size };
}
