#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false
#:property SkipNDScopeWeave=true
#:property GeneratePackageOnBuild=false

// Hyperparameter importance — after a study, ask which parameters actually mattered. optuna's
// get_param_importances (fANOVA / PedAnova / MDI) turns a finished study into a ranked, normalized
// importance per parameter; OptunaSharp exposes the same via ImportanceFunctions.GetParamImportances.
//
// source: https://optuna.readthedocs.io/en/stable/reference/importance.html
//
// the objective is rigged so the ranking is known — 'a' dominates (quadratic), 'b' has a faint linear
// pull, 'c' does nothing — so you can watch the evaluators recover a ≫ b ≫ c. pure managed; a real
// xgboost/sklearn objective is the PackageParity track.
//
//   Run:  dotnet run examples/HyperparameterImportance.cs

using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Importance;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

// RandomSampler (uniform coverage) so the evaluators see the whole space — the cleanest way to read
// importance. TPE concentrates around good points, which blurs the picture.
var study = Optuna.CreateStudy(
    storage: new InMemoryStorage(),
    sampler: new RandomSampler(seed: 0),
    direction: StudyDirection.Minimize,
    studyName: "importance");

study.Optimize(Objective, nTrials: 128);   // more trials → steadier importance estimates

Console.WriteLine($"Best value: {study.BestValue:F4}   (expect the ranking a ≫ b ≫ c)\n");
// fANOVA decomposes the objective's variance across params; PedAnova (Optuna's default) is a faster,
// distribution-based estimate. showing both makes the ranking, and the two methods, easy to compare.
PrintImportances("fANOVA (variance decomposition)", ImportanceFunctions.GetParamImportances(study, new FanovaImportanceEvaluator()));
PrintImportances("PedAnova (Optuna's default)", ImportanceFunctions.GetParamImportances(study));

// print each param's normalized importance (they sum to 1) with a little bar
static void PrintImportances(string label, OrderedDict<string, double> importances)
{
    Console.WriteLine(label + ":");
    foreach (var (name, importance) in importances)
        Console.WriteLine($"    {name,-3} {importance,7:F4}  {new string('#', (int)Math.Round(importance * 40))}");
    Console.WriteLine();
}

// 'a' dominates (quadratic), 'b' has a small linear pull, 'c' is pure noise.
static double Objective(Trial t)
{
    double a = t.SuggestFloat("a", -10, 10);
    double b = t.SuggestFloat("b", -10, 10);
    t.SuggestFloat("c", -10, 10);   // suggested but unused → importance should come out ~0
    return a * a + 0.1 * b;
}
