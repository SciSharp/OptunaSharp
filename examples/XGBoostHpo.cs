#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// XGBoost hyperparameter optimization — an OptunaSharp port of a widely-starred public Optuna gist
// (16 stars, 5 forks), by `bexgboost`:
//   https://gist.github.com/bexgboost/823df08d2e3760538e9b931d38439a68
//
// The gist minimizes RMSE by tuning the classic 8-parameter XGBoost grid with early-stopping pruning
// (XGBoostPruningCallback). The xgb.train objective is replaced here by a deterministic surrogate so
// the example runs without XGBoost or a dataset — only the flat search space and direction='minimize'
// are the portable, battle-tested part.
//
// The exact search space below is byte-for-byte parity-checked against live Optuna in
// test/OptunaSharp.Tests.Interop/GistExampleParityTests.cs.
//
//   Run:  dotnet run examples/XGBoostHpo.cs

using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

var study = Optuna.CreateStudy(
    storage: new InMemoryStorage(),
    sampler: new RandomSampler(seed: 0),   // seeded for reproducibility; swap for `new TPESampler()` for the gist's default
    direction: StudyDirection.Minimize,
    studyName: "xgboost-hpo");

study.Optimize(Objective, nTrials: 25);

Console.WriteLine($"Finished trials: {study.Trials.Count}");
Console.WriteLine($"Best surrogate RMSE: {study.BestValue:0.000000}");
Console.WriteLine("Best hyperparameters:");
foreach (var kv in study.BestParams)
    Console.WriteLine($"    {kv.Key}: {kv.Value}");

// the gist's flat 8-parameter XGBoost grid; xgb.train is replaced by a deterministic surrogate RMSE.
// the exact space is byte-for-byte parity-checked in GistExampleParityTests.cs.
static double Objective(Trial t)
{
    double eta = t.SuggestFloat("eta", 0.01, 0.3);              // learning rate (gist uses a linear range — matched)
    t.SuggestInt("max_depth", 3, 10);                          // tree depth (suggested + recorded; unused by the surrogate)
    double subsample = t.SuggestFloat("subsample", 0.5, 1.0);  // row subsampling per tree
    t.SuggestFloat("colsample_bytree", 0.5, 1.0);              // column subsampling per tree
    t.SuggestFloat("gamma", 0.0, 10.0);                        // min loss reduction to make a split
    t.SuggestFloat("min_child_weight", 0.1, 10.0);             // min summed child weight
    t.SuggestFloat("lambda", 0.1, 10.0);                       // L2 regularization
    t.SuggestFloat("alpha", 0.0, 10.0);                        // L1 regularization
    // closed-form stand-in for RMSE: a bowl with its minimum at eta=0.1, subsample=0.8. Math.Pow (not
    // eta*eta) matches Python's `x ** 2` bit-for-bit — the rule that keeps the parity test exact.
    return Math.Pow(eta - 0.1, 2.0) + Math.Pow(subsample - 0.8, 2.0);
}
