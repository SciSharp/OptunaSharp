#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false
#:property SkipNDScopeWeave=true
#:property GeneratePackageOnBuild=false

// Median-pruned training — the canonical Optuna pruning loop: train iteratively, report the metric
// after each epoch, and let a MedianPruner stop trials that are tracking worse than the median so far.
// This is the pattern knottwill/sesame-finetune (113★) uses (trial.report(val_loss, step) +
// trial.should_prune()) and the shape of Optuna's own `pytorch_simple` example.
//
// Distinct from examples/FFmpegX264Tuning.cs, which prunes with SuccessiveHalving at fixed cut-points;
// here the pruner compares each trial's per-epoch loss against the running median. The training loss
// is a deterministic surrogate curve (converges to a param-dependent floor at a param-dependent rate)
// so the example runs anywhere — a real per-round LightGBM/XGBoost curve is the OptunaSharp.Tests.
// PackageParity track.
//
//   Run:  dotnet run examples/MedianPrunedTraining.cs

using OptunaSharp;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

var study = Optuna.CreateStudy(
    storage: new InMemoryStorage(),
    sampler: new TPESampler(seed: 0),
    pruner: new MedianPruner(nStartupTrials: 5, nWarmupSteps: 2),
    direction: StudyDirection.Minimize,
    studyName: "median-pruned-training");

study.Optimize(Objective, nTrials: 40);

int completed = 0, pruned = 0;
foreach (var t in study.Trials)
{
    if (t.State == TrialState.Complete) completed++;
    else if (t.State == TrialState.Pruned) pruned++;
}
Console.WriteLine($"Trials: {study.Trials.Count}  |  completed: {completed}  |  pruned early: {pruned}");
Console.WriteLine($"Best validation loss: {study.BestValue:F4}");
Console.WriteLine("Best hyperparameters:");
foreach (var kv in study.BestParams)
    Console.WriteLine($"    {kv.Key}: {kv.Value}");

// A deterministic stand-in for a training run: 20 "epochs" of validation loss decaying to a floor.
// Good hyperparameters (learning_rate ≈ 1e-2, more units, light dropout) reach a lower floor faster,
// so laggards fall below the running median and get pruned.
static double Objective(Trial t)
{
    double lr = t.SuggestFloat("learning_rate", 1e-4, 1e-1, log: true);
    long units = t.SuggestInt("n_units", 16, 256, log: true);
    double dropout = t.SuggestFloat("dropout", 0.0, 0.6);

    double offLr = Math.Log10(lr) + 2.0;                       // 0 at the sweet spot lr = 1e-2
    double floor = 0.08 + 0.12 * offLr * offLr
                        + 0.40 * Math.Abs(dropout - 0.15)
                        + 0.30 / Math.Log2(units);
    double rate = 0.5 / (1.0 + Math.Abs(offLr));               // best lr decays fastest

    double loss = 0.0;
    for (int epoch = 0; epoch < 20; epoch++)
    {
        loss = floor + (1.2 - floor) * Math.Exp(-rate * (epoch + 1));
        t.Report(loss, epoch);
        if (t.ShouldPrune())
            throw new TrialPruned();
    }
    return loss;
}
