#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false
#:property SkipNDScopeWeave=true
#:property GeneratePackageOnBuild=false

// Median-pruned training — the canonical Optuna pruning loop: train iteratively, report the metric
// after each epoch, and let a MedianPruner stop trials tracking worse than the median so far.
//
// source (real-world pattern): https://github.com/knottwill/sesame-finetune  (trial.report(val_loss,
//   step) + trial.should_prune())
// source (canonical example):  https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
//
// distinct from examples/FFmpegX264Tuning.cs, which prunes with SuccessiveHalving at fixed cut-points;
// here the pruner compares each trial's per-epoch loss against the running median. the training loss is
// a deterministic surrogate curve (converges to a param-dependent floor at a param-dependent rate) so
// the example runs anywhere — a real per-round LightGBM/XGBoost curve is the PackageParity track.
//
//   Run:  dotnet run examples/MedianPrunedTraining.cs

using OptunaSharp;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

var study = Optuna.CreateStudy(
    storage: new InMemoryStorage(),                                 // in-memory — nothing persisted, fine for a demo
    sampler: new TPESampler(seed: 0),                               // seeded, so the run reproduces exactly
    pruner: new MedianPruner(nStartupTrials: 5, nWarmupSteps: 2),   // don't prune during the first 5 trials, nor before epoch 2
    direction: StudyDirection.Minimize,                             // lower validation loss is better
    studyName: "median-pruned-training");

// run 40 trials; laggards are cut short mid-"training" by the pruner instead of running all 20 epochs
study.Optimize(Objective, nTrials: 40);

// tally how many trials finished vs. were pruned early
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

// a deterministic stand-in for a training run: 20 "epochs" of validation loss decaying toward a floor.
// good hyperparameters (learning_rate ≈ 1e-2, more units, light dropout) reach a lower floor faster, so
// laggards fall below the running median and get pruned.
static double Objective(Trial t)
{
    // the search space — a typical small-net configuration
    double lr = t.SuggestFloat("learning_rate", 1e-4, 1e-1, log: true);   // log scale: search across orders of magnitude
    long units = t.SuggestInt("n_units", 16, 256, log: true);            // log scale: 16..256 hidden units
    double dropout = t.SuggestFloat("dropout", 0.0, 0.6);

    // shape the surrogate curve from the params: how low it converges (floor) and how fast (rate)
    double offLr = Math.Log10(lr) + 2.0;                                 // 0 at the sweet spot lr = 1e-2
    double floor = 0.08 + 0.12 * offLr * offLr                           // penalize being far from the best lr
                        + 0.40 * Math.Abs(dropout - 0.15)                // best around dropout 0.15
                        + 0.30 / Math.Log2(units);                       // more units → lower floor
    double rate = 0.5 / (1.0 + Math.Abs(offLr));                         // best lr also decays fastest

    // "train" for up to 20 epochs, reporting loss each epoch so the pruner can act
    double loss = 0.0;
    for (int epoch = 0; epoch < 20; epoch++)
    {
        loss = floor + (1.2 - floor) * Math.Exp(-rate * (epoch + 1));    // monotonically decreasing toward floor
        t.Report(loss, epoch);                                           // hand the intermediate value to the study
        if (t.ShouldPrune())                                             // worse than the median at this epoch?
            throw new TrialPruned();                                     // → stop early; Optimize records it as PRUNED
    }
    return loss;
}
