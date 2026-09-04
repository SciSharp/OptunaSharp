#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false
#:property SkipNDScopeWeave=true
#:property GeneratePackageOnBuild=false

// Automatic termination — instead of guessing n_trials, let Optuna's Terminator decide when to stop.
// A TerminatorCallback checks after each trial and calls study.Stop() once the study should end. Here
// the criterion is best-value stagnation: stop when the best value hasn't improved for a run of trials
// (after at least minNTrials). So optimize() ends early — short of the 300-trial cap.
//
// The default Terminator is fancier: it fits a GP to bound remaining "regret" and compares it to the
// evaluation error you report per trial via TerminatorFunctions.ReportCrossValidationScores(trial,
// foldScores) — swap that in when you have real CV fold metrics. NOTE: `optuna.terminator` is
// deprecated upstream in v4.9 (removed in v6); OptunaSharp mirrors it, hence the one FutureWarning.
//
//   Run:  dotnet run examples/AutomaticTermination.cs

using System;
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

var study = Optuna.CreateStudy(
    storage: new InMemoryStorage(),
    sampler: new TPESampler(seed: 0),
    direction: StudyDirection.Minimize,
    studyName: "auto-terminate");

var terminate = (Action<Study, FrozenTrial>)new OptunaSharp.Terminator.TerminatorCallback(
    new OptunaSharp.Terminator.Terminator(
        improvementEvaluator: new OptunaSharp.Terminator.BestValueStagnationEvaluator(maxStagnationTrials: 25),
        minNTrials: 20));

study.Optimize(Objective, nTrials: 300, callbacks: new[] { terminate });

Console.WriteLine($"\nTerminator stopped after {study.Trials.Count} trials (the n_trials cap was 300).");
Console.WriteLine($"Best value: {study.BestValue:F6}   (min of (x-2)² + (y+3)² is 0 at x=2, y=-3)");
foreach (var kv in study.BestParams)
    Console.WriteLine($"    {kv.Key}: {kv.Value}");

static double Objective(Trial t)
{
    double x = t.SuggestFloat("x", -10, 10);
    double y = t.SuggestFloat("y", -10, 10);
    return (x - 2) * (x - 2) + (y + 3) * (y + 3);
}
