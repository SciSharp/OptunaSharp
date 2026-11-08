#:project ../src/OptunaSharp.Integration.LightGbm/OptunaSharp.Integration.LightGbm.csproj
#:property PublishAot=false

// Loan-default model with LightGBMTunerCV — Optuna's stepwise LightGBM tuner (feature_fraction →
// num_leaves → bagging → feature_fraction again → L1/L2 → min_child_samples) cross-validates a real
// gradient-boosted model of loan default, trained in-process by ML.NET's native LightGBM.
//
// source: https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_tuner_cv.py
//         (the Optuna example this follows: LightGBMTunerCV on a binary task, binary_logloss, 5 folds)
//         https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.lightgbm.LightGBMTunerCV.html
//
// Real objective, real training: a deterministic loan book of 4,000 applications (income, debt-to-income,
// credit history length, utilization, recent inquiries, loan amount/term/purpose, employment length) with a
// non-linear default mechanism and ~7 % defaults. Each step of the tuner is its own short Optuna study;
// every trial trains a LightGBM booster with 5-fold stratified CV and is scored by binary log-loss. The
// tuner's study is returned, so the usual Optuna tooling (best trial, trial count, user attrs) works on it.
// The LightGBM `seed` is set explicitly — without it the ML.NET adapter is not reproducible (an open bug).
//
//   Run:  dotnet run examples/0_LoanDefaultLightGbmTunerCv.cs

using System.Diagnostics;
using System.Globalization;
using OptunaSharp;
using OptunaSharp.Integration.LightGbm;
using OptunaSharp.Logging;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // the tuner logs every trial at INFO

var (x, y) = LoanBook.Generate(rows: 4000, seed: 19);
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "loan book: {0} applications, default rate {1:P1}", y.Length, y.Average()));

var lgbmParams = new Dictionary<string, object?>
{
    ["objective"] = "binary",
    ["metric"] = "binary_logloss",
    ["verbosity"] = -1,
    ["boosting_type"] = "gbdt",
    // Python's LightGBM is reproducible without a seed (fixed default bagging/feature-fraction seeds); the
    // ML.NET adapter is not yet — it draws a random seed per fit when none is given (open bug, see
    // test/OptunaSharp.Tests.Integration.LightGbm/LightGbmRealWorldBugTests.cs). Pin it explicitly.
    ["seed"] = 0L,
};
var sw = Stopwatch.StartNew();
var tuner = new LightGBMTunerCV(lgbmParams, new Dataset(x, y), numBoostRound: 200, nFold: 5, stratified: true, seed: 0, optunaSeed: 42);
tuner.Run();
sw.Stop();

Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "stepwise tuning: {0} trials in {1:F1} s", tuner.Study.Trials.Count, sw.Elapsed.TotalSeconds));
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "best 5-fold binary_logloss: {0:F5}", tuner.BestScore));
Console.WriteLine("best parameters:");
foreach (var kv in tuner.BestParams)
    Console.WriteLine($"    {kv.Key,-20} {kv.Value}");
// the tuner tags each trial with its step in a system attribute, as optuna-integration does.
var steps = tuner.Study.Trials.GroupBy(t => t.SystemAttrs.TryGetValue("lightgbm_tuner:step_name", out var s) ? (string)s! : "?")
    .Select(g => $"{g.Key} ({g.Count()})");
Console.WriteLine("steps: " + string.Join(" → ", steps));

/// <summary>The synthetic loan book.</summary>
static class LoanBook
{
    /// <summary>Generates applications and default outcomes; the same seed gives the same book.</summary>
    /// <param name="rows">Number of applications.</param>
    /// <param name="seed">Data seed.</param>
    /// <returns>Ten numeric features per application and a 0/1 default label.</returns>
    public static (double[,] X, double[] Y) Generate(int rows, int seed)
    {
        var rng = new Random(seed);   // data generation only — samplers never use System.Random
        var x = new double[rows, 10];
        var y = new double[rows];
        for (int i = 0; i < rows; i++)
        {
            double income = Math.Exp(Math.Log(60_000) + 0.5 * Normal(rng));
            double dti = Math.Clamp(0.25 + 0.12 * Normal(rng), 0, 0.8);          // debt-to-income
            double history = Math.Max(0, 10 + 6 * Normal(rng));                  // years of credit history
            double util = Math.Clamp(0.3 + 0.25 * Normal(rng), 0, 1.2);          // revolving utilization
            double inquiries = Math.Floor(Math.Abs(1.2 * Normal(rng)));           // last 6 months
            double amount = Math.Exp(Math.Log(15_000) + 0.6 * Normal(rng));
            double term = rng.NextDouble() < 0.7 ? 36 : 60;
            double purpose = rng.Next(5);                                         // 0 debt consolidation … 4 small business
            double employment = Math.Floor(Math.Max(0, 6 + 4 * Normal(rng)));
            double homeowner = rng.NextDouble() < 0.45 ? 1 : 0;
            double z = -2.4 + 3.2 * Math.Max(0, dti - 0.35) + 1.4 * Math.Max(0, util - 0.6) + 0.35 * inquiries
                       - 0.05 * Math.Min(history, 15) + 0.5 * (term == 60 ? 1 : 0) + 0.7 * (purpose == 4 ? 1 : 0)
                       + 0.4 * Math.Log(amount / income * 4) - 0.3 * homeowner - 0.03 * Math.Min(employment, 10);
            y[i] = rng.NextDouble() < 1 / (1 + Math.Exp(-z)) ? 1 : 0;
            double[] row = { income, dti, history, util, inquiries, amount, term, purpose, employment, homeowner };
            for (int j = 0; j < 10; j++)
                x[i, j] = row[j];
        }
        return (x, y);
    }

    /// <summary>A standard normal draw (Box–Muller).</summary>
    /// <param name="rng">The random stream.</param>
    /// <returns>A N(0, 1) variate.</returns>
    private static double Normal(Random rng) => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
}
