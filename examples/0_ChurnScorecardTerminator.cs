#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// When to stop tuning — Optuna's Terminator in its documented default form: every trial reports its
// cross-validation fold scores (`report_cross_validation_scores`), and TerminatorCallback ends the study once
// the GP-estimated room for improvement (the regret bound) falls below the noise of the CV estimate itself —
// past that point, a "better" trial is indistinguishable from a lucky fold split.
//
// source: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.terminator.TerminatorCallback.html
//         (TerminatorCallback with the default RegretBoundEvaluator + CrossValidationErrorEvaluator, and
//         `report_cross_validation_scores`)
//
// Real objective, real training: a deterministic telecom-churn table (2,000 customers, 10 features with
// interactions) and an L2-regularized logistic regression with optional quadratic features, fitted by
// gradient descent; 5-fold CV accuracy is maximized. The trial budget is 120, but the terminator usually
// stops much earlier — the printout shows where, and how little the best score moved after that point in a
// separate run that ignores the terminator. (Optuna has deprecated `optuna.terminator` — v4.9, removal in
// v6 — so a FutureWarning prints; it is still the canonical recipe in Optuna 5.)
//
//   Run:  dotnet run examples/0_ChurnScorecardTerminator.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.Terminator;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var (x, y) = Customers.Generate(rows: 2000, seed: 43);

// the run with the terminator.
var stopped = Optuna.CreateStudy(sampler: new TPESampler(seed: 44), direction: StudyDirection.Maximize, studyName: "churn-terminated");
stopped.Optimize(t => Objective(t, x, y), nTrials: 120, callbacks: new Action<Study, FrozenTrial>[] { new TerminatorCallback() });

// the same search without it, to see what the remaining budget would have bought.
var full = Optuna.CreateStudy(sampler: new TPESampler(seed: 44), direction: StudyDirection.Maximize, studyName: "churn-full");
full.Optimize(t => Objective(t, x, y), nTrials: 120);

int stopAt = stopped.Trials.Count;
double bestAtStop = full.Trials.Take(stopAt).Max(t => t.Value!.Value);
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "terminator stopped after {0} of 120 trials; best CV accuracy {1:P2}", stopAt, stopped.BestValue));
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "the full 120-trial run: best {0:P2} — {1:+0.00;-0.00} percentage points for {2} more trials",
    full.BestValue, 100 * (full.BestValue - bestAtStop), 120 - stopAt));
var folds = (IEnumerable<object?>)stopped.BestTrial.SystemAttrs["terminator:cv_scores"]!;
Console.WriteLine("the best trial's fold scores (the noise the terminator compares against): "
                  + string.Join(", ", folds.Select(v => Convert.ToDouble(v, CultureInfo.InvariantCulture).ToString("P1", CultureInfo.InvariantCulture))));

/// <summary>Cross-validates one scorecard configuration and reports its fold scores to the terminator.</summary>
/// <param name="t">The trial proposing C, the feature expansion and the iteration budget.</param>
/// <param name="x">Customer features.</param>
/// <param name="y">Churn labels.</param>
/// <returns>Mean 5-fold accuracy.</returns>
static double Objective(Trial t, double[][] x, double[] y)
{
    double c = t.SuggestFloat("C", 1e-3, 1e3, log: true);
    bool quadratic = t.SuggestCategorical("quadratic", new[] { false, true });
    long iterations = t.SuggestInt("iterations", 20, 300, log: true);
    var rows = quadratic ? x.Select(r => r.Concat(r.Select(v => v * v)).Append(r[0] * r[2]).Append(r[1] * r[3]).ToArray()).ToArray() : x;
    var scores = new List<double>();
    for (int f = 0; f < 5; f++)
        scores.Add(FoldAccuracy(rows, y, f, c, (int)iterations));
    TerminatorFunctions.ReportCrossValidationScores(t, scores);   // what CrossValidationErrorEvaluator reads
    return scores.Average();
}

/// <summary>Trains on four folds by gradient descent and scores the fifth.</summary>
/// <param name="x">Rows.</param>
/// <param name="y">Labels.</param>
/// <param name="fold">The held-out fold (rows with index % 5 == fold).</param>
/// <param name="c">Inverse L2 strength.</param>
/// <param name="iterations">Gradient-descent steps.</param>
/// <returns>Held-out accuracy.</returns>
static double FoldAccuracy(double[][] x, double[] y, int fold, double c, int iterations)
{
    int d = x[0].Length, nTrain = x.Length - x.Length / 5;
    var w = new double[d];
    double b = 0;
    for (int it = 0; it < iterations; it++)
    {
        var g = new double[d];
        double gb = 0;
        for (int i = 0; i < x.Length; i++)
        {
            if (i % 5 == fold) continue;
            double err = 1 / (1 + Math.Exp(-(Dot(w, x[i]) + b))) - y[i];
            for (int j = 0; j < d; j++) g[j] += err * x[i][j];
            gb += err;
        }
        for (int j = 0; j < d; j++) w[j] -= 0.5 * (g[j] / nTrain + w[j] / (c * nTrain));
        b -= 0.5 * gb / nTrain;
    }
    int correct = 0, total = 0;
    for (int i = fold; i < x.Length; i += 5)
    {
        correct += ((Dot(w, x[i]) + b) > 0 ? 1.0 : 0.0) == y[i] ? 1 : 0;
        total++;
    }
    return (double)correct / total;

    static double Dot(double[] w, double[] r)
    {
        double s = 0;
        for (int j = 0; j < w.Length; j++) s += w[j] * r[j];
        return s;
    }
}

/// <summary>The synthetic telecom customer table.</summary>
static class Customers
{
    /// <summary>Generates standardized features and churn labels; the same seed gives the same table.</summary>
    /// <param name="rows">Customer count.</param>
    /// <param name="seed">Data seed.</param>
    /// <returns>Ten features per customer and a 0/1 churn label.</returns>
    public static (double[][] X, double[] Y) Generate(int rows, int seed)
    {
        var rng = new Random(seed);   // data generation only — samplers never use System.Random
        var x = new double[rows][];
        var y = new double[rows];
        for (int i = 0; i < rows; i++)
        {
            var r = Enumerable.Range(0, 10).Select(_ => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble())).ToArray();
            double z = -0.8 + 1.1 * r[0] - 0.9 * r[1] + 0.6 * r[0] * r[2] + 0.5 * r[3] * r[3] - 0.4 * r[4];
            y[i] = rng.NextDouble() < 1 / (1 + Math.Exp(-z)) ? 1 : 0;
            x[i] = r;
        }
        return (x, y);
    }
}
