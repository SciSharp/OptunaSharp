#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Credit-scoring feature selection — which credit-bureau attributes are worth paying for? Sixteen boolean
// "buy this attribute" switches plus the regularization strength of a logistic-regression scorecard are
// tuned on cross-validated log-loss plus the attributes' data cost; fANOVA importance then says which
// switches mattered, and the trials dataframe gives the analyst the raw table.
//
// source: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.importance.get_param_importances.html
//         (hyperparameter importance; fANOVA is the evaluator used here)
//         https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.trials_dataframe
//
// Real objective: a deterministic applicant table (3,000 rows) where 5 attributes truly drive default risk,
// 3 are noisy near-copies of informative ones (they look useful but add little on top), and 8 are noise.
// Each trial fits an L2-regularized logistic regression by gradient descent on the chosen attributes with
// 5-fold cross-validation, and pays $0.004 of "log-loss equivalent" per attribute bought — so the optimum
// buys the informative attributes and skips the copies. TPE in multivariate mode models the switches
// jointly, which matters when two features substitute for each other.
//
//   Run:  dotnet run examples/0_CreditFeatureSelection.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Importance;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var (x, y, names, informative) = Applicants.Generate(rows: 3000, seed: 8);
var study = Optuna.CreateStudy(
    sampler: new TPESampler(seed: 6, nStartupTrials: 30, multivariate: true),
    direction: StudyDirection.Minimize,
    studyName: "credit-attributes");
study.Optimize(t => Objective(t, x, y, names), nTrials: 150);

var best = study.BestTrial;
var bought = names.Where(n => (bool)best.Params["use_" + n]!).ToList();
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "best objective {0:F4} (cv log-loss {1:F4} + data cost), trial #{2}",
    best.Value, best.UserAttrs["cv_logloss"], best.Number));
Console.WriteLine($"attributes bought ({bought.Count}): {string.Join(", ", bought)}");
Console.WriteLine($"truly informative     : {string.Join(", ", informative)}");
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "regularization C = {0:G4}", best.Params["C"]));

// which decisions mattered? fANOVA decomposes the objective's variance over the trials seen.
var importances = ImportanceFunctions.GetParamImportances(study, new FanovaImportanceEvaluator(seed: 0));
Console.WriteLine();
Console.WriteLine("fANOVA importance (top 8):");
foreach (var kv in importances.Take(8))
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "  {0,-22} {1,6:F3} {2}", kv.Key, kv.Value, new string('#', (int)Math.Round(kv.Value * 60))));

// the analyst's raw table: one row per trial, flattened `params_*` / `user_attrs_*` columns.
var table = study.TrialsDataframe(attrs: new[] { "number", "value", "params", "user_attrs", "state" });
Console.WriteLine();
Console.WriteLine($"trials dataframe: {table.RowCount} rows × {table.ColumnCount} columns; first columns: {string.Join(", ", table.Columns.Take(6))}");

/// <summary>Cross-validates a scorecard on the attributes the trial buys.</summary>
/// <param name="t">The trial proposing the 16 switches and C.</param>
/// <param name="x">Standardized applicant attributes.</param>
/// <param name="y">Default labels (1 = defaulted).</param>
/// <param name="names">Attribute names, aligned with the columns of <paramref name="x" />.</param>
/// <returns>Mean 5-fold log-loss plus 0.004 per attribute bought.</returns>
static double Objective(Trial t, double[][] x, double[] y, string[] names)
{
    var cols = new List<int>();
    for (int j = 0; j < names.Length; j++)
        if (t.SuggestCategorical("use_" + names[j], new[] { false, true }))   // buy this attribute?
            cols.Add(j);
    double c = t.SuggestFloat("C", 1e-3, 1e2, log: true);                     // inverse L2 strength
    double cv = Scorecard.CrossValidatedLogLoss(x, y, cols, c, folds: 5);
    t.SetUserAttr("cv_logloss", cv);
    t.SetUserAttr("n_bought", cols.Count);
    return cv + 0.004 * cols.Count;
}

/// <summary>The synthetic applicant table.</summary>
static class Applicants
{
    /// <summary>Generates standardized attributes and default labels; the same seed gives the same table.</summary>
    /// <param name="rows">Number of applicants.</param>
    /// <param name="seed">Data seed.</param>
    /// <returns>Attributes, labels, attribute names and the names of the truly informative ones.</returns>
    public static (double[][] X, double[] Y, string[] Names, string[] Informative) Generate(int rows, int seed)
    {
        string[] names =
        {
            "utilization", "delinquencies", "inquiries_6m", "account_age", "income_to_debt",     // informative
            "utilization_alt", "delinq_bureau2", "age_alt",                                    // noisy copies
            "zip_digit", "phone_brand", "email_len", "signup_hour", "browser", "name_len", "dob_day", "screen_dpi",
        };
        var rng = new Random(seed);   // data generation only — samplers never use System.Random
        var x = new double[rows][];
        var y = new double[rows];
        for (int i = 0; i < rows; i++)
        {
            var v = new double[names.Length];
            for (int j = 0; j < 5; j++)
                v[j] = Normal(rng);
            v[5] = v[0] + 0.7 * Normal(rng);                       // a second bureau's (noisier) utilization
            v[6] = v[1] + 0.8 * Normal(rng);
            v[7] = v[3] + 0.9 * Normal(rng);
            for (int j = 8; j < names.Length; j++)
                v[j] = Normal(rng);
            double z = -1.6 + 0.9 * v[0] + 0.8 * v[1] + 0.5 * v[2] - 0.6 * v[3] - 0.7 * v[4];
            y[i] = rng.NextDouble() < 1 / (1 + Math.Exp(-z)) ? 1 : 0;
            x[i] = v;
        }
        return (x, y, names, names[..5]);
    }

    /// <summary>A standard normal draw (Box–Muller).</summary>
    /// <param name="rng">The random stream.</param>
    /// <returns>A N(0, 1) variate.</returns>
    private static double Normal(Random rng) => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
}

/// <summary>An L2-regularized logistic-regression scorecard fitted by full-batch gradient descent.</summary>
static class Scorecard
{
    /// <summary>Mean held-out log-loss over contiguous folds.</summary>
    /// <param name="x">Attributes.</param>
    /// <param name="y">Labels.</param>
    /// <param name="cols">The attribute columns to use (may be empty: intercept-only model).</param>
    /// <param name="c">Inverse regularization strength (the penalty is ‖w‖² / (2·C·n)).</param>
    /// <param name="folds">Number of folds.</param>
    /// <returns>The mean validation log-loss.</returns>
    public static double CrossValidatedLogLoss(double[][] x, double[] y, List<int> cols, double c, int folds)
    {
        int n = x.Length, size = n / folds;
        double total = 0;
        for (int f = 0; f < folds; f++)
        {
            int lo = f * size, hi = f == folds - 1 ? n : lo + size;
            var w = new double[cols.Count];
            double b = 0;
            int nTrain = n - (hi - lo);
            for (int it = 0; it < 150; it++)
            {
                var g = new double[cols.Count];
                double gb = 0;
                for (int i = 0; i < n; i++)
                {
                    if (i >= lo && i < hi)
                        continue;
                    double err = Sigmoid(Dot(x[i], cols, w) + b) - y[i];
                    for (int k = 0; k < cols.Count; k++)
                        g[k] += err * x[i][cols[k]];
                    gb += err;
                }
                for (int k = 0; k < cols.Count; k++)
                    w[k] -= 0.5 * (g[k] / nTrain + w[k] / (c * nTrain));
                b -= 0.5 * gb / nTrain;
            }
            double loss = 0;
            for (int i = lo; i < hi; i++)
            {
                double p = Math.Clamp(Sigmoid(Dot(x[i], cols, w) + b), 1e-12, 1 - 1e-12);
                loss -= y[i] * Math.Log(p) + (1 - y[i]) * Math.Log(1 - p);
            }
            total += loss / (hi - lo);
        }
        return total / folds;
    }

    /// <summary>Dot product of a row's selected columns with the weights.</summary>
    /// <param name="row">The row.</param>
    /// <param name="cols">Selected column indexes.</param>
    /// <param name="w">Weights aligned with <paramref name="cols" />.</param>
    /// <returns>The linear score.</returns>
    private static double Dot(double[] row, List<int> cols, double[] w)
    {
        double s = 0;
        for (int k = 0; k < cols.Count; k++)
            s += row[cols[k]] * w[k];
        return s;
    }

    /// <summary>The logistic function, stable for large negative inputs.</summary>
    /// <param name="z">The logit.</param>
    /// <returns>A probability in (0, 1).</returns>
    private static double Sigmoid(double z) => z >= 0 ? 1 / (1 + Math.Exp(-z)) : Math.Exp(z) / (1 + Math.Exp(z));
}
