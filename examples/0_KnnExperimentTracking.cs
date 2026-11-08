#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:project ../src/OptunaSharp.Storage.Sqlite/OptunaSharp.Storage.Sqlite.csproj
#:property PublishAot=false

// Experiment tracking with rich metadata — every trial of a k-NN classifier search records what a reviewer
// asks for later (per-fold scores, a confusion matrix, per-class F1, the run's config and code version) as
// user attributes; the study lives in a journal file, is reopened by a fresh process-like storage, copied
// into SQLite, and — when a Python with optuna is on PATH — read back by Python, checking that the
// metadata survives every hop unchanged.
//
// source: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.set_user_attr
//         (user attributes: any JSON-serializable value)
//         https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.JournalStorage.html
//
// Real objective: a deterministic 3-class dataset (two informative features with a curved boundary, two
// noise features, one feature on a 1000× scale) and a k-nearest-neighbours classifier with 5-fold CV; the
// knobs are k, the vote weighting, the distance metric and whether features are standardized first (the
// 1000×-scale feature makes that decisive). The score is mean CV accuracy.
//
//   Run:  dotnet run examples/0_KnnExperimentTracking.cs

using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var (x, y) = Dataset.Make(rows: 600, seed: 13);
string journal = Path.Combine(Path.GetTempPath(), "optunasharp-knn-tracking.journal");
File.Delete(journal);

var study = Optuna.CreateStudy(
    storage: new JournalStorage(new JournalFileBackend(journal)),
    sampler: new TPESampler(seed: 3, nStartupTrials: 10),
    direction: StudyDirection.Maximize,
    studyName: "knn-tracking");
study.SetUserAttr("dataset", new Dictionary<string, object?> { ["name"] = "three-moons-v2", ["rows"] = 600, ["features"] = 5 });
study.Optimize(t => Objective(t, x, y), nTrials: 40);
var best = study.BestTrial;
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "best CV accuracy {0:P1} (trial #{1}): {2}", best.Value, best.Number,
    string.Join(", ", best.Params.Select(kv => $"{kv.Key}={kv.Value}"))));
Console.WriteLine("best trial's metadata as stored:");
foreach (var kv in best.UserAttrs)
    Console.WriteLine($"    {kv.Key,-16} {JsonSerializer.Serialize(kv.Value)}");

// hop 1: a fresh storage object over the same journal file (what a second process sees).
var reopened = Optuna.LoadStudy("knn-tracking", new JournalStorage(new JournalFileBackend(journal)));
// hop 2: copy into SQLite (Optuna's own schema).
string db = Path.Combine(Path.GetTempPath(), "optunasharp-knn-tracking.db");
File.Delete(db);
using var sqlite = new RDBStorage("sqlite:///" + db.Replace('\\', '/'));
Optuna.CopyStudy("knn-tracking", new JournalStorage(new JournalFileBackend(journal)), sqlite);
var fromSqlite = Optuna.LoadStudy("knn-tracking", sqlite);

string Canon(Study s) => JsonSerializer.Serialize(s.Trials.Select(t => t.UserAttrs.OrderBy(kv => kv.Key, StringComparer.Ordinal)
    .ToDictionary(kv => kv.Key, kv => kv.Value)).Prepend(s.UserAttrs.ToDictionary(kv => kv.Key, kv => kv.Value)));
Console.WriteLine();
Console.WriteLine($"metadata identical after reopening the journal: {Canon(study) == Canon(reopened)}");
Console.WriteLine($"metadata identical after copying into SQLite  : {Canon(study) == Canon(fromSqlite)}");
PythonCheck(journal, db, best.Number);

/// <summary>Cross-validates one k-NN configuration and records the evidence on the trial.</summary>
/// <param name="t">The trial proposing k, the weighting, the metric and the scaling.</param>
/// <param name="x">Feature rows.</param>
/// <param name="y">Class labels 0–2.</param>
/// <returns>Mean 5-fold accuracy.</returns>
static double Objective(Trial t, double[][] x, int[] y)
{
    int k = (int)t.SuggestInt("k", 1, 50, log: true);
    string weights = t.SuggestCategorical("weights", new[] { "uniform", "distance" });
    string metric = t.SuggestCategorical("metric", new[] { "euclidean", "manhattan", "cosine" });
    bool scale = t.SuggestCategorical("standardize", new[] { false, true });

    var foldAcc = new List<double>();
    var confusion = new int[3][] { new int[3], new int[3], new int[3] };
    for (int f = 0; f < 5; f++)
    {
        var testIdx = Enumerable.Range(0, x.Length).Where(i => i % 5 == f).ToArray();
        var trainIdx = Enumerable.Range(0, x.Length).Where(i => i % 5 != f).ToArray();
        var (trX, teX) = scale ? Knn.Standardize(trainIdx.Select(i => x[i]).ToArray(), testIdx.Select(i => x[i]).ToArray())
                               : (trainIdx.Select(i => x[i]).ToArray(), testIdx.Select(i => x[i]).ToArray());
        var trY = trainIdx.Select(i => y[i]).ToArray();
        int correct = 0;
        for (int j = 0; j < teX.Length; j++)
        {
            int pred = Knn.Predict(trX, trY, teX[j], k, weights, metric);
            int truth = y[testIdx[j]];
            confusion[truth][pred]++;
            correct += pred == truth ? 1 : 0;
        }
        foldAcc.Add((double)correct / teX.Length);
    }
    var f1 = new Dictionary<string, object?>();
    for (int c = 0; c < 3; c++)
    {
        double tp = confusion[c][c], fp = Enumerable.Range(0, 3).Sum(r => confusion[r][c]) - tp, fn = confusion[c].Sum() - tp;
        f1[$"class_{c}"] = tp == 0 ? 0.0 : 2 * tp / (2 * tp + fp + fn);
    }
    t.SetUserAttr("fold_accuracy", foldAcc);                        // a list of floats
    t.SetUserAttr("confusion", confusion.Select(r => r.Select(v => (long)v).ToList()).ToList());   // nested lists of ints
    t.SetUserAttr("f1", f1);                                        // a dict
    t.SetUserAttr("code_version", "a1b2c3d");                       // a string
    t.SetUserAttr("standardized", scale);                           // a bool
    return foldAcc.Average();
}

/// <summary>Reads both stores from Python Optuna, if available, and prints what it sees for the best trial.</summary>
/// <param name="journal">The journal file.</param>
/// <param name="db">The SQLite file.</param>
/// <param name="bestNumber">The best trial's number.</param>
static void PythonCheck(string journal, string db, int bestNumber)
{
    string code = "import optuna,sys,json; optuna.logging.set_verbosity(optuna.logging.ERROR)\n"
                  + "from optuna.storages.journal import JournalFileBackend\n"
                  + "j=optuna.load_study(study_name='knn-tracking', storage=optuna.storages.JournalStorage(JournalFileBackend(sys.argv[1])))\n"
                  + "s=optuna.load_study(study_name='knn-tracking', storage='sqlite:///'+sys.argv[2])\n"
                  + "b=int(sys.argv[3])\n"
                  + "same = [t.user_attrs for t in j.trials] == [t.user_attrs for t in s.trials] and j.user_attrs == s.user_attrs\n"
                  + "print('python reads journal and sqlite alike:', same)\n"
                  + "print('python sees best trial attrs:', json.dumps(j.trials[b].user_attrs, sort_keys=True))\n";
    try
    {
        var psi = new ProcessStartInfo("python") { RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (var a in new[] { "-c", code, journal, db.Replace('\\', '/'), bestNumber.ToString(CultureInfo.InvariantCulture) })
            psi.ArgumentList.Add(a);
        using var p = Process.Start(psi)!;
        string output = p.StandardOutput.ReadToEnd(), err = p.StandardError.ReadToEnd();
        p.WaitForExit();
        Console.WriteLine(p.ExitCode == 0 ? output.TrimEnd() : "python failed: " + err.Trim().Split('\n').Last());
    }
    catch (System.ComponentModel.Win32Exception)
    {
        Console.WriteLine("(no python on PATH — skipped the cross-language read)");
    }
}

/// <summary>The synthetic three-class dataset.</summary>
static class Dataset
{
    /// <summary>Generates rows and labels; the same seed gives the same data.</summary>
    /// <param name="rows">Row count.</param>
    /// <param name="seed">Data seed.</param>
    /// <returns>Five features per row and a class label per row.</returns>
    public static (double[][] X, int[] Y) Make(int rows, int seed)
    {
        var rng = new Random(seed);   // data generation only — samplers never use System.Random
        var x = new double[rows][];
        var y = new int[rows];
        for (int i = 0; i < rows; i++)
        {
            int c = i % 3;
            double angle = Math.PI * rng.NextDouble();
            double a = Math.Cos(angle) + c * 0.9 + 0.25 * Normal(rng);
            double b = Math.Sin(angle) * (c == 1 ? -1 : 1) + 0.25 * Normal(rng);
            x[i] = new[] { a, b, Normal(rng), Normal(rng), 1000 * Normal(rng) };   // last: a noisy feature on a huge scale
            y[i] = c;
        }
        return (x, y);
    }

    /// <summary>A standard normal draw (Box–Muller).</summary>
    /// <param name="rng">The random stream.</param>
    /// <returns>A N(0, 1) variate.</returns>
    private static double Normal(Random rng) => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
}

/// <summary>Brute-force k-nearest-neighbours.</summary>
static class Knn
{
    /// <summary>Standardizes both sets with the training set's mean and standard deviation.</summary>
    /// <param name="train">Training rows.</param>
    /// <param name="test">Test rows.</param>
    /// <returns>The scaled copies.</returns>
    public static (double[][] Train, double[][] Test) Standardize(double[][] train, double[][] test)
    {
        int d = train[0].Length;
        var mean = Enumerable.Range(0, d).Select(j => train.Average(r => r[j])).ToArray();
        var sd = Enumerable.Range(0, d).Select(j => Math.Sqrt(train.Average(r => Math.Pow(r[j] - mean[j], 2))) + 1e-12).ToArray();
        double[][] Scale(double[][] rows) => rows.Select(r => r.Select((v, j) => (v - mean[j]) / sd[j]).ToArray()).ToArray();
        return (Scale(train), Scale(test));
    }

    /// <summary>Predicts one row's class by a (weighted) vote of its k nearest training rows.</summary>
    /// <param name="trX">Training rows.</param>
    /// <param name="trY">Training labels.</param>
    /// <param name="q">The query row.</param>
    /// <param name="k">Neighbour count.</param>
    /// <param name="weights">"uniform" or "distance" (1/d votes).</param>
    /// <param name="metric">"euclidean", "manhattan" or "cosine".</param>
    /// <returns>The predicted class.</returns>
    public static int Predict(double[][] trX, int[] trY, double[] q, int k, string weights, string metric)
    {
        var nearest = Enumerable.Range(0, trX.Length).Select(i => (D: Distance(trX[i], q, metric), Y: trY[i]))
            .OrderBy(p => p.D).Take(k);
        var votes = new double[3];
        foreach (var (d, label) in nearest)
            votes[label] += weights == "distance" ? 1 / (d + 1e-9) : 1;
        return Array.IndexOf(votes, votes.Max());
    }

    /// <summary>The distance between two rows.</summary>
    /// <param name="a">First row.</param>
    /// <param name="b">Second row.</param>
    /// <param name="metric">The metric name.</param>
    /// <returns>A non-negative distance.</returns>
    private static double Distance(double[] a, double[] b, string metric)
    {
        double s = 0, na = 0, nb = 0;
        for (int j = 0; j < a.Length; j++)
        {
            if (metric == "euclidean") s += (a[j] - b[j]) * (a[j] - b[j]);
            else if (metric == "manhattan") s += Math.Abs(a[j] - b[j]);
            else { s += a[j] * b[j]; na += a[j] * a[j]; nb += b[j] * b[j]; }
        }
        return metric == "euclidean" ? Math.Sqrt(s) : metric == "manhattan" ? s : 1 - s / (Math.Sqrt(na * nb) + 1e-12);
    }
}
