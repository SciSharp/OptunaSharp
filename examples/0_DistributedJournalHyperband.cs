#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Distributed tuning on a shared journal file — three worker processes, each running two trial threads
// (`n_jobs=2`), optimize one study through a single JournalFileBackend file (Optuna's recipe for clusters
// that only share a file system), and HyperbandPruner coordinates early stopping across all six workers
// because every intermediate value lands in the same journal.
//
// source: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.journal.JournalFileBackend.html
//         (the append-only, lock-protected journal file several processes can share)
//         https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
//         https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html
//
// Real objective, real training: a deterministic house-price regression set (8 features with interactions,
// 3,000 train / 1,000 validation rows) and a polynomial-feature ridge model fitted by mini-batch SGD; every
// epoch reports the validation RMSE, so Hyperband's brackets can compare trials from different processes at
// the same epoch. The parent creates the study, starts the workers, waits, and then audits the shared file:
// unique trial numbers, every trial finished, and how much training the pruner saved.
//
//   Run:  dotnet run examples/0_DistributedJournalHyperband.cs

using System.Diagnostics;
using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;

const string StudyName = "house-prices-sgd";
const int MaxEpochs = 27;
string journal = args.Length > 1 ? args[1] : Path.Combine(Path.GetTempPath(), "optunasharp-house-prices.journal");

if (args.Length > 0 && args[0] == "--worker")
{
    // child mode: join the study through the shared file and run 16 trials on two threads.
    OptunaLogging.SetVerbosity(LogLevel.Warning);
    var data = Houses.Generate(seed: 41);
    var worker = Optuna.LoadStudy(StudyName, new JournalStorage(new JournalFileBackend(journal)),
        sampler: new TPESampler(seed: int.Parse(args[2], CultureInfo.InvariantCulture), nStartupTrials: 8),
        pruner: new HyperbandPruner(minResource: 1, maxResource: MaxEpochs, reductionFactor: 3));
    worker.Optimize(t => Objective(t, data), nTrials: 16, nJobs: 2);
    return;
}

OptunaLogging.SetVerbosity(LogLevel.Warning);
File.Delete(journal);
Optuna.CreateStudy(storage: new JournalStorage(new JournalFileBackend(journal)), studyName: StudyName, direction: StudyDirection.Minimize);

var sw = Stopwatch.StartNew();
string dll = typeof(Houses).Assembly.Location;
var workers = Enumerable.Range(1, 3).Select(id =>
{
    var psi = new ProcessStartInfo("dotnet") { RedirectStandardError = true, RedirectStandardOutput = true };
    foreach (var a in new[] { "exec", dll, "--worker", journal, id.ToString(CultureInfo.InvariantCulture) })
        psi.ArgumentList.Add(a);
    return Process.Start(psi)!;
}).ToList();
var errors = workers.Select(w => { w.StandardOutput.ReadToEnd(); string e = w.StandardError.ReadToEnd(); w.WaitForExit(); return (w.ExitCode, e); }).ToList();
sw.Stop();

var study = Optuna.LoadStudy(StudyName, new JournalStorage(new JournalFileBackend(journal)));
var trials = study.Trials;
var numbers = trials.Select(t => t.Number).OrderBy(n => n).ToList();
int epochs = trials.Sum(t => t.IntermediateValues.Count);
Console.WriteLine($"3 processes × 2 threads finished in {sw.Elapsed.TotalSeconds:F1} s, exit codes: {string.Join(", ", errors.Select(e => e.ExitCode))}");
foreach (var (code, err) in errors.Where(e => e.ExitCode != 0))
    Console.WriteLine("worker stderr: " + err.Trim().Split('\n').Last());
Console.WriteLine($"trials in the journal: {trials.Count}; numbers 0..{trials.Count - 1} each exactly once: {numbers.SequenceEqual(Enumerable.Range(0, trials.Count))}");
Console.WriteLine("states: " + string.Join(", ", trials.GroupBy(t => t.State).Select(g => $"{g.Key} {g.Count()}")));
Console.WriteLine($"epochs trained: {epochs} of {trials.Count * MaxEpochs} without pruning ({1 - (double)epochs / (trials.Count * MaxEpochs):P0} saved)");
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "best validation RMSE: {0:F4} (trial #{1}); predicting the mean: {2:F4}",
    study.BestValue, study.BestTrial.Number, Houses.Generate(seed: 41).MeanBaselineRmse()));
foreach (var kv in study.BestParams)
    Console.WriteLine($"    {kv.Key,-8} {kv.Value}");

/// <summary>Fits one SGD ridge configuration epoch by epoch, reporting the validation RMSE.</summary>
/// <param name="t">The trial proposing the learning rate, its decay, the L2 strength, the batch size and the feature degree.</param>
/// <param name="d">The house-price data.</param>
/// <returns>The validation RMSE after the last epoch.</returns>
/// <exception cref="TrialPruned">Hyperband stopped the configuration at one of its rungs.</exception>
static double Objective(Trial t, Houses d)
{
    double lr = t.SuggestFloat("lr", 1e-4, 0.5, log: true);
    double decay = t.SuggestFloat("decay", 0.85, 1.0);
    double l2 = t.SuggestFloat("l2", 1e-6, 1.0, log: true);
    int batch = (int)t.SuggestCategorical("batch", new[] { 8, 32, 128 });
    bool quadratic = t.SuggestCategorical("quadratic", new[] { false, true });   // add pairwise products
    var xs = quadratic ? d.TrainQ : d.Train;
    var vs = quadratic ? d.ValidQ : d.Valid;
    var w = new double[xs[0].Length];
    double b = 0, rmse = double.NaN;
    for (int epoch = 1; epoch <= MaxEpochs; epoch++)
    {
        var order = Enumerable.Range(0, xs.Length).ToArray();
        new Random(epoch).Shuffle(order);
        for (int s = 0; s < order.Length; s += batch)
        {
            var g = new double[w.Length];
            double gb = 0;
            int end = Math.Min(s + batch, order.Length);
            for (int k = s; k < end; k++)
            {
                int i = order[k];
                double err = Dot(w, xs[i]) + b - d.TrainY[i];
                for (int j = 0; j < w.Length; j++) g[j] += err * xs[i][j];
                gb += err;
            }
            for (int j = 0; j < w.Length; j++) w[j] -= lr * (g[j] / (end - s) + l2 * w[j]);
            b -= lr * gb / (end - s);
        }
        lr *= decay;
        rmse = Math.Sqrt(Enumerable.Range(0, vs.Length).Average(i => Math.Pow(Dot(w, vs[i]) + b - d.ValidY[i], 2)));
        if (!double.IsFinite(rmse)) rmse = 1e3;                                  // diverged: a bad finite value
        t.Report(rmse, epoch);
        if (t.ShouldPrune())
            throw new TrialPruned();
    }
    return rmse;

    static double Dot(double[] w, double[] x)
    {
        double s = 0;
        for (int j = 0; j < w.Length; j++) s += w[j] * x[j];
        return s;
    }
}

/// <summary>A standardized house-price regression set, with an optional quadratic feature expansion.</summary>
/// <param name="Train">Training rows (8 standardized features).</param>
/// <param name="TrainQ">Training rows with all pairwise products appended.</param>
/// <param name="TrainY">Training targets (log price, centred).</param>
/// <param name="Valid">Validation rows.</param>
/// <param name="ValidQ">Validation rows with pairwise products.</param>
/// <param name="ValidY">Validation targets.</param>
sealed record Houses(double[][] Train, double[][] TrainQ, double[] TrainY, double[][] Valid, double[][] ValidQ, double[] ValidY)
{
    /// <summary>Generates the data; the same seed gives the same houses.</summary>
    /// <param name="seed">Data seed.</param>
    /// <returns>The data set.</returns>
    public static Houses Generate(int seed)
    {
        var rng = new Random(seed);   // data generation only — samplers never use System.Random
        int n = 4000;
        var x = new double[n][];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            var r = Enumerable.Range(0, 8).Select(_ => Normal(rng)).ToArray();   // area, rooms, age, distance, …
            y[i] = 0.5 * r[0] + 0.2 * r[1] - 0.25 * r[2] - 0.3 * r[3] + 0.15 * r[0] * r[4] - 0.1 * r[3] * r[3] + 0.1 * Normal(rng);
            x[i] = r;
        }
        double[] Quad(double[] r) => r.Concat(from a in Enumerable.Range(0, 8) from c in Enumerable.Range(a, 8 - a) select r[a] * r[c]).ToArray();
        double mean = y.Take(3000).Average();
        var yc = y.Select(v => v - mean).ToArray();
        return new Houses(x[..3000], x[..3000].Select(Quad).ToArray(), yc[..3000], x[3000..], x[3000..].Select(Quad).ToArray(), yc[3000..]);
    }

    /// <summary>RMSE of predicting the training mean on the validation rows.</summary>
    /// <returns>The baseline RMSE.</returns>
    public double MeanBaselineRmse() => Math.Sqrt(ValidY.Average(v => v * v));

    /// <summary>A standard normal draw (Box–Muller).</summary>
    /// <param name="rng">The random stream.</param>
    /// <returns>A N(0, 1) variate.</returns>
    private static double Normal(Random rng) => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
}
