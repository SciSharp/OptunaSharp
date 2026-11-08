#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Recommender tuning in parallel — four worker threads (`n_jobs=4`) tune a matrix-factorization
// recommender trained by SGD, while SuccessiveHalvingPruner (asynchronous successive halving, ASHA) promotes
// only the best configurations to more epochs and TPE's constant liar keeps the parallel workers from
// proposing the same point.
//
// source: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html
//         (ASHA: rungs at min_resource × reduction_factor^k, asynchronous promotion)
//         https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
//         (`n_jobs`: parallel trials in one process)
//         https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
//         (`constant_liar`: running trials are treated as bad so parallel workers spread out)
//
// Real objective: a deterministic MovieLens-sized rating matrix (800 users × 500 items, 40,000 ratings from
// a rank-6 taste model plus user/item biases and noise) split into train/validation, and a biased
// matrix-factorization model fitted by SGD — rank, learning rate, regularization and its decay are the
// knobs. Each epoch reports the validation RMSE; the pruner compares trials at its rungs (1, 3, 9, 27
// epochs). The trials run on four threads, so the order of completions — and therefore the run — varies.
//
//   Run:  dotnet run examples/0_RecommenderParallelAsha.cs

using System.Diagnostics;
using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

const int MaxEpochs = 27;
var ratings = Ratings.Generate(users: 800, items: 500, count: 40_000, seed: 4);

var study = Optuna.CreateStudy(
    sampler: new TPESampler(seed: 9, nStartupTrials: 16, constantLiar: true),
    pruner: new SuccessiveHalvingPruner(minResource: 1, reductionFactor: 3),
    direction: StudyDirection.Minimize,
    studyName: "mf-asha");

var sw = Stopwatch.StartNew();
study.Optimize(t => Objective(t, ratings), nTrials: 96, nJobs: 4);
sw.Stop();

var byState = study.Trials.GroupBy(t => t.State).ToDictionary(g => g.Key, g => g.Count());
var epochsPerTrial = study.Trials.Select(t => t.IntermediateValues.Count).ToList();
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0} trials on 4 workers in {1:F1} s: {2} complete, {3} pruned",
    study.Trials.Count, sw.Elapsed.TotalSeconds, byState.GetValueOrDefault(TrialState.Complete), byState.GetValueOrDefault(TrialState.Pruned)));
Console.WriteLine("epochs reached (rung histogram): " + string.Join("  ",
    new[] { 1, 3, 9, 27 }.Select(r => $"≥{r}: {epochsPerTrial.Count(e => e >= r)}")));
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "best validation RMSE {0:F4} (trial #{1}); a global-mean predictor scores {2:F4}",
    study.BestValue, study.BestTrial.Number, ratings.MeanBaselineRmse()));
foreach (var kv in study.BestParams)
    Console.WriteLine($"    {kv.Key,-8} {kv.Value}");
var numbers = study.Trials.Select(t => t.Number).OrderBy(n => n).ToList();
Console.WriteLine($"trial numbers unique and contiguous: {numbers.SequenceEqual(Enumerable.Range(0, numbers.Count))}");

/// <summary>Fits one factorization configuration epoch by epoch, reporting the validation RMSE.</summary>
/// <param name="t">The trial proposing rank, learning rate, regularization and decay.</param>
/// <param name="r">The rating data.</param>
/// <returns>The validation RMSE after the last epoch.</returns>
/// <exception cref="TrialPruned">ASHA did not promote this configuration to the next rung.</exception>
static double Objective(Trial t, Ratings r)
{
    int rank = (int)t.SuggestInt("rank", 2, 64, log: true);
    double lr = t.SuggestFloat("lr", 1e-3, 0.1, log: true);
    double reg = t.SuggestFloat("reg", 1e-4, 0.5, log: true);
    double decay = t.SuggestFloat("lr_decay", 0.8, 1.0);           // per-epoch learning-rate multiplier
    var model = new BiasedMf(r.Users, r.Items, rank, r.GlobalMean, seed: 1);
    double rmse = double.NaN;
    for (int epoch = 1; epoch <= MaxEpochs; epoch++)
    {
        model.Epoch(r.Train, lr, reg, seed: epoch);
        lr *= decay;
        rmse = model.Rmse(r.Valid);
        if (!double.IsFinite(rmse))
            rmse = 10.0;                                             // diverged: report a bad finite value
        t.Report(rmse, epoch);
        if (t.ShouldPrune())
            throw new TrialPruned();
    }
    return rmse;
}

/// <summary>A synthetic explicit-feedback rating set with a train/validation split.</summary>
/// <param name="Users">User count.</param>
/// <param name="Items">Item count.</param>
/// <param name="GlobalMean">Mean training rating.</param>
/// <param name="Train">Training (user, item, rating) triples.</param>
/// <param name="Valid">Validation triples.</param>
sealed record Ratings(int Users, int Items, double GlobalMean, (int U, int I, double R)[] Train, (int U, int I, double R)[] Valid)
{
    /// <summary>Generates ratings from a low-rank taste model; the same seed gives the same data.</summary>
    /// <param name="users">User count.</param>
    /// <param name="items">Item count.</param>
    /// <param name="count">Number of observed ratings (distinct user–item pairs).</param>
    /// <param name="seed">Data seed.</param>
    /// <returns>The data set, 85 % train / 15 % validation.</returns>
    public static Ratings Generate(int users, int items, int count, int seed)
    {
        var rng = new Random(seed);   // data generation only — samplers never use System.Random
        const int TrueRank = 6;
        double[][] p = Enumerable.Range(0, users).Select(_ => Enumerable.Range(0, TrueRank).Select(_ => Normal(rng) * 0.5).ToArray()).ToArray();
        double[][] q = Enumerable.Range(0, items).Select(_ => Enumerable.Range(0, TrueRank).Select(_ => Normal(rng) * 0.5).ToArray()).ToArray();
        double[] bu = Enumerable.Range(0, users).Select(_ => Normal(rng) * 0.4).ToArray();
        double[] bi = Enumerable.Range(0, items).Select(_ => Normal(rng) * 0.5).ToArray();
        var seen = new HashSet<long>();
        var all = new List<(int, int, double)>();
        while (all.Count < count)
        {
            // popular items get more ratings (a Zipf-like skew), as in real catalogs.
            int u = rng.Next(users), i = (int)(items * Math.Pow(rng.NextDouble(), 2));
            if (!seen.Add((long)u * items + i))
                continue;
            double score = 3.6 + bu[u] + bi[i] + Enumerable.Range(0, TrueRank).Sum(k => p[u][k] * q[i][k]) + Normal(rng) * 0.5;
            all.Add((u, i, Math.Clamp(Math.Round(score * 2) / 2, 0.5, 5.0)));   // half-star scale
        }
        int cut = (int)(all.Count * 0.85);
        var train = all.Take(cut).ToArray();
        return new Ratings(users, items, train.Average(x => x.Item3), train, all.Skip(cut).ToArray());
    }

    /// <summary>RMSE of always predicting the training mean (the floor any model must beat).</summary>
    /// <returns>The baseline validation RMSE.</returns>
    public double MeanBaselineRmse() => Math.Sqrt(Valid.Average(x => (x.R - GlobalMean) * (x.R - GlobalMean)));

    /// <summary>A standard normal draw (Box–Muller).</summary>
    /// <param name="rng">The random stream.</param>
    /// <returns>A N(0, 1) variate.</returns>
    private static double Normal(Random rng) => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
}

/// <summary>Biased matrix factorization (Funk-SVD style): r̂ = μ + b_u + b_i + p_u·q_i.</summary>
sealed class BiasedMf
{
    private readonly double[,] _p, _q;
    private readonly double[] _bu, _bi;
    private readonly double _mu;
    private readonly int _rank;

    /// <summary>Initializes small random factors from a fixed seed.</summary>
    /// <param name="users">User count.</param>
    /// <param name="items">Item count.</param>
    /// <param name="rank">Latent dimension.</param>
    /// <param name="mu">Global mean rating.</param>
    /// <param name="seed">Initialization seed.</param>
    public BiasedMf(int users, int items, int rank, double mu, int seed)
    {
        var rng = new Random(seed);
        (_rank, _mu) = (rank, mu);
        _p = new double[users, rank];
        _q = new double[items, rank];
        for (int u = 0; u < users; u++)
            for (int k = 0; k < rank; k++)
                _p[u, k] = (rng.NextDouble() - 0.5) * 0.1;
        for (int i = 0; i < items; i++)
            for (int k = 0; k < rank; k++)
                _q[i, k] = (rng.NextDouble() - 0.5) * 0.1;
        _bu = new double[users];
        _bi = new double[items];
    }

    /// <summary>One SGD pass over the training ratings in a seeded shuffled order.</summary>
    /// <param name="train">Training triples.</param>
    /// <param name="lr">Learning rate.</param>
    /// <param name="reg">L2 regularization on factors and biases.</param>
    /// <param name="seed">Shuffle seed.</param>
    public void Epoch((int U, int I, double R)[] train, double lr, double reg, int seed)
    {
        var order = Enumerable.Range(0, train.Length).ToArray();
        new Random(seed).Shuffle(order);
        foreach (int idx in order)
        {
            var (u, i, r) = train[idx];
            double err = r - Predict(u, i);
            _bu[u] += lr * (err - reg * _bu[u]);
            _bi[i] += lr * (err - reg * _bi[i]);
            for (int k = 0; k < _rank; k++)
            {
                double pu = _p[u, k];
                _p[u, k] += lr * (err * _q[i, k] - reg * pu);
                _q[i, k] += lr * (err * pu - reg * _q[i, k]);
            }
        }
    }

    /// <summary>Root-mean-square error over a set of ratings.</summary>
    /// <param name="data">The triples to score.</param>
    /// <returns>The RMSE (NaN or infinity when the factors diverged).</returns>
    public double Rmse((int U, int I, double R)[] data) => Math.Sqrt(data.Average(x => Math.Pow(x.R - Predict(x.U, x.I), 2)));

    /// <summary>Predicts one rating.</summary>
    /// <param name="u">User index.</param>
    /// <param name="i">Item index.</param>
    /// <returns>The predicted rating (unclipped).</returns>
    private double Predict(int u, int i)
    {
        double s = _mu + _bu[u] + _bi[i];
        for (int k = 0; k < _rank; k++)
            s += _p[u, k] * _q[i, k];
        return s;
    }
}
