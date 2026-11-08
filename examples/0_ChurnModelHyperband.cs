#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Churn model tuning — TPE + HyperbandPruner tune a small neural network that predicts customer churn,
// trained for real (mini-batch SGD with momentum) inside every trial; Hyperband stops the hopeless
// configurations after a few epochs so the budget goes to the promising ones.
//
// source: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html
//         (the pruner, and its `min_resource` / `max_resource` = epochs)
//         https://github.com/optuna/optuna-examples/blob/main/README.md
//         (the optuna-examples suite whose `*_simple.py` training loops this follows: report per epoch,
//         raise TrialPruned when `should_prune()`)
//
// Real objective: a deterministic synthetic churn table (12 features — tenure, monthly charge, support
// calls, contract type, … with non-linear interactions and label noise; 4,000 train / 1,000 validation
// rows) and a one-hidden-layer MLP written out by hand. Each epoch the trial reports the validation
// log-loss; Hyperband's brackets compare trials at the same epoch and prune the laggards. The tuned model is
// retrained at the end and scored on a held-out test set, so the number printed is a real generalization
// estimate, not the search's own validation score.
//
//   Run:  dotnet run examples/0_ChurnModelHyperband.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

const int MaxEpochs = 30;
var data = ChurnData.Generate(seed: 3, train: 4000, valid: 1000, test: 2000);

var study = Optuna.CreateStudy(
    sampler: new TPESampler(seed: 5, nStartupTrials: 12),
    pruner: new HyperbandPruner(minResource: 2, maxResource: MaxEpochs, reductionFactor: 3),
    direction: StudyDirection.Minimize,
    studyName: "churn-mlp");
study.Optimize(t => Objective(t, data), nTrials: 80);

var states = study.Trials.GroupBy(t => t.State).ToDictionary(g => g.Key, g => g.Count());
long epochsRun = study.Trials.Sum(t => (long)t.IntermediateValues.Count);
Console.WriteLine($"trials: {study.Trials.Count}  complete: {states.GetValueOrDefault(TrialState.Complete)}  "
                  + $"pruned: {states.GetValueOrDefault(TrialState.Pruned)}");
Console.WriteLine($"epochs trained: {epochsRun} of {study.Trials.Count * MaxEpochs} an unpruned search would spend");
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "best validation log-loss: {0:F4} (trial #{1})",
    study.BestValue, study.BestTrial.Number));
foreach (var kv in study.BestParams)
    Console.WriteLine($"    {kv.Key,-10} {kv.Value}");

// retrain the winner from scratch and score it on rows no trial ever saw.
var bestModel = Mlp.FromParams(study.BestParams, data.Features);
for (int epoch = 0; epoch < MaxEpochs; epoch++)
    bestModel.TrainEpoch(data.TrainX, data.TrainY, seed: epoch);
var (testLoss, testAcc) = bestModel.Evaluate(data.TestX, data.TestY);
var (baseLoss, baseAcc) = Baseline(data);
Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
    "held-out test: log-loss {0:F4}, accuracy {1:P1}   (majority-class baseline: {2:F4}, {3:P1})",
    testLoss, testAcc, baseLoss, baseAcc));

/// <summary>Trains one MLP configuration, reporting the validation loss after every epoch.</summary>
/// <param name="t">The trial proposing the architecture and optimizer settings.</param>
/// <param name="d">The churn dataset.</param>
/// <returns>The validation log-loss after the last epoch.</returns>
/// <exception cref="TrialPruned">Hyperband decided this configuration cannot catch up.</exception>
static double Objective(Trial t, ChurnData d)
{
    var model = new Mlp(
        inputs: d.Features,
        hidden: (int)t.SuggestInt("hidden", 4, 128, log: true),            // width of the hidden layer
        activation: t.SuggestCategorical("activation", new[] { "relu", "tanh" }),
        lr: t.SuggestFloat("lr", 1e-4, 1.0, log: true),                    // SGD step size
        momentum: t.SuggestFloat("momentum", 0.0, 0.99),
        l2: t.SuggestFloat("l2", 1e-6, 1e-1, log: true),                   // weight decay
        batch: (int)t.SuggestCategorical("batch", new[] { 16, 32, 64, 128 }));
    double loss = double.NaN;
    for (int epoch = 0; epoch < MaxEpochs; epoch++)
    {
        model.TrainEpoch(d.TrainX, d.TrainY, seed: epoch);
        loss = model.Evaluate(d.ValidX, d.ValidY).LogLoss;
        if (double.IsNaN(loss))
            loss = 10.0;                                   // a diverged run: worst-case loss, not NaN
        t.Report(loss, epoch + 1);                         // step = epochs completed (Hyperband's resource)
        if (t.ShouldPrune())
            throw new TrialPruned();
    }
    return loss;
}

/// <summary>Scores the constant "predict the training churn rate" model.</summary>
/// <param name="d">The dataset.</param>
/// <returns>The baseline's test log-loss and accuracy.</returns>
static (double LogLoss, double Accuracy) Baseline(ChurnData d)
{
    double p = d.TrainY.Average();
    double loss = -d.TestY.Average(y => y * Math.Log(p) + (1 - y) * Math.Log(1 - p));
    double acc = d.TestY.Average(y => (p >= 0.5 ? 1.0 : 0.0) == y ? 1.0 : 0.0);
    return (loss, acc);
}

/// <summary>A deterministic synthetic customer-churn dataset, standardized, split three ways.</summary>
/// <param name="Features">Number of input columns.</param>
/// <param name="TrainX">Training rows.</param>
/// <param name="TrainY">Training labels (1 = churned).</param>
/// <param name="ValidX">Validation rows (what the study scores).</param>
/// <param name="ValidY">Validation labels.</param>
/// <param name="TestX">Held-out rows, used once at the end.</param>
/// <param name="TestY">Held-out labels.</param>
sealed record ChurnData(int Features, double[][] TrainX, double[] TrainY, double[][] ValidX, double[] ValidY,
    double[][] TestX, double[] TestY)
{
    /// <summary>Generates the dataset; the same seed always produces the same rows.</summary>
    /// <param name="seed">Seed of the data generator (not of any sampler).</param>
    /// <param name="train">Training row count.</param>
    /// <param name="valid">Validation row count.</param>
    /// <param name="test">Test row count.</param>
    /// <returns>The standardized dataset.</returns>
    public static ChurnData Generate(int seed, int train, int valid, int test)
    {
        var rng = new Random(seed);   // data generation only — samplers never use System.Random
        int n = train + valid + test;
        var x = new double[n][];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            double tenure = rng.NextDouble() * 72;                      // months as a customer
            double monthly = 20 + rng.NextDouble() * 100;               // monthly charge
            double calls = Poisson(rng, 1.5);                           // support calls last quarter
            double contract = rng.Next(3);                              // 0 monthly, 1 one-year, 2 two-year
            double fiber = rng.NextDouble() < 0.45 ? 1 : 0;
            double paperless = rng.NextDouble() < 0.6 ? 1 : 0;
            double seniors = rng.NextDouble() < 0.16 ? 1 : 0;
            double streaming = rng.NextDouble() < 0.4 ? 1 : 0;
            double outages = Poisson(rng, 0.7);
            double discount = rng.NextDouble() < 0.25 ? 1 : 0;
            double tickets = Poisson(rng, 0.4);
            double noise = rng.NextDouble();                            // an irrelevant column
            // the true churn propensity: interactions a linear model misses (calls hurt more when monthly;
            // fiber + outages compounds; long tenure saturates).
            double z = -1.2 + 1.6 * (contract == 0 ? 1 : 0) - 0.04 * Math.Min(tenure, 36)
                       + 0.018 * (monthly - 70) + 0.45 * calls * (contract == 0 ? 1.0 : 0.3)
                       + 0.8 * fiber * Math.Min(outages, 3) / 3.0 + 0.3 * paperless + 0.25 * seniors
                       - 0.6 * discount + 0.35 * tickets - 0.2 * streaming;
            y[i] = rng.NextDouble() < 1 / (1 + Math.Exp(-z)) ? 1 : 0;
            x[i] = new[] { tenure, monthly, calls, contract, fiber, paperless, seniors, streaming, outages,
                discount, tickets, noise };
        }
        int f = x[0].Length;
        // standardize with the TRAINING rows' statistics only (no peeking at validation/test).
        for (int j = 0; j < f; j++)
        {
            double mean = 0, sq = 0;
            for (int i = 0; i < train; i++)
                mean += x[i][j];
            mean /= train;
            for (int i = 0; i < train; i++)
                sq += (x[i][j] - mean) * (x[i][j] - mean);
            double sd = Math.Sqrt(sq / train) + 1e-12;
            for (int i = 0; i < n; i++)
                x[i][j] = (x[i][j] - mean) / sd;
        }
        return new ChurnData(f, x[..train], y[..train], x[train..(train + valid)], y[train..(train + valid)],
            x[(train + valid)..], y[(train + valid)..]);
    }

    /// <summary>Draws a Poisson variate by Knuth's multiplication method.</summary>
    /// <param name="rng">The data generator.</param>
    /// <param name="lambda">The Poisson mean.</param>
    /// <returns>A non-negative count.</returns>
    private static double Poisson(Random rng, double lambda)
    {
        double l = Math.Exp(-lambda), p = 1;
        int k = 0;
        do { k++; p *= rng.NextDouble(); } while (p > l);
        return k - 1;
    }
}

/// <summary>
///     A one-hidden-layer binary classifier trained by mini-batch SGD with momentum and L2 decay — small, but
///     a real training loop with real divergence when the learning rate is too high.
/// </summary>
sealed class Mlp
{
    private readonly int _in, _hidden, _batch;
    private readonly bool _relu;
    private readonly double _lr, _momentum, _l2;
    private readonly double[,] _w1;
    private readonly double[] _b1, _w2;
    private double _b2;
    private readonly double[,] _vw1;
    private readonly double[] _vb1, _vw2;
    private double _vb2;

    /// <summary>Creates the network with Glorot-uniform weights from a fixed seed.</summary>
    /// <param name="inputs">Input width.</param>
    /// <param name="hidden">Hidden-layer width.</param>
    /// <param name="activation">"relu" or "tanh".</param>
    /// <param name="lr">Learning rate.</param>
    /// <param name="momentum">Momentum coefficient in [0, 1).</param>
    /// <param name="l2">L2 weight-decay coefficient.</param>
    /// <param name="batch">Mini-batch size.</param>
    public Mlp(int inputs, int hidden, string activation, double lr, double momentum, double l2, int batch)
    {
        (_in, _hidden, _batch, _relu, _lr, _momentum, _l2) = (inputs, hidden, batch, activation == "relu", lr, momentum, l2);
        var rng = new Random(17);
        double a1 = Math.Sqrt(6.0 / (inputs + hidden)), a2 = Math.Sqrt(6.0 / (hidden + 1));
        _w1 = new double[hidden, inputs];
        _b1 = new double[hidden];
        _w2 = new double[hidden];
        for (int h = 0; h < hidden; h++)
        {
            for (int i = 0; i < inputs; i++)
                _w1[h, i] = (rng.NextDouble() * 2 - 1) * a1;
            _w2[h] = (rng.NextDouble() * 2 - 1) * a2;
        }
        _vw1 = new double[hidden, inputs];
        _vb1 = new double[hidden];
        _vw2 = new double[hidden];
    }

    /// <summary>Builds a network from a finished study's best parameters.</summary>
    /// <param name="p">The parameter dictionary (<c>study.BestParams</c>).</param>
    /// <param name="inputs">Input width.</param>
    /// <returns>An untrained network with those hyperparameters.</returns>
    public static Mlp FromParams(IReadOnlyDictionary<string, object?> p, int inputs) => new(
        inputs, (int)(long)p["hidden"]!, (string)p["activation"]!, (double)p["lr"]!, (double)p["momentum"]!,
        (double)p["l2"]!, (int)(long)p["batch"]!);

    /// <summary>One pass over the training rows in a seeded shuffled order.</summary>
    /// <param name="x">Training rows.</param>
    /// <param name="y">Training labels.</param>
    /// <param name="seed">Shuffle seed (the epoch number, so every configuration sees the same order).</param>
    public void TrainEpoch(double[][] x, double[] y, int seed)
    {
        var order = Enumerable.Range(0, x.Length).ToArray();
        new Random(seed).Shuffle(order);
        var gw1 = new double[_hidden, _in];
        var gb1 = new double[_hidden];
        var gw2 = new double[_hidden];
        var act = new double[_hidden];
        for (int start = 0; start < order.Length; start += _batch)
        {
            int end = Math.Min(start + _batch, order.Length);
            Array.Clear(gw1);
            Array.Clear(gb1);
            Array.Clear(gw2);
            double gb2 = 0;
            for (int r = start; r < end; r++)
            {
                double[] row = x[order[r]];
                double logit = Forward(row, act);
                double dz = Sigmoid(logit) - y[order[r]];          // d(log-loss)/d(logit)
                gb2 += dz;
                for (int h = 0; h < _hidden; h++)
                {
                    gw2[h] += dz * act[h];
                    double da = dz * _w2[h] * (_relu ? (act[h] > 0 ? 1 : 0) : 1 - act[h] * act[h]);
                    gb1[h] += da;
                    for (int i = 0; i < _in; i++)
                        gw1[h, i] += da * row[i];
                }
            }
            double scale = 1.0 / (end - start);
            for (int h = 0; h < _hidden; h++)
            {
                for (int i = 0; i < _in; i++)
                {
                    _vw1[h, i] = _momentum * _vw1[h, i] - _lr * (gw1[h, i] * scale + _l2 * _w1[h, i]);
                    _w1[h, i] += _vw1[h, i];
                }
                _vb1[h] = _momentum * _vb1[h] - _lr * gb1[h] * scale;
                _b1[h] += _vb1[h];
                _vw2[h] = _momentum * _vw2[h] - _lr * (gw2[h] * scale + _l2 * _w2[h]);
                _w2[h] += _vw2[h];
            }
            _vb2 = _momentum * _vb2 - _lr * gb2 * scale;
            _b2 += _vb2;
        }
    }

    /// <summary>Mean log-loss and accuracy over a set of rows.</summary>
    /// <param name="x">Rows to score.</param>
    /// <param name="y">Their labels.</param>
    /// <returns>The log-loss (probabilities clipped to [1e-12, 1 − 1e-12]) and the 0.5-threshold accuracy.</returns>
    public (double LogLoss, double Accuracy) Evaluate(double[][] x, double[] y)
    {
        var act = new double[_hidden];
        double loss = 0, correct = 0;
        for (int r = 0; r < x.Length; r++)
        {
            double p = Math.Clamp(Sigmoid(Forward(x[r], act)), 1e-12, 1 - 1e-12);
            loss -= y[r] * Math.Log(p) + (1 - y[r]) * Math.Log(1 - p);
            correct += (p >= 0.5 ? 1.0 : 0.0) == y[r] ? 1 : 0;
        }
        return (loss / x.Length, correct / x.Length);
    }

    /// <summary>Forward pass for one row.</summary>
    /// <param name="row">The input row.</param>
    /// <param name="act">Receives the hidden activations (reused buffer).</param>
    /// <returns>The output logit.</returns>
    private double Forward(double[] row, double[] act)
    {
        double logit = _b2;
        for (int h = 0; h < _hidden; h++)
        {
            double z = _b1[h];
            for (int i = 0; i < _in; i++)
                z += _w1[h, i] * row[i];
            act[h] = _relu ? Math.Max(0, z) : Math.Tanh(z);
            logit += _w2[h] * act[h];
        }
        return logit;
    }

    /// <summary>The logistic function, computed without overflow for large negative inputs.</summary>
    /// <param name="z">The logit.</param>
    /// <returns>A probability in (0, 1).</returns>
    private static double Sigmoid(double z) => z >= 0 ? 1 / (1 + Math.Exp(-z)) : Math.Exp(z) / (1 + Math.Exp(z));
}
