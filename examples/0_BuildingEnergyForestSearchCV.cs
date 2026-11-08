#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Building energy model with OptunaSearchCV — the scikit-learn-style integration: OptunaSearchCV
// cross-validates a random-forest regressor (OptunaSharp's managed, byte-exact port of scikit-learn's
// RandomForestRegressor) that predicts a building's heating load from its geometry and glazing, the way an
// energy consultant screens designs before running a full thermal simulation.
//
// source: https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.OptunaSearchCV.html
//         (OptunaSearchCV: an estimator, `param_distributions`, `n_trials`, CV scoring, refit)
//
// Real training: a deterministic table of 768 building designs shaped like the classic energy-efficiency
// study (relative compactness, surface/wall/roof area, height, orientation, glazing area and its
// distribution) with a heating load that depends non-linearly on them; 5-fold KFold CV with R² as the score.
// The forest's depth, leaf size, feature subsampling and size are searched; the refitted best forest is then
// scored on 200 held-out designs it never saw.
//
//   Run:  dotnet run examples/0_BuildingEnergyForestSearchCV.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Integration;
using OptunaSharp.Integration.Sklearn;
using OptunaSharp.Logging;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var (x, y) = Buildings.Generate(rows: 968, seed: 47);
var (xTrain, yTrain) = (Slice(x, 0, 768), y[..768]);
var (xTest, yTest) = (Slice(x, 768, 968), y[768..]);

var search = new OptunaSearchCV(
    new SklearnEstimator(new RandomForestRegressor(randomState: 0), xTrain, yTrain, cv: new KFold(5, shuffle: true, randomState: 0)),
    new Dictionary<string, BaseDistribution>
    {
        ["n_estimators"] = new IntDistribution(20, 120, step: 20),
        ["max_depth"] = new IntDistribution(2, 20),
        ["min_samples_leaf"] = new IntDistribution(1, 20, log: true),
        ["max_features"] = new FloatDistribution(0.2, 1.0),
    },
    nTrials: 25,
    randomState: 0);
search.Fit();

Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "best 5-fold CV R²: {0:F4} after {1} trials", search.BestScore, search.Study.Trials.Count));
foreach (var kv in search.BestParams)
    Console.WriteLine($"    {kv.Key,-17} {kv.Value}");
var forest = new RandomForestRegressor(
    nEstimators: (int)(long)search.BestParams["n_estimators"]!, maxDepth: (int)(long)search.BestParams["max_depth"]!,
    minSamplesLeaf: (int)(long)search.BestParams["min_samples_leaf"]!, maxFeatures: search.BestParams["max_features"], randomState: 0);
forest.Fit(xTrain, yTrain);
double[] pred = forest.Predict(xTest);
double mean = yTest.Average();
double r2 = 1 - yTest.Zip(pred, (a, p) => (a - p) * (a - p)).Sum() / yTest.Sum(a => (a - mean) * (a - mean));
double mae = yTest.Zip(pred, (a, p) => Math.Abs(a - p)).Average();
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "held-out designs: R² {0:F4}, mean absolute error {1:F2} kWh/m²", r2, mae));

/// <summary>Copies a block of rows out of a matrix.</summary>
/// <param name="m">The matrix.</param>
/// <param name="from">First row (inclusive).</param>
/// <param name="to">Last row (exclusive).</param>
/// <returns>The rows as a new matrix.</returns>
static double[,] Slice(double[,] m, int from, int to)
{
    var o = new double[to - from, m.GetLength(1)];
    for (int i = from; i < to; i++)
        for (int j = 0; j < m.GetLength(1); j++)
            o[i - from, j] = m[i, j];
    return o;
}

/// <summary>The synthetic building-design table.</summary>
static class Buildings
{
    /// <summary>Generates designs and heating loads; the same seed gives the same table.</summary>
    /// <param name="rows">Number of designs.</param>
    /// <param name="seed">Data seed.</param>
    /// <returns>Eight features per design and its heating load in kWh/m².</returns>
    public static (double[,] X, double[] Y) Generate(int rows, int seed)
    {
        var rng = new Random(seed);   // data generation only — samplers never use System.Random
        var x = new double[rows, 8];
        var y = new double[rows];
        for (int i = 0; i < rows; i++)
        {
            double compactness = 0.62 + 0.36 * rng.NextDouble();
            double surface = 515 + 300 * (1 - compactness) / 0.36 + 10 * rng.NextDouble();
            double wall = 245 + 170 * rng.NextDouble();
            double roof = Math.Max(110, (surface - wall) / 2);
            double height = compactness > 0.75 ? 7.0 : 3.5;
            double orientation = 2 + rng.Next(4);
            double glazing = new[] { 0.0, 0.1, 0.25, 0.4 }[rng.Next(4)];
            double distribution = glazing == 0 ? 0 : 1 + rng.Next(5);
            // heating load: tall compact buildings lose less per m²; glazing and roof area add losses.
            y[i] = 6 + 28 * (height / 7.0) * Math.Pow(compactness, 1.5) + 0.018 * roof * (height < 5 ? 0.4 : 1.0)
                   + 22 * glazing + 0.6 * (distribution is 2 or 3 ? glazing * 4 : 0) + 0.2 * (orientation - 3.5) + 0.8 * (rng.NextDouble() - 0.5);
            double[] r = { compactness, surface, wall, roof, height, orientation, glazing, distribution };
            for (int j = 0; j < 8; j++)
                x[i, j] = r[j];
        }
        return (x, y);
    }
}
