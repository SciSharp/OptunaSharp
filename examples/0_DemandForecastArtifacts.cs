#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Demand forecasting with model artifacts — TPE tunes a Holt-Winters exponential-smoothing forecaster for a
// store's daily sales with rolling-origin cross-validation; every finished trial saves its fitted model as an
// Optuna artifact (FileSystemArtifactStore), and the best trial's model file is downloaded at the end, the
// way a team hands the winning model to the deployment job.
//
// source: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/012_artifact_tutorial.html
//         (artifacts: `upload_artifact`, `get_all_artifact_meta`, `download_artifact`)
//         https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html
//         https://otexts.com/fpp3/holt-winters.html (Holt-Winters' seasonal method, damped trend)
//
// Real objective: two years of deterministic daily sales (trend, weekly and yearly seasonality,
// promotions, noise). Holt-Winters with a damped additive trend and additive or multiplicative weekly
// seasonality is fitted, and five rolling origins each forecast the next 28 days; the score is the mean
// sMAPE. After every fold the running mean so far is reported, so MedianPruner compares configurations on
// the same folds and stops one whose running mean is worse than the median of earlier trials at that fold
// (folds are not a learning curve, so a "has it stopped improving" pruner such as PatientPruner would be
// the wrong tool here). A completed trial
// writes {parameters, final level/trend/season, next-28-day forecast} to JSON and uploads it.
//
//   Run:  dotnet run examples/0_DemandForecastArtifacts.cs

using System.Globalization;
using System.Text.Json;
using OptunaSharp;
using OptunaSharp.Artifacts;
using OptunaSharp.Logging;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

double[] sales = SalesHistory.Generate(days: 730, seed: 12);
string artifactDir = Path.Combine(Path.GetTempPath(), "optunasharp-forecast-artifacts");
Directory.CreateDirectory(artifactDir);
var store = new FileSystemArtifactStore(artifactDir);       // where the model files live

var storage = new InMemoryStorage();                       // kept: artifact lookups on a FrozenTrial need it
var study = Optuna.CreateStudy(
    storage: storage,
    sampler: new TPESampler(seed: 14, nStartupTrials: 15),
    pruner: new MedianPruner(nStartupTrials: 8, nWarmupSteps: 1),   // judge from the second fold on
    direction: StudyDirection.Minimize,
    studyName: "store-17-forecast");
study.Optimize(t => Objective(t, sales, store, artifactDir), nTrials: 80);

var best = study.BestTrial;
var metas = ArtifactFunctions.GetAllArtifactMeta(best, storage);   // what the best trial uploaded
string downloaded = Path.Combine(artifactDir, $"best-model-{Guid.NewGuid():N}.json");
ArtifactFunctions.DownloadArtifact(store, downloaded, metas[0].ArtifactId);
using var model = JsonDocument.Parse(File.ReadAllText(downloaded));

int pruned = study.Trials.Count(t => t.State == TrialState.Pruned);
int withArtifacts = study.Trials.Count(t => ArtifactFunctions.GetAllArtifactMeta(t, storage).Count > 0);
Console.WriteLine($"trials: {study.Trials.Count}, pruned by MedianPruner: {pruned}, with a model artifact: {withArtifacts}");
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "best mean sMAPE {0:F3} % (trial #{1}); seasonal-naive forecast: {2:F3} %",
    best.Value, best.Number, SalesHistory.SeasonalNaiveSmape(sales)));
Console.WriteLine($"best model artifact: {metas[0].Filename} ({metas[0].Mimetype}), downloaded to {downloaded}");
Console.WriteLine("  parameters : " + model.RootElement.GetProperty("params").GetRawText());
var forecast = model.RootElement.GetProperty("forecast_28d").EnumerateArray().Select(e => e.GetDouble()).ToArray();
Console.WriteLine("  next 7 days: " + string.Join(", ", forecast.Take(7).Select(v => v.ToString("F0", CultureInfo.InvariantCulture))));

/// <summary>Cross-validates one Holt-Winters configuration and stores the refitted model as an artifact.</summary>
/// <param name="t">The trial proposing the smoothing parameters and the seasonality type.</param>
/// <param name="y">The daily sales history.</param>
/// <param name="store">The artifact store.</param>
/// <param name="dir">A scratch folder for the model file before upload.</param>
/// <returns>The mean sMAPE (%) over the five rolling origins.</returns>
/// <exception cref="TrialPruned">The configuration's running mean trailed the median of earlier trials.</exception>
static double Objective(Trial t, double[] y, IArtifactStore store, string dir)
{
    var p = new HwParams(
        Alpha: t.SuggestFloat("alpha", 0.01, 0.9),           // level smoothing
        Beta: t.SuggestFloat("beta", 0.001, 0.3, log: true), // trend smoothing
        Gamma: t.SuggestFloat("gamma", 0.01, 0.9),           // seasonal smoothing
        Phi: t.SuggestFloat("phi", 0.8, 1.0),                // trend damping (1 = undamped)
        Multiplicative: t.SuggestCategorical("season", new[] { "additive", "multiplicative" }) == "multiplicative");

    var scores = new List<double>();
    for (int fold = 0; fold < 5; fold++)
    {
        int origin = y.Length - 28 * (5 - fold);             // rolling origin: train on everything before it
        var fit = HoltWinters.Fit(y[..origin], p);
        scores.Add(Smape(y[origin..(origin + 28)], fit.Forecast(28)));
        t.Report(scores.Average(), fold);                    // the running mean: comparable across trials
        if (t.ShouldPrune())
            throw new TrialPruned();
    }

    // the configuration survived: refit on the full history and save what deployment needs.
    var final = HoltWinters.Fit(y, p);
    string file = Path.Combine(dir, $"trial-{t.Number}-model.json");
    File.WriteAllText(file, JsonSerializer.Serialize(new
    {
        @params = p,
        level = final.Level,
        trend = final.Trend,
        season = final.Season,
        forecast_28d = final.Forecast(28),
    }));
    ArtifactFunctions.UploadArtifact(store, file, t);       // attached to this trial in the study's storage
    File.Delete(file);
    return scores.Average();
}

/// <summary>Symmetric mean absolute percentage error.</summary>
/// <param name="actual">Observed values.</param>
/// <param name="forecast">Forecast values (same length).</param>
/// <returns>sMAPE in percent.</returns>
static double Smape(double[] actual, double[] forecast) =>
    100.0 * actual.Zip(forecast, (a, f) => Math.Abs(a - f) / ((Math.Abs(a) + Math.Abs(f)) / 2)).Average();

/// <summary>The smoothing parameters of one Holt-Winters model.</summary>
/// <param name="Alpha">Level smoothing in (0, 1).</param>
/// <param name="Beta">Trend smoothing in (0, 1).</param>
/// <param name="Gamma">Seasonal smoothing in (0, 1).</param>
/// <param name="Phi">Trend damping factor in (0, 1].</param>
/// <param name="Multiplicative">Multiplicative (true) or additive (false) weekly seasonality.</param>
sealed record HwParams(double Alpha, double Beta, double Gamma, double Phi, bool Multiplicative);

/// <summary>A fitted Holt-Winters state: level, trend and the seven seasonal indexes.</summary>
/// <param name="P">The parameters it was fitted with.</param>
/// <param name="Level">Final level.</param>
/// <param name="Trend">Final trend.</param>
/// <param name="Season">Final seasonal indexes, aligned so Season[0] belongs to the first forecast day.</param>
sealed record HoltWinters(HwParams P, double Level, double Trend, double[] Season)
{
    private const int M = 7;   // weekly season

    /// <summary>Runs the Holt-Winters recursions over a history.</summary>
    /// <param name="y">The history (at least two seasons long).</param>
    /// <param name="p">The smoothing parameters.</param>
    /// <returns>The final state.</returns>
    public static HoltWinters Fit(double[] y, HwParams p)
    {
        // classical initialization: level = first week's mean, trend = mean week-over-week change.
        double level = y[..M].Average();
        double trend = (y[M..(2 * M)].Average() - level) / M;
        var season = new double[M];
        for (int i = 0; i < M; i++)
            season[i] = p.Multiplicative ? y[i] / level : y[i] - level;
        for (int t = M; t < y.Length; t++)
        {
            double s = season[t % M];
            double prevLevel = level;
            level = p.Multiplicative
                ? p.Alpha * (y[t] / s) + (1 - p.Alpha) * (prevLevel + p.Phi * trend)
                : p.Alpha * (y[t] - s) + (1 - p.Alpha) * (prevLevel + p.Phi * trend);
            trend = p.Beta * (level - prevLevel) + (1 - p.Beta) * p.Phi * trend;
            season[t % M] = p.Multiplicative
                ? p.Gamma * (y[t] / level) + (1 - p.Gamma) * s
                : p.Gamma * (y[t] - level) + (1 - p.Gamma) * s;
        }
        var aligned = Enumerable.Range(0, M).Select(h => season[(y.Length + h) % M]).ToArray();
        return new HoltWinters(p, level, trend, aligned);
    }

    /// <summary>Forecasts the next days from the final state (damped trend: Σ φ^k).</summary>
    /// <param name="horizon">Number of days ahead.</param>
    /// <returns>The point forecasts.</returns>
    public double[] Forecast(int horizon)
    {
        var f = new double[horizon];
        double damp = 0, phiK = 1;
        for (int h = 0; h < horizon; h++)
        {
            phiK *= P.Phi;
            damp += phiK;
            double baseLine = Level + damp * Trend;
            f[h] = P.Multiplicative ? baseLine * Season[h % M] : baseLine + Season[h % M];
        }
        return f;
    }
}

/// <summary>The synthetic store's sales history.</summary>
static class SalesHistory
{
    /// <summary>Generates daily unit sales; the same seed gives the same history.</summary>
    /// <param name="days">History length.</param>
    /// <param name="seed">Data seed.</param>
    /// <returns>Daily sales.</returns>
    public static double[] Generate(int days, int seed)
    {
        var rng = new Random(seed);   // data generation only — samplers never use System.Random
        double[] weekly = { 0.85, 0.9, 0.92, 0.97, 1.12, 1.35, 0.89 };   // Mon..Sun shape
        var y = new double[days];
        for (int d = 0; d < days; d++)
        {
            double trend = 400 + 0.25 * d;
            double yearly = 1 + 0.12 * Math.Sin(2 * Math.PI * (d - 300) / 365.25);
            double promo = rng.NextDouble() < 0.04 ? 1.4 : 1.0;          // occasional promotion days
            y[d] = trend * yearly * weekly[d % 7] * promo * (1 + 0.05 * (rng.NextDouble() - 0.5));
        }
        return y;
    }

    /// <summary>The seasonal-naive benchmark: "same weekday last week", over the same five folds.</summary>
    /// <param name="y">The history.</param>
    /// <returns>Mean sMAPE (%) of the benchmark.</returns>
    public static double SeasonalNaiveSmape(double[] y)
    {
        var scores = new List<double>();
        for (int fold = 0; fold < 5; fold++)
        {
            int origin = y.Length - 28 * (5 - fold);
            var lastWeek = y[(origin - 7)..origin];
            var f = Enumerable.Range(0, 28).Select(h => lastWeek[h % 7]).ToArray();
            scores.Add(100.0 * y[origin..(origin + 28)].Zip(f, (a, b) => Math.Abs(a - b) / ((Math.Abs(a) + Math.Abs(b)) / 2)).Average());
        }
        return scores.Average();
    }
}
