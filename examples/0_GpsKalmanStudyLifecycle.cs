#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:project ../src/OptunaSharp.Storage.Sqlite/OptunaSharp.Storage.Sqlite.csproj
#:property PublishAot=false

// GPS tracker tuning, with a study's whole lifecycle — prototype a Kalman filter's noise tuning in memory,
// promote the study to a shared SQLite database with `copy_study`, keep optimizing it there with a better
// sampler, tag it with metadata, list every study in the database, and delete the scratch copy: the path a
// notebook experiment takes on its way to a team's experiment store.
//
// source: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.copy_study.html
//         https://optuna.readthedocs.io/en/stable/reference/generated/optuna.get_all_study_summaries.html
//         https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.set_metric_names
//         https://en.wikipedia.org/wiki/Kalman_filter
//
// Real objective: a recorded (deterministically simulated) 10-minute drive at 1 Hz — accelerations, turns,
// a stop at a light — seen through a GPS receiver with 4 m noise and 3 % multipath outliers that jump 30–80
// m. A constant-velocity Kalman filter per axis smooths it; the knobs are the process noise (how much the
// car may accelerate), the measurement noise, and an innovation gate that rejects outliers. The score is
// the position RMSE against the true track, in metres.
//
//   Run:  dotnet run examples/0_GpsKalmanStudyLifecycle.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var drive = Drive.Record(seconds: 600, seed: 5);
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "raw GPS RMSE: {0:F2} m", drive.RawRmse()));

// 1. the notebook prototype: an in-memory study and a quick random scan.
var scratch = new InMemoryStorage();
var proto = Optuna.CreateStudy(storage: scratch, sampler: new RandomSampler(seed: 1), studyName: "kf-prototype");
proto.SetMetricNames(new[] { "rmse_m" });
proto.Optimize(t => Objective(t, drive), nTrials: 20);
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "prototype: {0} trials, best {1:F3} m", proto.Trials.Count, proto.BestValue));

// 2. promote it: copy every trial, attribute and the metric names into the team's SQLite database.
string dbPath = Path.Combine(Path.GetTempPath(), "optunasharp-team-experiments.db");
File.Delete(dbPath);
using var team = new RDBStorage("sqlite:///" + dbPath.Replace('\\', '/'));
Optuna.CopyStudy(fromStudyName: "kf-prototype", fromStorage: scratch, toStorage: team, toStudyName: "gps-kf-v1");

// 3. continue in the shared database with TPE, which now learns from the copied random trials too.
var study = Optuna.LoadStudy("gps-kf-v1", team, sampler: new TPESampler(seed: 2, nStartupTrials: 5));
study.SetUserAttr("dataset", "drive-2026-11-08-1hz");
study.SetUserAttr("owner", "tracking-team");
study.Optimize(t => Objective(t, drive), nTrials: 60);
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "team study: {0} trials (20 copied + 60 new), best {1:F3} m: {2}",
    study.Trials.Count, study.BestValue, string.Join(", ", study.BestParams.Select(kv => string.Format(CultureInfo.InvariantCulture, "{0}={1:G4}", kv.Key, kv.Value)))));

// 4. a throwaway study in the same database, then the inventory, then clean-up.
Optuna.CreateStudy(storage: team, studyName: "gps-kf-scratch").Optimize(t => Objective(t, drive), nTrials: 3);
Console.WriteLine();
Console.WriteLine("studies in the database:");
foreach (var s in Optuna.GetAllStudySummaries(team))
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "  {0,-16} trials={1,3}  best={2,8:F3}  user_attrs={3}",
        s.StudyName, s.NTrials, s.BestTrial?.Value, string.Join(";", s.UserAttrs.Select(kv => $"{kv.Key}={kv.Value}"))));
Optuna.DeleteStudy("gps-kf-scratch", team);
Console.WriteLine("after delete_study: " + string.Join(", ", Optuna.GetAllStudyNames(team)));

// 5. what an analyst pulls out: the dataframe uses the metric name for the value column.
var df = study.TrialsDataframe();
Console.WriteLine("trials_dataframe columns: " + string.Join(", ", df.Columns));

/// <summary>Filters the drive with the trial's noise settings and scores it against the true track.</summary>
/// <param name="t">The trial proposing process noise, measurement noise and the outlier gate.</param>
/// <param name="d">The recorded drive.</param>
/// <returns>The filtered position RMSE in metres.</returns>
static double Objective(Trial t, Drive d)
{
    double q = t.SuggestFloat("accel_noise", 0.01, 20, log: true);      // m²/s³: how hard the car may accelerate
    double r = t.SuggestFloat("gps_sigma", 0.5, 30, log: true);        // m: assumed receiver noise
    double gate = t.SuggestFloat("gate_chi2", 2, 50, log: true);       // innovation gate (χ², 1 dof)
    return d.FilteredRmse(q, r, gate);
}

/// <summary>A simulated drive: the true track and what the GPS receiver reported.</summary>
/// <param name="TrueX">True east positions (m), one per second.</param>
/// <param name="TrueY">True north positions (m).</param>
/// <param name="GpsX">Reported east positions.</param>
/// <param name="GpsY">Reported north positions.</param>
sealed record Drive(double[] TrueX, double[] TrueY, double[] GpsX, double[] GpsY)
{
    /// <summary>Simulates the drive; the same seed gives the same recording.</summary>
    /// <param name="seconds">Duration at 1 Hz.</param>
    /// <param name="seed">Recording seed.</param>
    /// <returns>The drive.</returns>
    public static Drive Record(int seconds, int seed)
    {
        var rng = new Random(seed);   // data generation only — samplers never use System.Random
        var tx = new double[seconds]; var ty = new double[seconds];
        var gx = new double[seconds]; var gy = new double[seconds];
        double x = 0, y = 0, heading = 0, speed = 0;
        for (int s = 0; s < seconds; s++)
        {
            double target = s is > 200 and < 230 ? 0 : 14;               // a red light at 200–230 s
            speed += Math.Clamp(target - speed, -3, 2) * 0.5 + 0.3 * Normal(rng);
            speed = Math.Max(0, speed);
            if (s % 90 > 75) heading += 0.1;                              // a turn every 90 s
            x += speed * Math.Cos(heading);
            y += speed * Math.Sin(heading);
            (tx[s], ty[s]) = (x, y);
            bool multipath = rng.NextDouble() < 0.03;
            double jump = multipath ? 30 + 50 * rng.NextDouble() : 0;
            double dir = 2 * Math.PI * rng.NextDouble();
            gx[s] = x + 4 * Normal(rng) + jump * Math.Cos(dir);
            gy[s] = y + 4 * Normal(rng) + jump * Math.Sin(dir);
        }
        return new Drive(tx, ty, gx, gy);
    }

    /// <summary>RMSE of the raw fixes.</summary>
    /// <returns>Metres.</returns>
    public double RawRmse() => Rmse(GpsX, GpsY);

    /// <summary>Runs one constant-velocity Kalman filter per axis and scores the result.</summary>
    /// <param name="q">Process (acceleration) noise spectral density.</param>
    /// <param name="sigma">Measurement noise standard deviation.</param>
    /// <param name="gate">Reject a fix whose normalized squared innovation exceeds this.</param>
    /// <returns>Filtered RMSE in metres.</returns>
    public double FilteredRmse(double q, double sigma, double gate) => Rmse(Filter(GpsX, q, sigma, gate), Filter(GpsY, q, sigma, gate));

    /// <summary>One axis of the filter: state (position, velocity), dt = 1 s.</summary>
    /// <param name="z">The axis's fixes.</param>
    /// <param name="q">Process noise spectral density.</param>
    /// <param name="sigma">Measurement noise standard deviation.</param>
    /// <param name="gate">Innovation gate.</param>
    /// <returns>The filtered positions.</returns>
    private static double[] Filter(double[] z, double q, double sigma, double gate)
    {
        double p = z[0], v = 0, p00 = 100, p01 = 0, p11 = 100, r = sigma * sigma;
        var o = new double[z.Length];
        o[0] = p;
        for (int k = 1; k < z.Length; k++)
        {
            // predict: x = F x, P = F P Fᵀ + Q (white-noise acceleration, dt = 1).
            p += v;
            double n00 = p00 + 2 * p01 + p11 + q / 3, n01 = p01 + p11 + q / 2, n11 = p11 + q;
            (p00, p01, p11) = (n00, n01, n11);
            // update with the fix unless the gate calls it an outlier.
            double innov = z[k] - p, s = p00 + r;
            if (innov * innov / s <= gate)
            {
                double k0 = p00 / s, k1 = p01 / s;
                p += k0 * innov;
                v += k1 * innov;
                (p00, p01, p11) = ((1 - k0) * p00, (1 - k0) * p01, p11 - k1 * p01);
            }
            o[k] = p;
        }
        return o;
    }

    /// <summary>Position RMSE of an estimated track against the true one.</summary>
    /// <param name="ex">Estimated east positions.</param>
    /// <param name="ey">Estimated north positions.</param>
    /// <returns>Metres.</returns>
    private double Rmse(double[] ex, double[] ey) =>
        Math.Sqrt(Enumerable.Range(0, TrueX.Length).Average(i => Math.Pow(ex[i] - TrueX[i], 2) + Math.Pow(ey[i] - TrueY[i], 2)));

    /// <summary>A standard normal draw (Box–Muller).</summary>
    /// <param name="rng">The random stream.</param>
    /// <returns>A N(0, 1) variate.</returns>
    private static double Normal(Random rng) => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
}
