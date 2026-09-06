using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Integration;
using OptunaSharp.Integration.Sklearn;

namespace OptunaSharp.Tests.Integration.Sklearn;

/// <summary>
///     Byte-exact parity against a committed oracle captured from real scikit-learn 1.8.0 +
///     optuna_integration (oracle/gen_sklearn_oracle.py) — the OptunaSharp.Tests.Oracle model (no Python
///     at test time), as the LightGBM tuner does.
///
///     <list type="bullet">
///       <item><b>Components:</b> <see cref="RandomForestRegressor" /> predictions + <c>r2</c>,
///         <see cref="KFold" /> folds, and <c>cross_validate</c> per-fold/mean scores — bit-for-bit vs
///         scikit-learn across a sweep of configs (default / sqrt / fractional <c>max_features</c>,
///         <c>bootstrap=False</c>, <c>min_samples_*</c>, unlimited depth, <c>neg_mean_squared_error</c>).</item>
///       <item><b>End-to-end:</b> the Core <see cref="OptunaSearchCV" /> over the native
///         <see cref="SklearnEstimator" /> reproduces Python's <c>optuna_integration.OptunaSearchCV</c>
///         trial-for-trial — sampled params, per-split test scores, <c>mean_test_score</c>, and the best
///         result — because integer <c>param_distributions</c> make TPE's suggestions bit-exact and the
///         forest/CV/scorer are byte-exact.</item>
///     </list>
///
///     <para>Value parity is host-pinned to x64 (the float32 forest accumulation + the platform
///     <c>np.int_</c> bootstrap width); the tests self-skip (Inconclusive) elsewhere, matching the fANOVA
///     parity suite.</para>
/// </summary>
[TestClass]
public class SklearnOracleParityTests
{
    private static JsonElement _oracle;

    [ClassInitialize]
    public static void Load(TestContext _)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "oracle", "sklearn_oracle.json");
        _oracle = JsonDocument.Parse(File.ReadAllText(path)).RootElement.Clone();
    }

    private static void RequireBitExactHost()
    {
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            Assert.Inconclusive(
                "scikit-learn value parity is host-pinned to x64 (float32 forest accumulation + platform " +
                "np.int_ bootstrap width); the committed oracle was captured on x64.");
    }

    [TestMethod]
    public void Components_MatchScikitLearnByteForByte()
    {
        RequireBitExactHost();

        foreach (var sc in _oracle.GetProperty("component").EnumerateArray())
        {
            string name = sc.GetProperty("name").GetString()!;
            var (x, y) = ReadXy(sc);

            var config = ReadConfig(sc.GetProperty("config"));
            var rf = (RandomForestRegressor)new RandomForestRegressor().WithParams(config);
            rf.Fit(x, y);

            // predictions + estimator.score (r2), bit-for-bit.
            var preds = rf.Predict(x);
            var oraclePreds = ReadDoubles(sc.GetProperty("preds"));
            preds.Should().Equal(oraclePreds, $"predictions must match scikit-learn ({name})");
            rf.Score(x, y).Should().Be(sc.GetProperty("r2").GetDouble(), $"r2 must match scikit-learn ({name})");

            // KFold folds (integer membership).
            var folds = new KFold(5).Split(y.Length).Select(f => f.Test).ToList();
            var oracleFolds = sc.GetProperty("folds").EnumerateArray()
                .Select(a => a.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToList();
            for (int i = 0; i < folds.Count; i++)
                folds[i].Should().Equal(oracleFolds[i], $"KFold fold {i} must match ({name})");

            // cross_validate test/train/mean, bit-for-bit.
            var scorer = Scorers.CheckScoring(sc.GetProperty("scoring").GetString());
            var cv = CrossValidation.CrossValidate(rf, x, y, new KFold(5), scorer, returnTrainScore: true, double.NaN);
            var cvTest = cv.Select(f => f.TestScore).ToArray();
            var cvTrain = cv.Select(f => f.TrainScore!.Value).ToArray();
            cvTest.Should().Equal(ReadDoubles(sc.GetProperty("cv_test")), $"cross_validate test_score ({name})");
            cvTrain.Should().Equal(ReadDoubles(sc.GetProperty("cv_train")), $"cross_validate train_score ({name})");
            cvTest.Average().Should().Be(sc.GetProperty("cv_mean").GetDouble(), $"cross_validate mean ({name})");
        }
    }

    [TestMethod]
    public void OptunaSearchCV_ReproducesOptunaIntegrationTrialForTrial()
    {
        RequireBitExactHost();

        foreach (var sc in _oracle.GetProperty("search").EnumerateArray())
        {
            string name = sc.GetProperty("name").GetString()!;
            var (x, y) = ReadXy(sc);

            var dists = new Dictionary<string, BaseDistribution>();
            foreach (var d in sc.GetProperty("dists").EnumerateArray())
                dists[d.GetProperty("name").GetString()!] =
                    new IntDistribution(d.GetProperty("low").GetInt64(), d.GetProperty("high").GetInt64());

            int cv = sc.GetProperty("cv").GetInt32();
            var estimator = new SklearnEstimator(new RandomForestRegressor(randomState: 0), x, y, new KFold(cv));
            var search = new OptunaSearchCV(
                estimator, dists,
                nTrials: sc.GetProperty("n_trials").GetInt32(),
                randomState: sc.GetProperty("random_state").GetInt32(),
                refit: true, returnTrainScore: true).Fit();

            var oracleTrials = sc.GetProperty("trials").EnumerateArray().ToList();
            search.NTrials.Should().Be(oracleTrials.Count, $"trial count ({name})");
            search.NSplits.Should().Be(sc.GetProperty("n_splits").GetInt32(), $"n_splits ({name})");

            for (int i = 0; i < oracleTrials.Count; i++)
            {
                var ot = oracleTrials[i];
                var trial = search.Trials[i];

                // sampled params (bit-exact TPE for integer distributions).
                foreach (var p in ot.GetProperty("params").EnumerateObject())
                    Convert.ToInt64(trial.Params[p.Name]).Should()
                        .Be(p.Value.GetInt64(), $"trial {i} param {p.Name} ({name})");

                // per-split and mean test score, bit-for-bit.
                var oracleSplits = ReadDoubles(ot.GetProperty("splits"));
                for (int s = 0; s < oracleSplits.Length; s++)
                    ((double)trial.UserAttrs[$"split{s}_test_score"]!).Should()
                        .Be(oracleSplits[s], $"trial {i} split{s}_test_score ({name})");
                ((double)trial.UserAttrs["mean_test_score"]!).Should()
                    .Be(ot.GetProperty("mean_test_score").GetDouble(), $"trial {i} mean_test_score ({name})");
            }

            search.BestScore.Should().Be(sc.GetProperty("best_score").GetDouble(), $"best_score ({name})");
            foreach (var p in sc.GetProperty("best_params").EnumerateObject())
                Convert.ToInt64(search.BestParams[p.Name]).Should()
                    .Be(p.Value.GetInt64(), $"best_params {p.Name} ({name})");
        }
    }

    // ---- oracle readers -----------------------------------------------------------------------

    private static (double[,] X, double[] Y) ReadXy(JsonElement sc)
    {
        var xrows = sc.GetProperty("X");
        int n = xrows.GetArrayLength();
        int d = xrows[0].GetArrayLength();
        var x = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                x[i, j] = xrows[i][j].GetDouble();
        var y = ReadDoubles(sc.GetProperty("y"));
        return (x, y);
    }

    private static double[] ReadDoubles(JsonElement arr)
        => arr.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    private static Dictionary<string, object?> ReadConfig(JsonElement config)
    {
        var dict = new Dictionary<string, object?>();
        foreach (var p in config.EnumerateObject())
            dict[p.Name] = ClrValue(p.Value);
        return dict;
    }

    // JSON -> CLR: integer number -> long (an integral count), real number -> double (a fraction),
    // matching how RandomForestRegressor.SetParam dispatches on Integral vs Real.
    private static object? ClrValue(JsonElement v) => v.ValueKind switch
    {
        JsonValueKind.String => v.GetString(),
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        JsonValueKind.Null => null,
        // NB: cast to object so the ternary's common type stays object — otherwise long/double unify to
        // double, silently turning an integer count (e.g. min_samples_split=5) into a fraction (5.0).
        JsonValueKind.Number => IsIntegerText(v.GetRawText()) ? (object)v.GetInt64() : v.GetDouble(),
        _ => throw new InvalidOperationException($"unexpected config value kind {v.ValueKind}"),
    };

    private static bool IsIntegerText(string raw)
        => !raw.Contains('.') && !raw.Contains('e') && !raw.Contains('E');
}
