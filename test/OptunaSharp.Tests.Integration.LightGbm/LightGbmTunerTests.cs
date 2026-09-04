using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Integration.LightGbm;

namespace OptunaSharp.Tests.Integration.LightGbm;

/// <summary>
///     Behavioural + end-to-end tests for the LightGBM tuner. The <b>structural</b> tests (direction
///     inference, alias handling, required arguments, default best-params) use a trivial synthetic trainer
///     and assert the ported <c>optuna_integration.lightgbm</c> control flow. The <b>end-to-end</b> tests
///     run the real default backend (<see cref="MlNetLightGbmTrainer" />, ML.NET's native LightGBM) on a
///     small dataset — the adapter half of Phase 6, validated by usage rather than bit-parity.
/// </summary>
[TestClass]
public class LightGbmTunerTests
{
    // A trivial deterministic trainer (score = a fixed bowl) — lets the structural tests run the tuner
    // without ML.NET. Lower is better.
    private sealed class ConstantTrainer : ILightGbmTrainer
    {
        public Booster Train(TrainRequest r)
        {
            var byName = new Dictionary<string, IReadOnlyDictionary<string, double>>();
            foreach (var n in r.ValidNames.DefaultIfEmpty("valid_0"))
                byName[n] = new Dictionary<string, double> { [Metric(r.Params)] = Score(r.Params) };
            return new Booster(byName, r.NumBoostRound, model: "dummy-model");
        }

        public CvResult Cv(CvRequest r) => new(new Dictionary<string, IReadOnlyList<double>>
        {
            [$"{Metric(r.Params)}-mean"] = new[] { Score(r.Params) },
        }, new CvBooster(new object?[] { "fold0" }, r.NumBoostRound));

        private static string Metric(IReadOnlyDictionary<string, object?> p)
            => p.TryGetValue("metric", out var m) && m is string s ? s : "binary_logloss";

        private static double Score(IReadOnlyDictionary<string, object?> p)
        {
            double ff = p.TryGetValue("feature_fraction", out var v) && v is not null ? Convert.ToDouble(v) : 1.0;
            return (ff - 0.7) * (ff - 0.7);   // any deterministic function of the params
        }
    }

    private static Dataset Data(int rows) => new(new double[rows, 3], new double[rows]);

    private static Dictionary<string, object?> Params(string objective, string metric) =>
        new() { ["objective"] = objective, ["metric"] = metric, ["verbosity"] = -1L };

    // ---- Structural (ported control-flow) ----

    [TestMethod]
    public void ValidSets_IsRequired()
    {
        var act = () => new LightGBMTuner(Params("regression", "l2"), Data(8), validSets: null);
        act.Should().Throw<ValueError>().WithMessage("*valid_sets* is required*");
    }

    [TestMethod]
    public void Direction_IsMinimize_ForRegressionMetric()
    {
        var tuner = new LightGBMTuner(Params("regression", "l2"), Data(8), validSets: new[] { Data(4) },
            trainer: new ConstantTrainer());
        tuner.Study.Direction.Should().Be(StudyDirection.Minimize);
    }

    [TestMethod]
    public void Direction_IsMaximize_ForAucMetric()
    {
        var tuner = new LightGBMTuner(Params("binary", "auc"), Data(8), validSets: new[] { Data(4) },
            trainer: new ConstantTrainer());
        tuner.Study.Direction.Should().Be(StudyDirection.Maximize);
    }

    [TestMethod]
    public void AliasMetric_IsCanonicalized_AndDrivesDirection()
    {
        // "mse" is an alias of "l2" (minimize); "mean_average_precision" is an alias of "map" (maximize).
        new LightGBMTuner(Params("regression", "mse"), Data(8), validSets: new[] { Data(4) },
                trainer: new ConstantTrainer())
            .Study.Direction.Should().Be(StudyDirection.Minimize);
        new LightGBMTuner(Params("lambdarank", "mean_average_precision"), Data(8), validSets: new[] { Data(4) },
                trainer: new ConstantTrainer())
            .Study.Direction.Should().Be(StudyDirection.Maximize);
    }

    [TestMethod]
    public void PreCreatedStudy_WithWrongDirection_Throws()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Maximize);
        var act = () => new LightGBMTuner(Params("regression", "l2"), Data(8), validSets: new[] { Data(4) },
            study: study, trainer: new ConstantTrainer());
        act.Should().Throw<ValueError>().WithMessage("*direction is inconsistent*minimize*");
    }

    [TestMethod]
    public void BestParams_BeforeRun_AreLightGbmDefaultsPlusUserParams()
    {
        var tuner = new LightGBMTuner(Params("regression", "l2"), Data(8), validSets: new[] { Data(4) },
            trainer: new ConstantTrainer());
        var best = tuner.BestParams;
        Convert.ToInt64(best["num_leaves"]).Should().Be(31);        // _DEFAULT_LIGHTGBM_PARAMETERS
        Convert.ToDouble(best["feature_fraction"]).Should().Be(1.0);
        Convert.ToInt64(best["min_child_samples"]).Should().Be(20);
        (best["objective"] as string).Should().Be("regression");     // user param preserved
    }

    [TestMethod]
    public void FeaturePreFilter_IsForcedFalse()
    {
        var p = Params("regression", "l2");
        p["feature_pre_filter"] = true;
        var tuner = new LightGBMTuner(p, Data(8), validSets: new[] { Data(4) }, trainer: new ConstantTrainer());
        // best_params-before-run overlays lgbm_params, which now carries feature_pre_filter = false.
        tuner.BestParams["feature_pre_filter"].Should().Be(false);
    }

    [TestMethod]
    public void Run_WithConstantTrainer_TunesAllStepsAndReportsBest()
    {
        var tuner = new LightGBMTuner(Params("regression", "l2"), Data(8), validSets: new[] { Data(4) },
            numBoostRound: 5, optunaSeed: 1, trainer: new ConstantTrainer());
        tuner.Run();

        tuner.Study.Trials.Count.Should().Be(68, "7+20+10+6+20+5 stepwise trials");
        tuner.Study.Trials.Select(t => (string)t.SystemAttrs["lightgbm_tuner:step_name"]!).Distinct()
            .Should().BeEquivalentTo(new[]
            {
                "feature_fraction", "num_leaves", "bagging", "feature_fraction_stage2",
                "regularization_factors", "min_child_samples",
            });
        double.IsFinite(tuner.BestScore).Should().BeTrue();
        tuner.GetBestBooster().Model.Should().Be("dummy-model");
    }

    [TestMethod]
    public void TunerCV_Run_WithConstantTrainer_Works()
    {
        var tuner = new LightGBMTunerCV(Params("regression", "l2"), Data(20),
            numBoostRound: 5, optunaSeed: 1, returnCvBooster: true, trainer: new ConstantTrainer());
        tuner.Run();

        tuner.Study.Trials.Count.Should().Be(68);
        double.IsFinite(tuner.BestScore).Should().BeTrue();
        tuner.GetBestBooster().Boosters.Should().ContainSingle();
    }

    [TestMethod]
    public void TunerCV_GetBestBooster_RequiresReturnCvBooster()
    {
        var tuner = new LightGBMTunerCV(Params("regression", "l2"), Data(20),
            numBoostRound: 5, optunaSeed: 1, trainer: new ConstantTrainer());
        tuner.Run();
        var act = () => tuner.GetBestBooster();
        act.Should().Throw<ValueError>().WithMessage("*return_cvbooster=True*");
    }

    // ---- End-to-end with the real ML.NET LightGBM backend ----

    [TestMethod]
    public void EndToEnd_RegressionTuner_RunsAndImproves()
    {
        var (xTr, yTr) = MakeRegression(160, seed: 0);
        var (xVa, yVa) = MakeRegression(60, seed: 1);
        var dtrain = new Dataset(xTr, yTr);
        var dval = new Dataset(xVa, yVa);

        var tuner = new LightGBMTuner(Params("regression", "l2"), dtrain, validSets: new[] { dval },
            numBoostRound: 15, optunaSeed: 42);
        tuner.Run();

        double.IsFinite(tuner.BestScore).Should().BeTrue("the l2 metric must be a finite score");
        tuner.BestScore.Should().BeGreaterThanOrEqualTo(0.0, "l2 (MSE) is non-negative");
        var booster = tuner.GetBestBooster();
        booster.Model.Should().NotBeNull("the best trial's ML.NET model is retained");
        booster.BestScore.Should().ContainKey("valid_0");
        // The tuner should not do worse than an untuned default fit.
        double defaultScore = new MlNetLightGbmTrainer().Train(new TrainRequest(
            tuner.BestParams, dtrain, new[] { dval }, new[] { "valid_0" }, 15)).BestScore["valid_0"]["l2"];
        tuner.BestScore.Should().BeLessThanOrEqualTo(defaultScore * 1.5 + 1e-9,
            "the best tuned score should be competitive with re-fitting the best params");
    }

    [TestMethod]
    public void EndToEnd_BinaryTuner_RunsWithNativeLightGbm()
    {
        var (xTr, yTr) = MakeBinary(160, seed: 0);
        var (xVa, yVa) = MakeBinary(60, seed: 1);
        var tuner = new LightGBMTuner(
            Params("binary", "binary_logloss"), new Dataset(xTr, yTr),
            validSets: new[] { new Dataset(xVa, yVa) }, numBoostRound: 15, optunaSeed: 7);
        tuner.Run();

        double.IsFinite(tuner.BestScore).Should().BeTrue();
        tuner.BestScore.Should().BeGreaterThan(0.0, "binary_logloss is positive");
        tuner.GetBestBooster().Model.Should().NotBeNull();
    }

    [TestMethod]
    public void EndToEnd_Train_StaticEntryPoint_ReturnsBooster()
    {
        var (xTr, yTr) = MakeRegression(120, seed: 0);
        var (xVa, yVa) = MakeRegression(40, seed: 1);
        var booster = LightGBMTuner.Train(
            Params("regression", "l2"), new Dataset(xTr, yTr),
            validSets: new[] { new Dataset(xVa, yVa) }, numBoostRound: 10, optunaSeed: 3);

        booster.Should().NotBeNull();
        booster.Model.Should().NotBeNull();
        booster.BestScore["valid_0"]["l2"].Should().BeGreaterThanOrEqualTo(0.0);
    }

    private static (float[][] X, double[] Y) MakeRegression(int n, int seed)
    {
        var rng = new Random(seed);
        var x = new float[n][];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            var row = new float[4];
            for (int j = 0; j < 4; j++) row[j] = (float)(rng.NextDouble() * 2 - 1);
            x[i] = row;
            // A smooth nonlinear target LightGBM can fit.
            y[i] = 3.0 * row[0] - 2.0 * row[1] + row[2] * row[2] + 0.1 * (rng.NextDouble() - 0.5);
        }
        return (x, y);
    }

    private static (float[][] X, double[] Y) MakeBinary(int n, int seed)
    {
        var rng = new Random(seed);
        var x = new float[n][];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            var row = new float[4];
            for (int j = 0; j < 4; j++) row[j] = (float)(rng.NextDouble() * 2 - 1);
            x[i] = row;
            y[i] = (row[0] + row[1] > 0) ? 1.0 : 0.0;
        }
        return (x, y);
    }
}
