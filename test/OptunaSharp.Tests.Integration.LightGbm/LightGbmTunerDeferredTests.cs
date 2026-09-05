using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using Microsoft.ML;
using NumSharp;
using OptunaSharp.Integration.LightGbm;

namespace OptunaSharp.Tests.Integration.LightGbm;

/// <summary>
///     Coverage for the features that finished Phase 6.1 (previously deferred from the first cut):
///     <b>ndcg / map ranking objectives</b> and <b>multiclass</b> in the default ML.NET trainer, and
///     <b>model_dir cross-process booster sharing</b>. The metric math (ndcg@k, map@k, multiclass
///     log-loss) is tested deterministically on hand-computed inputs (no native LightGBM); the tuner
///     end-to-end tests run the real ML.NET backend; the persistence tests exercise the save → resume →
///     reload round-trip that backs <c>get_best_booster</c> across processes.
/// </summary>
[TestClass]
public class LightGbmTunerDeferredTests
{
    // ---- Ranking metric math (deterministic, hand-computed — no native LightGBM) ----

    [TestMethod]
    public void RankingMetrics_PerfectOrder_NdcgAndMapAreOne()
    {
        // One group, predicted order equals the ideal order.
        var rel = new[] { 2.0, 1.0, 0.0 };
        var score = new[] { 3.0, 2.0, 1.0 };
        var m = MlNetLightGbmTrainer.RankingMetrics(rel, score, new[] { 3 }, Empty());

        m["ndcg@1"].Should().BeApproximately(1.0, 1e-12);
        m["ndcg@3"].Should().BeApproximately(1.0, 1e-12);
        m["map@3"].Should().BeApproximately(1.0, 1e-12);
    }

    [TestMethod]
    public void RankingMetrics_KnownNdcg()
    {
        // rel = [3,2,1,0]; scores rank them as order [1,2,0,3] -> rels seen = [2,1,3,0].
        var rel = new[] { 3.0, 2.0, 1.0, 0.0 };
        var score = new[] { 1.0, 4.0, 2.0, 0.0 };
        var m = MlNetLightGbmTrainer.RankingMetrics(rel, score, new[] { 4 }, Empty());

        // dcg@2 = (2^2-1)/log2(2) + (2^1-1)/log2(3) = 3 + 0.6309297535714574 = 3.6309297535714573
        // idcg@2 = (2^3-1)/log2(2) + (2^2-1)/log2(3) = 7 + 1.892789260714372 = 8.892789260714372
        m["ndcg@2"].Should().BeApproximately(3.6309297535714573 / 8.892789260714372, 1e-12);
    }

    [TestMethod]
    public void RankingMetrics_KnownMap()
    {
        // rel = [0,1,0,1] (relevant at ranks 2 and 4 in score order); scores keep index order.
        var rel = new[] { 0.0, 1.0, 0.0, 1.0 };
        var score = new[] { 4.0, 3.0, 2.0, 1.0 };
        var m = MlNetLightGbmTrainer.RankingMetrics(rel, score, new[] { 4 }, Empty());

        // map@4 = (1/2 + 2/4) / min(2,4) = 1.0 / 2 = 0.5 ; map@2 = (1/2) / min(2,2) = 0.25
        m["map@4"].Should().BeApproximately(0.5, 1e-12);
        m["map@2"].Should().BeApproximately(0.25, 1e-12);
    }

    [TestMethod]
    public void RankingMetrics_HonorsEvalAt()
    {
        var p = new Dictionary<string, object?> { ["eval_at"] = new List<object?> { 1L, 10L } };
        var m = MlNetLightGbmTrainer.RankingMetrics(new[] { 1.0, 0.0 }, new[] { 1.0, 0.0 }, new[] { 2 }, p);
        m.Keys.Should().Contain(new[] { "ndcg@1", "ndcg@10", "map@1", "map@10" });
        m.Keys.Should().NotContain("ndcg@3");
    }

    // ---- Multiclass metric math (deterministic) ----

    [TestMethod]
    public void MulticlassMetrics_PerfectProbabilities_ZeroLossZeroError()
    {
        var y = new[] { 0.0, 1.0, 2.0 };
        var probs = new[] { new[] { 1f, 0f, 0f }, new[] { 0f, 1f, 0f }, new[] { 0f, 0f, 1f } };
        var m = MlNetLightGbmTrainer.MulticlassMetrics(y, probs);
        m["multi_logloss"].Should().BeApproximately(0.0, 1e-12);
        m["multi_error"].Should().Be(0.0);
    }

    [TestMethod]
    public void MulticlassMetrics_KnownLoglossAndError()
    {
        // class 0 true, p=0.5 on it (argmax correct) -> loss -ln(0.5), error 0.
        MlNetLightGbmTrainer.MulticlassMetrics(new[] { 0.0 }, new[] { new[] { 0.5f, 0.3f, 0.2f } })["multi_logloss"]
            .Should().BeApproximately(-Math.Log(0.5), 1e-6);
        // class 0 true but argmax is class 1 -> error 1, loss -ln(0.2).
        var m = MlNetLightGbmTrainer.MulticlassMetrics(new[] { 0.0 }, new[] { new[] { 0.2f, 0.5f, 0.3f } });
        m["multi_error"].Should().Be(1.0);
        m["multi_logloss"].Should().BeApproximately(-Math.Log(0.2), 1e-6);
    }

    // ---- Dataset group validation ----

    [TestMethod]
    public void Dataset_Group_MustSumToRows()
    {
        var act = () => new Dataset(new double[6, 2], new double[6], group: new[] { 2, 2 });
        act.Should().Throw<ValueError>().WithMessage("*sum to the number of rows*");
    }

    // ---- Multiclass tuner end-to-end (native ML.NET) ----

    [TestMethod]
    public void EndToEnd_MulticlassTuner_RunsWithNativeLightGbm()
    {
        var (xTr, yTr) = MakeMulticlass(150, 3, seed: 0);
        var (xVa, yVa) = MakeMulticlass(60, 3, seed: 1);
        var p = new Dictionary<string, object?>
        {
            ["objective"] = "multiclass", ["metric"] = "multi_logloss", ["num_class"] = 3L, ["verbosity"] = -1L,
        };

        var tuner = new LightGBMTuner(p, new Dataset(xTr, yTr), validSets: new[] { new Dataset(xVa, yVa) },
            numBoostRound: 12, optunaSeed: 5);
        tuner.Study.Direction.Should().Be(StudyDirection.Minimize, "multi_logloss is minimized");
        tuner.Run();

        // The trial count is 68 minus however many stage-2 feature_fraction grid points fall outside
        // [0.4, 1.0] (fewer when the best feature_fraction sits on a grid boundary), so >= 65.
        tuner.Study.Trials.Count.Should().BeInRange(65, 68);
        AllStepsRan(tuner);
        double.IsFinite(tuner.BestScore).Should().BeTrue();
        tuner.BestScore.Should().BeGreaterThan(0.0, "multi_logloss is positive");
        var booster = tuner.GetBestBooster();
        booster.Model.Should().NotBeNull();
        booster.BestScore["valid_0"].Should().ContainKey("multi_logloss");
    }

    // ---- Ranking tuner end-to-end (native ML.NET) ----

    [TestMethod]
    public void EndToEnd_RankingTuner_RunsWithNativeLightGbm()
    {
        // A realistic ranking size; degenerate stumps (from the regularization / min_child_samples steps)
        // are handled by the trainer's constant-predictor fallback, so the full 68-trial run completes.
        var (xTr, relTr, gTr) = MakeRanking(60, 10, seed: 0);
        var (xVa, relVa, gVa) = MakeRanking(20, 10, seed: 1);
        var p = new Dictionary<string, object?>
        {
            ["objective"] = "lambdarank", ["metric"] = "ndcg", ["verbosity"] = -1L,
        };

        var tuner = new LightGBMTuner(
            p, new Dataset(xTr, relTr, group: gTr),
            validSets: new[] { new Dataset(xVa, relVa, group: gVa) }, numBoostRound: 12, optunaSeed: 9);
        tuner.Study.Direction.Should().Be(StudyDirection.Maximize, "ndcg is maximized");
        tuner.Run();

        tuner.Study.Trials.Count.Should().BeInRange(65, 68);
        AllStepsRan(tuner);
        double.IsFinite(tuner.BestScore).Should().BeTrue();
        tuner.BestScore.Should().BeInRange(0.0, 1.0 + 1e-9, "ndcg is a normalized score");
        var booster = tuner.GetBestBooster();
        booster.Model.Should().NotBeNull();
        booster.BestScore["valid_0"].Should().ContainKey("ndcg@1");
    }

    [TestMethod]
    public void EndToEnd_RankingTunerCV_RunsWithGroupFolds()
    {
        // 90 groups over nFold=3 -> each fold trains on 60 groups (600 rows), enough for min_child_samples=100.
        var (x, rel, g) = MakeRanking(90, 10, seed: 3);
        var p = new Dictionary<string, object?>
        {
            ["objective"] = "lambdarank", ["metric"] = "ndcg", ["verbosity"] = -1L,
        };

        var tuner = new LightGBMTunerCV(p, new Dataset(x, rel, group: g),
            numBoostRound: 10, nFold: 3, optunaSeed: 2);
        tuner.Run();

        tuner.Study.Trials.Count.Should().BeInRange(65, 68);
        tuner.BestScore.Should().BeInRange(0.0, 1.0 + 1e-9);
    }

    [TestMethod]
    public void Ranking_DegenerateStump_DoesNotThrow_AndStillScores()
    {
        // Tiny ranking data + strong regularization + a large min_child_samples forces LightGBM to a
        // single-leaf stump that ML.NET cannot build; the trainer must fall back to constant predictions
        // (Booster.Model == null) rather than throwing, so the metric is still produced.
        var (x, rel, g) = MakeRanking(10, 6, seed: 0);
        var p = new Dictionary<string, object?>
        {
            ["objective"] = "lambdarank", ["metric"] = "ndcg",
            ["lambda_l1"] = 10.0, ["lambda_l2"] = 10.0, ["min_child_samples"] = 100L, ["verbosity"] = -1L,
        };

        var booster = new MlNetLightGbmTrainer().Train(new TrainRequest(
            p, new Dataset(x, rel, group: g),
            new[] { new Dataset(x, rel, group: g) }, new[] { "valid_0" }, 12));

        booster.BestScore["valid_0"].Should().ContainKey("ndcg@1");
        double.IsFinite(booster.BestScore["valid_0"]["ndcg@1"]).Should().BeTrue();
    }

    // ---- model_dir persistence + resume ----

    [TestMethod]
    public void ModelDir_SavesBoosters_AndResumeReloadsBest()
    {
        string dir = FreshDir();
        try
        {
            var (xTr, yTr) = MakeRegression(120, seed: 0);
            var (xVa, yVa) = MakeRegression(40, seed: 1);
            var dtrain = new Dataset(xTr, yTr);
            var dval = new Dataset(xVa, yVa);
            var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);

            // First "process": tune one step; every trial's booster is persisted to `dir`.
            var tunerA = new LightGBMTuner(RegParams(), dtrain, validSets: new[] { dval },
                numBoostRound: 10, optunaSeed: 1, study: study, modelDir: dir);
            tunerA.TuneFeatureFraction();

            Directory.GetFiles(dir, "*.meta.json").Should().NotBeEmpty("every trial's booster is saved");
            Directory.GetFiles(dir, "*.model.zip").Should().NotBeEmpty();

            // Second "process": a fresh tuner over the SAME study but with no in-memory booster — it must
            // reload the best booster from `dir`.
            var tunerB = new LightGBMTuner(RegParams(), dtrain, validSets: new[] { dval },
                numBoostRound: 10, optunaSeed: 1, study: study, modelDir: dir);
            var reloaded = tunerB.GetBestBooster();

            reloaded.Model.Should().NotBeNull("the best booster is reloaded from disk");
            reloaded.BestScore["valid_0"]["l2"].Should().Be(study.BestValue,
                "the persisted score is the best trial's objective value, round-tripped losslessly");

            // The reloaded model predicts identically to the in-memory best booster (same trained model).
            var inMemory = tunerA.GetBestBooster();
            PredictReg((ITransformer)reloaded.Model!, dval)
                .Should().Equal(PredictReg((ITransformer)inMemory.Model!, dval));
        }
        finally
        {
            TryDelete(dir);
        }
    }

    [TestMethod]
    public void ModelDir_BestBoosterMissingOnDisk_Throws()
    {
        string dirA = FreshDir();
        string dirB = FreshDir();   // empty — no boosters
        try
        {
            var (xTr, yTr) = MakeRegression(80, seed: 0);
            var dtrain = new Dataset(xTr, yTr);
            var dval = new Dataset(MakeRegression(30, seed: 1).X, MakeRegression(30, seed: 1).Y);
            var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);

            new LightGBMTuner(RegParams(), dtrain, validSets: new[] { dval }, numBoostRound: 8,
                optunaSeed: 1, study: study, modelDir: dirA).TuneFeatureFraction();

            // Same study (so trials exist) but a directory that has no boosters -> the "cannot be found" error.
            var tuner = new LightGBMTuner(RegParams(), dtrain, validSets: new[] { dval }, numBoostRound: 8,
                optunaSeed: 1, study: study, modelDir: dirB);
            var act = () => tuner.GetBestBooster();
            act.Should().Throw<ValueError>().WithMessage("*cannot be found in*");
        }
        finally
        {
            TryDelete(dirA);
            TryDelete(dirB);
        }
    }

    [TestMethod]
    public void ModelDir_CreatesDirectoryIfMissing()
    {
        string dir = Path.Combine(Path.GetTempPath(), "optunasharp_lgbm_" + Guid.NewGuid().ToString("N"));
        Directory.Exists(dir).Should().BeFalse();
        try
        {
            _ = new LightGBMTuner(RegParams(), new Dataset(MakeRegression(20, 0).X, MakeRegression(20, 0).Y),
                validSets: new[] { new Dataset(MakeRegression(10, 1).X, MakeRegression(10, 1).Y) },
                modelDir: dir);
            Directory.Exists(dir).Should().BeTrue("the constructor creates model_dir");
        }
        finally
        {
            TryDelete(dir);
        }
    }

    [TestMethod]
    public void ModelDir_WithTrainerLackingStore_ThrowsAtConstruction()
    {
        string dir = FreshDir();
        try
        {
            var act = () => new LightGBMTuner(RegParams(),
                new Dataset(new double[8, 2], new double[8]), validSets: new[] { new Dataset(new double[4, 2], new double[4]) },
                modelDir: dir, trainer: new NonStoreTrainer());
            act.Should().Throw<NotSupportedException>().WithMessage("*ILightGbmBoosterStore*");
        }
        finally
        {
            TryDelete(dir);
        }
    }

    [TestMethod]
    public void ModelDirCV_ResumeReloadsCvBooster()
    {
        string dir = FreshDir();
        try
        {
            var (x, y) = MakeRegression(90, seed: 0);
            var dtrain = new Dataset(x, y);
            var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);

            var tunerA = new LightGBMTunerCV(RegParams(), dtrain, numBoostRound: 8, nFold: 3,
                optunaSeed: 1, study: study, modelDir: dir, returnCvBooster: true);
            tunerA.TuneFeatureFraction();

            Directory.GetFiles(dir, "*.cv.json").Should().NotBeEmpty("CV boosters are persisted");

            var tunerB = new LightGBMTunerCV(RegParams(), dtrain, numBoostRound: 8, nFold: 3,
                optunaSeed: 1, study: study, modelDir: dir, returnCvBooster: true);
            var reloaded = tunerB.GetBestBooster();

            reloaded.Boosters.Count.Should().Be(3, "the 3 fold models are reloaded from disk");
            reloaded.BestIteration.Should().Be(8);
        }
        finally
        {
            TryDelete(dir);
        }
    }

    // ---- helpers ----

    private sealed class NonStoreTrainer : ILightGbmTrainer
    {
        public Booster Train(TrainRequest r) => new(
            new Dictionary<string, IReadOnlyDictionary<string, double>>
            {
                ["valid_0"] = new Dictionary<string, double> { ["l2"] = 0.0 },
            }, r.NumBoostRound);

        public CvResult Cv(CvRequest r) => new(new Dictionary<string, IReadOnlyList<double>>
        {
            ["l2-mean"] = new[] { 0.0 },
        });
    }

    private static void AllStepsRan(LightGbmBaseTuner tuner)
        => tuner.Study.Trials
            .Select(t => (string)t.SystemAttrs["lightgbm_tuner:step_name"]!).Distinct()
            .Should().BeEquivalentTo(new[]
            {
                "feature_fraction", "num_leaves", "bagging", "feature_fraction_stage2",
                "regularization_factors", "min_child_samples",
            });

    private static IReadOnlyDictionary<string, object?> Empty() => new Dictionary<string, object?>();

    private static Dictionary<string, object?> RegParams() =>
        new() { ["objective"] = "regression", ["metric"] = "l2", ["verbosity"] = -1L };

    private static double[] PredictReg(ITransformer model, Dataset set)
    {
        var ml = new MLContext();
        var view = model.Transform(LoadRegressionView(ml, set));
        return ml.Data.CreateEnumerable<RegOut>(view, reuseRowObject: false).Select(r => (double)r.Score).ToArray();
    }

    private static Microsoft.ML.IDataView LoadRegressionView(MLContext ml, Dataset set)
    {
        int d = set.NumFeatures;
        var rows = new RegRow[set.NumRows];
        var y = set.GetLabelOrZeros();
        for (int i = 0; i < set.NumRows; i++)
            rows[i] = new RegRow { Features = set.Data[i], Label = (float)y[i] };
        var schema = Microsoft.ML.Data.SchemaDefinition.Create(typeof(RegRow));
        schema[nameof(RegRow.Features)].ColumnType =
            new Microsoft.ML.Data.VectorDataViewType(Microsoft.ML.Data.NumberDataViewType.Single, d);
        return ml.Data.LoadFromEnumerable(rows, schema);
    }

    private sealed class RegRow
    {
        public float[] Features { get; set; } = Array.Empty<float>();
        public float Label { get; set; }
    }

    private sealed class RegOut
    {
        public float Score { get; set; }
    }

    private static string FreshDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), "optunasharp_lgbm_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void TryDelete(string dir)
    {
        try { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
        catch { /* best-effort temp cleanup */ }
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
            y[i] = 3.0 * row[0] - 2.0 * row[1] + row[2] * row[2] + 0.1 * (rng.NextDouble() - 0.5);
        }
        return (x, y);
    }

    private static (float[][] X, double[] Y) MakeMulticlass(int n, int numClass, int seed)
    {
        var rng = new Random(seed);
        var x = new float[n][];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            var row = new float[4];
            for (int j = 0; j < 4; j++) row[j] = (float)(rng.NextDouble() * 2 - 1);
            x[i] = row;
            double s = row[0] + row[1];
            int cls = s > 0.5 ? 2 : (s > -0.5 ? 1 : 0);
            y[i] = Math.Min(numClass - 1, cls);
        }
        return (x, y);
    }

    // Grouped ranking data: `numGroups` query groups of `perGroup` rows; relevance correlates with feature 0.
    private static (float[][] X, double[] Rel, int[] Group) MakeRanking(int numGroups, int perGroup, int seed)
    {
        var rng = new Random(seed);
        var x = new List<float[]>();
        var rel = new List<double>();
        var group = new int[numGroups];
        for (int g = 0; g < numGroups; g++)
        {
            for (int r = 0; r < perGroup; r++)
            {
                var row = new float[3];
                for (int j = 0; j < 3; j++) row[j] = (float)rng.NextDouble();
                x.Add(row);
                rel.Add(Math.Round(row[0] * 4));   // relevance grade 0..4 from feature 0
            }
            group[g] = perGroup;
        }
        return (x.ToArray(), rel.ToArray(), group);
    }
}
