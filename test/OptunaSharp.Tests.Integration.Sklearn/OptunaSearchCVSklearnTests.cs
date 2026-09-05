using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Integration;
using OptunaSharp.Integration.Sklearn;

namespace OptunaSharp.Tests.Integration.Sklearn;

/// <summary>
///     End-to-end wiring of the Core <see cref="OptunaSearchCV" /> over the native
///     <see cref="SklearnEstimator" /> (no Python): the search runs, records scikit-learn-style user
///     attributes, refits, and is reproducible for a fixed <c>random_state</c>. The byte-for-byte parity
///     vs Python's <c>optuna_integration.OptunaSearchCV</c> lives in the interop suite.
/// </summary>
[TestClass]
public class OptunaSearchCVSklearnTests
{
    private static (double[,] X, double[] y) Data(int n = 60, int seed = 11)
    {
        var rng = new Random(seed);
        var x = new double[n, 3];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            double a = rng.NextDouble() * 4 - 2, b = rng.NextDouble() * 4 - 2, c = rng.NextDouble() * 4 - 2;
            x[i, 0] = a; x[i, 1] = b; x[i, 2] = c;
            y[i] = 2 * a - b + 0.5 * c * c;
        }
        return (x, y);
    }

    private static IReadOnlyDictionary<string, BaseDistribution> Dists() => new Dictionary<string, BaseDistribution>
    {
        ["n_estimators"] = new IntDistribution(5, 20),
        ["max_depth"] = new IntDistribution(2, 8),
        ["min_samples_split"] = new IntDistribution(2, 10),
    };

    private static OptunaSearchCV Run(int nTrials = 12, int randomState = 42, bool refit = true, bool returnTrain = false)
    {
        var (x, y) = Data();
        var estimator = new SklearnEstimator(new RandomForestRegressor(randomState: 0), x, y, new KFold(5));
        return new OptunaSearchCV(estimator, Dists(), nTrials: nTrials, randomState: randomState,
            refit: refit, returnTrainScore: returnTrain).Fit();
    }

    [TestMethod]
    public void Search_Runs_AndExposesBestResultSurface()
    {
        var search = Run();

        search.NTrials.Should().Be(12);
        search.NSplits.Should().Be(5);
        search.Study.Direction.Should().Be(StudyDirection.Maximize);
        search.BestScore.Should().Be(search.Study.BestValue);
        search.BestParams.Should().ContainKeys("n_estimators", "max_depth", "min_samples_split");
        search.BestIndex.Should().Be(search.BestTrial.Number);
        search.BestScore.Should().BeGreaterThan(0.0);   // a forest fits the smooth target
    }

    [TestMethod]
    public void Search_StoresScikitLearnStyleUserAttrs()
    {
        var search = Run(returnTrain: true);
        var attrs = search.BestTrial.UserAttrs;

        attrs.Should().ContainKey("mean_test_score");
        attrs.Should().ContainKey("std_test_score");
        attrs.Should().ContainKey("mean_fit_time");
        attrs.Should().ContainKey("mean_score_time");
        attrs.Should().ContainKey("mean_train_score");   // returnTrainScore = true
        for (int i = 0; i < 5; i++)
        {
            attrs.Should().ContainKey($"split{i}_test_score");
            attrs.Should().ContainKey($"split{i}_train_score");
        }

        // The objective value == the stored mean_test_score.
        ((double)attrs["mean_test_score"]!).Should().Be(search.BestTrial.Value);
    }

    [TestMethod]
    public void Search_IsReproducibleForAFixedRandomState()
    {
        var a = Run(randomState: 7);
        var b = Run(randomState: 7);

        a.BestScore.Should().Be(b.BestScore);
        a.NTrials.Should().Be(b.NTrials);
        for (int i = 0; i < a.NTrials; i++)
        {
            ((double)a.Trials[i].UserAttrs["mean_test_score"]!)
                .Should().Be((double)b.Trials[i].UserAttrs["mean_test_score"]!);
            a.Trials[i].Params["max_depth"].Should().Be(b.Trials[i].Params["max_depth"]);
        }
    }

    [TestMethod]
    public void CvResults_ExposesPerTrialColumns()
    {
        var search = Run();
        var cv = search.CvResults;

        cv.Should().ContainKey("mean_test_score");
        cv["mean_test_score"].Count.Should().Be(search.NTrials);
        cv["mean_test_score"].Should().OnlyContain(v => v is double);
    }

    [TestMethod]
    public void Refit_MakesThePredictSurfaceAvailable()
    {
        var (xTest, _) = Data(n: 5, seed: 99);
        var search = Run(refit: true);

        var estimator = (SklearnEstimator)search.Estimator;
        search.RefitTimeSeconds.Should().BeGreaterThanOrEqualTo(0.0);   // refit_time_ was set
        estimator.BestEstimator.Should().NotBeNull();

        var preds = estimator.Predict(xTest);
        preds.Length.Should().Be(5);
        preds.Should().OnlyContain(v => !double.IsNaN(v));
    }

    [TestMethod]
    public void NoRefit_LeavesPredictUnavailable()
    {
        var (xTest, _) = Data(n: 5, seed: 99);
        var search = Run(refit: false);
        var estimator = (SklearnEstimator)search.Estimator;

        estimator.BestEstimator.Should().BeNull();
        Action predict = () => estimator.Predict(xTest);
        predict.Should().Throw<InvalidOperationException>().WithMessage("*not been refit*");
    }

    [TestMethod]
    public void Study_DirectionGuard_RejectsMinimize()
    {
        var (x, y) = Data();
        var estimator = new SklearnEstimator(new RandomForestRegressor(randomState: 0), x, y);
        var minimize = Optuna.CreateStudy(direction: StudyDirection.Minimize);

        Action bad = () => new OptunaSearchCV(estimator, Dists(), study: minimize).Fit();
        bad.Should().Throw<NumSharp.ValueError>().WithMessage("*must be 'maximize'*");
    }
}
