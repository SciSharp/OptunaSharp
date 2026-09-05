using System;
using System.Collections.Generic;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Integration;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Integration.MLNet;

/// <summary>
///     Tests for <see cref="OptunaSearchCV" /> — the .NET port of
///     <c>optuna_integration.sklearn.OptunaSearchCV</c>. The search logic, the user-attribute bookkeeping
///     (<c>_store_scores</c>), the maximize direction, reproducibility, and the refit surface are asserted
///     against a deterministic in-memory <see cref="ICrossValidationEstimator" /> (no ML.NET) — the
///     analogue of the LightGBM tuner's synthetic-scorer tests. A separate end-to-end test drives the real
///     <see cref="MLNetEstimator" />.
/// </summary>
[TestClass]
public class OptunaSearchCVTests
{
    // A deterministic estimator: the fold score is a function of the sampled params (so the search is
    // meaningful and reproducible), or a fixed per-fold vector (so the mean/std bookkeeping is checkable).
    private sealed class SyntheticEstimator : ICrossValidationEstimator
    {
        private readonly int _folds;
        private readonly Func<IReadOnlyDictionary<string, object?>, double> _score;
        private readonly double[]? _fixedFoldScores;

        public IReadOnlyDictionary<string, object?>? LastRefitParams;
        public int RefitCount;

        public SyntheticEstimator(
            Func<IReadOnlyDictionary<string, object?>, double> score, int folds = 3, double[]? fixedFoldScores = null)
        {
            _score = score;
            _fixedFoldScores = fixedFoldScores;
            _folds = fixedFoldScores?.Length ?? folds;
        }

        public int GetNumSplits() => _folds;

        public IReadOnlyList<CrossValidationFoldResult> CrossValidate(
            IReadOnlyDictionary<string, object?> parameters, bool returnTrainScore)
        {
            var list = new List<CrossValidationFoldResult>(_folds);
            for (int i = 0; i < _folds; i++)
            {
                double test = _fixedFoldScores is not null ? _fixedFoldScores[i] : _score(parameters);
                double? train = returnTrainScore ? test + 1.0 : (double?)null;
                list.Add(new CrossValidationFoldResult(test, train, 0.001 * (i + 1), 0.0005 * (i + 1)));
            }

            return list;
        }

        public void Refit(IReadOnlyDictionary<string, object?> parameters)
        {
            LastRefitParams = new Dictionary<string, object?>(parameters);
            RefitCount++;
        }
    }

    private static Dictionary<string, BaseDistribution> BowlSpace()
        => new() { ["x"] = new FloatDistribution(-10.0, 10.0) };

    // The classic 1-D bowl, maximized at x = 2 (negated so higher is better, per the scorer convention).
    private static double BowlScore(IReadOnlyDictionary<string, object?> p)
        => -Math.Pow((double)p["x"]! - 2.0, 2.0);

    [TestMethod]
    public void Fit_RunsRequestedTrials_AndMaximizes()
    {
        var estimator = new SyntheticEstimator(BowlScore);
        var search = new OptunaSearchCV(estimator, BowlSpace(), nTrials: 30, randomState: 0);

        search.Fit();

        search.NTrials.Should().Be(30);
        search.NSplits.Should().Be(3);
        search.BestScore.Should().Be(search.Study.BestValue);
        search.BestScore.Should().BeLessThanOrEqualTo(0.0);
        // 30 TPE trials on a 1-D bowl over [-10, 10] land comfortably within |x - 2| < 3 (score > -9).
        search.BestScore.Should().BeGreaterThan(-9.0);
        search.BestIndex.Should().Be(search.BestTrial.Number);
        search.BestParams.Should().ContainKey("x");
    }

    [TestMethod]
    public void Fit_IsReproducible_ForTheSameRandomState()
    {
        double RunOnce()
        {
            var estimator = new SyntheticEstimator(BowlScore);
            var search = new OptunaSearchCV(estimator, BowlSpace(), nTrials: 15, randomState: 42);
            search.Fit();
            return (double)search.BestParams["x"]!;
        }

        // Deterministic estimator + a seeded TPE sampler -> bit-identical best parameter.
        RunOnce().Should().Be(RunOnce());
    }

    [TestMethod]
    public void Fit_StoresCrossValidationUserAttrs()
    {
        // Fixed per-fold test scores [1, 2, 3]: mean = 2, population std = sqrt(2/3).
        var estimator = new SyntheticEstimator(_ => 0.0, fixedFoldScores: new[] { 1.0, 2.0, 3.0 });
        var search = new OptunaSearchCV(estimator, BowlSpace(), nTrials: 1, randomState: 0);

        search.Fit();

        var attrs = search.BestTrial.UserAttrs;
        attrs.Should().ContainKeys(
            "split0_test_score", "split1_test_score", "split2_test_score",
            "mean_test_score", "std_test_score",
            "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time");

        ((double)attrs["split1_test_score"]!).Should().Be(2.0);
        ((double)attrs["mean_test_score"]!).Should().Be(2.0);
        ((double)attrs["std_test_score"]!).Should().BeApproximately(Math.Sqrt(2.0 / 3.0), 1e-12);
        // The objective returns the mean test score, so it is the (single) best value.
        search.BestScore.Should().Be(2.0);
    }

    [TestMethod]
    public void Fit_WithReturnTrainScore_AddsTrainAttrs()
    {
        var estimator = new SyntheticEstimator(BowlScore);
        var search = new OptunaSearchCV(estimator, BowlSpace(), nTrials: 3, randomState: 0, returnTrainScore: true);

        search.Fit();

        search.BestTrial.UserAttrs.Should().ContainKeys(
            "split0_train_score", "mean_train_score", "std_train_score");
    }

    [TestMethod]
    public void Refit_IsInvokedWithBestParams_WhenRefitTrue()
    {
        var estimator = new SyntheticEstimator(BowlScore);
        var search = new OptunaSearchCV(estimator, BowlSpace(), nTrials: 5, randomState: 0, refit: true);

        search.Fit();

        estimator.RefitCount.Should().Be(1);
        estimator.LastRefitParams.Should().BeEquivalentTo(search.BestParams);
        search.RefitTimeSeconds.Should().BeGreaterThanOrEqualTo(0.0);
    }

    [TestMethod]
    public void Refit_IsSkipped_WhenRefitFalse()
    {
        var estimator = new SyntheticEstimator(BowlScore);
        var search = new OptunaSearchCV(estimator, BowlSpace(), nTrials: 5, randomState: 0, refit: false);

        search.Fit();

        estimator.RefitCount.Should().Be(0);
    }

    [TestMethod]
    public void CvResults_HasOneEntryPerTrial()
    {
        var estimator = new SyntheticEstimator(BowlScore);
        var search = new OptunaSearchCV(estimator, BowlSpace(), nTrials: 4, randomState: 0);

        search.Fit();

        search.CvResults.Should().ContainKey("mean_test_score");
        search.CvResults["mean_test_score"].Count.Should().Be(4);
    }

    [TestMethod]
    public void Fit_UsesProvidedStudy()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Maximize, sampler: new TPESampler(seed: 0));
        var estimator = new SyntheticEstimator(BowlScore);
        var search = new OptunaSearchCV(estimator, BowlSpace(), study: study, nTrials: 6);

        search.Fit();

        search.Study.Should().BeSameAs(study);
        study.Trials.Count.Should().Be(6);
    }

    [TestMethod]
    public void Fit_WithNonMaximizeStudy_Throws()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);
        var estimator = new SyntheticEstimator(BowlScore);
        var search = new OptunaSearchCV(estimator, BowlSpace(), study: study);

        Action act = () => search.Fit();

        act.Should().Throw<ValueError>().WithMessage("*direction of study must be 'maximize'*");
    }

    [TestMethod]
    public void Properties_ThrowBeforeFit()
    {
        var estimator = new SyntheticEstimator(BowlScore);
        var search = new OptunaSearchCV(estimator, BowlSpace());

        Action act = () => _ = search.BestParams;

        act.Should().Throw<InvalidOperationException>().WithMessage("*not fitted*");
    }
}
