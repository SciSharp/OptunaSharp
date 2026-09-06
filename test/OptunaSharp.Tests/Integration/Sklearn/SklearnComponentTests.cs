using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Integration.Sklearn;

namespace OptunaSharp.Tests.Integration.Sklearn;

/// <summary>
///     Unit coverage for the byte-exact scikit-learn components (no Python): <see cref="KFold" />,
///     <see cref="Metrics" /> (r2 / MSE) against scikit-learn's own documented constants, and
///     <see cref="RandomForestRegressor" /> reproducibility / clone semantics. The live byte-for-byte
///     parity vs scikit-learn 1.8.0 lives in the interop suite.
/// </summary>
[TestClass]
public class SklearnComponentTests
{
    // ---- KFold (sklearn.model_selection.KFold) -------------------------------------------------

    [TestMethod]
    public void KFold_ContiguousFolds_MatchSklearnLayout()
    {
        // n=13, k=5 -> sizes [3,3,3,2,2], contiguous (the first n%k folds get the extra sample).
        var kf = new KFold(5);
        kf.GetNumSplits(13).Should().Be(5);

        var tests = kf.Split(13).Select(f => f.Test).ToList();
        tests[0].Should().Equal(0, 1, 2);
        tests[1].Should().Equal(3, 4, 5);
        tests[2].Should().Equal(6, 7, 8);
        tests[3].Should().Equal(9, 10);
        tests[4].Should().Equal(11, 12);

        // train = ascending complement of the test block.
        var first = kf.Split(13).First();
        first.Train.Should().Equal(3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
    }

    [TestMethod]
    public void KFold_EveryFold_TrainAndTest_ArePartitionAndAscending()
    {
        foreach (int n in new[] { 10, 12, 13, 25 })
        {
            var seen = new List<int>();
            foreach (var (train, test) in new KFold(5).Split(n))
            {
                test.Should().BeInAscendingOrder();
                train.Should().BeInAscendingOrder();
                train.Intersect(test).Should().BeEmpty();
                (train.Length + test.Length).Should().Be(n);
                seen.AddRange(test);
            }

            seen.OrderBy(x => x).Should().Equal(Enumerable.Range(0, n));
        }
    }

    [TestMethod]
    public void KFold_Shuffle_IsDeterministicForASeed_AndStillAscendingWithinFolds()
    {
        var a = new KFold(4, shuffle: true, randomState: 0).Split(20).Select(f => f.Test).ToList();
        var b = new KFold(4, shuffle: true, randomState: 0).Split(20).Select(f => f.Test).ToList();
        for (int i = 0; i < a.Count; i++)
        {
            a[i].Should().Equal(b[i]);                 // same seed -> same folds
            a[i].Should().BeInAscendingOrder();        // membership shuffled, order ascending
        }

        // Different seed -> different membership (overwhelmingly likely).
        var c = new KFold(4, shuffle: true, randomState: 1).Split(20).Select(f => f.Test).ToList();
        a.SequenceEqual(c, EqualityComparer<int[]>.Default).Should().BeFalse();
    }

    [TestMethod]
    public void KFold_Validation_Errors_MatchSklearn()
    {
        Action tooFew = () => new KFold(1);
        tooFew.Should().Throw<ValueError>().WithMessage("*at least one train/test split*");

        Action seedNoShuffle = () => new KFold(5, shuffle: false, randomState: 3);
        seedNoShuffle.Should().Throw<ValueError>().WithMessage("*random_state has no effect*");

        Action moreSplitsThanSamples = () => new KFold(5).Split(3).ToList();
        moreSplitsThanSamples.Should().Throw<ValueError>().WithMessage("*greater than the number of samples*");
    }

    [TestMethod]
    public void CheckCv_ResolvesLikeSklearnRegressorPath()
    {
        CrossValidation.CheckCv(null).Should().BeOfType<KFold>().Which.NSplits.Should().Be(5);
        CrossValidation.CheckCv(3).Should().BeOfType<KFold>().Which.NSplits.Should().Be(3);
        var kf = new KFold(7);
        CrossValidation.CheckCv(kf).Should().BeSameAs(kf);
    }

    // ---- Metrics (sklearn.metrics.r2_score / mean_squared_error) -------------------------------

    [TestMethod]
    public void R2Score_MatchesSklearnDocstringConstants()
    {
        // Exact rational cases (the sklearn r2_score docstring).
        Metrics.R2Score(new[] { 1.0, 2, 3 }, new[] { 1.0, 2, 3 }).Should().Be(1.0);
        Metrics.R2Score(new[] { 1.0, 2, 3 }, new[] { 2.0, 2, 2 }).Should().Be(0.0);
        Metrics.R2Score(new[] { 1.0, 2, 3 }, new[] { 3.0, 2, 1 }).Should().Be(-3.0);

        // The headline docstring example (0.948...).
        Metrics.R2Score(new[] { 3.0, -0.5, 2, 7 }, new[] { 2.5, 0.0, 2, 8 })
            .Should().BeApproximately(0.9486081370449679, 1e-15);
    }

    [TestMethod]
    public void R2Score_ForceFinite_ConstantTargetEdgeCases()
    {
        // Constant y_true, perfect prediction -> numerator 0 -> 1.0 (force_finite default).
        Metrics.R2Score(new[] { -2.0, -2, -2 }, new[] { -2.0, -2, -2 }).Should().Be(1.0);
        // Constant y_true, imperfect -> numerator != 0, denominator 0 -> 0.0.
        Metrics.R2Score(new[] { -2.0, -2, -2 }, new[] { -2.0, -2, -2 + 1e-8 }).Should().Be(0.0);
        // Without force_finite: perfect -> NaN, imperfect -> -inf.
        Metrics.R2Score(new[] { -2.0, -2, -2 }, new[] { -2.0, -2, -2 }, forceFinite: false)
            .Should().Be(double.NaN);
        Metrics.R2Score(new[] { -2.0, -2, -2 }, new[] { -2.0, -2, -2 + 1e-8 }, forceFinite: false)
            .Should().Be(double.NegativeInfinity);
    }

    [TestMethod]
    public void R2Score_FewerThanTwoSamples_IsNaN()
    {
        Metrics.R2Score(new[] { 5.0 }, new[] { 4.0 }).Should().Be(double.NaN);
    }

    [TestMethod]
    public void MeanSquaredError_MatchesHandComputedValue()
    {
        // ((0.5)^2 + (0.5)^2 + 0 + 1^2) / 4 = 1.5 / 4.
        Metrics.MeanSquaredError(new[] { 3.0, -0.5, 2, 7 }, new[] { 2.5, 0.0, 2, 8 })
            .Should().Be(0.375);
    }

    // ---- RandomForestRegressor ----------------------------------------------------------------

    private static (double[,] X, double[] y) SampleData(int n = 40, int seed = 7)
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

    [TestMethod]
    public void RandomForest_SameSeed_IsReproducible_DifferentSeedDiffers()
    {
        var (x, y) = SampleData();

        var p0a = Fit(new RandomForestRegressor(nEstimators: 8, maxDepth: 5, randomState: 0), x).Predict(x);
        var p0b = Fit(new RandomForestRegressor(nEstimators: 8, maxDepth: 5, randomState: 0), x).Predict(x);
        p0a.Should().Equal(p0b);   // bit-identical for a fixed seed

        var p1 = Fit(new RandomForestRegressor(nEstimators: 8, maxDepth: 5, randomState: 1), x).Predict(x);
        p0a.SequenceEqual(p1).Should().BeFalse();

        RandomForestRegressor Fit(RandomForestRegressor rf, double[,] data) { rf.Fit(data, y); return rf; }
    }

    [TestMethod]
    public void RandomForest_Score_EqualsR2OfPredictions()
    {
        var (x, y) = SampleData();
        var rf = new RandomForestRegressor(nEstimators: 10, maxDepth: 6, randomState: 0);
        rf.Fit(x, y);
        rf.Score(x, y).Should().Be(Metrics.R2Score(y, rf.Predict(x)));
        rf.Score(x, y).Should().BeGreaterThan(0.9);   // fits the smooth target well
    }

    [TestMethod]
    public void RandomForest_WithParams_ClonesAndOverrides_LeavingOriginalUnfitted()
    {
        var (x, y) = SampleData();
        var baseRf = new RandomForestRegressor(nEstimators: 5, maxDepth: 3, randomState: 0);

        var tuned = (RandomForestRegressor)baseRf.WithParams(new Dictionary<string, object?>
        {
            ["n_estimators"] = 12L,   // Optuna IntDistribution suggests boxed long
            ["max_depth"] = 7L,
        });
        tuned.NEstimators.Should().Be(12);

        // WithParams returns a fresh, unfitted clone; fitting it must not touch the base estimator.
        tuned.Fit(x, y);
        tuned.Trees.Should().NotBeNull();
        baseRf.Trees.Should().BeNull();
        baseRf.NEstimators.Should().Be(5);
    }

    [TestMethod]
    public void RandomForest_Predict_BeforeFit_Throws()
    {
        var (x, _) = SampleData();
        Action predict = () => new RandomForestRegressor().Predict(x);
        predict.Should().Throw<InvalidOperationException>().WithMessage("*not fitted*");
    }
}
