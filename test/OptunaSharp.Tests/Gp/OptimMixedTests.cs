using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Gp;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Gp;

/// <summary>
///     Unit coverage for the mixed continuous/discrete acqf optimizer (PORTING_PLAN Phase 3.2 part 3d —
///     <c>optuna/_gp/optim_mixed.py</c>) and its two building blocks: the Brent scalar minimizer
///     (<see cref="BrentMinimizer" />, scipy's <c>minimize_scalar(method="brent")</c>) and the numpy
///     weighted sampling-without-replacement (<see cref="OptimMixed.ChoiceNoReplaceWeighted" />). The
///     numpy/scipy figures are pinned from real numpy/scipy; live-Optuna parity of the whole optimizer
///     lives in the interop <c>GpOptimMixedParityTests</c>.
/// </summary>
[TestClass]
public class OptimMixedTests
{
    [TestMethod]
    public void Brent_FindsMinimumOfParabola()
    {
        // scipy.optimize.minimize_scalar(method='brent') on (x-shift)^2 with a 3-point bracket.
        BrentMinimizer.MinimizeScalar(x => (x - 0.37) * (x - 0.37), 0.0, 0.4, 1.0, 1e-8)
            .Should().BeApproximately(0.37, 1e-6);
        BrentMinimizer.MinimizeScalar(x => (x - 0.812) * (x - 0.812), 0.5, 0.7, 1.0, 1e-8)
            .Should().BeApproximately(0.812, 1e-6);
    }

    [TestMethod]
    public void Brent_RejectsInvalidBracket()
    {
        // f(xb) must be below f(xa) and f(xc) — the same guard scipy's Brent raises.
        Action bad = () => BrentMinimizer.MinimizeScalar(x => x * x, 0.0, 0.5, 1.0, 1e-8);
        bad.Should().Throw<ArgumentException>();
    }

    [TestMethod]
    public void ChoiceNoReplaceWeighted_MatchesNumpy()
    {
        // np.random.RandomState(42).choice(100, size=8, replace=False, p=arange(1,101)/sum).
        var rng = new LazyRandomState(42).Rng;
        var p = new double[100];
        double sum = 0;
        for (int i = 0; i < 100; i++) { p[i] = i + 1; sum += p[i]; }
        for (int i = 0; i < 100; i++) p[i] /= sum;
        OptimMixed.ChoiceNoReplaceWeighted(rng, 100, 8, p)
            .Should().Equal(61, 97, 85, 77, 39, 23, 93, 75);

        // np.random.RandomState(7).choice(50, size=5, replace=False, p=uniform).
        var rng2 = new LazyRandomState(7).Rng;
        var pu = new double[50];
        for (int i = 0; i < 50; i++) pu[i] = 1.0 / 50.0;
        OptimMixed.ChoiceNoReplaceWeighted(rng2, 50, 5, pu)
            .Should().Equal(3, 38, 21, 36, 48);
    }

    [TestMethod]
    public void ChoiceNoReplaceWeighted_ReturnsDistinctInBounds()
    {
        var rng = new LazyRandomState(123).Rng;
        var p = new double[40];
        for (int i = 0; i < 40; i++) p[i] = 1.0 / 40.0;
        int[] result = OptimMixed.ChoiceNoReplaceWeighted(rng, 40, 12, p);
        result.Should().HaveCount(12);
        result.Distinct().Should().HaveCount(12);           // without replacement
        result.Should().OnlyContain(i => i >= 0 && i < 40);
    }

    [TestMethod]
    public void OptimizeAcqfMixed_IsDeterministic()
    {
        (double[] p1, double v1) = RunOptimize(seed: 5);
        (double[] p2, double v2) = RunOptimize(seed: 5);
        p1.Should().Equal(p2);
        v1.Should().Be(v2);
    }

    [TestMethod]
    public void OptimizeAcqfMixed_ReturnsInBoundsOnGridPoint()
    {
        var (space, gpr, x) = BuildScenario();
        var gpSpace = new GpSearchSpace(space);
        double threshold = 0.5;
        var acqf = new LogEI(gpr, gpSpace, threshold);
        var best = new double[1, 2];
        best[0, 0] = x[0, 0];
        best[0, 1] = x[0, 1];

        var rng = new LazyRandomState(9).Rng;
        (double[] param, double val) = OptimMixed.OptimizeAcqfMixed(acqf, best, 2048, 10, 1e-4, rng);

        param.Should().HaveCount(2);
        param[0].Should().BeInRange(0.0, 1.0);              // continuous float dim, normalized box
        // The discrete int dim must land on its normalized grid (one of the 11 choices for Int(0,10)).
        double[] grid = gpSpace.GetChoicesOfDiscreteParams()[0];
        grid.Should().Contain(g => Math.Abs(g - param[1]) < 1e-12);
        // The reported value is exactly the acqf at the returned point.
        val.Should().BeApproximately(acqf.EvalAcqfNoGrad(param), 1e-12);
    }

    [TestMethod]
    public void OptimizeAcqfMixed_AllCategorical_NoContinuousDims()
    {
        // With no continuous dim the gradient ascent is a no-op; the exhaustive search still runs and
        // returns an on-grid categorical code in [0, C).
        var space = new Dictionary<string, BaseDistribution>
        {
            ["c0"] = new CategoricalDistribution("a", "b", "c"),
            ["c1"] = new CategoricalDistribution("x", "y"),
        };
        var gpSpace = new GpSearchSpace(space);
        gpSpace.ContinuousIndices.Should().BeEmpty();

        int n = 8;
        var gen = new LazyRandomState(3).Rng;
        double[,] xTrain = gpSpace.SampleNormalizedParams(n, gen);
        var y = new double[n];
        var raw = gen.random_sample((long)n).ToArray<double>();
        for (int i = 0; i < n; i++) y[i] = raw[i];
        var gpr = new GPRegressor(gpSpace.IsCategorical, xTrain, y, new[] { 1.0, 1.0 }, 1.0, 0.05);
        gpr.CacheMatrix();
        var acqf = new LogEI(gpr, gpSpace, y.Max());
        var best = new double[1, 2] { { xTrain[0, 0], xTrain[0, 1] } };

        var rng = new LazyRandomState(1).Rng;
        (double[] param, _) = OptimMixed.OptimizeAcqfMixed(acqf, best, 2048, 10, 1e-4, rng);
        param[0].Should().BeInRange(0.0, 2.0);   // code in {0,1,2}
        param[1].Should().BeInRange(0.0, 1.0);   // code in {0,1}
        (param[0] == Math.Floor(param[0])).Should().BeTrue("categorical codes are integers");
    }

    // ---- shared scenario builders ---------------------------------------------------------------

    private static (Dictionary<string, BaseDistribution> Space, GPRegressor Gpr, double[,] X) BuildScenario()
    {
        var space = new Dictionary<string, BaseDistribution>
        {
            ["f"] = new FloatDistribution(0, 1),
            ["i"] = new IntDistribution(0, 10),
        };
        var gpSpace = new GpSearchSpace(space);
        int n = 12;
        var gen = new LazyRandomState(20).Rng;
        double[,] x = gpSpace.SampleNormalizedParams(n, gen);
        var y = new double[n];
        var raw = gen.random_sample((long)n).ToArray<double>();
        for (int i = 0; i < n; i++) y[i] = raw[i] * 2 - 1;
        var isc = new bool[] { false, false };
        var gpr = new GPRegressor(isc, x, y, new[] { 2.0, 3.0 }, 1.0, 0.01);
        gpr.CacheMatrix();
        return (space, gpr, x);
    }

    private static (double[] Param, double Val) RunOptimize(int seed)
    {
        var (space, gpr, x) = BuildScenario();
        var gpSpace = new GpSearchSpace(space);
        var acqf = new LogEI(gpr, gpSpace, 0.5);
        var best = new double[1, 2] { { x[0, 0], x[0, 1] } };
        var rng = new LazyRandomState(seed).Rng;
        return OptimMixed.OptimizeAcqfMixed(acqf, best, 2048, 10, 1e-4, rng);
    }
}
