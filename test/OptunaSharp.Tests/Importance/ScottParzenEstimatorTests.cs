using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Importance.PedAnova;
using OptunaSharp.Samplers.Tpe;

namespace OptunaSharp.Tests.Importance;

/// <summary>
///     Port of Optuna's <c>tests/importance_tests/pedanova_tests/test_scott_parzen_estimator.py</c>,
///     plus bit-exact grid-count / PDF anchors captured from live Optuna 5.0.0.dev.
/// </summary>
[TestClass]
public class ScottParzenEstimatorTests
{
    private static FrozenTrial ParamTrial(BaseDistribution dist, object? value)
        => FrozenTrial.CreateTrial(
            value: 0.0,
            @params: new OrderedDict<string, object?> { ["a"] = value },
            distributions: new OrderedDict<string, BaseDistribution> { ["a"] = dist });

    // ------------------------------------------------------------------ grid counting

    [TestMethod]
    public void CountNumerical_IntLinear()
    {
        var dist = new IntDistribution(-5, 5);
        var trials = new long[] { -5, -5, 1, 5, 5 }.Select(p => ParamTrial(dist, p)).ToList();
        ScottParzenEstimatorFunctions.CountNumericalParamInGrid("a", dist, trials, nSteps: 5)
            .Should().Equal(2L, 0L, 1L, 0L, 2L);
    }

    [TestMethod]
    public void CountNumerical_IntLog()
    {
        var dist = new IntDistribution(1, 8, log: true);
        var trials = Enumerable.Range(1, 8).Select(p => ParamTrial(dist, (long)p)).ToList();
        ScottParzenEstimatorFunctions.CountNumericalParamInGrid("a", dist, trials, nSteps: 5)
            .Should().Equal(1L, 1L, 3L, 3L);
    }

    [TestMethod]
    public void CountNumerical_FloatLinear()
    {
        var dist = new FloatDistribution(-5.0, 5.0);
        // np.linspace(-5, 5, 100).
        var values = Enumerable.Range(0, 100).Select(i => -5.0 + 10.0 * i / 99.0).ToArray();
        var trials = values.Select(p => ParamTrial(dist, p)).ToList();
        ScottParzenEstimatorFunctions.CountNumericalParamInGrid("a", dist, trials, nSteps: 5)
            .Should().Equal(13L, 25L, 24L, 25L, 13L);
    }

    [TestMethod]
    public void CountNumerical_FloatLog()
    {
        var dist = new FloatDistribution(1, 8, log: true);
        var trials = Enumerable.Range(1, 8).Select(p => ParamTrial(dist, (double)p)).ToList();
        ScottParzenEstimatorFunctions.CountNumericalParamInGrid("a", dist, trials, nSteps: 5)
            .Should().Equal(1L, 1L, 1L, 3L, 2L);
    }

    [TestMethod]
    public void CountCategorical()
    {
        var dist = new CategoricalDistribution("a", "b", "c", "d");
        var trials = new[] { "a", "b", "a", "d", "a", "a", "d" }.Select(p => ParamTrial(dist, p)).ToList();
        ScottParzenEstimatorFunctions.CountCategoricalParamInGrid("a", dist, trials)
            .Should().Equal(4L, 1L, 0L, 2L);
    }

    // ------------------------------------------------------------------ Scott's-rule bandwidth

    [DataTestMethod]
    [DataRow(new[] { 0, 0, 0, 1 }, 0.3048780487804878)]
    [DataRow(new[] { 1, 2, 3, 4 }, 0.7043275971840145)]
    [DataRow(new[] { 90, 0, 0, 90 }, 0.5638225678863902)]
    [DataRow(new[] { 1, 0, 0, 1 }, 1.955672900598399)]
    public void BuildInt_SigmaBitExact(int[] counts, double expectedSigma)
    {
        var pe = BuildFromCounts(counts, new IntDistribution(0, counts.Length - 1));
        var dist = (BatchedDiscreteTruncNormDistributions)pe.MixtureDistribution.Distributions[0];
        dist.Sigma[0].Should().Be(expectedSigma);
        // The prior kernel's sigma is always high - low + 1.
        dist.Sigma[^1].Should().Be(counts.Length);
        // The prior kernel's mu is always the domain midpoint.
        dist.Mu[^1].Should().Be((0 + (counts.Length - 1)) / 2.0);
    }

    [TestMethod]
    public void BuildInt_MuAndWeights()
    {
        var pe = BuildFromCounts(new[] { 1, 2, 3, 4 }, new IntDistribution(0, 3));
        var dist = (BatchedDiscreteTruncNormDistributions)pe.MixtureDistribution.Distributions[0];
        dist.Mu.Should().Equal(0.0, 1.0, 2.0, 3.0, 1.5);
        pe.MixtureDistribution.Weights.Should().Equal(0.1, 0.2, 0.3, 0.4, 0.0);
    }

    [TestMethod]
    public void BuildCat_Weights()
    {
        var pe = BuildFromCounts(new[] { 1, 2, 3, 4 },
            new CategoricalDistribution("", "a", "aa", "aaa"), isCategorical: true);
        var dist = (BatchedCategoricalDistributions)pe.MixtureDistribution.Distributions[0];
        // Identity rows for the observed choices, then a zero prior row (prior_weight == 0).
        dist.Weights[0, 0].Should().Be(1.0);
        dist.Weights[1, 1].Should().Be(1.0);
        dist.Weights[2, 2].Should().Be(1.0);
        dist.Weights[3, 3].Should().Be(1.0);
        for (int c = 0; c < 4; c++)
            dist.Weights[4, c].Should().Be(0.0);
    }

    // ------------------------------------------------------------------ pdf bit-exactness

    [TestMethod]
    public void NumericalPdf_BitExact()
    {
        var pe = new ScottParzenEstimator(
            observations: new double[] { 3, 7, 20, 41 },
            searchSpace: new IntDistribution(0, 49),
            predeterminedWeights: new double[] { 2, 5, 1, 3 },
            priorWeight: 1.0);
        var grid = Enumerable.Range(0, 50).Select(i => (double)i).ToArray();
        var pdf = pe.Pdf(grid);

        var expectedHead = new[]
        {
            0.028218362047150353, 0.029533988159983513, 0.030674099434163063,
            0.031614612401403334, 0.032335938266966346, 0.032823778735507014,
        };
        // The numerical Parzen density rides on erf, so it matches Python only to a few ULP (the
        // erf/ucrt divergence documented on TruncNorm). The categorical path below is bit-exact.
        for (int i = 0; i < expectedHead.Length; i++)
            pdf[i].Should().BeApproximately(expectedHead[i], 1e-13);
        pdf.Sum().Should().BeApproximately(1.0, 1e-12);
    }

    [TestMethod]
    public void CategoricalPdf_BitExact()
    {
        var pe = new ScottParzenEstimator(
            observations: new double[] { 0, 3 },
            searchSpace: new CategoricalDistribution("a", "b", "c", "d"),
            predeterminedWeights: new double[] { 1, 1 },
            priorWeight: 1.0);
        var pdf = pe.Pdf(new double[] { 0, 1, 2, 3 });
        pdf.Should().Equal(
            0.3214285714285715, 0.17857142857142855, 0.17857142857142855, 0.3214285714285715);
    }

    // ------------------------------------------------------------------ builder dispatch

    [TestMethod]
    public void BuildParzenEstimator_TypeAndGridSize()
    {
        var intDist = new IntDistribution(-5, 5);
        var intTrials = new long[] { 1, 2, 3 }.Select(p => ParamTrial(intDist, p)).ToList();
        var (pe, gridSize) = ScottParzenEstimatorFunctions.BuildParzenEstimatorOnGrid(
            "a", intDist, intTrials, nSteps: 50, priorWeight: 0.0);
        pe.MixtureDistribution.Distributions[0].Should().BeOfType<BatchedDiscreteTruncNormDistributions>();
        gridSize.Should().Be(11);  // min(round((5 - -5)/1) + 1, 50).

        var catDist = new CategoricalDistribution("a", "b", "c");
        var catTrials = new[] { "a", "b", "b" }.Select(p => ParamTrial(catDist, p)).ToList();
        var (catPe, catGrid) = ScottParzenEstimatorFunctions.BuildParzenEstimatorOnGrid(
            "a", catDist, catTrials, nSteps: 50, priorWeight: 0.0);
        catPe.MixtureDistribution.Distributions[0].Should().BeOfType<BatchedCategoricalDistributions>();
        catGrid.Should().Be(3);
    }

    private static ScottParzenEstimator BuildFromCounts(int[] counts, BaseDistribution roundedDist, bool isCategorical = false)
    {
        var observations = new List<double>();
        var weights = new List<double>();
        for (int i = 0; i < counts.Length; i++)
        {
            if (counts[i] != 0)
            {
                observations.Add(i);
                weights.Add(counts[i]);
            }
        }
        return new ScottParzenEstimator(observations.ToArray(), roundedDist, weights.ToArray(), priorWeight: 0.0);
    }
}
