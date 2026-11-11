using System;
using System.Reflection;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.AgeMoea;
using OptunaSharp.Samplers.Janus;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Pins the reductions moved onto NumSharp on 2026-11-11 to NumPy 2.4.2's results for inputs where the replaced
///     hand-written loops rounded differently, so each test fails on the old order and passes on NumPy's. NumPy's order
///     is not always a sequential sum: a contiguous run is summed pairwise (eight accumulators from eight elements on),
///     an outer axis of a matrix is summed row by row, and <c>np.linalg.norm(x, p, axis=1)</c> reduces each row pairwise.
///     Every expected value below was printed by NumPy itself (bit patterns, so signs and last bits are exact).
/// </summary>
[TestClass]
public class NumPyReductionOrderTests
{
    /// <summary>Asserts two doubles are the same bits, with a readable message naming both values.</summary>
    /// <param name="actual">The value OptunaSharp computed.</param>
    /// <param name="expectedBits">NumPy's result as its IEEE-754 bit pattern.</param>
    /// <param name="what">What the value is, for the failure message.</param>
    private static void ShouldBeBits(double actual, long expectedBits, string what)
        => BitConverter.DoubleToInt64Bits(actual).Should().Be(expectedBits,
            $"{what} must be NumPy's {BitConverter.Int64BitsToDouble(expectedBits):R}, not {actual:R}");

    /// <summary>
    ///     GPSampler's <c>_standardize_values</c> over two objectives and ten trials: NumPy's <c>np.mean(values, axis=0)</c>
    ///     and <c>np.std(values, axis=0)</c> add the rows in order (a sequential sum per column). The former per-column
    ///     pairwise sums gave means of <c>100.420794</c> and <c>100.54714200000001</c> where NumPy gives
    ///     <c>100.42079399999999</c> and <c>100.547142</c>.
    /// </summary>
    [TestMethod]
    public void GpStandardize_TwoObjectives_ColumnStatisticsFollowNumPysAxisZeroOrder()
    {
        double[,] values =
        {
            { 100.65128, 97.15331 }, { 104.6579, 104.64801 }, { 101.65433, 99.91153 }, { 100.57431, 96.71228 },
            { 99.28281, 99.6852 }, { 97.4433, 102.54764 }, { 95.72056, 98.63974 }, { 93.24137, 101.49046 },
            { 102.22782, 101.86388 }, { 108.75426, 102.81937 },
        };
        MethodInfo standardize = typeof(GPSampler).GetMethod("StandardizeValues", BindingFlags.NonPublic | BindingFlags.Static)!;

        var (standardized, means, stds) = ((double[,], double[], double[]))standardize.Invoke(null, new object[] { values })!;

        ShouldBeBits(means[0], unchecked((long)0x40591AEE49F51697UL), "np.mean(values, axis=0)[0]");
        ShouldBeBits(means[1], unchecked((long)0x405923045FE11127UL), "np.mean(values, axis=0)[1]");
        ShouldBeBits(stds[0], unchecked((long)0x4010C8A7C6404E4BUL), "np.std(values, axis=0)[0]");
        ShouldBeBits(stds[1], unchecked((long)0x4003887101742D6CUL), "np.std(values, axis=0)[1]");
        // (v - mean) / max(EPS, std), one subtraction and one division per element, as NumPy's broadcast ufuncs.
        standardized[3, 1].Should().Be((values[3, 1] - means[1]) / stds[1]);
    }

    /// <summary>
    ///     AGE-MOEA's survival norms over a ten-objective front: pymoo's <c>np.linalg.norm(front, 2, axis=1)</c> is
    ///     <c>sqrt(add.reduce(x * x, axis=1))</c>, a pairwise sum of the squares. The former left-to-right sum gave
    ///     <c>1.9441723203124253</c> where NumPy gives <c>1.9441723203124255</c>.
    /// </summary>
    [TestMethod]
    public void AgeMoeaRowNorms_TenObjectives_FollowNumPysPairwiseRowSums()
    {
        double[][] front =
        {
            new[] { 0.207855, 0.335135, 0.567565, 0.957842, 0.936312, 0.073896, 0.853314, 0.167824, 0.811433, 0.296347 },
        };

        double[] norms = AgeMoeaSurvival.RowNorms(front, 2.0);

        norms.Should().HaveCount(1);
        ShouldBeBits(norms[0], unchecked((long)0x3FFF1B546F587C20UL), "np.linalg.norm(front, 2, axis=1)[0]");
    }

    /// <summary>
    ///     JANUS's fallback objective for a valueless trial, the reference's <c>float(np.mean(rae))</c> over ten component
    ///     losses: NumPy sums them pairwise. A LINQ <c>Average()</c> gave <c>0.738784</c> where NumPy gives
    ///     <c>0.7387840000000001</c>.
    /// </summary>
    [TestMethod]
    public void JanusComponentMean_TenComponents_IsNumPysPairwiseMean()
    {
        double[] rae = { 0.41115, 1.10275, 0.52537, 0.83257, 0.04189, 0.56352, 0.54136, 1.83826, 1.17594, 0.35503 };
        MethodInfo componentMean = typeof(JanusOptimizer).GetMethod("ComponentMean", BindingFlags.NonPublic | BindingFlags.Static)!;

        double mean = (double)componentMean.Invoke(null, new object[] { rae })!;

        ShouldBeBits(mean, unchecked((long)0x3FE7A41E57D9DBAAUL), "np.mean(rae)");
    }
}
