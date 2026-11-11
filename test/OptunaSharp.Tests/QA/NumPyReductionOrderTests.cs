using System;
using System.Collections.Generic;
using System.Reflection;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.AgeMoea;
using OptunaSharp.Samplers.Janus;
using OptunaSharp.Samplers.Nsgaiii;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Pins the reductions (and the argmax / max / isclose ports) moved onto NumSharp or onto NumPy's statements on
///     2026-11-11 to NumPy 2.4.2's results for inputs where the replaced hand-written loops answered differently, so each
///     test fails on the old code and passes on NumPy's. NumPy's order
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

    /// <summary>
    ///     NSGA-III's reference-point association over eight objectives (Optuna's
    ///     <c>_associate_individuals_with_reference_points</c> with the default 120 reference points):
    ///     <c>np.linalg.norm(differences, axis=2)</c> sums each row's eight squares pairwise. The former left-to-right
    ///     sum gave the closest line (index 50) a distance of <c>0.8832043678688792</c> where NumPy gives
    ///     <c>0.8832043678688793</c>; with seven objectives or fewer the two orders agree.
    /// </summary>
    [TestMethod]
    public void NsgaiiiAssociation_EightObjectives_DistanceIsNumPysPairwiseRowNorm()
    {
        double[,] objectives = { { 0.076308, 0.779919, 0.438409, 0.723465, 0.97799, 0.538496, 0.50112, 0.072051 } };
        double[,] referencePoints = NSGAIIIElitePopulationSelectionStrategy.GenerateDefaultReferencePoint(8, 3);
        MethodInfo associate = typeof(NSGAIIIElitePopulationSelectionStrategy)
            .GetMethod("AssociateIndividualsWithReferencePoints", BindingFlags.NonPublic | BindingFlags.Static)!;

        var (closest, distance) = ((int[], double[]))associate.Invoke(null, new object[] { objectives, referencePoints })!;

        closest.Should().Equal(50);
        ShouldBeBits(distance[0], unchecked((long)0x3FEC4335CE75CC4FUL), "np.min(np.linalg.norm(..., axis=2), axis=1)[0]");
    }

    /// <summary>
    ///     Every port of Optuna's <c>np.argmax</c> calls follows NumPy's rule: a NaN is maximal, so the first NaN wins;
    ///     otherwise the first maximum wins, a tie keeping the earlier index (<c>-0.0</c> and <c>0.0</c> tie). The
    ///     expected indices are NumPy 2.4.2's. The strict <c>&gt;</c> loops these helpers used until 2026-11-11 skipped
    ///     NaNs (QA-29 for TPE's <c>_compare</c>, QA-53's class for the GP sampler, its optimizer and the EMMR terminator,
    ///     and the HSSP greedy step, whose contributions turn NaN once a <c>-inf</c> loss makes the volume infinite). The
    ///     TPE and HSSP helpers keep a loop (they run per parameter / per greedy step); the others call NumSharp's
    ///     <c>np.argmax</c>.
    /// </summary>
    /// <param name="values">The scores, as NumPy sees them.</param>
    /// <param name="expected">NumPy's <c>np.argmax(values)</c>.</param>
    [TestMethod]
    [DataRow(new[] { 0.1, -0.2, double.NaN, 0.3, double.NaN }, 2)]
    [DataRow(new[] { double.NaN, 5.0 }, 0)]
    [DataRow(new[] { 0.0, -0.0, 1.0, 1.0 }, 2)]
    [DataRow(new[] { -0.0, 0.0 }, 0)]
    [DataRow(new[] { double.NegativeInfinity, double.NegativeInfinity }, 0)]
    [DataRow(new[] { 1.0, double.PositiveInfinity, double.NaN }, 2)]
    [DataRow(new[] { 2.0, 3.0, 3.0 }, 1)]
    public void ArgMaxPorts_FirstNanThenFirstMaximum_LikeNumPysArgmax(double[] values, int expected)
    {
        const BindingFlags PrivateStatic = BindingFlags.NonPublic | BindingFlags.Static;
        int Call(Type type, string name, params object[] arguments)
            => (int)type.GetMethod(name, PrivateStatic)!.Invoke(null, arguments)!;

        Call(typeof(TPESampler), "ArgMax", values).Should().Be(expected, "TPESampler._compare's np.argmax");
        Call(typeof(GPSampler), "ArgMax", values).Should().Be(expected, "GPSampler's np.argmax(y_with_neginf)");
        Call(typeof(OptunaSharp.Internal.Hypervolume), "ArgMax", (IReadOnlyList<double>)values)
            .Should().Be(expected, "hssp's np.argmax(contribs)");
        Call(typeof(OptunaSharp.Gp.OptimMixed), "Argmax", values).Should().Be(expected, "optim_mixed's np.argmax");
        Call(typeof(OptunaSharp.Terminator.EMMREvaluator), "ArgMax", values, values.Length)
            .Should().Be(expected, "EMMREvaluator's np.argmax(standarized_score_vals)");

        // GPSampler's single-objective np.argmax(standardized_score_vals) over the (n, 1) score matrix.
        var column = new double[values.Length, 1];
        for (int i = 0; i < values.Length; i++)
            column[i, 0] = values[i];
        Call(typeof(GPSampler), "ArgMaxColumn", column, 0).Should().Be(expected, "np.argmax(standardized_score_vals)");
    }

    /// <summary>
    ///     GPSampler's LogEI threshold, <c>standardized_score_vals[:, 0].max()</c>: NumPy's max propagates a NaN (QA-53:
    ///     overflowing objective values leave NaNs after standardization, and Optuna's line search then fails). The loop
    ///     used until 2026-11-11 skipped NaNs.
    /// </summary>
    [TestMethod]
    public void GpColumnMax_ColumnWithNaN_IsNaNLikeNumPysMax()
    {
        double[,] scores = { { 1.0, 9.0 }, { double.NaN, 8.0 }, { 3.0, 7.0 } };
        MethodInfo columnMax = typeof(GPSampler).GetMethod("ColumnMax", BindingFlags.NonPublic | BindingFlags.Static)!;

        ((double)columnMax.Invoke(null, new object[] { scores, 0 })!).Should().Be(double.NaN);
        ((double)columnMax.Invoke(null, new object[] { scores, 1 })!).Should().Be(9.0);
    }

    /// <summary>
    ///     The regret bound's <c>np.max(lcb_acqf.eval_acqf_no_grad(...))</c> and <c>.max()</c> of the UCB values: NumPy's
    ///     max propagates a NaN acquisition value (QA-53's class); the loop used until 2026-11-11 skipped it.
    /// </summary>
    [TestMethod]
    public void RegretBoundMax_ValuesWithNaN_IsNaNLikeNumPysMax()
    {
        MethodInfo max = typeof(OptunaSharp.Terminator.RegretBoundEvaluator)
            .GetMethod("Max", BindingFlags.NonPublic | BindingFlags.Static)!;

        ((double)max.Invoke(null, new object[] { new[] { 0.5, double.NaN, 2.0 } })!).Should().Be(double.NaN);
        ((double)max.Invoke(null, new object[] { new[] { 0.5, 2.0, -1.0 } })!).Should().Be(2.0);
    }

    /// <summary>
    ///     <see cref="OptunaSharp.Internal.MathUtil.Clip" /> is NumPy's const clip loop (scalar bounds: a value on a bound
    ///     keeps its own zero sign, a NaN bound is the result) and <see cref="OptunaSharp.Internal.MathUtil.ClipGeneral" />
    ///     its general loop (array bounds: a value on a bound becomes the bound). Each expected value is NumPy 2.4.2's,
    ///     and NumSharp's <c>np.clip</c> must give the same on the matching layout: a scalar call for the const loop, an
    ///     (n, 2) block against per-column bounds for the general loop. QA-30 is the const half: a stepped float with
    ///     the upper bound <c>-0.0</c>.
    /// </summary>
    /// <param name="value">The clipped value.</param>
    /// <param name="low">The lower bound.</param>
    /// <param name="high">The upper bound.</param>
    /// <param name="constBits">NumPy's <c>np.clip(value, low, high)</c> on scalars, as IEEE bits.</param>
    /// <param name="generalBits">NumPy's result for the same element in an (n, 2) block with per-column bounds, as IEEE bits.</param>
    [TestMethod]
    [DataRow(-0.0, 0.0, 1.0, long.MinValue, 0L)]
    [DataRow(0.0, -0.0, 1.0, 0L, long.MinValue)]
    [DataRow(0.0, -1.0, -0.0, 0L, long.MinValue)]
    [DataRow(-0.0, -1.0, 0.0, long.MinValue, 0L)]
    [DataRow(0.5, 0.0, 1.0, 0x3FE0000000000000L, 0x3FE0000000000000L)]
    [DataRow(2.0, 0.0, 1.0, 0x3FF0000000000000L, 0x3FF0000000000000L)]
    [DataRow(-3.0, 0.0, 1.0, 0L, 0L)]
    public void MathUtilClip_ConstAndGeneralLoops_MatchNumPyAndNumSharp(
        double value, double low, double high, long constBits, long generalBits)
    {
        BitConverter.DoubleToInt64Bits(OptunaSharp.Internal.MathUtil.Clip(value, low, high))
            .Should().Be(constBits, "NumPy's const loop (scalar bounds)");
        BitConverter.DoubleToInt64Bits(OptunaSharp.Internal.MathUtil.ClipGeneral(value, low, high))
            .Should().Be(generalBits, "NumPy's general loop (array bounds)");

        using (NumSharp.NDArray scalar = NumSharp.NDArray.Scalar(value))
        using (NumSharp.NDArray lo = NumSharp.NDArray.Scalar(low))
        using (NumSharp.NDArray hi = NumSharp.NDArray.Scalar(high))
        using (NumSharp.NDArray clipped = NumSharp.np.clip(scalar, lo, hi))
            BitConverter.DoubleToInt64Bits((double)clipped).Should().Be(constBits, "NumSharp's np.clip on scalars");

        // The element sits in column 0 of a (3, 2) block; column 1 is an unrelated parameter with bounds [0, 1].
        using (NumSharp.NDArray block = NumSharp.np.array(new[,] { { value, 0.3 }, { 0.5, 0.2 }, { 0.25, 0.1 } }))
        using (NumSharp.NDArray lows = NumSharp.np.array(new[] { low, 0.0 }))
        using (NumSharp.NDArray highs = NumSharp.np.array(new[] { high, 1.0 }))
        using (NumSharp.NDArray clipped = NumSharp.np.clip(block, lows, highs))
            BitConverter.DoubleToInt64Bits(clipped.GetDouble(0, 0)).Should().Be(generalBits, "NumSharp's np.clip with array bounds");
    }

    /// <summary>
    ///     <see cref="OptunaSharp.Internal.MathUtil.IsClose" /> is NumPy's <c>np.isclose</c> for two floats: each case is
    ///     NumPy 2.4.2's answer, and NumSharp's <c>np.isclose</c> (which the helper does not call, for speed) must agree
    ///     on it too. The infinite rows are QA-50's: <c>enqueue_trial(skip_if_exists=True)</c> calls
    ///     <c>np.isclose(new, existing, atol=0.0, equal_nan=True)</c>, whose <c>isfinite(b)</c> and <c>a == b</c> terms the
    ///     former bare inequality lacked (it called a repeated <c>inf</c> new and a finite value over an <c>inf</c> a
    ///     repeat).
    /// </summary>
    /// <param name="a">NumPy's <c>a</c>.</param>
    /// <param name="b">NumPy's <c>b</c>.</param>
    /// <param name="atol">The absolute tolerance (NumPy's default <c>1e-8</c>, or the enqueue check's <c>0.0</c>).</param>
    /// <param name="equalNan">Whether two NaNs are close.</param>
    /// <param name="expected">NumPy's <c>np.isclose(a, b, atol=atol, equal_nan=equal_nan)</c>.</param>
    [TestMethod]
    [DataRow(double.PositiveInfinity, double.PositiveInfinity, 0.0, true, true)]
    [DataRow(5.0, double.PositiveInfinity, 0.0, true, false)]
    [DataRow(double.NegativeInfinity, double.PositiveInfinity, 0.0, true, false)]
    [DataRow(double.PositiveInfinity, double.NegativeInfinity, 0.0, true, false)]
    [DataRow(double.NaN, double.NaN, 0.0, true, true)]
    [DataRow(double.NaN, double.NaN, 1e-8, false, false)]
    [DataRow(1.0, 1.00001, 1e-8, false, true)]
    [DataRow(1.0, 1.0000101, 1e-8, false, false)]
    [DataRow(0.0, -0.0, 1e-8, false, true)]
    [DataRow(1e-8, 0.0, 1e-8, false, true)]
    [DataRow(0.0, 1e-8, 1e-8, false, true)]
    [DataRow(2e-8, 0.0, 1e-8, false, false)]
    public void MathUtilIsClose_EdgeCases_MatchNumPyAndNumSharp(double a, double b, double atol, bool equalNan, bool expected)
    {
        OptunaSharp.Internal.MathUtil.IsClose(a, b, atol: atol, equalNan: equalNan).Should().Be(expected, "NumPy's np.isclose");

        using NumSharp.NDArray left = NumSharp.NDArray.Scalar(a);
        using NumSharp.NDArray right = NumSharp.NDArray.Scalar(b);
        using NumSharp.NDArray close = NumSharp.np.isclose(left, right, atol: atol, equal_nan: equalNan);
        ((bool)close).Should().Be(expected, "NumSharp's np.isclose");
    }
}
