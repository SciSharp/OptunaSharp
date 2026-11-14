using System;
using System.Reflection;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Importance.Forests.Internal;
using OptunaSharp.Visualization;

namespace OptunaSharp.Tests.Internal;

/// <summary>
///     Pins the port's three hand-written sequential products to NumSharp's <c>np.prod</c>, bit for bit. Optuna computes
///     them with <c>np.prod</c>, which NumPy runs as a sequential <c>multiply.reduce</c>; NumSharp 0.81.0's float
///     <c>np.prod</c> multiplies in that same order (0.80.0's SIMD lanes did not, which is why the port wrote loops). The
///     loops stay, because each runs once per point inside the hypervolume and fANOVA loops, where a NumSharp call (about
///     2 µs) costs more than the multiplications; these tests are what lets them stay: if a NumSharp release ever reorders
///     <c>np.prod</c> again, they fail and name the helper that no longer equals it.
/// </summary>
[TestClass]
public class SequentialProductTests
{
    /// <summary>The private static helpers are reached by reflection, as the reduction-order pins reach theirs.</summary>
    private const BindingFlags PrivateStatic = BindingFlags.NonPublic | BindingFlags.Static;

    /// <summary>
    ///     Builds a row of objective-like values: magnitudes from 1e-3 to 1e3 with both signs, so the products round in
    ///     their last bits and an order change would show.
    /// </summary>
    /// <param name="rng">The generator to draw from.</param>
    /// <param name="length">The number of values.</param>
    /// <returns>The row.</returns>
    private static double[] Row(NumPyRandom rng, int length)
    {
        using NDArray exponents = rng.uniform(-3.0, 3.0, new Shape(length));
        using NDArray signs = rng.uniform(0.0, 1.0, new Shape(length));
        var row = new double[length];
        for (int i = 0; i < length; i++)
            row[i] = (signs.GetDouble(i) < 0.5 ? -1.0 : 1.0) * Math.Pow(10.0, exponents.GetDouble(i));
        return row;
    }

    /// <summary>NumSharp's <c>np.prod(a - b)</c> over two rows of the same length, as one double.</summary>
    /// <param name="a">The minuend row.</param>
    /// <param name="b">The subtrahend row.</param>
    /// <returns>The product of the differences.</returns>
    private static double NpProdOfDifference(double[] a, double[] b)
    {
        using NDArray left = np.array(a);
        using NDArray right = np.array(b);
        using NDArray difference = left - right;
        using NDArray product = np.prod(difference);
        return (double)product;
    }

    /// <summary>Asserts two doubles are the same value: equal bits, or both NaN (whose payload no Python caller sees).</summary>
    /// <param name="actual">The helper's result.</param>
    /// <param name="expected">NumSharp's result.</param>
    /// <param name="what">The helper and case, for the failure message.</param>
    private static void ShouldBeSameValue(double actual, double expected, string what)
    {
        if (double.IsNaN(expected))
        {
            double.IsNaN(actual).Should().BeTrue($"{what}: NumSharp's np.prod is NaN, the helper gave {actual:R}");
            return;
        }
        BitConverter.DoubleToInt64Bits(actual).Should().Be(BitConverter.DoubleToInt64Bits(expected),
            $"{what}: NumSharp's np.prod is {expected:R}, the helper gave {actual:R}");
    }

    /// <summary>
    ///     <c>Hypervolume.ProdRefMinus</c> (<c>np.prod(reference_point - row)</c> in Optuna's hypervolume and HSSP) and
    ///     <c>HypervolumeHistory.Prod</c> (<c>np.prod(ref - values)</c> in <c>_get_hypervolume_history</c>) equal NumSharp's
    ///     <c>np.prod</c> over 2 to 16 objectives, 400 random rows, plus rows holding an infinity, a zero and a signed zero.
    /// </summary>
    [TestMethod]
    public void HypervolumeProducts_EqualNumSharpsNpProd()
    {
        MethodInfo prodRefMinus = typeof(OptunaSharp.Internal.Hypervolume).GetMethod("ProdRefMinus", PrivateStatic)!;
        MethodInfo historyProd = typeof(HypervolumeHistory).GetMethod("Prod", PrivateStatic)!;
        NumPyRandom rng = np.random.RandomState(20261114);

        void Check(double[] reference, double[] row, string label)
        {
            double expected = NpProdOfDifference(reference, row);
            ShouldBeSameValue((double)prodRefMinus.Invoke(null, new object[] { reference, row })!, expected, $"ProdRefMinus {label}");
            ShouldBeSameValue((double)historyProd.Invoke(null, new object[] { reference, row })!, expected, $"HypervolumeHistory.Prod {label}");
        }

        for (int i = 0; i < 400; i++)
        {
            int m = 2 + i % 15;
            Check(Row(rng, m), Row(rng, m), $"random row {i} (m={m})");
        }
        Check(new[] { 1.0, 2.0, 3.0 }, new[] { 1.0, double.NegativeInfinity, 0.5 }, "zero times infinity");
        Check(new[] { 1.0, 2.0, 3.0, 4.0 }, new[] { 0.5, double.NegativeInfinity, 0.5, 1.0 }, "one infinite distance");
        Check(new[] { -0.0, 2.0, 3.0 }, new[] { 0.0, 1.0, 2.5 }, "a signed zero distance");
    }

    /// <summary>
    ///     <c>FanovaTree.GetCardinality</c> (<c>_get_cardinality(search_spaces) = np.prod(hi - lo)</c>) equals NumSharp's
    ///     <c>np.prod</c> over 1 to 16 features, 400 random bound pairs.
    /// </summary>
    [TestMethod]
    public void FanovaCardinality_EqualsNumSharpsNpProd()
    {
        MethodInfo cardinality = typeof(FanovaTree).GetMethod("GetCardinality", PrivateStatic)!;
        NumPyRandom rng = np.random.RandomState(1114);

        for (int i = 0; i < 400; i++)
        {
            int features = 1 + i % 16;
            double[] lo = Row(rng, features);
            double[] hi = Row(rng, features);
            var searchSpaces = new double[features, 2];
            for (int f = 0; f < features; f++)
            {
                searchSpaces[f, 0] = lo[f];
                searchSpaces[f, 1] = hi[f];
            }
            ShouldBeSameValue((double)cardinality.Invoke(null, new object[] { searchSpaces })!, NpProdOfDifference(hi, lo),
                $"GetCardinality row {i} (features={features})");
        }
    }
}
