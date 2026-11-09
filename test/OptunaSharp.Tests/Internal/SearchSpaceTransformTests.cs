using System;
using System.Collections.Generic;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Internal;

/// <summary>
///     Pins <see cref="SearchSpaceTransform" /> (Python <c>_SearchSpaceTransform</c>) to live
///     Optuna output for a mixed search space — bounds layout, transform values and the
///     untransform round trip below are copied from a real optuna run.
/// </summary>
[TestClass]
public class SearchSpaceTransformTests
{
    private static OrderedDict<string, BaseDistribution> MixedSpace() => new()
    {
        ["f"] = new FloatDistribution(-5.0, 5.0),
        ["fl"] = new FloatDistribution(1e-3, 10.0, log: true),
        ["fs"] = new FloatDistribution(0.0, 1.0, step: 0.25),
        ["i"] = new IntDistribution(1, 20),
        ["il"] = new IntDistribution(1, 64, log: true),
        ["c"] = new CategoricalDistribution(new CategoricalChoice[] { "u", "v", "w" }),
    };

    [TestMethod]
    public void Bounds_Match_Python_Exactly()
    {
        var trans = new SearchSpaceTransform(MixedSpace());
        var bounds = trans.RawBounds;

        // Pinned from optuna._transform._SearchSpaceTransform(sp).bounds.tolist().
        double[][] expected =
        {
            new[] { -5.0, 5.0 },
            new[] { -6.907755278982137, 2.302585092994046 },
            new[] { -0.125, 1.125 },
            new[] { 0.5, 20.5 },
            new[] { -0.6931471805599453, 4.1666652238017265 },
            new[] { 0.0, 1.0 },
            new[] { 0.0, 1.0 },
            new[] { 0.0, 1.0 },
        };
        bounds.GetLength(0).Should().Be(8);
        for (int r = 0; r < 8; r++)
        {
            bounds[r, 0].Should().Be(expected[r][0], $"row {r} low");
            bounds[r, 1].Should().Be(expected[r][1], $"row {r} high");
        }
    }

    [TestMethod]
    public void Transform_Then_Untransform_Match_Python()
    {
        var trans = new SearchSpaceTransform(MixedSpace());
        var @params = new OrderedDict<string, object?>
        {
            ["f"] = 2.5,
            ["fl"] = 0.05,
            ["fs"] = 0.75,
            ["i"] = 7L,
            ["il"] = 32L,
            ["c"] = "v",
        };
        var transParams = trans.Transform(@params).ToArray<double>();
        // Pinned from Python: [2.5, -2.995732273553991, 0.75, 7.0, 3.4657359027997265, 0.0, 1.0, 0.0].
        transParams.Should().Equal(
            2.5, -2.995732273553991, 0.75, 7.0, 3.4657359027997265, 0.0, 1.0, 0.0);

        var back = trans.Untransform(NumSharp.np.array(transParams));
        back["f"].Should().Be(2.5);
        back["fl"].Should().Be(0.05000000000000001, "the exp/log round trip Python produces");
        back["fs"].Should().Be(0.75);
        back["i"].Should().Be(7L);
        back["il"].Should().Be(32L);
        back["c"].Should().Be("v");
    }

    [TestMethod]
    public void ColumnMappings_Cover_OneHot_Choices()
    {
        var trans = new SearchSpaceTransform(MixedSpace());
        trans.ColumnToEncodedColumns.Should().HaveCount(6);
        trans.ColumnToEncodedColumns[5].Should().Equal(5L, 6L, 7L);
        trans.EncodedColumnToColumn.Should().Equal(0L, 1L, 2L, 3L, 4L, 5L, 5L, 5L);
    }

    [TestMethod]
    public void Untransform_Clamps_Float_Below_High()
    {
        var space = new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(0.0, 1.0) };
        var trans = new SearchSpaceTransform(space);
        // Values at/above high clamp to nextafter(high, high-1) — high itself is unreachable.
        var back = trans.Untransform(NumSharp.np.array(new[] { 1.0 }));
        ((double)back["x"]!).Should().Be(MathUtil.NextAfter(1.0, 0.0));
    }

    /// <summary>
    ///     Optuna 5.0.0 (#6799) clips a log float to <c>[low, nextafter(high, high - 1)]</c> on BOTH ends.
    ///     <c>exp(log(1e-7))</c> rounds to <c>9.999999999999994e-08</c>, so before the fix a coordinate on
    ///     the lower edge of the transformed box came back one ULP below <c>low</c> — outside the
    ///     distribution the value is then suggested from.
    /// </summary>
    [TestMethod]
    public void Untransform_Clamps_LogFloat_To_Both_Bounds()
    {
        var distribution = new FloatDistribution(1e-7, 1.0, log: true);
        var space = new OrderedDict<string, BaseDistribution> { ["x"] = distribution };
        var trans = new SearchSpaceTransform(space);

        // Guard: the exp/log round trip genuinely undershoots this bound, so the assertion below keeps
        // exercising the low clamp rather than passing because the round trip happened to be exact.
        Math.Exp(Math.Log(1e-7)).Should().BeLessThan(1e-7);

        ((double)trans.Untransform(NumSharp.np.array(new[] { Math.Log(1e-7) }))["x"]!)
            .Should().Be(1e-7, "the lower edge clamps to low exactly");
        ((double)trans.Untransform(NumSharp.np.array(new[] { Math.Log(1.0) }))["x"]!)
            .Should().Be(MathUtil.NextAfter(1.0, 0.0), "high itself stays unreachable, as before");
        double.IsNaN((double)SearchSpaceTransform.UntransformNumericalParam(double.NaN, distribution, transformLog: true))
            .Should().BeTrue("np.clip propagates NaN instead of clamping it");
    }

    [TestMethod]
    public void Untransform_Int_Uses_HalfUp_Rounding()
    {
        var space = new OrderedDict<string, BaseDistribution> { ["n"] = new IntDistribution(0, 10) };
        var trans = new SearchSpaceTransform(space);
        // Python: floor((t - low)/step + 0.5) — half rounds UP (unlike banker's).
        trans.Untransform(NumSharp.np.array(new[] { 2.5 }))["n"].Should().Be(3L);
        trans.Untransform(NumSharp.np.array(new[] { 3.5 }))["n"].Should().Be(4L);
        trans.Untransform(NumSharp.np.array(new[] { -0.4 }))["n"].Should().Be(0L, "clamped at low");
    }

    [TestMethod]
    public void Transform01_Maps_To_Unit_Cube_And_Back()
    {
        var space = new OrderedDict<string, BaseDistribution>
        {
            ["x"] = new FloatDistribution(-10.0, 10.0),
            ["single"] = new FloatDistribution(3.0, 3.0),
        };
        var trans = new SearchSpaceTransform(space, transform01: true);
        var transParams = trans.Transform(new OrderedDict<string, object?> { ["x"] = 5.0, ["single"] = 3.0 })
            .ToArray<double>();
        transParams[0].Should().Be(0.75);
        transParams[1].Should().Be(0.5, "single-valued bounds map to 0.5");

        var back = trans.Untransform(NumSharp.np.array(new[] { 0.75, 0.5 }));
        back["x"].Should().Be(5.0);
        back["single"].Should().Be(3.0);
    }

    [TestMethod]
    public void Categorical_Untransform_Picks_First_ArgMax()
    {
        var space = new OrderedDict<string, BaseDistribution>
        {
            ["c"] = new CategoricalDistribution(new CategoricalChoice[] { "a", "b", "c" }),
        };
        var trans = new SearchSpaceTransform(space);
        trans.Untransform(NumSharp.np.array(new[] { 0.4, 0.4, 0.1 }))["c"]
            .Should().Be("a", "np.argmax returns the first maximum");
    }

    /// <summary>
    ///     Plan 14: <see cref="SearchSpaceTransform.TransformValues" /> returns exactly the values <see cref="SearchSpaceTransform.Transform" />
    ///     wraps in a NumSharp array, for the mixed space, with and without the unit-cube rescaling, over many random
    ///     configurations (one-hot categorical columns included).
    /// </summary>
    /// <param name="transform01">Whether the transform rescales numerical columns to the unit cube.</param>
    [TestMethod]
    [DataRow(false)]
    [DataRow(true)]
    public void TransformValues_Equal_Transform_BitForBit(bool transform01)
    {
        var space = MixedSpace();
        var trans = new SearchSpaceTransform(space, transform01: transform01);
        var rng = new Random(7);
        string[] choices = { "u", "v", "w" };
        for (int round = 0; round < 200; round++)
        {
            var config = new OrderedDict<string, object?>
            {
                ["f"] = -5.0 + 10.0 * rng.NextDouble(),
                ["fl"] = Math.Exp(Math.Log(1e-3) + (Math.Log(10.0) - Math.Log(1e-3)) * rng.NextDouble()),
                ["fs"] = 0.25 * rng.Next(0, 5),
                ["i"] = (long)rng.Next(1, 21),
                ["il"] = (long)rng.Next(1, 65),
                ["c"] = choices[rng.Next(0, 3)],
            };
            using NumSharp.NDArray expected = trans.Transform(config);
            double[] want = expected.ToArray<double>();
            double[] actual = trans.TransformValues(config);
            actual.Should().HaveCount(want.Length);
            for (int i = 0; i < want.Length; i++)
                BitConverter.DoubleToInt64Bits(actual[i]).Should().Be(BitConverter.DoubleToInt64Bits(want[i]), $"column {i}, round {round}");
        }
    }

    /// <summary>
    ///     Plan 14: <see cref="SearchSpaceTransform.UntransformValues" /> returns exactly what <see cref="SearchSpaceTransform.Untransform(NumSharp.NDArray)" />
    ///     returns for the same float64 values (same keys in the same order, same values and types), with and without the
    ///     unit-cube rescaling, including coordinates outside the bounds that the untransform clamps and one-hot ties.
    /// </summary>
    /// <param name="transform01">Whether the transform rescales numerical columns to the unit cube.</param>
    [TestMethod]
    [DataRow(false)]
    [DataRow(true)]
    public void UntransformValues_Equal_Untransform_ForTheSameValues(bool transform01)
    {
        var trans = new SearchSpaceTransform(MixedSpace(), transform01: transform01);
        double[,] bounds = trans.RawBounds;
        int n = bounds.GetLength(0);
        var rng = new Random(11);
        for (int round = 0; round < 300; round++)
        {
            var values = new double[n];
            for (int i = 0; i < n; i++)
            {
                // Mostly inside the box, sometimes a little outside (the clamping branches), sometimes a tie (argmax).
                double low = transform01 ? 0.0 : bounds[i, 0];
                double high = transform01 ? 1.0 : bounds[i, 1];
                values[i] = round % 7 == 0 ? 0.5 * (low + high) : low + (high - low) * (1.2 * rng.NextDouble() - 0.1);
            }
            using NumSharp.NDArray array = NumSharp.np.array(values);
            var expected = trans.Untransform(array);
            var actual = trans.UntransformValues(values);
            actual.Keys.Should().Equal(expected.Keys);
            foreach (var (name, value) in expected)
            {
                actual[name]!.GetType().Should().Be(value!.GetType(), $"{name}, round {round}");
                if (value is double d)
                    BitConverter.DoubleToInt64Bits((double)actual[name]!).Should().Be(BitConverter.DoubleToInt64Bits(d), $"{name}, round {round}");
                else
                    actual[name].Should().Be(value, $"{name}, round {round}");
            }
        }
    }

    /// <summary>
    ///     Plan 14: the NumSharp overload still reads any numeric dtype through <c>astype(float64)</c> (now disposing that copy):
    ///     a float32 and an int64 input untransform as their float64 conversions do, and the caller's array stays usable.
    /// </summary>
    [TestMethod]
    public void Untransform_NonFloat64Input_IsReadThroughAFloat64Copy_AndTheInputSurvives()
    {
        var space = new OrderedDict<string, BaseDistribution>
        {
            ["x"] = new FloatDistribution(-10.0, 10.0),
            ["n"] = new IntDistribution(0, 9),
        };
        var trans = new SearchSpaceTransform(space);
        using NumSharp.NDArray asFloat32 = NumSharp.np.array(new[] { 2.5f, 3.4f });
        var fromFloat32 = trans.Untransform(asFloat32);
        fromFloat32["x"].Should().Be((double)2.5f);
        fromFloat32["n"].Should().Be(3L);
        asFloat32.ToArray<float>().Should().Equal(2.5f, 3.4f);

        using NumSharp.NDArray asInt64 = NumSharp.np.array(new[] { 4L, 7L });
        var fromInt64 = trans.Untransform(asInt64);
        fromInt64["x"].Should().Be(4.0);
        fromInt64["n"].Should().Be(7L);
        asInt64.ToArray<long>().Should().Equal(4L, 7L);
    }
}
