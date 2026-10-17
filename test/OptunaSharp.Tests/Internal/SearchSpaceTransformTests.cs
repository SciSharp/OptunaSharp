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
}
