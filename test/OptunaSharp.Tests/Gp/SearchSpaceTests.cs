using System.Collections.Generic;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Gp;

namespace OptunaSharp.Tests.Gp;

/// <summary>
///     Unit coverage for the GP-internal <see cref="GpSearchSpace" /> (PORTING_PLAN Phase 3.2 part 3 —
///     the normalize/unnormalize data layer of <c>optuna/_gp/search_space.py</c>). The transforms are
///     pure numpy in Optuna and reproduced bit-for-bit here; values are pinned to figures produced by
///     live Optuna. (Byte-exact parity over random spaces lives in the interop
///     <c>GpSearchSpaceParityTests</c>.)
/// </summary>
[TestClass]
public class SearchSpaceTests
{
    private static GpSearchSpace MixedSpace() => new(new Dictionary<string, BaseDistribution>
    {
        ["f"] = new FloatDistribution(-3.0, 5.0),
        ["lf"] = new FloatDistribution(1e-3, 10.0, log: true),
        ["i"] = new IntDistribution(0, 9),
        ["df"] = new FloatDistribution(0.0, 1.0, step: 0.25),
        ["c"] = new CategoricalDistribution((CategoricalChoice)"a", (CategoricalChoice)"b", (CategoricalChoice)"c"),
    });

    [TestMethod]
    public void Construction_ScaleTypesStepsIndices_MatchOptuna()
    {
        var s = MixedSpace();
        s.ScaleTypes.Should().Equal(0L, 1L, 0L, 0L, 2L);
        s.Steps.Should().Equal(0.0, 0.0, 1.0, 0.25, 1.0);
        s.IsCategorical.Should().Equal(false, false, false, false, true);
        s.DiscreteIndices.Should().Equal(2, 3, 4);
        s.ContinuousIndices.Should().Equal(0, 1);
        s.Dim.Should().Be(5);
    }

    [TestMethod]
    public void GetChoicesOfDiscreteParams_MatchOptuna()
    {
        List<double[]> choices = MixedSpace().GetChoicesOfDiscreteParams();
        choices.Should().HaveCount(3);
        choices[0].Should().Equal(0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95);  // int 0..9
        choices[1].Should().Equal(0.1, 0.3, 0.5, 0.7, 0.9);   // discretized float step 0.25
        choices[2].Should().Equal(0.0, 1.0, 2.0);              // categorical codes
    }

    [TestMethod]
    public void NormalizeOneParam_LinearAndLog_MatchOptuna()
    {
        GpSearchSpace.NormalizeOneParam(2.0, GpSearchSpace.ScaleType.Linear, (-3.0, 5.0), 0.0)
            .Should().Be(0.625);
        GpSearchSpace.NormalizeOneParam(0.1, GpSearchSpace.ScaleType.Log, (1e-3, 10.0), 0.0)
            .Should().Be(0.5);
        // Single-value dimension normalizes to 0.5.
        GpSearchSpace.NormalizeOneParam(4.0, GpSearchSpace.ScaleType.Linear, (4.0, 4.0), 0.0)
            .Should().Be(0.5);
    }

    [TestMethod]
    public void UnnormalizeOneParam_LogAndInt_MatchOptuna()
    {
        GpSearchSpace.UnnormalizeOneParam(0.5, GpSearchSpace.ScaleType.Log, (1e-3, 10.0), 0.0)
            .Should().BeApproximately(0.10000000000000006, 1e-15);
        GpSearchSpace.UnnormalizeOneParam(0.7, GpSearchSpace.ScaleType.Linear, (0.0, 9.0), 1.0)
            .Should().Be(6.5);
    }

    [TestMethod]
    public void GetUnnormalizedParam_MatchesOptuna()
    {
        var un = MixedSpace().GetUnnormalizedParam(new[] { 0.4, 0.6, 0.55, 0.3, 1.0 });
        ((double)un["f"]!).Should().BeApproximately(0.20000000000000018, 1e-15);
        ((double)un["lf"]!).Should().BeApproximately(0.2511886431509581, 1e-14);
        un["i"].Should().Be(5L);          // int, rounded
        ((double)un["df"]!).Should().BeApproximately(0.25, 1e-15);
        un["c"].Should().Be("b");         // categorical external value
    }

    [TestMethod]
    public void RoundOneNormalizedParam_SnapsToGrid()
    {
        // A value near the 0.5 grid point of a step-0.25 [0,1] float rounds onto it.
        double rounded = GpSearchSpace.RoundOneNormalizedParam(0.52, GpSearchSpace.ScaleType.Linear, (0.0, 1.0), 0.25);
        // 0.52 -> unnormalize -> ~0.525 -> snap to 0.5 -> normalize -> 0.5.
        rounded.Should().Be(0.5);
    }
}
