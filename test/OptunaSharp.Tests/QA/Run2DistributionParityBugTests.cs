using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Distribution parity findings of the 2026-11-10 parity QA run (<c>docs/qa/PARITY_QA_2026-11-10.md</c>). Optuna's
///     values are what Optuna 5.0.0 produces on CPython 3.12. A test marked <c>[OpenBugs]</c> fails until its divergence is
///     fixed; a test marked <c>[Misaligned]</c> pins a divergence the owner chose to keep.
/// </summary>
[TestClass]
public class Run2DistributionParityBugTests
{
    /// <summary>
    ///     QA-62 (the QA-3 root cause, beyond the range QA-3 describes): Python parses <c>str(x)</c> into a
    ///     <c>Decimal</c> exactly and computes in a 28-significant-digit context; .NET's <c>decimal</c> keeps at most 28
    ///     digits after the decimal point. A value whose shortest repr needs more places (a 17-digit double below about
    ///     <c>1e-11</c>, such as <c>4.1271208719049127e-14</c>) is rounded on the way in, so the stepped-high adjustment
    ///     and BruteForce's candidate grid come out differently, even though the value is well inside decimal's range.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Measured over 100,000 random stepped ranges after the QA-2 / QA-11 fix (<c>PyNumbers.DecimalToDouble</c>,
    ///         which made the final conversion correct): 1,439 adjusted <c>high</c>s differ by one ULP and 184 of 5,000
    ///         candidate grids differ, every one among the 3,021 inputs whose repr needs more than 28 decimal places.
    ///         Optuna: <c>FloatDistribution(4.1271208719049127e-14, 4.085121318968367e-13, step=1e-15).high</c> is
    ///         <c>4.0827120871904913e-13</c> (the port: one ULP below); <c>_enumerate_candidates(2.942974008882182e-14,
    ///         5.614297400888218e-13, 2e-15)</c> has 266 values ending at <c>5.594297400888218e-13</c> (the port: 267).
    ///     </para>
    ///     <para>
    ///         <b>Misaligned</b> by the owner's decision of 2026-11-12 (asked together with QA-3's question Q1): the port
    ///         keeps .NET <c>decimal</c>, and this test pins its values. The FloatDistribution and BruteForce docs state
    ///         the rounding.
    ///     </para>
    /// </remarks>
    [TestMethod]
    [Misaligned]
    public void SteppedHigh_ValuesBeyond28DecimalPlaces_RoundLikeDotNetDecimal()
    {
        var adjusted = new FloatDistribution(4.1271208719049127e-14, 4.085121318968367e-13, step: 1e-15);
        adjusted.High.Should().Be(4.082712087190491e-13, "Optuna gives 4.0827120871904913e-13, one ULP above");

        var grid = new FloatDistribution(2.942974008882182e-14, 5.615524728866719e-13, step: 2e-15);
        grid.High.Should().Be(5.614297400888218e-13);
        var candidates = BruteForceSampler.EnumerateCandidates(
            new BruteForceSampler.ChoicesArgs(grid.Low, grid.High, grid.Step, isInt: false));
        candidates.Should().HaveCount(267, "Optuna lists 266 values");
        candidates[^1].Should().Be(5.614297400888218e-13, "Optuna's last value is 5.594297400888218e-13");
    }
}
