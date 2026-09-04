using AwesomeAssertions;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Internal;

/// <summary>
///     Unit coverage for the scrambled Sobol' engine (PORTING_PLAN Phase 3.2 part 3c — the port of
///     <c>scipy.stats.qmc.Sobol(scramble=True)</c> that closes the 2.12 scrambling gap). Values are
///     pinned to figures produced by scipy 1.16.3; byte-exact parity over random d/n/seed lives in the
///     interop <c>GpSobolParityTests</c>.
/// </summary>
[TestClass]
public class ScrambledSobolTests
{
    [TestMethod]
    public void Random_MatchesScipy_D2Seed42()
    {
        double[,] p = ScrambledSobol.Random(2, 6, 42);
        var expected = new[,]
        {
            { 0.431029474362731, 0.8143679723143578 },
            { 0.9615787928923965, 0.31140994280576706 },
            { 0.5645319204777479, 0.5978343049064279 },
            { 0.0350473215803504, 0.026509271003305912 },
            { 0.23806727584451437, 0.6396976113319397 },
            { 0.6427621599286795, 0.23440346121788025 },
        };
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 2; j++)
                p[i, j].Should().Be(expected[i, j], $"scrambled Sobol' point [{i},{j}] is byte-identical to scipy");
    }

    [TestMethod]
    public void Random_FirstPointIsInUnitCube_AndDeterministic()
    {
        double[,] a = ScrambledSobol.Random(3, 5, 7);
        double[,] b = ScrambledSobol.Random(3, 5, 7);
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 3; j++)
            {
                a[i, j].Should().BeInRange(0.0, 1.0);
                a[i, j].Should().Be(b[i, j], "the same seed reproduces the same sequence");
            }
    }
}
