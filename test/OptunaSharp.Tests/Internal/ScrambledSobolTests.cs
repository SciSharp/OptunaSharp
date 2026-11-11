using System;
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

    /// <summary>
    ///     Plan 15: QMCSampler now computes the point at <c>sample_id</c> directly from the Gray code of the index
    ///     (<see cref="ScrambledSobol.Engine.Point" />) instead of generating every earlier point. Pins the direct form
    ///     to the stepping form it replaced, by bits, over every index of a long prefix and several dimensions and
    ///     seeds, including the indexes where the Gray code gains a new top bit.
    /// </summary>
    [TestMethod]
    public void EnginePoint_EqualsTheSteppedSequenceRowByRow()
    {
        foreach (var (d, seed) in new[] { (1, 0), (2, 42), (5, 7), (10, 0), (21, 123456) })
        {
            ScrambledSobol.Engine engine = ScrambledSobol.Create(d, seed);
            const int n = 4100;
            double[,] stepped = engine.Generate(n);
            for (int i = 0; i < n; i++)
            {
                double[] direct = engine.Point((ulong)i);
                for (int j = 0; j < d; j++)
                    BitConverter.DoubleToInt64Bits(direct[j]).Should().Be(BitConverter.DoubleToInt64Bits(stepped[i, j]),
                        $"d={d}, seed={seed}: point {i}, coordinate {j} is the same as stepping there");
            }
        }
    }

    /// <summary>
    ///     Plan 15: the engine is built once and reused, so building it twice must give the same sequence as
    ///     <see cref="ScrambledSobol.Random" />, which builds it per call.
    /// </summary>
    [TestMethod]
    public void Create_ThenGenerate_EqualsRandom()
    {
        double[,] viaRandom = ScrambledSobol.Random(4, 300, 99);
        double[,] viaEngine = ScrambledSobol.Create(4, 99).Generate(300);
        for (int i = 0; i < 300; i++)
            for (int j = 0; j < 4; j++)
                viaEngine[i, j].Should().Be(viaRandom[i, j]);
    }

    /// <summary>Plan 15: the direct form refuses an index past the 2^30 points of a 30-bit sequence, as stepping could not reach it.</summary>
    [TestMethod]
    public void EnginePoint_PastTheEndOfTheSequence_Throws()
    {
        ScrambledSobol.Engine engine = ScrambledSobol.Create(2, 0);
        Action past = () => engine.Point(1UL << QuasiMonteCarlo.SobolBits);
        past.Should().Throw<NumSharp.ValueError>();
        engine.Point((1UL << QuasiMonteCarlo.SobolBits) - 1).Should().HaveCount(2);
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
