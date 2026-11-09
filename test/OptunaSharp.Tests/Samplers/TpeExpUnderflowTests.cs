using System;
using AwesomeAssertions;
using OptunaSharp.Samplers.Tpe;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Plan 14: TPE's log-sum-exp stores <c>0.0</c> instead of calling <see cref="Math.Exp" /> for every term below
///     <see cref="MixtureOfProductDistribution.ExpIsZeroBelow" />. That is exact only if <see cref="Math.Exp" /> itself
///     returns <c>+0.0</c> (the bits, not merely a value equal to zero) for every such input on the platform running the
///     sampler; this pins it on every platform CI runs the unit suite on.
/// </summary>
[TestClass]
public class TpeExpUnderflowTests
{
    /// <summary>
    ///     Every input from the bound down to -1e6, densely near the bound and spread below it, plus -inf, gives
    ///     <see cref="Math.Exp" /> bits <c>0</c> (+0.0).
    /// </summary>
    [TestMethod]
    public void MathExp_IsPositiveZero_ForEveryInputBelowTheSkipBound()
    {
        double bound = MixtureOfProductDistribution.ExpIsZeroBelow;
        // The first 100,000 doubles below the bound, one ulp apart.
        double x = Math.BitDecrement(bound);
        for (int i = 0; i < 100_000; i++, x = Math.BitDecrement(x))
            BitConverter.DoubleToInt64Bits(Math.Exp(x)).Should().Be(0L, $"exp({x:R})");
        // A spread of inputs down to -1e6.
        var rng = new Random(746);
        for (int i = 0; i < 200_000; i++)
        {
            double y = bound - Math.Pow(10, 6 * rng.NextDouble());
            BitConverter.DoubleToInt64Bits(Math.Exp(y)).Should().Be(0L, $"exp({y:R})");
        }
        BitConverter.DoubleToInt64Bits(Math.Exp(double.NegativeInfinity)).Should().Be(0L);
        BitConverter.DoubleToInt64Bits(Math.Exp(double.MinValue)).Should().Be(0L);
    }

    /// <summary>
    ///     The bound sits below the underflow point, not above it: inputs just above -745.1332 still produce a nonzero
    ///     subnormal, and so must still go through <see cref="Math.Exp" />.
    /// </summary>
    [TestMethod]
    public void TheSkipBound_IsBelowTheUnderflowPoint()
    {
        Math.Exp(-745.13).Should().BeGreaterThan(0.0, "exp still has a subnormal result here");
        (MixtureOfProductDistribution.ExpIsZeroBelow < -745.14).Should().BeTrue();
    }
}
