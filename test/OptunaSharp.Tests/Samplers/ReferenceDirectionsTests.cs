using System;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Moead;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Unit tests for the shared Das-Dennis <see cref="ReferenceDirections" /> helper (weight/reference
///     vectors for <see cref="MoeaDSampler" /> and <see cref="RveaSampler" />) and the MOEA/D
///     <see cref="MoeadDecomposition" /> scalarizing functions. These are deterministic math checks (no
///     RNG), verified against the closed-form Das-Dennis lattice and the textbook decomposition formulas.
/// </summary>
[TestClass]
public class ReferenceDirectionsTests
{
    // ---- Das-Dennis lattice ----------------------------------------------------------------------

    [TestMethod]
    public void DasDennis_TwoObjectives_TwoPartitions_IsTheThreePointSimplex()
    {
        var dirs = ReferenceDirections.DasDennis(nPartitions: 2, nDim: 2);
        dirs.Length.Should().Be(3);
        dirs[0].Should().Equal(0.0, 1.0);
        dirs[1].Should().Equal(0.5, 0.5);
        dirs[2].Should().Equal(1.0, 0.0);
    }

    [TestMethod]
    public void DasDennis_RowsAreNonNegativeAndSumToOne()
    {
        foreach (var (m, h) in new[] { (2, 12), (3, 6), (4, 4) })
        {
            var dirs = ReferenceDirections.DasDennis(h, m);
            dirs.Length.Should().Be((int)ReferenceDirections.NumberOfUniformPoints(h, m));
            foreach (var row in dirs)
            {
                row.Length.Should().Be(m);
                row.Should().OnlyContain(v => v >= 0.0);
                row.Sum().Should().BeApproximately(1.0, 1e-12);
            }
        }
    }

    [TestMethod]
    public void DasDennis_ZeroPartitions_IsTheCentroid()
    {
        var dirs = ReferenceDirections.DasDennis(nPartitions: 0, nDim: 3);
        dirs.Length.Should().Be(1);
        dirs[0].Should().Equal(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }

    [TestMethod]
    public void NumberOfUniformPoints_MatchesTheBinomialFormula()
    {
        // C(m + H - 1, H): 2 objectives -> H+1; 3 objectives / H=12 -> C(14,12) = 91 (pymoo's default).
        ReferenceDirections.NumberOfUniformPoints(99, 2).Should().Be(100);
        ReferenceDirections.NumberOfUniformPoints(12, 3).Should().Be(91);
        ReferenceDirections.NumberOfUniformPoints(2, 3).Should().Be(6);
    }

    [TestMethod]
    public void PartitionClosestToPoints_PicksTheLargestLatticeAtOrBelowTheTarget()
    {
        // 2 objectives: N = H + 1, so a target of 50 gives H = 49 (exactly 50 points).
        int h2 = ReferenceDirections.PartitionClosestToPoints(50, 2);
        h2.Should().Be(49);
        ReferenceDirections.NumberOfUniformPoints(h2, 2).Should().BeLessThanOrEqualTo(50);

        // 3 objectives: H = 8 -> 45 points (H = 9 would overshoot to 55).
        int h3 = ReferenceDirections.PartitionClosestToPoints(50, 3);
        h3.Should().Be(8);
        ReferenceDirections.NumberOfUniformPoints(h3, 3).Should().BeLessThanOrEqualTo(50);
        ReferenceDirections.NumberOfUniformPoints(h3 + 1, 3).Should().BeGreaterThan(50);
    }

    // ---- MOEA/D scalarizing functions ------------------------------------------------------------

    [TestMethod]
    public void Tchebycheff_IsTheWeightedMaxAbsoluteDeviationFromTheIdeal()
    {
        var f = new[] { 1.0, 3.0 };
        var ideal = new[] { 0.0, 0.0 };

        // Boundary weight [1,0] drops the 2nd objective -> only |f0 - z0| counts.
        MoeadDecomposition.Scalarize(MoeadScalarization.Tchebycheff, f, new[] { 1.0, 0.0 }, ideal)
            .Should().BeApproximately(1.0, 1e-12);
        // Even weight [0.5,0.5] -> max(0.5*1, 0.5*3) = 1.5.
        MoeadDecomposition.Scalarize(MoeadScalarization.Tchebycheff, f, new[] { 0.5, 0.5 }, ideal)
            .Should().BeApproximately(1.5, 1e-12);
    }

    [TestMethod]
    public void WeightedSum_IsTheWeightedSumOfDeviations()
    {
        var f = new[] { 1.0, 3.0 };
        var ideal = new[] { 0.0, 0.0 };
        MoeadDecomposition.Scalarize(MoeadScalarization.WeightedSum, f, new[] { 0.5, 0.5 }, ideal)
            .Should().BeApproximately(2.0, 1e-12);
    }

    [TestMethod]
    public void Pbi_OnTheWeightDirection_HasNoPerpendicularPenalty()
    {
        // A point exactly along the (normalized) weight direction has d2 = 0, so PBI = d1 = its distance
        // from the ideal along that direction.
        var ideal = new[] { 0.0, 0.0 };
        var weight = new[] { 1.0, 1.0 };
        double norm = Math.Sqrt(2.0);
        var f = new[] { 2.0 / norm, 2.0 / norm };   // distance 2 along the unit weight direction

        MoeadDecomposition.Scalarize(MoeadScalarization.Pbi, f, weight, ideal)
            .Should().BeApproximately(2.0, 1e-9);

        // A point off the direction incurs the theta-weighted perpendicular penalty, so it scores worse.
        var offAxis = new[] { 2.0, 0.0 };
        MoeadDecomposition.Scalarize(MoeadScalarization.Pbi, offAxis, weight, ideal)
            .Should().BeGreaterThan(MoeadDecomposition.Scalarize(MoeadScalarization.Pbi, f, weight, ideal));
    }
}
