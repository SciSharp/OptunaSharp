using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests;

/// <summary>
///     Unit coverage for the hypervolume calculator (<c>wfg.py</c>) and the greedy HSSP solver
///     (<c>hssp.py</c>) — PORTING_PLAN Phase 2.10. The structured-input assertions are ports of
///     Optuna's own <c>tests/hypervolume_tests/*</c> (exact integer hypervolumes and the 1−1/e
///     approximation guarantee); the pinned scalar values were produced by running live Optuna.
/// </summary>
[TestClass]
public class HypervolumeTests
{
    // ---- compute_hypervolume: structured exact values (port of test_wfg_2d / test_wfg_3d) -------

    [DataTestMethod]
    [DataRow(2)]
    [DataRow(5)]
    [DataRow(9)]
    [DataRow(20)]
    public void ComputeHypervolume_2dStructured_ExactCount(int n)
    {
        // s[:n] = (i, n-1-i); s[n:] = (i, n-i); ref = (n, n). HV = n*n - n*(n-1)/2.
        var pts = new List<double[]>();
        for (int i = 0; i < n; i++)
            pts.Add(new double[] { i, n - 1 - i });
        for (int i = 0; i < n + 1; i++)
            pts.Add(new double[] { i, n - i });

        double hv = Hypervolume.ComputeHypervolume(pts.ToArray(), new double[] { n, n });
        hv.Should().Be(n * n - n * (n - 1) / 2);
    }

    [DataTestMethod]
    [DataRow(2)]
    [DataRow(4)]
    [DataRow(6)]
    [DataRow(9)]
    public void ComputeHypervolume_3dStructured_ExactCount(int n)
    {
        // s = {(x, y, n-1-x-y)}; ref = (n, n, n). HV = n^3 - (n-1)*n*(n+1)/6.
        var pts = new List<double[]>();
        for (int x = 0; x < n; x++)
            for (int y = 0; y < n - x; y++)
                pts.Add(new double[] { x, y, n - 1 - x - y });

        double hv = Hypervolume.ComputeHypervolume(pts.ToArray(), new double[] { n, n, n });
        hv.Should().Be((double)n * n * n - (long)(n - 1) * n * (n + 1) / 6);
    }

    [DataTestMethod]
    [DataRow(2)]
    [DataRow(4)]
    [DataRow(6)]
    public void ComputeHypervolume_ndIdentity_ExactCount(int nObjs)
    {
        // The identity rows e_i (ref = 10) cover everything except the unit corner cube [0,1)^n,
        // so HV = 10^n - 1 (the identity half of Optuna's test_wfg_nd).
        var pts = new List<double[]>();
        for (int i = 0; i < nObjs; i++)
        {
            var row = new double[nObjs];
            row[i] = 1.0;  // e_i (the rest are 0).
            pts.Add(row);
        }
        var reff = Enumerable.Repeat(10.0, nObjs).ToArray();
        double hv = Hypervolume.ComputeHypervolume(pts.ToArray(), reff);
        hv.Should().Be(Math.Pow(10, nObjs) - 1);
    }

    // ---- compute_hypervolume: pinned values + edge cases (vs live Optuna) ----------------------

    [TestMethod]
    public void ComputeHypervolume_Pinned2d_MatchesOptuna()
    {
        var s = new[]
        {
            new[] { 1.0, 4.0 }, new[] { 2.0, 2.0 }, new[] { 3.0, 1.0 },
            new[] { 0.5, 5.0 }, new[] { 4.0, 0.5 },
        };
        Hypervolume.ComputeHypervolume(s, new[] { 5.0, 6.0 }).Should().Be(17.0);
    }

    [TestMethod]
    public void ComputeHypervolume_Pinned3d_MatchesOptuna()
    {
        var s = new[]
        {
            new[] { 1.0, 2.0, 3.0 }, new[] { 3.0, 1.0, 2.0 }, new[] { 2.0, 3.0, 1.0 },
            new[] { 1.0, 1.0, 4.0 }, new[] { 4.0, 1.0, 1.0 },
        };
        Hypervolume.ComputeHypervolume(s, new[] { 5.0, 5.0, 5.0 }).Should().Be(48.0);
    }

    [DataTestMethod]
    [DataRow(2)]
    [DataRow(3)]
    [DataRow(5)]
    public void ComputeHypervolume_Empty_ReturnsZero(int nObjs)
    {
        Hypervolume.ComputeHypervolume(Array.Empty<double[]>(), Enumerable.Repeat(1.0, nObjs).ToArray())
            .Should().Be(0.0);
    }

    [TestMethod]
    public void ComputeHypervolume_InfiniteReference_ReturnsInfinity()
    {
        var s = new[] { new[] { 1.0, double.PositiveInfinity, 1.0 } };
        var reff = new[] { 1.1, double.PositiveInfinity, 1.1 };
        Hypervolume.ComputeHypervolume(s, reff).Should().Be(double.PositiveInfinity);
    }

    [TestMethod]
    public void ComputeHypervolume_NanReference_Throws()
    {
        var s = new[] { new[] { 1.0, 1.0, 1.0 } };
        var reff = new[] { 1.1, 1.1, double.NaN };
        var act = () => Hypervolume.ComputeHypervolume(s, reff);
        act.Should().Throw<ValueError>().WithMessage("*dominate or equal the reference point*");
    }

    [TestMethod]
    public void ComputeHypervolume_PointExceedsReference_Throws()
    {
        var s = new[] { new[] { 2.0, 2.0, 2.0 } };
        var act = () => Hypervolume.ComputeHypervolume(s, new[] { 1.0, 1.0, 1.0 });
        act.Should().Throw<ValueError>();
    }

    [TestMethod]
    public void ComputeHypervolume_DuplicatePoints_UnaffectedByDuplication()
    {
        var s = new[]
        {
            new[] { 1.0, 3.0, 2.0 }, new[] { 2.0, 1.0, 3.0 }, new[] { 3.0, 2.0, 1.0 },
        };
        var reff = new[] { 5.0, 5.0, 5.0 };
        double baseline = Hypervolume.ComputeHypervolume(s, reff);
        var withDup = s.Concat(new[] { s[^1] }).ToArray();  // Add an existing point.
        Hypervolume.ComputeHypervolume(withDup, reff).Should().Be(baseline);
    }

    // ---- solve_hssp: pinned selections + the 1 − 1/e approximation guarantee -------------------

    [TestMethod]
    public void SolveHssp_Pinned2d_MatchesOptuna()
    {
        var s = new[]
        {
            new[] { 1.0, 4.0 }, new[] { 2.0, 2.0 }, new[] { 3.0, 1.0 },
            new[] { 0.5, 5.0 }, new[] { 4.0, 0.5 },
        };
        var reff = new[] { 5.0, 6.0 };
        Hypervolume.SolveHssp(s, Enumerable.Range(0, 5).ToArray(), 2, reff)
            .OrderBy(i => i).Should().Equal(0, 1);
        Hypervolume.SolveHssp(s, Enumerable.Range(0, 5).ToArray(), 3, reff)
            .OrderBy(i => i).Should().Equal(0, 1, 2);
    }

    [TestMethod]
    public void SolveHssp_Pinned3d_MatchesOptuna()
    {
        var s = new[]
        {
            new[] { 1.0, 2.0, 3.0 }, new[] { 3.0, 1.0, 2.0 }, new[] { 2.0, 3.0, 1.0 },
            new[] { 1.0, 1.0, 4.0 }, new[] { 4.0, 1.0, 1.0 },
        };
        Hypervolume.SolveHssp(s, Enumerable.Range(0, 5).ToArray(), 2, new[] { 5.0, 5.0, 5.0 })
            .OrderBy(i => i).Should().Equal(0, 2);
    }

    [DataTestMethod]
    [DataRow(2)]
    [DataRow(3)]
    public void SolveHssp_Achieves1MinusInvE_Approximation(int dim)
    {
        // Port of Optuna's test_solve_hssp: greedy HSSP is a 1 − 1/e ≈ 0.6321 approximation of the
        // optimal hypervolume subset. Truth is brute-forced over all subsets of the given size.
        var testCase = FixedPoints(dim);
        var reff = ReferencePoint(testCase, 1.1);

        for (int subset = 1; subset <= testCase.Length; subset++)
        {
            double truth = 0.0;
            foreach (var combo in Combinations(testCase.Length, subset))
            {
                var subPts = combo.Select(i => testCase[i]).ToArray();
                double hv = Hypervolume.ComputeHypervolume(subPts, reff);
                double.IsNaN(hv).Should().BeFalse();
                truth = Math.Max(truth, hv);
            }

            var selected = Hypervolume.SolveHssp(testCase, Enumerable.Range(0, testCase.Length).ToArray(), subset, reff);
            double approx = Hypervolume.ComputeHypervolume(selected.Select(i => testCase[i]).ToArray(), reff);
            double.IsNaN(approx).Should().BeFalse();
            (approx / truth).Should().BeGreaterThan(0.6321);
        }
    }

    [TestMethod]
    public void SolveHssp_FullSubset_ReturnsAllIndices()
    {
        var s = new[] { new[] { 1.0, 2.0 }, new[] { 2.0, 1.0 }, new[] { 3.0, 0.5 } };
        var idx = new[] { 10, 20, 30 };
        Hypervolume.SolveHssp(s, idx, 3, new[] { 5.0, 5.0 }).Should().Equal(10, 20, 30);
    }

    // ---- helpers -------------------------------------------------------------------------------

    private static double[][] FixedPoints(int dim) => dim == 2
        ? new[]
        {
            new[] { 0.7, 0.2 }, new[] { 0.1, 0.9 }, new[] { 0.4, 0.5 }, new[] { 0.9, 0.05 },
            new[] { 0.3, 0.6 }, new[] { 0.55, 0.35 }, new[] { 0.2, 0.8 }, new[] { 0.65, 0.25 },
        }
        : new[]
        {
            new[] { 0.7, 0.2, 0.5 }, new[] { 0.1, 0.9, 0.3 }, new[] { 0.4, 0.5, 0.6 },
            new[] { 0.9, 0.05, 0.2 }, new[] { 0.3, 0.6, 0.4 }, new[] { 0.55, 0.35, 0.7 },
            new[] { 0.2, 0.8, 0.1 }, new[] { 0.65, 0.25, 0.45 },
        };

    private static double[] ReferencePoint(double[][] pts, double scale)
    {
        int m = pts[0].Length;
        var reff = new double[m];
        for (int k = 0; k < m; k++)
            reff[k] = scale * pts.Max(p => p[k]);
        return reff;
    }

    private static IEnumerable<int[]> Combinations(int n, int k)
    {
        var idx = Enumerable.Range(0, k).ToArray();
        if (k == 0 || k > n)
            yield break;
        while (true)
        {
            yield return (int[])idx.Clone();
            int i = k - 1;
            while (i >= 0 && idx[i] == n - k + i)
                i--;
            if (i < 0)
                yield break;
            idx[i]++;
            for (int j = i + 1; j < k; j++)
                idx[j] = idx[j - 1] + 1;
        }
    }
}
