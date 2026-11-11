using System;
using System.Globalization;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Bit-for-bit parity of <see cref="Hypervolume.ComputeHypervolume" /> and <see cref="Hypervolume.SolveHssp" /> with
///     Optuna's own <c>optuna._hypervolume.compute_hypervolume</c> and <c>_solve_hssp</c>, live. Optuna computes the
///     two- and three-objective volumes with NumPy's <c>@</c> / <c>np.dot</c>, i.e. BLAS <c>ddot</c> and <c>dgemv</c>, whose
///     accumulation order no sequential loop reproduces; OptunaSharp routes those products through NumSharp's
///     <c>np.dot</c> on the same OpenBLAS binary NumPy bundles, so the volumes, and the greedy subset selection built on
///     them (multi-objective TPE's tie-break, SMS-EMOA's survival), match to the last bit. Fronts use full-precision
///     random coordinates, so every product and sum rounds; duplicated and weakly dominated rows exercise the
///     zero-contribution ties of the three-objective subset selection.
/// </summary>
[TestClass]
public class HypervolumeParityTests : OptunaInteropTestBase
{
    /// <summary>
    ///     240 fronts in 2 to 5 objectives: half random rows of 1 to 80 points (dominated rows and duplicates included,
    ///     so Optuna's own Pareto filtering and sorting run, leaving short fronts), half mutually non-dominated simplex
    ///     fronts of up to 80 points (long dot products, through BLAS's unrolled kernels). Every volume equals Optuna's bit
    ///     for bit.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ComputeHypervolume_MatchesOptunaBitForBit_TwoToFiveObjectives()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec("import struct\nimport numpy as np\nfrom optuna._hypervolume import compute_hypervolume\n");
        // A seeded System.Random only generates test fronts; no sampler path reads it.
        var rng = new Random(20261111);
        int mismatches = 0;
        string firstMismatch = "";
        for (int trial = 0; trial < 240; trial++)
        {
            int m = 2 + trial % 4;
            int n = 1 + rng.Next(80);
            double[][] loss = (trial / 4) % 2 == 0
                ? RandomRows(rng, n, m, duplicateEvery: 7)
                : SimplexFront(rng, n, m, duplicateEvery: 11);
            double[] reference = Enumerable.Repeat(1.1, m).ToArray();
            double hv = Hypervolume.ComputeHypervolume(loss, reference);

            PyExec($"loss = np.array({Lit(loss)}).reshape(-1, {m})\nref = np.array({Lit(reference)})\n" +
                   "hv_hex = struct.pack('>d', compute_hypervolume(loss, ref)).hex()\n");
            string expected = PyStr("hv_hex");
            string actual = BitConverter.DoubleToInt64Bits(hv).ToString("x16");
            if (expected != actual)
            {
                mismatches++;
                if (firstMismatch.Length == 0)
                    firstMismatch = $"trial {trial}: m={m} n={n} OptunaSharp {actual} vs Optuna {expected}";
            }
        }
        mismatches.Should().Be(0, $"every hypervolume must be Optuna's to the last bit; first mismatch: {firstMismatch}");
    }

    /// <summary>
    ///     Greedy hypervolume subset selection on 240 three- and four-objective fronts: the selected indices equal
    ///     Optuna's, in order. Half the fronts carry duplicated rows (Optuna de-duplicates them first); the other half
    ///     put some points on the reference point's boundary (a coordinate equal to the reference, so their boxes have
    ///     zero volume) and ask for more points than have positive volume, so the greedy step must choose among
    ///     contributions that are zero up to the sub-ulp residual of the hypervolume differences, which only Optuna's
    ///     own BLAS accumulation order resolves the same way.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void SolveHssp_SelectsOptunasIndices_ThreeAndFourObjectives()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec("import numpy as np\nfrom optuna._hypervolume import _solve_hssp\n");
        var rng = new Random(424242);
        int mismatches = 0;
        string firstMismatch = "";
        for (int trial = 0; trial < 240; trial++)
        {
            int m = 3 + trial % 2;
            int n = 4 + rng.Next(40);
            double[] reference = Enumerable.Repeat(1.05, m).ToArray();
            double[][] front = SimplexFront(rng, n, m, duplicateEvery: 3);
            int subsetSize = 1 + rng.Next(front.Length - 1);
            if ((trial / 2) % 2 == 1)
            {
                // Append zero-volume points: one coordinate on the reference, the others a tiny simplex of total 0.01.
                // The simplex points' coordinates are all at least ~0.03, so none dominates or is dominated by these,
                // and the subset size reaches past the positive-volume points so some zero contribution is chosen.
                int zeros = 2 + rng.Next(6);
                var withZeros = front.ToList();
                for (int z = 0; z < zeros; z++)
                {
                    var v = new double[m];
                    int pinned = rng.Next(m);
                    double sum = 0;
                    for (int d = 0; d < m; d++)
                        if (d != pinned)
                            sum += v[d] = 0.1 + rng.NextDouble();
                    for (int d = 0; d < m; d++)
                        v[d] = d == pinned ? reference[d] : v[d] / sum * 0.01;
                    withZeros.Add(v);
                }
                front = withZeros.ToArray();
                subsetSize = Math.Min(front.Length - 1, front.Length - zeros + 1 + rng.Next(zeros - 1));
            }
            int[] indices = Enumerable.Range(0, front.Length).ToArray();
            int[] selected = Hypervolume.SolveHssp(front, indices, subsetSize, reference);

            PyExec($"front = np.array({Lit(front)}).reshape(-1, {m})\nref = np.array({Lit(reference)})\n" +
                   $"sel = _solve_hssp(front, np.arange({front.Length}), {subsetSize}, ref)\n" +
                   "sel_str = ','.join(str(int(i)) for i in sel)\n");
            string expected = PyStr("sel_str");
            string actual = string.Join(",", selected);
            if (expected != actual)
            {
                mismatches++;
                if (firstMismatch.Length == 0)
                    firstMismatch = $"trial {trial}: m={m} n={front.Length} k={subsetSize} OptunaSharp [{actual}] vs Optuna [{expected}]";
            }
        }
        mismatches.Should().Be(0, $"every selection must be Optuna's; first mismatch: {firstMismatch}");
    }

    /// <summary>Uniform rows in [0, 1) with every <paramref name="duplicateEvery" />-th row a copy of an earlier one.</summary>
    /// <param name="rng">The seeded test-input generator.</param>
    /// <param name="n">The number of rows.</param>
    /// <param name="m">The number of objectives.</param>
    /// <param name="duplicateEvery">The period of duplicated rows (exact copies, so Optuna's de-duplication runs).</param>
    /// <returns>The rows; most are dominated, so the Pareto filter does real work.</returns>
    private static double[][] RandomRows(Random rng, int n, int m, int duplicateEvery)
    {
        var rows = new double[n][];
        for (int i = 0; i < n; i++)
        {
            if (i > 0 && i % duplicateEvery == 0)
            {
                rows[i] = (double[])rows[rng.Next(i)].Clone();
                continue;
            }
            rows[i] = new double[m];
            for (int d = 0; d < m; d++)
                rows[i][d] = rng.NextDouble();
        }
        return rows;
    }

    /// <summary>
    ///     A mutually non-dominated front: positive random vectors scaled to sum to one (no point on that simplex can
    ///     dominate another), with every <paramref name="duplicateEvery" />-th row a copy of an earlier one.
    /// </summary>
    /// <param name="rng">The seeded test-input generator.</param>
    /// <param name="n">The number of rows before duplicates are appended.</param>
    /// <param name="m">The number of objectives.</param>
    /// <param name="duplicateEvery">The period of appended duplicates.</param>
    /// <returns>The front, duplicates included, as HSSP receives a non-dominated rank's loss values.</returns>
    private static double[][] SimplexFront(Random rng, int n, int m, int duplicateEvery)
    {
        var rows = new System.Collections.Generic.List<double[]>();
        for (int i = 0; i < n; i++)
        {
            var v = new double[m];
            double sum = 0;
            for (int d = 0; d < m; d++)
            {
                v[d] = 0.05 + rng.NextDouble();
                sum += v[d];
            }
            for (int d = 0; d < m; d++)
                v[d] /= sum;
            rows.Add(v);
            if (i % duplicateEvery == duplicateEvery - 1)
                rows.Add((double[])rows[rng.Next(rows.Count)].Clone());
        }
        return rows.ToArray();
    }

    /// <summary>A Python list literal of rows, every value printed round-trip exact.</summary>
    /// <param name="a">The rows.</param>
    /// <returns>The literal, e.g. <c>[[0.1, 0.2], [0.3, 0.4]]</c>.</returns>
    private static string Lit(double[][] a) => "[" + string.Join(", ", a.Select(Lit)) + "]";

    /// <summary>A Python list literal of one row, every value printed round-trip exact.</summary>
    /// <param name="a">The values.</param>
    /// <returns>The literal, e.g. <c>[0.1, 0.2]</c>.</returns>
    private static string Lit(double[] a) => "[" + string.Join(", ", a.Select(x => x.ToString("R", CultureInfo.InvariantCulture))) + "]";
}
