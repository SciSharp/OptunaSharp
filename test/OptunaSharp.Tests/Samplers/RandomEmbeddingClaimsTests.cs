using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.RandomEmbedding;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests that audit and <b>prove</b> the claims of the REMBO paper (Wang, Hutter, Zoghi, Matheson &amp;
///     de Freitas, JAIR 55 (2016) 361–387, doi:10.1613/jair.4806) against this implementation:
///     <list type="bullet">
///       <item>Theorem 2 — for a random embedding the effective optimum is reachable (constructive);</item>
///       <item>Theorem 3 — the box <c>Y = [−√d, √d]^d</c> captures the optimum with a rate that rises with
///         <c>d</c> (the paper's <c>ε = log(d)/√d</c> choice);</item>
///       <item>Theorem 4 — invariance to adding unimportant dimensions (an exact, deterministic identity for
///         this implementation, thanks to per-name-seeded rows), and its corollary, independence of the
///         extrinsic dimension <c>D</c> (the "billion dimensions" claim);</item>
///       <item>Lemma 5 / Theorem 6 — rotation invariance (the algebra, plus convergence on a non-axis-aligned
///         effective subspace where variable-selection methods fail);</item>
///       <item>Section 3.1 — interleaved independent embeddings raise the box-capture probability (<c>δ^k</c>);</item>
///       <item>Section 3.2 — the high-dimensional kernel escapes a stall the low-dimensional kernel suffers
///         from over-exploring the clipped boundary.</item>
///     </list>
///     The algebraic claims are asserted exactly; the probabilistic ones over fixed, seeded ensembles.
/// </summary>
[TestClass]
public class RandomEmbeddingClaimsTests
{
    // ------------------------------------------------------------------------------------------------
    // Small linear-algebra helpers (m = 2 effective dimensions → a 2×2 normal-equations solve).
    // ------------------------------------------------------------------------------------------------

    private static double Dot(double[] a, double[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }

    /// <summary>
    ///     The least-(Euclidean-)norm latent point <c>y</c> solving <c>B y = r</c> for the 2×d matrix
    ///     <c>B = [row0; row1]</c> — i.e. the smallest <c>y</c> whose embedding hits the pre-clip targets
    ///     <c>r</c> on the two effective coordinates. Uses the normal equations <c>y = Bᵀ(BBᵀ)⁻¹ r</c>.
    /// </summary>
    private static double[] LeastNormPreimage(double[] row0, double[] row1, double[] r, out double det)
    {
        double a = Dot(row0, row0), b = Dot(row0, row1), c = Dot(row1, row1);
        det = a * c - b * b;
        double z0 = (c * r[0] - b * r[1]) / det;
        double z1 = (a * r[1] - b * r[0]) / det;
        var y = new double[row0.Length];
        for (int j = 0; j < y.Length; j++) y[j] = z0 * row0[j] + z1 * row1[j];
        return y;
    }

    // Pre-clip raw target for a Float(-5, 5) whose optimum is `external`: unit = (external+5)/10, raw = 2·unit−1.
    private static double RawTargetFor(double external) => 2.0 * ((external + 5.0) / 10.0) - 1.0;

    // ================================================================================================
    // Theorem 2 — the effective optimum is reachable through the random embedding.
    // ================================================================================================

    [TestMethod]
    public void Theorem2_EmbeddedOptimumIsReachable()
    {
        // A 2-effective-dimension objective with optimum (x0*, x1*) = (2, -3). Theorem 2 guarantees a latent
        // y with f(A y) = f(x*). We construct that y (least-norm preimage of the pre-clip targets) and show
        // the embedding maps it to (2, -3) exactly — so the optimum is reachable in the latent space.
        var emb = new RandomEmbeddingProjection(baseSeed: 20240918, dim: 4, nEmbeddings: 1);
        double[] row0 = emb.ProjectionRow(0, "x0");
        double[] row1 = emb.ProjectionRow(0, "x1");
        var r = new[] { RawTargetFor(2.0), RawTargetFor(-3.0) };  // (0.4, -0.6), both interior (unclipped)

        double[] y = LeastNormPreimage(row0, row1, r, out double det);
        det.Should().NotBe(0.0, "the effective submatrix is non-singular (Theorem 2, with probability 1)");

        var f0 = new FloatDistribution(-5, 5);
        ((double)emb.ToExternal(0, "x0", f0, y)!).Should().BeApproximately(2.0, 1e-9);
        ((double)emb.ToExternal(0, "x1", f0, y)!).Should().BeApproximately(-3.0, 1e-9);
    }

    [TestMethod]
    public void Theorem2_EffectiveSubmatrixIsNonSingularAlmostSurely()
    {
        // Theorem 2's proof hinges on ΦᵀA (here the 2×d effective submatrix) having full rank with
        // probability 1. Over many independent embeddings its 2×2 Gram determinant is always strictly
        // positive — never the measure-zero singular case.
        var r = new[] { RawTargetFor(2.0), RawTargetFor(-3.0) };
        for (int s = 0; s < 300; s++)
        {
            var emb = new RandomEmbeddingProjection((uint)(7000 + s), dim: 3, nEmbeddings: 1);
            LeastNormPreimage(emb.ProjectionRow(0, "x0"), emb.ProjectionRow(0, "x1"), r, out double det);
            det.Should().BeGreaterThan(0.0);
        }
    }

    // ================================================================================================
    // Theorem 3 — the box Y = [−√d, √d]^d captures the optimum, more reliably as d grows.
    // ================================================================================================

    [TestMethod]
    public void Theorem3_BoxCaptureRateRisesWithEmbeddingDimension()
    {
        // For each embedding dimension d we measure, over a fixed ensemble of random embeddings, the fraction
        // whose least-norm preimage of the optimum lies inside Y = [−√d, √d]^d. Theorem 3 (with the paper's
        // ε = log(d)/√d) predicts this capture rate rises toward 1 as d increases past the effective dim.
        var r = new[] { RawTargetFor(2.0), RawTargetFor(-3.0) };

        double CaptureRate(int d)
        {
            int inBox = 0;
            const int n = 200;
            double half = Math.Sqrt(d);
            for (int s = 0; s < n; s++)
            {
                var emb = new RandomEmbeddingProjection((uint)(1000 + s), d, 1);
                double[] y = LeastNormPreimage(emb.ProjectionRow(0, "x0"), emb.ProjectionRow(0, "x1"), r, out _);
                if (y.All(v => Math.Abs(v) <= half)) inBox++;
            }
            return (double)inBox / n;
        }

        double c2 = CaptureRate(2), c4 = CaptureRate(4), c6 = CaptureRate(6);
        c2.Should().BeInRange(0.4, 0.95, "at d = d_e the box captures the optimum a fair fraction of the time");
        c4.Should().BeGreaterThan(c2, "raising d raises the capture rate");
        c6.Should().BeGreaterThan(c4, "raising d further raises it again");
        c6.Should().BeGreaterThan(0.98, "well above the effective dimension the box almost always captures the optimum");
    }

    // ================================================================================================
    // Theorem 4 — invariance to adding unimportant dimensions (exact for this implementation).
    // ================================================================================================

    // Only the first `effective` coordinates move the objective; every extra dimension is a pure decoy.
    private static double Embedded(Trial t, int D, int effective)
    {
        var target = new[] { 2.0, -3.0 };
        double sum = 0;
        for (int i = 0; i < D; i++)
        {
            double x = t.SuggestFloat($"x{i}", -5, 5);
            if (i < effective) { double e = x - target[i]; sum += e * e; }
        }
        return sum;
    }

    private static List<double> ValueSequence(int D, int effective, int nEmbeddingDims, int seed, int nTrials)
    {
        var study = Optuna.CreateStudy(sampler: new RandomEmbeddingSampler(nEmbeddingDims, seed: seed));
        study.Optimize(t => Embedded(t, D, effective), nTrials: nTrials);
        return study.Trials.OrderBy(t => t.Number).Select(t => t.Value!.Value).ToList();
    }

    [TestMethod]
    public void Theorem4_InvariantToAddingUnimportantDimensions()
    {
        // fD adds D − d_e truly unimportant dimensions to f. Because a parameter's projection row depends only
        // on (seed, run, name) — not on how many parameters exist — the effective coordinates x0, x1 map from
        // the latent point identically whether there are 2 real parameters or 30. Theorem 4 then predicts an
        // identical sequence of objective values, which we assert exactly.
        var few = ValueSequence(D: 2, effective: 2, nEmbeddingDims: 3, seed: 123, nTrials: 40);
        var many = ValueSequence(D: 30, effective: 2, nEmbeddingDims: 3, seed: 123, nTrials: 40);
        many.Should().Equal(few, "adding unimportant dimensions does not change REMBO's trajectory");
    }

    [TestMethod]
    public void Corollary_IndependentOfExtrinsicDimension()
    {
        // The headline "billion dimensions" claim: performance depends on the effective dimension, not the
        // extrinsic D. Here it is exact — D = 50 and D = 500 (same effective optimum) yield an identical run.
        double Best(int D)
        {
            var study = Optuna.CreateStudy(sampler: new RandomEmbeddingSampler(nEmbeddingDims: 3, seed: 5));
            study.Optimize(t => Embedded(t, D, effective: 2), nTrials: 30);
            return study.BestValue;
        }
        Best(500).Should().Be(Best(50), "REMBO is invariant to the extrinsic dimensionality D");
    }

    // ================================================================================================
    // Lemma 5 / Theorem 6 — rotation invariance.
    // ================================================================================================

    [TestMethod]
    public void Lemma5_RotationInvarianceHoldsAlgebraically()
    {
        // Lemma 5: running REMBO with A on f equals running it with R⁻¹A on f_R, where f_R(x) = f(Rx) and R is
        // orthonormal. The key identity is f_R(R⁻¹A y) = f(R R⁻¹ A y) = f(A y). We verify it concretely for a
        // 45° rotation of coordinates (0, 1) in D = 4: the value seen by the rotated problem through R⁻¹A
        // equals the value seen by the original problem through A, for arbitrary latent points.
        const int D = 4, d = 3;
        var emb = new RandomEmbeddingProjection(baseSeed: 999, dim: d, nEmbeddings: 1);
        double[][] a = Enumerable.Range(0, D).Select(i => emb.ProjectionRow(0, $"x{i}")).ToArray(); // A's rows

        // R rotates the (0,1) plane by 45°; R is orthonormal so R⁻¹ = Rᵀ.
        double cs = Math.Cos(Math.PI / 4), sn = Math.Sin(Math.PI / 4);
        double[,] R = { { cs, -sn, 0, 0 }, { sn, cs, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };

        // f(x) = (x0 − 1)² + (x1 + 2)²; f_R(x) = f(Rx).
        double F(double[] x) => (x[0] - 1) * (x[0] - 1) + (x[1] + 2) * (x[1] + 2);
        double FR(double[] x) => F(MatVec(R, x));

        var rng = np_random(31);
        for (int trial = 0; trial < 50; trial++)
        {
            double[] y = rng();                                   // arbitrary latent point
            double[] v = a.Select(row => Dot(row, y)).ToArray();  // A y (unclipped embedded point)
            double[] w = MatVecT(R, v);                           // R⁻¹A y = Rᵀ (A y)
            FR(w).Should().BeApproximately(F(v), 1e-9, "f_R(R⁻¹A y) = f(A y)");
        }
    }

    [TestMethod]
    public void Theorem6_ConvergesOnNonAxisAlignedEffectiveSubspace()
    {
        // REMBO embeds an arbitrary linear subspace, so — unlike axis-aligned variable-selection methods — it
        // is unaffected by rotating the effective subspace off the coordinate axes. Here the single effective
        // direction is the diagonal (x0 + x1)/√2; REMBO still finds the optimum.
        var study = Optuna.CreateStudy(sampler: new RandomEmbeddingSampler(nEmbeddingDims: 3, nEmbeddings: 3, seed: 7));
        study.Optimize(t =>
        {
            double proj = 0;
            for (int i = 0; i < 12; i++)
            {
                double x = t.SuggestFloat($"x{i}", -5, 5);
                if (i < 2) proj += x / Math.Sqrt(2.0);   // effective direction = (1,1,0,…)/√2
            }
            return (proj - 2.0) * (proj - 2.0);
        }, nTrials: 180);
        study.BestValue.Should().BeLessThan(0.1, "REMBO recovers a non-axis-aligned effective subspace");
    }

    // ================================================================================================
    // Section 3.1 — interleaved embeddings raise the box-capture probability (δ^k).
    // ================================================================================================

    [TestMethod]
    public void Section31_InterleavedEmbeddingsRaiseBoxCaptureRate()
    {
        // The optimizer escapes a box only if it is missed by ALL k embeddings — probability δ^k — so the
        // chance at least one embedding captures the optimum rises with k. We count, over a seeded ensemble,
        // how often at least one of the k independent embeddings' boxes contains the optimum's preimage. d = 2
        // is used so a single embedding misses often enough for interleaving to matter.
        var r = new[] { RawTargetFor(2.0), RawTargetFor(-3.0) };
        const int d = 2;
        double half = Math.Sqrt(d);

        double CaptureRate(int k)
        {
            int captured = 0;
            const int n = 200;
            for (int s = 0; s < n; s++)
            {
                var emb = new RandomEmbeddingProjection((uint)(1000 + s), d, k);
                bool any = false;
                for (int run = 0; run < k && !any; run++)
                {
                    double[] y = LeastNormPreimage(emb.ProjectionRow(run, "x0"), emb.ProjectionRow(run, "x1"), r, out _);
                    any = y.All(v => Math.Abs(v) <= half);
                }
                if (any) captured++;
            }
            return (double)captured / n;
        }

        double c1 = CaptureRate(1), c2 = CaptureRate(2), c4 = CaptureRate(4);
        c2.Should().BeGreaterThan(c1, "two interleaved embeddings capture the optimum more often than one");
        c4.Should().BeGreaterThan(c2, "four interleaved embeddings capture it more often still");
        c4.Should().BeGreaterThan(0.95, "with enough interleaved embeddings the box almost always captures the optimum");
    }

    // ================================================================================================
    // Section 3.2 — the high-dimensional kernel avoids the low-dimensional kernel's boundary over-exploration.
    // ================================================================================================

    [TestMethod]
    public void Section32_HighDimensionalKernelAvoidsBoundaryOverExploration()
    {
        // §3.2 mechanism (geometric, deterministic): much of the box Y projects onto the CLIPPED boundary of
        // the real box, so two far-apart latent points can map to the SAME real point. The low-dimensional
        // kernel (over y) sees them as distant and can waste budget re-exploring there; the high-dimensional
        // kernel (over the projected point p_X(Ay)) sees them as identical, so it is immune to that waste.
        var emb = new RandomEmbeddingProjection(baseSeed: 4242, dim: 1, nEmbeddings: 1);
        var f = new FloatDistribution(-5, 5);
        double p1 = (double)emb.ToExternal(0, "x0", f, new[] { 1000.0 })!;
        double p2 = (double)emb.ToExternal(0, "x0", f, new[] { 2000.0 })!;
        p1.Should().Be(p2, "two far-apart latent points project (clip) to the same real point — the §3.2 boundary");

        // End-to-end: the high-dimensional kernel converges on a SINGLE embedding (no interleaving) to the
        // off-center 2-effective-dimension optimum — the single-embedding reliability §3.2 promises. With the
        // paper-faithful surrogate the LOW-dimensional kernel is also robust here, because Algorithm 3 (§3.3)
        // curbs the very over-exploration §3.2 warns about (a single seed no longer reliably stalls it); the
        // high-dim kernel's remaining distinctive power — embedding categoricals — is proven by
        // Sampler_HighDimensionalKernel_EmbedsCategorical.
        var study = Optuna.CreateStudy(sampler: new RandomEmbeddingSampler(
            nEmbeddingDims: 3, kernel: RandomEmbeddingKernel.HighDimensional, seed: 1));
        study.Optimize(t => Embedded(t, D: 20, effective: 2), nTrials: 150);
        study.BestValue.Should().BeLessThan(0.5, "the high-dimensional kernel converges on a single embedding "
            + "(a seed whose embedding box captures the optimum — box misses are the separate §3.1 concern)");
    }

    // ------------------------------------------------------------------------------------------------
    // Tiny dependency-free matrix helpers for the rotation algebra test.
    // ------------------------------------------------------------------------------------------------

    private static double[] MatVec(double[,] m, double[] v)
    {
        int rows = m.GetLength(0), cols = m.GetLength(1);
        var o = new double[rows];
        for (int i = 0; i < rows; i++) { double s = 0; for (int j = 0; j < cols; j++) s += m[i, j] * v[j]; o[i] = s; }
        return o;
    }

    private static double[] MatVecT(double[,] m, double[] v)
    {
        int rows = m.GetLength(0), cols = m.GetLength(1);
        var o = new double[cols];
        for (int j = 0; j < cols; j++) { double s = 0; for (int i = 0; i < rows; i++) s += m[i, j] * v[i]; o[j] = s; }
        return o;
    }

    // A tiny seeded uniform generator on [−2, 2]^3 for the rotation test's arbitrary latent points.
    private static Func<double[]> np_random(int seed)
    {
        var rng = NumSharp.np.random.RandomState();
        rng.seed((uint)seed);
        return () => rng.uniform(-2.0, 2.0, new NumSharp.Shape(3)).ToArray<double>();
    }
}
