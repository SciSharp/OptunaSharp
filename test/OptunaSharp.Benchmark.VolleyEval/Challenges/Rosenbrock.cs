using System;
using System.Collections.Generic;
using OptunaSharp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Benchmark.VolleyEval.Challenges;

/// <summary>
///     <b>Challenge 1 — the coupled continuous radical.</b> Rotated, shifted Rosenbrock (BBOB f9),
///     the canonical non-separable smooth test function (Best-Practice §1; COCO/BBOB f8/f9). Its bent,
///     ill-scaled valley is the textbook case where full-covariance adaptation (CMA-ES) and its
///     limited-memory form (LM-CMA at high D) win, GP is viable only at low D, and GA/QMC lag.
///
///     <para>
///         <b>Objective</b> (minimize):
///         <c>f(x) = Σ_{i=1}^{D-1} [100·(z_{i+1} − z_i²)² + (1 − z_i)²]</c>, where
///         <c>z = Q·(x − x_opt) + 1</c>. The optimum is exactly <c>0</c> at <c>x = x_opt</c> (because
///         <c>z = (1,…,1)</c> there), so shifting/rotating keeps the optimum known while defeating
///         axis-aligned exploitation.
///     </para>
///
///     <para>
///         <b>Instance transform.</b> <see cref="InstanceId" /> seeds the mandated
///         <see cref="LazyRandomState" /> (NumPy-exact MT19937, never <c>System.Random</c>): a uniform
///         shift <c>x_opt ∈ [-3, 3]^D</c> (inside the [-5, 5] box so the optimum is reachable) and an
///         orthogonal rotation <c>Q</c> built by modified Gram-Schmidt on a Gaussian matrix (the
///         dependency-free equivalent of the plan's "NumSharp <c>qr</c> of a Gaussian").
///     </para>
///
///     <para>
///         <b>Random baseline.</b> The empirical mean of <c>f</c> under uniform sampling of the box,
///         estimated once with a large seeded sample — the score = 0 reference.
///     </para>
/// </summary>
public sealed class Rosenbrock : IChallenge
{
    private readonly double[] _shift;
    private readonly double[,] _q;

    /// <param name="dimension">D. v0 uses 40, where covariance adaptation (CMA/LM-CMA) separates clearly from TPE/GA/QMC.</param>
    /// <param name="instanceId">Seeds the shift + rotation (and, offset, the baseline estimate).</param>
    /// <param name="budget">Per-run trial budget (COCO fixed-budget).</param>
    public Rosenbrock(int dimension = 40, int instanceId = 0, int budget = 300)
    {
        Dimension = dimension;
        InstanceId = instanceId;
        Budget = budget;
        (_shift, _q) = BuildInstance(dimension, instanceId);
        RandomBaseline = EstimateUniformMean(dimension, instanceId, _shift, _q);
    }

    /// <inheritdoc />
    public string Name => "rosenbrock";

    /// <inheritdoc />
    public IReadOnlyList<string> Tags => new[] { "continuous", "coupled", "non-separable", "single-obj" };

    /// <inheritdoc />
    public int Dimension { get; }

    /// <inheritdoc />
    public int NumObjectives => 1;

    /// <inheritdoc />
    public IReadOnlyList<StudyDirection> Directions => new[] { StudyDirection.Minimize };

    /// <inheritdoc />
    public int InstanceId { get; }

    /// <summary>Exact minimum of the Rosenbrock family.</summary>
    public double Optimum => 0.0;

    /// <inheritdoc />
    public double RandomBaseline { get; }

    /// <inheritdoc />
    public IReadOnlyList<double> ReferencePoint => Array.Empty<double>();

    /// <inheritdoc />
    public int Budget { get; }

    /// <summary>Continuous space — the inert (≡ random) check would never fire, so skip it.</summary>
    public bool DetectInert => false;

    /// <inheritdoc />
    public IReadOnlyList<double> Evaluate(Trial trial)
    {
        var x = new double[Dimension];
        for (int i = 0; i < Dimension; i++)
            x[i] = trial.SuggestFloat($"x{i}", -5.0, 5.0);
        return new[] { Value(x) };
    }

    /// <summary>The raw (untransformed-return) Rosenbrock value at <paramref name="x" /> for this instance. Exposed for tests (optimum reachability).</summary>
    public double Value(double[] x)
    {
        // z = Q·(x − shift) + 1  → optimum at x = shift gives z = 1 → f = 0.
        int d = Dimension;
        var z = new double[d];
        for (int i = 0; i < d; i++)
        {
            double s = 0.0;
            for (int k = 0; k < d; k++) s += _q[i, k] * (x[k] - _shift[k]);
            z[i] = s + 1.0;
        }
        double f = 0.0;
        for (int i = 0; i < d - 1; i++)
        {
            double a = 1.0 - z[i];
            double b = z[i + 1] - z[i] * z[i];
            f += a * a + 100.0 * b * b;
        }
        return f;
    }

    /// <summary>The location of the exact optimum (x_opt); a copy, for tests.</summary>
    public double[] OptimumLocation() => (double[])_shift.Clone();

    /// <inheritdoc />
    public double Score(Study study)
        => (RandomBaseline - study.BestValue) / (RandomBaseline - Optimum);

    // ---- instance construction (deterministic in instanceId) ----

    private static (double[] shift, double[,] q) BuildInstance(int d, int instanceId)
    {
        var rng = new LazyRandomState(1000 + instanceId).Rng;

        var shift = new double[d];
        for (int i = 0; i < d; i++)
            shift[i] = (double)rng.uniform(-3.0, 3.0);

        // Gaussian matrix A, then modified Gram-Schmidt → orthonormal columns Q (a rotation).
        var a = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                a[i, j] = rng.randn<double>();

        var q = new double[d, d];
        for (int j = 0; j < d; j++)
        {
            var v = new double[d];
            for (int i = 0; i < d; i++) v[i] = a[i, j];
            for (int k = 0; k < j; k++)
            {
                double dot = 0.0;
                for (int i = 0; i < d; i++) dot += q[i, k] * a[i, j];
                for (int i = 0; i < d; i++) v[i] -= dot * q[i, k];
            }
            double norm = 0.0;
            for (int i = 0; i < d; i++) norm += v[i] * v[i];
            norm = Math.Sqrt(norm);
            for (int i = 0; i < d; i++) q[i, j] = v[i] / norm;
        }
        return (shift, q);
    }

    private double EstimateUniformMean(int d, int instanceId, double[] shift, double[,] q)
    {
        // A separate seed stream (9000+id) so the baseline draw never correlates with the instance draw.
        var rng = new LazyRandomState(9000 + instanceId).Rng;
        const int samples = 20_000;
        double sum = 0.0;
        var x = new double[d];
        for (int t = 0; t < samples; t++)
        {
            for (int i = 0; i < d; i++) x[i] = (double)rng.uniform(-5.0, 5.0);
            sum += Value(x);
        }
        return sum / samples;
    }
}
