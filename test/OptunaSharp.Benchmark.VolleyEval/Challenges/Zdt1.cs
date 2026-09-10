using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp;
using OptunaSharp.Visualization;

namespace OptunaSharp.Benchmark.VolleyEval.Challenges;

/// <summary>
///     <b>Challenge 3 — the multi-objective radical.</b> ZDT1 (Zitzler-Deb-Thiele 2000), the canonical
///     bi-objective test with a convex analytic Pareto front. Hypervolume of a 2-objective front is
///     NSGA-II's home; the single-objective methods can only scalarize or random-sample the front and
///     degrade — the third distinct outcome pattern. The covariance samplers (CmaEs, LmCma) reject a
///     multi-objective study outright (a first-class <b>DNF</b>).
///
///     <para>
///         <b>Objectives</b> (both minimized): <c>f₁(x) = x₁</c>;
///         <c>g(x) = 1 + (9/(D−1))·Σ_{i=2}^{D} x_i</c>; <c>f₂(x) = g·(1 − √(f₁/g))</c>, with
///         <c>x_i ∈ [0, 1]</c> and <c>D = 30</c>.
///     </para>
///
///     <para>
///         <b>Score = hypervolume ratio</b> <c>HV(study) / HV(true front)</c> to the reference point
///         <c>(1.1, 1.1)</c>. The true front is <c>f₂ = 1 − √f₁</c> at <c>g = 1</c>; its hypervolume to
///         that reference is analytic (see <see cref="TrueFrontHypervolume" />).
///     </para>
///
///     <para>
///         <b>Instance transform.</b> ZDT1's front is invariant to coordinate shuffling and its optimum
///         is the whole front, so a shift/rotation cannot "hide" it. The instance id therefore only
///         permutes which decision coordinate plays the <c>f₁</c> role — enough to vary the problem for
///         seed robustness without changing the analytic front or ceiling.
///     </para>
/// </summary>
public sealed class Zdt1 : IChallenge
{
    private readonly int[] _perm; // _perm[0] is the coordinate used as f1; the rest are distance vars.

    /// <param name="dimension">D (v0: 30). At D=30 the 29-variable g-function needs a large trial budget to converge, which is exactly why the sample-efficient-but-cubic GP is trial-capped while cheap NSGA runs to convergence.</param>
    /// <param name="instanceId">Permutes which coordinate is f₁ (front and ceiling unchanged).</param>
    /// <param name="budget">Per-run trial budget. v0 uses a large budget so the cheap evolutionary champion (NSGA) converges; capped contenders (GP/TPE) degrade — the two-leaderboards view made concrete.</param>
    public Zdt1(int dimension = 30, int instanceId = 0, int budget = 8000)
    {
        if (dimension < 2) throw new ArgumentOutOfRangeException(nameof(dimension));
        Dimension = dimension;
        InstanceId = instanceId;
        Budget = budget;
        _perm = BuildPermutation(dimension, instanceId);
    }

    /// <inheritdoc />
    public string Name => "zdt1";

    /// <inheritdoc />
    public IReadOnlyList<string> Tags => new[] { "continuous", "multi-objective", "pareto", "convex-front" };

    /// <inheritdoc />
    public int Dimension { get; }

    /// <inheritdoc />
    public int NumObjectives => 2;

    /// <inheritdoc />
    public IReadOnlyList<StudyDirection> Directions => new[] { StudyDirection.Minimize, StudyDirection.Minimize };

    /// <inheritdoc />
    public int InstanceId { get; }

    /// <summary>The analytic true-front hypervolume ceiling (score = 1).</summary>
    public double Optimum => TrueFrontHypervolume();

    /// <summary>Unused for the hypervolume-ratio score.</summary>
    public double RandomBaseline => 0.0;

    /// <inheritdoc />
    public IReadOnlyList<double> ReferencePoint => new[] { 1.1, 1.1 };

    /// <inheritdoc />
    public int Budget { get; }

    /// <summary>Continuous, multi-objective — the inert (≡ random) check does not apply.</summary>
    public bool DetectInert => false;

    /// <inheritdoc />
    public IReadOnlyList<double> Evaluate(Trial trial)
    {
        var x = new double[Dimension];
        for (int i = 0; i < Dimension; i++)
            x[i] = trial.SuggestFloat($"x{i}", 0.0, 1.0);
        return Objectives(x);
    }

    /// <summary>The two ZDT1 objectives for a decision vector (using this instance's coordinate permutation). Exposed for tests.</summary>
    public double[] Objectives(double[] x)
    {
        double f1 = x[_perm[0]];
        double g = 1.0;
        for (int i = 1; i < Dimension; i++)
            g += 9.0 / (Dimension - 1) * x[_perm[i]];
        double f2 = g * (1.0 - Math.Sqrt(f1 / g));
        return new[] { f1, f2 };
    }

    /// <inheritdoc />
    public double Score(Study study)
    {
        double hv = HypervolumeHistory
            .GetInfo(study, ReferencePoint)
            .Values.DefaultIfEmpty(0.0)
            .Max();
        return hv / Optimum;
    }

    /// <summary>
    ///     Closed-form hypervolume of the ZDT1 true front <c>f₂ = 1 − √f₁</c>, <c>f₁ ∈ [0, 1]</c>, to the
    ///     reference <c>(r, r)</c> with <c>r = 1.1</c>. The dominated area is
    ///     <c>∫₀¹ (r − (1 − √a)) da + (r − 1)·r</c>
    ///     = <c>(r − 1) + 2/3 + (r − 1)·r</c> = <c>0.1 + 2/3 + 0.11</c> ≈ <c>0.87667</c>.
    /// </summary>
    public double TrueFrontHypervolume()
    {
        double r = ReferencePoint[1];
        return (r - 1.0) + 2.0 / 3.0 + (r - 1.0) * r;
    }

    private static int[] BuildPermutation(int d, int instanceId)
    {
        // A deterministic rotation of coordinate indices by instanceId — the f1 role moves, the front
        // (and its hypervolume ceiling) are unchanged, giving cheap instance variety for seed robustness.
        var perm = new int[d];
        for (int i = 0; i < d; i++)
            perm[i] = (i + instanceId) % d;
        return perm;
    }
}
