using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Benchmark.VolleyEval.Challenges;

/// <summary>
///     <b>Challenge 5 — the feasibility radical.</b> A quadratic whose unconstrained optimum lies
///     <i>outside</i> the feasible region, so the best point must sit on the constraint boundary. This
///     breaks the other assumption every earlier challenge shares — being <i>unconstrained</i>. It
///     exercises OptunaSharp's native constraint machinery: the objective reports a constraint via
///     <see cref="Trial.SetConstraint(string, double)" /> (convention <c>g ≤ 0</c> = feasible) and the
///     constraint-aware samplers (GP, NSGA-II/III) auto-detect and steer toward feasibility, while the
///     constraint-blind ones drive into the infeasible basin and waste their budget.
///
///     <para>
///         <b>Objective</b> (minimize): <c>f(x) = ‖x − t‖²</c> with target <c>t = (τ,…,τ)</c> chosen so
///         <c>‖t‖ &gt; R</c> (infeasible). <b>Constraint:</b> <c>g(x) = ‖x‖² − R² ≤ 0</c> (a feasible
///         ball of radius R at the origin). The constrained optimum is the ball point nearest <c>t</c>,
///         <c>x* = t·R/‖t‖</c>, with the exact value <c>f* = (‖t‖ − R)²</c>.
///     </para>
///
///     <para>
///         <b>Scoring is feasibility-aware:</b> <see cref="Score" /> considers only trials whose
///         constraint is satisfied and reports the best <i>feasible</i> objective (score 0 if the sampler
///         never found a feasible point). Random baseline = the uniform-sampling mean of <c>f</c> over the
///         box.
///     </para>
/// </summary>
public sealed class ConstrainedQuadratic : IChallenge
{
    private readonly double[] _target;
    private readonly double _radius;

    /// <param name="dimension">D (v0: 4 — low D keeps the feasible ball a findable ~2% of the box so even random lands a few feasible points).</param>
    /// <param name="radius">Feasible-ball radius R (v0: 2.5).</param>
    /// <param name="target">Per-coordinate target value τ (v0: 3.5, giving ‖t‖ = τ√D &gt; R, infeasible).</param>
    /// <param name="instanceId">Reserved for instance variation (kept simple: the geometry is already rotation-invariant).</param>
    /// <param name="budget">Per-run trial budget.</param>
    public ConstrainedQuadratic(int dimension = 4, double radius = 2.5, double target = 3.5, int instanceId = 0, int budget = 300)
    {
        Dimension = dimension;
        _radius = radius;
        InstanceId = instanceId;
        Budget = budget;
        _target = Enumerable.Repeat(target, dimension).ToArray();
        double normT = Math.Sqrt(_target.Sum(v => v * v));
        if (normT <= radius)
            throw new ArgumentException("The target must be infeasible (‖t‖ > R) for the constraint to bind.");
        Optimum = (normT - radius) * (normT - radius);
        RandomBaseline = EstimateUniformMean(dimension, _target);
    }

    /// <inheritdoc />
    public string Name => "constrained_quadratic";

    /// <inheritdoc />
    public IReadOnlyList<string> Tags => new[] { "continuous", "constrained", "feasibility", "single-obj" };

    /// <inheritdoc />
    public int Dimension { get; }

    /// <inheritdoc />
    public int NumObjectives => 1;

    /// <inheritdoc />
    public IReadOnlyList<StudyDirection> Directions => new[] { StudyDirection.Minimize };

    /// <inheritdoc />
    public int InstanceId { get; }

    /// <summary>Exact constrained optimum value (‖t‖ − R)² — the boundary point nearest the target.</summary>
    public double Optimum { get; }

    /// <inheritdoc />
    public double RandomBaseline { get; }

    /// <inheritdoc />
    public IReadOnlyList<double> ReferencePoint => Array.Empty<double>();

    /// <inheritdoc />
    public int Budget { get; }

    /// <summary>Constrained continuous — the inert (≡ random) check does not apply.</summary>
    public bool DetectInert => false;

    /// <inheritdoc />
    public IReadOnlyList<double> Evaluate(Trial trial)
    {
        var x = new double[Dimension];
        for (int i = 0; i < Dimension; i++)
            x[i] = trial.SuggestFloat($"x{i}", -5.0, 5.0);
        // g ≤ 0 is feasible; the constraint-aware samplers read this back off the trial.
        trial.SetConstraint("ball", Constraint(x));
        return new[] { Objective(x) };
    }

    /// <summary>f(x) = ‖x − t‖². Exposed for tests.</summary>
    public double Objective(double[] x)
    {
        double s = 0.0;
        for (int i = 0; i < Dimension; i++) { double d = x[i] - _target[i]; s += d * d; }
        return s;
    }

    /// <summary>g(x) = ‖x‖² − R² (≤ 0 feasible). Exposed for tests.</summary>
    public double Constraint(double[] x)
    {
        double s = 0.0;
        for (int i = 0; i < Dimension; i++) s += x[i] * x[i];
        return s - _radius * _radius;
    }

    /// <summary>The exact constrained optimum location x* = t·R/‖t‖ (a copy), for tests.</summary>
    public double[] OptimumLocation()
    {
        double normT = Math.Sqrt(_target.Sum(v => v * v));
        return _target.Select(v => v * _radius / normT).ToArray();
    }

    /// <inheritdoc />
    public double Score(Study study)
    {
        // Only feasible trials count; a sampler that never found a feasible point scores 0.
        double bestFeasible = double.PositiveInfinity;
        foreach (var trial in study.Trials)
        {
            if (trial.Value is not double value) continue;
            if (trial.Constraints.Values.All(c => c <= 0.0) && value < bestFeasible)
                bestFeasible = value;
        }
        if (double.IsPositiveInfinity(bestFeasible)) return 0.0;
        return (RandomBaseline - bestFeasible) / (RandomBaseline - Optimum);
    }

    private static double EstimateUniformMean(int d, double[] target)
    {
        var rng = new LazyRandomState(9700).Rng;
        const int samples = 20_000;
        double sum = 0.0;
        var x = new double[d];
        for (int t = 0; t < samples; t++)
        {
            double s = 0.0;
            for (int i = 0; i < d; i++) { x[i] = (double)rng.uniform(-5.0, 5.0); double diff = x[i] - target[i]; s += diff * diff; }
            sum += s;
        }
        return sum / samples;
    }
}
