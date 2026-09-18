// The paper-faithful global optimizer REMBO uses for both the expected-improvement acquisition and the
// length-scale marginal-likelihood fit: "the algorithm runs both DIRECT and CMA-ES and uses the result of
// the best of the two options" (Wang et al. 2016, §3.3). This pairs the deterministic DIRECT (Direct.cs)
// with a CMA-ES search that REUSES the repo's byte-exact CMA core (OptunaSharp.Samplers.CmaEs.Cma) as a
// black-box box-constrained minimizer, driven through its own ask/tell surface WITHOUT modification — so
// no shared component changes (the project rule). Both searches run in normalized [0,1]^k coordinates and
// map to the real box inside the objective, so a single CMA-ES step size is well scaled across dimensions.

using System;
using System.Collections.Generic;
using OptunaSharp.Samplers.CmaEs;

namespace OptunaSharp.Samplers.ModelBased.Gp;

/// <summary>
///     Box-constrained global <b>minimization</b> of a cheap deterministic function by running DIRECT and
///     CMA-ES and returning the better result — REMBO's §3.3 acquisition/length-scale optimizer. Callers
///     maximizing (expected improvement, marginal likelihood) pass the negated objective.
/// </summary>
internal static class GlobalMinimizer
{
    /// <summary>
    ///     Minimizes <paramref name="f" /> over <c>[lb, ub]</c> by running both DIRECT and CMA-ES and taking
    ///     the lower result (Wang et al. 2016, §3.3). DIRECT contributes deterministic global coverage;
    ///     CMA-ES contributes a seeded local-to-global refinement whose randomness is a pure function of
    ///     <paramref name="seed" /> so the whole optimizer is reproducible.
    /// </summary>
    /// <param name="f">The objective over the original box; a non-finite value is treated as very large.</param>
    /// <param name="lb">Per-dimension lower bounds (length <c>k ≥ 1</c>).</param>
    /// <param name="ub">Per-dimension upper bounds; each must exceed its lower bound.</param>
    /// <param name="seed">Seed for the CMA-ES stream (DIRECT is deterministic and ignores it).</param>
    /// <param name="directEvals">DIRECT's evaluation budget.</param>
    /// <param name="cmaEvals">CMA-ES's evaluation budget.</param>
    /// <returns>The best point found (in the original box) and its objective value.</returns>
    /// <exception cref="System.ArgumentException">The bounds are empty, mismatched, or an upper bound does not exceed its lower bound.</exception>
    internal static (double[] X, double F) Minimize(
        Func<double[], double> f, double[] lb, double[] ub, int seed, int directEvals, int cmaEvals)
    {
        (double[] xDirect, double fDirect) = Direct.Minimize(f, lb, ub, directEvals);
        (double[] xCma, double fCma) = CmaEsMinimize(f, lb, ub, seed, cmaEvals);
        return fCma < fDirect ? (xCma, fCma) : (xDirect, fDirect);
    }

    /// <summary>
    ///     Minimizes <paramref name="f" /> with CMA-ES, reusing <see cref="Cma" /> as a black-box minimizer.
    ///     The search runs in the normalized unit cube (mean <c>0.5</c>, step size <c>0.2</c>, box bounds
    ///     <c>[0, 1]^k</c> enforced by the core's resampling) and maps each candidate to <c>[lb, ub]</c> when
    ///     evaluating, so one scalar step size is well scaled across heterogeneous coordinate ranges. It runs
    ///     generation by generation until the core signals convergence or the evaluation budget is spent, and
    ///     returns the best center it ever evaluated.
    /// </summary>
    /// <param name="f">The objective over the original box; a non-finite value is treated as very large.</param>
    /// <param name="lb">Per-dimension lower bounds.</param>
    /// <param name="ub">Per-dimension upper bounds.</param>
    /// <param name="seed">Seed for the CMA-ES random stream.</param>
    /// <param name="maxEvals">The evaluation budget.</param>
    /// <returns>The best point found (in the original box) and its objective value.</returns>
    internal static (double[] X, double F) CmaEsMinimize(
        Func<double[], double> f, double[] lb, double[] ub, int seed, int maxEvals)
    {
        int k = lb.Length;

        // Evaluate on the original box from a normalized (and clamped) unit-cube point.
        double Eval(double[] unit)
        {
            var x = new double[k];
            for (int d = 0; d < k; d++)
            {
                double u = unit[d] < 0.0 ? 0.0 : unit[d] > 1.0 ? 1.0 : unit[d];
                x[d] = lb[d] + u * (ub[d] - lb[d]);
            }
            double v = f(x);
            return double.IsNaN(v) || double.IsPositiveInfinity(v) ? double.MaxValue : v;
        }

        var mean = new double[k];
        for (int d = 0; d < k; d++) mean[d] = 0.5;
        var bounds = new double[k, 2];
        for (int d = 0; d < k; d++) { bounds[d, 0] = 0.0; bounds[d, 1] = 1.0; }

        double[] bestUnit = (double[])mean.Clone();
        double bestF = Eval(bestUnit);
        int evals = 1;

        // sigma 0.2 covers the unit box without immediately saturating the [0,1] bounds — the usual choice
        // for CMA-ES on a normalized domain.
        Cma cma;
        try
        {
            cma = new Cma(mean, sigma: 0.2, bounds: bounds, seed: seed);
        }
        catch (ArgumentException)
        {
            // Degenerate dimension (e.g. k below the core's minimum): fall back to the center evaluation.
            return (Unit2Box(bestUnit, lb, ub), bestF);
        }

        int popsize = cma.PopulationSize;
        while (evals < maxEvals && !cma.ShouldStop())
        {
            var solutions = new List<(double[] X, double Value)>(popsize);
            for (int i = 0; i < popsize; i++)
            {
                double[] unit = cma.Ask();
                double val = Eval(unit);
                evals++;
                solutions.Add((unit, val));
                if (val < bestF) { bestF = val; bestUnit = (double[])unit.Clone(); }
            }
            cma.Tell(solutions);
        }

        return (Unit2Box(bestUnit, lb, ub), bestF);
    }

    /// <summary>Maps a (possibly slightly out-of-range) unit-cube point to the original box, clamping to <c>[lb, ub]</c>.</summary>
    /// <param name="unit">The normalized point.</param>
    /// <param name="lb">Per-dimension lower bounds.</param>
    /// <param name="ub">Per-dimension upper bounds.</param>
    /// <returns>The corresponding point in the original box.</returns>
    private static double[] Unit2Box(double[] unit, double[] lb, double[] ub)
    {
        int k = lb.Length;
        var x = new double[k];
        for (int d = 0; d < k; d++)
        {
            double u = unit[d] < 0.0 ? 0.0 : unit[d] > 1.0 ? 1.0 : unit[d];
            x[d] = lb[d] + u * (ub[d] - lb[d]);
        }
        return x;
    }
}
