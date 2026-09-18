// SAASBO-MAP inference (Eriksson & Jankowiak 2021, §4.2.2 & §A.2) — the faster point-estimate alternative
// to NUTS. It learns separate GP models for a small set S of pre-selected shrinkage values τ_s (default
// {1, 1e-1, 1e-2, 1e-3}), each by maximizing the log posterior U_s(ψ|τ_s) with Adam (1500 steps, lr 0.02,
// β1 0.50 — the paper's A.2 settings), then selects the single ψ_s that maximizes the leave-one-out
// predictive log likelihood. MAP does not require the α hyperparameter (the τ grid replaces the τ prior).

using System;
using System.Collections.Generic;

namespace OptunaSharp.Samplers.ModelBased.Saas;

/// <summary>
///     SAASBO-MAP: maximum-a-posteriori inference for the SAAS GP (Eriksson &amp; Jankowiak 2021, §A.2). For
///     each pre-selected global-shrinkage value <c>τ_s</c> it runs Adam to maximize the log posterior over
///     the kernel hyperparameters <c>ψ = (ρ, σ²_k)</c>, then picks the single model whose leave-one-out
///     predictive log likelihood is largest. The result is one hyperparameter point estimate — the "bag" of
///     size <c>L = 1</c> the acquisition averages EI over. It is faster than NUTS but, as the paper notes,
///     less robust (a single posterior point rather than a spread), so NUTS is SAASBO's default.
/// </summary>
internal static class SaasMapOptimizer
{
    /// <summary>The paper's pre-selected shrinkage grid <c>τ_s ∈ {1, 1e-1, 1e-2, 1e-3}</c> (§A.2).</summary>
    internal static readonly double[] DefaultTaus = { 1.0, 1e-1, 1e-2, 1e-3 };

    /// <summary>
    ///     Runs MAP over the shrinkage grid and returns the leave-one-out-selected hyperparameters. Each
    ///     model is optimized independently with Adam from a prior-centered start (<c>ρ_i = τ_s</c>,
    ///     <c>σ²_k = 1</c>); the selection then re-scores each converged model by its LOO predictive log
    ///     likelihood (§A.2) and returns the best. A non-finite objective at convergence excludes that model
    ///     from selection (its LOO is treated as <c>−∞</c>).
    /// </summary>
    /// <param name="model">The SAAS model wrapping the GP and priors.</param>
    /// <param name="taus">The pre-selected shrinkage values (default <see cref="DefaultTaus" />).</param>
    /// <param name="steps">The number of Adam iterations per model (paper: 1500).</param>
    /// <param name="learningRate">The Adam learning rate (paper: 0.02).</param>
    /// <param name="beta1">The Adam first-moment decay (paper: 0.50).</param>
    /// <param name="beta2">The Adam second-moment decay (Adam default 0.999).</param>
    /// <param name="epsilon">The Adam denominator floor (Adam default 1e-8).</param>
    /// <returns>The hyperparameters of the LOO-selected model.</returns>
    /// <exception cref="System.ArgumentException"><paramref name="taus" /> is empty, or a budget is non-positive.</exception>
    /// <exception cref="System.InvalidOperationException">No model yielded a finite leave-one-out score (a degenerate problem).</exception>
    internal static SaasHyperparameters Optimize(
        SaasModel model, IReadOnlyList<double>? taus = null,
        int steps = 1500, double learningRate = 0.02, double beta1 = 0.50, double beta2 = 0.999, double epsilon = 1e-8)
    {
        taus ??= DefaultTaus;
        if (taus.Count == 0)
            throw new ArgumentException("At least one tau must be supplied.", nameof(taus));
        if (steps < 1)
            throw new ArgumentException("steps must be at least 1.", nameof(steps));

        int dim = model.Dim;
        double bestLoo = double.NegativeInfinity;
        SaasHyperparameters? best = null;

        foreach (double tau in taus)
        {
            SaasHyperparameters hp = AdamMaximize(model, tau, dim, steps, learningRate, beta1, beta2, epsilon);
            double loo = model.Gp.LeaveOneOutLogLikelihood(hp);
            if (loo > bestLoo)
            {
                bestLoo = loo;
                best = hp;
            }
        }

        // Every model degenerated (e.g. a constant target after standardization) — the caller cannot proceed.
        return best ?? throw new InvalidOperationException(
            "SAASBO-MAP found no model with a finite leave-one-out score.");
    }

    /// <summary>
    ///     Maximizes the fixed-<c>τ</c> MAP objective <c>U_s(ψ|τ)</c> with Adam in the log-coordinate
    ///     <c>q = [log ρ, log σ²_k]</c>, starting from the prior-centered point <c>ρ_i = τ</c>,
    ///     <c>σ²_k = 1</c>. Adam ascends (the objective is a value to maximize), and any non-finite gradient
    ///     step is skipped so a transient infeasible covariance does not corrupt the moment estimates.
    /// </summary>
    /// <param name="model">The SAAS model.</param>
    /// <param name="tau">The fixed shrinkage value for this model.</param>
    /// <param name="dim">The input dimension <c>D</c>.</param>
    /// <param name="steps">The number of Adam iterations.</param>
    /// <param name="lr">The Adam learning rate.</param>
    /// <param name="beta1">The Adam first-moment decay.</param>
    /// <param name="beta2">The Adam second-moment decay.</param>
    /// <param name="epsilon">The Adam denominator floor.</param>
    /// <returns>The converged hyperparameters for this <paramref name="tau" />.</returns>
    private static SaasHyperparameters AdamMaximize(
        SaasModel model, double tau, int dim, int steps, double lr, double beta1, double beta2, double epsilon)
    {
        // q = [log ρ_1..D, log σ²_k]; prior-centered start ρ_i = τ, σ²_k = 1.
        var q = new double[dim + 1];
        double logTau = Math.Log(tau);
        for (int i = 0; i < dim; i++) q[i] = logTau;
        q[dim] = 0.0;

        var m = new double[dim + 1];   // Adam first moment
        var v = new double[dim + 1];   // Adam second moment
        for (int t = 1; t <= steps; t++)
        {
            (double obj, double[] grad) = model.MapObjectiveAndGrad(q, tau);
            if (double.IsNaN(obj) || double.IsInfinity(obj))
                break;   // the objective diverged; keep the last good q.

            double bc1 = 1.0 - Math.Pow(beta1, t);
            double bc2 = 1.0 - Math.Pow(beta2, t);
            for (int d = 0; d < q.Length; d++)
            {
                double g = grad[d];
                if (double.IsNaN(g) || double.IsInfinity(g))
                    g = 0.0;   // skip a bad coordinate rather than poisoning the moments.
                m[d] = beta1 * m[d] + (1.0 - beta1) * g;
                v[d] = beta2 * v[d] + (1.0 - beta2) * g * g;
                double mHat = m[d] / bc1;
                double vHat = v[d] / bc2;
                q[d] += lr * mHat / (Math.Sqrt(vHat) + epsilon);   // ascent (maximize U).
            }
        }
        return model.HyperparametersFromMapCoord(q);
    }
}
