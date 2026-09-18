// The SAAS probabilistic model — the sparsity-inducing prior of Eriksson & Jankowiak 2021 (§4.1, Eqn (8))
// wrapped around the RBF ARD GP, exposing exactly the two log-densities inference needs:
//
//   * the reparameterized log-joint over the UNCONSTRAINED latents θ = [log τ, log ρ̃_1..D, log σ²_k],
//     used by NUTS (§4.2.1). The reparameterization ρ_i = τ·ρ̃_i with ρ̃_i ~ HC(1) (Eqn (12)) is the
//     paper's device for a better-conditioned HMC geometry;
//   * the MAP objective U_s(ψ | τ_s) = log p(y|X,ψ) + log p(ψ|τ_s) at a FIXED shrinkage τ_s (§4.2.2 /
//     §A.2), maximized over ψ = (ρ, σ²_k) with Adam.
//
// SAAS prior (Eqn (8)):  σ²_k ~ LogNormal(0, 10²);  τ ~ HalfCauchy(α);  ρ_i ~ HalfCauchy(τ);  noise → 1e-6.
// Half-Cauchy density (scale s):  p(t|s) ∝ (s² + t²)⁻¹ · 1(t>0),  normalized  p(t|s) = (2 / (π s)) / (1 + (t/s)²).

using System;

namespace OptunaSharp.Samplers.ModelBased.Saas;

/// <summary>
///     The SAAS model (Eriksson &amp; Jankowiak 2021, §4.1) over a fixed <see cref="SaasGaussianProcess" />:
///     the RBF ARD marginal likelihood plus the sparsity-inducing SAAS prior on <c>(σ²_k, τ, ρ)</c>. It
///     supplies the two objectives SAASBO inference optimizes — the reparameterized unconstrained log-joint
///     for NUTS (<see cref="LogJointAndGrad" />) and the fixed-<c>τ</c> MAP objective for Adam
///     (<see cref="MapObjectiveAndGrad" />) — with fully analytic gradients (no autodiff).
///
///     <para><b>Layout of the NUTS latent <c>θ</c>.</b> <c>θ[0] = log τ</c>; <c>θ[1..D] = log ρ̃_i</c>
///     (the reparameterized length scales, <c>ρ_i = τ·ρ̃_i</c>); <c>θ[D+1] = log σ²_k</c>. The half-Cauchy
///     priors on <c>τ</c> and <c>ρ̃_i</c> and the log-Normal prior on <c>σ²_k</c> are all expressed in this
///     log coordinate (each transform's Jacobian is folded into the log-density), so <c>θ</c> is
///     unconstrained and HMC needs no boundary handling.</para>
/// </summary>
internal sealed class SaasModel
{
    private const double LogTwoOverPi = -0.4515827052894548;   // log(2/π), the half-Cauchy normalizer constant

    private readonly SaasGaussianProcess _gp;
    private readonly double _alpha;              // global-shrinkage half-Cauchy scale α (default 0.1)
    private readonly double _kernelVarLogStd;    // σ²_k ~ LogNormal(0, kernelVarLogStd²) — the paper's 10
    private readonly int _dim;

    /// <summary>The dimension of the NUTS latent <c>θ</c>: <c>D + 2</c> (τ, the <c>D</c> length scales, σ²_k).</summary>
    internal int LatentDim => _dim + 2;

    /// <summary>The underlying GP — MAP model selection scores each candidate's leave-one-out predictive likelihood on it.</summary>
    internal SaasGaussianProcess Gp => _gp;

    /// <summary>The input dimension <c>D</c> of the modeled search space.</summary>
    internal int Dim => _dim;

    /// <summary>The global-shrinkage hyperparameter <c>α</c> of the half-Cauchy prior on <c>τ</c> (Eqn (8)).</summary>
    internal double Alpha => _alpha;

    /// <summary>Constructs the SAAS model over a GP.</summary>
    /// <param name="gp">The RBF ARD GP holding the (standardized, maximization-oriented) training data.</param>
    /// <param name="alpha">The global-shrinkage half-Cauchy scale <c>α &gt; 0</c> (paper default <c>0.1</c>); smaller
    ///     is more conservative (prefers smaller subspaces).</param>
    /// <param name="kernelVarLogStd">The standard deviation of the log-Normal prior on <c>σ²_k</c> (paper's <c>10</c>).</param>
    /// <exception cref="System.ArgumentException"><paramref name="alpha" /> or <paramref name="kernelVarLogStd" /> is not positive.</exception>
    internal SaasModel(SaasGaussianProcess gp, double alpha = 0.1, double kernelVarLogStd = 10.0)
    {
        if (!(alpha > 0.0))
            throw new ArgumentException("alpha must be positive.", nameof(alpha));
        if (!(kernelVarLogStd > 0.0))
            throw new ArgumentException("kernelVarLogStd must be positive.", nameof(kernelVarLogStd));
        _gp = gp;
        _alpha = alpha;
        _kernelVarLogStd = kernelVarLogStd;
        _dim = gp.Dim;
    }

    /// <summary>
    ///     Maps a NUTS latent <c>θ</c> to the kernel hyperparameters <c>(ρ, σ²_k)</c> the GP consumes:
    ///     <c>τ = e^{θ_0}</c>, <c>ρ̃_i = e^{θ_{1+i}}</c>, <c>ρ_i = τ·ρ̃_i</c>, <c>σ²_k = e^{θ_{D+1}}</c>.
    /// </summary>
    /// <param name="theta">The unconstrained latent vector (length <see cref="LatentDim" />).</param>
    /// <returns>The kernel hyperparameters at <paramref name="theta" />.</returns>
    /// <exception cref="System.ArgumentException"><paramref name="theta" /> does not have length <see cref="LatentDim" />.</exception>
    internal SaasHyperparameters HyperparametersFromTheta(double[] theta)
    {
        if (theta.Length != LatentDim)
            throw new ArgumentException($"theta length must be {LatentDim}.", nameof(theta));
        double tau = Math.Exp(theta[0]);
        var rho = new double[_dim];
        for (int i = 0; i < _dim; i++)
            rho[i] = tau * Math.Exp(theta[1 + i]);
        double kernelVar = Math.Exp(theta[_dim + 1]);
        return new SaasHyperparameters(rho, kernelVar);
    }

    /// <summary>
    ///     The reparameterized SAAS log-joint <c>log p(y | X, ψ(θ)) + log p(θ)</c> and its gradient w.r.t.
    ///     the unconstrained latent <c>θ</c> — the target density for NUTS (Eqn (9), in the reparameterized
    ///     coordinates of Eqn (12)). The gradient chains the GP's marginal-likelihood gradient through
    ///     <c>ρ = τ·ρ̃</c> and adds the analytic prior-plus-Jacobian derivative of each latent.
    ///
    ///     <para>When the covariance will not factor the log-joint is <see cref="double.NegativeInfinity" />
    ///     with a zero GP-likelihood gradient (the prior gradient is still returned), so a leapfrog step into
    ///     an infeasible region is rejected by the Metropolis correction rather than throwing.</para>
    /// </summary>
    /// <param name="theta">The unconstrained latent vector (length <see cref="LatentDim" />).</param>
    /// <returns>The log-joint density and its gradient in <c>θ</c> coordinates.</returns>
    /// <exception cref="System.ArgumentException"><paramref name="theta" /> does not have length <see cref="LatentDim" />.</exception>
    internal (double LogJoint, double[] Grad) LogJointAndGrad(double[] theta)
    {
        if (theta.Length != LatentDim)
            throw new ArgumentException($"theta length must be {LatentDim}.", nameof(theta));

        double logTau = theta[0];
        double tau = Math.Exp(logTau);
        double logKernelVar = theta[_dim + 1];
        double kernelVar = Math.Exp(logKernelVar);

        var rho = new double[_dim];
        for (int i = 0; i < _dim; i++)
            rho[i] = tau * Math.Exp(theta[1 + i]);

        (double mll, double[] dRho, double dKernelVar) =
            _gp.MarginalLogLikelihoodAndGrad(new SaasHyperparameters(rho, kernelVar));

        var grad = new double[LatentDim];

        // ---- prior: τ ~ HalfCauchy(α) in log-space (θ_0 = log τ) ----
        double logJoint = HalfCauchyLogInLog(logTau, _alpha);
        grad[0] = HalfCauchyLogInLogDeriv(logTau, _alpha);

        // ---- prior: ρ̃_i ~ HalfCauchy(1) in log-space (θ_{1+i} = log ρ̃_i) ----
        for (int i = 0; i < _dim; i++)
        {
            double logRhoTilde = theta[1 + i];
            logJoint += HalfCauchyLogInLog(logRhoTilde, 1.0);
            grad[1 + i] = HalfCauchyLogInLogDeriv(logRhoTilde, 1.0);
        }

        // ---- prior: σ²_k ~ LogNormal(0, kernelVarLogStd²); in θ_{D+1} = log σ²_k this is N(·; 0, std²) ----
        double invVar = 1.0 / (_kernelVarLogStd * _kernelVarLogStd);
        logJoint += -0.5 * logKernelVar * logKernelVar * invVar
                    - 0.5 * Math.Log(2.0 * Math.PI * _kernelVarLogStd * _kernelVarLogStd);
        grad[_dim + 1] = -logKernelVar * invVar;

        // ---- likelihood + chain to θ ----
        // An infeasible covariance zeroes the likelihood gradient but keeps the (finite) prior gradient, so
        // the leapfrog still has a direction to leave the infeasible region on the next Metropolis proposal.
        if (double.IsNegativeInfinity(mll))
            return (double.NegativeInfinity, grad);

        logJoint += mll;
        // ∂ρ_i/∂(log τ) = τ·ρ̃_i = ρ_i;  ∂ρ_i/∂(log ρ̃_i) = ρ_i;  ∂σ²_k/∂(log σ²_k) = σ²_k.
        double dTau = 0.0;
        for (int i = 0; i < _dim; i++)
        {
            double contrib = dRho[i] * rho[i];
            dTau += contrib;
            grad[1 + i] += contrib;
        }
        grad[0] += dTau;
        grad[_dim + 1] += dKernelVar * kernelVar;

        return (logJoint, grad);
    }

    /// <summary>
    ///     Maps a MAP optimization coordinate <c>q = [log ρ_1..D, log σ²_k]</c> (length <c>D+1</c>, at a
    ///     fixed <c>τ</c>) to the kernel hyperparameters.
    /// </summary>
    /// <param name="q">The MAP log-coordinate (length <c>D+1</c>).</param>
    /// <returns>The kernel hyperparameters at <paramref name="q" />.</returns>
    /// <exception cref="System.ArgumentException"><paramref name="q" /> does not have length <c>D+1</c>.</exception>
    internal SaasHyperparameters HyperparametersFromMapCoord(double[] q)
    {
        if (q.Length != _dim + 1)
            throw new ArgumentException($"MAP coordinate length must be {_dim + 1}.", nameof(q));
        var rho = new double[_dim];
        for (int i = 0; i < _dim; i++)
            rho[i] = Math.Exp(q[i]);
        double kernelVar = Math.Exp(q[_dim]);
        return new SaasHyperparameters(rho, kernelVar);
    }

    /// <summary>
    ///     The MAP objective <c>U_s(ψ | τ_s) = log p(y|X,ψ) + log p(ψ|τ_s)</c> (§A.2, Eqn (13)) and its
    ///     gradient in the log-coordinate <c>q</c>, at the FIXED shrinkage <paramref name="tau" />. This is
    ///     the mode-in-ψ objective the paper maximizes over <c>ψ = (ρ, σ²_k)</c> with Adam: the prior on
    ///     <c>ρ_i</c> is <c>HalfCauchy(τ_s)</c> and on <c>σ²_k</c> is the log-Normal; there is no Jacobian
    ///     term (the target is the mode in the constrained parameter, so optimizing in <c>q = log ψ</c> only
    ///     chains the constrained-space gradient through <c>∂ψ/∂q = ψ</c>).
    /// </summary>
    /// <param name="q">The MAP log-coordinate <c>[log ρ_1..D, log σ²_k]</c> (length <c>D+1</c>).</param>
    /// <param name="tau">The fixed global-shrinkage value <c>τ_s &gt; 0</c> for this model.</param>
    /// <returns>The MAP objective value and its gradient in <paramref name="q" /> coordinates.</returns>
    /// <exception cref="System.ArgumentException"><paramref name="q" /> length is wrong, or <paramref name="tau" /> is not positive.</exception>
    internal (double Objective, double[] Grad) MapObjectiveAndGrad(double[] q, double tau)
    {
        if (q.Length != _dim + 1)
            throw new ArgumentException($"MAP coordinate length must be {_dim + 1}.", nameof(q));
        if (!(tau > 0.0))
            throw new ArgumentException("tau must be positive.", nameof(tau));

        var rho = new double[_dim];
        for (int i = 0; i < _dim; i++)
            rho[i] = Math.Exp(q[i]);
        double kernelVar = Math.Exp(q[_dim]);

        (double mll, double[] dRho, double dKernelVar) =
            _gp.MarginalLogLikelihoodAndGrad(new SaasHyperparameters(rho, kernelVar));

        var grad = new double[_dim + 1];
        double obj = mll;

        // ρ_i ~ HalfCauchy(τ): log p(ρ|τ) = log(2/π) − log τ − log(1 + (ρ/τ)²); d/dρ = −2ρ/(τ² + ρ²).
        double tauSq = tau * tau;
        for (int i = 0; i < _dim; i++)
        {
            double r = rho[i];
            obj += LogTwoOverPi - Math.Log(tau) - Math.Log(1.0 + r * r / tauSq);
            double dObjDRho = dRho[i] - 2.0 * r / (tauSq + r * r);
            grad[i] = dObjDRho * r;   // chain through ∂ρ/∂(log ρ) = ρ.
        }

        // σ²_k ~ LogNormal(0, std²): log p = −log σ²_k − ½log(2π std²) − (log σ²_k)²/(2 std²).
        double std = _kernelVarLogStd;
        double logKv = q[_dim];
        obj += -logKv - 0.5 * Math.Log(2.0 * Math.PI * std * std) - 0.5 * logKv * logKv / (std * std);
        double dObjDKernelVar = dKernelVar;                    // ∂mll/∂σ²_k
        // chain: ∂/∂(log σ²_k) = σ²_k·∂mll/∂σ²_k + σ²_k·(−1/σ²_k − logσ²_k/(std²·σ²_k)) = σ²_k·∂mll − 1 − logσ²_k/std².
        grad[_dim] = kernelVar * dObjDKernelVar - 1.0 - logKv / (std * std);

        return (obj, grad);
    }

    /// <summary>
    ///     The log-density of <c>u = log t</c> where <c>t ~ HalfCauchy(scale)</c> — i.e. the half-Cauchy
    ///     log-density plus the <c>log|dt/du| = u</c> Jacobian: <c>log(2/π) − log s − log(1 + (e^u/s)²) + u</c>.
    /// </summary>
    /// <param name="u">The log of the (positive) half-Cauchy variate.</param>
    /// <param name="scale">The half-Cauchy scale <c>s &gt; 0</c>.</param>
    /// <returns>The log-density in the <c>u = log t</c> coordinate.</returns>
    internal static double HalfCauchyLogInLog(double u, double scale)
    {
        double r = Math.Exp(u) / scale;
        return LogTwoOverPi - Math.Log(scale) - Math.Log(1.0 + r * r) + u;
    }

    /// <summary>
    ///     The derivative of <see cref="HalfCauchyLogInLog" /> w.r.t. <c>u</c>: <c>1 − 2r²/(1 + r²)</c> with
    ///     <c>r = e^u/s</c>.
    /// </summary>
    /// <param name="u">The log of the (positive) half-Cauchy variate.</param>
    /// <param name="scale">The half-Cauchy scale <c>s &gt; 0</c>.</param>
    /// <returns>The derivative of the log-density in the <c>u = log t</c> coordinate.</returns>
    internal static double HalfCauchyLogInLogDeriv(double u, double scale)
    {
        double r = Math.Exp(u) / scale;
        double rSq = r * r;
        return 1.0 - 2.0 * rSq / (1.0 + rSq);
    }
}
