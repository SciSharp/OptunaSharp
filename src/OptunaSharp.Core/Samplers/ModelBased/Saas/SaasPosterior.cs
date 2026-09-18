// The factored SAAS GP posterior at a fixed hyperparameter sample — the closed-form predictive of
// Eriksson & Jankowiak 2021, Eqns (4)-(5), specialized to the RBF kernel. Produced by
// SaasGaussianProcess.Condition (which factors C = LLᵀ once); the acquisition queries Predict across the
// Q Sobol candidates and PredictWithGrad during the L-BFGS-B polish, averaging expected improvement over
// the bag of posteriors (one per NUTS/MAP hyperparameter sample), Eqn (10).

using System;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.ModelBased.Saas;

/// <summary>
///     A SAAS GP posterior conditioned on the training data at one fixed hyperparameter sample
///     <c>ψ = (ρ, σ²_k)</c> — the closed-form predictive of Eriksson &amp; Jankowiak 2021, Eqns (4)-(5).
///     It caches the Cholesky factor of the training covariance and <c>C⁻¹y</c>, so a single query costs one
///     triangular solve. <see cref="Predict" /> returns the mean/variance the acquisition screens on;
///     <see cref="PredictWithGrad" /> additionally returns their input gradients for the L-BFGS-B EI
///     maximization (Eqn (11)).
///
///     <para>The mean is in the standardized, maximization-oriented coordinate the training targets were
///     given in; the variance is clamped at zero (a query is treated as noise-free, its prior variance being
///     <c>σ²_k</c>).</para>
/// </summary>
internal sealed class SaasPosterior
{
    private readonly SaasGaussianProcess _gp;
    private readonly double[] _rho;       // per-dimension inverse-squared length scales
    private readonly double _kernelVar;   // σ²_k
    private readonly double[,] _cholL;    // lower Cholesky factor of C = K + σ²·I
    private readonly double[] _invY;      // C⁻¹y
    private readonly double[,] _x;        // training inputs (n, dim)
    private readonly int _n;
    private readonly int _dim;

    /// <summary>Constructs the factored posterior. Called by <see cref="SaasGaussianProcess.Condition" />.</summary>
    /// <param name="gp">The GP that produced this factorization (supplies the training inputs).</param>
    /// <param name="rho">The per-dimension inverse-squared length scales of this sample.</param>
    /// <param name="kernelVar">The kernel variance <c>σ²_k</c> of this sample.</param>
    /// <param name="cholL">The lower Cholesky factor of the training covariance.</param>
    /// <param name="invY">The precomputed <c>C⁻¹y</c>.</param>
    internal SaasPosterior(SaasGaussianProcess gp, double[] rho, double kernelVar, double[,] cholL, double[] invY)
    {
        _gp = gp;
        _rho = rho;
        _kernelVar = kernelVar;
        _cholL = cholL;
        _invY = invY;
        _x = gp.Inputs;
        _n = gp.Count;
        _dim = gp.Dim;
    }

    /// <summary>
    ///     The posterior mean and (nonnegative) variance at one query point. With the kernel row
    ///     <c>k_j = σ²_k·exp(−½ Σ_d ρ_d (x_d − X_{j,d})²)</c>: <c>mean = k·C⁻¹y</c> and
    ///     <c>var = σ²_k − k·C⁻¹kᵀ = σ²_k − ‖L⁻¹kᵀ‖²</c> (clamped at zero).
    /// </summary>
    /// <param name="xStar">One normalized query point (length <see cref="SaasGaussianProcess.Dim" />).</param>
    /// <returns>The posterior mean and nonnegative variance at the query point.</returns>
    /// <exception cref="System.ArgumentException"><paramref name="xStar" /> does not have length <c>dim</c>.</exception>
    internal (double Mean, double Var) Predict(double[] xStar)
    {
        if (xStar.Length != _dim)
            throw new ArgumentException($"Query point must have length {_dim}.", nameof(xStar));

        var k = new double[_n];
        for (int j = 0; j < _n; j++)
        {
            double s = 0.0;
            for (int d = 0; d < _dim; d++)
            {
                double diff = xStar[d] - _x[j, d];
                s += _rho[d] * diff * diff;
            }
            k[j] = _kernelVar * Math.Exp(-0.5 * s);
        }

        double mean = 0.0;
        for (int j = 0; j < _n; j++)
            mean += k[j] * _invY[j];

        double[] invLk = LinAlg.SolveTriangular(_cholL, k, upper: false);
        double rCinvR = 0.0;
        for (int j = 0; j < _n; j++)
            rCinvR += invLk[j] * invLk[j];

        double var = Math.Max(0.0, _kernelVar - rCinvR);
        return (mean, var);
    }

    /// <summary>
    ///     The posterior mean and variance at one query point together with their gradients w.r.t. the query
    ///     coordinates — the analytic quantities the EI maximization's L-BFGS-B needs. With
    ///     <c>∂k_j/∂x_d = −σ²_k·M(S_j)·ρ_d·(x_d − X_{j,d})</c> (the RBF kernel derivative) and
    ///     <c>V = C⁻¹kᵀ</c>: <c>∂mean/∂x_d = Σ_j ∂k_j/∂x_d·(C⁻¹y)_j</c> and
    ///     <c>∂var/∂x_d = −2·Σ_j ∂k_j/∂x_d·V_j</c>. Where the variance is clamped at zero its gradient is
    ///     zero (matching the clamp's flat region).
    /// </summary>
    /// <param name="xStar">One normalized query point (length <see cref="SaasGaussianProcess.Dim" />).</param>
    /// <returns>The mean, variance, their per-coordinate gradients, and whether the variance was clamped.</returns>
    /// <exception cref="System.ArgumentException"><paramref name="xStar" /> does not have length <c>dim</c>.</exception>
    internal (double Mean, double Var, double[] DMean, double[] DVar, bool VarClamped) PredictWithGrad(double[] xStar)
    {
        if (xStar.Length != _dim)
            throw new ArgumentException($"Query point must have length {_dim}.", nameof(xStar));

        var k = new double[_n];
        for (int j = 0; j < _n; j++)
        {
            double s = 0.0;
            for (int d = 0; d < _dim; d++)
            {
                double diff = xStar[d] - _x[j, d];
                s += _rho[d] * diff * diff;
            }
            k[j] = _kernelVar * Math.Exp(-0.5 * s);
        }

        double mean = 0.0;
        for (int j = 0; j < _n; j++)
            mean += k[j] * _invY[j];

        // V = C⁻¹kᵀ via L⁻ᵀ(L⁻¹k).
        double[] invLk = LinAlg.SolveTriangular(_cholL, k, upper: false);
        double[] v = LinAlg.SolveTriangular(SaasGaussianProcess.Transpose(_cholL), invLk, upper: true);

        double rawVar = _kernelVar;
        for (int j = 0; j < _n; j++)
            rawVar -= k[j] * v[j];
        bool clamped = rawVar < 0.0;
        double variance = clamped ? 0.0 : rawVar;

        var dMean = new double[_dim];
        var dVar = new double[_dim];
        for (int d = 0; d < _dim; d++)
        {
            double dm = 0.0;
            double dv = 0.0;
            for (int j = 0; j < _n; j++)
            {
                // ∂k_j/∂x_d = σ²_k·M(S_j)·(−ρ_d·(x_d − X_{j,d})) = −k_j·ρ_d·(x_d − X_{j,d}).
                double dkj = -k[j] * _rho[d] * (xStar[d] - _x[j, d]);
                dm += dkj * _invY[j];
                dv += dkj * v[j];
            }
            dMean[d] = dm;
            dVar[d] = clamped ? 0.0 : -2.0 * dv;
        }
        return (mean, variance, dMean, dVar, clamped);
    }
}
