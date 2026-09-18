// SAASBO's surrogate — the RBF (squared-exponential) ARD Gaussian process of Eriksson & Jankowiak,
// "High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces" (UAI 2021, arXiv:2103.00349),
// §3.1 Eqn (1) and §4.1 Eqn (8). This is a NEW, beyond-core class (a fork of the model-based toolkit's GP
// machinery, NOT a copy of it) so SAASBO can carry exactly the paper's surrogate — an RBF kernel with a
// PER-DIMENSION inverse-squared length scale ρ_i and an EXPLICIT kernel variance σ²_k that the SAAS prior
// governs — without touching the Matérn-5/2 GPRegressor that TuRBO/REMBO/GPSampler depend on (the project
// rule: clone rather than modify a preexisting component when a sampler needs it different).
//
//   * §3.1 Eqn (1) (RBF kernel):  k_ψ(x, y) = σ²_k · exp{ −½ Σ_i ρ_i (x_i − y_i)² }
//                                 — ρ_i are inverse SQUARED length scales (ρ_i = 1/ℓ_i²), one per dimension,
//                                   and σ²_k is the amplitude/kernel variance (a modeled hyperparameter with
//                                   a LogNormal prior, NOT profiled out as in REMBO's SquaredExponentialGp).
//   * §4.1: the objective is noise-free, so the observation noise σ² is fixed to a tiny constant (1e-6).
//
// The class holds the GP math at a FIXED hyperparameter setting (marginal log likelihood + its analytic
// gradient w.r.t. (ρ, σ²_k) for MAP/NUTS inference, leave-one-out predictive log likelihood for MAP model
// selection) plus a factory (Condition) that factors the covariance once so the posterior can be queried
// cheaply across the many acquisition candidates. Linear algebra rides the repo's OpenBLAS-backed LinAlg
// (Cholesky / triangular solve), so results are tolerance-reproducible on a platform — SAASBO is a
// behavioral, paper-faithful port, not a byte-exact match against a foreign GP/NUTS library.

using System;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.ModelBased.Saas;

/// <summary>
///     A point in SAASBO's kernel-hyperparameter space: the per-dimension inverse-squared length scales
///     <c>ρ</c> (Eriksson &amp; Jankowiak 2021, Eqn (1)) and the scalar kernel variance <c>σ²_k</c>. The
///     observation noise is not carried here because SAASBO fixes it to a tiny constant (noise-free
///     objective, §4.1). A single one of these represents either a NUTS posterior draw <c>ψ_ℓ</c> or a MAP
///     point estimate <c>ψ_s</c>; the acquisition averages EI over a bag of them (Eqn (10)).
/// </summary>
/// <param name="Rho">The per-dimension inverse-squared length scales <c>ρ_i &gt; 0</c> (length = input dimension);
///     large <c>ρ_i</c> means dimension <c>i</c> is "on" (the SAAS prior keeps most near zero).</param>
/// <param name="KernelVariance">The kernel amplitude <c>σ²_k &gt; 0</c> multiplying the RBF covariance.</param>
internal readonly record struct SaasHyperparameters(double[] Rho, double KernelVariance);

/// <summary>
///     SAASBO's RBF ARD Gaussian process over a standardized, maximization-oriented objective on a
///     normalized <c>[0, 1]^D</c> input box (Eriksson &amp; Jankowiak 2021, §3.1 &amp; §4.1). It stores the
///     training data and the precomputed pairwise squared differences, and evaluates the GP quantities
///     inference needs at an arbitrary hyperparameter setting: the marginal log likelihood and its analytic
///     gradient (<see cref="MarginalLogLikelihoodAndGrad" />), the leave-one-out predictive log likelihood
///     for MAP model selection (<see cref="LeaveOneOutLogLikelihood" />), and a factored
///     <see cref="SaasPosterior" /> (<see cref="Condition" />) for the acquisition.
///
///     <para><b>Not the Matérn GPRegressor.</b> The sibling <see cref="Gp.GPRegressor" /> in the model-based
///     toolkit is a Matérn-5/2 ARD GP (TuRBO's paper choice); this class exists because SAASBO's paper
///     specifies the RBF kernel with an explicit prior-governed kernel variance, which that GP does not
///     express. The kernel-generic marginal-likelihood gradient (<c>∂/∂ρ</c>, <c>∂/∂σ²_k</c>) is the same
///     algebra as <see cref="Gp.GPRegressor.LossAndGrad" />, specialized to the RBF value/derivative
///     <c>M(S) = e^{−S/2}</c>, <c>M'(S) = −½e^{−S/2}</c>.</para>
///
///     <para><b>Noise-free numerics.</b> The fixed noise is a tiny <c>1e-6</c> (§4.1), so the covariance can
///     be ill-conditioned once the search concentrates; a jitter ladder is added until the Cholesky
///     succeeds. When even the largest jitter fails, the marginal likelihood is reported as
///     <see cref="double.NegativeInfinity" /> (with a zero gradient) so a NUTS/MAP optimizer treats that
///     region as infeasible rather than throwing.</para>
/// </summary>
internal sealed class SaasGaussianProcess
{
    // Jitter added to the covariance diagonal (on top of the fixed noise) until the Cholesky factors —
    // escalated only when near-duplicate training points make C numerically singular at the noise floor.
    private static readonly double[] JitterLevels = { 0.0, 1e-8, 1e-6, 1e-4, 1e-2 };

    private readonly double[,] _x;             // (n, dim) normalized training inputs
    private readonly double[] _y;              // (n,) standardized, maximization-oriented targets
    private readonly double[,,] _squaredXDiff; // (n, n, dim): (x_i - x_j)² per coordinate
    private readonly double _noiseVar;         // fixed observation noise σ² (1e-6 for the noise-free paper)
    private readonly int _n;
    private readonly int _dim;

    /// <summary>The number of training observations the GP is conditioned on.</summary>
    internal int Count => _n;

    /// <summary>The input dimension <c>D</c> — the length of a hyperparameter's <c>ρ</c> vector.</summary>
    internal int Dim => _dim;

    /// <summary>The fixed observation-noise variance <c>σ²</c> added to the covariance diagonal.</summary>
    internal double NoiseVar => _noiseVar;

    /// <summary>The standardized training targets (maximization-oriented), retained by reference.</summary>
    internal double[] Targets => _y;

    /// <summary>The normalized training inputs (<c>n × dim</c>), retained by reference.</summary>
    internal double[,] Inputs => _x;

    /// <summary>
    ///     Builds a GP over fixed training observations. The targets must already be <b>standardized</b>
    ///     (zero mean, unit variance) and <b>maximization-oriented</b> — the caller sign-flips a minimizing
    ///     study — matching the coordinate the acquisition's incumbent is expressed in.
    /// </summary>
    /// <param name="x">Normalized training inputs in <c>[0, 1]^dim</c>, one point per row (<c>n × dim</c>, <c>n ≥ 1</c>).</param>
    /// <param name="y">Standardized, maximization-oriented targets, length <c>n</c>.</param>
    /// <param name="noiseVar">The fixed observation-noise variance (SAASBO uses <c>1e-6</c>; must be positive).</param>
    /// <exception cref="System.ArgumentException">The shapes are inconsistent, or <paramref name="noiseVar" /> is not positive.</exception>
    internal SaasGaussianProcess(double[,] x, double[] y, double noiseVar)
    {
        _n = x.GetLength(0);
        _dim = x.GetLength(1);
        if (_n < 1)
            throw new ArgumentException("At least one training point is required.", nameof(x));
        if (y.Length != _n)
            throw new ArgumentException("y length must equal the number of training rows.", nameof(y));
        if (!(noiseVar > 0.0))
            throw new ArgumentException("noiseVar must be positive.", nameof(noiseVar));

        _x = x;
        _y = y;
        _noiseVar = noiseVar;

        // squared_X_diff[i,j,d] = (x_i,d − x_j,d)². SAASBO models continuous/integer coordinates only, so
        // there is no categorical (Hamming) branch here — categoricals are delegated by the sampler.
        _squaredXDiff = new double[_n, _n, _dim];
        for (int i = 0; i < _n; i++)
            for (int j = 0; j < _n; j++)
                for (int d = 0; d < _dim; d++)
                {
                    double diff = _x[i, d] - _x[j, d];
                    _squaredXDiff[i, j, d] = diff * diff;
                }
    }

    /// <summary>
    ///     The RBF kernel value <c>M(S) = e^{−S/2}</c> and derivative <c>M'(S) = dM/dS = −½e^{−S/2}</c> at
    ///     the length-scale-weighted squared distance <c>S = Σ_d ρ_d (x_d − y_d)²</c>. Splitting the value
    ///     and its <c>S</c>-derivative lets the marginal-likelihood and posterior gradients reuse one
    ///     exponential per pair (mirroring <see cref="Gp.GPRegressor" />'s Matérn helper).
    /// </summary>
    /// <param name="s">The nonnegative weighted squared distance between two input points.</param>
    /// <returns>The RBF kernel value and its derivative with respect to the squared distance.</returns>
    private static (double Value, double Deriv) Rbf(double s)
    {
        double e = Math.Exp(-0.5 * s);
        return (e, -0.5 * e);
    }

    /// <summary>Weighted pairwise squared distance <c>S[i,j] = Σ_d squaredXDiff[i,j,d] · ρ_d</c>.</summary>
    /// <param name="rho">The per-dimension inverse-squared length scales.</param>
    /// <returns>The <c>n × n</c> matrix of weighted squared distances between training rows.</returns>
    private double[,] WeightedSquaredDistance(double[] rho)
    {
        var s = new double[_n, _n];
        for (int i = 0; i < _n; i++)
            for (int j = 0; j < _n; j++)
            {
                double acc = 0.0;
                for (int d = 0; d < _dim; d++)
                    acc += _squaredXDiff[i, j, d] * rho[d];
                s[i, j] = acc;
            }
        return s;
    }

    /// <summary>
    ///     The marginal log likelihood <c>log p(y | X, ψ)</c> (Eqn (3)) and its analytic gradient w.r.t. the
    ///     hyperparameters <c>(ρ, σ²_k)</c> at the setting <paramref name="hp" />. With
    ///     <c>S = D·ρ</c>, <c>K = σ²_k·M(S)</c>, <c>C = K + σ²I</c>, <c>C = L·Lᵀ</c>, <c>α = C⁻¹y</c>,
    ///     <c>G = ½(ααᵀ − C⁻¹)</c>:
    ///     <list type="bullet">
    ///       <item><c>MLL = −½ logdet C − ½ yᵀC⁻¹y − ½ n log 2π</c>,</item>
    ///       <item><c>∂MLL/∂ρ_d = σ²_k · Σ_{ij} (G ⊙ M'(S))[i,j] · D[i,j,d]</c>,</item>
    ///       <item><c>∂MLL/∂σ²_k = Σ_{ij} G[i,j] · M(S)[i,j]</c>.</item>
    ///     </list>
    ///     These are the exploited quantities for both MAP (Adam) and NUTS (leapfrog); the reparameterized
    ///     SAAS log-joint (<see cref="SaasModel" />) chains them through <c>ρ = τ·ρ̃</c> and adds the prior.
    ///
    ///     <para><b>Infeasible regions.</b> If the covariance will not Cholesky-factor even at the largest
    ///     jitter, the likelihood is <see cref="double.NegativeInfinity" /> and the gradient is all zeros —
    ///     so an optimizer/sampler rejects the point rather than propagating a <c>LinAlgError</c>.</para>
    /// </summary>
    /// <param name="hp">The hyperparameter setting <c>(ρ, σ²_k)</c> at which to evaluate.</param>
    /// <returns>The marginal log likelihood, its per-dimension <c>ρ</c> gradient, and its <c>σ²_k</c> gradient.</returns>
    /// <exception cref="System.ArgumentException"><paramref name="hp" />'s <c>ρ</c> length does not equal <see cref="Dim" />.</exception>
    internal (double Mll, double[] DRho, double DKernelVariance) MarginalLogLikelihoodAndGrad(SaasHyperparameters hp)
    {
        double[] rho = hp.Rho;
        double sigma = hp.KernelVariance;
        if (rho.Length != _dim)
            throw new ArgumentException($"rho length must equal the input dimension {_dim}.", nameof(hp));

        double[,] sqd = WeightedSquaredDistance(rho);

        // forward: M(S), M'(S), C = σ²_k·M(S) + σ²·I.
        var mVal = new double[_n, _n];
        var mDer = new double[_n, _n];
        var c = new double[_n, _n];
        for (int i = 0; i < _n; i++)
            for (int j = 0; j < _n; j++)
            {
                (double v, double dv) = Rbf(sqd[i, j]);
                mVal[i, j] = v;
                mDer[i, j] = dv;
                c[i, j] = sigma * v + (i == j ? _noiseVar : 0.0);
            }

        if (!TryFactor(c, out double[,] l))
            return (double.NegativeInfinity, new double[_dim], 0.0);

        // MLL = −½ logdet C − ½ yᵀC⁻¹y − ½ n log 2π, using logdet C = 2·Σ log L_ii and yᵀC⁻¹y = ‖L⁻¹y‖².
        double logdet = 0.0;
        for (int i = 0; i < _n; i++)
            logdet += 2.0 * Math.Log(l[i, i]);
        double[] invLy = LinAlg.SolveTriangular(l, _y, upper: false);
        var invLySq = new double[_n];
        for (int i = 0; i < _n; i++)
            invLySq[i] = invLy[i] * invLy[i];
        double mll = -0.5 * logdet - 0.5 * MathUtil.PairwiseSum(invLySq) - 0.5 * _n * Math.Log(2.0 * Math.PI);

        // backward: C⁻¹ = L⁻ᵀL⁻¹, α = C⁻¹y, G = ½(ααᵀ − C⁻¹).
        var identity = new double[_n, _n];
        for (int i = 0; i < _n; i++)
            identity[i, i] = 1.0;
        double[,] invL = LinAlg.SolveTriangular(l, identity, upper: false);      // L⁻¹
        double[,] cInv = LinAlg.SolveTriangular(Transpose(l), invL, upper: true); // L⁻ᵀL⁻¹ = C⁻¹
        double[] alpha = LinAlg.Dot(cInv, _y);

        // ∂MLL/∂σ²_k = Σ G ⊙ M(S); P = G ⊙ M'(S); ∂MLL/∂ρ_d = σ²_k·Σ P ⊙ D[:,:,d].
        var g = new double[_n, _n];
        var gM = new double[_n * _n];
        var p = new double[_n, _n];
        int flat = 0;
        for (int i = 0; i < _n; i++)
            for (int j = 0; j < _n; j++)
            {
                double gij = 0.5 * (alpha[i] * alpha[j] - cInv[i, j]);
                g[i, j] = gij;
                gM[flat++] = gij * mVal[i, j];
                p[i, j] = gij * mDer[i, j];
            }
        double dKernelVar = MathUtil.PairwiseSum(gM);

        var dRho = new double[_dim];
        var pd = new double[_n * _n];
        for (int d = 0; d < _dim; d++)
        {
            flat = 0;
            for (int i = 0; i < _n; i++)
                for (int j = 0; j < _n; j++)
                    pd[flat++] = p[i, j] * _squaredXDiff[i, j, d];
            dRho[d] = sigma * MathUtil.PairwiseSum(pd);
        }

        return (mll, dRho, dKernelVar);
    }

    /// <summary>
    ///     The sum over training points of the leave-one-out predictive log likelihood (Rasmussen &amp;
    ///     Williams 2006, Eqns (5.10)–(5.12)) at the hyperparameter setting <paramref name="hp" /> — the
    ///     measure SAASBO-MAP uses to choose among its <c>S</c> pre-selected <c>τ</c> models (§A.2). With
    ///     <c>C⁻¹</c> and <c>α = C⁻¹y</c>, the LOO mean/variance are
    ///     <c>μ_i = y_i − α_i / [C⁻¹]_{ii}</c> and <c>σ²_i = 1 / [C⁻¹]_{ii}</c>, and the score is
    ///     <c>Σ_i (−½ log σ²_i − (y_i − μ_i)² / (2σ²_i) − ½ log 2π)</c>. Returns
    ///     <see cref="double.NegativeInfinity" /> when the covariance will not factor (so that model is
    ///     never selected).
    /// </summary>
    /// <param name="hp">The hyperparameter setting <c>(ρ, σ²_k)</c> at which to score the model.</param>
    /// <returns>The summed leave-one-out predictive log likelihood.</returns>
    /// <exception cref="System.ArgumentException"><paramref name="hp" />'s <c>ρ</c> length does not equal <see cref="Dim" />.</exception>
    internal double LeaveOneOutLogLikelihood(SaasHyperparameters hp)
    {
        double[] rho = hp.Rho;
        double sigma = hp.KernelVariance;
        if (rho.Length != _dim)
            throw new ArgumentException($"rho length must equal the input dimension {_dim}.", nameof(hp));

        double[,] sqd = WeightedSquaredDistance(rho);
        var c = new double[_n, _n];
        for (int i = 0; i < _n; i++)
            for (int j = 0; j < _n; j++)
                c[i, j] = sigma * Rbf(sqd[i, j]).Value + (i == j ? _noiseVar : 0.0);

        if (!TryFactor(c, out double[,] l))
            return double.NegativeInfinity;

        var identity = new double[_n, _n];
        for (int i = 0; i < _n; i++)
            identity[i, i] = 1.0;
        double[,] invL = LinAlg.SolveTriangular(l, identity, upper: false);
        double[,] cInv = LinAlg.SolveTriangular(Transpose(l), invL, upper: true);
        double[] alpha = LinAlg.Dot(cInv, _y);

        var terms = new double[_n];
        for (int i = 0; i < _n; i++)
        {
            double cii = cInv[i, i];
            if (!(cii > 0.0))
                return double.NegativeInfinity;   // degenerate diagonal → the model is unusable.
            double predVar = 1.0 / cii;
            double residual = alpha[i] / cii;     // y_i − μ_i = α_i / [C⁻¹]_ii.
            terms[i] = -0.5 * Math.Log(predVar) - 0.5 * (residual * residual) / predVar - 0.5 * Math.Log(2.0 * Math.PI);
        }
        return MathUtil.PairwiseSum(terms);
    }

    /// <summary>
    ///     Factors the covariance at <paramref name="hp" /> once and returns a <see cref="SaasPosterior" />
    ///     that evaluates the GP posterior (and its input gradient) cheaply across acquisition candidates.
    ///     Escalates the diagonal jitter until the Cholesky succeeds so a concentrated search still yields a
    ///     usable posterior.
    /// </summary>
    /// <param name="hp">The hyperparameter setting <c>(ρ, σ²_k)</c> to condition on.</param>
    /// <returns>A factored posterior over the training data at the given hyperparameters.</returns>
    /// <exception cref="System.ArgumentException"><paramref name="hp" />'s <c>ρ</c> length does not equal <see cref="Dim" />.</exception>
    /// <exception cref="NumSharp.LinAlgError">The covariance stays non-positive-definite even at the largest jitter.</exception>
    internal SaasPosterior Condition(SaasHyperparameters hp)
    {
        if (hp.Rho.Length != _dim)
            throw new ArgumentException($"rho length must equal the input dimension {_dim}.", nameof(hp));

        double[,] sqd = WeightedSquaredDistance(hp.Rho);
        var c = new double[_n, _n];
        for (int i = 0; i < _n; i++)
            for (int j = 0; j < _n; j++)
                c[i, j] = hp.KernelVariance * Rbf(sqd[i, j]).Value + (i == j ? _noiseVar : 0.0);

        if (!TryFactor(c, out double[,] l))
            throw new NumSharp.LinAlgError("SAAS covariance is not positive definite at the largest jitter.");

        double[] invLy = LinAlg.SolveTriangular(l, _y, upper: false);
        double[] invY = LinAlg.SolveTriangular(Transpose(l), invLy, upper: true);   // C⁻¹y
        return new SaasPosterior(this, hp.Rho, hp.KernelVariance, l, invY);
    }

    /// <summary>Attempts a Cholesky factorization, escalating the diagonal jitter until it succeeds.</summary>
    /// <param name="c">The covariance matrix (mutated: jitter is added in place on retries).</param>
    /// <param name="l">The lower Cholesky factor when the method returns <c>true</c>.</param>
    /// <returns><c>true</c> if some jitter level factored; <c>false</c> if even the largest failed.</returns>
    private bool TryFactor(double[,] c, out double[,] l)
    {
        double prev = 0.0;
        foreach (double jitter in JitterLevels)
        {
            // Jitter is cumulative on the diagonal — add only the increment over the previous level so the
            // matrix is not re-cloned each attempt.
            double add = jitter - prev;
            prev = jitter;
            if (add != 0.0)
                for (int i = 0; i < _n; i++)
                    c[i, i] += add;
            try
            {
                l = LinAlg.Cholesky(c);
                return true;
            }
            catch (NumSharp.LinAlgError)
            {
                // not positive definite at this jitter; escalate.
            }
        }
        l = new double[0, 0];
        return false;
    }

    /// <summary>The pairwise squared differences <c>(x_i − x_j)²</c> per coordinate — the posterior's kernel derivative reuses this.</summary>
    /// <returns>The precomputed <c>n × n × dim</c> squared-difference tensor (retained by reference).</returns>
    internal double[,,] SquaredXDiff => _squaredXDiff;

    /// <summary>Returns a new matrix with rows and columns exchanged (for the <c>L⁻ᵀ</c> triangular solve).</summary>
    /// <param name="a">The matrix to transpose.</param>
    /// <returns>The transposed matrix.</returns>
    internal static double[,] Transpose(double[,] a)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        var t = new double[cols, rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t[j, i] = a[i, j];
        return t;
    }
}
