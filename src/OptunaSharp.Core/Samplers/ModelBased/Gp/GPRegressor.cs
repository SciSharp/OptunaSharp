// Beyond-core copy of the Matérn-5/2 ARD Gaussian-process regressor for the model-based samplers
// (TuRBO-1 / TuRBO-m / REMBO). This is a VERBATIM duplicate of the Optuna-parity GP stack's
// OptunaSharp.Gp.GPRegressor, moved into its own namespace so these beyond-core samplers can own and
// evolve their GP machinery — e.g. REMBO's paper-faithful squared-exponential kernel lives beside this
// class in the same toolkit — WITHOUT modifying the shared parity stack that OptunaSharp.Samplers.GPSampler
// and the GP terminators depend on (the project rule: never change a preexisting component when a sampler
// needs it different — copy it). The kernel here stays Matérn-5/2 ARD because that is TuRBO's own paper
// choice (Eriksson et al. 2019, §C); REMBO instead uses the SquaredExponentialGp in this namespace.
// Keep this diff-identical to the parity GPRegressor so the two can be cross-referenced; any change meant
// for GPSampler belongs in OptunaSharp.Gp, not here.

using System;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Samplers.ModelBased.Gp;

/// <summary>
///     Gaussian-process regressor over a Matérn-5/2 ARD kernel — port of <c>GPRegressor</c> in
///     <c>optuna/_gp/gp.py</c>, the surrogate <see cref="OptunaSharp.Samplers.GPSampler" /> fits to the observed trials.
///     This type holds the kernel <b>forward</b> pass (<see cref="Kernel()" />,
///     <see cref="MarginalLogLikelihood" />) and the <b>hand-derived analytic gradient</b> of the
///     fit objective (<see cref="LossAndGrad" />) that replaces Optuna's PyTorch autograd.
///
///     <para><b>The §7 autodiff decision gate.</b> Optuna computes the marginal-log-likelihood
///     gradient with <c>torch.autograd</c> (its <c>Matern52Kernel</c> even writes its own backward).
///     Keeping the GP in Tier N (BCL + NumSharp only, no ~600 MB TorchSharp) requires deriving that
///     gradient by hand. The Matérn-5/2 marginal-likelihood gradient is closed form, so this is a
///     pure-managed computation; it is validated to match live PyTorch autograd to ~1e-11 (a
///     finite-difference self-check confirms correctness independently). See PORTING_PLAN §7.</para>
///
///     <para><b>Tolerance, not bit-exact (documented boundary).</b> The forward rides
///     <see cref="LinAlg.Cholesky" /> (byte-identical to NumPy via OpenBLAS) and the managed
///     <see cref="LinAlg.SolveTriangular(double[,],double[],bool)" />; the gradient is a different
///     sequence of float ops than autograd's reverse pass, so loss and gradient agree with Python to
///     ~1e-11, not bit-for-bit — the managed-vs-LAPACK / autograd-order class the plan already
///     excuses for the GP stack. Everything downstream (the fit trajectory, the sampled params) is
///     therefore parity-verified to a tolerance rather than byte-for-byte.</para>
///
///     <para><b>The gradient (for a future reader diffing against Python).</b> With hyperparameters
///     <c>w = inverse_squared_lengthscales</c>, <c>s = kernel_scale</c>, <c>ν = noise_var</c>, the
///     squared-difference tensor <c>D[i,j,d]</c>, <c>S = D · w</c>, the Matérn value <c>M(S)</c> and
///     its derivative <c>M'(S) = dM/dS</c>, kernel <c>K = s·M(S)</c>, covariance <c>C = K + ν·I</c>,
///     Cholesky <c>C = L·Lᵀ</c>, <c>α = C⁻¹y</c>, and <c>G = ½(ααᵀ − C⁻¹)</c>:
///     <list type="bullet">
///       <item><c>∂MLL/∂w[d] = s · Σᵢⱼ (G ⊙ M'(S))[i,j] · D[i,j,d]</c></item>
///       <item><c>∂MLL/∂s   = Σᵢⱼ G[i,j] · M(S)[i,j]</c></item>
///       <item><c>∂MLL/∂ν   = ½(α·α − tr(C⁻¹))</c></item>
///     </list>
///     The fit optimizes raw (log-transformed) parameters, so each derivative is chained through
///     <c>w = eᵘ</c>, <c>s = eᵘ</c>, <c>ν = eᵘ + minimum_noise</c>, and the analytic
///     <see cref="Prior" /> derivative is added.</para>
/// </summary>
internal sealed class GPRegressor
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna._gp.gp");

    private readonly bool[] _isCategorical;
    private readonly double[,] _xTrain;   // (N, n_params) — completed trials only (kernel()/MLL/fit)
    private readonly double[] _yTrain;     // (N,)
    private readonly double[,,] _squaredXDiff;  // (N, N, n_params)
    private readonly int _n;               // number of training (completed) points
    private readonly int _nParams;

    // The "all" set — completed trials plus any running trials appended by the Kriging-Believer path
    // (append_running_data). Equal to the train set until AppendRunningData is called. The posterior
    // (mean/variance and their gradients) reads these, matching Optuna's posterior() over _X_all.
    private double[,] _xAll;   // (N_all, n_params)
    private double[] _yAll;    // (N_all,)
    private int _nAll;         // number of "all" points

    /// <summary><c>inverse_squared_lengthscales</c> (one per parameter). Mutated by the fit loop.</summary>
    internal double[] InverseSquaredLengthscales { get; set; }

    /// <summary><c>kernel_scale</c>. Mutated by the fit loop.</summary>
    internal double KernelScale { get; set; }

    /// <summary><c>noise_var</c>. Mutated by the fit loop.</summary>
    internal double NoiseVar { get; set; }

    /// <summary>The number of normalized input coordinates modeled by the GP.</summary>
    internal int Dim => _nParams;

    /// <summary><c>length_scales = 1 / √(inverse_squared_lengthscales)</c> — the per-parameter length scales the acqf optimizer preconditions with.</summary>
    internal double[] LengthScales
    {
        get
        {
            var ls = new double[_nParams];
            for (int d = 0; d < _nParams; d++)
                ls[d] = 1.0 / Math.Sqrt(InverseSquaredLengthscales[d]);
            return ls;
        }
    }

    /// <summary>Creates a GP with fixed training observations and the supplied Matern kernel and noise parameters.</summary>
    /// <param name="isCategorical">A flag per coordinate indicating that the kernel must use categorical equality instead of numerical distance.</param>
    /// <param name="xTrain">The training inputs, with one normalized point per row and one parameter per column.</param>
    /// <param name="yTrain">The observed objective values used to train or condition the model.</param>
    /// <param name="inverseSquaredLengthscales">The positive inverse squared kernel length scale of each coordinate.</param>
    /// <param name="kernelScale">The positive amplitude multiplying the kernel covariance.</param>
    /// <param name="noiseVar">The observation-noise variance added to the training covariance diagonal.</param>
    /// <exception cref="System.ArgumentException">The training target count, categorical flags, or length-scale count does not agree with the training matrix dimensions.</exception>
    /// <remarks>Training arrays and categorical flags are retained by reference. Callers must keep their contents stable while this model or its cached covariance is in use.</remarks>
    internal GPRegressor(
        bool[] isCategorical,
        double[,] xTrain,
        double[] yTrain,
        double[] inverseSquaredLengthscales,
        double kernelScale,
        double noiseVar)
    {
        _n = xTrain.GetLength(0);
        _nParams = xTrain.GetLength(1);
        if (yTrain.Length != _n)
            throw new ArgumentException("y_train length must equal the number of rows in X_train.");
        if (isCategorical.Length != _nParams)
            throw new ArgumentException("is_categorical length must equal the number of parameters.");
        if (inverseSquaredLengthscales.Length != _nParams)
            throw new ArgumentException("inverse_squared_lengthscales length must equal the number of parameters.");

        _isCategorical = isCategorical;
        // The fit owns the training buffers and retains them across posterior queries. Keeping the
        // same row order is essential because cached distances and covariance factors index them.
        _xTrain = xTrain;
        _yTrain = yTrain;
        _xAll = xTrain;   // == train set until AppendRunningData appends the running (fantasy) data.
        _yAll = yTrain;
        _nAll = _n;
        InverseSquaredLengthscales = inverseSquaredLengthscales;
        KernelScale = kernelScale;
        NoiseVar = noiseVar;

        // squared_X_diff[i,j,d] = (X[i,d] - X[j,d])**2, or the Hamming indicator for categoricals.
        _squaredXDiff = new double[_n, _n, _nParams];
        for (int i = 0; i < _n; i++)
            for (int j = 0; j < _n; j++)
                for (int d = 0; d < _nParams; d++)
                {
                    double diff = _xTrain[i, d] - _xTrain[j, d];
                    double sq = diff * diff;
                    _squaredXDiff[i, j, d] = _isCategorical[d] ? (sq > 0.0 ? 1.0 : 0.0) : sq;
                }
    }

    /// <summary>
    ///     <c>Matern52Kernel</c>: value <c>M(t)</c> and derivative <c>dM/dt</c> at a squared distance
    ///     <c>t</c>. With <c>u = √(5t)</c>, <c>M = e^{-u}(5/3·t + u + 1)</c> and
    ///     <c>dM/dt = -5/6·(u + 1)·e^{-u}</c> (Optuna saves this derivative by hand to avoid the
    ///     autograd divide-by-zero at <c>t = 0</c>).
    /// </summary>
    /// <param name="t">The nonnegative squared kernel distance between two input points.</param>
    /// <returns>The Matern-5/2 kernel value and derivative for the supplied distance argument.</returns>
    private static (double Value, double Deriv) Matern52(double t)
    {
        double u = Math.Sqrt(5.0 * t);
        double expPart = Math.Exp(-u);
        double value = expPart * ((5.0 / 3.0) * t + u + 1.0);
        double deriv = (-5.0 / 6.0) * (u + 1.0) * expPart;
        return (value, deriv);
    }

    /// <summary>The squared distance <c>S[i,j] = Σ_d squared_X_diff[i,j,d] · w[d]</c>.</summary>
    /// <returns>The pairwise length-scale-weighted squared distances between input rows.</returns>
    private double[,] SquaredDistance()
    {
        double[] w = InverseSquaredLengthscales;
        var s = new double[_n, _n];
        for (int i = 0; i < _n; i++)
            for (int j = 0; j < _n; j++)
            {
                double acc = 0.0;
                for (int d = 0; d < _nParams; d++)
                    acc += _squaredXDiff[i, j, d] * w[d];
                s[i, j] = acc;
            }
        return s;
    }

    /// <summary>
    ///     The training kernel matrix <c>K = kernel_scale · M(S)</c> (no noise on the diagonal),
    ///     matching <c>GPRegressor.kernel()</c> with no arguments.
    /// </summary>
    /// <returns>The Matern-5/2 covariance matrix before adding observation noise.</returns>
    internal double[,] Kernel()
    {
        double[,] s = SquaredDistance();
        double scale = KernelScale;
        var k = new double[_n, _n];
        for (int i = 0; i < _n; i++)
            for (int j = 0; j < _n; j++)
                k[i, j] = scale * Matern52(s[i, j]).Value;
        return k;
    }

    /// <summary>
    ///     The cross-kernel matrix <c>K[b,j] = kernel_scale · M(sqdist(x1_b, x2_j))</c> — the two-argument
    ///     <c>GPRegressor.kernel(X1, X2)</c>, used by the posterior. <paramref name="x1" /> is
    ///     <c>(B, dim)</c>, <paramref name="x2" /> is <c>(N, dim)</c>; the result is <c>(B, N)</c>.
    /// </summary>
    /// <param name="x1">The first matrix of normalized input rows.</param>
    /// <param name="x2">The second matrix of normalized input rows.</param>
    /// <returns>The Matern-5/2 covariance matrix before adding observation noise.</returns>
    internal double[,] Kernel(double[,] x1, double[,] x2)
    {
        int b = x1.GetLength(0);
        int nn = x2.GetLength(0);
        double[] w = InverseSquaredLengthscales;
        double scale = KernelScale;
        var k = new double[b, nn];
        for (int i = 0; i < b; i++)
            for (int j = 0; j < nn; j++)
            {
                double acc = 0.0;
                for (int d = 0; d < _nParams; d++)
                {
                    double diff = x1[i, d] - x2[j, d];
                    double sq = diff * diff;
                    if (_isCategorical[d])
                        sq = sq > 0.0 ? 1.0 : 0.0;
                    acc += sq * w[d];
                }
                k[i, j] = scale * Matern52(acc).Value;
            }
        return k;
    }

    /// <summary>
    ///     The GP posterior mean and variance at the query points <paramref name="x" /> (<c>(B, dim)</c>)
    ///     — <c>GPRegressor.posterior(x, joint=False)</c>. With <c>k = kernel(x, X)</c>,
    ///     <c>mean = k · C⁻¹y</c> and <c>var = kernel_scale − k · (C⁻¹kᵀ)</c> (clamped at 0), reusing the
    ///     cached Cholesky factor and <c>C⁻¹y</c> from <see cref="CacheMatrix" />. Consumed by the
    ///     acquisition functions (Phase 3.2 part 3).
    /// </summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The posterior mean and marginal variance at each query row.</returns>
    /// <exception cref="System.InvalidOperationException">The training covariance has not been prepared by CacheMatrix.</exception>
    internal (double[] Mean, double[] Var) Posterior(double[,] x)
    {
        if (_covYYChol is null)
            throw new InvalidOperationException("Call CacheMatrix before Posterior.");

        int b = x.GetLength(0);
        double[,] covFxFX = Kernel(x, _xAll);   // (B, N_all)

        var mean = new double[b];
        for (int i = 0; i < b; i++)
        {
            double s = 0.0;
            for (int j = 0; j < _nAll; j++)
                s += covFxFX[i, j] * _covYYInvY[j];
            mean[i] = s;
        }

        // V = cov_fx_fX @ C⁻¹ = solve( X (L Lᵀ) = cov_fx_fX ) via two right-side triangular solves.
        double[,] lt = Transpose(_covYYChol);
        double[,] tmp = LinAlg.SolveTriangular(lt, covFxFX, upper: true, left: false);   // X Lᵀ = cov_fx_fX
        double[,] v = LinAlg.SolveTriangular(_covYYChol, tmp, upper: false, left: false); // X L = tmp

        var variance = new double[b];
        for (int i = 0; i < b; i++)
        {
            double s = 0.0;
            for (int j = 0; j < _nAll; j++)
                s += covFxFX[i, j] * v[i, j];
            variance[i] = Math.Max(0.0, KernelScale - s);
        }
        return (mean, variance);
    }

    /// <summary>
    ///     The <b>joint</b> GP posterior mean and full covariance at the query points <paramref name="x" />
    ///     (<c>(B, dim)</c>) — <c>GPRegressor.posterior(x, joint=True)</c>. With <c>k = kernel(x, X_all)</c>,
    ///     <c>mean = k · C⁻¹y</c>, and the covariance <c>cov = kernel(x, x) − k·C⁻¹·kᵀ</c> whose diagonal is
    ///     clamped ≥ 0 (the off-diagonal covariances stay signed). Used by <see cref="OptunaSharp.Gp.ConditionalGPRegressor" />
    ///     to condition on the running fantasies (Phase 3.2 part 3e-6).
    /// </summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The posterior mean vector and joint covariance matrix of the query rows.</returns>
    /// <exception cref="System.InvalidOperationException">The training covariance has not been prepared by CacheMatrix.</exception>
    internal (double[] Mean, double[,] Cov) PosteriorJoint(double[,] x)
    {
        if (_covYYChol is null)
            throw new InvalidOperationException("Call CacheMatrix before PosteriorJoint.");

        int b = x.GetLength(0);
        double[,] covFxFX = Kernel(x, _xAll);       // (B, N_all)
        double[,] covFxFx = Kernel(x, x);           // (B, B) joint prior covariance

        var mean = new double[b];
        for (int i = 0; i < b; i++)
        {
            double s = 0.0;
            for (int j = 0; j < _nAll; j++)
                s += covFxFX[i, j] * _covYYInvY[j];
            mean[i] = s;
        }

        // V = cov_fx_fX @ C⁻¹  (X (L Lᵀ) = cov_fx_fX).
        double[,] v = SolveCholesky(_covYYChol, covFxFX, left: false);   // (B, N_all)
        var cov = new double[b, b];
        for (int a = 0; a < b; a++)
            for (int c = 0; c < b; c++)
            {
                double s = 0.0;
                for (int j = 0; j < _nAll; j++)
                    s += v[a, j] * covFxFX[c, j];   // (V @ cov_fx_fX.T)[a,c]
                double val = covFxFx[a, c] - s;
                cov[a, c] = a == c ? Math.Max(0.0, val) : val;   // clamp only the diagonal
            }
        return (mean, cov);
    }

    /// <summary>
    ///     The posterior mean and variance at a single query point <paramref name="x" /> together with
    ///     their gradients w.r.t. <paramref name="x" /> — the analytic replacement for autograd through
    ///     <c>posterior</c>, used by the acquisition-function gradient (Phase 3.2 part 3b). With
    ///     <c>k_j = kernel_scale·M(S_j)</c>, <c>V = C⁻¹kᵀ</c>: <c>mean = k·C⁻¹y</c>,
    ///     <c>var = kernel_scale − k·V</c> (clamped at 0), and for a continuous dimension <c>d</c>
    ///     (<c>∂k_j/∂x_d = kernel_scale·M'(S_j)·w_d·2(x_d − X_{j,d})</c>):
    ///     <c>∂mean/∂x_d = Σ_j ∂k_j/∂x_d·C⁻¹y_j</c> and <c>∂var/∂x_d = −2·Σ_j ∂k_j/∂x_d·V_j</c>. The
    ///     categorical (Hamming) dimensions have zero derivative, and where the variance is clamped the
    ///     variance gradient is 0 (torch's <c>clamp_min_</c> backward) — matching autograd, including the
    ///     NaNs Optuna itself produces where the posterior collapses.
    /// </summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The posterior mean, variance, their coordinate derivatives, and whether numerical variance clamping occurred.</returns>
    /// <exception cref="System.InvalidOperationException">The training covariance has not been prepared by CacheMatrix.</exception>
    internal (double Mean, double Var, double[] DMean, double[] DVar, bool VarClamped) PosteriorWithGrad(double[] x)
    {
        if (_covYYChol is null)
            throw new InvalidOperationException("Call CacheMatrix before PosteriorWithGrad.");

        double[] w = InverseSquaredLengthscales;
        double scale = KernelScale;
        var k = new double[_nAll];
        var mp = new double[_nAll];
        for (int j = 0; j < _nAll; j++)
        {
            double s = 0.0;
            for (int d = 0; d < _nParams; d++)
            {
                double diff = x[d] - _xAll[j, d];
                double sq = diff * diff;
                if (_isCategorical[d])
                    sq = sq > 0.0 ? 1.0 : 0.0;
                s += sq * w[d];
            }
            (double value, double deriv) = Matern52(s);
            k[j] = scale * value;
            mp[j] = deriv;
        }

        double mean = 0.0;
        for (int j = 0; j < _nAll; j++)
            mean += k[j] * _covYYInvY[j];

        // V = C⁻¹ kᵀ via inv(Lᵀ)·inv(L)·k.
        double[] invLk = LinAlg.SolveTriangular(_covYYChol, k, upper: false);
        double[,] lt = Transpose(_covYYChol);
        double[] v = LinAlg.SolveTriangular(lt, invLk, upper: true);

        double rawVar = scale;
        for (int j = 0; j < _nAll; j++)
            rawVar -= k[j] * v[j];
        bool clamped = rawVar < 0.0;
        double variance = clamped ? 0.0 : rawVar;

        var dMean = new double[_nParams];
        var dVar = new double[_nParams];
        for (int d = 0; d < _nParams; d++)
        {
            if (_isCategorical[d])
                continue;  // Hamming distance is piecewise constant → zero derivative.
            double dm = 0.0;
            double dv = 0.0;
            for (int j = 0; j < _nAll; j++)
            {
                double dkj = scale * mp[j] * w[d] * 2.0 * (x[d] - _xAll[j, d]);
                dm += dkj * _covYYInvY[j];
                dv += dkj * v[j];
            }
            dMean[d] = dm;
            dVar[d] = clamped ? 0.0 : -2.0 * dv;
        }
        return (mean, variance, dMean, dVar, clamped);
    }

    /// <summary>
    ///     <c>marginal_log_likelihood()</c> (constant omitted): <c>-½·logdet(C) - ½·yᵀC⁻¹y</c> with
    ///     <c>C = K + noise_var·I</c>, exploiting the Cholesky factor <c>C = L·Lᵀ</c>:
    ///     <c>logdet_part = -Σ log(diag L)</c> and <c>quad_part = -½·‖L⁻¹y‖²</c>.
    /// </summary>
    /// <returns>The Gaussian marginal log likelihood at the current kernel parameters.</returns>
    /// <exception cref="NumSharp.LinAlgError">The covariance matrix is not numerically positive definite and cannot be Cholesky-factored.</exception>
    internal double MarginalLogLikelihood()
    {
        double[,] c = Kernel();
        for (int i = 0; i < _n; i++)
            c[i, i] += NoiseVar;
        double[,] l = LinAlg.Cholesky(c);

        double logdetPart = 0.0;
        for (int i = 0; i < _n; i++)
            logdetPart -= Math.Log(l[i, i]);

        double[] invLy = LinAlg.SolveTriangular(l, _yTrain, upper: false);
        var squares = new double[_n];
        for (int i = 0; i < _n; i++)
            squares[i] = invLy[i] * invLy[i];
        double quadPart = -0.5 * MathUtil.PairwiseSum(squares);
        return logdetPart + quadPart;
    }

    /// <summary>
    ///     The fit objective <c>loss(raw_params) = -MLL - log_prior</c> and its gradient w.r.t. the
    ///     raw (log-transformed) parameters — the analytic replacement for Optuna's
    ///     <c>loss_func</c> (autograd) inside <c>_fit_kernel_params</c>. This sets the three
    ///     hyperparameters from <paramref name="rawParams" /> (as the fit loop does on every
    ///     evaluation): <c>w = exp(raw[:n])</c>, <c>s = exp(raw[n])</c>, and, unless
    ///     <paramref name="deterministicObjective" />, <c>ν = exp(raw[n+1]) + minimum_noise</c>
    ///     (else <c>ν = minimum_noise</c>, and the noise gradient is exactly 0).
    /// </summary>
    /// <param name="rawParams">The unconstrained kernel-optimization coordinates whose value and gradient are evaluated.</param>
    /// <param name="minimumNoise">The positive floor on observation noise used when fitting the GP.</param>
    /// <param name="deterministicObjective">Whether the objective is treated as noiseless apart from the minimum variance floor.</param>
    /// <returns>The negative marginal log likelihood plus prior penalty and its gradient in raw parameter coordinates.</returns>
    internal (double Loss, double[] Grad) LossAndGrad(
        double[] rawParams, double minimumNoise, bool deterministicObjective)
    {
        int n = _nParams;
        var w = new double[n];
        for (int d = 0; d < n; d++)
            w[d] = Math.Exp(rawParams[d]);
        double s = Math.Exp(rawParams[n]);
        double nv = deterministicObjective ? minimumNoise : Math.Exp(rawParams[n + 1]) + minimumNoise;

        InverseSquaredLengthscales = w;
        KernelScale = s;
        NoiseVar = nv;

        // ---- forward: S, M(S), M'(S), K, C = K + ν·I, Cholesky ----
        double[,] sq = SquaredDistance();
        var k0 = new double[_n, _n];   // M(S)
        var mp = new double[_n, _n];   // M'(S)
        var c = new double[_n, _n];    // C = s·M(S) + ν·I
        for (int i = 0; i < _n; i++)
            for (int j = 0; j < _n; j++)
            {
                (double value, double deriv) = Matern52(sq[i, j]);
                k0[i, j] = value;
                mp[i, j] = deriv;
                c[i, j] = s * value + (i == j ? nv : 0.0);
            }

        double[,] l = LinAlg.Cholesky(c);
        double logdetPart = 0.0;
        for (int i = 0; i < _n; i++)
            logdetPart -= Math.Log(l[i, i]);
        double[] invLy = LinAlg.SolveTriangular(l, _yTrain, upper: false);
        var invLySq = new double[_n];
        for (int i = 0; i < _n; i++)
            invLySq[i] = invLy[i] * invLy[i];
        double mll = logdetPart - 0.5 * MathUtil.PairwiseSum(invLySq);
        double logPrior = Prior.DefaultLogPrior(this);
        double loss = -mll - logPrior;

        // ---- backward: C⁻¹ = L⁻ᵀL⁻¹, α = C⁻¹y, G = ½(ααᵀ - C⁻¹) ----
        var identity = new double[_n, _n];
        for (int i = 0; i < _n; i++)
            identity[i, i] = 1.0;
        double[,] invL = LinAlg.SolveTriangular(l, identity, upper: false);   // L⁻¹
        double[,] lt = Transpose(l);
        double[,] cInv = LinAlg.SolveTriangular(lt, invL, upper: true);        // L⁻ᵀL⁻¹ = C⁻¹
        double[] alpha = LinAlg.Dot(cInv, _yTrain);

        // ∂MLL/∂ν = ½(α·α - tr(C⁻¹))
        var alphaSq = new double[_n];
        double traceCInv = 0.0;
        for (int i = 0; i < _n; i++)
        {
            alphaSq[i] = alpha[i] * alpha[i];
            traceCInv += cInv[i, i];
        }
        double dmllDNoise = 0.5 * (MathUtil.PairwiseSum(alphaSq) - traceCInv);

        // G = ½(ααᵀ - C⁻¹); ∂MLL/∂s = Σ G⊙M(S); P = G⊙M'(S); ∂MLL/∂w[d] = s·Σ P⊙D[:,:,d]
        var g = new double[_n, _n];
        var gK0 = new double[_n * _n];
        var p = new double[_n, _n];
        int flat = 0;
        for (int i = 0; i < _n; i++)
            for (int j = 0; j < _n; j++)
            {
                double gij = 0.5 * (alpha[i] * alpha[j] - cInv[i, j]);
                g[i, j] = gij;
                gK0[flat++] = gij * k0[i, j];
                p[i, j] = gij * mp[i, j];
            }
        double dmllDScale = MathUtil.PairwiseSum(gK0);

        var dmllDW = new double[n];
        var pd = new double[_n * _n];
        for (int d = 0; d < n; d++)
        {
            flat = 0;
            for (int i = 0; i < _n; i++)
                for (int j = 0; j < _n; j++)
                    pd[flat++] = p[i, j] * _squaredXDiff[i, j, d];
            dmllDW[d] = s * MathUtil.PairwiseSum(pd);
        }

        // ---- chain to raw params and add the analytic prior gradient ----
        //   d(log_prior)/d(raw_w[d]) = (0.1/w²-0.1)·w = 0.1/w - 0.1·w
        //   d(log_prior)/d(raw_s)    = (1/s-1)·s      = 1 - s
        //   d(log_prior)/d(raw_ν)    = (0.1/ν-30)·(ν - minimum_noise)
        var grad = new double[n + 2];
        for (int d = 0; d < n; d++)
            grad[d] = -(dmllDW[d] * w[d]) - (0.1 / w[d] - 0.1 * w[d]);
        grad[n] = -(dmllDScale * s) - (1.0 - s);
        grad[n + 1] = deterministicObjective
            ? 0.0
            : -(dmllDNoise * (nv - minimumNoise)) - ((0.1 / nv - 30.0) * (nv - minimumNoise));

        return (loss, grad);
    }

    private double[,]? _covYYChol;
    private double[] _covYYInvY = Array.Empty<double>();

    /// <summary>
    ///     <c>_cache_matrix()</c> — factor <c>C = K + noise_var·I = L·Lᵀ</c> once and precompute
    ///     <c>C⁻¹y</c>, the quantities the posterior (acquisition functions, Phase 3.2 part 3) needs.
    /// </summary>
    /// <exception cref="NumSharp.LinAlgError">The covariance matrix is not numerically positive definite and cannot be Cholesky-factored.</exception>
    internal void CacheMatrix()
    {
        double[,] c = Kernel();
        for (int i = 0; i < _n; i++)
            c[i, i] += NoiseVar;
        _covYYChol = LinAlg.Cholesky(c);
        // C⁻¹y = L⁻ᵀ(L⁻¹y).
        double[] invLy = LinAlg.SolveTriangular(_covYYChol, _yTrain, upper: false);
        double[,] lt = Transpose(_covYYChol);
        _covYYInvY = LinAlg.SolveTriangular(lt, invLy, upper: true);
    }

    /// <summary>
    ///     <c>append_running_data(X_running, y_running)</c> — the Kriging-Believer step. Extend the cached
    ///     covariance factorization with the running trials' points (using the posterior mean as their
    ///     fantasy objective values), so subsequent <see cref="Posterior" /> / <see cref="PosteriorWithGrad" />
    ///     calls condition on the running trials. The completed-only <c>kernel()</c>/fit state is left
    ///     untouched (only the "all" set and the covariance grow); <b>must be called once</b>, after
    ///     <see cref="CacheMatrix" />. Mirrors Optuna's <c>GPRegressor.append_running_data</c>.
    /// </summary>
    /// <param name="xRunning">The normalized input rows of pending trials.</param>
    /// <param name="yRunning">The values assigned to the pending trials, in matching row order.</param>
    /// <exception cref="System.InvalidOperationException">The training covariance has not been prepared by CacheMatrix.</exception>
    /// <exception cref="NumSharp.LinAlgError">The covariance matrix is not numerically positive definite and cannot be Cholesky-factored.</exception>
    internal void AppendRunningData(double[,] xRunning, double[] yRunning)
    {
        if (_covYYChol is null)
            throw new InvalidOperationException("Call CacheMatrix before AppendRunningData.");

        int r = xRunning.GetLength(0);
        double[,] kernelRunningTrain = Kernel(xRunning, _xTrain);        // (r, N)
        double[,] kernelRunningRunning = Kernel(xRunning, xRunning);     // (r, r)
        for (int i = 0; i < r; i++)
            kernelRunningRunning[i, i] += NoiseVar;

        // _X_all = cat(X_train, X_running); _y_all = cat(y_train, y_running).
        var xAll = new double[_n + r, _nParams];
        for (int i = 0; i < _n; i++)
            for (int d = 0; d < _nParams; d++)
                xAll[i, d] = _xTrain[i, d];
        for (int i = 0; i < r; i++)
            for (int d = 0; d < _nParams; d++)
                xAll[_n + i, d] = xRunning[i, d];
        var yAll = new double[_n + r];
        Array.Copy(_yTrain, yAll, _n);
        Array.Copy(yRunning, 0, yAll, _n, r);

        _covYYChol = ExtendCholesky(_covYYChol, kernelRunningTrain, kernelRunningRunning);
        _covYYInvY = SolveCholesky(_covYYChol, yAll);
        _xAll = xAll;
        _yAll = yAll;
        _nAll = _n + r;
    }

    /// <summary>The cached lower Cholesky factor <c>L</c> of <c>C = K + noise_var·I</c> (<see cref="OptunaSharp.Gp.ConditionalGPRegressor" /> conditions on it).</summary>
    /// <exception cref="System.InvalidOperationException">The training covariance has not been prepared by CacheMatrix.</exception>
    internal double[,] CovYYChol => _covYYChol ?? throw new InvalidOperationException("Call CacheMatrix first.");

    /// <summary><c>kernel(X1)</c> — the cross kernel <c>kernel(X1, X_train)</c> against the completed-trial inputs.</summary>
    /// <param name="x1">The first matrix of normalized input rows.</param>
    /// <returns>The covariance between each query row and each training row.</returns>
    internal double[,] KernelToTrain(double[,] x1) => Kernel(x1, _xTrain);

    /// <summary>
    ///     The kernel row <c>k[j] = kernel(x, others[j])</c> at a single query point <paramref name="x" />
    ///     together with its gradient <c>dk[j,d] = ∂k[j]/∂x_d</c> — the cross-kernel building block the
    ///     <see cref="OptunaSharp.Gp.ConditionalGPRegressor" /> gradient chains through (<c>∂k/∂x_d =
    ///     kernel_scale·M'(S)·w_d·2(x_d − others[j,d])</c>; categorical/Hamming dims have zero derivative).
    /// </summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <param name="others">The input rows against which the query kernel row is computed.</param>
    /// <returns>The query-to-input covariance row and its normalized-coordinate derivatives.</returns>
    internal (double[] K, double[,] Dk) KernelRowWithGrad(double[] x, double[,] others)
    {
        int m = others.GetLength(0);
        double[] w = InverseSquaredLengthscales;
        double scale = KernelScale;
        var k = new double[m];
        var dk = new double[m, _nParams];
        for (int j = 0; j < m; j++)
        {
            double s = 0.0;
            for (int d = 0; d < _nParams; d++)
            {
                double diff = x[d] - others[j, d];
                double sq = diff * diff;
                if (_isCategorical[d])
                    sq = sq > 0.0 ? 1.0 : 0.0;
                s += sq * w[d];
            }
            (double value, double deriv) = Matern52(s);
            k[j] = scale * value;
            for (int d = 0; d < _nParams; d++)
                dk[j, d] = _isCategorical[d] ? 0.0 : scale * deriv * w[d] * 2.0 * (x[d] - others[j, d]);
        }
        return (k, dk);
    }

    /// <summary><c>kernel(x, X_train)</c> row and its x-gradient — <see cref="KernelRowWithGrad" /> against the completed-trial inputs.</summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <returns>The query-to-training covariance row and its normalized-coordinate derivatives.</returns>
    internal (double[] K, double[,] Dk) KernelRowToTrainWithGrad(double[] x) => KernelRowWithGrad(x, _xTrain);

    /// <summary>
    ///     <c>_solve_cholesky(L, B, left)</c> — solve <c>(L Lᵀ) X = B</c> (<paramref name="left" />
    ///     <c>true</c>) or <c>X (L Lᵀ) = B</c> (<c>false</c>) via two triangular solves, where
    ///     <c>C = L Lᵀ</c>.
    /// </summary>
    /// <param name="l">The lower-triangular Cholesky factor of the positive-definite coefficient matrix.</param>
    /// <param name="b">The vector or matrix right-hand side aligned with the Cholesky factor.</param>
    /// <param name="left">True solves with the coefficient matrix on the left; false solves with it on the right.</param>
    /// <returns>The solution obtained by applying the two triangular factors of the covariance matrix.</returns>
    internal static double[,] SolveCholesky(double[,] l, double[,] b, bool left = true)
    {
        if (left)
        {
            double[,] tmp = LinAlg.SolveTriangular(l, b, upper: false, left: true);           // L (Lᵀ X) = B
            return LinAlg.SolveTriangular(Transpose(l), tmp, upper: true, left: true);         // Lᵀ X = tmp
        }
        double[,] rt = LinAlg.SolveTriangular(Transpose(l), b, upper: true, left: false);       // (X L) Lᵀ = B
        return LinAlg.SolveTriangular(l, rt, upper: false, left: false);                        // X L = rt
    }

    /// <summary><c>_solve_cholesky(L, b)</c> for a vector right-hand side (<c>left=True</c>): <c>C⁻¹b = L⁻ᵀ(L⁻¹b)</c>.</summary>
    /// <param name="l">The lower-triangular Cholesky factor of the positive-definite coefficient matrix.</param>
    /// <param name="b">The vector or matrix right-hand side aligned with the Cholesky factor.</param>
    /// <returns>The solution obtained by applying the two triangular factors of the covariance matrix.</returns>
    internal static double[] SolveCholesky(double[,] l, double[] b)
    {
        double[] invLb = LinAlg.SolveTriangular(l, b, upper: false);
        return LinAlg.SolveTriangular(Transpose(l), invLb, upper: true);
    }

    /// <summary>
    ///     <c>_extend_cholesky(L11, K21, K22)</c> — the Cholesky factor of the bordered matrix
    ///     <c>[[K11, K21ᵀ], [K21, K22]]</c> given <c>K11 = L11·L11ᵀ</c>:
    ///     <c>L = [[L11, 0], [L21, chol(K22 − L21·L21ᵀ)]]</c> with <c>L21ᵀ = L11⁻¹·K21ᵀ</c>.
    /// </summary>
    /// <param name="l11">The existing lower-triangular Cholesky factor for the first block.</param>
    /// <param name="k21">The covariance block between newly appended and existing observations.</param>
    /// <param name="k22">The lower-right covariance block for the new observations, including any required diagonal noise.</param>
    /// <returns>The lower-triangular Cholesky factor after appending the new covariance block.</returns>
    /// <exception cref="NumSharp.LinAlgError">The covariance matrix is not numerically positive definite and cannot be Cholesky-factored.</exception>
    internal static double[,] ExtendCholesky(double[,] l11, double[,] k21, double[,] k22)
    {
        int n1 = l11.GetLength(0);
        int n2 = k22.GetLength(0);
        double[,] l21T = LinAlg.SolveTriangular(l11, Transpose(k21), upper: false, left: true);  // L11 L21ᵀ = K21ᵀ  → (n1, n2)
        double[,] l21 = Transpose(l21T);                                                          // (n2, n1)
        double[,] l21l21T = LinAlg.Dot(l21, l21T);                                                // (n2, n2)
        var schur = new double[n2, n2];
        for (int i = 0; i < n2; i++)
            for (int j = 0; j < n2; j++)
                schur[i, j] = k22[i, j] - l21l21T[i, j];
        double[,] l22 = LinAlg.Cholesky(schur);

        var l = new double[n1 + n2, n1 + n2];
        for (int i = 0; i < n1; i++)
            for (int j = 0; j < n1; j++)
                l[i, j] = l11[i, j];
        for (int i = 0; i < n2; i++)
        {
            for (int j = 0; j < n1; j++)
                l[n1 + i, j] = l21[i, j];
            for (int j = 0; j < n2; j++)
                l[n1 + i, n1 + j] = l22[i, j];
        }
        return l;
    }

    /// <summary>
    ///     <c>_fit_kernel_params</c> — maximize the marginal log likelihood over the (log-transformed)
    ///     kernel hyperparameters with the ported <see cref="LBfgsB" /> (scipy's <c>l-bfgs-b</c>), then
    ///     set the fitted hyperparameters and cache the covariance. <paramref name="gtol" /> maps to
    ///     L-BFGS-B's <c>pgtol</c> exactly as <c>scipy.optimize.minimize(method="l-bfgs-b")</c> does
    ///     (the other options keep their defaults: <c>factr = 1e7</c>, <c>m = 10</c>). Throws when the
    ///     optimizer does not converge (Optuna's <c>RuntimeError("Optimization failed: ...")</c>).
    /// </summary>
    /// <param name="minimumNoise">The positive floor on observation noise used when fitting the GP.</param>
    /// <param name="deterministicObjective">Whether the objective is treated as noiseless apart from the minimum variance floor.</param>
    /// <param name="gtol">The gradient-norm tolerance used to stop the kernel-parameter optimizer.</param>
    /// <returns>A GP fitted from the supplied initial kernel parameters.</returns>
    /// <exception cref="System.InvalidOperationException">The numerical optimizer terminates without a successful kernel fit.</exception>
    private GPRegressor FitInner(double minimumNoise, bool deterministicObjective, double gtol)
    {
        int nParams = _nParams;
        var initial = new double[nParams + 2];
        for (int d = 0; d < nParams; d++)
            initial[d] = Math.Log(InverseSquaredLengthscales[d]);
        initial[nParams] = Math.Log(KernelScale);
        // 0.99 * minimum_noise is subtracted to keep the initial noise strictly above the floor.
        initial[nParams + 1] = Math.Log(NoiseVar - 0.99 * minimumNoise);

        LBfgsB.Result res = LBfgsB.Minimize(
            rawParams => LossAndGrad(rawParams, minimumNoise, deterministicObjective),
            initial, bounds: null, m: 10, factr: 1e7, pgtol: gtol, maxfun: 15000, maxiter: 15000, maxls: 20);
        if (res.WarnFlag != 0)
            throw new InvalidOperationException("Optimization failed");

        double[] xOpt = res.X;
        for (int d = 0; d < nParams; d++)
            InverseSquaredLengthscales[d] = Math.Exp(xOpt[d]);
        KernelScale = Math.Exp(xOpt[nParams]);
        NoiseVar = deterministicObjective ? minimumNoise : minimumNoise + Math.Exp(xOpt[nParams + 1]);
        CacheMatrix();
        return this;
    }

    /// <summary>
    ///     <c>fit_kernel_params</c> — the robust entry point: fit starting from
    ///     <paramref name="gprCache" /> (the previous trial's kernel parameters) and, if that fails,
    ///     retry from the all-ones default; if both fail, fall back to the default parameters and log
    ///     a warning (matching Optuna's <c>fit_kernel_params</c>). <paramref name="isCategorical" /> flags the
    ///     Hamming-distance dimensions; <paramref name="x" /> is the normalized training inputs and
    ///     <paramref name="y" /> the standardized objective.
    /// </summary>
    /// <param name="x">Normalized input coordinates in search-space order; rectangular matrices contain one point per row.</param>
    /// <param name="y">The second numerical input, aligned with the first input where both are sequences.</param>
    /// <param name="isCategorical">A flag per coordinate indicating that the kernel must use categorical equality instead of numerical distance.</param>
    /// <param name="minimumNoise">The positive floor on observation noise used when fitting the GP.</param>
    /// <param name="deterministicObjective">Whether the objective is treated as noiseless apart from the minimum variance floor.</param>
    /// <param name="gprCache">A previously fitted GP whose kernel parameters provide a warm start, or null for the default initialization.</param>
    /// <param name="gtol">The gradient-norm tolerance used to stop the kernel-parameter optimizer.</param>
    /// <returns>A fitted GP with cached training covariance factors.</returns>
    internal static GPRegressor FitKernelParams(
        double[,] x, double[] y, bool[] isCategorical,
        double minimumNoise, bool deterministicObjective,
        GPRegressor? gprCache = null, double gtol = 1e-2)
    {
        int dim = x.GetLength(1);
        GPRegressor Default() => new(isCategorical, x, y, Ones(dim), 1.0, 1.0);

        GPRegressor defaultCache = Default();
        gprCache ??= Default();

        // First try optimizing from the provided cache, then from the default parameters; if both
        // fail, fall back to the default parameters — Optuna's fit_kernel_params robustness retry.
        Exception? error = null;
        foreach (GPRegressor cache in new[] { gprCache, defaultCache })
        {
            try
            {
                return new GPRegressor(
                    isCategorical, x, y,
                    (double[])cache.InverseSquaredLengthscales.Clone(),
                    cache.KernelScale, cache.NoiseVar)
                    .FitInner(minimumNoise, deterministicObjective, gtol);
            }
            catch (Exception e) when (e is InvalidOperationException or NumSharp.LinAlgError)
            {
                // Optuna catches RuntimeError here, which covers both scipy's "Optimization failed"
                // (our InvalidOperationException from a nonzero L-BFGS-B warnflag) and torch's Cholesky
                // failure (our NumSharp.LinAlgError). Retry from the default kernel parameters next.
                error = e;
            }
        }

        _logger.Warning(
            $"The optimization of kernel parameters failed: \n{error?.Message}\n"
            + "The default initial kernel parameters will be used instead.");
        GPRegressor fallback = Default();
        fallback.CacheMatrix();
        return fallback;
    }

    /// <summary>Returns a new vector of the requested length filled with one.</summary>
    /// <param name="n">The number of elements or points to process.</param>
    /// <returns>A new vector of the requested length filled with one.</returns>
    private static double[] Ones(int n)
    {
        var a = new double[n];
        Array.Fill(a, 1.0);
        return a;
    }

    /// <summary>Returns a new matrix with rows and columns exchanged.</summary>
    /// <param name="a">The first numerical operand; its dimensions must agree with the operation described in the summary.</param>
    /// <returns>A new matrix with rows and columns exchanged.</returns>
    private static double[,] Transpose(double[,] a)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        var t = new double[cols, rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t[j, i] = a[i, j];
        return t;
    }
}
