using System;
using OptunaSharp.Internal;

namespace OptunaSharp.Gp;

/// <summary>
///     Gaussian-process regressor over a Matérn-5/2 ARD kernel — port of <c>GPRegressor</c> in
///     <c>optuna/_gp/gp.py</c>, the surrogate <see cref="GPSampler" /> fits to the observed trials.
///     This type holds the kernel <b>forward</b> pass (<see cref="Kernel" />,
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
    private readonly bool[] _isCategorical;
    private readonly double[,] _xTrain;   // (N, n_params)
    private readonly double[] _yTrain;     // (N,)
    private readonly double[,,] _squaredXDiff;  // (N, N, n_params)
    private readonly int _n;               // number of training points
    private readonly int _nParams;

    /// <summary><c>inverse_squared_lengthscales</c> (one per parameter). Mutated by the fit loop.</summary>
    internal double[] InverseSquaredLengthscales { get; set; }

    /// <summary><c>kernel_scale</c>. Mutated by the fit loop.</summary>
    internal double KernelScale { get; set; }

    /// <summary><c>noise_var</c>. Mutated by the fit loop.</summary>
    internal double NoiseVar { get; set; }

    internal int Dim => _nParams;

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
        _xTrain = xTrain;
        _yTrain = yTrain;
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
    private static (double Value, double Deriv) Matern52(double t)
    {
        double u = Math.Sqrt(5.0 * t);
        double expPart = Math.Exp(-u);
        double value = expPart * ((5.0 / 3.0) * t + u + 1.0);
        double deriv = (-5.0 / 6.0) * (u + 1.0) * expPart;
        return (value, deriv);
    }

    /// <summary>The squared distance <c>S[i,j] = Σ_d squared_X_diff[i,j,d] · w[d]</c>.</summary>
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
    ///     <c>marginal_log_likelihood()</c> (constant omitted): <c>-½·logdet(C) - ½·yᵀC⁻¹y</c> with
    ///     <c>C = K + noise_var·I</c>, exploiting the Cholesky factor <c>C = L·Lᵀ</c>:
    ///     <c>logdet_part = -Σ log(diag L)</c> and <c>quad_part = -½·‖L⁻¹y‖²</c>.
    /// </summary>
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
    ///     <c>_fit_kernel_params</c> — maximize the marginal log likelihood over the (log-transformed)
    ///     kernel hyperparameters with the ported <see cref="LBfgsB" /> (scipy's <c>l-bfgs-b</c>), then
    ///     set the fitted hyperparameters and cache the covariance. <paramref name="gtol" /> maps to
    ///     L-BFGS-B's <c>pgtol</c> exactly as <c>scipy.optimize.minimize(method="l-bfgs-b")</c> does
    ///     (the other options keep their defaults: <c>factr = 1e7</c>, <c>m = 10</c>). Throws when the
    ///     optimizer does not converge (Optuna's <c>RuntimeError("Optimization failed: ...")</c>).
    /// </summary>
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
    ///     retry from the all-ones default; if both fail, fall back to the default parameters
    ///     (matching Optuna, which then logs a warning). <paramref name="isCategorical" /> flags the
    ///     Hamming-distance dimensions; <paramref name="x" /> is the normalized training inputs and
    ///     <paramref name="y" /> the standardized objective.
    /// </summary>
    internal static GPRegressor FitKernelParams(
        double[,] x, double[] y, bool[] isCategorical,
        double minimumNoise, bool deterministicObjective,
        GPRegressor? gprCache = null, double gtol = 1e-2)
    {
        int dim = x.GetLength(1);
        GPRegressor Default() => new(isCategorical, x, y, Ones(dim), 1.0, 1.0);

        GPRegressor defaultCache = Default();
        gprCache ??= Default();

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
            catch (InvalidOperationException)
            {
                // Retry from the default kernel parameters (improves robustness), as Optuna does.
            }
        }

        GPRegressor fallback = Default();
        fallback.CacheMatrix();
        return fallback;
    }

    private static double[] Ones(int n)
    {
        var a = new double[n];
        Array.Fill(a, 1.0);
        return a;
    }

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
