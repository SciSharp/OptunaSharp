// VkD-CMA — Projection-Based Restricted Covariance Matrix Adaptation for high dimension.
//
// Reference papers:
//   [1] Youhei Akimoto and Nikolaus Hansen. "Projection-Based Restricted Covariance Matrix
//       Adaptation for High Dimension." Proc. of GECCO 2016, pp. 197–204.
//   [2] Youhei Akimoto and Nikolaus Hansen. "Online Model Selection for Restricted Covariance
//       Matrix Adaptation." Proc. of PPSN 2016, pp. 3–13. (the online k-adaptation)
//
// Reference implementation: Akimoto's own `GaussVkDSampler` in pycma
//   (cma/restricted_gaussian_sampler.py). That class is a covariance *model* — the restricted
//   representation C = D(I + Vᵀ diag(S) V)D, its natural-gradient update, the TPA step-size, and the
//   PPSN-2016 k-adaptation — that pycma's `CMAEvolutionStrategy` drives (mean recombination +
//   selection live in the ES). This file ports that model AND wraps it in a self-contained
//   ask/tell ES loop (the same shape as `LmCma`), so the covariance math is a faithful port of
//   `GaussVkDSampler` while the surrounding recombination is the standard weighted-mean ES. Method
//   comments cite the pycma method / paper each block realizes; the type <summary> records the
//   fidelity decisions and the two protocol concessions the per-trial sampler forces.

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.CmaEs;

/// <summary>
///     VkD-CMA-ES optimizer with an ask-and-tell interface — a faithful port of Akimoto's
///     <c>GaussVkDSampler</c> restricted-covariance model (pycma <c>cma/restricted_gaussian_sampler.py</c>),
///     wrapped in a self-contained evolution-strategy loop. VkD-CMA restricts the covariance to a
///     <b>rank-<c>k</c> term plus a diagonal</b>, <c>C = σ² D(I + Σⱼ Sⱼ Vⱼ Vⱼᵀ)D</c> with the <c>Vⱼ</c>
///     orthonormal, and adapts <c>D</c>, <c>V</c>, <c>S</c> AND the rank <c>k</c> itself online — cost
///     <c>O(N k² + k³)</c> per generation when <c>N &gt; k</c>, versus full <see cref="Cma" />'s
///     <c>O(N²)</c>. It is the middle point of the restricted-covariance family between <see cref="SepCma" />
///     (diagonal only) and full <see cref="Cma" />.
///
///     <para><b>What is ported from <c>GaussVkDSampler</c> verbatim</b>: the <c>transform</c> /
///     <c>transform_inverse</c> whitening; the U-matrix natural-gradient update of <c>V</c>, <c>S</c>,
///     <c>D</c> (the two-branch eigendecomposition of <c>UᵀU</c> when <c>N&gt;rankU</c>, else <c>UUᵀ</c>);
///     the determinant normalization; the <b>TPA</b> (Two-Point step-size Adaptation) with the injected
///     mirrored pair; and the <b>PPSN-2016 k-adaptation</b> heuristic. The surrounding mean
///     recombination — standard positive log-weights, <c>m ← m + Σ wᵢ (x_{i:λ} − m)</c> — is the ES the
///     reference class assumes around it.</para>
///
///     <para><b>Faithful quirks preserved from the reference</b> (do not "fix" these — they are the
///     published behavior the doctests converge under): (1) pycma's <c>ExponentialMovingAverage</c> is
///     constructed with <c>flg_init_with_data=False</c>, so <c>flg_init = -False = 0</c> makes the blend
///     factor <c>a = 1</c> on every step — the "EMA" collapses to <b>the latest one-step datum</b> and its
///     variance estimate stays 0. Ported verbatim as <see cref="Ema" /> so this parity does not rest on a
///     re-derivation. (2) The reference re-unpacks <c>_get_params</c> into <c>(cone, cmu, cc)</c> after a
///     k-change — the wrong order, and the result is never read before the method returns — so it is
///     <b>dead</b> and omitted here.</para>
///
///     <para><b>Two protocol concessions the per-trial sampler forces</b> (documented on
///     <see cref="VkdCmaSampler" />): TPA needs a mirrored ±<c>dx</c> pair as the generation's first two
///     samples, and it recovers them <b>by direction</b> (<c>dot(ŷ, d̂x)</c>, length-invariant). The batch
///     <see cref="Ask" /> emits an exact mirror (same drawn length); the per-trial
///     <see cref="AskCandidate" /> — asked one candidate at a time from freshly-restored state across
///     separately-seeded trials — emits the pair with <b>independent</b> lengths but the exact <c>±dx</c>
///     directions, which is all TPA's rank recovery needs. Like <see cref="LmCma" />, the optimizer itself
///     is unconstrained; the sampler clamps to the box.</para>
/// </summary>
internal sealed class VkdCma : ICmaOptimizer
{
    private const double Eps = 1e-30;      // guard for ‖dx‖ ≈ 0 (no mean shift yet → skip TPA injection)

    // ---- fixed configuration (a pure function of the constructor inputs) ----------------------
    private readonly int _nDim;            // N
    private readonly int _popsize;         // λ
    private readonly int _mu;              // μ = ⌊λ/2⌋
    private readonly double[] _weights;    // wᵢ, i = 1..μ, positive, normalized to sum 1
    private readonly double _muEff;        // μ_eff = 1 / Σ wᵢ²
    private readonly double _sigma0;       // initial σ — needed to rebuild config on state restore
    private readonly int _kInit;           // k the optimizer was constructed with (for restore)
    private readonly bool _kAdapt;         // whether the rank k adapts online (PPSN 2016)

    // TPA step-size parameters (GaussVkDSampler.__init__)
    private readonly double _cs;           // cs = 0.3
    private readonly double _ds;           // ds = 4 − 3/N

    // k-adaptation static parameters (GaussVkDSampler.__init__, only when _kAdapt)
    private readonly int _kmin;
    private readonly int _kmax;
    private readonly double _kIncCond;
    private readonly double _kDecCond;
    private readonly double _kAdaptFactor;
    private readonly double _factorSigmaSlope;
    private readonly double _factorDiagSlope;
    private readonly double _acceptedSlowdown;
    private readonly double _kAdaptDecay;
    private readonly double _kAdaptWait;
    private readonly double _optConv;

    // ---- evolving state -----------------------------------------------------------------------
    private double[] _mean;                // m
    private double _sigma;                 // σ (TPA-adapted step-size)
    private double[] _D;                   // coordinate-wise diagonal scaling (length N)
    private readonly List<double[]> _V;    // orthonormal low-rank directions; Count == _k, each length N
    private double[] _S;                   // eigenvalue corrections (length N; first _kActive used)
    private double[] _pc;                  // evolution path (length N)
    private double[] _dx;                  // last weighted mean shift in D-space (for TPA injection)
    private int _k;                        // current rank
    private int _kActive;                  // active rank (≤ _k)
    private bool _flgInjection;            // whether TPA injection is armed (true after the first tell)
    private double _ps;                    // TPA cumulation scalar
    private int _g;                        // generation counter

    // k-adaptation running state (initialized on construction; == the reference's lazy first-tell init,
    // whose pre-update D/σ equal the constructor values)
    private double _lastLogSigma;
    private double[] _lastLogD;
    private double[] _lastLogCondCorr;
    private readonly Ema _emaLogSigma;     // dim 1
    private readonly Ema _emaLogD;         // dim N
    private readonly Ema _emaLogS;         // dim N
    private int _itrAfterKInc;

    private double _lnSigmaChange;         // scalar  (ema_log_sigma.M / (opt_conv/accepted_slowdown))
    private double[] _lnDiagChange;        // N       (ema_log_d.M / (cmu+cone))
    private double[] _lnLambdaChange;      // N       (ema_log_s.M / (cmu+cone))

    // per-generation ask bookkeeping
    private int _askIndex;                 // 1-based index of the next candidate within the generation
    private double[]? _lastInject;         // the +dy injected step, mirrored by the next even candidate

    private NumPyRandom _rng;

    /// <summary>Initializes the diagonal-plus-low-rank proposal and optional covariance-rank adaptation.</summary>
    /// <param name="mean">Initial or current proposal center in transformed coordinates.</param>
    /// <param name="sigma">Initial positive global proposal standard deviation.</param>
    /// <param name="seed">Seed used to reset the optimizer random stream.</param>
    /// <param name="populationSize">Number of individuals retained or evaluated per generation.</param>
    /// <param name="kInit">Initial low-rank covariance width; null selects the algorithm default.</param>
    /// <param name="kAdapt">Whether covariance rank may change during optimization.</param>
    /// <param name="kmin">Optional lower bound on covariance rank.</param>
    /// <param name="kmax">Optional upper bound on covariance rank.</param>
    /// <exception cref="System.ArgumentException">The scale or population size is nonpositive, mean dimension is below two, or rank bounds do not satisfy 0 &lt;= kmin &lt;= kmax &lt; dimension.</exception>
    internal VkdCma(
        double[] mean,
        double sigma,
        int? seed = null,
        int? populationSize = null,
        int? kInit = null,
        bool kAdapt = true,
        int? kmin = null,
        int? kmax = null)
    {
        if (sigma <= 0) throw new ArgumentException("sigma must be a non-zero positive value");
        int nDim = mean.Length;
        if (nDim <= 1) throw new ArgumentException("The dimension of mean must be larger than 1");

        // λ = 4 + ⌊3 ln N⌋ ; μ = ⌊λ/2⌋   (the CMA default population)
        int popsize = populationSize ?? 4 + (int)Math.Floor(3 * Math.Log(nDim));
        if (popsize <= 0) throw new ArgumentException("popsize must be a non-zero positive value.");
        int mu = popsize / 2;

        // Standard positive recombination log-weights, normalized to sum 1 (VkD runs CMA_active=False,
        // so there are no negative weights). Same scheme as `LmCma`. μ_eff = 1 / Σ wᵢ².
        var weightsPrime = new double[mu];
        for (int i = 0; i < mu; i++) weightsPrime[i] = Math.Log(mu + 1) - Math.Log(i + 1);
        double wpSum = 0; foreach (double w in weightsPrime) wpSum += w;
        var weights = new double[mu];
        for (int i = 0; i < mu; i++) weights[i] = weightsPrime[i] / wpSum;
        double sumWSq = 0; foreach (double w in weights) sumWSq += w * w;
        double muEff = 1.0 / sumWSq;

        _nDim = nDim;
        _popsize = popsize;
        _mu = mu;
        _weights = weights;
        _muEff = muEff;
        _sigma0 = sigma;
        _kAdapt = kAdapt;

        // TPA parameters
        _cs = 0.3;
        _ds = 4.0 - 3.0 / nDim;

        int k = kInit ?? 0;
        _kInit = k;
        _k = k;
        _kActive = 0;

        if (kAdapt)
        {
            _kmin = kmin ?? 0;
            _kmax = kmax ?? nDim - 1;
            if (!(0 <= _kmin && _kmin <= _kmax && _kmax < nDim))
                throw new ArgumentException("k-adaptation bounds must satisfy 0 <= kmin <= kmax < N.");
            _kIncCond = 30.0;
            _kDecCond = _kIncCond;
            _kAdaptFactor = 1.414;
            _factorSigmaSlope = 0.1;
            _factorDiagSlope = 2.0;             // 0.3 in PPSN (different cc); GaussVkDSampler uses 2
            _acceptedSlowdown = Math.Max(1.0, _kIncCond / 10.0);
            _kAdaptDecay = 1.0 / nDim;
            _kAdaptWait = 2.0 / _kAdaptDecay - 1;
            _optConv = 0.5 * Math.Min(1.0, (double)popsize / nDim);
        }
        else
        {
            _kmax = k;
        }

        _mean = (double[])mean.Clone();
        _sigma = sigma;
        _D = new double[nDim];
        for (int i = 0; i < nDim; i++) _D[i] = 1.0;
        _V = new List<double[]>(_k);
        for (int j = 0; j < _k; j++) _V.Add(new double[nDim]);
        _S = new double[nDim];
        _pc = new double[nDim];
        _dx = new double[nDim];
        _flgInjection = false;
        _ps = 0.0;
        _g = 0;

        // k-adaptation lazy-init constants captured at pre-first-tell state (D = ones, σ = σ0, S = 0).
        _lastLogSigma = Math.Log(sigma);
        _lastLogD = new double[nDim];          // 2·log(1) = 0
        _lastLogCondCorr = new double[nDim];   // log(1 + 0) = 0
        _emaLogSigma = new Ema(1);
        _emaLogD = new Ema(nDim);
        _emaLogS = new Ema(nDim);
        _itrAfterKInc = 0;
        _lnSigmaChange = 0;
        _lnDiagChange = new double[nDim];
        _lnLambdaChange = new double[nDim];

        _askIndex = 0;
        _lastInject = null;

        _rng = np.random.RandomState();
        if (seed is not null) _rng.seed(seed.Value);
    }

    /// <summary>Number of numerical decision coordinates.</summary>
    public int Dim => _nDim;
    /// <summary>Number of evaluated individuals required for one generation.</summary>
    public int PopulationSize => _popsize;
    /// <summary>Number of completed optimizer updates.</summary>
    public int Generation => _g;

    /// <summary>Current Gaussian proposal center in transformed coordinates.</summary>
    internal double[] Mean => _mean;
    /// <summary>Global Gaussian proposal step size.</summary>
    internal double Sigma => _sigma;
    /// <summary>Allocated covariance rank.</summary>
    internal int K => _k;
    /// <summary>Number of active low-rank covariance directions.</summary>
    internal int KActive => _kActive;

    /// <summary>Resets the optimizer random stream for subsequent proposals.</summary>
    /// <param name="seed">Seed used to reset the optimizer random stream.</param>
    public void ReseedRng(int seed) => _rng.seed((uint)seed);

    // Whitening functions exposed for the live pycma parity gate (VkdCmaSamplerParityTests): the sampled
    // step for a standard normal, and its inverse. Internal — the sampler drives these through Ask/Tell.
    /// <summary>Applies the optimizer proposal transform to supplied standard-normal coordinates for parity checks.</summary>
    /// <param name="z">Standardized observation or standard-normal coordinate array.</param>
    /// <returns>The vector transformed by the current proposal geometry.</returns>
    internal double[] TransformForParity(double[] z) => Transform(z);
    /// <summary>Applies the inverse proposal transform to supplied coordinates for parity checks.</summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <returns>The vector mapped back through the inverse proposal geometry.</returns>
    internal double[] TransformInverseForParity(double[] x) => TransformInverse(x);

    // ---- Ask (GaussVkDSampler.sample) ---------------------------------------------------------

    /// <summary>
    ///     Generate one candidate of the current generation (port of <c>GaussVkDSampler.sample</c>). When TPA
    ///     injection is armed, the first two candidates are the mirrored pair <c>m ± dy</c> (batch mirror is
    ///     exact — the even candidate reuses the odd one's drawn length); the rest are fresh
    ///     <c>m + transform(N(0,I))</c> draws. This is the batch ask/tell path; the per-trial sampler uses
    ///     <see cref="AskCandidate" /> instead.
    /// </summary>
    /// <returns>One candidate vector in transformed coordinates.</returns>
    public double[] Ask()
    {
        int k = ++_askIndex;   // 1-based candidate index within the generation
        double[] step;
        if (_flgInjection && k == 1)
        {
            step = InjectedStep();
            _lastInject = step;
        }
        else if (_flgInjection && k == 2)
        {
            double[] prev = _lastInject!;
            step = new double[_nDim];
            for (int i = 0; i < _nDim; i++) step[i] = -prev[i];   // exact mirror  m − dy
        }
        else
        {
            step = Transform(Randn(_nDim));
        }
        var x = new double[_nDim];
        for (int i = 0; i < _nDim; i++) x[i] = _mean[i] + step[i];
        return x;
    }

    /// <summary>
    ///     Generate the candidate at 1-based generation position <paramref name="k" /> for the per-trial
    ///     protocol. Positions 1 and 2 are the TPA injection pair <c>m ± dy</c> with <b>independent</b> drawn
    ///     lengths but the exact <c>±dx</c> directions — a mirror in direction, not in length. Because
    ///     <c>Tell</c> recovers the pair by direction alignment (length-invariant), this drives TPA correctly
    ///     even though separately-seeded per-trial asks cannot share the odd candidate's length the way batch
    ///     <see cref="Ask" /> does.
    /// </summary>
    /// <param name="k">Zero-based candidate slot in the current generation.</param>
    /// <returns>The proposed decision vector for the requested population slot.</returns>
    internal double[] AskCandidate(int k)
    {
        double[] step;
        if (_flgInjection && k == 1) step = InjectedStep();
        else if (_flgInjection && k == 2) { var s = InjectedStep(); step = new double[_nDim]; for (int i = 0; i < _nDim; i++) step[i] = -s[i]; }
        else step = Transform(Randn(_nDim));
        var x = new double[_nDim];
        for (int i = 0; i < _nDim; i++) x[i] = _mean[i] + step[i];
        return x;
    }

    /// <summary>
    ///     The TPA injection step <c>dy = (‖N(0,I)‖ / ‖dx‖_M) · dx</c> (GaussVkDSampler.sample). <c>dx</c> is
    ///     the previous generation's weighted mean shift (D-space); dividing by its Mahalanobis norm
    ///     <c>‖dx‖_M = ‖transform⁻¹(dx)‖</c> and multiplying by a fresh sample length puts the injected step
    ///     on the same whitened scale as a <see cref="Transform" /> draw. If no mean shift exists yet
    ///     (‖dx‖ ≈ 0), fall back to an ordinary transformed draw.
    /// </summary>
    /// <returns>The displacement-scaled paired injection vector.</returns>
    private double[] InjectedStep()
    {
        double mnorm = Norm(_dx);
        double[] r = Randn(_nDim);
        double rn = 0; for (int i = 0; i < _nDim; i++) rn += r[i] * r[i]; rn = Math.Sqrt(rn);
        if (!(mnorm > Eps))
            return Transform(r);   // reuse the just-drawn normals as the fallback sample
        double scale = rn / mnorm;
        var step = new double[_nDim];
        for (int i = 0; i < _nDim; i++) step[i] = scale * _dx[i];
        return step;
    }

    /// <summary>
    ///     <c>GaussVkDSampler.transform</c>: whiten a standard normal <paramref name="z" /> into a candidate
    ///     step, <c>y = D·σ·(z + Σⱼ (√(1+Sⱼ) − 1)(z·Vⱼ) Vⱼ)</c> over the active directions.
    /// </summary>
    /// <param name="z">Standardized observation or standard-normal coordinate array.</param>
    /// <returns>The standard-normal vector transformed by the current low-rank correlation factors.</returns>
    private double[] Transform(double[] z)
    {
        var y = (double[])z.Clone();
        int ka = _kActive;
        if (ka > 0)
        {
            var coeff = new double[ka];
            for (int j = 0; j < ka; j++)
            {
                double dot = 0; double[] vj = _V[j];
                for (int i = 0; i < _nDim; i++) dot += z[i] * vj[i];
                coeff[j] = (Math.Sqrt(1.0 + _S[j]) - 1.0) * dot;
            }
            for (int j = 0; j < ka; j++)
            {
                double cj = coeff[j]; double[] vj = _V[j];
                for (int i = 0; i < _nDim; i++) y[i] += cj * vj[i];
            }
        }
        for (int i = 0; i < _nDim; i++) y[i] *= _D[i] * _sigma;
        return y;
    }

    /// <summary>
    ///     <c>GaussVkDSampler.transform_inverse</c>: the inverse whitening,
    ///     <c>x/(Dσ) + Σⱼ (1/√(1+Sⱼ) − 1)(Vⱼ·(x/(Dσ))) Vⱼ</c>.
    /// </summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <returns>The vector mapped through the inverse proposal transform.</returns>
    private double[] TransformInverse(double[] x)
    {
        var y = new double[_nDim];
        for (int i = 0; i < _nDim; i++) y[i] = x[i] / _D[i] / _sigma;
        int ka = _kActive;
        if (ka > 0)
        {
            var coeff = new double[ka];
            for (int j = 0; j < ka; j++)
            {
                double dot = 0; double[] vj = _V[j];
                for (int i = 0; i < _nDim; i++) dot += vj[i] * y[i];
                coeff[j] = (1.0 / Math.Sqrt(1.0 + _S[j]) - 1.0) * dot;
            }
            for (int j = 0; j < ka; j++)
            {
                double cj = coeff[j]; double[] vj = _V[j];
                for (int i = 0; i < _nDim; i++) y[i] += cj * vj[i];
            }
        }
        return y;
    }

    /// <summary>Mahalanobis norm w.r.t. the statistical model, <c>‖transform⁻¹(x)‖</c>.</summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <returns>The Euclidean vector length.</returns>
    private double Norm(double[] x)
    {
        double[] y = TransformInverse(x);
        double s = 0; for (int i = 0; i < _nDim; i++) s += y[i] * y[i];
        return Math.Sqrt(s);
    }

    /// <summary>Draws a vector of independent standard-normal values from the optimizer stream.</summary>
    /// <param name="n">Number of vector entries to produce.</param>
    /// <returns>The newly drawn standard-normal vector.</returns>
    private double[] Randn(int n) => _rng.randn(n).ToArray<double>();

    // ---- Tell (GaussVkDSampler.update + the ES mean recombination) ----------------------------

    /// <summary>
    ///     Complete one generation: recover the TPA step-size signal from the injected pair, recombine the
    ///     best μ into the new mean, update the evolution path, and run the restricted-covariance natural
    ///     gradient (V/S/D) plus the online k-adaptation — a faithful port of <c>GaussVkDSampler.update</c>
    ///     with the surrounding weighted-mean ES made explicit.
    /// </summary>
    /// <param name="solutions">Evaluated proposal vectors and minimization-oriented values for one coherent generation.</param>
    /// <exception cref="System.ArgumentException">The solution count differs from population size.</exception>
    public void Tell(List<(double[] X, double Value)> solutions)
    {
        if (solutions.Count != _popsize)
            throw new ArgumentException("Must tell popsize-length solutions.");

        double sigmaOld = _sigma;                          // sary is formed with σ BEFORE the TPA update
        double[] m = _mean;
        int lam = _popsize, mu = _mu, N = _nDim;

        // Sort ascending by value (stable). `sary` = the best-μ steps in D-space (÷ σ_old).
        var sorted = solutions.OrderBy(s => s.Value).ToList();
        var sary = new double[mu][];
        for (int l = 0; l < mu; l++)
        {
            var row = new double[N];
            for (int i = 0; i < N; i++) row[i] = (sorted[l].X[i] - m[i]) / sigmaOld;
            sary[l] = row;
        }
        double[] w = _weights;
        var (cc, cone, cmu) = GetParams(_k);

        // ---- TPA (Two-Point step-size Adaptation) ----
        double hsig;
        if (_flgInjection)
        {
            // ndx = previous mean shift direction; ip/im = fitness ranks of the samples most aligned
            // with +dx and −dx. Recovery is by DIRECTION (each step normalized), so the injected pair's
            // lengths are irrelevant. (GaussVkDSampler.update, the flg_injection branch.)
            double dxNorm = 0; for (int i = 0; i < N; i++) dxNorm += _dx[i] * _dx[i]; dxNorm = Math.Sqrt(dxNorm);
            int ip = 0, im = 0; double best = double.NegativeInfinity, worst = double.PositiveInfinity;
            for (int l = 0; l < lam; l++)
            {
                double yn = 0, dot = 0;
                for (int i = 0; i < N; i++) { double yi = sorted[l].X[i] - m[i]; yn += yi * yi; dot += yi * _dx[i]; }
                double inner = dot / (Math.Sqrt(yn) * dxNorm);
                if (inner > best) { best = inner; ip = l; }
                if (inner < worst) { worst = inner; im = l; }
            }
            double alphaAct = (im - ip) / (double)(lam - 1);
            _ps += _cs * (alphaAct - _ps);
            _sigma *= Math.Exp(_ps / _ds);
            hsig = _ps < 0.5 ? 1.0 : 0.0;
        }
        else
        {
            _flgInjection = true;
            hsig = 1.0;
        }

        // ---- mean recombination:  Δm = Σ wᵢ (x_{i:λ} − m),  dx = Δm / σ_old ----
        var meanShift = new double[N];
        for (int i = 0; i < N; i++)
        {
            double s = 0;
            for (int l = 0; l < mu; l++) s += w[l] * (sorted[l].X[i] - m[i]);
            meanShift[i] = s;
        }
        for (int i = 0; i < N; i++) _dx[i] = meanShift[i] / sigmaOld;

        // ---- cumulation:  pc = (1−cc) pc + hsig·√(cc(2−cc)μ_eff)·dx ----
        double pcCoef = hsig * Math.Sqrt(cc * (2 - cc) * _muEff);
        for (int i = 0; i < N; i++) _pc[i] = (1 - cc) * _pc[i] + pcCoef * _dx[i];

        // ---- restricted-covariance natural gradient: build U so that Cov ∝ D(α²I + UUᵀ)D ----
        int ka = _kActive, k = _k;
        double alpha = Math.Sqrt(Math.Abs(1 - cmu - cone + cone * (1 - hsig) * cc * (2 - cc)));

        int rankU;
        double[,] U;    // (N × rankU)
        if (cmu == 0.0)
        {
            rankU = ka + 1;
            U = new double[N, rankU];
            FillActiveVColumns(U, ka, alpha);
            for (int i = 0; i < N; i++) U[i, rankU - 1] = Math.Sqrt(cone) * (_pc[i] / _D[i]);
        }
        else if (cone == 0.0)
        {
            rankU = ka + mu;
            U = new double[N, rankU];
            FillActiveVColumns(U, ka, alpha);
            for (int l = 0; l < mu; l++)
            {
                double sq = Math.Sqrt(cmu * w[l]);
                for (int i = 0; i < N; i++) U[i, ka + l] = sq * (sary[l][i] / _D[i]);
            }
        }
        else
        {
            rankU = ka + mu + 1;
            U = new double[N, rankU];
            FillActiveVColumns(U, ka, alpha);
            for (int l = 0; l < mu; l++)
            {
                double sq = Math.Sqrt(cmu * w[l]);
                for (int i = 0; i < N; i++) U[i, ka + l] = sq * (sary[l][i] / _D[i]);
            }
            for (int i = 0; i < N; i++) U[i, rankU - 1] = Math.Sqrt(cone) * (_pc[i] / _D[i]);
        }

        // Eigendecomposition of UUᵀ, taken through the small Gram UᵀU when N > rankU (the O(Nk²+k³)
        // regime), else directly on the N×N UUᵀ (when N ≤ rankU the full CMA would be preferable).
        double[] DD; int[] idxeig; double gamma, beta;
        if (N > rankU)
        {
            var gram = new double[rankU, rankU];
            for (int a = 0; a < rankU; a++)
                for (int b = a; b < rankU; b++)
                {
                    double s = 0; for (int i = 0; i < N; i++) s += U[i, a] * U[i, b];
                    gram[a, b] = s; gram[b, a] = s;
                }
            var (dd, R) = LinAlg.Eigh(gram);
            DD = dd;
            idxeig = ArgsortDesc(DD);
            gamma = rankU <= k ? 0 : TailMean(DD, idxeig, k, N);
            beta = alpha * alpha + gamma;
            int kaNew = Math.Min(CountNonNegative(DD), k);
            for (int j = 0; j < kaNew; j++) _S[j] = (DD[idxeig[j]] - gamma) / beta;
            for (int j = 0; j < kaNew; j++)
            {
                int col = idxeig[j];
                double inv = 1.0 / Math.Sqrt(DD[col]);
                double[] vj = _V[j];
                for (int i = 0; i < N; i++)
                {
                    double s = 0; for (int c = 0; c < rankU; c++) s += U[i, c] * R[c, col];
                    vj[i] = s * inv;
                }
            }
            _kActive = ka = kaNew;
        }
        else
        {
            var uut = new double[N, N];
            for (int i = 0; i < N; i++)
                for (int jj = i; jj < N; jj++)
                {
                    double s = 0; for (int c = 0; c < rankU; c++) s += U[i, c] * U[jj, c];
                    uut[i, jj] = s; uut[jj, i] = s;
                }
            var (dd, L) = LinAlg.Eigh(uut);
            DD = dd;
            idxeig = ArgsortDesc(DD);
            gamma = rankU <= k ? 0 : TailMean(DD, idxeig, k, N);
            beta = alpha * alpha + gamma;
            int kaNew = Math.Min(CountNonNegative(DD), k);
            for (int j = 0; j < kaNew; j++) _S[j] = (DD[idxeig[j]] - gamma) / beta;
            for (int j = 0; j < kaNew; j++)
            {
                int col = idxeig[j];
                double[] vj = _V[j];
                for (int i = 0; i < N; i++) vj[i] = L[i, col];
            }
            _kActive = ka = kaNew;
        }

        // ---- diagonal update:  Dᵢ ·= √( (α² + Σ_c Uᵢc²) / (1 + Σⱼ Sⱼ Vⱼᵢ²) ) ----
        for (int i = 0; i < N; i++)
        {
            double num = alpha * alpha;
            for (int c = 0; c < rankU; c++) num += U[i, c] * U[i, c];
            double den = 1.0;
            for (int j = 0; j < ka; j++) { double v = _V[j][i]; den += _S[j] * v * v; }
            _D[i] *= Math.Sqrt(num / den);
        }

        NormalizeByDeterminant();

        // ---- online k-adaptation (PPSN 2016) ----
        if (_kAdapt)
            KAdaptation(cc, cone, cmu);

        // ---- commit the mean shift ----
        var newMean = new double[N];
        for (int i = 0; i < N; i++) newMean[i] = m[i] + meanShift[i];
        _mean = newMean;

        _g += 1;
        _askIndex = 0;
        _lastInject = null;
    }

    /// <summary>Fill U's first <paramref name="ka" /> columns with the current active directions,
    /// <c>Uᵢⱼ = Vⱼᵢ·√(Sⱼ)·α</c> (GaussVkDSampler.update, the <c>U[:, :ka]</c> assignment).</summary>
    /// <param name="U">Workspace receiving scaled active low-rank direction columns.</param>
    /// <param name="ka">Number of active low-rank covariance columns.</param>
    /// <param name="alpha">Scale applied to active covariance direction columns.</param>
    private void FillActiveVColumns(double[,] U, int ka, double alpha)
    {
        for (int j = 0; j < ka; j++)
        {
            double sq = Math.Sqrt(_S[j]) * alpha;
            double[] vj = _V[j];
            for (int i = 0; i < _nDim; i++) U[i, j] = vj[i] * sq;
        }
    }

    /// <summary>Covariance normalization by its determinant (GaussVkDSampler): divide D and pc by the
    /// geometric-mean eigenvalue so <c>det(C)</c> is pinned.</summary>
    private void NormalizeByDeterminant()
    {
        double logdet = 0;
        for (int i = 0; i < _nDim; i++) logdet += Math.Log(_D[i]);
        logdet *= 2.0;
        for (int j = 0; j < _kActive; j++) logdet += Math.Log(1.0 + _S[j]);
        double gmean = Math.Exp(logdet / _nDim / 2.0);
        for (int i = 0; i < _nDim; i++) { _D[i] /= gmean; _pc[i] /= gmean; }
    }

    /// <summary>
    ///     Online model selection of the rank <c>k</c> (GaussVkDSampler.update, the PPSN-2016 block): grow
    ///     <c>k</c> when all directions are saturated and σ / the diagonal have stabilized; shrink it by
    ///     dropping under-used directions. The "EMA" changes are the latest one-step deltas (see
    ///     <see cref="Ema" />).
    /// </summary>
    /// <param name="cc">Evolution-path accumulation coefficient.</param>
    /// <param name="cone">Rank-one covariance learning coefficient.</param>
    /// <param name="cmu">Rank-mu covariance learning coefficient.</param>
    private void KAdaptation(double cc, double cone, double cmu)
    {
        int N = _nDim;
        _itrAfterKInc += 1;

        // one-step change trackers
        _emaLogSigma.Update(new[] { Math.Log(_sigma) - _lastLogSigma });
        _lnSigmaChange = _emaLogSigma.M[0] / (_optConv / _acceptedSlowdown);
        _lastLogSigma = Math.Log(_sigma);

        var curLogD = new double[N];
        for (int i = 0; i < N; i++)
        {
            double sv = 0; for (int j = 0; j < _k; j++) { double v = _V[j][i]; sv += _S[j] * v * v; }
            curLogD[i] = 2.0 * Math.Log(_D[i]) + Math.Log(1.0 + sv);
        }
        var dLogD = new double[N];
        for (int i = 0; i < N; i++) dLogD[i] = curLogD[i] - _lastLogD[i];
        _emaLogD.Update(dLogD);
        for (int i = 0; i < N; i++) _lnDiagChange[i] = _emaLogD.M[i] / (cmu + cone);
        _lastLogD = curLogD;

        var curLogCond = new double[N];
        for (int i = 0; i < N; i++) curLogCond[i] = Math.Log(1.0 + _S[i]);
        var dLogCond = new double[N];
        for (int i = 0; i < N; i++) dLogCond[i] = curLogCond[i] - _lastLogCondCorr[i];
        _emaLogS.Update(dLogCond);
        for (int i = 0; i < N; i++) _lnLambdaChange[i] = _emaLogS.M[i] / (cmu + cone);
        _lastLogCondCorr = curLogCond;

        // increase condition: waited long enough, k below cap, ALL k directions saturated, σ and diagonal
        // both stable.  all(1+S[:k] > k_inc_cond) is vacuously true when k == 0 (start diagonal, grow rank).
        bool flgIncrease = _itrAfterKInc > _kAdaptWait && _k < _kmax
                           && Math.Abs(_lnSigmaChange) < _factorSigmaSlope;
        if (flgIncrease)
            for (int j = 0; j < _k; j++) if (!((1.0 + _S[j]) > _kIncCond)) { flgIncrease = false; break; }
        if (flgIncrease)
            for (int i = 0; i < N; i++) if (!(Math.Abs(_lnDiagChange[i]) < _factorDiagSlope)) { flgIncrease = false; break; }

        // per-direction decrease flags: direction j is under-used (1+Sⱼ small) and its eigenvalue shrinking.
        var flgDecrease = new bool[_k];
        bool anyDecrease = false;
        if (_k > _kmin)
            for (int j = 0; j < _k; j++)
            {
                flgDecrease[j] = (1.0 + _S[j] < _kDecCond) && (_lnLambdaChange[j] < 0.0);
                if (flgDecrease[j]) anyDecrease = true;
            }

        if (flgIncrease)
        {
            // ----- increase k -----
            _kActive = _k;                              // activate all current directions (reference: k_active = k)
            int newk = Math.Min(
                Math.Max((int)Math.Ceiling(_k * _kAdaptFactor), _k + 1),
                _kmax);
            for (int j = _k; j < newk; j++) _V.Add(new double[N]);   // append zero direction rows
            _k = newk;
            _itrAfterKInc = 0;
        }
        else if (_itrAfterKInc > _k * _kAdaptWait && anyDecrease)
        {
            // ----- decrease k: keep the directions not flagged -----
            var keptV = new List<double[]>();
            var keptS = new List<double>();
            for (int j = 0; j < _k; j++)
                if (!flgDecrease[j]) { keptV.Add(_V[j]); keptS.Add(_S[j]); }
            int newk = Math.Max(keptV.Count, _kmin);
            var newV = new List<double[]>(newk);
            for (int j = 0; j < newk; j++) newV.Add(j < keptV.Count ? keptV[j] : new double[N]);
            _V.Clear(); _V.AddRange(newV);
            for (int j = 0; j < keptS.Count; j++) _S[j] = keptS[j];
            for (int j = keptS.Count; j < N; j++) _S[j] = 0.0;
            _k = _kActive = newk;
        }

        NormalizeByDeterminant();
    }

    // ---- learning-rate schedule (GaussVkDSampler._get_params2) ---------------------------------

    /// <summary>Return <c>(cc, cone, cmu)</c> for the current rank <paramref name="k" /> — the PPSN-2016
    /// learning rates. μ_eff is fixed by the (positive, sum-1) weights, so only <paramref name="k" /> varies.</summary>
    /// <param name="k">Active covariance rank.</param>
    /// <returns>The adaptation coefficients for the current covariance rank.</returns>
    private (double Cc, double Cone, double Cmu) GetParams(int k)
    {
        double mueff = _muEff;
        double nelem = (double)_nDim * (k + 1);
        double cone = 2.0 / (nelem + _nDim + 2 * (k + 2) + mueff);
        double cc = Math.Sqrt(cone);
        double cmu = Math.Min(1 - cone, 2.0 * (mueff - 2 + 1.0 / mueff) / (nelem + 4 * (k + 2) + mueff));
        return (cc, cone, cmu);
    }

    // ---- small helpers ------------------------------------------------------------------------

    /// <summary>Indices of <paramref name="a" /> sorted by value DESCENDING (numpy's <c>argsort()[::-1]</c>).</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>Indices ordered from largest to smallest value.</returns>
    private static int[] ArgsortDesc(double[] a)
    {
        var idx = Enumerable.Range(0, a.Length).ToArray();
        Array.Sort(idx, (i, j) => a[j].CompareTo(a[i]));
        return idx;
    }

    /// <summary>Mean of the eigenvalues ranked <paramref name="k" /> and beyond (the tail), <c>/(N−k)</c>.</summary>
    /// <param name="dd">Eigenvalues whose trailing entries are averaged.</param>
    /// <param name="idxeig">Indices ordering covariance eigenvalues and their eigenvectors.</param>
    /// <param name="k">Number of leading eigenvalues excluded from the tail.</param>
    /// <param name="n">Number of numerical coordinates.</param>
    /// <returns>The mean of the selected eigenvalue tail.</returns>
    private static double TailMean(double[] dd, int[] idxeig, int k, int n)
    {
        double s = 0; for (int r = k; r < idxeig.Length; r++) s += dd[idxeig[r]];
        return s / (n - k);
    }

    /// <summary>Counts entries greater than or equal to zero for rank-adaptation bookkeeping.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>The number of nonnegative entries.</returns>
    private static int CountNonNegative(double[] a)
    {
        int c = 0; foreach (double v in a) if (v >= 0) c++;
        return c;
    }

    // ---- state serialization (own JSON; mirrors the SepCma / LmCma pattern) --------------------

    /// <summary>Captures constructor settings and evolving optimizer arrays in a serializable state object.</summary>
    /// <returns>A serializable snapshot of optimizer settings and evolving state.</returns>
    internal VkdCmaState ToState() => new()
    {
        NDim = _nDim,
        Sigma0 = _sigma0,
        Popsize = _popsize,
        KInit = _kInit,
        KAdapt = _kAdapt,
        Kmin = _kmin,
        Kmax = _kmax,
        Mean = (double[])_mean.Clone(),
        Sigma = _sigma,
        D = (double[])_D.Clone(),
        V = _V.Select(r => (double[])r.Clone()).ToArray(),
        S = (double[])_S.Clone(),
        Pc = (double[])_pc.Clone(),
        Dx = (double[])_dx.Clone(),
        K = _k,
        KActive = _kActive,
        FlgInjection = _flgInjection,
        Ps = _ps,
        G = _g,
        LastLogSigma = _lastLogSigma,
        LastLogD = (double[])_lastLogD.Clone(),
        LastLogCondCorr = (double[])_lastLogCondCorr.Clone(),
        ItrAfterKInc = _itrAfterKInc,
        EmaLogSigmaM = (double[])_emaLogSigma.M.Clone(),
        EmaLogDM = (double[])_emaLogD.M.Clone(),
        EmaLogSM = (double[])_emaLogS.M.Clone(),
    };

    /// <summary>Recreates optimizer constants and restores evolving state from a serialized snapshot.</summary>
    /// <param name="s">Serialized optimizer settings and evolving proposal state.</param>
    /// <returns>An optimizer restored from the supplied snapshot.</returns>
    /// <exception cref="System.ArgumentException">Stored constructor settings or population dimensions are rejected when the optimizer is recreated.</exception>
    internal static VkdCma FromState(VkdCmaState s)
    {
        var cma = new VkdCma(new double[s.NDim], s.Sigma0, seed: null,
            populationSize: s.Popsize, kInit: s.KInit, kAdapt: s.KAdapt,
            kmin: s.KAdapt ? s.Kmin : null, kmax: s.KAdapt ? s.Kmax : null);
        cma._mean = (double[])s.Mean.Clone();
        cma._sigma = s.Sigma;
        cma._D = (double[])s.D.Clone();
        cma._V.Clear();
        foreach (var r in s.V) cma._V.Add((double[])r.Clone());
        cma._S = (double[])s.S.Clone();
        cma._pc = (double[])s.Pc.Clone();
        cma._dx = (double[])s.Dx.Clone();
        cma._k = s.K;
        cma._kActive = s.KActive;
        cma._flgInjection = s.FlgInjection;
        cma._ps = s.Ps;
        cma._g = s.G;
        cma._lastLogSigma = s.LastLogSigma;
        cma._lastLogD = (double[])s.LastLogD.Clone();
        cma._lastLogCondCorr = (double[])s.LastLogCondCorr.Clone();
        cma._itrAfterKInc = s.ItrAfterKInc;
        Array.Copy(s.EmaLogSigmaM, cma._emaLogSigma.M, cma._emaLogSigma.M.Length);
        Array.Copy(s.EmaLogDM, cma._emaLogD.M, cma._emaLogD.M.Length);
        Array.Copy(s.EmaLogSM, cma._emaLogS.M, cma._emaLogS.M.Length);
        return cma;
    }

    /// <summary>
    ///     Verbatim port of pycma's <c>ExponentialMovingAverage</c> as used by <c>GaussVkDSampler</c>: it is
    ///     always constructed with <c>flg_init_with_data=False</c>, so <c>flg_init = −False = 0</c> forces the
    ///     blend factor <c>a = 1</c> on <b>every</b> update — the mean estimate <see cref="M" /> collapses to
    ///     the latest datum and the variance estimate stays 0. Preserved exactly rather than "corrected".
    /// </summary>
    private sealed class Ema
    {
        /// <summary>Mutable vector holding the exponential moving average.</summary>
        public readonly double[] M;
        private readonly double[] _s;

        /// <summary>Allocates the per-coordinate state for an exponential moving average.</summary>
        /// <param name="dim">Number of numerical coordinates.</param>
        public Ema(int dim) { M = new double[dim]; _s = new double[dim]; }

        /// <summary>Incorporates one vector observation into the exponential moving average.</summary>
        /// <param name="datum">Vector observation incorporated into the moving average.</param>
        public void Update(double[] datum)
        {
            const double a = 1.0;   // flg_init == 0 (falsy) ⇒ a = decay-if-flg_init-else-1 == 1, always
            for (int i = 0; i < M.Length; i++)
            {
                double d = datum[i] - M[i];
                _s[i] += a * ((1 - a) * d * d - _s[i]);
                M[i] += a * d;
            }
        }
    }
}

/// <summary>Serialized evolving state of a <see cref="VkdCma" /> optimizer (own JSON; see <see cref="LmCmaState" />).</summary>
internal sealed class VkdCmaState
{
    /// <summary>Serialized number of numerical decision coordinates.</summary>
    public int NDim { get; set; }
    /// <summary>Initial proposal step size retained for reconstruction.</summary>
    public double Sigma0 { get; set; }
    /// <summary>Serialized number of evaluated individuals in one generation.</summary>
    public int Popsize { get; set; }
    /// <summary>Initial configured covariance rank.</summary>
    public int KInit { get; set; }
    /// <summary>Whether covariance rank adapts during optimization.</summary>
    public bool KAdapt { get; set; }
    /// <summary>Minimum permitted covariance rank.</summary>
    public int Kmin { get; set; }
    /// <summary>Maximum permitted covariance rank.</summary>
    public int Kmax { get; set; }
    /// <summary>Current Gaussian proposal center in transformed coordinates.</summary>
    public double[] Mean { get; set; } = Array.Empty<double>();
    /// <summary>Global Gaussian proposal step size.</summary>
    public double Sigma { get; set; }
    /// <summary>Per-coordinate diagonal scales of the low-rank proposal transform.</summary>
    public double[] D { get; set; } = Array.Empty<double>();
    /// <summary>Stored inverse-transformed evolution-path vectors.</summary>
    public double[][] V { get; set; } = Array.Empty<double[]>();
    /// <summary>Low-rank covariance eigenvalue correction strengths.</summary>
    public double[] S { get; set; } = Array.Empty<double>();
    /// <summary>Evolution path used by the rank-one covariance update.</summary>
    public double[] Pc { get; set; } = Array.Empty<double>();
    /// <summary>Stored displacement used for paired proposal injection.</summary>
    public double[] Dx { get; set; } = Array.Empty<double>();
    /// <summary>Allocated covariance rank.</summary>
    public int K { get; set; }
    /// <summary>Number of active low-rank covariance directions.</summary>
    public int KActive { get; set; }
    /// <summary>Whether paired mean-displacement proposals are enabled for the next generation.</summary>
    public bool FlgInjection { get; set; }
    /// <summary>Scalar step-size adaptation path.</summary>
    public double Ps { get; set; }
    /// <summary>Serialized number of completed optimizer updates.</summary>
    public int G { get; set; }
    /// <summary>Previous log step size used to estimate adaptation trends.</summary>
    public double LastLogSigma { get; set; }
    /// <summary>Previous log diagonal scales used to estimate adaptation trends.</summary>
    public double[] LastLogD { get; set; } = Array.Empty<double>();
    /// <summary>Previous log correlation condition estimates used by rank adaptation.</summary>
    public double[] LastLogCondCorr { get; set; } = Array.Empty<double>();
    /// <summary>Number of generations since the most recent covariance-rank increase.</summary>
    public int ItrAfterKInc { get; set; }
    /// <summary>Moving-average state of log step-size changes.</summary>
    public double[] EmaLogSigmaM { get; set; } = Array.Empty<double>();
    /// <summary>Moving-average state of log diagonal-scale changes.</summary>
    public double[] EmaLogDM { get; set; } = Array.Empty<double>();
    /// <summary>Moving-average state of low-rank spectral changes.</summary>
    public double[] EmaLogSM { get; set; } = Array.Empty<double>();
}
