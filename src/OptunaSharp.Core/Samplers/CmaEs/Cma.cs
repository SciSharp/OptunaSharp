using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.CmaEs;

/// <summary>
///     CMA-ES stochastic optimizer with an ask-and-tell interface. Port of the external
///     <c>cmaes</c> package's <c>CMA</c> class (<c>cmaes/_cma.py</c>, v0.13.0) that Optuna's
///     <c>CmaEsSampler</c> drives.
///
///     <para><b>Bit-exact vs the Python <c>cmaes</c> package.</b> Every numpy operation maps to a
///     bit-identical managed route: <c>.dot</c>/<c>@</c> → <c>LinAlg.Dot</c> (CBLAS via
///     OpenBLAS), <c>np.linalg.eigh</c> → <see cref="LinAlg.Eigh" /> (LAPACK <c>syevd</c>),
///     <c>np.outer</c>/<c>np.linalg.norm</c> → <see cref="LinAlg" />, contiguous <c>np.sum</c> →
///     <c>MathUtil.PairwiseSum</c> (numpy's pairwise summation) while strided (axis) reductions
///     accumulate sequentially — mixing the two drifts ~1 ULP and would cascade; scalar
///     <c>math.*</c> → <see cref="Math" />, and <c>rng.randn</c> →
///     NumSharp's MT19937 <c>RandomState</c>. So a seeded ask/tell run reproduces the package's mean,
///     sigma and covariance byte-for-byte.</para>
/// </summary>
internal class Cma : ICmaOptimizer
{
    private const double Eps = 1e-8;
    private const double MeanMax = 1e32;
    private const double SigmaMax = 1e32;

    private readonly int _nDim;
    private readonly int _popsize;
    private readonly int _mu;
    private readonly double _muEff;

    private readonly double _cc;
    private readonly double _c1;
    private readonly double _cmu;
    private readonly double _cSigma;
    private readonly double _dSigma;
    private readonly double _cm;
    private readonly double _chiN;
    private readonly double[] _weights;

    private readonly double _sigma0;  // the initial sigma — needed to rebuild config on state restore
    private double[] _pSigma;
    private double[] _pc;
    private double[] _mean;
    private double[,] _c;
    private double _sigma;
    private double[]? _d;
    private double[,]? _b;

    private double[,]? _bounds;
    private readonly int _nMaxResampling;
    private int _g;
    private NumPyRandom _rng;

    // Learning rate adaptation (https://arxiv.org/abs/2304.03473).
    private readonly bool _lrAdapt;
    private readonly double _alpha = 1.4;
    private readonly double _betaMean = 0.1;
    private readonly double _betaSigma = 0.03;
    private readonly double _gamma = 0.1;
    private double[,] _eMean;
    private double[,] _eSigma;
    private double _vMean;
    private double _vSigma;
    private double _etaMean = 1.0;
    private double _etaSigma = 1.0;

    // Termination criteria.
    private readonly double _tolx;
    private readonly double _tolxup = 1e4;
    private readonly double _tolfun = 1e-12;
    private readonly double _tolconditioncov = 1e14;
    private readonly int _funhistTerm;
    private readonly double[] _funhistValues;

    /// <summary>Initializes a Gaussian proposal, evolution paths, and covariance learning rates.</summary>
    /// <param name="mean">Initial or current proposal center in transformed coordinates.</param>
    /// <param name="sigma">Initial positive global proposal standard deviation.</param>
    /// <param name="bounds">Optional box with one row per decision coordinate and lower/upper bound columns.</param>
    /// <param name="nMaxResampling">Maximum rejected box-infeasible proposals before coordinate repair.</param>
    /// <param name="seed">Seed used to reset the optimizer random stream.</param>
    /// <param name="populationSize">Number of individuals retained or evaluated per generation.</param>
    /// <param name="cov">Initial covariance matrix, or null for the identity default.</param>
    /// <param name="lrAdapt">Whether to adapt mean and covariance learning rates from their measured signal-to-noise ratios.</param>
    /// <exception cref="System.ArgumentException">The scale is nonpositive, the mean is empty or exceeds the supported magnitude, population size is nonpositive, or covariance shape differs from the decision dimension.</exception>
    internal Cma(
        double[] mean,
        double sigma,
        double[,]? bounds = null,
        int nMaxResampling = 100,
        int? seed = null,
        int? populationSize = null,
        double[,]? cov = null,
        bool lrAdapt = false)
    {
        if (sigma <= 0) throw new ArgumentException("sigma must be non-zero positive value");
        foreach (double m in mean)
            if (!(Math.Abs(m) < MeanMax))
                throw new ArgumentException($"Abs of all elements of mean vector must be less than {MeanMax}");

        int nDim = mean.Length;
        if (nDim <= 0) throw new ArgumentException("The dimension of mean must be positive");

        populationSize ??= 4 + (int)Math.Floor(3 * Math.Log(nDim));  // (eq. 48)
        if (populationSize <= 0) throw new ArgumentException("popsize must be non-zero positive value.");
        int popsize = populationSize.Value;

        int mu = popsize / 2;

        // (eq.49)
        var weightsPrime = new double[popsize];
        for (int i = 0; i < popsize; i++)
            weightsPrime[i] = Math.Log((popsize + 1) / 2.0) - Math.Log(i + 1);

        double muEff = Square(PairwiseSum(weightsPrime, 0, mu)) / PairwiseSum(Squared(weightsPrime, 0, mu));
        double muEffMinus = Square(PairwiseSum(weightsPrime, mu, popsize - mu))
                            / PairwiseSum(Squared(weightsPrime, mu, popsize - mu));

        const double alphaCov = 2;
        double c1 = alphaCov / (Square(nDim + 1.3) + muEff);
        double cmu = Math.Min(
            1 - c1 - 1e-8,  // 1e-8 is for large popsize.
            alphaCov * (muEff - 2 + 1 / muEff) / (Square(nDim + 2) + alphaCov * muEff / 2));

        double minAlpha = Math.Min(
            Math.Min(
                1 + c1 / cmu,                             // eq.50
                1 + 2 * muEffMinus / (muEff + 2)),        // eq.51
            (1 - c1 - cmu) / (nDim * cmu));               // eq.52

        // (eq.53) — match numpy's `np.where(wp>=0, 1/positive_sum * wp, min_alpha/negative_sum * wp)`:
        // the scalar coefficient is formed FIRST (reciprocal, then multiply), not a direct divide.
        double positiveSum = PairwiseSum(weightsPrime.Where(w => w > 0).ToArray());
        double negativeSum = PairwiseSum(weightsPrime.Where(w => w < 0).Select(Math.Abs).ToArray());
        double posScale = 1 / positiveSum;
        double negScale = minAlpha / negativeSum;
        var weights = new double[popsize];
        for (int i = 0; i < popsize; i++)
            weights[i] = weightsPrime[i] >= 0
                ? posScale * weightsPrime[i]
                : negScale * weightsPrime[i];

        double cm = 1;  // (eq. 54)

        // learning rate for the cumulation for the step-size control (eq.55)
        double cSigma = (muEff + 2) / (nDim + muEff + 5);
        double dSigma = 1 + 2 * Math.Max(0, Math.Sqrt((muEff - 1) / (nDim + 1)) - 1) + cSigma;

        // learning rate for cumulation for the rank-one update (eq.56)
        double cc = (4 + muEff / nDim) / (nDim + 4 + 2 * muEff / nDim);

        _nDim = nDim;
        _popsize = popsize;
        _mu = mu;
        _muEff = muEff;
        _cc = cc;
        _c1 = c1;
        _cmu = cmu;
        _cSigma = cSigma;
        _dSigma = dSigma;
        _cm = cm;

        // E||N(0, I)|| (p.28)
        _chiN = Math.Sqrt(_nDim) * (1.0 - 1.0 / (4.0 * _nDim) + 1.0 / (21.0 * (_nDim * _nDim)));
        _weights = weights;

        _pSigma = new double[nDim];
        _pc = new double[nDim];
        _mean = (double[])mean.Clone();

        if (cov is null)
            _c = Eye(nDim);
        else
        {
            if (cov.GetLength(0) != nDim || cov.GetLength(1) != nDim)
                throw new ArgumentException("Invalid shape of covariance matrix");
            _c = cov;
        }

        _sigma = sigma;
        _sigma0 = sigma;
        _d = null;
        _b = null;

        _bounds = bounds;
        _nMaxResampling = nMaxResampling;
        _g = 0;
        _rng = np.random.RandomState();
        if (seed is not null) _rng.seed(seed.Value);

        _lrAdapt = lrAdapt;
        _eMean = new double[nDim, 1];
        _eSigma = new double[nDim * nDim, 1];

        _tolx = 1e-12 * sigma;
        _funhistTerm = 10 + (int)Math.Ceiling(30.0 * nDim / popsize);
        _funhistValues = new double[_funhistTerm * 2];
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
    /// <summary>Current proposal covariance representation.</summary>
    internal double[,] Cov => _c;

    /// <summary>
    ///     Replace the covariance matrix and invalidate the cached eigendecomposition (<c>B</c>/<c>D</c>),
    ///     so the next <see cref="Ask" /> re-derives it from the new <c>C</c> (which
    ///     <see cref="EigenDecomposition" /> also re-symmetrizes and reconstructs). This is the
    ///     <c>es.C = …; safe_decompose()</c> hook JANUS's <c>cmix_mode="reshape"</c> uses to mix an
    ///     inverse-curvature term into the proposal covariance; it is not part of the ordinary CMA-ES loop
    ///     (a seeded ask/tell run that never calls it is byte-for-byte unchanged).
    /// </summary>
    /// <param name="c">Replacement proposal covariance in transformed coordinates.</param>
    internal void SetCovariance(double[,] c)
    {
        _c = (double[,])c.Clone();
        _b = null;
        _d = null;
    }

    /// <summary>Optional lower/upper proposal bounds in coordinate order.</summary>
    internal double[,]? Bounds => _bounds;      // needed by CmaWm to rebuild its decomposition on restore
    /// <summary>Maximum rejected proposals before box repair is applied.</summary>
    internal int NMaxResampling => _nMaxResampling;
    /// <summary>Evolution path used to adapt the global step size.</summary>
    internal double[] PSigma => _pSigma;
    /// <summary>Evolution path used by the rank-one covariance update.</summary>
    internal double[] Pc => _pc;
    /// <summary>Rank-ordered recombination weights, including negative active-covariance weights.</summary>
    internal double[] Weights => _weights;
    /// <summary>Variance-effective number of recombination parents.</summary>
    internal double MuEff => _muEff;
    /// <summary>Learning rate for the rank-one covariance update.</summary>
    internal double C1 => _c1;
    /// <summary>Learning rate for the rank-mu covariance update.</summary>
    internal double Cmu => _cmu;

    /// <summary>Resets the optimizer random stream for subsequent proposals.</summary>
    /// <param name="seed">Seed used to reset the optimizer random stream.</param>
    public void ReseedRng(int seed) => _rng.seed((uint)seed);

    // ---- state serialization ------------------------------------------------------------------
    // Optuna pickles the optimizer into system_attrs; we serialize our OWN JSON instead (pickle is
    // a non-goal — [Misaligned], no cross-language mid-CMA resume; see CmaEsSampler). The readonly
    // configuration (weights, learning rates, tolerances, …) is a pure function of the constructor
    // inputs, so restore reruns the constructor and overwrites only the evolving state — no field
    // is re-derived by hand. The RNG is not persisted (Optuna drops it too); the sampler reseeds
    // the optimizer every trial before ask().

    /// <summary>Captures constructor settings and evolving optimizer arrays in a serializable state object.</summary>
    /// <returns>A serializable snapshot of optimizer settings and evolving state.</returns>
    internal CmaState ToState() => new()
    {
        NDim = _nDim,
        Sigma0 = _sigma0,
        Bounds = ToJagged(_bounds),
        NMaxResampling = _nMaxResampling,
        Popsize = _popsize,
        LrAdapt = _lrAdapt,
        G = _g,
        Mean = (double[])_mean.Clone(),
        Sigma = _sigma,
        // Mirror Optuna's pickle: `_compress_symmetric` keeps only the UPPER triangle and
        // `_decompress_symmetric` mirrors it into both, so a restored covariance is EXACTLY
        // symmetric-from-upper. `B·diag(D²)·Bᵀ` via CBLAS is ~1 ULP asymmetric, and the next
        // eigendecomposition's `(C+Cᵀ)/2` would otherwise average `(upper+lower)/2 ≠ upper`,
        // diverging the eigenvectors vs Python. Store the upper triangle mirrored, verbatim.
        C = ToJagged(SymmetrizeFromUpper(_c))!,
        PSigma = (double[])_pSigma.Clone(),
        Pc = (double[])_pc.Clone(),
        FunhistValues = (double[])_funhistValues.Clone(),
        EMean = ToJagged(_eMean)!,
        ESigma = ToJagged(_eSigma)!,
        VMean = _vMean,
        VSigma = _vSigma,
        EtaMean = _etaMean,
        EtaSigma = _etaSigma,
    };

    /// <summary>Recreates optimizer constants and restores evolving state from a serialized snapshot.</summary>
    /// <param name="s">Serialized optimizer settings and evolving proposal state.</param>
    /// <returns>An optimizer restored from the supplied snapshot.</returns>
    /// <exception cref="System.ArgumentException">Stored constructor settings or population dimensions are rejected when the optimizer is recreated.</exception>
    internal static Cma FromState(CmaState s)
    {
        var cma = new Cma(new double[s.NDim], s.Sigma0, FromJagged(s.Bounds), s.NMaxResampling,
            seed: null, populationSize: s.Popsize, cov: null, lrAdapt: s.LrAdapt);
        cma.RestoreEvolvingState(s);
        return cma;
    }

    /// <summary>Restores proposal state and adaptation statistics after constructor constants have been rebuilt.</summary>
    /// <param name="s">Serialized optimizer settings and evolving proposal state.</param>
    private void RestoreEvolvingState(CmaState s)
    {
        _g = s.G;
        _mean = (double[])s.Mean.Clone();
        _sigma = s.Sigma;
        _c = FromJagged(s.C)!;
        _pSigma = (double[])s.PSigma.Clone();
        _pc = (double[])s.Pc.Clone();
        Array.Copy(s.FunhistValues, _funhistValues, _funhistValues.Length);
        _b = null;
        _d = null;
        if (_lrAdapt)
        {
            _eMean = FromJagged(s.EMean)!;
            _eSigma = FromJagged(s.ESigma)!;
            _vMean = s.VMean;
            _vSigma = s.VSigma;
            _etaMean = s.EtaMean;
            _etaSigma = s.EtaSigma;
        }
    }

    /// <summary>Sample a parameter.</summary>
    /// <returns>One candidate vector in transformed coordinates.</returns>
    public double[] Ask()
    {
        for (int i = 0; i < _nMaxResampling; i++)
        {
            double[] x = SampleSolution();
            if (IsFeasible(x))
                return x;
        }
        double[] last = SampleSolution();
        return RepairInfeasibleParams(last);
    }

    /// <summary>Refreshes the covariance factorization used to sample and whiten proposal vectors.</summary>
    /// <returns>Orthogonal eigenvector columns B and square roots D of the covariance eigenvalues.</returns>
    private (double[,] B, double[] D) EigenDecomposition()
    {
        if (_b is not null && _d is not null)
            return (_b, _d);

        // _C = (_C + _C.T) / 2
        for (int i = 0; i < _nDim; i++)
            for (int j = i + 1; j < _nDim; j++)
            {
                double avg = (_c[i, j] + _c[j, i]) / 2;
                _c[i, j] = avg;
                _c[j, i] = avg;
            }

        var (d2, b) = LinAlg.Eigh(_c);
        var d = new double[_nDim];
        for (int i = 0; i < _nDim; i++)
            d[i] = Math.Sqrt(d2[i] < 0 ? Eps : d2[i]);

        // _C = B @ diag(D**2) @ B.T
        var diagD2 = Diag(d.Select(v => v * v).ToArray());
        _c = LinAlg.Dot(LinAlg.Dot(b, diagD2), Transpose(b));

        _b = b;
        _d = d;
        return (b, d);
    }

    /// <summary>Python <c>CMA._sample_solution()</c> — one raw draw <c>~ N(m, σ² C)</c> (no bounds
    /// check/repair). <see cref="CmaWm" /> calls this directly, bypassing <see cref="Ask" />'s
    /// feasibility loop, because it does its own continuous-only feasibility handling.</summary>
    /// <returns>A proposal vector before box-feasibility repair.</returns>
    internal double[] SampleSolution()
    {
        var (b, d) = EigenDecomposition();
        double[] z = _rng.randn(_nDim).ToArray<double>();  // ~ N(0, I)
        double[] y = LinAlg.Dot(LinAlg.Dot(b, Diag(d)), z);  // ~ N(0, C)
        var x = new double[_nDim];
        for (int i = 0; i < _nDim; i++)
            x[i] = _mean[i] + _sigma * y[i];  // ~ N(m, σ² C)
        return x;
    }

    /// <summary>Checks whether every proposal coordinate lies within its configured box.</summary>
    /// <param name="param">Decision vector whose box feasibility is checked or repaired.</param>
    /// <returns>True if every coordinate lies inside its configured bounds.</returns>
    private bool IsFeasible(double[] param)
    {
        if (_bounds is null) return true;
        for (int i = 0; i < _nDim; i++)
            if (!(param[i] >= _bounds[i, 0] && param[i] <= _bounds[i, 1]))
                return false;
        return true;
    }

    /// <summary>Clamps a proposal to the configured box after rejection sampling is exhausted.</summary>
    /// <param name="param">Decision vector whose box feasibility is checked or repaired.</param>
    /// <returns>The proposal with out-of-box coordinates clamped to the nearest bound.</returns>
    private double[] RepairInfeasibleParams(double[] param)
    {
        if (_bounds is null) return param;
        var result = new double[_nDim];
        for (int i = 0; i < _nDim; i++)
        {
            double v = param[i];
            if (v < _bounds[i, 0]) v = _bounds[i, 0];
            if (v > _bounds[i, 1]) v = _bounds[i, 1];
            result[i] = v;
        }
        return result;
    }

    /// <summary>Tell evaluation values.</summary>
    /// <param name="solutions">Evaluated proposal vectors and minimization-oriented values for one coherent generation.</param>
    /// <exception cref="System.ArgumentException">The solution count differs from population size or a coordinate exceeds the supported magnitude.</exception>
    public void Tell(List<(double[] X, double Value)> solutions)
    {
        if (solutions.Count != _popsize)
            throw new ArgumentException("Must tell popsize-length solutions.");
        foreach (var s in solutions)
            foreach (double v in s.X)
                if (!(Math.Abs(v) < MeanMax))
                    throw new ArgumentException(
                        $"Abs of all param values must be less than {MeanMax} to avoid overflow errors");

        _g += 1;
        // Python's list.sort is stable; OrderBy is a stable sort.
        var sorted = solutions.OrderBy(s => s.Value).ToList();

        int funhistIdx = 2 * (Generation % _funhistTerm);
        _funhistValues[funhistIdx] = sorted[0].Value;
        _funhistValues[funhistIdx + 1] = sorted[^1].Value;

        var (b, d) = EigenDecomposition();
        _b = null;
        _d = null;

        double[]? oldMean = null;
        double oldSigma = 0;
        double[,]? oldSigmaMat = null;
        double[,]? oldInvSqrtC = null;
        if (_lrAdapt)
        {
            oldMean = (double[])_mean.Clone();
            oldSigma = _sigma;
            oldSigmaMat = Scale(_c, Math.Pow(_sigma, 2));  // Python scalar sigma**2 (not sigma*sigma)
            oldInvSqrtC = LinAlg.Dot(LinAlg.Dot(b, Diag(d.Select(v => 1 / v).ToArray())), Transpose(b));
        }

        // x_k ~ N(m, σ² C); y_k = (x_k - m) / σ ~ N(0, C).
        var yk = new double[_popsize, _nDim];
        for (int k = 0; k < _popsize; k++)
            for (int i = 0; i < _nDim; i++)
                yk[k, i] = (sorted[k].X[i] - _mean[i]) / _sigma;

        // y_w = np.sum(y_k[:μ].T * weights[:μ], axis=1)  (eq.41). `y_k[:μ].T` is an F-contiguous
        // view, so the broadcast product stays F-contiguous and the axis=1 reduction is STRIDED →
        // numpy sums it SEQUENTIALLY (its pairwise 8-way only fires on a contiguous/last-axis
        // reduction). A plain accumulation is required — PairwiseSum here diverges once μ ≥ 8.
        var yw = new double[_nDim];
        for (int i = 0; i < _nDim; i++)
        {
            double s = 0;
            for (int k = 0; k < _mu; k++)
                s += yk[k, i] * _weights[k];
            yw[i] = s;
        }

        for (int i = 0; i < _nDim; i++)
            _mean[i] += _cm * _sigma * yw[i];

        // C^(-1/2) = B D^(-1) B^T.
        double[,] c2 = LinAlg.Dot(LinAlg.Dot(b, Diag(d.Select(v => 1 / v).ToArray())), Transpose(b));

        // Step-size control.
        double[] c2yw = LinAlg.Dot(c2, yw);
        double coef = Math.Sqrt(_cSigma * (2 - _cSigma) * _muEff);
        for (int i = 0; i < _nDim; i++)
            _pSigma[i] = (1 - _cSigma) * _pSigma[i] + coef * c2yw[i];

        double normPSigma = LinAlg.Norm(_pSigma);
        _sigma *= Math.Exp((_cSigma / _dSigma) * (normPSigma / _chiN - 1));
        _sigma = Math.Min(_sigma, SigmaMax);

        // Covariance matrix adaptation.
        double hSigmaCondLeft = normPSigma / Math.Sqrt(1 - Math.Pow(1 - _cSigma, 2 * (_g + 1)));
        double hSigmaCondRight = (1.4 + 2.0 / (_nDim + 1)) * _chiN;
        double hSigma = hSigmaCondLeft < hSigmaCondRight ? 1.0 : 0.0;  // (p.28)

        // (eq.45)
        double pcCoef = hSigma * Math.Sqrt(_cc * (2 - _cc) * _muEff);
        for (int i = 0; i < _nDim; i++)
            _pc[i] = (1 - _cc) * _pc[i] + pcCoef * yw[i];

        // (eq.46) w_io = weights * where(weights>=0, 1, n_dim / (‖C^(-1/2) y_kᵀ‖_axis0² + eps)).
        // `np.linalg.norm(M, axis=0)` reduces M·M over the STRIDED axis 0 sequentially (numpy's
        // pairwise 8-way only fires on a contiguous/last-axis reduction), then takes sqrt; the
        // outer `** 2` squares that result back — so it is (√(Σᵢ colᵢ²))², NOT Σ colᵢ² (the
        // sqrt→square round-trip is not the identity in float64). Both are reproduced verbatim.
        double[,] c2ykT = LinAlg.Dot(c2, Transpose(yk));  // (n × popsize)
        var wio = new double[_popsize];
        for (int k = 0; k < _popsize; k++)
        {
            double factor;
            if (_weights[k] >= 0)
                factor = 1;
            else
            {
                double colSumSq = 0;
                for (int i = 0; i < _nDim; i++) colSumSq += c2ykT[i, k] * c2ykT[i, k];  // axis=0 sequential
                double colNorm = Math.Sqrt(colSumSq);
                factor = _nDim / (colNorm * colNorm + Eps);  // norm**2 (the sqrt→square round-trip)
            }
            wio[k] = _weights[k] * factor;
        }

        double deltaHSigma = (1 - hSigma) * _cc * (2 - _cc);  // (p.28)

        // (eq.47)
        double[,] rankOne = LinAlg.Outer(_pc, _pc);
        // rank_mu = np.sum(w_io.reshape(-1,1,1) * einsum("...i,...j->...ij", y_k, y_k), axis=0).
        // The einsum forms the outer product yₖᵢ·yₖⱼ FIRST, then the broadcast multiplies by w_io[k];
        // so each term is w_io[k]·(yₖᵢ·yₖⱼ), NOT (w_io[k]·yₖᵢ)·yₖⱼ (a different float rounding). The
        // sum is over axis 0 (popsize) of a (popsize,n,n) array — the strided outer axis → sequential.
        var rankMu = new double[_nDim, _nDim];
        for (int i = 0; i < _nDim; i++)
            for (int j = 0; j < _nDim; j++)
            {
                double s = 0;
                for (int k = 0; k < _popsize; k++)
                    s += wio[k] * (yk[k, i] * yk[k, j]);
                rankMu[i, j] = s;
            }

        double sumWeights = MathUtil.PairwiseSum(_weights);
        double cCoef = 1 + _c1 * deltaHSigma - _c1 - _cmu * sumWeights;
        for (int i = 0; i < _nDim; i++)
            for (int j = 0; j < _nDim; j++)
                _c[i, j] = cCoef * _c[i, j] + _c1 * rankOne[i, j] + _cmu * rankMu[i, j];

        if (_lrAdapt)
            LrAdaptation(oldMean!, oldSigma, oldSigmaMat!, oldInvSqrtC!);
    }

    /// <summary>Adjusts mean and covariance learning rates using measured update signal-to-noise ratios.</summary>
    /// <param name="oldMean">Proposal center before the current generation update.</param>
    /// <param name="oldSigma">Global proposal scale before the current generation update.</param>
    /// <param name="oldSigmaMat">Scaled covariance before the current generation update.</param>
    /// <param name="oldInvSqrtC">Inverse covariance square root before the current generation update.</param>
    private void LrAdaptation(double[] oldMean, double oldSigma, double[,] oldSigmaMat, double[,] oldInvSqrtC)
    {
        // one-step difference of the parameters
        var deltaMean = new double[_nDim, 1];
        for (int i = 0; i < _nDim; i++) deltaMean[i, 0] = _mean[i] - oldMean[i];

        double[,] sigmaMat = Scale(_c, Math.Pow(_sigma, 2));  // Python scalar sigma**2
        var deltaSigma = new double[_nDim, _nDim];
        for (int i = 0; i < _nDim; i++)
            for (int j = 0; j < _nDim; j++)
                deltaSigma[i, j] = sigmaMat[i, j] - oldSigmaMat[i, j];

        // local coordinate — `old_invsqrtC / old_sigma` is an elementwise DIVIDE, not `* (1/σ)`.
        double[,] oldInvSqrtSigma = DivideScalar(oldInvSqrtC, oldSigma);
        double[,] locDeltaMean = LinAlg.Dot(oldInvSqrtSigma, deltaMean);  // (n × 1)
        double[,] locDeltaSigmaMat = LinAlg.Dot(oldInvSqrtSigma, LinAlg.Dot(deltaSigma, oldInvSqrtSigma));
        // reshape to (n*n × 1) / sqrt(2)
        var locDeltaSigma = new double[_nDim * _nDim, 1];
        double sqrt2 = Math.Sqrt(2);
        for (int i = 0; i < _nDim; i++)
            for (int j = 0; j < _nDim; j++)
                locDeltaSigma[i * _nDim + j, 0] = locDeltaSigmaMat[i, j] / sqrt2;

        // moving averages E, V
        for (int i = 0; i < _nDim; i++)
            _eMean[i, 0] = (1 - _betaMean) * _eMean[i, 0] + _betaMean * locDeltaMean[i, 0];
        for (int i = 0; i < _nDim * _nDim; i++)
            _eSigma[i, 0] = (1 - _betaSigma) * _eSigma[i, 0] + _betaSigma * locDeltaSigma[i, 0];
        // `norm(...)**2` is a Python scalar power (ucrt pow), not x*x.
        _vMean = (1 - _betaMean) * _vMean + _betaMean * Math.Pow(NormCol(locDeltaMean), 2);
        _vSigma = (1 - _betaSigma) * _vSigma + _betaSigma * Math.Pow(NormCol(locDeltaSigma), 2);

        // estimate SNR
        double sqnormEMean = Math.Pow(NormCol(_eMean), 2);
        double hatSnrMean = (sqnormEMean - (_betaMean / (2 - _betaMean)) * _vMean) / (_vMean - sqnormEMean);
        double sqnormESigma = Math.Pow(NormCol(_eSigma), 2);
        double hatSnrSigma = (sqnormESigma - (_betaSigma / (2 - _betaSigma)) * _vSigma) / (_vSigma - sqnormESigma);

        // update learning rate
        double beforeEtaMean = _etaMean;
        double relSnrMean = Clip(hatSnrMean / _alpha / _etaMean - 1, -1, 1);
        _etaMean *= Math.Exp(Math.Min(_gamma * _etaMean, _betaMean) * relSnrMean);
        double relSnrSigma = Clip(hatSnrSigma / _alpha / _etaSigma - 1, -1, 1);
        _etaSigma *= Math.Exp(Math.Min(_gamma * _etaSigma, _betaSigma) * relSnrSigma);
        _etaMean = Math.Min(_etaMean, 1.0);
        _etaSigma = Math.Min(_etaSigma, 1.0);

        // update parameters
        for (int i = 0; i < _nDim; i++)
            _mean[i] = oldMean[i] + _etaMean * deltaMean[i, 0];
        var sigmaNew = new double[_nDim, _nDim];
        for (int i = 0; i < _nDim; i++)
            for (int j = 0; j < _nDim; j++)
                sigmaNew[i, j] = oldSigmaMat[i, j] + _etaSigma * deltaSigma[i, j];

        // decompose Sigma into sigma and C
        var (eigs, _) = LinAlg.Eigh(sigmaNew);
        double logEigSum = 0;
        for (int i = 0; i < _nDim; i++) logEigSum += Math.Log(eigs[i]);  // Python: sum([...]) sequential
        _sigma = Math.Exp(logEigSum / 2.0 / _nDim);
        _sigma = Math.Min(_sigma, SigmaMax);
        _c = DivideScalar(sigmaNew, Math.Pow(_sigma, 2));  // `Sigma / sigma**2` elementwise divide

        // step-size correction
        _sigma *= beforeEtaMean / _etaMean;
    }

    /// <summary>Checks numerical and progress criteria that indicate the current optimizer has converged or stalled.</summary>
    /// <returns>True when a numerical or convergence stopping criterion is met.</returns>
    internal bool ShouldStop()
    {
        var (b, d) = EigenDecomposition();
        double[] dc = DiagOf(_c);

        if (Generation > _funhistTerm && Max(_funhistValues) - Min(_funhistValues) < _tolfun)
            return true;

        bool allTolx = true;
        for (int i = 0; i < _nDim; i++)
            if (!(_sigma * dc[i] < _tolx) || !(_sigma * _pc[i] < _tolx)) { allTolx = false; break; }
        if (allTolx) return true;

        if (_sigma * Max(d) > _tolxup) return true;

        for (int i = 0; i < _nDim; i++)
            if (_mean[i] == _mean[i] + 0.2 * _sigma * Math.Sqrt(dc[i]))
                return true;

        int axis = Generation % Dim;
        bool noEffectAxis = true;
        for (int i = 0; i < _nDim; i++)
            if (_mean[i] != _mean[i] + 0.1 * _sigma * d[axis] * b[i, axis]) { noEffectAxis = false; break; }
        if (noEffectAxis) return true;

        double conditionCov = Max(d) / Min(d);
        if (conditionCov > _tolconditioncov) return true;

        return false;
    }

    // ---- small numpy-shaped helpers -----------------------------------------------------------

    /// <summary>Accumulates a vector using the NumPy-compatible pairwise reduction order.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>The accumulated value using the reference pairwise reduction order.</returns>
    private static double PairwiseSum(double[] a) => MathUtil.PairwiseSum(a);
    /// <summary>Accumulates a vector using the NumPy-compatible pairwise reduction order.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="offset">Zero-based start of the active vector slice.</param>
    /// <param name="n">Number of entries in the active vector slice.</param>
    /// <returns>The accumulated value using the reference pairwise reduction order.</returns>
    private static double PairwiseSum(double[] a, int offset, int n) => MathUtil.PairwiseSum(a, offset, n);
    /// <summary>Computes the scalar second power used by adaptation equations.</summary>
    /// <param name="x">Scalar operand used in the comparison or numerical transformation.</param>
    /// <returns>The product of the scalar with itself.</returns>
    private static double Square(double x) => x * x;

    /// <summary>Copies a vector slice while squaring each entry.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="offset">Zero-based start of the active vector slice.</param>
    /// <param name="n">Number of entries in the active vector slice.</param>
    /// <returns>A newly allocated vector of squared slice entries.</returns>
    private static double[] Squared(double[] a, int offset, int n)
    {
        var r = new double[n];
        for (int i = 0; i < n; i++) r[i] = a[offset + i] * a[offset + i];
        return r;
    }

    /// <summary>Allocates a square identity matrix.</summary>
    /// <param name="n">Number of numerical coordinates.</param>
    /// <returns>The square identity matrix.</returns>
    private static double[,] Eye(int n)
    {
        var m = new double[n, n];
        for (int i = 0; i < n; i++) m[i, i] = 1.0;
        return m;
    }

    /// <summary>Allocates a diagonal matrix with the supplied vector on its diagonal.</summary>
    /// <param name="v">Vector operand aligned with the matrix coordinates.</param>
    /// <returns>A square matrix whose off-diagonal entries are zero.</returns>
    private static double[,] Diag(double[] v)
    {
        int n = v.Length;
        var m = new double[n, n];
        for (int i = 0; i < n; i++) m[i, i] = v[i];
        return m;
    }

    /// <summary>Extracts the main diagonal of a square matrix.</summary>
    /// <param name="m">Rectangular matrix operand.</param>
    /// <returns>A newly allocated main-diagonal vector.</returns>
    private static double[] DiagOf(double[,] m)
    {
        int n = m.GetLength(0);
        var v = new double[n];
        for (int i = 0; i < n; i++) v[i] = m[i, i];
        return v;
    }

    /// <summary>Allocates a matrix with its row and column axes exchanged.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>A newly allocated transposed matrix.</returns>
    private static double[,] Transpose(double[,] a)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        var t = new double[cols, rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t[j, i] = a[i, j];
        return t;
    }

    /// <summary>Allocates a matrix multiplied by a scalar factor.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="s">Scalar multiplier applied to every matrix entry.</param>
    /// <returns>A newly allocated scaled matrix.</returns>
    private static double[,] Scale(double[,] a, double s)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        var r = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                r[i, j] = a[i, j] * s;
        return r;
    }

    /// <summary>Elementwise <c>a / s</c> — numpy's <c>ndarray / scalar</c> is a divide, not <c>a * (1/s)</c>.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="s">Scalar multiplier applied to every matrix entry.</param>
    /// <returns>A matrix copy divided entry-wise by the scalar.</returns>
    private static double[,] DivideScalar(double[,] a, double s)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        var r = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                r[i, j] = a[i, j] / s;
        return r;
    }

    /// <summary>Computes the Euclidean norm of a column-shaped matrix.</summary>
    /// <param name="col">Zero-based column index or column-shaped matrix.</param>
    /// <returns>The Euclidean length of the supplied column.</returns>
    private static double NormCol(double[,] col)
    {
        int n = col.GetLength(0);
        var v = new double[n];
        for (int i = 0; i < n; i++) v[i] = col[i, 0];
        return LinAlg.Norm(v);
    }

    /// <summary>Clamps a scalar to an inclusive interval.</summary>
    /// <param name="v">Scalar operand used in the comparison or numerical transformation.</param>
    /// <param name="lo">Inclusive lower clipping bound.</param>
    /// <param name="hi">Inclusive upper clipping bound.</param>
    /// <returns>The value restricted to the inclusive interval.</returns>
    private static double Clip(double v, double lo, double hi) => Math.Min(Math.Max(v, lo), hi);
    /// <summary>Finds the largest vector entry using the reference reduction semantics.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>The largest vector entry.</returns>
    private static double Max(double[] a) { double m = a[0]; foreach (double v in a) if (v > m) m = v; return m; }
    /// <summary>Finds the smallest vector entry using the reference reduction semantics.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>The smallest vector entry.</returns>
    private static double Min(double[] a) { double m = a[0]; foreach (double v in a) if (v < m) m = v; return m; }

    /// <summary>
    ///     Return a copy of the symmetric matrix <paramref name="m" /> with both triangles taken
    ///     from its UPPER triangle (<c>r[i,j] = r[j,i] = m[i,j]</c> for <c>i ≤ j</c>) — the exact
    ///     effect of Optuna's <c>_compress_symmetric</c>/<c>_decompress_symmetric</c> pickle pair.
    /// </summary>
    /// <param name="m">Rectangular matrix operand.</param>
    /// <returns>A symmetric matrix formed by mirroring the upper triangle.</returns>
    private static double[,] SymmetrizeFromUpper(double[,] m)
    {
        int n = m.GetLength(0);
        var r = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = i; j < n; j++)
            {
                r[i, j] = m[i, j];
                r[j, i] = m[i, j];
            }
        return r;
    }

    /// <summary>Copies a rectangular matrix into row arrays for serialization.</summary>
    /// <param name="m">Rectangular matrix operand.</param>
    /// <returns>Copied matrix rows, or null when the source matrix is null.</returns>
    private static double[][]? ToJagged(double[,]? m)
    {
        if (m is null) return null;
        int rows = m.GetLength(0), cols = m.GetLength(1);
        var j = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            j[i] = new double[cols];
            for (int c = 0; c < cols; c++) j[i][c] = m[i, c];
        }
        return j;
    }

    /// <summary>Copies serialized row arrays into rectangular matrix storage.</summary>
    /// <param name="j">Serialized matrix row arrays.</param>
    /// <returns>A rectangular copy of the rows, or null when the source is null.</returns>
    private static double[,]? FromJagged(double[][]? j)
    {
        if (j is null) return null;
        int rows = j.Length, cols = rows == 0 ? 0 : j[0].Length;
        var m = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int c = 0; c < cols; c++) m[i, c] = j[i][c];
        return m;
    }
}

/// <summary>
///     Serialized evolving state of a <see cref="Cma" /> optimizer — the OptunaSharp analogue of
///     Optuna's pickled <c>cmaes.CMA</c> (see <see cref="Cma.ToState" />/<see cref="Cma.FromState" />).
///     Plain properties so it round-trips through <c>System.Text.Json</c>. Two-dimensional matrices
///     are stored as jagged arrays for portable JSON.
/// </summary>
internal sealed class CmaState
{
    /// <summary>Serialized number of numerical decision coordinates.</summary>
    public int NDim { get; set; }
    /// <summary>Initial proposal step size retained for reconstruction.</summary>
    public double Sigma0 { get; set; }
    /// <summary>Optional lower/upper proposal bounds in coordinate order.</summary>
    public double[][]? Bounds { get; set; }
    /// <summary>Maximum rejected proposals before box repair is applied.</summary>
    public int NMaxResampling { get; set; }
    /// <summary>Serialized number of evaluated individuals in one generation.</summary>
    public int Popsize { get; set; }
    /// <summary>Whether signal-to-noise learning-rate adaptation is enabled.</summary>
    public bool LrAdapt { get; set; }
    /// <summary>Serialized number of completed optimizer updates.</summary>
    public int G { get; set; }
    /// <summary>Current Gaussian proposal center in transformed coordinates.</summary>
    public double[] Mean { get; set; } = System.Array.Empty<double>();
    /// <summary>Global Gaussian proposal step size.</summary>
    public double Sigma { get; set; }
    /// <summary>Serialized covariance representation used to restore the proposal geometry.</summary>
    public double[][] C { get; set; } = System.Array.Empty<double[]>();
    /// <summary>Evolution path used to adapt the global step size.</summary>
    public double[] PSigma { get; set; } = System.Array.Empty<double>();
    /// <summary>Evolution path used by the rank-one covariance update.</summary>
    public double[] Pc { get; set; } = System.Array.Empty<double>();
    /// <summary>Recent objective-value history used by convergence checks.</summary>
    public double[] FunhistValues { get; set; } = System.Array.Empty<double>();
    /// <summary>Moving-average state of normalized mean updates.</summary>
    public double[][] EMean { get; set; } = System.Array.Empty<double[]>();
    /// <summary>Moving-average state of normalized covariance updates.</summary>
    public double[][] ESigma { get; set; } = System.Array.Empty<double[]>();
    /// <summary>Estimated variance of the normalized mean-update signal.</summary>
    public double VMean { get; set; }
    /// <summary>Estimated variance of the normalized covariance-update signal.</summary>
    public double VSigma { get; set; }
    /// <summary>Adaptive learning-rate multiplier for mean updates.</summary>
    public double EtaMean { get; set; }
    /// <summary>Adaptive learning-rate multiplier for covariance updates.</summary>
    public double EtaSigma { get; set; }
}
