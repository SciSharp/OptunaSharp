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

    public int Dim => _nDim;
    public int PopulationSize => _popsize;
    public int Generation => _g;
    internal double[] Mean => _mean;
    internal double Sigma => _sigma;
    internal double[,] Cov => _c;
    internal double[] PSigma => _pSigma;
    internal double[] Pc => _pc;
    internal double[] Weights => _weights;
    internal double MuEff => _muEff;
    internal double C1 => _c1;
    internal double Cmu => _cmu;

    public void ReseedRng(int seed) => _rng.seed((uint)seed);

    // ---- state serialization ------------------------------------------------------------------
    // Optuna pickles the optimizer into system_attrs; we serialize our OWN JSON instead (pickle is
    // a non-goal — [Misaligned], no cross-language mid-CMA resume; see CmaEsSampler). The readonly
    // configuration (weights, learning rates, tolerances, …) is a pure function of the constructor
    // inputs, so restore reruns the constructor and overwrites only the evolving state — no field
    // is re-derived by hand. The RNG is not persisted (Optuna drops it too); the sampler reseeds
    // the optimizer every trial before ask().

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

    internal static Cma FromState(CmaState s)
    {
        var cma = new Cma(new double[s.NDim], s.Sigma0, FromJagged(s.Bounds), s.NMaxResampling,
            seed: null, populationSize: s.Popsize, cov: null, lrAdapt: s.LrAdapt);
        cma.RestoreEvolvingState(s);
        return cma;
    }

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

    private double[] SampleSolution()
    {
        var (b, d) = EigenDecomposition();
        double[] z = _rng.randn(_nDim).ToArray<double>();  // ~ N(0, I)
        double[] y = LinAlg.Dot(LinAlg.Dot(b, Diag(d)), z);  // ~ N(0, C)
        var x = new double[_nDim];
        for (int i = 0; i < _nDim; i++)
            x[i] = _mean[i] + _sigma * y[i];  // ~ N(m, σ² C)
        return x;
    }

    private bool IsFeasible(double[] param)
    {
        if (_bounds is null) return true;
        for (int i = 0; i < _nDim; i++)
            if (!(param[i] >= _bounds[i, 0] && param[i] <= _bounds[i, 1]))
                return false;
        return true;
    }

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

    private static double PairwiseSum(double[] a) => MathUtil.PairwiseSum(a);
    private static double PairwiseSum(double[] a, int offset, int n) => MathUtil.PairwiseSum(a, offset, n);
    private static double Square(double x) => x * x;

    private static double[] Squared(double[] a, int offset, int n)
    {
        var r = new double[n];
        for (int i = 0; i < n; i++) r[i] = a[offset + i] * a[offset + i];
        return r;
    }

    private static double[,] Eye(int n)
    {
        var m = new double[n, n];
        for (int i = 0; i < n; i++) m[i, i] = 1.0;
        return m;
    }

    private static double[,] Diag(double[] v)
    {
        int n = v.Length;
        var m = new double[n, n];
        for (int i = 0; i < n; i++) m[i, i] = v[i];
        return m;
    }

    private static double[] DiagOf(double[,] m)
    {
        int n = m.GetLength(0);
        var v = new double[n];
        for (int i = 0; i < n; i++) v[i] = m[i, i];
        return v;
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
    private static double[,] DivideScalar(double[,] a, double s)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        var r = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                r[i, j] = a[i, j] / s;
        return r;
    }

    private static double NormCol(double[,] col)
    {
        int n = col.GetLength(0);
        var v = new double[n];
        for (int i = 0; i < n; i++) v[i] = col[i, 0];
        return LinAlg.Norm(v);
    }

    private static double Clip(double v, double lo, double hi) => Math.Min(Math.Max(v, lo), hi);
    private static double Max(double[] a) { double m = a[0]; foreach (double v in a) if (v > m) m = v; return m; }
    private static double Min(double[] a) { double m = a[0]; foreach (double v in a) if (v < m) m = v; return m; }

    /// <summary>
    ///     Return a copy of the symmetric matrix <paramref name="m" /> with both triangles taken
    ///     from its UPPER triangle (<c>r[i,j] = r[j,i] = m[i,j]</c> for <c>i ≤ j</c>) — the exact
    ///     effect of Optuna's <c>_compress_symmetric</c>/<c>_decompress_symmetric</c> pickle pair.
    /// </summary>
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
    public int NDim { get; set; }
    public double Sigma0 { get; set; }
    public double[][]? Bounds { get; set; }
    public int NMaxResampling { get; set; }
    public int Popsize { get; set; }
    public bool LrAdapt { get; set; }
    public int G { get; set; }
    public double[] Mean { get; set; } = System.Array.Empty<double>();
    public double Sigma { get; set; }
    public double[][] C { get; set; } = System.Array.Empty<double[]>();
    public double[] PSigma { get; set; } = System.Array.Empty<double>();
    public double[] Pc { get; set; } = System.Array.Empty<double>();
    public double[] FunhistValues { get; set; } = System.Array.Empty<double>();
    public double[][] EMean { get; set; } = System.Array.Empty<double[]>();
    public double[][] ESigma { get; set; } = System.Array.Empty<double[]>();
    public double VMean { get; set; }
    public double VSigma { get; set; }
    public double EtaMean { get; set; }
    public double EtaSigma { get; set; }
}
