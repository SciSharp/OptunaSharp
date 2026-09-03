using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.CmaEs;

/// <summary>
///     Separable CMA-ES optimizer with an ask-and-tell interface. Port of the external <c>cmaes</c>
///     package's <c>SepCMA</c> class (<c>cmaes/_sepcma.py</c>, v0.13.0). The covariance is constrained
///     to be <b>diagonal</b> — stored as the 1-D vector <c>_C</c> — which drops the model complexity
///     (no eigendecomposition; <c>D = √C</c> elementwise) and raises the rank-µ learning rate, so it
///     outperforms full CMA-ES on separable functions.
///
///     <para><b>Bit-exact vs the Python <c>cmaes</c> package.</b> Same numeric discipline as
///     <see cref="Cma" />: <c>rng.randn</c> → NumSharp MT19937; the contiguous 1-D <c>np.sum(weights)</c>
///     → <see cref="MathUtil.PairwiseSum" />; but the constructor's Python <c>sum(...)</c> and the
///     strided <c>y_w</c>/<c>rank_mu</c> axis reductions accumulate <b>sequentially</b>; scalar
///     <c>**2</c> → multiply (array squaring), <c>y_w / D</c> is an elementwise divide. No LAPACK is
///     needed (the covariance is diagonal), so SepCMA sampling is byte-for-byte reproducible.</para>
/// </summary>
internal class SepCma : ICmaOptimizer
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
    private readonly double[] _weights;  // length mu

    private readonly double _sigma0;
    private double[] _pSigma;
    private double[] _pc;
    private double[] _mean;
    private double _sigma;
    private double[]? _d;
    private double[] _c;  // diagonal covariance (1-D)

    private readonly double[,]? _bounds;
    private readonly int _nMaxResampling;
    private int _g;
    private NumPyRandom _rng;

    private readonly double _tolx;
    private readonly double _tolxup = 1e4;
    private readonly double _tolfun = 1e-12;
    private readonly double _tolconditioncov = 1e14;
    private readonly int _funhistTerm;
    private readonly double[] _funhistValues;

    internal SepCma(
        double[] mean,
        double sigma,
        double[,]? bounds = null,
        int nMaxResampling = 100,
        int? seed = null,
        int? populationSize = null)
    {
        if (sigma <= 0) throw new ArgumentException("sigma must be non-zero positive value");
        foreach (double m in mean)
            if (!(Math.Abs(m) < MeanMax))
                throw new ArgumentException($"Abs of all elements of mean vector must be less than {MeanMax}");

        int nDim = mean.Length;
        if (nDim <= 1) throw new ArgumentException("The dimension of mean must be larger than 1");

        populationSize ??= 4 + (int)Math.Floor(3 * Math.Log(nDim));  // (eq. 48)
        if (populationSize <= 0) throw new ArgumentException("popsize must be non-zero positive value.");
        int popsize = populationSize.Value;

        int mu = popsize / 2;

        // (eq.49) weights_prime has length mu here (unlike CMA's popsize).
        var weightsPrime = new double[mu];
        for (int i = 0; i < mu; i++)
            weightsPrime[i] = Math.Log(mu + 1) - Math.Log(i + 1);

        // Python `sum(weights_prime)` / `sum(weights**2)` are the builtin sum → SEQUENTIAL.
        double wpSum = SeqSum(weightsPrime);
        var weights = new double[mu];
        for (int i = 0; i < mu; i++) weights[i] = weightsPrime[i] / wpSum;
        double sumWSq = 0;
        foreach (double w in weights) sumWSq += w * w;  // sum(weights**2), sequential
        double muEff = 1 / sumWSq;

        const double alphaCov = 2;
        double c1 = alphaCov / (Square(nDim + 1.3) + muEff);
        // cmu_full = 2/mu_eff/((n+√2)²) + (1 - 1/mu_eff) * min(1, (2·mu_eff-1)/((n+2)²+mu_eff))
        double cmuFull = 2 / muEff / Square(nDim + Math.Sqrt(2))
                         + (1 - 1 / muEff) * Math.Min(1, (2 * muEff - 1) / (Square(nDim + 2) + muEff));
        double cmu = (nDim + 2) / 3.0 * cmuFull;

        double cm = 1;  // (eq. 54)

        double cSigma = (muEff + 2) / (nDim + muEff + 3);
        double dSigma = 1 + 2 * Math.Max(0, Math.Sqrt((muEff - 1) / (nDim + 1)) - 1) + cSigma;
        double cc = 4.0 / (nDim + 4);

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

        _chiN = Math.Sqrt(_nDim) * (1.0 - 1.0 / (4.0 * _nDim) + 1.0 / (21.0 * (_nDim * _nDim)));
        _weights = weights;

        _pSigma = new double[nDim];
        _pc = new double[nDim];
        _mean = (double[])mean.Clone();
        _sigma = sigma;
        _sigma0 = sigma;
        _d = null;
        _c = Enumerable.Repeat(1.0, nDim).ToArray();  // np.ones(n_dim)

        _bounds = bounds;
        _nMaxResampling = nMaxResampling;
        _g = 0;
        _rng = np.random.RandomState();
        if (seed is not null) _rng.seed(seed.Value);

        _tolx = 1e-12 * sigma;
        _funhistTerm = 10 + (int)Math.Ceiling(30.0 * nDim / popsize);
        _funhistValues = new double[_funhistTerm * 2];
    }

    public int Dim => _nDim;
    public int PopulationSize => _popsize;
    public int Generation => _g;
    internal double[] Mean => _mean;
    internal double Sigma => _sigma;
    internal double[] Cov => _c;
    internal double[] PSigma => _pSigma;
    internal double[] Pc => _pc;

    public void ReseedRng(int seed) => _rng.seed((uint)seed);

    // ---- state serialization (own JSON; SepCMA's covariance is a 1-D vector — no symmetry trick) ---

    internal SepCmaState ToState() => new()
    {
        NDim = _nDim,
        Sigma0 = _sigma0,
        Bounds = ToJagged(_bounds),
        NMaxResampling = _nMaxResampling,
        Popsize = _popsize,
        G = _g,
        Mean = (double[])_mean.Clone(),
        Sigma = _sigma,
        C = (double[])_c.Clone(),
        PSigma = (double[])_pSigma.Clone(),
        Pc = (double[])_pc.Clone(),
        FunhistValues = (double[])_funhistValues.Clone(),
    };

    internal static SepCma FromState(SepCmaState s)
    {
        var cma = new SepCma(new double[s.NDim], s.Sigma0, FromJagged(s.Bounds), s.NMaxResampling,
            seed: null, populationSize: s.Popsize);
        cma._g = s.G;
        cma._mean = (double[])s.Mean.Clone();
        cma._sigma = s.Sigma;
        cma._c = (double[])s.C.Clone();
        cma._pSigma = (double[])s.PSigma.Clone();
        cma._pc = (double[])s.Pc.Clone();
        Array.Copy(s.FunhistValues, cma._funhistValues, cma._funhistValues.Length);
        cma._d = null;
        return cma;
    }

    public double[] Ask()
    {
        for (int i = 0; i < _nMaxResampling; i++)
        {
            double[] x = SampleSolution();
            if (IsFeasible(x))
                return x;
        }
        return RepairInfeasibleParams(SampleSolution());
    }

    private double[] EigenDecomposition()
    {
        if (_d is not null) return _d;
        var d = new double[_nDim];
        for (int i = 0; i < _nDim; i++) d[i] = Math.Sqrt(_c[i] < 0 ? Eps : _c[i]);
        _d = d;
        return d;
    }

    private double[] SampleSolution()
    {
        double[] d = EigenDecomposition();
        double[] z = _rng.randn(_nDim).ToArray<double>();  // ~ N(0, I)
        var x = new double[_nDim];
        for (int i = 0; i < _nDim; i++)
            x[i] = _mean[i] + _sigma * (d[i] * z[i]);  // y = D * z ; x = m + σ y
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
        var sorted = solutions.OrderBy(s => s.Value).ToList();

        int funhistIdx = 2 * (Generation % _funhistTerm);
        _funhistValues[funhistIdx] = sorted[0].Value;
        _funhistValues[funhistIdx + 1] = sorted[^1].Value;

        double[] d = EigenDecomposition();
        _d = null;

        // y_k = (x_k - m) / σ.
        var yk = new double[_popsize, _nDim];
        for (int k = 0; k < _popsize; k++)
            for (int i = 0; i < _nDim; i++)
                yk[k, i] = (sorted[k].X[i] - _mean[i]) / _sigma;

        // y_w = np.sum(y_k[:μ].T * weights[:μ], axis=1) — F-contiguous product → strided → sequential.
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

        // Step-size control: p_sigma = (1-cσ)·p_sigma + √(cσ(2-cσ)µeff)·(y_w / D).
        double coef = Math.Sqrt(_cSigma * (2 - _cSigma) * _muEff);
        for (int i = 0; i < _nDim; i++)
            _pSigma[i] = (1 - _cSigma) * _pSigma[i] + coef * (yw[i] / d[i]);

        double normPSigma = LinAlg.Norm(_pSigma);
        _sigma *= Math.Exp((_cSigma / _dSigma) * (normPSigma / _chiN - 1));
        _sigma = Math.Min(_sigma, SigmaMax);

        double hSigmaCondLeft = normPSigma / Math.Sqrt(1 - Math.Pow(1 - _cSigma, 2 * (_g + 1)));
        double hSigmaCondRight = (1.4 + 2.0 / (_nDim + 1)) * _chiN;
        double hSigma = hSigmaCondLeft < hSigmaCondRight ? 1.0 : 0.0;

        // (eq.45)
        double pcCoef = hSigma * Math.Sqrt(_cc * (2 - _cc) * _muEff);
        for (int i = 0; i < _nDim; i++)
            _pc[i] = (1 - _cc) * _pc[i] + pcCoef * yw[i];

        double deltaHSigma = (1 - hSigma) * _cc * (2 - _cc);

        // (eq.47) rank_one = pc**2 (elementwise); rank_mu = Σ_{k<μ} w[k]·y_k[k]²  (axis=0 → sequential).
        var rankOne = new double[_nDim];
        for (int i = 0; i < _nDim; i++) rankOne[i] = _pc[i] * _pc[i];
        var rankMu = new double[_nDim];
        for (int i = 0; i < _nDim; i++)
        {
            double s = 0;
            for (int k = 0; k < _mu; k++)  // zip(weights, y_k) stops at μ
                s += _weights[k] * (yk[k, i] * yk[k, i]);
            rankMu[i] = s;
        }

        double sumWeights = MathUtil.PairwiseSum(_weights);  // np.sum(weights) — 1-D contiguous → pairwise
        double cCoef = 1 + _c1 * deltaHSigma - _c1 - _cmu * sumWeights;
        for (int i = 0; i < _nDim; i++)
            _c[i] = cCoef * _c[i] + _c1 * rankOne[i] + _cmu * rankMu[i];
    }

    internal bool ShouldStop()
    {
        double[] d = EigenDecomposition();

        if (Generation > _funhistTerm && Max(_funhistValues) - Min(_funhistValues) < _tolfun)
            return true;

        bool allTolx = true;
        for (int i = 0; i < _nDim; i++)
            if (!(_sigma * _c[i] < _tolx) || !(_sigma * _pc[i] < _tolx)) { allTolx = false; break; }
        if (allTolx) return true;

        if (_sigma * Max(d) > _tolxup) return true;

        for (int i = 0; i < _nDim; i++)
            if (_mean[i] == _mean[i] + 0.2 * _sigma * Math.Sqrt(_c[i]))
                return true;

        int axis = Generation % Dim;
        bool noEffectAxis = true;
        for (int i = 0; i < _nDim; i++)
            if (_mean[i] != _mean[i] + 0.1 * _sigma * d[axis]) { noEffectAxis = false; break; }
        if (noEffectAxis) return true;

        double conditionCov = Max(d) / Min(d);
        if (conditionCov > _tolconditioncov) return true;

        return false;
    }

    // ---- helpers ------------------------------------------------------------------------------

    private static double SeqSum(double[] a) { double s = 0; foreach (double v in a) s += v; return s; }
    private static double Square(double x) => x * x;
    private static double Max(double[] a) { double m = a[0]; foreach (double v in a) if (v > m) m = v; return m; }
    private static double Min(double[] a) { double m = a[0]; foreach (double v in a) if (v < m) m = v; return m; }

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

/// <summary>Serialized evolving state of a <see cref="SepCma" /> optimizer (own JSON; see <see cref="CmaState" />).</summary>
internal sealed class SepCmaState
{
    public int NDim { get; set; }
    public double Sigma0 { get; set; }
    public double[][]? Bounds { get; set; }
    public int NMaxResampling { get; set; }
    public int Popsize { get; set; }
    public int G { get; set; }
    public double[] Mean { get; set; } = System.Array.Empty<double>();
    public double Sigma { get; set; }
    public double[] C { get; set; } = System.Array.Empty<double>();
    public double[] PSigma { get; set; } = System.Array.Empty<double>();
    public double[] Pc { get; set; } = System.Array.Empty<double>();
    public double[] FunhistValues { get; set; } = System.Array.Empty<double>();
}
