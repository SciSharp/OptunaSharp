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
///     → <see cref="MathUtil.PairwiseSum(double[])" />; but the constructor's Python <c>sum(...)</c> and the
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

    // Not readonly: FromState sets it after the constructor, so the constructor's "invalid bounds" check never tests the
    // placeholder mean against a stored box (as in Cma).
    private double[,]? _bounds;
    private readonly int _nMaxResampling;
    private int _g;
    private NumPyRandom _rng;

    private readonly double _tolx;
    private readonly double _tolxup = 1e4;
    private readonly double _tolfun = 1e-12;
    private readonly double _tolconditioncov = 1e14;
    private readonly int _funhistTerm;
    private readonly double[] _funhistValues;

    // Python's type of the scalars whose type cmaes' arithmetic decides (numpy.float64 or float), tracked only so a
    // pickled state is byte-identical to the one cmaes pickles (see Cma for the full story).
    private readonly bool _tolxIsNumpy;
    private bool _sigmaIsNumpy;

    /// <summary>Initializes a Gaussian proposal with diagonal covariance and separable adaptation.</summary>
    /// <param name="mean">Initial or current proposal center in transformed coordinates.</param>
    /// <param name="sigma">Initial positive global proposal standard deviation.</param>
    /// <param name="bounds">Optional box with one row per decision coordinate and lower/upper bound columns.</param>
    /// <param name="nMaxResampling">Maximum rejected box-infeasible proposals before coordinate repair.</param>
    /// <param name="seed">Seed used to reset the optimizer random stream.</param>
    /// <param name="populationSize">Number of individuals retained or evaluated per generation.</param>
    /// <param name="sigmaIsNumpyScalar">
    ///     Whether the Python caller would pass <paramref name="sigma" /> as a <c>numpy.float64</c>; it only decides
    ///     the type of the pickled <c>_tolx</c> (and of the untold <c>_sigma</c>). See <see cref="Cma" />.
    /// </param>
    /// <exception cref="System.ArgumentException">The scale is nonpositive, mean dimension is below two or exceeds the supported magnitude, population size is nonpositive, a learning rate fails cmaes' assertions, or the bounds are malformed or exclude the mean ("invalid bounds").</exception>
    /// <exception cref="System.DivideByZeroException">The population size is 1, so <c>mu</c> is 0 and cmaes' <c>1 / sum(weights**2)</c> divides the int 0 (Python's <c>ZeroDivisionError</c>).</exception>
    internal SepCma(
        double[] mean,
        double sigma,
        double[,]? bounds = null,
        int nMaxResampling = 100,
        int? seed = null,
        int? populationSize = null,
        bool sigmaIsNumpyScalar = false)
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
        // popsize 1 gives mu = 0: the builtin `sum` of an empty array is the int 0, so `1 / sum(weights**2)` is a Python
        // int division and raises ZeroDivisionError, which fails Optuna's first separable CMA-ES trial (QA-80). A double
        // division here would give +inf and sample on. DivideByZeroException is this port's ZeroDivisionError.
        if (mu == 0) throw new DivideByZeroException("division by zero");
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
        // cmaes' assertions, negated so that NaN fails them as in Python (QA-80).
        if (!(cSigma < 1)) throw new ArgumentException("invalid learning rate for cumulation for the step-size control");
        double cc = 4.0 / (nDim + 4);
        if (!(cc <= 1)) throw new ArgumentException("invalid learning rate for cumulation for the rank-one update");

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

        // cmaes' `assert bounds is None or _is_valid_bounds(bounds, mean), "invalid bounds"` (QA-80): an x0 or a
        // warm-start mean outside the search space fails Optuna's first CMA-ES trial.
        if (!IsValidBounds(bounds, mean)) throw new ArgumentException("invalid bounds");
        _bounds = bounds;
        _nMaxResampling = nMaxResampling;
        _g = 0;
        _rng = np.random.RandomState();
        if (seed is not null) _rng.seed(seed.Value);

        _tolx = 1e-12 * sigma;
        _funhistTerm = 10 + (int)Math.Ceiling(30.0 * nDim / popsize);
        _funhistValues = new double[_funhistTerm * 2];

        // `self._sigma = sigma` and `self._tolx = 1e-12 * sigma` take sigma's type.
        _tolxIsNumpy = sigmaIsNumpyScalar;
        _sigmaIsNumpy = sigmaIsNumpyScalar;
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
    internal double[] Cov => _c;
    /// <summary>Evolution path used to adapt the global step size.</summary>
    internal double[] PSigma => _pSigma;
    /// <summary>Evolution path used by the rank-one covariance update.</summary>
    internal double[] Pc => _pc;

    /// <summary>Resets the optimizer random stream for subsequent proposals.</summary>
    /// <param name="seed">Seed used to reset the optimizer random stream.</param>
    public void ReseedRng(int seed) => _rng.seed((uint)seed);

    // ---- state serialization -------------------------------------------------------------------
    // ToPyInstance/FromPyInstance map this class to cmaes' SepCMA state, which CmaPickle writes and reads as
    // Python's exact pickle. ToState/FromState are the JSON form stored before 2026-10-29, still read so older
    // studies keep restoring. SepCMA's covariance is a 1-D vector, so neither form needs a symmetry trick.

    /// <summary>Captures constructor settings and evolving optimizer arrays in a serializable state object.</summary>
    /// <returns>A serializable snapshot of optimizer settings and evolving state.</returns>
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

    /// <summary>Recreates optimizer constants and restores evolving state from a serialized snapshot.</summary>
    /// <param name="s">Serialized optimizer settings and evolving proposal state.</param>
    /// <returns>An optimizer restored from the supplied snapshot.</returns>
    /// <exception cref="System.ArgumentException">Stored constructor settings or population dimensions are rejected when the optimizer is recreated.</exception>
    internal static SepCma FromState(SepCmaState s)
    {
        // The JSON never recorded Python types; the sampler's default sigma0 (np.min) is numpy (see Cma.FromState).
        // The zero mean is a placeholder overwritten below, so the stored box goes in after the constructor, untested
        // against it, as an unpickled cmaes optimizer runs none of the constructor's assertions (see Cma.FromState).
        var cma = new SepCma(new double[s.NDim], s.Sigma0, bounds: null, s.NMaxResampling,
            seed: null, populationSize: s.Popsize, sigmaIsNumpyScalar: true);
        cma._bounds = FromJagged(s.Bounds);
        cma._g = s.G;
        cma._mean = (double[])s.Mean.Clone();
        cma._sigma = s.Sigma;
        cma._c = (double[])s.C.Clone();
        cma._pSigma = (double[])s.PSigma.Clone();
        cma._pc = (double[])s.Pc.Clone();
        Array.Copy(s.FunhistValues, cma._funhistValues, cma._funhistValues.Length);
        cma._d = null;
        // A stored state has been told: sigma is numpy unless the tell's clamp replaced it by the float bound.
        if (s.G > 0)
            cma._sigmaIsNumpy = s.Sigma != SigmaMax;
        return cma;
    }

    /// <summary>
    ///     Describes this optimizer as the state cmaes' <c>SepCMA.__getstate__</c> returns, attribute for attribute,
    ///     with Python's types and numpy's dtype sharing, so <see cref="PythonPickle.Dumps" /> of it is the pickle cmaes
    ///     writes for the same optimizer.
    /// </summary>
    /// <param name="layout">The pickle's dtype objects and whether this optimizer was restored before its last tell.</param>
    /// <returns>The <c>cmaes._sepcma.SepCMA</c> instance description; its arrays are copies.</returns>
    /// <remarks>
    ///     The key order is <c>__init__</c>'s in both a fresh and a restored optimizer (SepCMA's <c>__setstate__</c>
    ///     keeps the stored order). <c>_cc</c> is a float here (<c>4 / (n_dim + 4)</c>), unlike CMA's.
    /// </remarks>
    internal PyInstance ToPyInstance(CmaPickleLayout layout)
    {
        var state = new List<KeyValuePair<string, object?>>();
        void Add(string key, object? value) => state.Add(new(key, value));

        // Rebuilt by every tell: _p_sigma, _pc and _C; the rest is updated in place or never touched.
        Add("_n_dim", (long)_nDim);
        Add("_popsize", (long)_popsize);
        Add("_mu", (long)_mu);
        Add("_mu_eff", layout.Scalar(_muEff));
        Add("_cc", _cc);
        Add("_c1", layout.Scalar(_c1));
        Add("_cmu", layout.Scalar(_cmu));
        Add("_c_sigma", layout.Scalar(_cSigma));
        Add("_d_sigma", layout.Scalar(_dSigma));
        Add("_cm", CmaPickle.PyNumber(_cm));
        Add("_chi_n", _chiN);
        Add("_weights", layout.Vector(_weights, rebuiltByTell: false));
        Add("_p_sigma", layout.Vector(_pSigma, rebuiltByTell: true));
        Add("_pc", layout.Vector(_pc, rebuiltByTell: true));
        Add("_mean", layout.Vector(_mean, rebuiltByTell: false));
        Add("_sigma", layout.Float(_sigma, _sigmaIsNumpy));
        // Tell clears the sqrt(C) cache, so a stored state has None here.
        Add("_D", _d is null ? null : layout.Vector(_d, rebuiltByTell: true));
        Add("_C", layout.Vector(_c, rebuiltByTell: true));
        Add("_bounds", _bounds is null ? null : layout.Matrix(_bounds, rebuiltByTell: false));
        Add("_n_max_resampling", (long)_nMaxResampling);
        Add("_g", (long)_g);
        Add("_tolx", layout.Float(_tolx, _tolxIsNumpy));
        Add("_tolxup", _tolxup);
        Add("_tolfun", _tolfun);
        Add("_tolconditioncov", _tolconditioncov);
        Add("_funhist_term", (long)_funhistTerm);
        Add("_funhist_values", layout.Vector(_funhistValues, rebuiltByTell: false));
        return new PyInstance("cmaes._sepcma", "SepCMA", state);
    }

    /// <summary>
    ///     Restores an optimizer from a pickled cmaes <c>SepCMA</c> state, taking every value from the state, the
    ///     configuration included, so a state Python wrote continues exactly as Python would continue it.
    /// </summary>
    /// <param name="state">The unpickled <c>cmaes._sepcma.SepCMA</c> instance.</param>
    /// <returns>The restored optimizer; its random stream is unseeded, as after Python's <c>__setstate__</c>.</returns>
    /// <exception cref="ValueError">
    ///     The state lacks an attribute of cmaes' <c>SepCMA</c>, or an attribute has the wrong type or shape.
    /// </exception>
    internal static SepCma FromPyInstance(PyInstance state) => new(state);

    /// <summary>The constructor behind <see cref="FromPyInstance" />: assigns every field from the pickled state.</summary>
    /// <param name="s">The unpickled <c>cmaes._sepcma.SepCMA</c> instance.</param>
    /// <exception cref="ValueError">An attribute is missing, mistyped or misshapen.</exception>
    private SepCma(PyInstance s)
    {
        _nDim = CmaPickle.Int(s, "_n_dim");
        _popsize = CmaPickle.Int(s, "_popsize");
        if (_nDim <= 0 || _popsize <= 0)
            throw new ValueError($"The pickled cmaes SepCMA has n_dim {_nDim} and popsize {_popsize}; both must be positive.");
        _mu = CmaPickle.Int(s, "_mu");
        _muEff = CmaPickle.Float(s, "_mu_eff");
        _cc = CmaPickle.Float(s, "_cc");
        _c1 = CmaPickle.Float(s, "_c1");
        _cmu = CmaPickle.Float(s, "_cmu");
        _cSigma = CmaPickle.Float(s, "_c_sigma");
        _dSigma = CmaPickle.Float(s, "_d_sigma");
        _cm = CmaPickle.Float(s, "_cm");
        _chiN = CmaPickle.Float(s, "_chi_n");
        _weights = CmaPickle.Vector(s, "_weights", _mu);
        _pSigma = CmaPickle.Vector(s, "_p_sigma", _nDim);
        _pc = CmaPickle.Vector(s, "_pc", _nDim);
        _mean = CmaPickle.Vector(s, "_mean", _nDim);
        _sigma = CmaPickle.Float(s, "_sigma");
        _sigmaIsNumpy = CmaPickle.IsNumpy(s, "_sigma");
        _d = CmaPickle.OptionalVector(s, "_D", _nDim);
        _c = CmaPickle.Vector(s, "_C", _nDim);
        _bounds = CmaPickle.OptionalMatrix(s, "_bounds", _nDim, 2);
        _nMaxResampling = CmaPickle.Int(s, "_n_max_resampling");
        _g = CmaPickle.Int(s, "_g");
        _rng = np.random.RandomState();
        _tolx = CmaPickle.Float(s, "_tolx");
        _tolxIsNumpy = CmaPickle.IsNumpy(s, "_tolx");
        _tolxup = CmaPickle.Float(s, "_tolxup");
        _tolfun = CmaPickle.Float(s, "_tolfun");
        _tolconditioncov = CmaPickle.Float(s, "_tolconditioncov");
        _funhistTerm = CmaPickle.Int(s, "_funhist_term");
        _funhistValues = CmaPickle.Vector(s, "_funhist_values", 2 * _funhistTerm);
        // Only the legacy JSON form (ToState) reads _sigma0; `_tolx = 1e-12 * sigma0` is all a pickle keeps of it.
        _sigma0 = _tolx / 1e-12;
    }

    /// <summary>Draws one candidate from the current proposal without advancing the generation.</summary>
    /// <returns>One candidate vector in transformed coordinates.</returns>
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

    /// <summary>Refreshes the covariance factorization used to sample and whiten proposal vectors.</summary>
    /// <returns>Square roots of the diagonal covariance entries, cached for sampling and whitening.</returns>
    private double[] EigenDecomposition()
    {
        if (_d is not null) return _d;
        var d = new double[_nDim];
        for (int i = 0; i < _nDim; i++) d[i] = Math.Sqrt(_c[i] < 0 ? Eps : _c[i]);
        _d = d;
        return d;
    }

    /// <summary>Draws an unconstrained vector from the current Gaussian proposal.</summary>
    /// <returns>A proposal vector before box-feasibility repair.</returns>
    private double[] SampleSolution()
    {
        double[] d = EigenDecomposition();
        double[] z = NumPyDraws.Randn(_rng, _nDim);  // ~ N(0, I)
        var x = new double[_nDim];
        for (int i = 0; i < _nDim; i++)
            x[i] = _mean[i] + _sigma * (d[i] * z[i]);  // y = D * z ; x = m + σ y
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

    /// <summary>Updates the optimizer using evaluated candidates from one coherent generation.</summary>
    /// <param name="solutions">Evaluated proposal vectors and minimization-oriented values for one coherent generation.</param>
    /// <exception cref="System.ArgumentException">The solution count differs from population size or a coordinate exceeds the supported magnitude.</exception>
    /// <exception cref="NumSharp.ValueError">The vectors have different lengths, or a length NumPy cannot broadcast against the mean (<see cref="Cma.CheckTellShapes" />); the optimizer is left unchanged.</exception>
    public void Tell(List<(double[] X, double Value)> solutions)
    {
        if (solutions.Count != _popsize)
            throw new ArgumentException("Must tell popsize-length solutions.");
        foreach (var s in solutions)
            foreach (double v in s.X)
                if (!(Math.Abs(v) < MeanMax))
                    throw new ArgumentException(
                        $"Abs of all param values must be less than {MeanMax} to avoid overflow errors");
        // SepCMA.tell builds x_k as CMA.tell does, so NumPy's shape errors apply alike; checked before any state changes.
        solutions = Cma.CheckTellShapes(solutions, _nDim);

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
        // Numpy after `*= np.exp(...)`; Python's min() returns the float bound only when it is strictly smaller.
        _sigmaIsNumpy = !(SigmaMax < _sigma);
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

    /// <summary>Checks numerical and progress criteria that indicate the current optimizer has converged or stalled.</summary>
    /// <returns>True when a numerical or convergence stopping criterion is met.</returns>
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

    /// <summary>Accumulates values from left to right to preserve the reference reduction order.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>The left-to-right sum of the entries.</returns>
    private static double SeqSum(double[] a) { double s = 0; foreach (double v in a) s += v; return s; }

    /// <summary>
    ///     cmaes' <c>_is_valid_bounds(bounds, mean)</c> (shared by CMA and SepCMA): no bounds, or one <c>[low, high]</c>
    ///     row per coordinate with every coordinate of the mean inside; NaN fails, as Python's comparisons do.
    /// </summary>
    /// <param name="bounds">The optional box, one row per coordinate and two columns (low, high).</param>
    /// <param name="mean">The mean the box must contain.</param>
    /// <returns><c>true</c> when cmaes' assertion would hold.</returns>
    private static bool IsValidBounds(double[,]? bounds, double[] mean)
    {
        if (bounds is null)
            return true;
        if (bounds.GetLength(0) != mean.Length || bounds.GetLength(1) != 2)
            return false;
        for (int i = 0; i < mean.Length; i++)
            if (!(bounds[i, 0] <= mean[i] && mean[i] <= bounds[i, 1]))
                return false;
        return true;
    }
    /// <summary>Computes the scalar second power used by adaptation equations.</summary>
    /// <param name="x">Scalar operand used in the comparison or numerical transformation.</param>
    /// <returns>The product of the scalar with itself.</returns>
    private static double Square(double x) => x * x;
    /// <summary>Finds the largest vector entry using the reference reduction semantics.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>The largest vector entry.</returns>
    private static double Max(double[] a) { double m = a[0]; foreach (double v in a) if (v > m) m = v; return m; }
    /// <summary>Finds the smallest vector entry using the reference reduction semantics.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <returns>The smallest vector entry.</returns>
    private static double Min(double[] a) { double m = a[0]; foreach (double v in a) if (v < m) m = v; return m; }

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

/// <summary>Serialized evolving state of a <see cref="SepCma" /> optimizer (own JSON; see <see cref="CmaState" />).</summary>
internal sealed class SepCmaState
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
    /// <summary>Serialized number of completed optimizer updates.</summary>
    public int G { get; set; }
    /// <summary>Current Gaussian proposal center in transformed coordinates.</summary>
    public double[] Mean { get; set; } = System.Array.Empty<double>();
    /// <summary>Global Gaussian proposal step size.</summary>
    public double Sigma { get; set; }
    /// <summary>Serialized covariance representation used to restore the proposal geometry.</summary>
    public double[] C { get; set; } = System.Array.Empty<double>();
    /// <summary>Evolution path used to adapt the global step size.</summary>
    public double[] PSigma { get; set; } = System.Array.Empty<double>();
    /// <summary>Evolution path used by the rank-one covariance update.</summary>
    public double[] Pc { get; set; } = System.Array.Empty<double>();
    /// <summary>Recent objective-value history used by convergence checks.</summary>
    public double[] FunhistValues { get; set; } = System.Array.Empty<double>();
}
