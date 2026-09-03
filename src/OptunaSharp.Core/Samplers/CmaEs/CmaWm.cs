using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.CmaEs;

/// <summary>
///     CMA-ES with Margin (CMAwM) — port of the external <c>cmaes</c> package's <c>CMAwM</c> class
///     (<c>cmaes/_cmawm.py</c>, v0.13.0), the backend Optuna's <c>CmaEsSampler(with_margin=True)</c>
///     drives. It wraps a full-covariance <see cref="Cma" /> and, on every <see cref="Tell" />, applies
///     a <b>margin correction</b> that lower-bounds the marginal probability of each discrete
///     dimension so its samples never collapse onto a single grid point (Hamano et al., GECCO 2022).
///     <c>ask()</c> returns two vectors: an <em>encoded</em> x (discrete dims snapped to the grid) used
///     for evaluation, and the <em>raw</em> x used to update the distribution.
///
///     <para><b>[Misaligned] tolerance-bounded (erf/chi²), not bit-exact.</b> Unlike <see cref="Cma" />
///     /<see cref="SepCma" />, the margin correction rides SciPy's <c>norm.cdf</c> and
///     <c>chi2.ppf(df=1)</c>. OptunaSharp reproduces them managed: <c>norm.cdf</c> →
///     <see cref="Erf.NormCdf" /> (FreeBSD <c>s_erf</c>, ~1-3 ULP from SciPy/ucrt — the documented
///     erf class), and <c>chi2.ppf(q, df=1) = Φ⁻¹((1+q)/2)²</c> via <see cref="TruncNorm.NdtriExp" />
///     (the closed-form df=1 identity; verified to agree with SciPy's <c>chi2.ppf</c> end-to-end at
///     machine epsilon — the large relative error of the identity near <c>q→1</c> cancels because
///     those <c>χ</c> values only ever appear inside denominators/ratios). So a seeded CMAwM study
///     stays in exact RNG lock-step with Python (the raw draws come from the bit-exact <see cref="Cma" />)
///     but its post-correction mean and encoded values agree only to a few ULP. Everything before the
///     first correction (generation 0) is bit-exact.</para>
/// </summary>
internal sealed class CmaWm : ICmaOptimizer
{
    private readonly Cma _cma;
    private readonly int _nMaxResampling;

    // Discrete / continuous decomposition (a pure function of bounds+steps; rebuilt on restore).
    private double[] _steps = Array.Empty<double>();
    private int[] _discreteIdx = Array.Empty<int>();
    private int[] _continuousIdx = Array.Empty<int>();
    private double[,] _continuousSpace = new double[0, 2];   // bounds[continuous_idx]  (n_cont × 2)
    private int _nZdim;
    private double[,] _zSpace = new double[0, 0];            // (n_zdim × max_discrete) grid, NaN-filled → nanmax
    private double[,] _zLim = new double[0, 0];              // (n_zdim × max_discrete-1) midpoints, NaN-filled → nanmax

    // Evolving state (persisted verbatim across the optimizer round-trip).
    private double _margin;
    private double[] _a = Array.Empty<double>();             // the affine scaling `_A` (length n_dim)
    private double[] _mZLimLow = Array.Empty<double>();
    private double[] _mZLimUp = Array.Empty<double>();

    /// <summary>Public constructor — mirrors <c>CMAwM.__init__</c>.</summary>
    internal CmaWm(
        double[] mean,
        double sigma,
        double[,] bounds,
        double[] steps,
        int nMaxResampling = 100,
        int? seed = null,
        int? populationSize = null,
        double[,]? cov = null,
        double? margin = null)
    {
        // initialize `CMA` (no learning-rate adaptation — CMAwM is incompatible with lr_adapt).
        _cma = new Cma(mean, sigma, bounds, nMaxResampling, seed, populationSize, cov);
        int nDim = _cma.Dim;
        int populationSizeValue = _cma.PopulationSize;
        _nMaxResampling = nMaxResampling;

        if (bounds.GetLength(0) != steps.Length)
            throw new ArgumentException("bounds and steps must be the same length");
        foreach (double s in steps)
            if (double.IsNaN(s)) throw new ArgumentException("steps should not include NaN");

        BuildDiscreteDecomposition(bounds, steps, _cma.Mean);

        if (!IsValidBounds(_continuousSpace, Gather(mean, _continuousIdx)))
            throw new ArgumentException("invalid bounds");

        if (_nZdim == 0)
        {
            _a = Enumerable.Repeat(1.0, nDim).ToArray();  // harmless; unused when there are no discrete dims
            return;
        }

        _margin = margin ?? 1.0 / (nDim * populationSizeValue);
        if (!(_margin > 0)) throw new ArgumentException("margin must be non-zero positive value.");
        _a = Enumerable.Repeat(1.0, nDim).ToArray();  // np.full(n_dim, 1.0)
    }

    /// <summary>Private constructor for state restore: adopt a pre-built inner <see cref="Cma" /> and
    /// rebuild the (bounds, steps)-derived decomposition. <see cref="FromState" /> then overwrites the
    /// evolving <c>_A</c>/<c>m_z_lim</c>/<c>margin</c>.</summary>
    private CmaWm(Cma cma, double[] steps, int nMaxResampling)
    {
        _cma = cma;
        _nMaxResampling = nMaxResampling;
        BuildDiscreteDecomposition(_cma.Bounds!, steps, _cma.Mean);
        _a = Enumerable.Repeat(1.0, cma.Dim).ToArray();
    }

    public int Dim => _cma.Dim;
    public int PopulationSize => _cma.PopulationSize;
    public int Generation => _cma.Generation;

    // Test-surface accessors (the state parity tests inspect the decomposition + evolving fields).
    internal Cma Inner => _cma;
    internal double[] AffineScaling => _a;
    internal int[] DiscreteIdx => _discreteIdx;
    internal int[] ContinuousIdx => _continuousIdx;
    internal double[,] ZSpace => _zSpace;
    internal double[,] ZLim => _zLim;
    internal double[] MZLimLow => _mZLimLow;
    internal double[] MZLimUp => _mZLimUp;
    internal double Margin => _margin;
    internal int NZdim => _nZdim;

    public void ReseedRng(int seed) => _cma.ReseedRng(seed);

    internal bool ShouldStop() => _cma.ShouldStop();

    // ---- construction helpers -----------------------------------------------------------------

    /// <summary>
    ///     The <c>__init__</c> body that splits the space into discrete/continuous dimensions and
    ///     builds the per-discrete grid (<c>z_space</c>) and its midpoints (<c>z_lim</c>), NaN-padding
    ///     ragged rows and back-filling each with its row <c>nanmax</c>. Also seeds <c>m_z_lim_low/up</c>
    ///     from <paramref name="mean" /> exactly as the Python constructor does.
    /// </summary>
    private void BuildDiscreteDecomposition(double[,] bounds, double[] steps, double[] mean)
    {
        _steps = (double[])steps.Clone();
        var disc = new List<int>();
        var cont = new List<int>();
        for (int i = 0; i < steps.Length; i++)
        {
            if (steps[i] > 0) disc.Add(i);
            else cont.Add(i);   // np.where(steps <= 0)
        }
        _discreteIdx = disc.ToArray();
        _continuousIdx = cont.ToArray();

        // discrete_list[k] = np.arange(bounds[i,0], bounds[i,1] + steps[i]/2, steps[i]).
        var discreteList = new double[_discreteIdx.Length][];
        int maxDiscrete = 0;
        for (int k = 0; k < _discreteIdx.Length; k++)
        {
            int i = _discreteIdx[k];
            discreteList[k] = Arange(bounds[i, 0], bounds[i, 1] + steps[i] / 2, steps[i]);
            maxDiscrete = Math.Max(maxDiscrete, discreteList[k].Length);
        }

        // discrete_space = np.full((n_zdim, max_discrete), nan); fill each row's prefix.
        _zSpace = new double[_discreteIdx.Length, maxDiscrete];
        for (int k = 0; k < _discreteIdx.Length; k++)
            for (int j = 0; j < maxDiscrete; j++)
                _zSpace[k, j] = j < discreteList[k].Length ? discreteList[k][j] : double.NaN;

        // continuous_space = bounds[continuous_idx]  (n_cont × 2)
        _continuousSpace = new double[_continuousIdx.Length, 2];
        for (int k = 0; k < _continuousIdx.Length; k++)
        {
            _continuousSpace[k, 0] = bounds[_continuousIdx[k], 0];
            _continuousSpace[k, 1] = bounds[_continuousIdx[k], 1];
        }

        _nZdim = _discreteIdx.Length;   // == len(discrete_space)
        if (_nZdim == 0)
        {
            _zLim = new double[0, 0];
            _mZLimLow = Array.Empty<double>();
            _mZLimUp = Array.Empty<double>();
            return;
        }

        // z_lim = (z_space[:, 1:] + z_space[:, :-1]) / 2   (n_zdim × max_discrete-1)
        int limCols = Math.Max(0, maxDiscrete - 1);
        _zLim = new double[_nZdim, limCols];
        for (int k = 0; k < _nZdim; k++)
            for (int j = 0; j < limCols; j++)
                _zLim[k, j] = (_zSpace[k, j + 1] + _zSpace[k, j]) / 2;

        // Back-fill NaNs in each z_space / z_lim row with that row's nanmax.
        for (int k = 0; k < _nZdim; k++)
        {
            double zsMax = RowNanMax(_zSpace, k, maxDiscrete);
            for (int j = 0; j < maxDiscrete; j++)
                if (double.IsNaN(_zSpace[k, j])) _zSpace[k, j] = zsMax;
            double zlMax = RowNanMax(_zLim, k, limCols);
            for (int j = 0; j < limCols; j++)
                if (double.IsNaN(_zLim[k, j])) _zLim[k, j] = zlMax;
        }

        // m_z = mean[discrete_idx]; m_pos = searchsorted(z_lim[i], m_z[i]); clip to indices.
        _mZLimLow = new double[_nZdim];
        _mZLimUp = new double[_nZdim];
        for (int k = 0; k < _nZdim; k++)
        {
            double mz = mean[_discreteIdx[k]];
            int mPos = SearchSortedLeft(_zLim, k, limCols, mz);
            int lowIdx = Clip(mPos - 1, 0, limCols - 1);
            int upIdx = Clip(mPos, 0, limCols - 1);
            _mZLimLow[k] = _zLim[k, lowIdx];
            _mZLimUp[k] = _zLim[k, upIdx];
        }
    }

    // ---- ask ----------------------------------------------------------------------------------

    double[] ICmaOptimizer.Ask() =>
        throw new NotSupportedException("CMAwM has a two-vector ask(); call AskEncoded() instead.");

    /// <summary>
    ///     Python <c>CMAwM.ask()</c> — resample raw draws until the continuous dims are feasible (or
    ///     fall back to a clipped last draw), then return <c>(x_encoded, x_raw)</c>: the encoded x has
    ///     its discrete dims snapped to the grid (used for evaluation), the raw x updates the
    ///     distribution (stored as <c>x_for_tell</c> and fed back to <see cref="Tell" />).
    /// </summary>
    internal (double[] Encoded, double[] ForTell) AskEncoded()
    {
        for (int i = 0; i < _nMaxResampling; i++)
        {
            double[] x = _cma.SampleSolution();
            if (IsContinuousFeasible(x))
                return (EncodeIfNeeded(x), x);
        }
        double[] last = _cma.SampleSolution();
        RepairContinuousInPlace(last);
        return (EncodeIfNeeded(last), last);
    }

    /// <summary><c>x_encoded = x.copy(); x_encoded[discrete_idx] = _encode_discrete_params(x[discrete_idx])</c>.</summary>
    private double[] EncodeIfNeeded(double[] x)
    {
        var encoded = (double[])x.Clone();
        if (_nZdim > 0)
        {
            double[] enc = EncodeDiscreteParams(x);
            for (int k = 0; k < _nZdim; k++)
                encoded[_discreteIdx[k]] = enc[k];
        }
        return encoded;
    }

    /// <summary>Python <c>_is_continuous_feasible</c> — every continuous dim within its bounds (empty ⇒ true).</summary>
    private bool IsContinuousFeasible(double[] x)
    {
        for (int k = 0; k < _continuousIdx.Length; k++)
        {
            double v = x[_continuousIdx[k]];
            if (!(v >= _continuousSpace[k, 0] && v <= _continuousSpace[k, 1]))
                return false;
        }
        return true;
    }

    /// <summary>Python <c>_repair_continuous_params</c> — clip each continuous dim into its bounds, in place.</summary>
    private void RepairContinuousInPlace(double[] x)
    {
        for (int k = 0; k < _continuousIdx.Length; k++)
        {
            int i = _continuousIdx[k];
            double v = x[i];
            if (v < _continuousSpace[k, 0]) v = _continuousSpace[k, 0];
            if (v > _continuousSpace[k, 1]) v = _continuousSpace[k, 1];
            x[i] = v;
        }
    }

    /// <summary>
    ///     Python <c>_encode_discrete_params</c> — affine-scale the raw discrete values by <c>_A</c>
    ///     about the current mean, then snap each to its grid via <c>searchsorted(z_lim)</c> → <c>z_space</c>.
    /// </summary>
    private double[] EncodeDiscreteParams(double[] x)
    {
        double[] mean = _cma.Mean;
        int limCols = _zLim.GetLength(1);
        var enc = new double[_nZdim];
        for (int k = 0; k < _nZdim; k++)
        {
            int i = _discreteIdx[k];
            double scaled = (x[i] - mean[i]) * _a[i] + mean[i];
            int pos = SearchSortedLeft(_zLim, k, limCols, scaled);
            enc[k] = _zSpace[k, pos];
        }
        return enc;
    }

    // ---- tell (the margin correction) ---------------------------------------------------------

    public void Tell(List<(double[] X, double Value)> solutions)
    {
        _cma.Tell(solutions);
        double sigma = _cma.Sigma;
        double[,] c = _cma.Cov;
        if (_nZdim == 0) return;

        double[] mean = _cma.Mean;                          // reference — mutated in place below
        int limCols = _zLim.GetLength(1);

        // updated_m_integer = mean[discrete_idx]  (a snapshot copy — fancy indexing).
        var updated = Gather(mean, _discreteIdx);

        // Recompute m_z_lim_low/up from the just-updated mean.
        for (int k = 0; k < _nZdim; k++)
        {
            int mPos = SearchSortedLeft(_zLim, k, limCols, updated[k]);
            _mZLimLow[k] = _zLim[k, Clip(mPos - 1, 0, limCols - 1)];
            _mZLimUp[k] = _zLim[k, Clip(mPos, 0, limCols - 1)];
        }

        // z_scale = sigma * A[disc] * sqrt(diag(C)[disc]);  low/up/mid CDFs (norm.cdf → Erf.NormCdf).
        var lowCdf = new double[_nZdim];
        var upCdf = new double[_nZdim];
        var midCdf = new double[_nZdim];
        var cDiag = new double[_nZdim];   // diag(C)[disc]
        for (int k = 0; k < _nZdim; k++)
        {
            int i = _discreteIdx[k];
            cDiag[k] = c[i, i];
            double zScale = sigma * _a[i] * Math.Sqrt(cDiag[k]);
            lowCdf[k] = NormCdf(_mZLimLow[k], updated[k], zScale);
            upCdf[k] = 1.0 - NormCdf(_mZLimUp[k], updated[k], zScale);
            midCdf[k] = 1.0 - (lowCdf[k] + upCdf[k]);
        }

        // md tracks the discrete-mean under correction (edge, then side); written back at the end.
        var md = (double[])updated.Clone();

        // edge_mask = max(low, up) > 0.5 ; side_mask = max(low, up) <= 0.5.
        var edgeMask = new bool[_nZdim];
        var sideMask = new bool[_nZdim];
        bool anyEdge = false;
        for (int k = 0; k < _nZdim; k++)
        {
            double mx = Math.Max(lowCdf[k], upCdf[k]);
            edgeMask[k] = mx > 0.5;
            sideMask[k] = mx <= 0.5;
            anyEdge |= edgeMask[k];
        }

        if (anyEdge)
        {
            // dist = sigma * A[disc] * sqrt(chi2_ppf(1 - 2*margin) * diag(C)[disc]);  scalar chi2.
            double distChi = Chi2Ppf1(1.0 - 2.0 * _margin);
            for (int k = 0; k < _nZdim; k++)
            {
                int i = _discreteIdx[k];
                bool modifyMask = Math.Min(lowCdf[k], upCdf[k]) < _margin;
                double modifySign = Math.Sign(updated[k] - _mZLimUp[k]);   // reads pre-correction mean
                double dist = sigma * _a[i] * Math.Sqrt(distChi * cDiag[k]);
                double coeff = (modifyMask && edgeMask[k]) ? 1.0 : 0.0;
                md[k] = updated[k] + coeff * (_mZLimUp[k] + modifySign * dist - updated[k]);
            }
        }

        // correct probability: clamp low/up to >= margin/2, then the simultaneous-equation solve.
        double halfMargin = _margin / 2.0;
        var chiLow = new double[_nZdim];
        var chiUp = new double[_nZdim];
        var qLow = new double[_nZdim];
        var qUp = new double[_nZdim];
        for (int k = 0; k < _nZdim; k++)
        {
            double lowC = Math.Max(lowCdf[k], halfMargin);
            double upC = Math.Max(upCdf[k], halfMargin);
            double denom = lowC + midCdf[k] + upC - 3.0 * halfMargin;
            double slack = 1.0 - lowC - upC - midCdf[k];
            double modLow = lowC + slack * (lowC - halfMargin) / denom;
            double modUp = upC + slack * (upC - halfMargin) / denom;
            modLow = Clip(modLow, 1e-10, 0.5 - 1e-10);
            modUp = Clip(modUp, 1e-10, 0.5 - 1e-10);
            qLow[k] = 1.0 - 2.0 * modLow;
            qUp[k] = 1.0 - 2.0 * modUp;
        }
        // Batched chi2.ppf (the whole-array Newton iteration, matching numpy's vectorized frompyfunc).
        double[] chiLowSq = Chi2Ppf1(qLow);
        double[] chiUpSq = Chi2Ppf1(qUp);
        for (int k = 0; k < _nZdim; k++)
        {
            chiLow[k] = Math.Sqrt(chiLowSq[k]);
            chiUp[k] = Math.Sqrt(chiUpSq[k]);
        }

        // A[disc] += side_mask * ((up-low)/((chiLow+chiUp)*sigma*sqrt(diag(C)[disc])) - A[disc]).
        for (int k = 0; k < _nZdim; k++)
        {
            int i = _discreteIdx[k];
            double cDiagSq = Math.Sqrt(cDiag[k]);   // sqrt(diag(C))[disc]
            double aNew = (_mZLimUp[k] - _mZLimLow[k]) / ((chiLow[k] + chiUp[k]) * sigma * cDiagSq);
            if (sideMask[k]) _a[i] = _a[i] + (aNew - _a[i]);
        }

        // mean[disc] += side_mask * ((low*chiUp + up*chiLow)/(chiLow+chiUp) - mean[disc]).
        for (int k = 0; k < _nZdim; k++)
        {
            double target = (_mZLimLow[k] * chiUp[k] + _mZLimUp[k] * chiLow[k]) / (chiLow[k] + chiUp[k]);
            if (sideMask[k]) md[k] = md[k] + (target - md[k]);
            mean[_discreteIdx[k]] = md[k];   // write the corrected discrete mean back into the CMA
        }
    }

    // ---- state serialization (own JSON; the pickle analogue — see CmaEsSampler) ----------------

    internal CmaWmState ToState() => new()
    {
        Inner = _cma.ToState(),
        Steps = (double[])_steps.Clone(),
        Margin = _margin,
        A = (double[])_a.Clone(),
        MZLimLow = (double[])_mZLimLow.Clone(),
        MZLimUp = (double[])_mZLimUp.Clone(),
    };

    internal static CmaWm FromState(CmaWmState s)
    {
        var cma = Cma.FromState(s.Inner);
        var wm = new CmaWm(cma, s.Steps, s.Inner.NMaxResampling)
        {
            _margin = s.Margin,
        };
        wm._a = (double[])s.A.Clone();
        if (wm._nZdim > 0)
        {
            wm._mZLimLow = (double[])s.MZLimLow.Clone();
            wm._mZLimUp = (double[])s.MZLimUp.Clone();
        }
        return wm;
    }

    // ---- scipy-shaped scalar/array helpers (the erf/chi² tolerance surface) ---------------------

    /// <summary>SciPy <c>norm.cdf(x, loc, scale) = Φ((x-loc)/scale)</c> via <see cref="Erf.NormCdf" />.</summary>
    private static double NormCdf(double x, double loc, double scale) => Erf.NormCdf((x - loc) / scale);

    /// <summary>SciPy <c>chi2.ppf(q, df=1) = Φ⁻¹((1+q)/2)²</c> (scalar).</summary>
    private static double Chi2Ppf1(double q)
    {
        double p = (1.0 + q) / 2.0;
        double ndtri = TruncNorm.NdtriExp(new[] { Math.Log(p) })[0];
        return ndtri * ndtri;
    }

    /// <summary>SciPy <c>chi2.ppf(q, df=1)</c> element-wise — batched through the one whole-array
    /// <see cref="TruncNorm.NdtriExp" /> Newton iteration (matches numpy's vectorized evaluation).</summary>
    private static double[] Chi2Ppf1(double[] q)
    {
        var logs = new double[q.Length];
        for (int i = 0; i < q.Length; i++) logs[i] = Math.Log((1.0 + q[i]) / 2.0);
        double[] ndtri = TruncNorm.NdtriExp(logs);
        var r = new double[q.Length];
        for (int i = 0; i < q.Length; i++) r[i] = ndtri[i] * ndtri[i];
        return r;
    }

    // ---- small numpy-shaped helpers -----------------------------------------------------------

    /// <summary>numpy <c>np.arange(start, stop, step)</c> for floats — length <c>ceil((stop-start)/step)</c>,
    /// element <c>start + i*step</c> (verified bit-identical to numpy over the grids used).</summary>
    private static double[] Arange(double start, double stop, double step)
    {
        int n = (int)Math.Ceiling((stop - start) / step);
        if (n < 0) n = 0;
        var r = new double[n];
        for (int i = 0; i < n; i++) r[i] = start + i * step;
        return r;
    }

    /// <summary>numpy <c>np.searchsorted(a[row], v, side='left')</c> — leftmost index <c>i</c> with
    /// <c>a[i-1] &lt; v &lt;= a[i]</c> (the count of elements strictly less than <paramref name="v" />).</summary>
    private static int SearchSortedLeft(double[,] a, int row, int len, double v)
    {
        int i = 0;
        while (i < len && a[row, i] < v) i++;
        return i;
    }

    private static double RowNanMax(double[,] a, int row, int len)
    {
        double m = double.NegativeInfinity;
        for (int j = 0; j < len; j++)
        {
            double v = a[row, j];
            if (!double.IsNaN(v) && v > m) m = v;
        }
        return m;
    }

    private static double[] Gather(double[] a, int[] idx)
    {
        var r = new double[idx.Length];
        for (int k = 0; k < idx.Length; k++) r[k] = a[idx[k]];
        return r;
    }

    private static int Clip(int v, int lo, int hi) => Math.Min(Math.Max(v, lo), hi);
    private static double Clip(double v, double lo, double hi) => Math.Min(Math.Max(v, lo), hi);

    /// <summary>Python <c>_is_valid_bounds(bounds, mean)</c> — every mean within its [low, high] (empty ⇒ true).</summary>
    private static bool IsValidBounds(double[,] bounds, double[] mean)
    {
        if (mean.Length != bounds.GetLength(0)) return false;
        for (int i = 0; i < mean.Length; i++)
            if (!(bounds[i, 0] <= mean[i] && mean[i] <= bounds[i, 1])) return false;
        return true;
    }
}

/// <summary>
///     Serialized state of a <see cref="CmaWm" /> optimizer (own JSON; see <see cref="CmaState" />).
///     Stores the inner <see cref="Cma" />'s state plus the CMAwM-specific evolving fields; the
///     bounds+steps decomposition (grids, discrete/continuous split) is rebuilt from <see cref="Steps" />
///     and the inner state's bounds on restore.
/// </summary>
internal sealed class CmaWmState
{
    public CmaState Inner { get; set; } = new();
    public double[] Steps { get; set; } = System.Array.Empty<double>();
    public double Margin { get; set; }
    public double[] A { get; set; } = System.Array.Empty<double>();
    public double[] MZLimLow { get; set; } = System.Array.Empty<double>();
    public double[] MZLimUp { get; set; } = System.Array.Empty<double>();
}
