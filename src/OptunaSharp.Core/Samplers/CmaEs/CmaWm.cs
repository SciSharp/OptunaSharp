using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.CmaEs;

/// <summary>
///     CMA-ES with Margin (CMAwM) — port of the external <c>cmaes</c> package's <c>CMAwM</c> class
///     (<c>cmaes/_cmawm.py</c>, v0.13.0; its discrete-space description and pickled state follow v0.13.1,
///     see <see cref="ToPyInstance" />), the backend Optuna's <c>CmaEsSampler(with_margin=True)</c>
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
    // cmaes 0.13.1's description of each discrete dimension, which its pickle stores: the grid is
    // low + i·step for i < size (np.arange's values, with its actual step (low + step) - low).
    private double[] _discreteSpaceLow = Array.Empty<double>();
    private long[] _discreteSpaceSize = Array.Empty<long>();
    private double[] _discreteSpaceStep = Array.Empty<double>();
    private double[,] _zSpace = new double[0, 0];            // (n_zdim × max_discrete) grid, NaN-filled → nanmax
    private double[,] _zLim = new double[0, 0];              // (n_zdim × max_discrete-1) midpoints, NaN-filled → nanmax

    // Evolving state (persisted verbatim across the optimizer round-trip).
    private double _margin;
    private double[] _a = Array.Empty<double>();             // the affine scaling `_A` (length n_dim)
    private double[] _mZLimLow = Array.Empty<double>();
    private double[] _mZLimUp = Array.Empty<double>();

    /// <summary>Public constructor — mirrors <c>CMAwM.__init__</c>.</summary>
    /// <param name="mean">Initial or current proposal center in transformed coordinates.</param>
    /// <param name="sigma">Initial positive global proposal standard deviation.</param>
    /// <param name="bounds">Optional box with one row per decision coordinate and lower/upper bound columns.</param>
    /// <param name="steps">Quantization step per coordinate; zero identifies a continuous coordinate.</param>
    /// <param name="nMaxResampling">Maximum rejected box-infeasible proposals before coordinate repair.</param>
    /// <param name="seed">Seed used to reset the optimizer random stream.</param>
    /// <param name="populationSize">Number of individuals retained or evaluated per generation.</param>
    /// <param name="cov">Initial covariance matrix, or null for the identity default.</param>
    /// <param name="margin">Minimum probability retained near discrete choices; null selects the dimension-dependent default.</param>
    /// <param name="sigmaIsNumpyScalar">
    ///     Whether the Python caller would pass <paramref name="sigma" /> as a <c>numpy.float64</c>; passed to the inner
    ///     <see cref="Cma" />, where it only decides the types its pickle records.
    /// </param>
    /// <exception cref="System.ArgumentException">Bounds and steps have incompatible lengths, a step is NaN, bounds are invalid, a discrete parameter has fewer than two choices, margin is nonpositive, or the underlying CMA settings are invalid.</exception>
    internal CmaWm(
        double[] mean,
        double sigma,
        double[,] bounds,
        double[] steps,
        int nMaxResampling = 100,
        int? seed = null,
        int? populationSize = null,
        double[,]? cov = null,
        double? margin = null,
        bool sigmaIsNumpyScalar = false)
    {
        // initialize `CMA` (no learning-rate adaptation — CMAwM is incompatible with lr_adapt).
        _cma = new Cma(mean, sigma, bounds, nMaxResampling, seed, populationSize, cov, sigmaIsNumpyScalar: sigmaIsNumpyScalar);
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
        foreach (long size in _discreteSpaceSize)
            if (size < 2)
                throw new ArgumentException("each discrete parameter must have at least two choices");

        _margin = margin ?? 1.0 / (nDim * populationSizeValue);
        if (!(_margin > 0)) throw new ArgumentException("margin must be non-zero positive value.");
        _a = Enumerable.Repeat(1.0, nDim).ToArray();  // np.full(n_dim, 1.0)
    }

    /// <summary>Private constructor for state restore: adopt a pre-built inner <see cref="Cma" /> and
    /// rebuild the (bounds, steps)-derived decomposition. <see cref="FromState" /> then overwrites the
    /// evolving <c>_A</c>/<c>m_z_lim</c>/<c>margin</c>.</summary>
    /// <param name="cma">Existing full-covariance optimizer whose margin wrapper is restored.</param>
    /// <param name="steps">Quantization step per coordinate; zero identifies a continuous coordinate.</param>
    /// <param name="nMaxResampling">Maximum rejected box-infeasible proposals before coordinate repair.</param>
    private CmaWm(Cma cma, double[] steps, int nMaxResampling)
    {
        _cma = cma;
        _nMaxResampling = nMaxResampling;
        BuildDiscreteDecomposition(_cma.Bounds!, steps, _cma.Mean);
        _a = Enumerable.Repeat(1.0, cma.Dim).ToArray();
    }

    /// <summary>Number of numerical decision coordinates.</summary>
    public int Dim => _cma.Dim;
    /// <summary>Number of evaluated individuals required for one generation.</summary>
    public int PopulationSize => _cma.PopulationSize;
    /// <summary>Number of completed optimizer updates.</summary>
    public int Generation => _cma.Generation;

    // Test-surface accessors (the state parity tests inspect the decomposition + evolving fields).
    /// <summary>Underlying full-covariance CMA state used by the margin wrapper.</summary>
    internal Cma Inner => _cma;
    /// <summary>Per-coordinate affine scales applied by margin correction.</summary>
    internal double[] AffineScaling => _a;
    /// <summary>Indices of quantized decision coordinates.</summary>
    internal int[] DiscreteIdx => _discreteIdx;
    /// <summary>Indices of continuous decision coordinates.</summary>
    internal int[] ContinuousIdx => _continuousIdx;
    /// <summary>Discrete coordinate choices represented in the transformed space.</summary>
    internal double[,] ZSpace => _zSpace;
    /// <summary>Decision thresholds separating neighboring discrete choices.</summary>
    internal double[,] ZLim => _zLim;
    /// <summary>Nearest lower discrete threshold relative to the current mean.</summary>
    internal double[] MZLimLow => _mZLimLow;
    /// <summary>Nearest upper discrete threshold relative to the current mean.</summary>
    internal double[] MZLimUp => _mZLimUp;
    /// <summary>Minimum probability mass maintained near discrete coordinate boundaries.</summary>
    internal double Margin => _margin;
    /// <summary>Number of discrete coordinates requiring margin correction.</summary>
    internal int NZdim => _nZdim;

    /// <summary>Resets the optimizer random stream for subsequent proposals.</summary>
    /// <param name="seed">Seed used to reset the optimizer random stream.</param>
    public void ReseedRng(int seed) => _cma.ReseedRng(seed);

    /// <summary>Checks numerical and progress criteria that indicate the current optimizer has converged or stalled.</summary>
    /// <returns>True when a numerical or convergence stopping criterion is met.</returns>
    internal bool ShouldStop() => _cma.ShouldStop();

    // ---- construction helpers -----------------------------------------------------------------

    /// <summary>
    ///     The <c>__init__</c> body that splits the space into discrete/continuous dimensions and describes each
    ///     discrete one as cmaes 0.13.1 does (<c>_discrete_space_low/size/step</c>), then builds the grid from that
    ///     description (<see cref="BuildGrid" />).
    /// </summary>
    /// <param name="bounds">Optional box with one row per decision coordinate and lower/upper bound columns.</param>
    /// <param name="steps">Quantization step per coordinate; zero identifies a continuous coordinate.</param>
    /// <param name="mean">Initial or current proposal center in transformed coordinates.</param>
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

        // _discrete_space_low = bounds[idx, 0]; _discrete_space_size = ceil((high + step/2 - low) / step), the length
        // np.arange(low, high + step/2, step) has; _discrete_space_step = low + step - low, the step np.arange
        // actually takes. So the grid holds exactly the values cmaes <= 0.13.0 built with np.arange.
        _discreteSpaceLow = new double[_discreteIdx.Length];
        _discreteSpaceSize = new long[_discreteIdx.Length];
        _discreteSpaceStep = new double[_discreteIdx.Length];
        for (int k = 0; k < _discreteIdx.Length; k++)
        {
            int i = _discreteIdx[k];
            double low = bounds[i, 0];
            _discreteSpaceLow[k] = low;
            _discreteSpaceSize[k] = (long)Math.Ceiling((bounds[i, 1] + steps[i] / 2 - low) / steps[i]);
            _discreteSpaceStep[k] = low + steps[i] - low;
        }

        // continuous_space = bounds[continuous_idx]  (n_cont × 2)
        _continuousSpace = new double[_continuousIdx.Length, 2];
        for (int k = 0; k < _continuousIdx.Length; k++)
        {
            _continuousSpace[k, 0] = bounds[_continuousIdx[k], 0];
            _continuousSpace[k, 1] = bounds[_continuousIdx[k], 1];
        }

        BuildGrid(mean);
    }

    /// <summary>
    ///     Builds the per-discrete grid (<c>z_space</c>, row <c>k</c> = <c>low + i·step</c> for <c>i &lt; size</c>) and its
    ///     midpoints (<c>z_lim</c>), NaN-padding ragged rows and back-filling each with its row <c>nanmax</c>, from
    ///     <c>_discrete_space_low/size/step</c>. Also seeds <c>m_z_lim_low/up</c> from <paramref name="mean" /> exactly
    ///     as the Python constructor does.
    /// </summary>
    /// <param name="mean">Initial or current proposal center in transformed coordinates.</param>
    /// <remarks>
    ///     The padded grid is how cmaes 0.13.0 encoded and corrected (<c>searchsorted</c> over the midpoints); cmaes
    ///     0.13.1 computes the same nearest index (lower on ties) and the same midpoints <c>(lower + upper) / 2</c> per
    ///     dimension instead, so both give identical results on this grid.
    /// </remarks>
    private void BuildGrid(double[] mean)
    {
        var discreteList = new double[_discreteIdx.Length][];
        int maxDiscrete = 0;
        for (int k = 0; k < _discreteIdx.Length; k++)
        {
            // A size below one leaves an empty row, as np.arange gives for an empty range.
            var row = new double[Math.Max(0, checked((int)_discreteSpaceSize[k]))];
            for (int j = 0; j < row.Length; j++)
                row[j] = _discreteSpaceLow[k] + j * _discreteSpaceStep[k];
            discreteList[k] = row;
            maxDiscrete = Math.Max(maxDiscrete, row.Length);
        }

        // discrete_space = np.full((n_zdim, max_discrete), nan); fill each row's prefix.
        _zSpace = new double[_discreteIdx.Length, maxDiscrete];
        for (int k = 0; k < _discreteIdx.Length; k++)
            for (int j = 0; j < maxDiscrete; j++)
                _zSpace[k, j] = j < discreteList[k].Length ? discreteList[k][j] : double.NaN;

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

    /// <summary>Rejects the single-vector interface because margin adaptation requires both encoded and raw proposals.</summary>
    /// <returns>One candidate vector in transformed coordinates.</returns>
    /// <exception cref="System.NotSupportedException">This interface always rejects single-vector asks; use AskEncoded to preserve both required vectors.</exception>
    double[] ICmaOptimizer.Ask() =>
        throw new NotSupportedException("CMAwM has a two-vector ask(); call AskEncoded() instead.");

    /// <summary>
    ///     Python <c>CMAwM.ask()</c> — resample raw draws until the continuous dims are feasible (or
    ///     fall back to a clipped last draw), then return <c>(x_encoded, x_raw)</c>: the encoded x has
    ///     its discrete dims snapped to the grid (used for evaluation), the raw x updates the
    ///     distribution (stored as <c>x_for_tell</c> and fed back to <see cref="Tell" />).
    /// </summary>
    /// <returns>The encoded candidate for evaluation and the raw vector required by Tell.</returns>
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
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <returns>The decision vector with discrete coordinates mapped to their evaluation values.</returns>
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
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <returns>True if all continuous coordinates satisfy the configured box.</returns>
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
    /// <param name="x">Input numerical coordinates or observation rows.</param>
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
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <returns>A vector with each discrete coordinate assigned to its threshold bin.</returns>
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

    /// <summary>Updates the optimizer using evaluated candidates from one coherent generation.</summary>
    /// <param name="solutions">Evaluated proposal vectors and minimization-oriented values for one coherent generation.</param>
    /// <exception cref="System.ArgumentException">The underlying CMA rejects the solution count or a coordinate magnitude.</exception>
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

    // ---- state serialization -------------------------------------------------------------------
    // ToPyInstance/FromPyInstance map this class to cmaes 0.13.1's CMAwM state, which CmaPickle writes and reads as
    // Python's exact pickle. ToState/FromState are the JSON form stored before 2026-10-29, still read so older
    // studies keep restoring.

    /// <summary>
    ///     Describes this optimizer as the state cmaes 0.13.1 pickles for a <c>CMAwM</c> (its instance dict, in
    ///     <c>__init__</c>'s order), with the inner <c>CMA</c> nested, so <see cref="PythonPickle.Dumps" /> of it is the
    ///     pickle cmaes writes for the same optimizer.
    /// </summary>
    /// <param name="layout">The pickle's dtype objects and whether this optimizer was restored before its last tell.</param>
    /// <returns>The <c>cmaes._cmawm.CMAwM</c> instance description; its arrays are copies.</returns>
    /// <remarks>
    ///     Without discrete dimensions cmaes returns from <c>__init__</c> before setting <c>margin</c>,
    ///     <c>m_z_lim_low</c>, <c>m_z_lim_up</c> and <c>_A</c>, so the state omits them. Every outer array is updated in
    ///     place or rebuilt from <c>_discrete_space_low</c> (a numpy result keeps its first operand's dtype object), so
    ///     in a restored optimizer all of them keep the dtype unpickling gave them.
    /// </remarks>
    internal PyInstance ToPyInstance(CmaPickleLayout layout)
    {
        var state = new List<KeyValuePair<string, object?>>();
        void Add(string key, object? value) => state.Add(new(key, value));
        Add("_cma", _cma.ToPyInstance(layout));
        Add("_n_max_resampling", (long)_nMaxResampling);
        Add("_discrete_idx", layout.Int64Vector(_discreteIdx));
        Add("_discrete_space_low", layout.Vector(_discreteSpaceLow, rebuiltByTell: false));
        Add("_discrete_space_size", layout.Int64Vector(_discreteSpaceSize));
        Add("_discrete_space_step", layout.Vector(_discreteSpaceStep, rebuiltByTell: false));
        Add("_continuous_idx", layout.Int64Vector(_continuousIdx));
        Add("_continuous_space", layout.Matrix(_continuousSpace, rebuiltByTell: false));
        Add("_n_zdim", (long)_nZdim);
        if (_nZdim > 0)
        {
            Add("margin", _margin);
            Add("m_z_lim_low", layout.Vector(_mZLimLow, rebuiltByTell: false));
            Add("m_z_lim_up", layout.Vector(_mZLimUp, rebuiltByTell: false));
            Add("_A", layout.Vector(_a, rebuiltByTell: false));
        }
        return new PyInstance("cmaes._cmawm", "CMAwM", state);
    }

    /// <summary>
    ///     Restores an optimizer from a pickled cmaes 0.13.1 <c>CMAwM</c> state, taking every value from it (the inner
    ///     <c>CMA</c> through <see cref="Cma.FromPyInstance" />) and building the discrete grid from its
    ///     <c>_discrete_space_low/size/step</c>.
    /// </summary>
    /// <param name="state">The unpickled <c>cmaes._cmawm.CMAwM</c> instance.</param>
    /// <returns>The restored optimizer; its random stream is unseeded, as after Python's <c>__setstate__</c>.</returns>
    /// <exception cref="ValueError">
    ///     The state was written by cmaes 0.13.0 or older (whose <c>CMAwM</c> keeps <c>z_space</c>/<c>z_lim</c>
    ///     instead), or an attribute is missing, mistyped, misshapen or out of range.
    /// </exception>
    internal static CmaWm FromPyInstance(PyInstance state)
    {
        // cmaes 0.13.1 replaced the materialized z_space/z_lim grids by this triple; Python workers of one study must
        // run one cmaes version, and OptunaSharp speaks 0.13.1's.
        if (!state.TryGet("_discrete_space_low", out _))
            throw new ValueError(
                "The pickled cmaes CMAwM has no _discrete_space_low: cmaes 0.13.0 or older wrote it. OptunaSharp reads " +
                "the CMAwM state of cmaes 0.13.1; run that version on the Python workers of a with_margin study.");
        if (state.Get("_cma") is not PyInstance { Module: "cmaes._cma", QualName: "CMA" } inner)
            throw new ValueError("The pickled cmaes CMAwM's _cma is not a cmaes CMA.");
        return new CmaWm(Cma.FromPyInstance(inner), state);
    }

    /// <summary>The constructor behind <see cref="FromPyInstance" />: assigns every field from the pickled state.</summary>
    /// <param name="cma">The inner optimizer, already restored from the state's <c>_cma</c>.</param>
    /// <param name="s">The unpickled <c>cmaes._cmawm.CMAwM</c> instance.</param>
    /// <exception cref="ValueError">An attribute is missing, mistyped, misshapen or out of range.</exception>
    private CmaWm(Cma cma, PyInstance s)
    {
        _cma = cma;
        int nDim = cma.Dim;
        _nMaxResampling = CmaPickle.Int(s, "_n_max_resampling");
        int nZdim = CmaPickle.Int(s, "_n_zdim");
        if (nZdim < 0 || nZdim > nDim)
            throw new ValueError($"The pickled cmaes CMAwM has n_zdim {nZdim} for {nDim} dimensions.");
        _discreteIdx = CmaPickle.IndexVector(s, "_discrete_idx", nZdim, nDim);
        _discreteSpaceLow = CmaPickle.Vector(s, "_discrete_space_low", nZdim);
        _discreteSpaceSize = CmaPickle.Int64Vector(s, "_discrete_space_size", nZdim);
        _discreteSpaceStep = CmaPickle.Vector(s, "_discrete_space_step", nZdim);
        // The grid is materialized here (see BuildGrid), so a size must fit an array; cmaes requires two choices.
        foreach (long size in _discreteSpaceSize)
            if (size < 2 || size > int.MaxValue)
                throw new ValueError($"The pickled cmaes CMAwM has a discrete dimension with {size} choices.");
        _continuousIdx = CmaPickle.IndexVector(s, "_continuous_idx", nDim - nZdim, nDim);
        _continuousSpace = CmaPickle.Matrix(s, "_continuous_space", nDim - nZdim, 2);

        // The legacy JSON form (ToState) keeps per-dimension steps; on Optuna's [0, 1] space the actual step equals
        // the requested one.
        _steps = new double[nDim];
        for (int k = 0; k < nZdim; k++)
            _steps[_discreteIdx[k]] = _discreteSpaceStep[k];

        BuildGrid(cma.Mean);
        _a = Enumerable.Repeat(1.0, nDim).ToArray();
        if (nZdim > 0)
        {
            // The evolving margin state replaces what BuildGrid seeded from the mean.
            _margin = CmaPickle.Float(s, "margin");
            _mZLimLow = CmaPickle.Vector(s, "m_z_lim_low", nZdim);
            _mZLimUp = CmaPickle.Vector(s, "m_z_lim_up", nZdim);
            _a = CmaPickle.Vector(s, "_A", nDim);
        }
    }

    /// <summary>Captures constructor settings and evolving optimizer arrays in a serializable state object.</summary>
    /// <returns>A serializable snapshot of optimizer settings and evolving state.</returns>
    internal CmaWmState ToState() => new()
    {
        Inner = _cma.ToState(),
        Steps = (double[])_steps.Clone(),
        Margin = _margin,
        A = (double[])_a.Clone(),
        MZLimLow = (double[])_mZLimLow.Clone(),
        MZLimUp = (double[])_mZLimUp.Clone(),
    };

    /// <summary>Recreates optimizer constants and restores evolving state from a serialized snapshot.</summary>
    /// <param name="s">Serialized optimizer settings and evolving proposal state.</param>
    /// <returns>An optimizer restored from the supplied snapshot.</returns>
    /// <exception cref="System.ArgumentException">Stored constructor settings or population dimensions are rejected when the optimizer is recreated.</exception>
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
    /// <param name="x">Scalar operand used in the comparison or numerical transformation.</param>
    /// <param name="loc">Mean of the normal distribution.</param>
    /// <param name="scale">Standard deviation of the normal distribution.</param>
    /// <returns>The Gaussian cumulative probability at the supplied coordinate.</returns>
    private static double NormCdf(double x, double loc, double scale) => Erf.NormCdf((x - loc) / scale);

    /// <summary>SciPy <c>chi2.ppf(q, df=1) = Φ⁻¹((1+q)/2)²</c> (scalar).</summary>
    /// <param name="q">Cumulative probability or probabilities in [0, 1].</param>
    /// <returns>The one-degree-of-freedom chi-square quantile or quantile vector.</returns>
    private static double Chi2Ppf1(double q)
    {
        double p = (1.0 + q) / 2.0;
        double ndtri = TruncNorm.NdtriExp(new[] { Math.Log(p) })[0];
        return ndtri * ndtri;
    }

    /// <summary>SciPy <c>chi2.ppf(q, df=1)</c> element-wise — batched through the one whole-array
    /// <see cref="TruncNorm.NdtriExp" /> Newton iteration (matches numpy's vectorized evaluation).</summary>
    /// <param name="q">Cumulative probability or probabilities in [0, 1].</param>
    /// <returns>The one-degree-of-freedom chi-square quantile or quantile vector.</returns>
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

    /// <summary>numpy <c>np.searchsorted(a[row], v, side='left')</c> — leftmost index <c>i</c> with
    /// <c>a[i-1] &lt; v &lt;= a[i]</c> (the count of elements strictly less than <paramref name="v" />).</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="row">Zero-based matrix row.</param>
    /// <param name="len">Number of active entries to examine.</param>
    /// <param name="v">Scalar operand used in the comparison or numerical transformation.</param>
    /// <returns>The insertion position before entries equal to the target.</returns>
    private static int SearchSortedLeft(double[,] a, int row, int len, double v)
    {
        int i = 0;
        while (i < len && a[row, i] < v) i++;
        return i;
    }

    /// <summary>Finds the largest non-NaN value in the active prefix of one matrix row.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="row">Zero-based matrix row.</param>
    /// <param name="len">Number of active entries to examine.</param>
    /// <returns>The largest usable row entry, or NaN if the active prefix has none.</returns>
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

    /// <summary>Copies the requested vector entries in the supplied index order.</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="idx">Zero-based indices selecting vector entries or population rows.</param>
    /// <returns>A newly allocated vector following the index order.</returns>
    private static double[] Gather(double[] a, int[] idx)
    {
        var r = new double[idx.Length];
        for (int k = 0; k < idx.Length; k++) r[k] = a[idx[k]];
        return r;
    }

    /// <summary>Clamps a scalar to an inclusive interval.</summary>
    /// <param name="v">Scalar operand used in the comparison or numerical transformation.</param>
    /// <param name="lo">Inclusive lower clipping bound.</param>
    /// <param name="hi">Inclusive upper clipping bound.</param>
    /// <returns>The value restricted to the inclusive interval.</returns>
    private static int Clip(int v, int lo, int hi) => Math.Min(Math.Max(v, lo), hi);
    /// <summary>Clamps a scalar to an inclusive interval.</summary>
    /// <param name="v">Scalar operand used in the comparison or numerical transformation.</param>
    /// <param name="lo">Inclusive lower clipping bound.</param>
    /// <param name="hi">Inclusive upper clipping bound.</param>
    /// <returns>The value restricted to the inclusive interval.</returns>
    private static double Clip(double v, double lo, double hi) => Math.Min(Math.Max(v, lo), hi);

    /// <summary>Python <c>_is_valid_bounds(bounds, mean)</c> — every mean within its [low, high] (empty ⇒ true).</summary>
    /// <param name="bounds">Optional box with one row per decision coordinate and lower/upper bound columns.</param>
    /// <param name="mean">Initial or current proposal center in transformed coordinates.</param>
    /// <returns>True if the box shape and coordinate bounds are compatible with the mean.</returns>
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
    /// <summary>Underlying full-covariance CMA state used by the margin wrapper.</summary>
    public CmaState Inner { get; set; } = new();
    /// <summary>Quantization step for each coordinate; zero denotes a continuous coordinate.</summary>
    public double[] Steps { get; set; } = System.Array.Empty<double>();
    /// <summary>Minimum probability mass maintained near discrete coordinate boundaries.</summary>
    public double Margin { get; set; }
    /// <summary>Serialized per-coordinate affine scaling used by margin correction.</summary>
    public double[] A { get; set; } = System.Array.Empty<double>();
    /// <summary>Nearest lower discrete threshold relative to the current mean.</summary>
    public double[] MZLimLow { get; set; } = System.Array.Empty<double>();
    /// <summary>Nearest upper discrete threshold relative to the current mean.</summary>
    public double[] MZLimUp { get; set; } = System.Array.Empty<double>();
}
