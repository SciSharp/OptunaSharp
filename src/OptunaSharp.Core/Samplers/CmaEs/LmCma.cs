// LM-CMA — Limited-Memory Covariance Matrix Adaptation Evolution Strategy.
//
// Reference paper:
//   Ilya Loshchilov. "LM-CMA: an Alternative to L-BFGS for Large Scale Black-box Optimization."
//   Evolutionary Computation (MIT Press). arXiv:1511.00221 (2015).  https://arxiv.org/abs/1511.00221
//
// This file is a paper-identical port of the paper's Algorithms 3–7 and §4.1–4.6. Every method below
// cites the exact Algorithm / section it realizes; see the <summary> on `LmCma` for the fidelity
// decisions, the abstraction-reuse rationale, and the paper's under-specified edges.

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace OptunaSharp.Samplers.CmaEs;

/// <summary>
///     Limited-Memory CMA-ES (LM-CMA) optimizer with an ask-and-tell interface — a <b>paper-identical</b>
///     implementation of Ilya Loshchilov, <i>"LM-CMA: an Alternative to L-BFGS for Large Scale Black-box
///     Optimization"</i> (Evolutionary Computation journal, 2015; arXiv:1511.00221), Algorithms 3–7.
///
///     <para>Unlike full <see cref="Cma" /> (which stores an <c>n×n</c> covariance and eigendecomposes it),
///     LM-CMA stores only <c>m ≪ n</c> direction vectors and <b>reconstructs</b> the Cholesky factor
///     <c>A</c> (and its inverse) on the fly from those vectors — memory and per-sample time are both
///     <c>O(mn)</c>, targeting <c>n</c> up to millions. There is <b>no eigendecomposition and no LAPACK</b>.</para>
///
///     <para><b>Why this does NOT reuse the full CMA machinery.</b> Only the pieces of the OptunaSharp CMA
///     abstraction that match the paper 100% are reused: the <see cref="ICmaOptimizer" /> ask/tell contract
///     (pure-continuous, one candidate per <see cref="Ask" />), the NumSharp MT19937 RNG
///     (<c>np.random.RandomState</c>), and the scalar-loop numeric style of <see cref="SepCma" />. Everything
///     the paper defines differently is implemented the paper's way, not CMA's:
///     <list type="bullet">
///         <item>candidates are drawn from the <b>Rademacher</b> distribution (§4.4), not Gaussian;</item>
///         <item><b>mirrored sampling</b> (Alg. 7 l.9–10): every even candidate is the reflection of the
///             preceding odd one;</item>
///         <item>step-size is adapted by the <b>Population Success Rule</b> (§4.5), not CSA — so there is no
///             <c>p_sigma</c> path, no <c>chi_n</c>, no <c>h_sigma</c> trigger;</item>
///         <item>the covariance is implicit in stored vectors, so there is no <c>_C</c>, no
///             <c>EigenDecomposition</c>, and no <c>ShouldStop</c> (the paper specifies none);</item>
///         <item>the paper defines <b>no box-constraint / resampling</b>, so this optimizer has none — a
///             caller that needs bounds must clamp externally.</item>
///     </list></para>
///
///     <para><b>Genuinely under-specified points in the paper</b> (resolved with the minimal coherent reading,
///     flagged inline at each site): (1) Alg. 5 sets <c>N_i = n</c> for all <c>i</c> — the paper's own
///     instantiation — which removes the general-<c>N</c> indexing ambiguity; (2) Alg. 5 line 13's
///     <c>j_{min(t+1,m)}</c> reads inconsistently with the fill-phase line 4, so the newly stored vector is
///     placed at the newest ordering slot; (3) <c>T = ⌊log n⌋</c> is taken as natural log and floored to a
///     minimum of 1; (4) PSR is skipped on the very first generation (no previous population exists);
///     (5) mixed-set ranking ties break by (previous-then-current, ascending index) — a stable order.
///     Byte-exact parity against Loshchilov's original C++ would require his reference to pin these edges;
///     this file is faithful to the <b>paper</b>, which is the stated spec.</para>
/// </summary>
internal sealed class LmCma : ICmaOptimizer
{
    private const double Eps = 1e-30;   // guard for ‖v‖² ≈ 0 (a just-stored pc that is numerically zero)

    // ---- fixed strategy parameters (Alg. 7 line 1) --------------------------------------------
    private readonly int _nDim;         // n
    private readonly int _popsize;      // λ = 4 + ⌊3 ln n⌋
    private readonly int _mu;           // μ = ⌊λ/2⌋
    private readonly double[] _weights; // w_i, i = 1..μ, normalized to sum 1
    private readonly double _muEff;     // μ_w = 1 / Σ w_i²
    private readonly double _cc;        // c_c = 0.5/√n
    private readonly double _c1;        // c_1 = 1/(10 ln(n+1))
    private readonly double _cSigma;    // c_σ = 0.3
    private readonly double _dSigma;    // d_σ = 1
    private readonly double _zTarget;   // z* = 0.25
    private readonly int _nSteps;       // N_steps = n  (target temporal distance, N_i ≡ n)
    private readonly int _mVec;         // m = number of stored direction vectors (default 4 + ⌊3 ln n⌋)
    private readonly int _T;            // update period = max(1, ⌊ln n⌋)

    // reconstruction constants (§4.1): a = √(1-c1), c = 1/√(1-c1), rFactor = c1/(1-c1)
    private readonly double _a;
    private readonly double _cInv;
    private readonly double _rFactor;

    // ---- evolving state -----------------------------------------------------------------------
    private readonly double _sigma0;
    private double[] _mean;             // m^t
    private double _sigma;              // σ^t
    private double[] _pc;               // p_c^t (evolution path)
    private double _s;                  // PSR accumulator s
    private int _t;                     // iteration counter t (== Generation)

    // limited-memory store: m physical slots; ordering pointers j (jarr[0]=oldest .. jarr[nStored-1]=newest)
    private readonly double[,] _P;      // P ∈ ℝ^{m×n}: stored evolution-path (direction) vectors
    private readonly double[,] _V;      // V ∈ ℝ^{m×n}: inverse vectors v = A⁻¹ p_c
    private readonly double[] _b;       // per-vector Cholesky coefficient b (Alg. 3)
    private readonly double[] _d;       // per-vector inverse coefficient d (Alg. 4)
    private readonly int[] _l;          // l[slot] = iteration index when the slot's vector was stored
    private readonly int[] _jarr;       // ordering pointers into the physical slots
    private int _nStored;               // number of stored vectors so far (0..m)

    private double[]? _prevFitness;     // f^{t-1} raw values in ask order (null before first tell)

    // per-generation ask bookkeeping
    private int _askIndex;              // 1-based index of the next candidate within the generation
    private double[]? _lastAskX;        // last odd candidate, reflected by the following even one

    private NumPyRandom _rng;

    /// <summary>Initializes the limited-memory evolution-path representation of the Gaussian proposal.</summary>
    /// <param name="mean">Initial or current proposal center in transformed coordinates.</param>
    /// <param name="sigma">Initial positive global proposal standard deviation.</param>
    /// <param name="seed">Seed used to reset the optimizer random stream.</param>
    /// <param name="populationSize">Number of individuals retained or evaluated per generation.</param>
    /// <param name="nVectors">Maximum number of stored evolution-path vectors; null selects the algorithm default.</param>
    /// <exception cref="System.ArgumentException">The scale is nonpositive, mean dimension is below two, or population size is nonpositive.</exception>
    internal LmCma(
        double[] mean,
        double sigma,
        int? seed = null,
        int? populationSize = null,
        int? nVectors = null)
    {
        if (sigma <= 0) throw new ArgumentException("sigma must be a non-zero positive value");
        int nDim = mean.Length;
        if (nDim <= 1) throw new ArgumentException("The dimension of mean must be larger than 1");

        // λ = 4 + ⌊3 ln n⌋ ; μ = ⌊λ/2⌋   (Alg. 7 line 1)
        int popsize = populationSize ?? 4 + (int)Math.Floor(3 * Math.Log(nDim));
        if (popsize <= 0) throw new ArgumentException("popsize must be a non-zero positive value.");
        int mu = popsize / 2;

        // w_i = (ln(μ+1) - ln(i)) / Σ_{j=1}^μ (ln(μ+1) - ln(j)) ,  i = 1..μ   (normalized log-weights).
        // Σ builtin `sum` semantics → sequential accumulation (matches SepCma's weight sums).
        var weightsPrime = new double[mu];
        for (int i = 0; i < mu; i++)
            weightsPrime[i] = Math.Log(mu + 1) - Math.Log(i + 1);
        double wpSum = 0; foreach (double w in weightsPrime) wpSum += w;
        var weights = new double[mu];
        for (int i = 0; i < mu; i++) weights[i] = weightsPrime[i] / wpSum;
        double sumWSq = 0; foreach (double w in weights) sumWSq += w * w;
        double muEff = 1.0 / sumWSq;                       // μ_w = 1 / Σ w_i²

        _nDim = nDim;
        _popsize = popsize;
        _mu = mu;
        _weights = weights;
        _muEff = muEff;
        _cSigma = 0.3;                                     // c_σ
        _zTarget = 0.25;                                   // z*
        _dSigma = 1.0;                                     // d_σ
        _nSteps = nDim;                                    // N_steps = n
        _cc = 0.5 / Math.Sqrt(nDim);                       // c_c = 0.5/√n
        _c1 = 1.0 / (10.0 * Math.Log(nDim + 1));           // c_1 = 1/(10 ln(n+1))
        _mVec = nVectors ?? 4 + (int)Math.Floor(3 * Math.Log(nDim));  // m (default = λ's formula)
        _T = Math.Max(1, (int)Math.Floor(Math.Log(nDim))); // T = ⌊ln n⌋ (guarded ≥ 1)

        // reconstruction constants (§4.1)
        _a = Math.Sqrt(1 - _c1);
        _cInv = 1.0 / Math.Sqrt(1 - _c1);
        _rFactor = _c1 / (1 - _c1);

        _mean = (double[])mean.Clone();
        _sigma = sigma;
        _sigma0 = sigma;
        _pc = new double[nDim];
        _s = 0.0;
        _t = 0;

        _P = new double[_mVec, nDim];
        _V = new double[_mVec, nDim];
        _b = new double[_mVec];
        _d = new double[_mVec];
        _l = new int[_mVec];
        _jarr = new int[_mVec];
        _nStored = 0;

        _prevFitness = null;
        _askIndex = 0;
        _lastAskX = null;

        _rng = np.random.RandomState();
        if (seed is not null) _rng.seed(seed.Value);
    }

    /// <summary>Number of numerical decision coordinates.</summary>
    public int Dim => _nDim;
    /// <summary>Number of evaluated individuals required for one generation.</summary>
    public int PopulationSize => _popsize;
    /// <summary>Number of completed optimizer updates.</summary>
    public int Generation => _t;

    /// <summary>Current Gaussian proposal center in transformed coordinates.</summary>
    internal double[] Mean => _mean;
    /// <summary>Global Gaussian proposal step size.</summary>
    internal double Sigma => _sigma;

    /// <summary>Resets the optimizer random stream for subsequent proposals.</summary>
    /// <param name="seed">Seed used to reset the optimizer random stream.</param>
    public void ReseedRng(int seed) => _rng.seed((uint)seed);

    // ---- Ask (Loshchilov 2015, Algorithm 7 lines 5–10) ----------------------------------------

    /// <summary>
    ///     Generate one candidate of the current generation (Loshchilov 2015, Algorithm 7 lines 5–10):
    ///     odd positions draw a fresh Rademacher–Cholesky sample <c>m^t + σ^t A z</c>; even positions are the
    ///     <b>mirrored</b> reflection of the preceding odd one, <c>m^t - (x_{k-1} - m^t)</c> (§4.6). This is
    ///     the batch ask/tell path; the per-trial sampler uses <see cref="AskCandidate" /> instead.
    /// </summary>
    /// <returns>One candidate vector in transformed coordinates.</returns>
    public double[] Ask()
    {
        int k = ++_askIndex;   // 1-based candidate index within the generation
        double[] x;
        if (k % 2 == 1)
        {
            // odd: fresh Rademacher–Cholesky candidate (Alg. 7 l.6–8).
            x = OddCandidate(k);
            _lastAskX = x;
        }
        else
        {
            // even: mirror the preceding odd candidate — x_k = m^t - (x_{k-1} - m^t)   (Alg. 7 l.10)
            double[] prev = _lastAskX!;
            x = new double[_nDim];
            for (int i = 0; i < _nDim; i++) x[i] = _mean[i] - (prev[i] - _mean[i]);
        }
        return x;
    }

    /// <summary>
    ///     Generate one <b>non-mirrored</b> candidate for 1-based generation position <paramref name="k" />
    ///     (Loshchilov 2015, Algorithm 7 lines 6–8 only). This is the entry point Optuna's per-trial sampler
    ///     protocol needs: it asks a
    ///     single candidate per trial from freshly-restored optimizer state, which cannot carry the
    ///     intra-generation mirror pairing that the batch <see cref="Ask" /> maintains. Passing the true
    ///     position <paramref name="k" /> keeps SelectSubset's <c>k==1</c> old-information boost firing only
    ///     for the generation's first candidate. This is the original (2014) un-mirrored LM-CMA; mirroring
    ///     (§4.6) is an evaluation-halving efficiency add-on, not needed for convergence.
    /// </summary>
    /// <param name="k">Zero-based candidate slot in the current generation.</param>
    /// <returns>The proposed decision vector for the requested population slot.</returns>
    internal double[] AskCandidate(int k) => OddCandidate(k);

    /// <summary>
    ///     Odd-branch core (Loshchilov 2015, Algorithm 7 lines 6–8): Rademacher pre-image → subset of stored
    ///     vectors → <c>x = m^t + σ^t A z</c>. RNG order follows the paper: Rademacher() first (l.6), then
    ///     SelectSubset() (l.7).
    /// </summary>
    /// <param name="k">Zero-based candidate slot in the current generation.</param>
    /// <returns>The unmirrored proposal used to construct its antithetic pair.</returns>
    private double[] OddCandidate(int k)
    {
        double[] z = Rademacher(_nDim);
        int[] subset = SelectSubset(k);
        double[] az = Az(z, subset);
        var x = new double[_nDim];
        for (int i = 0; i < _nDim; i++) x[i] = _mean[i] + _sigma * az[i];
        return x;
    }

    /// <summary>Rademacher sampling (Loshchilov 2015, §4.4): each component is −1 or +1 with equal probability.</summary>
    /// <param name="n">Number of vector entries to produce.</param>
    /// <returns>A vector of independent values equal to minus one or plus one.</returns>
    private double[] Rademacher(int n)
    {
        int[] bits = _rng.randint(0, 2, new Shape(n), typeof(int)).ToArray<int>();
        var z = new double[n];
        for (int i = 0; i < n; i++) z[i] = 2 * bits[i] - 1;   // {0,1} → {-1,+1}
        return z;
    }

    /// <summary>
    ///     SelectSubset (Loshchilov 2015, Algorithm 6): return the ordering positions of the <c>m*</c> newest
    ///     stored vectors. The paper's <c>m</c> here is the CURRENT stored count <c>_nStored</c> (early on
    ///     fewer than <c>_mVec</c> vectors exist); <c>m*=0 ⇒</c> isotropic sampling, which falls out naturally
    ///     (an empty subset makes <see cref="Az" /> return <c>z</c>).
    /// </summary>
    /// <param name="k">Zero-based candidate slot in the current generation.</param>
    /// <returns>Stored evolution-path indices participating in the proposal transform.</returns>
    private int[] SelectSubset(int k)
    {
        double mSigma = 4.0;                 // m_σ = 4
        if (k == 1) mSigma *= 10.0;          // force the oldest information for one of λ (Alg. 6 l.2–3)
        double gauss = Math.Abs(_rng.randn(1).ToArray<double>()[0]);   // |N(0,1)|
        int mStar = Math.Min((int)Math.Floor(mSigma * gauss), _nStored);
        var subset = new int[mStar];
        for (int t = 0; t < mStar; t++) subset[t] = _nStored - mStar + t;  // newest m*, oldest→newest
        return subset;
    }

    /// <summary>
    ///     Az (Loshchilov 2015, Algorithm 3): reconstruct <c>x = A z</c> from the selected direction vectors.
    ///     The dot product uses the ORIGINAL <c>z</c> at every step (the unrolled <c>A^t z</c> expansion of
    ///     §4.1), not the running <c>x</c>.
    /// </summary>
    /// <param name="z">Standardized observation or standard-normal coordinate array.</param>
    /// <param name="subset">Ordered indices of stored evolution paths used by the transform.</param>
    /// <returns>The vector after applying the selected limited-memory covariance factors.</returns>
    private double[] Az(double[] z, int[] subset)
    {
        var x = (double[])z.Clone();
        foreach (int pos in subset)
        {
            int slot = _jarr[pos];
            double dot = Dot(_V, slot, z);                 // V[slot,:] · z
            double kk = _b[slot] * dot;
            for (int i = 0; i < _nDim; i++) x[i] = _a * x[i] + kk * _P[slot, i];
        }
        return x;
    }

    /// <summary>
    ///     Ainvz (Loshchilov 2015, Algorithm 4): reconstruct <c>x = A⁻¹ p</c> from the first
    ///     <paramref name="count" /> ordering positions. The dot uses the RUNNING <c>x</c> (a sequential
    ///     dependency), and the update subtracts along <c>V</c> (not <c>P</c>).
    /// </summary>
    /// <param name="p">Numerical coordinates used by the operator.</param>
    /// <param name="count">Number of active inverse-transform factors to apply.</param>
    /// <returns>The vector after applying the active inverse covariance factors.</returns>
    private double[] Ainvz(double[] p, int count)
    {
        var x = (double[])p.Clone();
        for (int pos = 0; pos < count; pos++)
        {
            int slot = _jarr[pos];
            double dot = Dot(_V, slot, x);                 // V[slot,:] · x
            double kk = _d[slot] * dot;
            for (int i = 0; i < _nDim; i++) x[i] = _cInv * x[i] - kk * _V[slot, i];
        }
        return x;
    }

    // ---- Tell (Loshchilov 2015, Algorithm 7 lines 11–21) --------------------------------------

    /// <summary>
    ///     Complete one generation (Loshchilov 2015, Algorithm 7 lines 11–21): recombine the best μ into the
    ///     new mean (l.12), update the evolution path (l.13), refresh the direction-vector store every
    ///     <c>T</c> iterations (l.14–16), and adapt the step-size by the Population Success Rule (§4.5, l.17–20).
    /// </summary>
    /// <param name="solutions">Evaluated proposal vectors and minimization-oriented values for one coherent generation.</param>
    /// <exception cref="System.ArgumentException">The solution count differs from population size.</exception>
    public void Tell(List<(double[] X, double Value)> solutions)
    {
        if (solutions.Count != _popsize)
            throw new ArgumentException("Must tell popsize-length solutions.");

        // f^t: raw values in ask order (used by the PSR mixed-set ranking).
        var fCur = new double[_popsize];
        for (int i = 0; i < _popsize; i++) fCur[i] = solutions[i].Value;

        // best μ by value → new mean  m^{t+1} = Σ_{i=1}^μ w_i x_{i:λ}   (Alg. 7 l.12)
        var sorted = solutions.OrderBy(s => s.Value).ToList();
        var newMean = new double[_nDim];
        for (int i = 0; i < _mu; i++)
            for (int j = 0; j < _nDim; j++)
                newMean[j] += _weights[i] * sorted[i].X[j];

        // evolution path  p_c^{t+1} = (1-c_c) p_c^t + √(c_c(2-c_c)) √μ_w (m^{t+1} - m^t)/σ^t   (l.13)
        double pcCoef = Math.Sqrt(_cc * (2 - _cc) * _muEff);
        var newPc = new double[_nDim];
        for (int j = 0; j < _nDim; j++)
            newPc[j] = (1 - _cc) * _pc[j] + pcCoef * (newMean[j] - _mean[j]) / _sigma;

        _mean = newMean;
        _pc = newPc;

        // store a new direction vector every T iterations, then refresh the affected inverses (l.14–16)
        if (_t % _T == 0)
        {
            int changePos = UpdateSet(_pc, _t);
            UpdateInverses(changePos);
        }

        // Population Success Rule step-size (l.17–20). Skipped on the first generation (no f^{t-1}).
        if (_prevFitness is not null)
        {
            double zPsr = ComputePsr(_prevFitness, fCur);
            _s = (1 - _cSigma) * _s + _cSigma * zPsr;
            _sigma *= Math.Exp(_s / _dSigma);
        }

        _prevFitness = fCur;
        _t += 1;
        _askIndex = 0;
        _lastAskX = null;
    }

    /// <summary>
    ///     UpdateSet (Loshchilov 2015, Algorithm 5): insert <c>p_c</c> into the limited-memory store, keeping
    ///     stored vectors spaced ~<c>N_i = n</c> iterations apart. Returns the ordering position from which
    ///     inverse vectors must be recomputed.
    /// </summary>
    /// <param name="pc">New covariance evolution-path vector.</param>
    /// <param name="tOrig">Generation timestamp attached to the stored path.</param>
    /// <returns>The earliest changed position in the ordered evolution-path set.</returns>
    private int UpdateSet(double[] pc, int tOrig)
    {
        int changePos;
        if (_nStored < _mVec)
        {
            // fill phase: append at the newest ordering slot (physical slot == ordering index).
            int slot = _nStored;
            _jarr[_nStored] = slot;
            _l[slot] = tOrig;
            SetRow(_P, slot, pc);
            changePos = _nStored;
            _nStored++;
        }
        else
        {
            // full: find the consecutive pair with the smallest temporal gap (N_i ≡ n is a constant,
            // so subtracting it does not change the argmin) — Alg. 5 l.6, 0-based i ∈ [0, m-2].
            int argmin = 0; double best = double.PositiveInfinity;
            for (int i = 0; i < _mVec - 1; i++)
            {
                double gap = (_l[_jarr[i + 1]] - _l[_jarr[i]]) - _nSteps;
                if (gap < best) { best = gap; argmin = i; }
            }
            int imin = argmin + 1;   // paper: i_min ← 1 + argmin (1-based)  →  0-based argmin+1

            // Alg. 5 l.7: if even the smallest gap is ≥ N, discard the oldest instead (i_min ← 1 → 0-based 0).
            if ((_l[_jarr[imin]] - _l[_jarr[imin - 1]]) - _nSteps >= 0) imin = 0;

            // remove ordering position imin, shift the rest down, recycle its physical slot at the end.
            int recycled = _jarr[imin];
            for (int i = imin; i < _mVec - 1; i++) _jarr[i] = _jarr[i + 1];
            _jarr[_mVec - 1] = recycled;
            _l[recycled] = tOrig;
            SetRow(_P, recycled, pc);
            changePos = imin;
        }
        return changePos;
    }

    /// <summary>
    ///     UpdateInverses (Loshchilov 2015, §4.1 / Algorithm 7 l.16): recompute <c>v = A⁻¹ p_c</c> and the
    ///     per-vector <c>b</c>, <c>d</c> coefficients for every ordering position ≥ <paramref name="changePos" />
    ///     (those whose set of older vectors changed).
    /// </summary>
    /// <param name="changePos">Earliest changed position whose dependent inverse factors must be recomputed.</param>
    private void UpdateInverses(int changePos)
    {
        for (int pos = changePos; pos < _nStored; pos++)
        {
            int slot = _jarr[pos];
            double[] pcVec = GetRow(_P, slot);
            double[] v = Ainvz(pcVec, pos);           // reconstruct with the `pos` older vectors
            SetRow(_V, slot, v);

            double vv = 0; for (int i = 0; i < _nDim; i++) vv += v[i] * v[i];   // ‖v‖²
            if (vv < Eps)
            {
                _b[slot] = 0.0;
                _d[slot] = 0.0;
            }
            else
            {
                double denom = Math.Sqrt(1 + _rFactor * vv);
                _b[slot] = (_a / vv) * (denom - 1);               // b = (√(1-c1)/‖v‖²)(√(1+r‖v‖²) - 1)
                _d[slot] = (1.0 / (_a * vv)) * (1 - 1.0 / denom);  // d = (1/(√(1-c1)‖v‖²))(1 - 1/√(1+r‖v‖²))
            }
        }
    }

    /// <summary>
    ///     Population Success Rule (Loshchilov 2015, §4.5): rank the mixed set <c>f^{t-1} ∪ f^t</c>, then
    ///     <c>z_PSR = (Σ_i r^{t-1}(i) − r^t(i))/λ² − z*</c>.
    /// </summary>
    /// <param name="fPrev">Previous generation objective scores.</param>
    /// <param name="fCur">Current generation objective scores.</param>
    /// <returns>The population success-ranking statistic comparing consecutive generations.</returns>
    private double ComputePsr(double[] fPrev, double[] fCur)
    {
        int lam = _popsize;
        // (value, group, index): group 0 = previous population, 1 = current. Stable ascending sort; ties
        // break by (group, index) so previous-population members take the lower rank.
        var mixed = new (double Val, int Grp, int Idx)[2 * lam];
        for (int i = 0; i < lam; i++)
        {
            mixed[i] = (fPrev[i], 0, i);
            mixed[lam + i] = (fCur[i], 1, i);
        }
        var order = mixed
            .Select((e, idx) => (e, idx))
            .OrderBy(t => t.e.Val).ThenBy(t => t.idx)
            .ToArray();

        double sumPrev = 0, sumCur = 0;
        for (int rank = 1; rank <= 2 * lam; rank++)
        {
            var e = order[rank - 1].e;
            if (e.Grp == 0) sumPrev += rank; else sumCur += rank;
        }
        return (sumPrev - sumCur) / ((double)lam * lam) - _zTarget;
    }

    // ---- helpers ------------------------------------------------------------------------------

    /// <summary>Computes a scalar dot product using the reference accumulation order.</summary>
    /// <param name="mat">Matrix whose specified row participates in the operation.</param>
    /// <param name="row">Zero-based matrix row.</param>
    /// <param name="vec">Vector with entries aligned to the matrix coordinates.</param>
    /// <returns>The scalar inner product.</returns>
    private double Dot(double[,] mat, int row, double[] vec)
    {
        double s = 0;
        for (int i = 0; i < _nDim; i++) s += mat[row, i] * vec[i];
        return s;
    }

    /// <summary>Copies a matrix row into a contiguous vector.</summary>
    /// <param name="mat">Matrix whose specified row participates in the operation.</param>
    /// <param name="row">Zero-based matrix row.</param>
    /// <returns>A detached copy of the selected row.</returns>
    private double[] GetRow(double[,] mat, int row)
    {
        var r = new double[_nDim];
        for (int i = 0; i < _nDim; i++) r[i] = mat[row, i];
        return r;
    }

    /// <summary>Writes a vector into the specified matrix row.</summary>
    /// <param name="mat">Matrix whose specified row participates in the operation.</param>
    /// <param name="row">Zero-based matrix row.</param>
    /// <param name="vec">Vector with entries aligned to the matrix coordinates.</param>
    private void SetRow(double[,] mat, int row, double[] vec)
    {
        for (int i = 0; i < _nDim; i++) mat[row, i] = vec[i];
    }

    // ---- state serialization (own JSON; mirrors the SepCma pattern) ---------------------------

    /// <summary>Captures constructor settings and evolving optimizer arrays in a serializable state object.</summary>
    /// <returns>A serializable snapshot of optimizer settings and evolving state.</returns>
    internal LmCmaState ToState() => new()
    {
        NDim = _nDim,
        Sigma0 = _sigma0,
        Popsize = _popsize,
        MVec = _mVec,
        Mean = (double[])_mean.Clone(),
        Sigma = _sigma,
        Pc = (double[])_pc.Clone(),
        S = _s,
        T = _t,
        P = ToJagged(_P),
        V = ToJagged(_V),
        B = (double[])_b.Clone(),
        D = (double[])_d.Clone(),
        L = (int[])_l.Clone(),
        Jarr = (int[])_jarr.Clone(),
        NStored = _nStored,
        PrevFitness = _prevFitness is null ? null : (double[])_prevFitness.Clone(),
    };

    /// <summary>Recreates optimizer constants and restores evolving state from a serialized snapshot.</summary>
    /// <param name="s">Serialized optimizer settings and evolving proposal state.</param>
    /// <returns>An optimizer restored from the supplied snapshot.</returns>
    /// <exception cref="System.ArgumentException">Stored constructor settings or population dimensions are rejected when the optimizer is recreated.</exception>
    internal static LmCma FromState(LmCmaState s)
    {
        var cma = new LmCma(new double[s.NDim], s.Sigma0, seed: null,
            populationSize: s.Popsize, nVectors: s.MVec);
        cma._mean = (double[])s.Mean.Clone();
        cma._sigma = s.Sigma;
        cma._pc = (double[])s.Pc.Clone();
        cma._s = s.S;
        cma._t = s.T;
        FromJagged(s.P, cma._P);
        FromJagged(s.V, cma._V);
        Array.Copy(s.B, cma._b, cma._b.Length);
        Array.Copy(s.D, cma._d, cma._d.Length);
        Array.Copy(s.L, cma._l, cma._l.Length);
        Array.Copy(s.Jarr, cma._jarr, cma._jarr.Length);
        cma._nStored = s.NStored;
        cma._prevFitness = s.PrevFitness is null ? null : (double[])s.PrevFitness.Clone();
        return cma;
    }

    /// <summary>Copies a rectangular matrix into row arrays for serialization.</summary>
    /// <param name="m">Rectangular matrix operand.</param>
    /// <returns>A detached row-array copy of the rectangular matrix.</returns>
    private double[][] ToJagged(double[,] m)
    {
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
    /// <param name="m">Rectangular matrix operand.</param>
    private static void FromJagged(double[][] j, double[,] m)
    {
        for (int i = 0; i < j.Length; i++)
            for (int c = 0; c < j[i].Length; c++) m[i, c] = j[i][c];
    }
}

/// <summary>Serialized evolving state of an <see cref="LmCma" /> optimizer (own JSON; see <see cref="SepCmaState" />).</summary>
internal sealed class LmCmaState
{
    /// <summary>Serialized number of numerical decision coordinates.</summary>
    public int NDim { get; set; }
    /// <summary>Initial proposal step size retained for reconstruction.</summary>
    public double Sigma0 { get; set; }
    /// <summary>Serialized number of evaluated individuals in one generation.</summary>
    public int Popsize { get; set; }
    /// <summary>Maximum number of evolution-path vectors retained by limited-memory CMA.</summary>
    public int MVec { get; set; }
    /// <summary>Current Gaussian proposal center in transformed coordinates.</summary>
    public double[] Mean { get; set; } = Array.Empty<double>();
    /// <summary>Global Gaussian proposal step size.</summary>
    public double Sigma { get; set; }
    /// <summary>Evolution path used by the rank-one covariance update.</summary>
    public double[] Pc { get; set; } = Array.Empty<double>();
    /// <summary>Accumulated success-ranking statistic used for limited-memory step-size control.</summary>
    public double S { get; set; }
    /// <summary>Serialized generation/update counter.</summary>
    public int T { get; set; }
    /// <summary>Stored evolution-path vectors used by the limited-memory transform.</summary>
    public double[][] P { get; set; } = Array.Empty<double[]>();
    /// <summary>Stored inverse-transformed evolution-path vectors.</summary>
    public double[][] V { get; set; } = Array.Empty<double[]>();
    /// <summary>Coefficients of the forward limited-memory covariance transform.</summary>
    public double[] B { get; set; } = Array.Empty<double>();
    /// <summary>Coefficients of the inverse limited-memory covariance transform.</summary>
    public double[] D { get; set; } = Array.Empty<double>();
    /// <summary>Stored evolution-path timestamps used by replacement scheduling.</summary>
    public int[] L { get; set; } = Array.Empty<int>();
    /// <summary>Indices defining the active evolution-path order.</summary>
    public int[] Jarr { get; set; } = Array.Empty<int>();
    /// <summary>Number of evolution-path vectors currently retained.</summary>
    public int NStored { get; set; }
    /// <summary>Previous generation objective values used for step-size adaptation.</summary>
    public double[]? PrevFitness { get; set; }
}
