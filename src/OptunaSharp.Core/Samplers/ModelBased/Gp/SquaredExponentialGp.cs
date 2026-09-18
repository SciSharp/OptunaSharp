// REMBO's paper-faithful Gaussian-process surrogate — the isotropic squared-exponential kernel of
// Wang, Hutter, Zoghi, Matheson & de Freitas, "Bayesian Optimization in a Billion Dimensions via Random
// Embeddings", JAIR 55 (2016) 361-387, Definition 7 (continuous) and §3.2 (categorical). This is a NEW,
// beyond-core class (not a copy) that lives in the model-based GP toolkit so REMBO can use exactly the
// paper's surrogate instead of the shared stack's Matérn-5/2 ARD GPRegressor:
//
//   * Definition 7 (continuous):  k_ℓ(a,b) = K_SE((a−b)/ℓ),  K_SE(z) = exp(−‖z‖²/2)
//                              ⇒  exp(−‖a−b‖² / (2ℓ²))   — a SINGLE, isotropic length scale ℓ
//                                 (NOT the per-coordinate ARD length scales the Matérn stack fits).
//   * §3.2 (categorical):        k_λ(a,b) = exp(−(λ/2)·h(a,b)²),  h = Hamming distance (# differing coords).
//
// For a mixed space (REMBO's high-dimensional kernel over the projected real parameters can carry both
// kinds) the two are combined multiplicatively — exp(−contSqDist/(2ℓ²) − (λ/2)·hamming²) — which reduces
// to Def. 7 exactly when there are no categorical coordinates and to §3.2 exactly when there are no
// continuous ones. Both hyperparameters are single scalars, matching the paper (the whole point of the
// separate REMBO surrogate: the paper fits one ℓ, adapted by Algorithm 3, not a vector of ARD scales).
//
// The amplitude (signal variance) s² is PROFILED OUT in closed form given (ℓ,λ) — the standard kriging
// concentrated likelihood (Rasmussen & Williams 2006, eq. 2.30 profiled) — so the hyperparameter fit
// Algorithm 3 performs is a 1-D search over ℓ (2-D over (ℓ,λ) when categoricals are present), which is
// exactly what §3.3 optimizes with DIRECT + CMA-ES. Linear algebra reuses the repo's OpenBLAS-backed
// LinAlg (Cholesky / triangular solve) unchanged.

using System;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.ModelBased.Gp;

/// <summary>
///     A fitted isotropic squared-exponential Gaussian process — REMBO's paper surrogate (Wang et al.
///     2016, Def. 7 + §3.2). It models a standardized, maximization-oriented objective over a normalized
///     input space with a <b>single</b> length scale <c>ℓ</c> for the continuous coordinates and a single
///     Hamming scale <c>λ</c> for the categorical ones, the amplitude <c>s²</c> profiled out in closed
///     form. Construct one with <see cref="Fit" /> for a fixed <c>(ℓ, λ)</c>; score
///     <see cref="ConcentratedLogLikelihood" /> across candidate hyperparameters to drive the Algorithm-3
///     length-scale fit; then query <see cref="Posterior" /> for the acquisition.
///
///     <para><b>Not the Matérn GPRegressor.</b> The sibling <see cref="GPRegressor" /> in this toolkit is a
///     verbatim copy of the parity Matérn-5/2 ARD GP that TuRBO uses; this class exists because REMBO's
///     paper specifies a different kernel family and a single isotropic length scale, which cannot be
///     expressed by that GP without changing it.</para>
///
///     <para><b>Numerics.</b> A small relative nugget stabilizes the correlation matrix (REMBO's objectives
///     are deterministic, so this is a jitter, not a modeled noise); <see cref="Fit" /> escalates the
///     jitter until the Cholesky succeeds. The posterior variance is clamped at zero. Because the
///     factorization rides <see cref="LinAlg.Cholesky" /> (OpenBLAS), results are tolerance-reproducible on
///     a platform, not bit-exact against a foreign GP library — REMBO is a behavioral, convergence-validated
///     port with no upstream trajectory to match.</para>
/// </summary>
internal sealed class SquaredExponentialGp
{
    // Jitter ladder added to the correlation-matrix diagonal until the Cholesky is positive definite. The
    // correlation matrix has unit diagonal, so these are relative (signal-scaled) nuggets; the first is the
    // default deterministic-objective floor and the rest are escalations for near-duplicate training points.
    private static readonly double[] JitterLevels = { 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 };

    private readonly double[,] _x;          // (n, dim) normalized training inputs
    private readonly bool[] _isCategorical; // (dim,)
    private readonly double _invTwoLenSq;   // 1 / (2ℓ²) — the continuous coefficient
    private readonly double _halfLambda;    // λ / 2 — the categorical coefficient
    private readonly double[,] _cholL;      // lower Cholesky factor of R + jitter·I
    private readonly double[] _alpha;       // (R + jitter·I)⁻¹ y
    private readonly double _signalVar;     // profiled amplitude s²
    private readonly int _n;
    private readonly int _dim;

    /// <summary>The profiled signal variance <c>s²</c> — the amplitude of the posterior (variance is scaled by it).</summary>
    internal double SignalVariance => _signalVar;

    /// <summary>The isotropic continuous length scale <c>ℓ</c> this model was fit with.</summary>
    internal double LengthScale { get; }

    /// <summary>The categorical Hamming scale <c>λ</c> this model was fit with (unused when there are no categorical coordinates).</summary>
    internal double CategoricalScale { get; }

    private SquaredExponentialGp(
        double[,] x, bool[] isCategorical, double lengthScale, double categoricalScale,
        double[,] cholL, double[] alpha, double signalVar)
    {
        _x = x;
        _isCategorical = isCategorical;
        _n = x.GetLength(0);
        _dim = x.GetLength(1);
        LengthScale = lengthScale;
        CategoricalScale = categoricalScale;
        _invTwoLenSq = 1.0 / (2.0 * lengthScale * lengthScale);
        _halfLambda = 0.5 * categoricalScale;
        _cholL = cholL;
        _alpha = alpha;
        _signalVar = signalVar;
    }

    /// <summary>
    ///     Fits the model for a fixed <c>(ℓ, λ)</c>: builds the correlation matrix, escalates the jitter
    ///     until it factors, profiles the amplitude <c>s² = (1/n)·yᵀR⁻¹y</c>, and caches <c>R⁻¹y</c> for the
    ///     posterior. The training targets must already be <b>standardized</b> (zero mean, unit variance) and
    ///     <b>maximization-oriented</b> (the caller sign-flips a minimizing study), matching the convention the
    ///     acquisition's incumbent is expressed in.
    /// </summary>
    /// <param name="x">Normalized training inputs, one point per row (<c>n × dim</c>); <c>n ≥ 1</c>.</param>
    /// <param name="y">Standardized, maximization-oriented training targets, length <c>n</c>.</param>
    /// <param name="isCategorical">Per-coordinate flag: <c>true</c> uses the Hamming term, <c>false</c> the squared-exponential term.</param>
    /// <param name="lengthScale">The isotropic continuous length scale <c>ℓ &gt; 0</c>.</param>
    /// <param name="categoricalScale">The categorical Hamming scale <c>λ ≥ 0</c> (ignored when no coordinate is categorical).</param>
    /// <returns>The fitted model.</returns>
    /// <exception cref="System.ArgumentException">The shapes are inconsistent, or <paramref name="lengthScale" /> is not positive.</exception>
    /// <exception cref="NumSharp.LinAlgError">The correlation matrix stays non-positive-definite even at the largest jitter (degenerate/duplicated data).</exception>
    internal static SquaredExponentialGp Fit(
        double[,] x, double[] y, bool[] isCategorical, double lengthScale, double categoricalScale)
    {
        int n = x.GetLength(0);
        int dim = x.GetLength(1);
        if (n < 1)
            throw new ArgumentException("At least one training point is required.", nameof(x));
        if (y.Length != n)
            throw new ArgumentException("y length must equal the number of training rows.", nameof(y));
        if (isCategorical.Length != dim)
            throw new ArgumentException("isCategorical length must equal the input dimension.", nameof(isCategorical));
        if (!(lengthScale > 0.0))
            throw new ArgumentException("lengthScale must be positive.", nameof(lengthScale));

        double invTwoLenSq = 1.0 / (2.0 * lengthScale * lengthScale);
        double halfLambda = 0.5 * categoricalScale;
        double[,] r = CorrelationMatrix(x, isCategorical, invTwoLenSq, halfLambda);

        // Escalate the diagonal jitter until the Cholesky succeeds — near-duplicate training points (common
        // once the search concentrates) make R numerically singular at the smallest nugget.
        NumSharp.LinAlgError? lastError = null;
        foreach (double jitter in JitterLevels)
        {
            var c = (double[,])r.Clone();
            for (int i = 0; i < n; i++) c[i, i] += jitter;
            try
            {
                double[,] l = LinAlg.Cholesky(c);           // c = L·Lᵀ
                double[] invLy = LinAlg.SolveTriangular(l, y, upper: false);       // L⁻¹y
                double[] alpha = LinAlg.SolveTriangular(Transpose(l), invLy, upper: true);  // L⁻ᵀL⁻¹y = R⁻¹y

                // Profiled amplitude s² = yᵀR⁻¹y / n = ‖L⁻¹y‖² / n (Rasmussen & Williams, concentrated MLE).
                double quad = 0.0;
                for (int i = 0; i < n; i++) quad += invLy[i] * invLy[i];
                double signalVar = quad / n;

                return new SquaredExponentialGp(x, isCategorical, lengthScale, categoricalScale, l, alpha, signalVar);
            }
            catch (NumSharp.LinAlgError e)
            {
                lastError = e;   // not positive definite at this jitter; escalate.
            }
        }
        throw lastError!;
    }

    /// <summary>
    ///     The concentrated (profiled-amplitude) Gaussian log marginal likelihood at <c>(ℓ, λ)</c> — the
    ///     objective Algorithm 3 (Wang et al. §3.3) maximizes over the length scale. With the amplitude
    ///     profiled out, <c>2·ll = −n·log(ŝ²) − log|R| − n·(1 + log 2π)</c>, where
    ///     <c>ŝ² = yᵀR⁻¹y / n</c>. Constants are kept (unlike the parity stack, which omits them) so the
    ///     value is a genuine log-likelihood; it does not change the argmax. Returns
    ///     <see cref="double.NegativeInfinity" /> when the correlation matrix will not factor even at the
    ///     largest jitter, so a hyperparameter optimizer treats that region as infeasible rather than
    ///     throwing.
    /// </summary>
    /// <param name="x">Normalized training inputs, one point per row (<c>n × dim</c>).</param>
    /// <param name="y">Standardized, maximization-oriented training targets, length <c>n</c>.</param>
    /// <param name="isCategorical">Per-coordinate categorical flag.</param>
    /// <param name="lengthScale">The candidate isotropic length scale <c>ℓ &gt; 0</c>.</param>
    /// <param name="categoricalScale">The candidate categorical Hamming scale <c>λ ≥ 0</c>.</param>
    /// <returns>The concentrated log marginal likelihood, or <c>−∞</c> if <c>ℓ ≤ 0</c> or the matrix is not factorable.</returns>
    internal static double ConcentratedLogLikelihood(
        double[,] x, double[] y, bool[] isCategorical, double lengthScale, double categoricalScale)
    {
        if (!(lengthScale > 0.0))
            return double.NegativeInfinity;

        int n = x.GetLength(0);
        double invTwoLenSq = 1.0 / (2.0 * lengthScale * lengthScale);
        double halfLambda = 0.5 * categoricalScale;
        double[,] r = CorrelationMatrix(x, isCategorical, invTwoLenSq, halfLambda);

        foreach (double jitter in JitterLevels)
        {
            var c = (double[,])r.Clone();
            for (int i = 0; i < n; i++) c[i, i] += jitter;
            try
            {
                double[,] l = LinAlg.Cholesky(c);
                double[] invLy = LinAlg.SolveTriangular(l, y, upper: false);
                double quad = 0.0;
                for (int i = 0; i < n; i++) quad += invLy[i] * invLy[i];
                double signalVar = quad / n;
                if (!(signalVar > 0.0))
                    return double.NegativeInfinity;   // a constant target after standardization: no information.

                double logDet = 0.0;                  // log|R| = 2·Σ log(L_ii)
                for (int i = 0; i < n; i++) logDet += 2.0 * Math.Log(l[i, i]);

                return -0.5 * (n * Math.Log(signalVar) + logDet + n * (1.0 + Math.Log(2.0 * Math.PI)));
            }
            catch (NumSharp.LinAlgError)
            {
                // escalate the jitter and retry.
            }
        }
        return double.NegativeInfinity;
    }

    /// <summary>
    ///     The GP posterior mean and variance at one query point. With the correlation vector
    ///     <c>r_j = k(x⋆, X_j)</c>: <c>mean = r·(R⁻¹y)</c> and <c>var = s²·max(0, 1 − rᵀR⁻¹r)</c> (the query
    ///     is treated as noise-free, so its prior correlation with itself is 1). The mean is in the
    ///     standardized, maximization-oriented target coordinate the acquisition's incumbent uses.
    /// </summary>
    /// <param name="xStar">One normalized query point in input-space coordinate order (length <c>dim</c>).</param>
    /// <returns>The posterior mean and the (nonnegative) posterior variance at the query point.</returns>
    /// <exception cref="System.ArgumentException"><paramref name="xStar" /> does not have length <c>dim</c>.</exception>
    internal (double Mean, double Var) Posterior(double[] xStar)
    {
        if (xStar.Length != _dim)
            throw new ArgumentException($"Query point must have length {_dim}.", nameof(xStar));

        var r = new double[_n];
        for (int j = 0; j < _n; j++)
            r[j] = CorrelationTo(xStar, j);

        double mean = 0.0;
        for (int j = 0; j < _n; j++)
            mean += r[j] * _alpha[j];

        // rᵀR⁻¹r = ‖L⁻¹r‖².
        double[] invLr = LinAlg.SolveTriangular(_cholL, r, upper: false);
        double rRinvR = 0.0;
        for (int j = 0; j < _n; j++) rRinvR += invLr[j] * invLr[j];

        double var = _signalVar * Math.Max(0.0, 1.0 - rRinvR);
        return (mean, var);
    }

    /// <summary>The correlation between two training rows under <c>(ℓ, λ)</c> — the SE term over continuous
    ///     coordinates times the Hamming term over categorical ones.</summary>
    /// <param name="x">The normalized training inputs.</param>
    /// <param name="isCategorical">Per-coordinate categorical flag.</param>
    /// <param name="i">First training-row index.</param>
    /// <param name="j">Second training-row index.</param>
    /// <param name="invTwoLenSq">The precomputed continuous coefficient <c>1/(2ℓ²)</c>.</param>
    /// <param name="halfLambda">The precomputed categorical coefficient <c>λ/2</c>.</param>
    /// <returns>The kernel correlation in <c>(0, 1]</c>.</returns>
    private static double CorrelationBetween(
        double[,] x, bool[] isCategorical, int i, int j, double invTwoLenSq, double halfLambda)
    {
        int dim = x.GetLength(1);
        double contSq = 0.0;
        int hamming = 0;
        for (int d = 0; d < dim; d++)
        {
            if (isCategorical[d])
            {
                if (x[i, d] != x[j, d]) hamming++;
            }
            else
            {
                double diff = x[i, d] - x[j, d];
                contSq += diff * diff;
            }
        }
        return Math.Exp(-invTwoLenSq * contSq - halfLambda * (double)hamming * hamming);
    }

    /// <summary>The correlation between a query point and training row <paramref name="j" /> under this model's <c>(ℓ, λ)</c>.</summary>
    /// <param name="xStar">The normalized query point.</param>
    /// <param name="j">The training-row index.</param>
    /// <returns>The kernel correlation in <c>(0, 1]</c>.</returns>
    private double CorrelationTo(double[] xStar, int j)
    {
        double contSq = 0.0;
        int hamming = 0;
        for (int d = 0; d < _dim; d++)
        {
            if (_isCategorical[d])
            {
                if (xStar[d] != _x[j, d]) hamming++;
            }
            else
            {
                double diff = xStar[d] - _x[j, d];
                contSq += diff * diff;
            }
        }
        return Math.Exp(-_invTwoLenSq * contSq - _halfLambda * (double)hamming * hamming);
    }

    /// <summary>Builds the <c>n × n</c> training correlation matrix (unit diagonal) under <c>(ℓ, λ)</c>.</summary>
    /// <param name="x">The normalized training inputs.</param>
    /// <param name="isCategorical">Per-coordinate categorical flag.</param>
    /// <param name="invTwoLenSq">The continuous coefficient <c>1/(2ℓ²)</c>.</param>
    /// <param name="halfLambda">The categorical coefficient <c>λ/2</c>.</param>
    /// <returns>The symmetric unit-diagonal correlation matrix.</returns>
    private static double[,] CorrelationMatrix(double[,] x, bool[] isCategorical, double invTwoLenSq, double halfLambda)
    {
        int n = x.GetLength(0);
        var r = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            r[i, i] = 1.0;
            for (int j = i + 1; j < n; j++)
            {
                double c = CorrelationBetween(x, isCategorical, i, j, invTwoLenSq, halfLambda);
                r[i, j] = c;
                r[j, i] = c;
            }
        }
        return r;
    }

    /// <summary>Returns a new matrix with rows and columns exchanged (for the <c>L⁻ᵀ</c> triangular solve).</summary>
    /// <param name="a">The matrix to transpose.</param>
    /// <returns>The transposed matrix.</returns>
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
