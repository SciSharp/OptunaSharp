// Exponential Natural Evolution Strategies (xNES) — the full-covariance NES of Wierstra et al., ask/tell core.
//
// Reference:
//   Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, Jan Peters, Jürgen Schmidhuber.
//     "Natural Evolution Strategies." JMLR 15 (2014) 949–980. §4 Algorithm 4 (Exponential NES) + Table 1
//     (default learning rates and utility function as functions of the dimension d).
//
// xNES represents the search distribution with a full covariance factored as A = σ·B (a global step size σ
// and a unit-determinant transform B), and updates B through the MATRIX EXPONENTIAL of a symmetric
// natural-gradient direction — the parameterization that keeps the update invariant to linear
// transformations of the search space and closely connects xNES to CMA-ES's rank-µ update. Because
// OptunaSharp already ships full-covariance adaptation (CMA-ES, VkD-CMA, LM-CMA), xNES is the lower-priority
// member of the ES family; it is included for completeness of the NES story. Like `Cma`/`De`, it has no
// Optuna counterpart, so this is a native implementation validated by convergence, on NumSharp's
// NumPy-exact MT19937 and OpenBLAS-backed linear algebra. xNES is UNBOUNDED; `XnesSampler` clamps.
//
// [Implementation note] The JMLR Algorithm-4 box writes the SAMPLE as z = µ + σ·Bᵀ·s but the MEAN UPDATE
// as µ + η·σB·∇δ; those agree only for symmetric B (a transpose slip in the linearized box). We use B
// consistently in BOTH places — z = µ + σ·B·s and µ ← µ + η_µ·σ·B·∇δ — matching the reference pybrain /
// evosax xNES implementations, so the sample map and the mean-gradient map are the same operator σB.

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.EvolutionStrategy;

/// <summary>
///     An Exponential-NES optimizer (Wierstra 2014, Algorithm 4) with the per-generation ask/tell interface
///     the shared <see cref="EvolutionStrategySampler" /> drives. The search distribution is a full Gaussian
///     <c>N(µ, σ²·B·Bᵀ)</c>. Each <see cref="Ask" /> proposes <c>µ + σ·(B·s)</c> for <c>s ~ N(0, I)</c>;
///     <see cref="Tell" /> forms the rank utilities and takes natural-gradient steps:
///     <c>µ ← µ + η_µ·σ·(B·∇δ)</c>, <c>σ ← σ·exp(η_σ/2·∇σ)</c>, and <c>B ← B·exp(η_B/2·∇B)</c>, where
///     <c>∇δ = Σ u_k s_k</c>, <c>∇M = Σ u_k (s_k s_kᵀ − I)</c>, <c>∇σ = tr(∇M)/d</c>, and
///     <c>∇B = ∇M − ∇σ·I</c> (symmetric, traceless).
///
///     <para>The <c>exp(·)</c> on B is a <b>matrix</b> exponential of a symmetric matrix, computed via an
///     eigendecomposition (OpenBLAS <c>eigh</c>): <c>exp(Q Λ Qᵀ) = Q exp(Λ) Qᵀ</c>. Reconstructing the
///     per-candidate noise from a stored candidate requires a linear solve <c>s = (σB)⁻¹(x − µ)</c>.
///     <b>Orientation:</b> the optimizer minimizes; the best (lowest) value earns the largest utility.</para>
/// </summary>
internal sealed class XnesOptimizer : IEvolutionStrategyOptimizer
{
    private const double SigmaEps = 1e-300;   // keeps σ strictly positive

    private readonly int _nDim;
    private readonly int _popsize;
    private readonly double _etaMu;
    private readonly double _etaSigma;
    private readonly double _etaB;
    private readonly uint _seed;

    private double[] _mean;     // distribution mean µ
    private double _sigma;      // global step size σ
    private double[,] _b;       // unit-determinant transform B (d×d)
    private int _generation;

    private readonly NumPyRandom _rng;

    /// <param name="mean">Initial mean µ (the transformed box center); copied defensively.</param>
    /// <param name="sigma">Initial global step size σ (in the transformed <c>[0,1]</c> space).</param>
    /// <param name="b">Initial unit-determinant transform B (d×d); copied defensively (identity ⇒ isotropic start).</param>
    /// <param name="etaMu">Mean learning rate η_µ (Table 1 default 1).</param>
    /// <param name="etaSigma">Step-size learning rate η_σ (Table 1 default <c>(9 + 3 ln d)/(5 d √d)</c>).</param>
    /// <param name="etaB">Transform learning rate η_B (Table 1 default equals η_σ).</param>
    /// <param name="popsize">Population size λ per generation (Table 1 default <c>4 + ⌊3 ln d⌋</c>).</param>
    /// <param name="seed">Base seed for the per-generation noise streams.</param>
    /// <summary>Initializes the full-covariance Gaussian for Exponential NES.</summary>
    /// <exception cref="System.ArgumentException">The dimension is non-positive, B is not d×d, or the population is below two.</exception>
    internal XnesOptimizer(
        double[] mean, double sigma, double[,] b, double etaMu, double etaSigma, double etaB, int popsize, uint seed)
    {
        if (mean is null || mean.Length == 0) throw new ArgumentException("mean must be non-empty.", nameof(mean));
        int n = mean.Length;
        if (b is null || b.GetLength(0) != n || b.GetLength(1) != n)
            throw new ArgumentException("B must be a d×d matrix.", nameof(b));
        if (popsize < 2) throw new ArgumentException("Population size must be at least 2.", nameof(popsize));

        _nDim = n;
        _popsize = popsize;
        _etaMu = etaMu;
        _etaSigma = etaSigma;
        _etaB = etaB;
        _seed = seed;

        _mean = (double[])mean.Clone();
        _sigma = sigma;
        _b = (double[,])b.Clone();
        _generation = 0;

        _rng = np.random.RandomState();
    }

    /// <inheritdoc />
    public int Dim => _nDim;
    /// <inheritdoc />
    public int PopulationSize => _popsize;
    /// <inheritdoc />
    public int Generation => _generation;

    /// <summary>The current global step size σ (exposed for tests/diagnostics).</summary>
    internal double Sigma => _sigma;

    /// <inheritdoc />
    /// <exception cref="System.ArgumentOutOfRangeException"><paramref name="index" /> is outside the current generation.</exception>
    public double[] Ask(int index)
    {
        if (index < 0 || index >= _popsize) throw new ArgumentOutOfRangeException(nameof(index));

        _rng.seed(EsSampling.Combine(_seed, _generation));
        double[][] s = EsSampling.StandardNormalMatrix(_rng, _popsize, _nDim);
        double[] si = s[index];
        double[] bs = LinAlg.Dot(_b, si);   // B·s
        var x = new double[_nDim];
        for (int j = 0; j < _nDim; j++)
            x[j] = _mean[j] + _sigma * bs[j];   // candidate = µ + σ·(B·s)
        return x;
    }

    /// <inheritdoc />
    /// <exception cref="System.ArgumentException">The number of solutions differs from the population size.</exception>
    /// <exception cref="NumSharp.LinAlgError">The transform σB is singular, so the per-candidate noise cannot be recovered.</exception>
    public void Tell(IReadOnlyList<(double[] X, double Value)> solutions)
    {
        if (solutions.Count != _popsize)
            throw new ArgumentException($"Must tell exactly {_popsize} solutions for one generation.", nameof(solutions));

        // Reconstruct each standardized noise s = (σB)⁻¹(x − µ) from the actually-evaluated candidate. One
        // shared coefficient matrix σB is solved per candidate (small d/λ, so a per-solution solve is fine).
        var scaledB = new double[_nDim, _nDim];
        for (int r = 0; r < _nDim; r++)
            for (int c = 0; c < _nDim; c++)
                scaledB[r, c] = _sigma * _b[r, c];

        var s = new double[_popsize][];
        for (int i = 0; i < _popsize; i++)
        {
            var rhs = new double[_nDim];
            for (int j = 0; j < _nDim; j++) rhs[j] = solutions[i].X[j] - _mean[j];
            s[i] = LinAlg.Solve(scaledB, rhs);
        }

        // Rank ascending by objective (best = lowest); NaN sinks to the worst rank. Utilities aligned by rank.
        int[] order = Enumerable.Range(0, _popsize)
            .OrderBy(i => double.IsNaN(solutions[i].Value) ? double.PositiveInfinity : solutions[i].Value)
            .ToArray();
        double[] utilities = EsFitnessShaping.NesUtilities(_popsize);

        // ∇δ = Σ u_k s_k (d-vector); ∇M = Σ u_k (s_k s_kᵀ − I) (symmetric d×d).
        var gradDelta = new double[_nDim];
        var gradM = new double[_nDim, _nDim];
        for (int r = 0; r < _popsize; r++)
        {
            double u = utilities[r];
            double[] sr = s[order[r]];
            for (int a = 0; a < _nDim; a++)
            {
                gradDelta[a] += u * sr[a];
                for (int b = 0; b < _nDim; b++)
                    // The −I term is folded in below via the diagonal subtraction of u·(sum of utilities);
                    // here accumulate u·s_aᵀs_b, then subtract u on the diagonal.
                    gradM[a, b] += u * sr[a] * sr[b];
            }
        }
        // Subtract Σ u_k on the diagonal to complete Σ u_k (s_k s_kᵀ − I).
        double utilitySum = 0;
        for (int r = 0; r < _popsize; r++) utilitySum += utilities[r];
        for (int a = 0; a < _nDim; a++) gradM[a, a] -= utilitySum;

        // ∇σ = tr(∇M)/d (scalar); ∇B = ∇M − ∇σ·I (symmetric, traceless).
        double trace = 0;
        for (int a = 0; a < _nDim; a++) trace += gradM[a, a];
        double gradSigma = trace / _nDim;

        var gradB = new double[_nDim, _nDim];
        for (int a = 0; a < _nDim; a++)
            for (int b = 0; b < _nDim; b++)
                gradB[a, b] = gradM[a, b] - (a == b ? gradSigma : 0.0);

        // µ ← µ + η_µ·σ·(B·∇δ).
        double[] bDelta = LinAlg.Dot(_b, gradDelta);
        for (int j = 0; j < _nDim; j++)
            _mean[j] += _etaMu * _sigma * bDelta[j];

        // σ ← σ·exp(η_σ/2·∇σ).
        _sigma *= Math.Exp(0.5 * _etaSigma * gradSigma);
        if (_sigma < SigmaEps) _sigma = SigmaEps;

        // B ← B·exp(η_B/2·∇B): scale the symmetric direction, matrix-exponentiate it, and right-multiply B.
        var scaledGradB = new double[_nDim, _nDim];
        double bScale = 0.5 * _etaB;
        for (int a = 0; a < _nDim; a++)
            for (int b = 0; b < _nDim; b++)
                scaledGradB[a, b] = bScale * gradB[a, b];
        double[,] expB = ExpmSymmetric(scaledGradB);
        _b = LinAlg.Dot(_b, expB);

        _generation++;
    }

    /// <summary>
    ///     Matrix exponential of a real <b>symmetric</b> matrix via its eigendecomposition:
    ///     <c>exp(Q Λ Qᵀ) = Q exp(Λ) Qᵀ</c> (OpenBLAS <c>eigh</c> supplies <c>Q, Λ</c>). Valid only because
    ///     the ∇B direction is symmetric by construction; a general matrix exponential would need a Padé/
    ///     scaling-and-squaring routine we deliberately avoid here.
    /// </summary>
    /// <param name="m">A real symmetric matrix (only its lower triangle is read by <c>eigh</c>).</param>
    /// <returns>The symmetric-positive-definite matrix exponential of <paramref name="m" />.</returns>
    /// <exception cref="NumSharp.OpenBlasMissingBackendException">The native LAPACK backend is not loaded.</exception>
    private static double[,] ExpmSymmetric(double[,] m)
    {
        int n = m.GetLength(0);
        var (eigenvalues, q) = LinAlg.Eigh(m);   // m = Q · diag(eigenvalues) · Qᵀ
        // QD[i,k] = Q[i,k]·exp(λ_k); result = QD · Qᵀ ⇒ result[i,j] = Σ_k Q[i,k]·exp(λ_k)·Q[j,k].
        var qd = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++)
                qd[i, k] = q[i, k] * Math.Exp(eigenvalues[k]);
        var qt = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                qt[i, j] = q[j, i];   // Qᵀ
        return LinAlg.Dot(qd, qt);
    }

    /// <summary>Default step-size/transform learning rate <c>(9 + 3 ln d)/(5 d √d)</c> (Wierstra 2014, Table 1, xNES).</summary>
    /// <param name="d">Decision-space dimension.</param>
    /// <returns>The dimension-scaled default η_σ = η_B.</returns>
    internal static double DefaultEtaSigma(int d) => (9.0 + 3.0 * Math.Log(d)) / (5.0 * d * Math.Sqrt(d));

    /// <summary>Default population size λ = <c>4 + ⌊3 ln d⌋</c> (Wierstra 2014, Table 1), floored at 4.</summary>
    /// <param name="d">Decision-space dimension.</param>
    /// <returns>The dimension-scaled default population size (at least 4).</returns>
    internal static int DefaultPopulationSize(int d) => Math.Max(4, 4 + (int)Math.Floor(3.0 * Math.Log(d)));

    /// <summary>Builds a d×d identity matrix — the isotropic initial transform B.</summary>
    /// <param name="d">Matrix dimension.</param>
    /// <returns>The d×d identity.</returns>
    internal static double[,] Identity(int d)
    {
        var m = new double[d, d];
        for (int i = 0; i < d; i++) m[i, i] = 1.0;
        return m;
    }

    // ---- state serialization (own JSON; mirrors the De/LmCma pattern) -----------------------------

    /// <summary>Captures constructor settings and evolving state in a serializable object (B stored jagged).</summary>
    /// <returns>A serializable snapshot of the optimizer.</returns>
    internal XnesState ToState() => new()
    {
        NDim = _nDim,
        Popsize = _popsize,
        EtaMu = _etaMu,
        EtaSigma = _etaSigma,
        EtaB = _etaB,
        Seed = _seed,
        Mean = (double[])_mean.Clone(),
        Sigma = _sigma,
        B = ToJagged(_b),
        Generation = _generation,
    };

    /// <summary>Recreates the optimizer, restoring the mean, σ, transform B, and generation counter.</summary>
    /// <param name="s">Serialized optimizer state.</param>
    /// <returns>An optimizer restored from the supplied snapshot.</returns>
    /// <exception cref="System.ArgumentException">The stored settings are rejected by the constructor.</exception>
    internal static XnesOptimizer FromState(XnesState s)
    {
        var opt = new XnesOptimizer(
            s.Mean, s.Sigma, ToRectangular(s.B, s.NDim), s.EtaMu, s.EtaSigma, s.EtaB, s.Popsize, s.Seed);
        opt._generation = s.Generation;
        return opt;
    }

    /// <summary>Converts a rectangular matrix to jagged rows for JSON (System.Text.Json rejects <c>double[,]</c>).</summary>
    /// <param name="m">The rectangular matrix to convert.</param>
    /// <returns>A jagged copy with one array per row.</returns>
    private static double[][] ToJagged(double[,] m)
    {
        int rows = m.GetLength(0), cols = m.GetLength(1);
        var j = new double[rows][];
        for (int r = 0; r < rows; r++)
        {
            var row = new double[cols];
            for (int c = 0; c < cols; c++) row[c] = m[r, c];
            j[r] = row;
        }
        return j;
    }

    /// <summary>Converts jagged JSON rows back to a rectangular d×d matrix.</summary>
    /// <param name="j">The jagged matrix rows.</param>
    /// <param name="n">Expected square dimension d.</param>
    /// <returns>The rectangular d×d matrix.</returns>
    /// <exception cref="System.ArgumentException">The jagged data is not d×d.</exception>
    private static double[,] ToRectangular(double[][] j, int n)
    {
        if (j is null || j.Length != n) throw new ArgumentException("Stored B is not d×d.", nameof(j));
        var m = new double[n, n];
        for (int r = 0; r < n; r++)
        {
            if (j[r] is null || j[r].Length != n) throw new ArgumentException("Stored B is not d×d.", nameof(j));
            for (int c = 0; c < n; c++) m[r, c] = j[r][c];
        }
        return m;
    }
}

/// <summary>Serialized evolving state of an <see cref="XnesOptimizer" /> (own JSON; B stored jagged).</summary>
internal sealed class XnesState
{
    /// <summary>Serialized number of numerical decision coordinates.</summary>
    public int NDim { get; set; }
    /// <summary>Serialized population size per generation.</summary>
    public int Popsize { get; set; }
    /// <summary>Serialized mean learning rate η_µ.</summary>
    public double EtaMu { get; set; }
    /// <summary>Serialized step-size learning rate η_σ.</summary>
    public double EtaSigma { get; set; }
    /// <summary>Serialized transform learning rate η_B.</summary>
    public double EtaB { get; set; }
    /// <summary>Serialized base seed for the per-generation noise streams.</summary>
    public uint Seed { get; set; }
    /// <summary>Serialized mean µ.</summary>
    public double[] Mean { get; set; } = Array.Empty<double>();
    /// <summary>Serialized global step size σ.</summary>
    public double Sigma { get; set; }
    /// <summary>Serialized unit-determinant transform B, stored as jagged rows.</summary>
    public double[][] B { get; set; } = Array.Empty<double[]>();
    /// <summary>Serialized generation/update counter.</summary>
    public int Generation { get; set; }
}
