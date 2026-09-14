// Separable Natural Evolution Strategies (SNES) — the O(d) diagonal NES of Wierstra et al., ask/tell core.
//
// Reference:
//   Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, Jan Peters, Jürgen Schmidhuber.
//     "Natural Evolution Strategies." JMLR 15 (2014) 949–980. §5.1 Algorithm 7 (Separable NES) + Table 1
//     (default learning rates and utility function as functions of the dimension d).
//
// SNES restricts the NES search distribution to a Gaussian with DIAGONAL covariance — one mean and one
// standard deviation per coordinate — so both sampling and adaptation cost O(d) per individual. It scales
// to high-dimensional hyperparameter spaces where full-covariance CMA-ES/xNES become expensive, at the
// price of assuming (approximate) coordinate independence. Like `Cma`/`De`, SNES has no Optuna counterpart,
// so this is a native implementation validated by convergence; the RNG is NumSharp's NumPy-exact MT19937.
// SNES is UNBOUNDED (never clamps); `SnesSampler` clamps each candidate to the transformed box.

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace OptunaSharp.Samplers.EvolutionStrategy;

/// <summary>
///     A Separable-NES optimizer (Wierstra 2014, Algorithm 7) with the per-generation ask/tell interface
///     the shared <see cref="EvolutionStrategySampler" /> drives. The search distribution is a diagonal
///     Gaussian: per-coordinate mean <c>μ</c> and scale <c>σ</c>. Each <see cref="Ask" /> proposes
///     <c>μ + σ ⊙ s</c> for <c>s ~ N(0, I)</c>; <see cref="Tell" /> sorts the population, forms the rank
///     utilities, and takes natural-gradient steps: <c>μ ← μ + η_μ · σ ⊙ ∇_μ</c> and
///     <c>σ ← σ ⊙ exp(η_σ/2 · ∇_σ)</c>, with <c>∇_μ = Σ u_k s_k</c> and <c>∇_σ = Σ u_k (s_k² − 1)</c>.
///
///     <para>Because σ is adapted per coordinate, SNES both refines toward and expands away from the optimum
///     as the landscape warrants — so it converges to high precision without the fixed-σ floor that limits
///     bare OpenAI-ES. <b>Orientation:</b> the optimizer minimizes; the best (lowest) value earns the
///     largest utility.</para>
/// </summary>
internal sealed class SnesOptimizer : IEvolutionStrategyOptimizer
{
    private const double SigmaEps = 1e-300;   // keeps σ strictly positive so the (x−μ)/σ reconstruction is finite

    private readonly int _nDim;
    private readonly int _popsize;
    private readonly double _etaMu;
    private readonly double _etaSigma;
    private readonly uint _seed;

    private double[] _mean;    // per-coordinate distribution mean μ
    private double[] _sigma;   // per-coordinate scale σ (adapted multiplicatively)
    private int _generation;

    private readonly NumPyRandom _rng;

    /// <param name="mean">Initial per-coordinate mean μ (the transformed box center); copied defensively.</param>
    /// <param name="sigma">Initial per-coordinate scale σ; copied defensively (each entry must be positive).</param>
    /// <param name="etaMu">Mean learning rate η_μ (Table 1 default 1).</param>
    /// <param name="etaSigma">Scale learning rate η_σ (Table 1 default <c>(3 + ln d)/(5√d)</c>).</param>
    /// <param name="popsize">Population size λ per generation (Table 1 default <c>4 + ⌊3 ln d⌋</c>).</param>
    /// <param name="seed">Base seed for the per-generation noise streams.</param>
    /// <summary>Initializes the diagonal Gaussian for Separable NES.</summary>
    /// <exception cref="System.ArgumentException">The dimension is non-positive, the mean and σ lengths differ, or the population is below two.</exception>
    internal SnesOptimizer(double[] mean, double[] sigma, double etaMu, double etaSigma, int popsize, uint seed)
    {
        if (mean is null || mean.Length == 0) throw new ArgumentException("mean must be non-empty.", nameof(mean));
        if (sigma is null || sigma.Length != mean.Length)
            throw new ArgumentException("sigma must have one entry per coordinate.", nameof(sigma));
        if (popsize < 2) throw new ArgumentException("Population size must be at least 2.", nameof(popsize));

        _nDim = mean.Length;
        _popsize = popsize;
        _etaMu = etaMu;
        _etaSigma = etaSigma;
        _seed = seed;

        _mean = (double[])mean.Clone();
        _sigma = (double[])sigma.Clone();
        _generation = 0;

        _rng = np.random.RandomState();
    }

    /// <inheritdoc />
    public int Dim => _nDim;
    /// <inheritdoc />
    public int PopulationSize => _popsize;
    /// <inheritdoc />
    public int Generation => _generation;

    /// <summary>A defensive copy of the current per-coordinate scale σ (exposed for tests/diagnostics).</summary>
    /// <returns>The current σ vector.</returns>
    internal double[] SigmaSnapshot() => (double[])_sigma.Clone();

    /// <inheritdoc />
    /// <exception cref="System.ArgumentOutOfRangeException"><paramref name="index" /> is outside the current generation.</exception>
    public double[] Ask(int index)
    {
        if (index < 0 || index >= _popsize) throw new ArgumentOutOfRangeException(nameof(index));

        // Deterministic reseed: the whole generation's noise is a pure function of (seed, generation), so
        // any trial can reproduce any row.
        _rng.seed(EsSampling.Combine(_seed, _generation));
        double[][] s = EsSampling.StandardNormalMatrix(_rng, _popsize, _nDim);
        double[] si = s[index];
        var x = new double[_nDim];
        for (int j = 0; j < _nDim; j++)
            x[j] = _mean[j] + _sigma[j] * si[j];   // candidate = μ + σ ⊙ s
        return x;
    }

    /// <inheritdoc />
    /// <exception cref="System.ArgumentException">The number of solutions differs from the population size.</exception>
    public void Tell(IReadOnlyList<(double[] X, double Value)> solutions)
    {
        if (solutions.Count != _popsize)
            throw new ArgumentException($"Must tell exactly {_popsize} solutions for one generation.", nameof(solutions));

        // Reconstruct each standardized perturbation s = (x − μ)/σ from the actually-evaluated candidate.
        var s = new double[_popsize][];
        for (int i = 0; i < _popsize; i++)
        {
            var row = new double[_nDim];
            for (int j = 0; j < _nDim; j++)
                row[j] = (solutions[i].X[j] - _mean[j]) / _sigma[j];
            s[i] = row;
        }

        // Rank ascending by objective (best = lowest); NaN sinks to the worst rank. Utilities are then
        // aligned by rank so the best individual carries the largest weight.
        int[] order = Enumerable.Range(0, _popsize)
            .OrderBy(i => double.IsNaN(solutions[i].Value) ? double.PositiveInfinity : solutions[i].Value)
            .ToArray();
        double[] utilities = EsFitnessShaping.NesUtilities(_popsize);

        // Natural-gradient accumulators: ∇_μ = Σ u_k s_k, ∇_σ = Σ u_k (s_k² − 1), element-wise.
        var gradMu = new double[_nDim];
        var gradSigma = new double[_nDim];
        for (int r = 0; r < _popsize; r++)
        {
            double u = utilities[r];
            double[] sr = s[order[r]];
            for (int j = 0; j < _nDim; j++)
            {
                gradMu[j] += u * sr[j];
                gradSigma[j] += u * (sr[j] * sr[j] - 1.0);
            }
        }

        // Apply the updates: additive on μ (scaled by σ), multiplicative (exponential-map) on σ.
        for (int j = 0; j < _nDim; j++)
        {
            _mean[j] += _etaMu * _sigma[j] * gradMu[j];
            _sigma[j] *= Math.Exp(0.5 * _etaSigma * gradSigma[j]);
            if (_sigma[j] < SigmaEps) _sigma[j] = SigmaEps;   // never let a coordinate collapse to exactly 0
        }

        _generation++;
    }

    /// <summary>Default scale learning rate η_σ = <c>(3 + ln d)/(5√d)</c> (Wierstra 2014, Table 1, SNES).</summary>
    /// <param name="d">Decision-space dimension.</param>
    /// <returns>The dimension-scaled default σ learning rate.</returns>
    internal static double DefaultEtaSigma(int d) => (3.0 + Math.Log(d)) / (5.0 * Math.Sqrt(d));

    /// <summary>Default population size λ = <c>4 + ⌊3 ln d⌋</c> (Wierstra 2014, Table 1), floored at 4.</summary>
    /// <param name="d">Decision-space dimension.</param>
    /// <returns>The dimension-scaled default population size (at least 4).</returns>
    internal static int DefaultPopulationSize(int d) => Math.Max(4, 4 + (int)Math.Floor(3.0 * Math.Log(d)));

    // ---- state serialization (own JSON; mirrors the De/LmCma pattern) -----------------------------

    /// <summary>Captures constructor settings and evolving state in a serializable object.</summary>
    /// <returns>A serializable snapshot of the optimizer.</returns>
    internal SnesState ToState() => new()
    {
        NDim = _nDim,
        Popsize = _popsize,
        EtaMu = _etaMu,
        EtaSigma = _etaSigma,
        Seed = _seed,
        Mean = (double[])_mean.Clone(),
        Sigma = (double[])_sigma.Clone(),
        Generation = _generation,
    };

    /// <summary>Recreates the optimizer, restoring the mean, per-coordinate σ, and generation counter.</summary>
    /// <param name="s">Serialized optimizer state.</param>
    /// <returns>An optimizer restored from the supplied snapshot.</returns>
    /// <exception cref="System.ArgumentException">The stored settings are rejected by the constructor.</exception>
    internal static SnesOptimizer FromState(SnesState s)
    {
        var opt = new SnesOptimizer(s.Mean, s.Sigma, s.EtaMu, s.EtaSigma, s.Popsize, s.Seed);
        opt._generation = s.Generation;
        return opt;
    }
}

/// <summary>Serialized evolving state of a <see cref="SnesOptimizer" /> (own JSON; see <c>DeState</c>).</summary>
internal sealed class SnesState
{
    /// <summary>Serialized number of numerical decision coordinates.</summary>
    public int NDim { get; set; }
    /// <summary>Serialized population size per generation.</summary>
    public int Popsize { get; set; }
    /// <summary>Serialized mean learning rate η_μ.</summary>
    public double EtaMu { get; set; }
    /// <summary>Serialized scale learning rate η_σ.</summary>
    public double EtaSigma { get; set; }
    /// <summary>Serialized base seed for the per-generation noise streams.</summary>
    public uint Seed { get; set; }
    /// <summary>Serialized per-coordinate mean μ.</summary>
    public double[] Mean { get; set; } = Array.Empty<double>();
    /// <summary>Serialized per-coordinate scale σ.</summary>
    public double[] Sigma { get; set; } = Array.Empty<double>();
    /// <summary>Serialized generation/update counter.</summary>
    public int Generation { get; set; }
}
