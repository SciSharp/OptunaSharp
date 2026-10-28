// OpenAI Evolution Strategies (OpenAI-ES) — the isotropic-Gaussian NES of Salimans et al., ask/tell core.
//
// Reference:
//   Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever.
//     "Evolution Strategies as a Scalable Alternative to Reinforcement Learning." arXiv:1703.03864 (2017).
//     Algorithm 1 (the estimator θ ← θ + α·1/(nσ)·Σ Fᵢ εᵢ) plus the §2.1 refinements: antithetic
//     (mirrored) sampling, rank-based fitness shaping, and weight decay.
//
// OpenAI-ES has no `cmaes`/Optuna/OptunaHub counterpart, so — like `LmCma` / `De` — this is a native
// implementation validated by convergence, with no upstream trajectory to bit-match. It follows the
// widely-used `estool` / `evolution-strategies-starter` recipe, which sharpens the bare Algorithm-1
// estimator into a practical optimizer: centered-rank fitness shaping, a z-scored gradient, an Adam step
// on the mean, and a slowly annealed σ. The RNG is NumSharp's NumPy-exact MT19937 `RandomState` (the
// project's "never System.Random" discipline), so the whole trajectory is byte-reproducible against a
// pinned numpy reference. The optimizer is UNBOUNDED (it never clamps); the bounded caller
// (`OpenAiEsSampler`) clamps each candidate to the transformed box, exactly the concession `Cma`/`De`
// document for the per-trial protocol.

using System;
using System.Collections.Generic;
using NumSharp;

namespace OptunaSharp.Samplers.EvolutionStrategy;

/// <summary>
///     An OpenAI-ES optimizer (Salimans 2017) with the per-generation ask/tell interface the shared
///     <see cref="EvolutionStrategySampler" /> drives. The search distribution is an isotropic Gaussian
///     <c>N(mean, σ²I)</c>; each <see cref="Ask" /> proposes <c>mean + σ·εₖ</c> and <see cref="Tell" />
///     estimates the natural gradient of the mean from the shaped returns and takes an Adam step on
///     <c>mean</c>.
///
///     <para><b>Deviations from the bare paper estimator, following the canonical <c>estool</c>
///     implementation (documented so a reader can diff against it):</b></para>
///     <list type="bullet">
///       <item><b>Antithetic (mirrored) sampling</b> (paper §2.1): the population is drawn as pairs
///         <c>ε, −ε</c> to halve the estimator variance; this requires an <b>even</b> population size.</item>
///       <item><b>Centered-rank fitness shaping</b> (paper §2.1): raw returns are replaced by ranks in
///         <c>[-0.5, 0.5]</c>, then z-scored, before forming the gradient — making the step invariant to
///         monotone rescalings of the objective and robust to outliers.</item>
///       <item><b>Adam</b> on the estimated gradient (not plain SGD) for a well-conditioned step.</item>
///       <item><b>Annealed σ and learning rate</b>: geometric decay toward floors. The paper keeps σ
///         fixed (an RL choice); annealing lets the optimizer refine to a small neighborhood of the
///         optimum, which matters for hyperparameter search. Set <c>sigmaDecay = 1.0</c> to recover the
///         paper's fixed σ.</item>
///     </list>
///
///     <para><b>Orientation.</b> The optimizer minimizes: <see cref="Tell" /> receives
///     minimization-oriented values and internally shapes the negated value as the maximization "reward".</para>
/// </summary>
internal sealed class OpenEsOptimizer : IEvolutionStrategyOptimizer
{
    private const double StdEps = 1e-12;   // guards the z-score against a zero-variance shaped population

    private readonly int _nDim;
    private readonly int _popsize;
    private readonly bool _mirrored;
    private readonly double _weightDecay;
    private readonly double _sigmaDecay;
    private readonly double _sigmaLimit;
    private readonly double _learningRateDecay;
    private readonly double _learningRateLimit;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _adamEpsilon;
    private readonly uint _seed;

    private double[] _mean;             // θ — the distribution mean, ascended by the Adam step
    private double _sigma;              // isotropic perturbation scale (annealed)
    private double _learningRate;       // Adam base step size (annealed)
    private double[] _m;                // Adam first-moment accumulator
    private double[] _v;                // Adam second-moment accumulator
    private int _adamT;                 // Adam timestep (for bias correction)
    private int _generation;

    // Reseeded deterministically from (_seed, _generation) on every Ask, so a generation's whole noise
    // matrix is a pure function of the persisted state — never itself serialized.
    private readonly NumPyRandom _rng;

    /// <param name="mean">Initial distribution mean (the transformed box center); copied defensively.</param>
    /// <param name="sigma">Initial isotropic perturbation scale σ (in the transformed <c>[0,1]</c> space).</param>
    /// <param name="learningRate">Adam base step size α applied to the estimated mean-gradient.</param>
    /// <param name="popsize">Population size per generation; must be even when <paramref name="mirrored" /> is set.</param>
    /// <param name="mirrored">Use antithetic sampling (pairs <c>ε, −ε</c>) to reduce estimator variance.</param>
    /// <param name="weightDecay">L2 pull on candidates toward the origin (paper §2.1); <c>0</c> disables it.</param>
    /// <param name="sigmaDecay">Per-generation multiplicative σ decay (<c>1.0</c> keeps σ fixed as in the paper).</param>
    /// <param name="sigmaLimit">Lower floor below which σ stops decaying.</param>
    /// <param name="learningRateDecay">Per-generation multiplicative learning-rate decay (<c>1.0</c> keeps it fixed).</param>
    /// <param name="learningRateLimit">Lower floor below which the learning rate stops decaying.</param>
    /// <param name="beta1">Adam first-moment decay coefficient.</param>
    /// <param name="beta2">Adam second-moment decay coefficient.</param>
    /// <param name="adamEpsilon">Adam numerical-stability term added under the square root.</param>
    /// <param name="seed">Base seed for the per-generation noise streams.</param>
    /// <summary>Initializes the isotropic Gaussian and its Adam state for OpenAI-ES.</summary>
    /// <exception cref="System.ArgumentException">The dimension is non-positive, the population is below two, or mirrored sampling is requested with an odd population.</exception>
    internal OpenEsOptimizer(
        double[] mean, double sigma, double learningRate, int popsize, bool mirrored,
        double weightDecay, double sigmaDecay, double sigmaLimit,
        double learningRateDecay, double learningRateLimit,
        double beta1, double beta2, double adamEpsilon, uint seed)
    {
        if (mean is null || mean.Length == 0) throw new ArgumentException("mean must be non-empty.", nameof(mean));
        if (popsize < 2) throw new ArgumentException("Population size must be at least 2.", nameof(popsize));
        // Antithetic sampling forms popsize/2 mirrored pairs, so an odd population cannot be halved cleanly.
        if (mirrored && popsize % 2 != 0)
            throw new ArgumentException("Mirrored (antithetic) sampling requires an even population size.", nameof(popsize));

        _nDim = mean.Length;
        _popsize = popsize;
        _mirrored = mirrored;
        _weightDecay = weightDecay;
        _sigmaDecay = sigmaDecay;
        _sigmaLimit = sigmaLimit;
        _learningRateDecay = learningRateDecay;
        _learningRateLimit = learningRateLimit;
        _beta1 = beta1;
        _beta2 = beta2;
        _adamEpsilon = adamEpsilon;
        _seed = seed;

        _mean = (double[])mean.Clone();
        _sigma = sigma;
        _learningRate = learningRate;
        _m = new double[_nDim];
        _v = new double[_nDim];
        _adamT = 0;
        _generation = 0;

        _rng = np.random.RandomState();
    }

    /// <inheritdoc />
    public int Dim => _nDim;
    /// <inheritdoc />
    public int PopulationSize => _popsize;
    /// <inheritdoc />
    public int Generation => _generation;

    /// <summary>The current isotropic perturbation scale σ (exposed for tests/diagnostics).</summary>
    internal double Sigma => _sigma;

    /// <inheritdoc />
    /// <exception cref="System.ArgumentOutOfRangeException"><paramref name="index" /> is negative.</exception>
    public double[] Ask(int index)
    {
        if (index < 0) throw new ArgumentOutOfRangeException(nameof(index));

        double[] e = index < _popsize ? SampleEpsilon()[index] : SampleExtraEpsilon(index);
        var x = new double[_nDim];
        for (int j = 0; j < _nDim; j++)
            x[j] = _mean[j] + _sigma * e[j];   // candidate = mean + σ·ε
        return x;
    }

    /// <summary>
    ///     Re-derives the whole generation's perturbation matrix deterministically from
    ///     <c>(_seed, _generation)</c>. With antithetic sampling the first half is drawn and the second
    ///     half is its negation (<c>ε, −ε</c> pairs); otherwise all rows are independent draws. The layout
    ///     matches the <c>estool</c> <c>ask()</c> exactly, so it is byte-reproducible against numpy.
    /// </summary>
    /// <returns>The <c>popsize × dim</c> perturbation matrix for the current generation.</returns>
    private double[][] SampleEpsilon()
    {
        // A fresh reseed makes the draw a pure function of (seed, generation): every trial of a generation
        // reconstructs the identical matrix and simply indexes its own row.
        _rng.seed(EsSampling.Combine(_seed, _generation));

        if (!_mirrored)
            return EsSampling.StandardNormalMatrix(_rng, _popsize, _nDim);

        int half = _popsize / 2;
        double[][] h = EsSampling.StandardNormalMatrix(_rng, half, _nDim);
        var epsilon = new double[_popsize][];
        for (int i = 0; i < half; i++)
        {
            epsilon[i] = h[i];
            // Mirror: the paired row is the exact negation, halving the estimator's variance.
            var neg = new double[_nDim];
            for (int j = 0; j < _nDim; j++) neg[j] = -h[i][j];
            epsilon[half + i] = neg;
        }
        return epsilon;
    }

    /// <summary>
    ///     The perturbation of an extra candidate, past the generation's members (see
    ///     <see cref="IEvolutionStrategyOptimizer.Ask" />): an independent row of the same <c>(seed, generation)</c>
    ///     stream, drawn after the rows the members use. Unmirrored, so it has no antithetic partner; the update does
    ///     not pair candidates, it recovers each one's noise from its evaluated vector.
    /// </summary>
    /// <param name="index">The candidate's index; at least the population size.</param>
    /// <returns>The candidate's perturbation, of length <c>dim</c>.</returns>
    private double[] SampleExtraEpsilon(int index)
    {
        _rng.seed(EsSampling.Combine(_seed, _generation));
        // The members draw `half` rows when mirrored (their negations need no draws), else all `_popsize` rows.
        int memberRows = _mirrored ? _popsize / 2 : _popsize;
        int row = memberRows + (index - _popsize);
        return EsSampling.StandardNormalMatrix(_rng, row + 1, _nDim)[row];
    }

    /// <inheritdoc />
    /// <exception cref="System.ArgumentException">The number of solutions differs from the population size.</exception>
    public void Tell(IReadOnlyList<(double[] X, double Value)> solutions)
    {
        if (solutions.Count != _popsize)
            throw new ArgumentException($"Must tell exactly {_popsize} solutions for one generation.", nameof(solutions));

        // Reconstruct each perturbation from the actually-evaluated candidate using the CURRENT σ (the σ
        // that produced this generation — annealing happens only at the end of this method).
        var epsilon = new double[_popsize][];
        var rawReward = new double[_popsize];
        for (int i = 0; i < _popsize; i++)
        {
            var e = new double[_nDim];
            for (int j = 0; j < _nDim; j++)
                e[j] = (solutions[i].X[j] - _mean[j]) / _sigma;
            epsilon[i] = e;

            double value = solutions[i].Value;
            // Minimization → maximization: reward = −value, so the best (lowest) value earns the highest
            // reward. NaN is treated as the worst possible reward so it sinks to rank 0 without poisoning
            // the sort.
            rawReward[i] = double.IsNaN(value) ? double.NegativeInfinity : -value;
        }

        // Fitness shaping: replace returns with centered ranks in [-0.5, 0.5].
        double[] reward = EsFitnessShaping.CenteredRanks(rawReward);

        // Optional weight decay: an L2 penalty on the candidate norm (paper §2.1). Off by default because
        // in the transformed [0,1] box the origin is a corner, so a nonzero decay biases the search toward
        // that corner rather than toward a neutral center.
        if (_weightDecay > 0)
        {
            for (int i = 0; i < _popsize; i++)
            {
                double meanSq = 0;
                for (int j = 0; j < _nDim; j++) meanSq += solutions[i].X[j] * solutions[i].X[j];
                meanSq /= _nDim;
                reward[i] += -_weightDecay * meanSq;
            }
        }

        // Z-score the shaped reward (estool `normalized_reward`): centers the estimator and normalizes its
        // scale, decoupling the effective step size from the population size and shaping choice.
        double rewardMean = 0;
        for (int i = 0; i < _popsize; i++) rewardMean += reward[i];
        rewardMean /= _popsize;
        double variance = 0;
        for (int i = 0; i < _popsize; i++) { double d = reward[i] - rewardMean; variance += d * d; }
        variance /= _popsize;
        double std = Math.Sqrt(variance);

        var normalized = new double[_popsize];
        if (std > StdEps)
            for (int i = 0; i < _popsize; i++) normalized[i] = (reward[i] - rewardMean) / std;
        // else: a degenerate (all-equal) generation carries no directional signal — leave the gradient at
        // zero so the mean does not drift on numerical noise.

        // Estimated gradient of the mean: g = 1/(popsize·σ) · Σᵢ normalizedᵢ · εᵢ.
        var changeMu = new double[_nDim];
        double scale = 1.0 / (_popsize * _sigma);
        for (int j = 0; j < _nDim; j++)
        {
            double acc = 0;
            for (int i = 0; i < _popsize; i++) acc += normalized[i] * epsilon[i][j];
            changeMu[j] = scale * acc;
        }

        // Adam minimizes, so ascending the reward means stepping on the negated gradient (estool passes
        // `-change_mu`). Uses the current learning rate; the rate is annealed only afterward.
        var globalGradient = new double[_nDim];
        for (int j = 0; j < _nDim; j++) globalGradient[j] = -changeMu[j];
        AdamStep(globalGradient);

        // Anneal σ and the learning rate toward their floors (estool). With the defaults these shrink the
        // exploration radius as the search settles.
        if (_sigma > _sigmaLimit) _sigma *= _sigmaDecay;
        if (_learningRate > _learningRateLimit) _learningRate *= _learningRateDecay;

        _generation++;
    }

    /// <summary>
    ///     One Adam update of the mean on <paramref name="globalGradient" /> (a descent gradient). Mirrors
    ///     the <c>estool</c> <c>Adam</c>: bias-corrected first/second moments, step
    ///     <c>−a·m̂/(√v̂ + ε)</c> added to the mean.
    /// </summary>
    /// <param name="globalGradient">The gradient to descend (the negated reward-gradient), length <see cref="Dim" />.</param>
    private void AdamStep(double[] globalGradient)
    {
        _adamT++;
        // Bias-corrected effective step size a = α·√(1−β2ᵗ)/(1−β1ᵗ).
        double a = _learningRate * Math.Sqrt(1 - Math.Pow(_beta2, _adamT)) / (1 - Math.Pow(_beta1, _adamT));
        for (int j = 0; j < _nDim; j++)
        {
            double g = globalGradient[j];
            _m[j] = _beta1 * _m[j] + (1 - _beta1) * g;
            _v[j] = _beta2 * _v[j] + (1 - _beta2) * g * g;
            double step = -a * _m[j] / (Math.Sqrt(_v[j]) + _adamEpsilon);
            _mean[j] += step;
        }
    }

    /// <summary>
    ///     Default population size <c>max(8, 4 + ⌊3 ln d⌋)</c> (the CMA/NES λ formula floored at 8 so the
    ///     mean-gradient estimate is not hopelessly noisy), rounded up to even when mirrored so it splits
    ///     into antithetic pairs. Kept modest so a generation completes within an Optuna trial budget.
    /// </summary>
    /// <param name="d">Decision-space dimension.</param>
    /// <param name="mirrored">Whether antithetic sampling (which needs an even population) is used.</param>
    /// <returns>The algorithm default population size for the supplied dimension.</returns>
    internal static int DefaultPopulationSize(int d, bool mirrored)
    {
        int popsize = Math.Max(8, 4 + (int)Math.Floor(3.0 * Math.Log(d)));
        if (mirrored && popsize % 2 != 0) popsize++;
        return popsize;
    }

    // ---- state serialization (own JSON; mirrors the De/LmCma pattern) -----------------------------

    /// <summary>Captures constructor settings and evolving optimizer state in a serializable object.</summary>
    /// <returns>A serializable snapshot of the optimizer.</returns>
    internal OpenEsState ToState() => new()
    {
        NDim = _nDim,
        Popsize = _popsize,
        Mirrored = _mirrored,
        WeightDecay = _weightDecay,
        SigmaDecay = _sigmaDecay,
        SigmaLimit = _sigmaLimit,
        LearningRateDecay = _learningRateDecay,
        LearningRateLimit = _learningRateLimit,
        Beta1 = _beta1,
        Beta2 = _beta2,
        AdamEpsilon = _adamEpsilon,
        Seed = _seed,
        Mean = (double[])_mean.Clone(),
        Sigma = _sigma,
        LearningRate = _learningRate,
        M = (double[])_m.Clone(),
        V = (double[])_v.Clone(),
        AdamT = _adamT,
        Generation = _generation,
    };

    /// <summary>Recreates the optimizer, restoring the mean, σ, Adam moments, and generation counter.</summary>
    /// <param name="s">Serialized optimizer state.</param>
    /// <returns>An optimizer restored from the supplied snapshot.</returns>
    /// <exception cref="System.ArgumentException">The stored settings are rejected by the constructor (e.g. an odd mirrored population).</exception>
    internal static OpenEsOptimizer FromState(OpenEsState s)
    {
        var opt = new OpenEsOptimizer(
            s.Mean, s.Sigma, s.LearningRate, s.Popsize, s.Mirrored,
            s.WeightDecay, s.SigmaDecay, s.SigmaLimit, s.LearningRateDecay, s.LearningRateLimit,
            s.Beta1, s.Beta2, s.AdamEpsilon, s.Seed);
        // Restore the evolving state the constructor zero-initialized.
        opt._m = (double[])s.M.Clone();
        opt._v = (double[])s.V.Clone();
        opt._adamT = s.AdamT;
        opt._generation = s.Generation;
        return opt;
    }
}

/// <summary>Serialized evolving state of an <see cref="OpenEsOptimizer" /> (own JSON; see <c>DeState</c>).</summary>
internal sealed class OpenEsState
{
    /// <summary>Serialized number of numerical decision coordinates.</summary>
    public int NDim { get; set; }
    /// <summary>Serialized population size per generation.</summary>
    public int Popsize { get; set; }
    /// <summary>Serialized antithetic-sampling flag.</summary>
    public bool Mirrored { get; set; }
    /// <summary>Serialized L2 weight-decay coefficient.</summary>
    public double WeightDecay { get; set; }
    /// <summary>Serialized per-generation σ decay factor.</summary>
    public double SigmaDecay { get; set; }
    /// <summary>Serialized σ decay floor.</summary>
    public double SigmaLimit { get; set; }
    /// <summary>Serialized per-generation learning-rate decay factor.</summary>
    public double LearningRateDecay { get; set; }
    /// <summary>Serialized learning-rate decay floor.</summary>
    public double LearningRateLimit { get; set; }
    /// <summary>Serialized Adam first-moment decay coefficient.</summary>
    public double Beta1 { get; set; }
    /// <summary>Serialized Adam second-moment decay coefficient.</summary>
    public double Beta2 { get; set; }
    /// <summary>Serialized Adam numerical-stability term.</summary>
    public double AdamEpsilon { get; set; }
    /// <summary>Serialized base seed for the per-generation noise streams.</summary>
    public uint Seed { get; set; }
    /// <summary>Serialized distribution mean θ.</summary>
    public double[] Mean { get; set; } = Array.Empty<double>();
    /// <summary>Serialized isotropic perturbation scale σ.</summary>
    public double Sigma { get; set; }
    /// <summary>Serialized current Adam base step size.</summary>
    public double LearningRate { get; set; }
    /// <summary>Serialized Adam first-moment accumulator.</summary>
    public double[] M { get; set; } = Array.Empty<double>();
    /// <summary>Serialized Adam second-moment accumulator.</summary>
    public double[] V { get; set; } = Array.Empty<double>();
    /// <summary>Serialized Adam timestep.</summary>
    public int AdamT { get; set; }
    /// <summary>Serialized generation/update counter.</summary>
    public int Generation { get; set; }
}
