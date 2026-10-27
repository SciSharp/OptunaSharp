// Sampler for OpenAI Evolution Strategies (OpenAI-ES) — the isotropic natural-gradient search of Salimans
// et al. (arXiv:1703.03864). Drives the `OpenEsOptimizer` through the Optuna study protocol via the shared
// `EvolutionStrategySampler` base. OpenAI-ES is NOT part of the Optuna ecosystem (no sampler, no OptunaHub
// package), so — like DifferentialEvolutionSampler / LmCmaSampler — this is a native implementation
// validated by convergence, on NumSharp's NumPy-exact MT19937 (never System.Random).

using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers.EvolutionStrategy;

namespace OptunaSharp.Samplers;

/// <summary>
///     A sampler driving OpenAI Evolution Strategies (Salimans et al. 2017): an isotropic Gaussian
///     <c>N(mean, σ²I)</c> whose mean ascends the natural gradient estimated from a population of mirrored,
///     rank-shaped perturbations, stepped with Adam. It is a robust, embarrassingly-parallel, gradient-free
///     optimizer that complements the covariance model of <see cref="CmaEsSampler" /> and the diagonal NES
///     of <see cref="SnesSampler" />.
///
///     <para><b>Convergence character.</b> OpenAI-ES uses a single global σ. With the defaults σ is annealed
///     only slowly (the <c>estool</c> recipe), so within a typical Optuna budget the search settles to a
///     neighborhood of the optimum rather than to machine precision — the natural regime for noisy,
///     high-dimensional objectives. When you need fine convergence on a smooth objective, prefer
///     <see cref="SnesSampler" /> (per-coordinate adaptive σ) or <see cref="CmaEsSampler" />.</para>
///
///     <para><b>Unsupported parameters.</b> Categorical parameters and single-valued distributions are
///     sampled by the independent sampler, as in <see cref="CmaEsSampler" />. Dynamic search spaces fall
///     back to independent sampling with a warning.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(sampler: new OpenAiEsSampler(seed: 42));
///     study.Optimize(t =>
///     {
///         double s = 0;
///         for (int i = 0; i &lt; 8; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
///         return s;
///     }, nTrials: 600);
///     </code>
/// </example>
public sealed class OpenAiEsSampler : EvolutionStrategySampler
{
    private const string Tag = "openaies";

    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers.openai_es");

    private readonly double _sigma;
    private readonly double _learningRate;
    private readonly int? _popsize;
    private readonly bool _mirrored;
    private readonly double _weightDecay;
    private readonly double _sigmaDecay;
    private readonly double _sigmaLimit;
    private readonly double _learningRateDecay;
    private readonly double _learningRateLimit;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _adamEpsilon;

    /// <param name="nStartupTrials">Independent sampling is used until this many trials complete.</param>
    /// <param name="independentSampler">Sampler for parameters outside the relative search space (default <see cref="RandomSampler" />).</param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently.</param>
    /// <param name="seed">Random seed for OpenAI-ES.</param>
    /// <param name="sigma">Initial isotropic perturbation scale σ in the transformed <c>[0,1]</c> space (default 0.1).</param>
    /// <param name="learningRate">Adam base step size α applied to the estimated mean-gradient (default 0.05).</param>
    /// <param name="popsize">Population size per generation (default <c>max(8, 4 + ⌊3 ln d⌋)</c>, rounded to even when mirrored).</param>
    /// <param name="mirrored">Use antithetic sampling (pairs <c>ε, −ε</c>); reduces variance, requires an even population (default true).</param>
    /// <param name="weightDecay">L2 pull on candidates toward the origin. <b>Off by default</b>: in the transformed
    ///     <c>[0,1]</c> box the origin is a corner, so a nonzero value biases the search toward that corner.</param>
    /// <param name="sigmaDecay">Per-generation multiplicative σ decay; <c>1.0</c> keeps σ fixed as in the paper (default 0.999).</param>
    /// <param name="sigmaLimit">Lower floor below which σ stops decaying (default 0.01).</param>
    /// <param name="learningRateDecay">Per-generation learning-rate decay; <c>1.0</c> keeps it fixed (default 1.0, suited to short budgets).</param>
    /// <param name="learningRateLimit">Lower floor below which the learning rate stops decaying (default 0.001).</param>
    /// <param name="beta1">Adam first-moment decay coefficient (default 0.9).</param>
    /// <param name="beta2">Adam second-moment decay coefficient (default 0.999).</param>
    /// <param name="adamEpsilon">Adam numerical-stability term (default 1e-8).</param>
    /// <summary>Configures OpenAI-ES proposals and its independent startup and unsupported-parameter fallback.</summary>
    public OpenAiEsSampler(
        int nStartupTrials = 1,
        BaseSampler? independentSampler = null,
        bool warnIndependentSampling = true,
        int? seed = null,
        double sigma = 0.1,
        double learningRate = 0.05,
        int? popsize = null,
        bool mirrored = true,
        double weightDecay = 0.0,
        double sigmaDecay = 0.999,
        double sigmaLimit = 0.01,
        double learningRateDecay = 1.0,
        double learningRateLimit = 0.001,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double adamEpsilon = 1e-8)
        : base(nStartupTrials, independentSampler, warnIndependentSampling, seed)
    {
        _sigma = sigma;
        _learningRate = learningRate;
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
    }

    /// <inheritdoc />
    protected override string AttrPrefix => "openaies:";

    /// <inheritdoc />
    protected override Logger SamplerLogger => Logger;

    /// <inheritdoc />
    private protected override IEvolutionStrategyOptimizer CreateOptimizer(double[] boxCenter, int nDim, uint seed)
    {
        int popsize = _popsize ?? OpenEsOptimizer.DefaultPopulationSize(nDim, _mirrored);
        return new OpenEsOptimizer(
            mean: boxCenter,
            sigma: _sigma,
            learningRate: _learningRate,
            popsize: popsize,
            mirrored: _mirrored,
            weightDecay: _weightDecay,
            sigmaDecay: _sigmaDecay,
            sigmaLimit: _sigmaLimit,
            learningRateDecay: _learningRateDecay,
            learningRateLimit: _learningRateLimit,
            beta1: _beta1,
            beta2: _beta2,
            adamEpsilon: _adamEpsilon,
            seed: seed);
    }

    /// <inheritdoc />
    /// <remarks>
    ///     Written through <see cref="OptimizerStateJson" />: should a diverging run push the mean, step size or
    ///     Adam moments to ±inf or NaN, the state is stored as named literals instead of failing the ask.
    /// </remarks>
    private protected override string SerializeOptimizer(IEvolutionStrategyOptimizer optimizer)
        => Tag + "\n" + OptimizerStateJson.Serialize(((OpenEsOptimizer)optimizer).ToState());

    /// <inheritdoc />
    /// <exception cref="System.NotSupportedException">The serialized optimizer tag is not this sampler's tag.</exception>
    /// <exception cref="System.Text.Json.JsonException">The persisted payload is not valid JSON for the state type, or is the literal null.</exception>
    /// <exception cref="System.ArgumentOutOfRangeException">The stored text has no <c>&lt;tag&gt;\n</c> line.</exception>
    private protected override IEvolutionStrategyOptimizer DeserializeOptimizer(string optimizerStr)
    {
        int nl = optimizerStr.IndexOf('\n');
        string tag = optimizerStr[..nl];
        if (tag != Tag)
            throw new System.NotSupportedException($"Unexpected optimizer state tag '{tag}' for OpenAiEsSampler.");
        return OpenEsOptimizer.FromState(OptimizerStateJson.Deserialize<OpenEsState>(optimizerStr[(nl + 1)..]));
    }
}
