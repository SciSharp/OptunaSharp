// Sampler for Separable Natural Evolution Strategies (SNES) — the O(d) diagonal NES of Wierstra et al.
// (JMLR 2014, Algorithm 7). Drives the `SnesOptimizer` through the Optuna study protocol via the shared
// `EvolutionStrategySampler` base. SNES is NOT part of the Optuna ecosystem, so — like
// DifferentialEvolutionSampler / LmCmaSampler — this is a native implementation validated by convergence,
// on NumSharp's NumPy-exact MT19937 (never System.Random).

using System;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers.EvolutionStrategy;

namespace OptunaSharp.Samplers;

/// <summary>
///     A sampler driving Separable Natural Evolution Strategies (Wierstra et al. 2014): a diagonal Gaussian
///     with a per-coordinate mean and scale, adapted by natural-gradient steps. Because σ is adapted per
///     coordinate, SNES converges to high precision without the fixed-σ floor of <see cref="OpenAiEsSampler" />,
///     while costing only <c>O(d)</c> per sample — the practical NES for higher-dimensional continuous
///     hyperparameter spaces where full-covariance <see cref="CmaEsSampler" /> becomes expensive. The price
///     is an assumption of (approximate) coordinate independence.
///
///     <para><b>Unsupported parameters.</b> Categorical parameters and single-valued distributions are
///     sampled by the independent sampler, as in <see cref="CmaEsSampler" />. Dynamic search spaces fall
///     back to independent sampling with a warning.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(sampler: new SnesSampler(seed: 42));
///     study.Optimize(t =>
///     {
///         double s = 0;
///         for (int i = 0; i &lt; 10; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
///         return s;
///     }, nTrials: 500);
///     </code>
/// </example>
public sealed class SnesSampler : EvolutionStrategySampler
{
    private const string Tag = "snes";

    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers.snes");

    private readonly double _sigma;
    private readonly double _etaMu;
    private readonly double? _etaSigma;
    private readonly int? _popsize;

    /// <param name="nStartupTrials">Independent sampling is used until this many trials complete.</param>
    /// <param name="independentSampler">Sampler for parameters outside the relative search space (default <see cref="RandomSampler" />).</param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently.</param>
    /// <param name="seed">Random seed for SNES.</param>
    /// <param name="sigma">Initial per-coordinate scale σ in the transformed <c>[0,1]</c> space (default 0.3).</param>
    /// <param name="etaMu">Mean learning rate η_μ (Table 1 default 1.0).</param>
    /// <param name="etaSigma">Scale learning rate η_σ; <c>null</c> uses the Table 1 default <c>(3 + ln d)/(5√d)</c>.</param>
    /// <param name="popsize">Population size λ per generation; <c>null</c> uses the Table 1 default <c>4 + ⌊3 ln d⌋</c>.</param>
    /// <summary>Configures SNES proposals and its independent startup and unsupported-parameter fallback.</summary>
    /// <exception cref="System.ArgumentException">The initial scale σ is not strictly positive.</exception>
    public SnesSampler(
        int nStartupTrials = 1,
        BaseSampler? independentSampler = null,
        bool warnIndependentSampling = true,
        int? seed = null,
        double sigma = 0.3,
        double etaMu = 1.0,
        double? etaSigma = null,
        int? popsize = null)
        : base(nStartupTrials, independentSampler, warnIndependentSampling, seed)
    {
        if (sigma <= 0) throw new ArgumentException("sigma must be strictly positive.", nameof(sigma));
        _sigma = sigma;
        _etaMu = etaMu;
        _etaSigma = etaSigma;
        _popsize = popsize;
    }

    /// <inheritdoc />
    protected override string AttrPrefix => "snes:";

    /// <inheritdoc />
    protected override Logger SamplerLogger => Logger;

    /// <inheritdoc />
    private protected override IEvolutionStrategyOptimizer CreateOptimizer(double[] boxCenter, int nDim, uint seed)
    {
        int popsize = _popsize ?? SnesOptimizer.DefaultPopulationSize(nDim);
        double etaSigma = _etaSigma ?? SnesOptimizer.DefaultEtaSigma(nDim);
        // A per-coordinate σ vector seeded from the single scalar initial scale.
        var sigmaVec = new double[nDim];
        for (int i = 0; i < nDim; i++) sigmaVec[i] = _sigma;
        return new SnesOptimizer(boxCenter, sigmaVec, _etaMu, etaSigma, popsize, seed);
    }

    /// <inheritdoc />
    /// <remarks>
    ///     Written through <see cref="OptimizerStateJson" />: should a diverging run push the mean or the
    ///     per-coordinate step sizes to ±inf or NaN, the state is stored as named literals instead of failing the ask.
    /// </remarks>
    private protected override string SerializeOptimizer(IEvolutionStrategyOptimizer optimizer)
        => Tag + "\n" + OptimizerStateJson.Serialize(((SnesOptimizer)optimizer).ToState());

    /// <inheritdoc />
    /// <exception cref="System.NotSupportedException">The serialized optimizer tag is not this sampler's tag.</exception>
    /// <exception cref="System.Text.Json.JsonException">The persisted payload is not valid JSON for the state type, or is the literal null.</exception>
    /// <exception cref="System.ArgumentOutOfRangeException">The stored text has no <c>&lt;tag&gt;\n</c> line.</exception>
    /// <remarks>Parses the JSON only when the text differs from the last one parsed (<see cref="_parsedState" />); a new optimizer is built from the state every time.</remarks>
    private protected override IEvolutionStrategyOptimizer DeserializeOptimizer(string optimizerStr)
    {
        // The same stored text as the last parse: the same state, without parsing the JSON again (see _parsedState).
        // The whole text, tag included, is the key, so a hit has passed the tag check before.
        if (_parsedState is { } parsed && string.Equals(parsed.Text, optimizerStr, System.StringComparison.Ordinal))
            return SnesOptimizer.FromState(parsed.State);

        int nl = optimizerStr.IndexOf('\n');
        string tag = optimizerStr[..nl];
        if (tag != Tag)
            throw new NotSupportedException($"Unexpected optimizer state tag '{tag}' for SnesSampler.");
        SnesState state = OptimizerStateJson.Deserialize<SnesState>(optimizerStr[(nl + 1)..]);
        _parsedState = (optimizerStr, state);
        return SnesOptimizer.FromState(state);
    }

    /// <summary>
    ///     The last stored optimizer text this instance parsed, and the state parsed from it (plan 15, as
    ///     <c>DifferentialEvolutionSampler</c> keeps it): the text changes once per generation, and every ask in between
    ///     parsed the same JSON again. Guarded by the sampler's monitor, like every hook; a per-worker instance has its
    ///     own.
    /// </summary>
    /// <remarks>
    ///     Safe to share across asks because FromState never changes the state it reads: the <see cref="SnesOptimizer" /> constructor clones the mean and the step sizes, and the generation is a number. A parse that throws
    ///     stores nothing, so the next ask parses (and throws) again, as before.
    /// </remarks>
    private (string Text, SnesState State)? _parsedState;
}
