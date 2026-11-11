// Sampler for Exponential Natural Evolution Strategies (xNES) — the full-covariance NES of Wierstra et al.
// (JMLR 2014, Algorithm 4). Drives the `XnesOptimizer` through the Optuna study protocol via the shared
// `EvolutionStrategySampler` base. xNES is NOT part of the Optuna ecosystem, so — like
// DifferentialEvolutionSampler / LmCmaSampler — this is a native implementation validated by convergence,
// on NumSharp's NumPy-exact MT19937 (never System.Random) and OpenBLAS-backed linear algebra.

using System;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers.EvolutionStrategy;

namespace OptunaSharp.Samplers;

/// <summary>
///     A sampler driving Exponential Natural Evolution Strategies (Wierstra et al. 2014): a full-covariance
///     Gaussian factored as <c>A = σ·B</c> and adapted through the matrix exponential of a natural-gradient
///     direction — the NES most closely related to CMA-ES's rank-µ update. It captures correlations between
///     parameters that the diagonal <see cref="SnesSampler" /> cannot, at <c>O(d²)</c>–<c>O(d³)</c> cost per
///     generation.
///
///     <para><b>When to prefer alternatives.</b> OptunaSharp already ships mature full-covariance adaptation
///     (<see cref="CmaEsSampler" />, <c>VkdCmaSampler</c>, <c>LmCmaSampler</c>); xNES is provided for
///     completeness of the NES family. For most correlated continuous problems <see cref="CmaEsSampler" />
///     is the better-tuned choice, and <see cref="SnesSampler" /> is cheaper in high dimension.</para>
///
///     <para><b>Unsupported parameters.</b> Categorical parameters and single-valued distributions are
///     sampled by the independent sampler, as in <see cref="CmaEsSampler" />. Dynamic search spaces fall
///     back to independent sampling with a warning.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(sampler: new XnesSampler(seed: 42));
///     study.Optimize(t =>
///     {
///         double x = t.SuggestFloat("x", -5, 5);
///         double y = t.SuggestFloat("y", -5, 5);
///         return (x - 1) * (x - 1) + 100 * (y - x * x) * (y - x * x);   // Rosenbrock
///     }, nTrials: 600);
///     </code>
/// </example>
public sealed class XnesSampler : EvolutionStrategySampler
{
    private const string Tag = "xnes";

    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers.xnes");

    private readonly double _sigma;
    private readonly double _etaMu;
    private readonly double? _etaSigma;
    private readonly double? _etaB;
    private readonly int? _popsize;

    /// <param name="nStartupTrials">Independent sampling is used until this many trials complete.</param>
    /// <param name="independentSampler">Sampler for parameters outside the relative search space (default <see cref="RandomSampler" />).</param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently.</param>
    /// <param name="seed">Random seed for xNES.</param>
    /// <param name="sigma">Initial global step size σ in the transformed <c>[0,1]</c> space (default 0.3).</param>
    /// <param name="etaMu">Mean learning rate η_µ (Table 1 default 1.0).</param>
    /// <param name="etaSigma">Step-size learning rate η_σ; <c>null</c> uses the Table 1 default <c>(9 + 3 ln d)/(5 d √d)</c>.</param>
    /// <param name="etaB">Transform learning rate η_B; <c>null</c> uses the Table 1 default (equal to η_σ).</param>
    /// <param name="popsize">Population size λ per generation; <c>null</c> uses the Table 1 default <c>4 + ⌊3 ln d⌋</c>.</param>
    /// <summary>Configures xNES proposals and its independent startup and unsupported-parameter fallback.</summary>
    /// <exception cref="System.ArgumentException">The initial step size σ is not strictly positive.</exception>
    public XnesSampler(
        int nStartupTrials = 1,
        BaseSampler? independentSampler = null,
        bool warnIndependentSampling = true,
        int? seed = null,
        double sigma = 0.3,
        double etaMu = 1.0,
        double? etaSigma = null,
        double? etaB = null,
        int? popsize = null)
        : base(nStartupTrials, independentSampler, warnIndependentSampling, seed)
    {
        if (sigma <= 0) throw new ArgumentException("sigma must be strictly positive.", nameof(sigma));
        _sigma = sigma;
        _etaMu = etaMu;
        _etaSigma = etaSigma;
        _etaB = etaB;
        _popsize = popsize;
    }

    /// <inheritdoc />
    protected override string AttrPrefix => "xnes:";

    /// <inheritdoc />
    protected override Logger SamplerLogger => Logger;

    /// <inheritdoc />
    private protected override IEvolutionStrategyOptimizer CreateOptimizer(double[] boxCenter, int nDim, uint seed)
    {
        int popsize = _popsize ?? XnesOptimizer.DefaultPopulationSize(nDim);
        double etaSigma = _etaSigma ?? XnesOptimizer.DefaultEtaSigma(nDim);
        double etaB = _etaB ?? etaSigma;   // Table 1: η_B = η_σ by default
        return new XnesOptimizer(boxCenter, _sigma, XnesOptimizer.Identity(nDim), _etaMu, etaSigma, etaB, popsize, seed);
    }

    /// <inheritdoc />
    /// <remarks>
    ///     Written through <see cref="OptimizerStateJson" />: should a diverging run push the mean, step size or
    ///     shape matrix to ±inf or NaN, the state is stored as named literals instead of failing the ask.
    /// </remarks>
    private protected override string SerializeOptimizer(IEvolutionStrategyOptimizer optimizer)
        => Tag + "\n" + OptimizerStateJson.Serialize(((XnesOptimizer)optimizer).ToState());

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
            return XnesOptimizer.FromState(parsed.State);

        int nl = optimizerStr.IndexOf('\n');
        string tag = optimizerStr[..nl];
        if (tag != Tag)
            throw new NotSupportedException($"Unexpected optimizer state tag '{tag}' for XnesSampler.");
        XnesState state = OptimizerStateJson.Deserialize<XnesState>(optimizerStr[(nl + 1)..]);
        _parsedState = (optimizerStr, state);
        return XnesOptimizer.FromState(state);
    }

    /// <summary>
    ///     The last stored optimizer text this instance parsed, and the state parsed from it (plan 15, as
    ///     <c>DifferentialEvolutionSampler</c> keeps it): the text changes once per generation, and every ask in between
    ///     parsed the same JSON again. Guarded by the sampler's monitor, like every hook; a per-worker instance has its
    ///     own.
    /// </summary>
    /// <remarks>
    ///     Safe to share across asks because FromState never changes the state it reads: the <see cref="XnesOptimizer" /> constructor clones the mean, receives the shape matrix as a fresh rectangular copy of the stored rows and clones that too, and the scale and generation are numbers. A parse that throws
    ///     stores nothing, so the next ask parses (and throws) again, as before.
    /// </remarks>
    private (string Text, XnesState State)? _parsedState;
}
