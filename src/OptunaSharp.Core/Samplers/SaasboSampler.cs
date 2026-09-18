// Sampler for SAASBO — Sparse Axis-Aligned Subspace Bayesian Optimization (Eriksson & Jankowiak, UAI 2021,
// arXiv:2103.00349). Drives the paper's Algorithm 1 through the Optuna study protocol: a Sobol
// initialization phase, then per-trial a fully-Bayesian (NUTS) or MAP fit of an RBF GP under the
// sparsity-inducing SAAS prior, followed by maximization of the sample-averaged expected improvement.
//
// SAASBO is NOT part of the Optuna ecosystem (Optuna offers it only through the BoTorch integration), so —
// like TurboSampler / RandomEmbeddingSampler — this is a native, paper-faithful implementation validated by
// convergence and by the paper's structural claims (it identifies a sparse relevant subspace), NOT byte-
// exact against BoTorch/NumPyro. All randomness rides NumSharp's MT19937 (LazyRandomState), never
// System.Random, so a fixed seed reproduces the trajectory on a platform. The SAAS surrogate lives in a
// beyond-core GP toolkit forked from the TuRBO/REMBO stack (Samplers/ModelBased/Saas), so this leaves the
// Matérn GPRegressor those samplers and GPSampler depend on untouched (the project rule: clone, don't modify).

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers.ModelBased.Gp;
using OptunaSharp.Samplers.ModelBased.Saas;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     A sampler using SAASBO — high-dimensional Bayesian optimization with sparse axis-aligned subspaces
///     (Eriksson &amp; Jankowiak 2021). Each trial it fits an <b>RBF ARD Gaussian process under the SAAS
///     prior</b> — a half-Cauchy sparsity prior on the per-dimension inverse-squared length scales that
///     "turns off" most dimensions unless the data demands otherwise (automatic relevance determination) —
///     and proposes the next point by maximizing the expected improvement averaged over posterior
///     hyperparameter samples. This is the paper's answer to the curse of dimensionality: work directly in
///     the full space, but let a sparsity-inducing function prior keep the surrogate parsimonious enough to
///     infer from a few hundred queries.
///
///     <para><b>Inference.</b> The default is the No-U-Turn Sampler (<see cref="SaasboInferenceMethod.Nuts" />,
///     the paper's "gold standard"), which averages EI over a bag of <c>L</c> posterior draws;
///     <see cref="SaasboInferenceMethod.Map" /> is a faster point-estimate alternative (Adam over a small
///     shrinkage grid + leave-one-out selection). The single prior hyperparameter is <c>α</c> (global
///     shrinkage, default <c>0.1</c>); every other knob is purely computational.</para>
///
///     <para><b>Cost footgun.</b> SAASBO is deliberately expensive: the paper's default NUTS budget
///     (512 warmup + 256 samples, thinned to 16) runs Hamiltonian Monte Carlo over an <c>O(N³D)</c> gradient
///     per leapfrog step, and the default EI maximization screens <c>Q = 5000</c> Sobol candidates before
///     three L-BFGS-B polishes — appropriate when the objective is expensive to evaluate (the paper targets
///     a few hundred queries) but heavy otherwise. Lower <c>numWarmup</c>/<c>numSamples</c>/<c>numCandidates</c>
///     (or use <see cref="SaasboInferenceMethod.Map" />) to trade fidelity for speed. Best for
///     <c>N ≲ 500</c> observations.</para>
///
///     <para><b>Continuous / integer only.</b> The SAAS GP models a continuous <c>[0, 1]^D</c> box, so the
///     relative search space is Float/Int only; categorical and single-valued parameters are delegated to
///     the independent sampler, exactly as in <see cref="TurboSampler" /> / <see cref="CmaEsSampler" />.
///     Dynamic search spaces fall back to independent sampling with a warning. Single-objective only.</para>
///
///     <para><b>State is reconstructed, not persisted.</b> Like <see cref="TurboSampler" />, SAASBO stores no
///     optimizer blob: each trial re-fits the GP to the completed observations, so a study resumes from
///     storage with no serialized state.</para>
/// </summary>
/// <example>
///     <code>
///     // A 2-D objective embedded in 20 dimensions — SAASBO discovers the two relevant axes.
///     var study = Optuna.CreateStudy(sampler: new SaasboSampler(
///         seed: 0, nStartupTrials: 8,
///         // light budget for a fast demo (paper defaults are far heavier):
///         numWarmup: 64, numSamples: 64, thinning: 8, numCandidates: 256));
///     study.Optimize(t =>
///     {
///         double a = t.SuggestFloat("x0", -5, 10), b = t.SuggestFloat("x1", 0, 15);
///         for (int i = 2; i &lt; 20; i++) t.SuggestFloat($"x{i}", 0, 1);   // irrelevant dimensions
///         return (b - 5.1 / (4 * Math.PI * Math.PI) * a * a + 5 / Math.PI * a - 6) * 2;  // Branin-like
///     }, nTrials: 40);
///     </code>
/// </example>
public sealed class SaasboSampler : BaseSampler
{
    private const double Eps = 1e-10;
    private const uint InitSeedSalt = 0x5aa5_b0b0u;   // de-correlates the init Sobol stream from the per-trial streams

    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers.saasbo");

    private readonly BaseSampler _independentSampler;
    private readonly IntersectionSearchSpace _searchSpace;
    private readonly bool _warnIndependentSampling;
    private readonly uint _baseSeed;
    private readonly int _nStartupTrials;
    private readonly SaasboConfig _config;

    /// <param name="independentSampler">Sampler for parameters outside the relative search space and for the
    ///     Sobol-init fallback (default <see cref="RandomSampler" />).</param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently.</param>
    /// <param name="seed">Random seed; a fixed seed reproduces the trajectory on a platform.</param>
    /// <param name="nStartupTrials">Sobol initialization trials before the SAAS GP takes over (paper's <c>m</c>;
    ///     default 10, the paper's Branin/Rosenbrock value). Must be at least 1.</param>
    /// <param name="inferenceMethod">NUTS (paper default) or MAP.</param>
    /// <param name="alpha">The SAAS global-shrinkage hyperparameter <c>α &gt; 0</c> (NUTS only; MAP ignores it).
    ///     Smaller is more conservative (prefers smaller subspaces); the paper's default is <c>0.1</c>.</param>
    /// <param name="numWarmup">NUTS warmup iterations (paper: 512). Ignored for MAP.</param>
    /// <param name="numSamples">NUTS post-warmup iterations before thinning (paper: 256). Ignored for MAP.</param>
    /// <param name="thinning">NUTS thinning factor — every <c>thinning</c>-th post-warmup draw is retained
    ///     (paper: 16, giving <c>L = 16</c> samples). Ignored for MAP.</param>
    /// <param name="maxTreeDepth">NUTS maximum tree depth (paper: 6). Ignored for MAP.</param>
    /// <param name="targetAccept">NUTS dual-averaging target acceptance rate (0.8). Ignored for MAP.</param>
    /// <param name="mapSteps">MAP Adam iterations per shrinkage value (paper: 1500). Ignored for NUTS.</param>
    /// <param name="numCandidates">EI-maximization Sobol screen size <c>Q</c> (paper: 5000).</param>
    /// <param name="numRestarts">EI-maximization L-BFGS-B restarts <c>K</c> (paper: 3).</param>
    /// <param name="maxLbfgsEvals">EI-maximization per-restart L-BFGS-B evaluation cap (paper: 100).</param>
    /// <param name="noiseVariance">Fixed GP observation-noise variance (paper: <c>1e-6</c>, noise-free). Must be positive.</param>
    /// <param name="kernelVarianceLogStd">Std of the log-Normal prior on the kernel variance <c>σ²_k</c> (paper: <c>10</c>).</param>
    /// <summary>Configures SAASBO's Sobol startup, SAAS-GP inference, and EI maximization.</summary>
    /// <exception cref="System.ArgumentException">A count is below one or a positive-valued knob is non-positive.</exception>
    public SaasboSampler(
        BaseSampler? independentSampler = null,
        bool warnIndependentSampling = true,
        int? seed = null,
        int nStartupTrials = 10,
        SaasboInferenceMethod inferenceMethod = SaasboInferenceMethod.Nuts,
        double alpha = 0.1,
        int numWarmup = 512,
        int numSamples = 256,
        int thinning = 16,
        int maxTreeDepth = 6,
        double targetAccept = 0.8,
        int mapSteps = 1500,
        int numCandidates = 5000,
        int numRestarts = 3,
        int maxLbfgsEvals = 100,
        double noiseVariance = 1e-6,
        double kernelVarianceLogStd = 10.0)
    {
        if (nStartupTrials < 1)
            throw new ArgumentException("nStartupTrials must be at least 1.", nameof(nStartupTrials));
        if (!(alpha > 0.0))
            throw new ArgumentException("alpha must be positive.", nameof(alpha));
        if (numWarmup < 0)
            throw new ArgumentException("numWarmup must be non-negative.", nameof(numWarmup));
        if (numSamples < 1)
            throw new ArgumentException("numSamples must be at least 1.", nameof(numSamples));
        if (thinning < 1)
            throw new ArgumentException("thinning must be at least 1.", nameof(thinning));
        if (maxTreeDepth < 1)
            throw new ArgumentException("maxTreeDepth must be at least 1.", nameof(maxTreeDepth));
        if (mapSteps < 1)
            throw new ArgumentException("mapSteps must be at least 1.", nameof(mapSteps));
        if (numCandidates < 1)
            throw new ArgumentException("numCandidates must be at least 1.", nameof(numCandidates));
        if (numRestarts < 1)
            throw new ArgumentException("numRestarts must be at least 1.", nameof(numRestarts));
        if (maxLbfgsEvals < 1)
            throw new ArgumentException("maxLbfgsEvals must be at least 1.", nameof(maxLbfgsEvals));
        if (!(noiseVariance > 0.0))
            throw new ArgumentException("noiseVariance must be positive.", nameof(noiseVariance));
        if (!(kernelVarianceLogStd > 0.0))
            throw new ArgumentException("kernelVarianceLogStd must be positive.", nameof(kernelVarianceLogStd));

        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _searchSpace = new IntersectionSearchSpace();
        _warnIndependentSampling = warnIndependentSampling;
        _nStartupTrials = nStartupTrials;
        _config = new SaasboConfig(
            inferenceMethod, alpha, noiseVariance, kernelVarianceLogStd,
            numWarmup, numSamples, thinning, maxTreeDepth, targetAccept,
            MapTaus: null, mapSteps, numCandidates, numRestarts, maxLbfgsEvals);

        // A fixed base seed makes each per-trial stream a pure function of (base, trial number). Derived
        // through LazyRandomState so a null seed draws from OS entropy without touching System.Random.
        var seedRng = new LazyRandomState(seed);
        _baseSeed = (uint)seedRng.Rng.randint(0, int.MaxValue);
    }

    /// <inheritdoc />
    public override void ReseedRng() => _independentSampler.ReseedRng();

    /// <inheritdoc />
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _searchSpace.Calculate(study))
        {
            // Single-valued distributions are sampled inside Trial; the SAAS GP box is continuous, so
            // categorical parameters are delegated to the independent sampler (as TurboSampler/CmaEsSampler).
            if (distribution.Single()) continue;
            if (distribution is not FloatDistribution && distribution is not IntDistribution) continue;
            searchSpace[name] = distribution;
        }
        return searchSpace;
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">The study has multiple objectives, but SAASBO requires a scalar objective.</exception>
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        RaiseErrorIfMultiObjective(study);

        if (searchSpace.Count == 0)
            return new OrderedDict<string, object?>();

        // Completed trials in evaluation (creation) order — the observation sequence Algorithm 1 conditions on.
        var completed = study.GetTrialsInternal(deepcopy: false, states: null, useCache: true)
            .Where(t => t.State == TrialState.Complete)
            .OrderBy(t => t.Number)
            .ToList();

        var internalSpace = new GpSearchSpace(searchSpace);
        int d = internalSpace.Dim;

        // ---- initialization phase: Sobol space-filling until nStartupTrials observations exist ----
        if (completed.Count < _nStartupTrials)
        {
            // A single Sobol stream (seeded from the base seed) makes the init reproducible; the k-th init
            // trial takes row k. Mask to 31 bits so the seed is non-negative (numpy rejects negative entropy).
            int initSeed = (int)((_baseSeed ^ InitSeedSalt) & 0x7FFFFFFFu);
            double[,] sobolInit = ScrambledSobol.Random(d, _nStartupTrials, initSeed);
            var point = new double[d];
            for (int j = 0; j < d; j++) point[j] = sobolInit[completed.Count, j];
            return Unnormalize(internalSpace, searchSpace, point);
        }

        // ---- Bayesian-optimization phase: fit the SAAS GP and maximize the averaged EI ----
        int n = completed.Count;
        double[,] normalizedParams = internalSpace.GetNormalizedParams(
            completed.Select(t => (IReadOnlyDictionary<string, object?>)t.ParamsDict).ToList());

        double sign = study.Direction == StudyDirection.Minimize ? -1.0 : 1.0;

        // Standardize the maximization-oriented objective before the GP fit (the paper standardizes y).
        var rawScore = new double[n];
        for (int i = 0; i < n; i++) rawScore[i] = ToScore(completed[i].Value, sign);
        double[] clipped = ClipFinite(rawScore);
        double mean = MathUtil.Mean(clipped);
        double std = Math.Max(Eps, MathUtil.Std(clipped));
        var y = new double[n];
        for (int i = 0; i < n; i++) y[i] = (clipped[i] - mean) / std;

        // Per-trial MT19937 stream: a pure function of (base seed, trial number), so the NUTS chain and the
        // Sobol EI screen reproduce regardless of execution order.
        NumPyRandom rng = np.random.RandomState();
        rng.seed(Mix(_baseSeed, (ulong)trial.Number));

        double[] proposal = SaasboOptimizer.Propose(normalizedParams, y, _config, rng);
        return Unnormalize(internalSpace, searchSpace, proposal);
    }

    /// <summary>Maps a completed trial's objective value to a maximization-oriented score (higher = better).</summary>
    /// <param name="value">The trial's scalar objective value (null for an unfinished trial).</param>
    /// <param name="sign">+1 for a maximizing study, −1 for a minimizing study.</param>
    /// <returns>The maximization-oriented score, with a missing or NaN value mapped to the worst score.</returns>
    private static double ToScore(double? value, double sign)
    {
        if (value is not double v || double.IsNaN(v)) return double.NegativeInfinity;
        return sign * v;
    }

    /// <summary>Replaces non-finite scores with the finite extremes so standardization stays finite:
    ///     <c>+∞ → max</c>, <c>−∞/NaN → min</c> (over the finite entries; all-non-finite yields all zeros).</summary>
    /// <param name="scores">The maximization-oriented scores of the completed trials.</param>
    /// <returns>A finite copy suitable for standardization.</returns>
    private static double[] ClipFinite(double[] scores)
    {
        double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
        foreach (double s in scores)
            if (!double.IsNaN(s) && !double.IsInfinity(s))
            {
                if (s < lo) lo = s;
                if (s > hi) hi = s;
            }
        if (double.IsPositiveInfinity(lo))   // no finite score at all
            return new double[scores.Length];

        var result = new double[scores.Length];
        for (int i = 0; i < scores.Length; i++)
        {
            double s = scores[i];
            result[i] = double.IsNaN(s) ? lo
                : double.IsPositiveInfinity(s) ? hi
                : double.IsNegativeInfinity(s) ? lo
                : s;
        }
        return result;
    }

    /// <summary>Maps a normalized <c>[0, 1]^d</c> point back to external parameter values in search-space order.</summary>
    /// <param name="internalSpace">The GP-internal normalized search space.</param>
    /// <param name="searchSpace">The relative search space defining the output key order.</param>
    /// <param name="point">The normalized point to convert.</param>
    /// <returns>External parameter values keyed by name in search-space order.</returns>
    private static OrderedDict<string, object?> Unnormalize(
        GpSearchSpace internalSpace, OrderedDict<string, BaseDistribution> searchSpace, double[] point)
    {
        Dictionary<string, object?> raw = internalSpace.GetUnnormalizedParam(point);
        var result = new OrderedDict<string, object?>();
        foreach (string name in searchSpace.Keys)
            result[name] = raw[name];
        return result;
    }

    /// <summary>SplitMix64 finalizer folding a base seed and a counter into a well-scrambled 32-bit seed —
    ///     avoids the MT19937 near-sequential-seed correlation when deriving per-trial streams.</summary>
    /// <param name="seed">The persisted base seed.</param>
    /// <param name="counter">The per-trial index.</param>
    /// <returns>A 32-bit seed for <c>RandomState.seed</c>.</returns>
    private static uint Mix(uint seed, ulong counter)
    {
        ulong z = ((ulong)seed << 32) ^ counter;
        z += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (uint)(z & 0xFFFFFFFFUL);
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">The study has multiple objectives, but SAASBO requires a scalar objective.</exception>
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        RaiseErrorIfMultiObjective(study);

        if (_warnIndependentSampling)
        {
            // Warn only once the BO phase would otherwise be active — a categorical parameter sampled during
            // the Sobol init phase is expected, not a degradation worth flagging repeatedly.
            var completed = study.GetTrialsInternal(deepcopy: false, states: null, useCache: true)
                .Count(t => t.State == TrialState.Complete);
            if (completed >= _nStartupTrials)
                LogIndependentSampling(trial, paramName);
        }

        return _independentSampler.SampleIndependent(study, trial, paramName, paramDistribution);
    }

    /// <summary>Explains why the current parameter is delegated to the independent sampler.</summary>
    /// <param name="trial">Trial snapshot being examined; copy its contents before modifying them.</param>
    /// <param name="paramName">Name of the parameter being suggested or looked up.</param>
    private void LogIndependentSampling(FrozenTrial trial, string paramName)
    {
        Logger.Warning(string.Format(
            IndependentSamplingWarningTemplate,
            paramName,
            trial.Number,
            _independentSampler.GetType().Name,
            GetType().Name,
            "dynamic search space and `CategoricalDistribution` are not supported by `SaasboSampler`"));
    }

    /// <inheritdoc />
    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _independentSampler.BeforeTrial(study, trial);

    /// <inheritdoc />
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        // Python `if trial.constraints:` — FrozenTrial.Constraints is never null (an empty dict when unset),
        // so warn only when at least one constraint was actually set on the trial.
        if (trial.Constraints.Count > 0)
            OptunaWarnings.Warn(
                "Constraints have been set, but SaasboSampler will not use them for optimization.",
                WarningCategory.UserWarning);
        _independentSampler.AfterTrial(study, trial, state, values);
    }
}
