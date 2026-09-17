// Sampler for TuRBO-1 — Trust-Region Bayesian Optimization (Eriksson et al., NeurIPS 2019). Drives the
// `TurboOptimizer` trust-region mechanics through the Optuna study protocol, reusing the native GP stack
// (GPRegressor: Matérn-5/2 ARD, OpenBLAS Cholesky) that GPSampler already ships. TuRBO is NOT part of the
// Optuna ecosystem, so — like DifferentialEvolutionSampler / LmCmaSampler — this is a native
// implementation validated by convergence (GP + Thompson sampling is stochastic and not bit-exact vs the
// BoTorch reference), on NumSharp's MT19937 (never System.Random).

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Gp;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers.Turbo;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     A sampler using TuRBO-1 — single-trust-region Bayesian optimization (Eriksson et al. 2019). It fits
///     a <b>local</b> Matérn-5/2 GP to the observations of the current local run, keeps a trust region (a
///     length-scale-weighted hyperrectangle) around the incumbent, and proposes the next point by Thompson
///     sampling over a Sobol perturbation set inside that region. The region expands after consecutive
///     improvements, shrinks after stalls, and restarts once it collapses — the mechanism that lets TuRBO
///     keep making progress in higher dimension where a single global GP (see <see cref="GPSampler" />)
///     over-explores.
///
///     <para><b>Beyond-core, convergence-validated.</b> TuRBO has no Optuna counterpart, so — like
///     <see cref="LmCmaSampler" /> and <see cref="DifferentialEvolutionSampler" /> — it is validated by
///     convergence rather than parity. GP fitting + Thompson sampling ride the same native GP stack as
///     <see cref="GPSampler" /> (hand-derived Matérn-5/2 marginal-likelihood gradient, managed L-BFGS-B,
///     OpenBLAS Cholesky), so results are tolerance-reproducible on a platform, not bit-exact vs BoTorch.</para>
///
///     <para><b>Continuous / integer only.</b> The trust-region box is a continuous construct, so the
///     relative search space is Float/Int only; categorical and single-valued parameters are delegated to
///     the independent sampler, exactly as in <see cref="CmaEsSampler" />. Dynamic search spaces fall back
///     to independent sampling with a warning. Single-objective only.</para>
///
///     <para><b>State is reconstructed, not persisted.</b> Unlike CMA-ES/DE, TuRBO stores no optimizer
///     blob: the trust-region length, resize counters, and restart boundaries are recomputed each trial by
///     a deterministic fold over the completed-trial objective sequence (<see cref="TurboOptimizer.ReconstructState" />),
///     so a study resumes from storage with no serialized state.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 42));
///     study.Optimize(t =>
///     {
///         double s = 0;
///         for (int i = 0; i &lt; 10; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
///         return s;
///     }, nTrials: 200);
///     </code>
/// </example>
public sealed class TurboSampler : BaseSampler
{
    private const double Eps = 1e-10;
    private const uint InitSeedSalt = 0x5eed_1117u;   // de-correlates the per-restart init Sobol stream

    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers.turbo");

    private readonly BaseSampler _independentSampler;
    private readonly IntersectionSearchSpace _searchSpace;
    private bool _warnIndependentSampling;
    private readonly uint _baseSeed;

    private readonly int? _nInit;
    private readonly int? _numCandidates;
    private readonly double _lengthInit;
    private readonly double _lengthMin;
    private readonly double _lengthMax;
    private readonly int _successTolerance;
    private readonly int? _failureTolerance;
    private readonly double _minimumNoise;
    private readonly bool _deterministicObjective;

    /// <param name="independentSampler">Sampler for parameters outside the relative search space, and for the
    ///     per-restart initialization when it is not Sobol (default <see cref="RandomSampler" />).</param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently.</param>
    /// <param name="seed">Random seed; a fixed seed reproduces the trajectory on a platform.</param>
    /// <param name="nInit">Per-restart initialization trials (Sobol space-filling) before the local GP takes
    ///     over; <c>null</c> uses <c>2·d</c>.</param>
    /// <param name="numCandidates">Thompson-sampling candidate-set size per trial; <c>null</c> uses
    ///     <c>min(100·d, 500)</c>. The paper uses <c>min(100·d, 5000)</c> with BoTorch's scalable posterior
    ///     sampler; this exact managed implementation draws the joint posterior via Cholesky (cost
    ///     <c>O(numCandidates³)</c> per trial), so the default caps lower to keep the per-trial overhead near
    ///     the GP-fit cost — raise it toward the paper's value for harder problems if you can afford it.</param>
    /// <param name="lengthInit">Initial trust-region base side length on a fresh restart (default 0.8).</param>
    /// <param name="lengthMin">Restart floor: when the base length falls below this the region is discarded
    ///     and re-initialized (default <c>2⁻⁷ ≈ 0.0078</c>).</param>
    /// <param name="lengthMax">Cap on the base side length under expansion (default 1.6).</param>
    /// <param name="successTolerance">Consecutive improvements required to double the region (default 3, the
    ///     paper supplement value; the BoTorch tutorial uses 10).</param>
    /// <param name="failureTolerance">Consecutive stalls required to halve the region; <c>null</c> uses the
    ///     paper's <c>max(4, d)</c> (i.e. <c>⌈max(4/q, d/q)⌉</c> with batch <c>q = 1</c>).</param>
    /// <param name="minimumNoise">Lower floor on the fitted GP observation-noise variance; <c>null</c> uses
    ///     the GP stack default.</param>
    /// <param name="deterministicObjective">Treat the objective as noise-free (fix the GP noise at the floor).</param>
    /// <summary>Configures TuRBO-1's trust-region schedule, local-GP fit, and independent startup/fallback.</summary>
    /// <exception cref="System.ArgumentException">A length bound is non-positive/ill-ordered, or a count is below one.</exception>
    public TurboSampler(
        BaseSampler? independentSampler = null,
        bool warnIndependentSampling = true,
        int? seed = null,
        int? nInit = null,
        int? numCandidates = null,
        double lengthInit = 0.8,
        double lengthMin = 1.0 / 128.0,   // 2^-7
        double lengthMax = 1.6,
        int successTolerance = 3,
        int? failureTolerance = null,
        double? minimumNoise = null,
        bool deterministicObjective = false)
    {
        if (!(lengthMin > 0 && lengthInit >= lengthMin && lengthMax >= lengthInit))
            throw new ArgumentException("Require 0 < lengthMin ≤ lengthInit ≤ lengthMax.");
        if (successTolerance < 1)
            throw new ArgumentException("successTolerance must be at least 1.", nameof(successTolerance));
        if (nInit is int ni && ni < 1)
            throw new ArgumentException("nInit must be at least 1.", nameof(nInit));
        if (numCandidates is int nc && nc < 1)
            throw new ArgumentException("numCandidates must be at least 1.", nameof(numCandidates));
        if (failureTolerance is int ft && ft < 1)
            throw new ArgumentException("failureTolerance must be at least 1.", nameof(failureTolerance));

        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _searchSpace = new IntersectionSearchSpace();
        _warnIndependentSampling = warnIndependentSampling;
        _nInit = nInit;
        _numCandidates = numCandidates;
        _lengthInit = lengthInit;
        _lengthMin = lengthMin;
        _lengthMax = lengthMax;
        _successTolerance = successTolerance;
        _failureTolerance = failureTolerance;
        _minimumNoise = minimumNoise ?? Prior.DefaultMinimumNoiseVar;
        _deterministicObjective = deterministicObjective;

        // A fixed base seed makes the per-trial noise a pure function of (base, trial number). Derived
        // through a LazyRandomState so a null seed draws from OS entropy without touching System.Random.
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
            // Single-valued distributions are sampled inside Trial; the trust-region box is a continuous
            // construct, so categorical parameters are delegated to the independent sampler (as CmaEsSampler).
            if (distribution.Single()) continue;
            if (distribution is not FloatDistribution && distribution is not IntDistribution) continue;
            searchSpace[name] = distribution;
        }
        return searchSpace;
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">The study has multiple objectives, but TuRBO-1 requires a scalar objective.</exception>
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        RaiseErrorIfMultiObjective(study);

        if (searchSpace.Count == 0)
            return new OrderedDict<string, object?>();

        // Completed trials in evaluation (creation) order — the fold's input sequence.
        var completed = study.GetTrialsInternal(deepcopy: false, states: null, useCache: true)
            .Where(t => t.State == TrialState.Complete)
            .OrderBy(t => t.Number)
            .ToList();

        var internalSpace = new GpSearchSpace(searchSpace);
        int d = internalSpace.Dim;
        int nInit = _nInit ?? 2 * d;

        double sign = study.Direction == StudyDirection.Minimize ? -1.0 : 1.0;

        // Maximization-oriented score of every completed trial (higher = better), for the trust-region fold.
        var foldScores = new double[completed.Count];
        for (int i = 0; i < completed.Count; i++)
            foldScores[i] = ToScore(completed[i].Value, sign);

        var optimizer = new TurboOptimizer(
            _lengthInit, _lengthMin, _lengthMax, _successTolerance,
            _failureTolerance ?? Math.Max(4, d), successTol: 1e-3);
        TurboTrState state = optimizer.ReconstructState(foldScores, nInit);

        int restartCount = completed.Count - state.RestartStart;   // completed trials in the current restart

        // ---- initialization phase: fill the current restart with Sobol space-filling points ----
        if (restartCount < nInit)
        {
            // A per-restart Sobol stream (seeded from the restart index) makes the init reproducible; each
            // init trial takes the next row of that restart's sequence. Mask to 31 bits so the seed is
            // non-negative (numpy's SeedSequence rejects negative entropy).
            int restartSeed = (int)(Mix(_baseSeed ^ InitSeedSalt, (ulong)state.RestartCount) & 0x7FFFFFFFu);
            double[,] sobolInit = ScrambledSobol.Random(d, nInit, restartSeed);
            var point = new double[d];
            for (int j = 0; j < d; j++) point[j] = sobolInit[restartCount, j];
            return Unnormalize(internalSpace, searchSpace, point);
        }

        // ---- Bayesian-optimization phase: fit the local GP and Thompson-sample inside the trust region ----
        var restartTrials = completed.Skip(state.RestartStart).ToList();
        int m = restartTrials.Count;

        double[,] normalizedParams = internalSpace.GetNormalizedParams(
            restartTrials.Select(t => (IReadOnlyDictionary<string, object?>)t.ParamsDict).ToList());

        // Standardize the local run's scores for the GP fit; the (finite-clipped) argmax is the incumbent.
        var rawScore = new double[m];
        for (int i = 0; i < m; i++) rawScore[i] = ToScore(restartTrials[i].Value, sign);
        double[] clipped = ClipFinite(rawScore);
        double mean = MathUtil.Mean(clipped);
        double std = Math.Max(Eps, MathUtil.Std(clipped));
        var y = new double[m];
        int incumbent = 0;
        for (int i = 0; i < m; i++)
        {
            y[i] = (clipped[i] - mean) / std;
            if (clipped[i] > clipped[incumbent]) incumbent = i;
        }

        GPRegressor gpr = GPRegressor.FitKernelParams(
            normalizedParams, y, internalSpace.IsCategorical, _minimumNoise, _deterministicObjective);

        var xCenter = new double[d];
        for (int j = 0; j < d; j++) xCenter[j] = normalizedParams[incumbent, j];

        (double[] lb, double[] ub) = optimizer.TrustRegionBounds(xCenter, gpr.LengthScales, state.Length);

        // Per-trial MT19937 stream: a pure function of (base seed, trial number), so the candidate set and
        // the Thompson draw reproduce regardless of execution order.
        NumPyRandom rng = np.random.RandomState();
        rng.seed(Mix(_baseSeed, (ulong)trial.Number));

        int numCandidates = _numCandidates ?? Math.Min(100 * d, 500);
        double[,] candidates = optimizer.GenerateCandidates(xCenter, lb, ub, numCandidates, rng);

        (double[] postMean, double[,] postCov) = gpr.PosteriorJoint(candidates);
        int chosen = TurboOptimizer.ThompsonArgmax(postMean, postCov, rng);

        var chosenPoint = new double[d];
        for (int j = 0; j < d; j++) chosenPoint[j] = candidates[chosen, j];
        return Unnormalize(internalSpace, searchSpace, chosenPoint);
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

    /// <summary>Replaces non-finite scores with the finite extremes so the GP standardization stays finite:
    ///     <c>+∞ → max</c>, <c>−∞/NaN → min</c> (over the finite entries; all-non-finite yields all zeros).</summary>
    /// <param name="scores">The maximization-oriented scores of the current local run.</param>
    /// <returns>A finite copy suitable for standardization and incumbent selection.</returns>
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
    ///     avoids the MT19937 near-sequential-seed correlation when deriving per-trial / per-restart streams.</summary>
    /// <param name="seed">The persisted base seed.</param>
    /// <param name="counter">The per-trial or per-restart index.</param>
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
    /// <exception cref="NumSharp.ValueError">The study has multiple objectives, but TuRBO-1 requires a scalar objective.</exception>
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        RaiseErrorIfMultiObjective(study);

        if (_warnIndependentSampling)
        {
            // Warn only once the trust-region phase would otherwise be active — a categorical parameter
            // sampled during the Sobol init phase is expected, not a degradation worth flagging repeatedly.
            var completed = study.GetTrialsInternal(deepcopy: false, states: null, useCache: true)
                .Count(t => t.State == TrialState.Complete);
            if (completed > 0)
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
            "dynamic search space and `CategoricalDistribution` are not supported by `TurboSampler`"));
    }

    /// <inheritdoc />
    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _independentSampler.BeforeTrial(study, trial);

    /// <inheritdoc />
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        // Python `if trial.constraints:` — FrozenTrial.Constraints is never null (an empty dict when
        // unset), so warn only when at least one constraint was actually set on the trial.
        if (trial.Constraints.Count > 0)
            OptunaWarnings.Warn(
                "Constraints have been set, but TurboSampler will not use them for optimization.",
                WarningCategory.UserWarning);
        _independentSampler.AfterTrial(study, trial, state, values);
    }
}
