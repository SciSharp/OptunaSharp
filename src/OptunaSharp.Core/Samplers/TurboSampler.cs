// Sampler for TuRBO — Trust-Region Bayesian Optimization (Eriksson et al., NeurIPS 2019): TuRBO-1 (a single
// trust region, the default) and TuRBO-m (m simultaneous trust regions with an implicit multi-armed bandit,
// selected by `numTrustRegions > 1`). Drives the `TurboOptimizer` trust-region mechanics through the Optuna
// study protocol, reusing the native GP stack (GPRegressor: Matérn-5/2 ARD, OpenBLAS Cholesky) that GPSampler
// already ships. Everything inside a region is unchanged TuRBO-1; the only mechanic TuRBO-m adds is the
// cross-region competition in `TurboMOptimizer`. TuRBO is NOT part of the Optuna ecosystem, so — like
// DifferentialEvolutionSampler / LmCmaSampler — this is a native implementation validated by convergence (GP +
// Thompson sampling is stochastic and not bit-exact vs the BoTorch/uber-research reference), on NumSharp's
// MT19937 (never System.Random).

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.ModelBased.Gp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers.Turbo;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     A sampler using TuRBO — trust-region Bayesian optimization (Eriksson et al. 2019). It fits a
///     <b>local</b> Matérn-5/2 GP to the observations of the current local run, keeps a trust region (a
///     length-scale-weighted hyperrectangle) around the incumbent, and proposes the next point by Thompson
///     sampling over a Sobol perturbation set inside that region. The region expands after consecutive
///     improvements, shrinks after stalls, and restarts once it collapses — the mechanism that lets TuRBO
///     keep making progress in higher dimension where a single global GP (see <see cref="GPSampler" />)
///     over-explores.
///
///     <para><b>TuRBO-1 vs TuRBO-m.</b> With the default <c>numTrustRegions = 1</c> this is TuRBO-1, a single
///     trust region restarted sequentially. With <c>numTrustRegions = m &gt; 1</c> it is TuRBO-m: <c>m</c>
///     trust regions run simultaneously, each an independent TuRBO-1 with its own local GP, incumbent, and
///     resize counters, and an implicit multi-armed bandit (a Thompson sample drawn from every region, global
///     best wins) allocates each evaluation to the most promising region. TuRBO-m concentrates the budget on
///     the best of several basins (§3.7) but <b>divides</b> that budget across the regions, so it is not
///     uniformly better — it wins on multimodal/deceptive problems and can tie or slightly lose on easy ones;
///     the paper accordingly reports TuRBO-1 <b>and</b> TuRBO-5/-20 by regime.</para>
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
///     <para><b>State is reconstructed, not persisted (with one small exception for TuRBO-m).</b> TuRBO-1
///     stores no optimizer blob: the trust-region length, resize counters, and restart boundaries are
///     recomputed each trial by a deterministic fold over the completed-trial objective sequence
///     (<see cref="TurboOptimizer.ReconstructState" />), so a study resumes from storage with no serialized
///     state. TuRBO-m adds the minimum needed to fold each region: because a trial's region is the
///     seeded competition winner (a function of which prior trials landed in which region, not of the values
///     alone), <c>m &gt; 1</c> persists one small per-trial region label (system attr <c>"turbo:tr"</c>);
///     each region then folds exactly as TuRBO-1 over its labeled subsequence.</para>
/// </summary>
/// <example>
///     <code>
///     // TuRBO-1 (single trust region, the default).
///     var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 42));
///     study.Optimize(t =>
///     {
///         double s = 0;
///         for (int i = 0; i &lt; 10; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
///         return s;
///     }, nTrials: 200);
///
///     // TuRBO-5 (five simultaneous trust regions) for a multimodal objective with several basins.
///     var multi = Optuna.CreateStudy(sampler: new TurboSampler(seed: 42, numTrustRegions: 5));
///     multi.Optimize(t =>
///     {
///         double s = 0;
///         for (int i = 0; i &lt; 10; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x - 10 * Math.Cos(2 * Math.PI * x); }
///         return s + 100;
///     }, nTrials: 400);
///     </code>
/// </example>
public sealed class TurboSampler : BaseSampler
{
    private const double Eps = 1e-10;
    private const uint InitSeedSalt = 0x5eed_1117u;   // de-correlates the per-restart init Sobol stream

    // Per-trial trust-region label persisted for TuRBO-m (m > 1). Unlike TuRBO-1, a trial's region is the
    // stochastic-but-seeded competition winner — a function of which prior trials landed in which region —
    // so it is NOT recoverable from the value sequence alone; the label is the one piece of state TuRBO-m
    // persists (a small int per trial), and each region's L/counters/restart boundaries then fold cheaply
    // from (labels, values). See docs/TURBO_M_PORTING.md §2.
    private const string AttrKeyTrustRegion = "turbo:tr";

    // [slot, restart, row] of the initialization design a trial took (see InitRow), so trials asked at the same time
    // never take one Sobol row.
    private const string AttrKeyInitRow = "turbo:init";

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
    private readonly double _successMargin;
    private readonly double _minimumNoise;
    private readonly bool _deterministicObjective;
    private readonly int _numTrustRegions;

    /// <param name="independentSampler">Sampler for parameters outside the relative search space, and for the
    ///     per-restart initialization when it is not Sobol (default <see cref="RandomSampler" />).</param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently.</param>
    /// <param name="seed">Random seed; a fixed seed reproduces the trajectory on a platform.</param>
    /// <param name="nInit">Per-restart initialization trials (Sobol space-filling) before the local GP takes
    ///     over; <c>null</c> uses <c>2·d</c>.</param>
    /// <param name="numCandidates">Thompson-sampling candidate-set size per trial; <c>null</c> uses the
    ///     paper's <c>min(100·d, 5000)</c> (§D). <b>Cost footgun:</b> this exact managed implementation draws
    ///     the joint posterior via a dense Cholesky (cost <c>O(numCandidates³)</c> per trial), whereas the
    ///     paper leans on BoTorch's scalable posterior sampler — so at high <c>d</c> the paper-faithful
    ///     default is genuinely heavy (a 5000-wide Cholesky per trial). Lower it (e.g. a few hundred) to trade
    ///     discretization fidelity for speed when the per-trial overhead dominates.</param>
    /// <param name="lengthInit">Initial trust-region base side length on a fresh restart (default 0.8).</param>
    /// <param name="lengthMin">Restart floor: when the base length falls below this the region is discarded
    ///     and re-initialized (default <c>2⁻⁷ ≈ 0.0078</c>).</param>
    /// <param name="lengthMax">Cap on the base side length under expansion (default 1.6).</param>
    /// <param name="successTolerance">Consecutive improvements required to double the region (default 3, the
    ///     paper supplement value; the BoTorch tutorial uses 10).</param>
    /// <param name="failureTolerance">Consecutive stalls required to halve the region; <c>null</c> uses the
    ///     paper's <c>τ_fail = ⌈d/q⌉ = d</c> (batch <c>q = 1</c>; §D). The authors' reference code and the
    ///     BoTorch tutorial instead use <c>⌈max(4/q, d/q)⌉ = max(4, d)</c> as a low-dimension guard —
    ///     identical for <c>d ≥ 4</c> (the paper's whole experimental regime) and different only for
    ///     <c>d ∈ {1, 2, 3}</c>. Pass <c>Math.Max(4, d)</c> to reproduce that reference-code behavior.</param>
    /// <param name="successMargin">Relative margin a candidate must clear over the incumbent to count as a
    ///     <i>success</i>: <c>score &gt; best + successMargin·|best|</c>. Default <c>0.0</c> is the paper's
    ///     strict improvement (§D); the BoTorch tutorial uses <c>1e-3</c> to ignore negligible gains — pass
    ///     that to recover BoTorch behavior.</param>
    /// <param name="minimumNoise">Lower floor on the fitted GP observation-noise variance and the effective
    ///     lower bound of the §C noise box; <c>null</c> uses the paper's <c>σ² ≥ 0.0005</c> (§C). Must stay
    ///     below the §C upper bound <c>0.1</c> for a well-posed noise box.</param>
    /// <param name="deterministicObjective">Treat the objective as noise-free: fix the GP noise at the floor
    ///     <b>and</b> center the trust region on the best <i>observation</i> (paper §2). When <c>false</c> (the
    ///     default, noisy case) the center is the training point with the best posterior <i>mean</i> instead.</param>
    /// <param name="numTrustRegions">Number of simultaneous trust regions <c>m</c> (paper §2). Default <c>1</c> is
    ///     TuRBO-1 (single region), a provably byte-identical path — the <c>m &gt; 1</c> multi-region bandit code
    ///     never runs at <c>m = 1</c>. Pass <c>5</c> for the paper's common "TuRBO-5". <b>Consequences of
    ///     <c>m &gt; 1</c>:</b> the evaluation budget is <b>divided</b> across the regions (each pays a <c>2·d</c>
    ///     init tax and gets fewer exploitation steps), so TuRBO-m is <i>not</i> uniformly better — it wins on
    ///     multimodal/deceptive problems with several basins (§3.7) and can tie or slightly lose on easy/unimodal
    ///     ones (which is why the paper reports TuRBO-1 <b>and</b> TuRBO-5/-20 by regime). And because a trial's
    ///     region is the seeded competition winner (not a function of the values alone), <c>m &gt; 1</c> persists
    ///     one small per-trial region label to storage (system attr <c>"turbo:tr"</c>); a trial whose relative
    ///     search space is empty (all-categorical) carries no label and never joins a region.</param>
    /// <summary>Configures TuRBO's trust-region schedule, local-GP fit, region count, and independent startup/fallback.</summary>
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
        double successMargin = 0.0,
        double? minimumNoise = null,
        bool deterministicObjective = false,
        int numTrustRegions = 1)
    {
        if (!(lengthMin > 0 && lengthInit >= lengthMin && lengthMax >= lengthInit))
            throw new ArgumentException("Require 0 < lengthMin ≤ lengthInit ≤ lengthMax.");
        if (successTolerance < 1)
            throw new ArgumentException("successTolerance must be at least 1.", nameof(successTolerance));
        if (numTrustRegions < 1)
            throw new ArgumentException("numTrustRegions must be at least 1.", nameof(numTrustRegions));
        if (successMargin < 0)
            throw new ArgumentException("successMargin must be non-negative.", nameof(successMargin));
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
        _successMargin = successMargin;
        // Paper §C floors observation-noise variance at 0.0005 (not the GP stack's 1e-6 default); this is
        // also the lower bound of the §C noise box the local GP is fitted under (see TurboGpFit).
        _minimumNoise = minimumNoise ?? TurboGpFit.NoiseVarMin;
        _deterministicObjective = deterministicObjective;
        _numTrustRegions = numTrustRegions;

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

        // TuRBO-1 (the default single region) runs the original, untouched path; m > 1 activates the
        // multi-trust-region bandit (TuRBO-m). Branching on m keeps the single-region trajectory provably
        // byte-identical — the multi-region code below never executes at m = 1.
        return _numTrustRegions == 1
            ? SampleRelativeSingleTrustRegion(study, trial, searchSpace)
            : SampleRelativeMultiTrustRegion(study, trial, searchSpace);
    }

    /// <summary>
    ///     The single-trust-region path (TuRBO-1): the deterministic state fold over the completed-trial
    ///     objective sequence, the Sobol initialization phase, then the Bayesian-optimization phase (fit the
    ///     local GP, choose the center, build the trust-region box + candidate set, Thompson-sample,
    ///     unnormalize). Reached only when <c>numTrustRegions == 1</c>; the body is unchanged from the original
    ///     TuRBO-1 sampler, so its trajectory is byte-identical to the pre-TuRBO-m release.
    /// </summary>
    /// <param name="study">The study whose completed trials drive the reconstructed trust-region state and direction.</param>
    /// <param name="trial">The trial being sampled; its number seeds the per-trial candidate/Thompson RNG.</param>
    /// <param name="searchSpace">The relative (Float/Int) search space to sample over.</param>
    /// <returns>The next point's external parameter values in search-space order.</returns>
    private OrderedDict<string, object?> SampleRelativeSingleTrustRegion(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
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
            _failureTolerance ?? DefaultFailureTolerance(d), successTol: _successMargin);
        TurboTrState state = optimizer.ReconstructState(foldScores, nInit);

        // Completed trials in the CURRENT restart (distinct from state.RestartCount, the number of restarts
        // so far): while this is below nInit we are still filling the restart's Sobol initialization.
        int trialsInRestart = completed.Count - state.RestartStart;

        // ---- initialization phase: fill the current restart with Sobol space-filling points ----
        if (trialsInRestart < nInit)
        {
            // A per-restart Sobol stream (seeded from the restart index) makes the init reproducible; each
            // init trial takes the next row of that restart's sequence. Mask to 31 bits so the seed is
            // non-negative (numpy's SeedSequence rejects negative entropy).
            int restartSeed = (int)(Mix(_baseSeed ^ InitSeedSalt, (ulong)state.RestartCount) & 0x7FFFFFFFu);
            // The next row is the first one no finished or running trial of this restart holds. It used to be the
            // finished count, so workers asking at the same time after a restart all took one row (7 of 64 trials at
            // 8 workers). A sequential run holds rows 0..count-1, so the row is still trialsInRestart.
            int row = InitRow(study, trial, slot: 0, state.RestartCount, completed.Skip(state.RestartStart));
            double[,] sobolInit = ScrambledSobol.Random(d, Math.Max(nInit, row + 1), restartSeed);
            var point = new double[d];
            for (int j = 0; j < d; j++) point[j] = sobolInit[row, j];
            PersistInitRow(study, trial, slot: 0, state.RestartCount, row);
            return Unnormalize(internalSpace, searchSpace, point);
        }

        // ---- Bayesian-optimization phase: fit the local GP and Thompson-sample inside the trust region ----
        var restartTrials = completed.Skip(state.RestartStart).ToList();
        int m = restartTrials.Count;

        double[,] normalizedParams = internalSpace.GetNormalizedParams(
            restartTrials.Select(t => t.Params).ToList());

        // Standardize the local run's scores for the GP fit (the paper standardizes objective values before
        // fitting; §C). `clipped` keeps the maximization-oriented observations for the center rule.
        var rawScore = new double[m];
        for (int i = 0; i < m; i++) rawScore[i] = ToScore(restartTrials[i].Value, sign);
        double[] clipped = ClipFinite(rawScore);
        double mean = MathUtil.Mean(clipped);
        double std = Math.Max(Eps, MathUtil.Std(clipped));
        var y = new double[m];
        for (int i = 0; i < m; i++) y[i] = (clipped[i] - mean) / std;

        // Fit the LOCAL GP under the paper's §C hyperparameter box (TurboGpFit), not the shared unconstrained
        // GPSampler fit — the fitted length scales rescale the trust-region box, so §C keeps them in range.
        GPRegressor gpr = TurboGpFit.Fit(
            normalizedParams, y, internalSpace.IsCategorical, _minimumNoise, _deterministicObjective);

        // Paper §2 trust-region center x⋆: best observation when noise-free, else the point with the best
        // posterior mean under the fitted GP (the noisy default). Compute the posterior mean only when needed.
        double[]? posteriorMean = _deterministicObjective ? null : gpr.Posterior(normalizedParams).Mean;
        int incumbent = TurboOptimizer.SelectCenterIndex(clipped, posteriorMean, _deterministicObjective);

        var xCenter = new double[d];
        for (int j = 0; j < d; j++) xCenter[j] = normalizedParams[incumbent, j];

        (double[] lb, double[] ub) = optimizer.TrustRegionBounds(xCenter, gpr.LengthScales, state.Length);

        // Per-trial MT19937 stream: a pure function of (base seed, trial number), so the candidate set and
        // the Thompson draw reproduce regardless of execution order.
        NumPyRandom rng = np.random.RandomState();
        rng.seed(Mix(_baseSeed, (ulong)trial.Number));

        int numCandidates = _numCandidates ?? DefaultNumCandidates(d);
        double[,] candidates = optimizer.GenerateCandidates(xCenter, lb, ub, numCandidates, rng);

        (double[] postMean, double[,] postCov) = gpr.PosteriorJoint(candidates);
        int chosen = TurboOptimizer.ThompsonArgmax(postMean, postCov, rng);

        var chosenPoint = new double[d];
        for (int j = 0; j < d; j++) chosenPoint[j] = candidates[chosen, j];
        return Unnormalize(internalSpace, searchSpace, chosenPoint);
    }

    /// <summary>
    ///     The multi-trust-region path (TuRBO-m, paper §2): maintain <c>m</c> simultaneous trust regions and let
    ///     an implicit multi-armed bandit decide which region spends this trial. Read the persisted per-trial
    ///     region labels, partition the completed trials into <c>m</c> slots, and fold each slot independently
    ///     (each slot's <c>L</c>/counters/restart boundaries are exactly TuRBO-1 over its labeled subsequence).
    ///     Then either fill the lowest slot still initializing (deterministic slot-by-slot / post-collapse
    ///     re-init) or, when every slot is ready, run the cross-region Thompson competition and propose the
    ///     winner's candidate. Finally persist this trial's winning region label so the fold reproduces next time.
    /// </summary>
    /// <param name="study">The study whose completed trials + persisted labels reconstruct the per-region state.</param>
    /// <param name="trial">The trial being sampled; its number seeds the per-trial competition RNG and receives the region label.</param>
    /// <param name="searchSpace">The relative (Float/Int) search space to sample over.</param>
    /// <returns>The next point's external parameter values in search-space order.</returns>
    private OrderedDict<string, object?> SampleRelativeMultiTrustRegion(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        // Completed trials in evaluation (creation) order — the per-region folds' input sequence.
        var completed = study.GetTrialsInternal(deepcopy: false, states: null, useCache: true)
            .Where(t => t.State == TrialState.Complete)
            .OrderBy(t => t.Number)
            .ToList();

        var internalSpace = new GpSearchSpace(searchSpace);
        int d = internalSpace.Dim;
        int nInit = _nInit ?? 2 * d;
        int m = _numTrustRegions;
        double sign = study.Direction == StudyDirection.Minimize ? -1.0 : 1.0;

        // Partition the completed trials into m region slots by their persisted label. A trial with no valid
        // label (a foreign-sampler warm-start, or an all-categorical trial whose relative space was empty) joins
        // no region — TuRBO-m builds each region only from its OWN proposals.
        var slotTrials = new List<FrozenTrial>[m];
        for (int s = 0; s < m; s++) slotTrials[s] = new List<FrozenTrial>();
        foreach (FrozenTrial t in completed)
        {
            int label = ReadTrustRegionLabel(t);
            if (label >= 0 && label < m) slotTrials[label].Add(t);
        }

        // The per-region mechanics are unchanged TuRBO-1: one shared, stateless optimizer folded per slot.
        var optimizer = new TurboOptimizer(
            _lengthInit, _lengthMin, _lengthMax, _successTolerance,
            _failureTolerance ?? DefaultFailureTolerance(d), successTol: _successMargin);

        // Fold every slot independently to its current restart's state + how far its initialization has filled.
        var states = new TurboTrState[m];
        var trialsInRestart = new int[m];
        for (int s = 0; s < m; s++)
        {
            var scores = new double[slotTrials[s].Count];
            for (int i = 0; i < scores.Length; i++) scores[i] = ToScore(slotTrials[s][i].Value, sign);
            states[s] = optimizer.ReconstructState(scores, nInit);
            trialsInRestart[s] = slotTrials[s].Count - states[s].RestartStart;
        }

        // ---- initialization phase: fill the lowest slot still short of its nInit Sobol points ----
        // This fills slots slot-by-slot at startup and forces a collapsed slot's fresh nInit re-init burst
        // before the competition resumes, keeping all m regions live (turbo_m.py's per-region init/re-init).
        int initSlot = TurboMOptimizer.NextInitSlot(trialsInRestart, nInit);
        if (initSlot >= 0)
        {
            // Per-(slot, restart) Sobol seed: pack the slot into the high bits and the restart count into the low
            // bits so different slots and restarts draw independent designs. Mask to 31 bits (numpy's
            // SeedSequence rejects negative entropy). No per-trial RNG is consumed in the init phase.
            ulong initCounter = ((ulong)(uint)initSlot << 32) | (uint)states[initSlot].RestartCount;
            int restartSeed = (int)(Mix(_baseSeed ^ InitSeedSalt, initCounter) & 0x7FFFFFFFu);
            // As in TuRBO-1: the first row of this slot's restart that no finished or running trial holds, so
            // concurrent asks never share a row; sequentially it is still trialsInRestart[initSlot].
            int row = InitRow(study, trial, initSlot, states[initSlot].RestartCount,
                slotTrials[initSlot].Skip(states[initSlot].RestartStart));
            double[,] sobolInit = ScrambledSobol.Random(d, Math.Max(nInit, row + 1), restartSeed);
            var point = new double[d];
            for (int j = 0; j < d; j++) point[j] = sobolInit[row, j];
            PersistTrustRegionLabel(study, trial, initSlot);
            PersistInitRow(study, trial, initSlot, states[initSlot].RestartCount, row);
            return Unnormalize(internalSpace, searchSpace, point);
        }

        // ---- competition phase: every slot is ready, so the bandit picks which region spends this trial ----
        // Per-trial MT19937 stream: a pure function of (base seed, trial number). Each region consumes the
        // stream in a fixed slot order (candidate generation here, then the Thompson z-draw in the competition),
        // so the winner reproduces bit-for-bit from (seed, labels).
        NumPyRandom rng = np.random.RandomState();
        rng.seed(Mix(_baseSeed, (ulong)trial.Number));

        int numCandidates = _numCandidates ?? DefaultNumCandidates(d);
        var regions = new List<TrustRegionSample>(m);
        for (int s = 0; s < m; s++)
            regions.Add(BuildRegionSample(s, slotTrials[s], states[s], optimizer, internalSpace, d, numCandidates, sign, rng));

        (int winnerIndex, int winnerCandidate) = TurboMOptimizer.SelectAcrossTrustRegions(regions, rng);
        TrustRegionSample winner = regions[winnerIndex];

        var chosen = new double[d];
        for (int j = 0; j < d; j++) chosen[j] = winner.Candidates[winnerCandidate, j];
        PersistTrustRegionLabel(study, trial, winner.Slot);
        return Unnormalize(internalSpace, searchSpace, chosen);
    }

    /// <summary>
    ///     Builds one ready region's cross-region-competition payload: fits the region's local GP on its
    ///     <b>current run only</b> (trials from its last restart onward, discarding pre-collapse points exactly
    ///     as the reference marks them <c>_idx = −1</c>), chooses the trust-region center (§2), builds the box +
    ///     Sobol candidate set, and evaluates the joint GP posterior over those candidates. The GP fit, center
    ///     rule, box, and candidate generation are the same per-region TuRBO-1 primitives the single-region path
    ///     uses. The returned payload also carries the region's score standardization constants so the
    ///     competition can un-standardize this region's Thompson draw into shared objective units.
    /// </summary>
    /// <param name="slot">The region's slot index (persisted as the label if this region wins).</param>
    /// <param name="slotTrials">All of this slot's labeled completed trials in evaluation order.</param>
    /// <param name="state">The slot's folded trust-region state (its current base length and restart boundary).</param>
    /// <param name="optimizer">The shared per-region trust-region optimizer (box + candidate generation).</param>
    /// <param name="internalSpace">The GP-internal normalized search space (parameter normalization + categorical flags).</param>
    /// <param name="d">The relative-search-space dimension.</param>
    /// <param name="numCandidates">The Thompson candidate-set size for this trial.</param>
    /// <param name="sign">+1 for a maximizing study, −1 for a minimizing study (maps values to maximization-oriented scores).</param>
    /// <param name="rng">The per-trial RNG; this region's candidate generation advances it in slot order.</param>
    /// <returns>The region's candidate set, standardized joint posterior, and score-standardization constants.</returns>
    private TrustRegionSample BuildRegionSample(
        int slot, IReadOnlyList<FrozenTrial> slotTrials, TurboTrState state, TurboOptimizer optimizer,
        GpSearchSpace internalSpace, int d, int numCandidates, double sign, NumPyRandom rng)
    {
        // Fit on the current run only — trials before the restart boundary were discarded on collapse.
        var restartTrials = new List<FrozenTrial>(slotTrials.Count - state.RestartStart);
        for (int i = state.RestartStart; i < slotTrials.Count; i++) restartTrials.Add(slotTrials[i]);
        int mRestart = restartTrials.Count;

        double[,] normalizedParams = internalSpace.GetNormalizedParams(
            restartTrials.Select(t => t.Params).ToList());

        // Standardize the run's maximization-oriented scores for the GP fit; keep mean/std to un-standardize the
        // Thompson draw for the cross-region comparison (each region standardizes by its OWN constants).
        var rawScore = new double[mRestart];
        for (int i = 0; i < mRestart; i++) rawScore[i] = ToScore(restartTrials[i].Value, sign);
        double[] clipped = ClipFinite(rawScore);
        double mean = MathUtil.Mean(clipped);
        double std = Math.Max(Eps, MathUtil.Std(clipped));
        var y = new double[mRestart];
        for (int i = 0; i < mRestart; i++) y[i] = (clipped[i] - mean) / std;

        // Local GP under the paper's §C hyperparameter box (same fit the single-region path uses per region).
        GPRegressor gpr = TurboGpFit.Fit(
            normalizedParams, y, internalSpace.IsCategorical, _minimumNoise, _deterministicObjective);

        // §2 center: best observation when noise-free, else the training point with the best posterior mean.
        double[]? posteriorMean = _deterministicObjective ? null : gpr.Posterior(normalizedParams).Mean;
        int incumbent = TurboOptimizer.SelectCenterIndex(clipped, posteriorMean, _deterministicObjective);
        var xCenter = new double[d];
        for (int j = 0; j < d; j++) xCenter[j] = normalizedParams[incumbent, j];

        (double[] lb, double[] ub) = optimizer.TrustRegionBounds(xCenter, gpr.LengthScales, state.Length);
        double[,] candidates = optimizer.GenerateCandidates(xCenter, lb, ub, numCandidates, rng);
        (double[] postMean, double[,] postCov) = gpr.PosteriorJoint(candidates);

        return new TrustRegionSample(slot, candidates, postMean, postCov, mean, std);
    }

    /// <summary>Reads a completed trial's persisted TuRBO-m region label (system attr <c>"turbo:tr"</c>), or
    ///     <c>−1</c> when it is absent — a trial from before this sampler ran, or an all-categorical trial whose
    ///     relative search space was empty and so never proposed a region.</summary>
    /// <param name="trial">The completed trial snapshot to read the label from.</param>
    /// <returns>The stored region slot index, or <c>−1</c> if the trial carries no valid label.</returns>
    private static int ReadTrustRegionLabel(FrozenTrial trial)
        => trial.SystemAttrs.TryGetValue(AttrKeyTrustRegion, out object? v) ? (int)Convert.ToInt64(v) : -1;

    /// <summary>Persists this trial's winning region slot as the system attr <c>"turbo:tr"</c> so the per-region
    ///     fold can reconstruct which region proposed it. With the init row (<see cref="PersistInitRow" />) it is the
    ///     only state TuRBO-m stores (a small int per trial); each worker writes only its own trial's label, so it is
    ///     parallel-worker-safe.</summary>
    /// <param name="study">The study whose storage backs the write.</param>
    /// <param name="trial">The trial receiving the label (identified by its trial id).</param>
    /// <param name="slot">The winning region slot index to persist.</param>
    private static void PersistTrustRegionLabel(Study study, FrozenTrial trial, int slot)
        => study.Storage.SetTrialSystemAttr(trial.TrialId, AttrKeyTrustRegion, slot);

    /// <summary>
    ///     The Sobol row of a (slot, restart) initialization design for this trial: the first row that no finished
    ///     trial of the restart and no other running trial of it holds, so trials asked at the same time take
    ///     distinct rows. Rows run past the design's <c>nInit</c> only when more trials are in flight than rows
    ///     remain; the scrambled Sobol sequence is prefix-stable, so those extra rows are its next points.
    /// </summary>
    /// <param name="study">
    ///     The study, whose complete and running trials are read from the storage in one read (their stamps are fresh
    ///     there, and a trial told since this trial read its history is still seen).
    /// </param>
    /// <param name="trial">The trial being sampled; it holds no row yet.</param>
    /// <param name="slot">The trust-region slot (0 for TuRBO-1).</param>
    /// <param name="restartCount">The slot's current restart, whose design is being filled.</param>
    /// <param name="finishedInRestart">The slot's finished trials since the restart began, in the history snapshot.</param>
    /// <returns>The row to take; equal to the finished count in a sequential run.</returns>
    /// <exception cref="NumSharp.KeyError">The study no longer exists in storage.</exception>
    private static int InitRow(
        Study study, FrozenTrial trial, int slot, int restartCount, IEnumerable<FrozenTrial> finishedInRestart)
    {
        var held = new List<long>();
        int unstamped = 0;
        foreach (FrozenTrial t in finishedInRestart)
        {
            if (TryReadInitRow(t, slot, restartCount, out long row))
                held.Add(row);
            else
                unstamped++;
        }

        // The stamp names the slot and restart, so trials of other restarts drop out here without the snapshot's
        // restart window.
        foreach (FrozenTrial t in ConcurrentAsks.OtherCompleteOrRunningTrials(study, trial))
            if (TryReadInitRow(t, slot, restartCount, out long row))
                held.Add(row);

        // A finished trial without a row (written before rows were stamped, or proposed in the previous restart's
        // optimization phase) took the next row in order back then: FirstFreeIndex gives each one the lowest row
        // still free before picking this trial's.
        return ConcurrentAsks.FirstFreeIndex(held, unstamped);
    }

    /// <summary>Reads a trial's init-row stamp when it belongs to the given slot's restart.</summary>
    /// <param name="trial">The trial whose system attributes are read.</param>
    /// <param name="slot">The slot the stamp must name.</param>
    /// <param name="restartCount">The restart the stamp must name.</param>
    /// <param name="row">The stamped row, when the stamp matches.</param>
    /// <returns><c>true</c> when the trial took a row of that slot's restart.</returns>
    private static bool TryReadInitRow(FrozenTrial trial, int slot, int restartCount, out long row)
    {
        row = -1;
        if (!trial.SystemAttrs.TryGetValue(AttrKeyInitRow, out object? v) || v is not IList<object?> parts || parts.Count != 3)
            return false;
        if (Convert.ToInt64(parts[0]) != slot || Convert.ToInt64(parts[1]) != restartCount)
            return false;
        row = Convert.ToInt64(parts[2]);
        return true;
    }

    /// <summary>
    ///     Stamps the trial with the initialization row it took, as the system attr <c>"turbo:init"</c> =
    ///     <c>[slot, restart, row]</c>, so concurrent asks of the same restart see the row as held.
    /// </summary>
    /// <param name="study">The study whose storage backs the write.</param>
    /// <param name="trial">The trial receiving the stamp.</param>
    /// <param name="slot">The trust-region slot (0 for TuRBO-1).</param>
    /// <param name="restartCount">The slot's current restart.</param>
    /// <param name="row">The Sobol row the trial took.</param>
    private static void PersistInitRow(Study study, FrozenTrial trial, int slot, int restartCount, int row)
        => study.Storage.SetTrialSystemAttr(trial.TrialId, AttrKeyInitRow, new List<object?> { slot, restartCount, row });

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

    /// <summary>
    ///     The paper's default failure tolerance <c>τ_fail = ⌈d/q⌉</c> (batch <c>q = 1</c> here, so
    ///     <c>= d</c>); Eriksson et al. 2019, §D. <b>Note the paper-vs-code discrepancy:</b> the authors' own
    ///     reference implementation and the BoTorch tutorial instead use <c>⌈max(4/q, d/q)⌉ = max(4, d)</c> as
    ///     a low-dimension guard (and the authors' TuRBO-<i>m</i> code uses <c>max(5, d)</c>). All three agree
    ///     for <c>d ≥ 5</c> — the paper's entire experimental regime — and differ only for small <c>d</c>; pass
    ///     an explicit <c>failureTolerance = max(4, d)</c> (or <c>max(5, d)</c> for the multi-region reference)
    ///     to reproduce that behavior there. This same default is used per region in TuRBO-m.
    /// </summary>
    /// <param name="d">The relative-search-space dimension.</param>
    /// <returns>The paper's <c>τ_fail = d</c>.</returns>
    internal static int DefaultFailureTolerance(int d) => d;

    /// <summary>
    ///     The paper's default Thompson-sampling candidate-set size, <c>min(100·d, 5000)</c> (Eriksson et al.
    ///     2019, §D). See the <c>numCandidates</c> constructor parameter for the <c>O(n³)</c>-per-trial cost
    ///     this exact managed joint-posterior draw pays at the upper end of that range.
    /// </summary>
    /// <param name="d">The relative-search-space dimension.</param>
    /// <returns>The paper's candidate-set size for dimension <paramref name="d" />.</returns>
    internal static int DefaultNumCandidates(int d) => Math.Min(100 * d, 5000);

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
