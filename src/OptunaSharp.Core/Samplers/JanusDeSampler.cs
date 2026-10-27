// JANUS-DE — Jacobian-Aligned infill on a Differential Evolution host (Plan 05 Phase 2b).
//
// Reference paper:
//   Hongyuan Yu, Pufan Xu, Jiaojiao Yi, Yiding Tian, Mingrui Sun, Jiayuan Lu, Changyuan Wen.
//   "JANUS: Online Jacobian-Aligned Infill for Black-Box Optimization." arXiv:2608.22862 (2026).
// Reference code: https://github.com/hongyuanyu/JANUS  (Apache-2.0).
//
// This is the paper's host-agnostic check: JANUS's Jacobian infill wrapped around a NON-CMA host. The
// same `JanusOptimizer` driver that powers `JanusSampler` runs here over a `DePopulationHost` instead of
// a `CmaPopulationHost`. Because DE exposes no proposal covariance, the driver automatically runs the
// conservative GN-only configuration (paper §"analogous box and step-length constraints"): the damped
// Gauss–Newton exploitation elite is bounded-injected into slot 0 of each DE generation and competes for
// its target through DE's ordinary greedy selection, while NO CMIX exploration candidates are offered
// (there is no covariance to build the inverse-curvature metric from). The residual metric that drives
// the elite still comes from the window (`JanusMetric` / `FitComponentJacobian`), exactly as for the CMA
// host. See JanusOptimizer.cs / DePopulationHost.cs for the algorithm and the documented divergences.

using System;
using System.Collections.Generic;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers.Differential;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     JANUS-DE: the training-free JANUS Jacobian infill on a Differential Evolution host. JANUS estimates
///     a local Jacobian from the recent evaluation trace and bounded-injects a damped Gauss–Newton
///     exploitation elite into each DE generation, while the <see cref="DifferentialEvolutionSampler" />-style
///     DE host keeps full control of selection. Because DE has no Gaussian proposal covariance, this is the
///     paper's <c>JANUS-GN</c> configuration — the exploitation injection only, no CMIX exploration
///     candidates. It is the host-agnostic complement to the CMA-hosted <see cref="JanusSampler" />.
///
///     <para><b>Residual coordinates (optional but recommended).</b> If the objective decomposes into
///     components <c>f(x) = mean_i r_i(x)</c>, expose the residual vector on each trial via
///     <c>trial.SetUserAttr("component_losses", new double[]{ r1, r2, … })</c>; JANUS then estimates the
///     local Jacobian directly from those residuals. Without it, JANUS falls back to a local
///     linear/quadratic fit on the recent window of scalar objective values (the constructor's <c>pseudoMode</c> option).</para>
///
///     <para><b>Beyond-core, in-memory state.</b> Like <see cref="JanusSampler" />, JANUS-DE holds its
///     optimizer (JANUS window + DE population) in memory for the study's duration; it does not resume its
///     infill state across process boundaries. Categorical and single-value distributions are handled by
///     the independent sampler.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(sampler: new JanusDeSampler(seed: 0, nStartupTrials: 16));
///     study.Optimize(t =>
///     {
///         double s = 0;
///         for (int i = 0; i &lt; 10; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); double r = (x - 0.5) * (x - 0.5); s += r; }
///         return s;
///     }, nTrials: 400);
///     </code>
/// </example>
public sealed class JanusDeSampler : BaseSampler
{
    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers.janus_de");

    private readonly int _seed;
    private readonly int? _popsize;
    private readonly int _nStartupTrials;
    private readonly BaseSampler _independentSampler;
    private bool _warnIndependentSampling;
    private readonly int _nJobs;
    private readonly IntersectionSearchSpace _searchSpace;

    private readonly double _f;
    private readonly double _cr;
    private readonly DeStrategy _strategy;

    private readonly int _fuseEvery;
    private readonly int? _fuseStart;
    private readonly double _kappa;
    private readonly int? _gnWindow;
    private readonly double _gnLambda;
    private readonly double _gnRidge;
    private readonly string _pseudoMode;
    private readonly string _gnPseudoMode;
    private readonly string _pseudoTarget;
    private readonly string _pcaK;

    // The optimizers of each study this instance serves, one per worker lane (trial.Number % nJobs, as in the
    // reference), keyed by storage object and study id: a single lane dictionary kept driving study A's DE
    // population in study B when the sampler was reused (pre-release review item H3). Lookups and inserts run
    // under _optimizersLock; the optimizers themselves are driven outside it, as before.
    private readonly PerStudyState<StudyLanes> _studyLanes = new();
    private readonly object _optimizersLock = new();

    /// <summary>The per-lane optimizers of one study.</summary>
    private sealed class StudyLanes
    {
        /// <summary>The optimizer of each worker lane, created on the lane's first relative sample.</summary>
        public readonly Dictionary<int, Janus.JanusOptimizer> Optimizers = new();

        /// <summary>
        ///     The lowest-numbered completed trial when the lanes were first used: if the study now stored under
        ///     this id no longer has it, the study was recreated and these optimizers are stale.
        /// </summary>
        public TrialAnchor? Anchor;
    }

    /// <param name="seed">Random seed. When <c>null</c>, a random seed is drawn (from MT19937, never System.Random).</param>
    /// <param name="popsize">DE population size NP (default <c>max(8, 4 + ⌊3 ln d⌋)</c>).</param>
    /// <param name="nStartupTrials">Independent sampling is used until this many trials finish.</param>
    /// <param name="independentSampler">Sampler for parameters outside the relative space (default <see cref="RandomSampler" />).</param>
    /// <param name="warnIndependentSampling">Warn when a parameter is sampled independently.</param>
    /// <param name="nJobs">Number of parallel workers; a separate optimizer is kept per <c>trial.Number % nJobs</c>.</param>
    /// <param name="f">DE differential weight F (default 0.8).</param>
    /// <param name="cr">DE crossover rate CR (default 0.9).</param>
    /// <param name="strategy">DE mutation strategy: <c>rand1bin</c> (default) or <c>best1bin</c>.</param>
    /// <param name="fuseEvery">Generations between Jacobian re-estimation (assembly cadence).</param>
    /// <param name="fuseStart">Window size before the first Jacobian assembly (<c>null</c> = auto).</param>
    /// <param name="kappa">Bounded-injection clip factor (whitened step ‖z‖ ≤ κ√d, in DE's per-dimension spread metric).</param>
    /// <param name="gnWindow">Sliding window of recent <c>(x, f)</c> pairs for the Jacobian regression (<c>null</c> = auto).</param>
    /// <param name="gnLambda">Levenberg–Marquardt damping for the Gauss–Newton step.</param>
    /// <param name="gnRidge">Ridge regularization for the residual-gradient regression.</param>
    /// <param name="pseudoMode">Scalar residual-coordinate fit: <c>linear</c> (default), <c>quadratic</c>, <c>diag_quadratic</c>, <c>pca_quadratic</c>, <c>diag_pca_quadratic</c>, or <c>none</c>.</param>
    /// <param name="gnPseudoMode">Override <paramref name="pseudoMode" /> for the GN direction (<c>null</c> = same).</param>
    /// <param name="pseudoTarget">Scalar-value transform for the fit: <c>raw</c> (default), <c>rank</c>, or <c>log</c>.</param>
    /// <param name="pcaK">PCA rank for the <c>pca_quadratic</c> fit (<c>auto</c> or an integer string).</param>
    /// <summary>Configures Jacobian-informed candidates injected into a differential-evolution population.</summary>
    public JanusDeSampler(
        int? seed = null,
        int? popsize = null,
        int nStartupTrials = 1,
        BaseSampler? independentSampler = null,
        bool warnIndependentSampling = true,
        int nJobs = 1,
        double f = 0.8,
        double cr = 0.9,
        string strategy = "rand1bin",
        int fuseEvery = 4,
        int? fuseStart = null,
        double kappa = 1.0,
        int? gnWindow = null,
        double gnLambda = 1.0,
        double gnRidge = 1.0,
        string pseudoMode = "linear",
        string? gnPseudoMode = null,
        string pseudoTarget = "raw",
        string pcaK = "auto")
    {
        // A missing seed is drawn from NumSharp's MT19937 (never System.Random — the project's RNG discipline).
        _seed = seed ?? (int)new LazyRandomState().Rng.randint(1, int.MaxValue);
        _popsize = popsize;
        _nStartupTrials = nStartupTrials;
        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _warnIndependentSampling = warnIndependentSampling;
        _nJobs = Math.Max(1, nJobs);
        _searchSpace = new IntersectionSearchSpace();

        _f = f;
        _cr = cr;
        _strategy = ParseStrategy(strategy);

        _fuseEvery = fuseEvery;
        _fuseStart = fuseStart;
        _kappa = kappa;
        _gnWindow = gnWindow;
        _gnLambda = gnLambda;
        _gnRidge = gnRidge;
        _pseudoMode = pseudoMode;
        _gnPseudoMode = gnPseudoMode ?? pseudoMode;
        _pseudoTarget = pseudoTarget;
        _pcaK = pcaK;
    }

    /// <summary>Maps supported differential-evolution strategy names to the internal strategy enum.</summary>
    /// <param name="strategy">Differential-evolution donor/crossover strategy name or enum.</param>
    /// <returns>The supported strategy, with unrecognized names resolved to the implementation default.</returns>
    private static DeStrategy ParseStrategy(string strategy)
    {
        string s = (strategy ?? "rand1bin").ToLowerInvariant().Replace("/", "").Replace("_", "").Replace("bin", "");
        return s switch
        {
            "best" or "best1" => DeStrategy.Best1Bin,
            _ => DeStrategy.Rand1Bin,
        };
    }

    /// <inheritdoc />
    public override void ReseedRng() => _independentSampler.ReseedRng();

    /// <inheritdoc />
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _searchSpace.Calculate(study))
        {
            if (distribution.Single()) continue;
            if (distribution is not FloatDistribution && distribution is not IntDistribution) continue;
            searchSpace[name] = distribution;
        }
        return searchSpace;
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">The study has multiple objectives, but this sampler requires a scalar objective.</exception>
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        RaiseErrorIfMultiObjective(study);

        if (searchSpace.Count <= 1)
            return new OrderedDict<string, object?>();

        var completed = GetTrials(study);
        if (completed.Count < _nStartupTrials)
            return new OrderedDict<string, object?>();

        int threadId = ((trial.Number % _nJobs) + _nJobs) % _nJobs;

        Janus.JanusOptimizer? optimizer;
        bool created;
        lock (_optimizersLock)
        {
            // This study's lanes. A completed trial never stops being completed, so lanes whose anchor is gone
            // were built for a study deleted and recreated under the same id: start this one afresh.
            StudyLanes lanes = _studyLanes.Resolve(
                study,
                static () => new StudyLanes(),
                state => state.Anchor is not { } anchor || anchor.IsIn(completed));
            if (lanes.Anchor is null && completed.Count > 0)
                lanes.Anchor = TrialAnchor.Of(completed[0]);

            created = !lanes.Optimizers.TryGetValue(threadId, out optimizer);
            if (created)
            {
                var trans = new SearchSpaceTransform(searchSpace, transformStep: true, transform01: true);
                // The DE host owns the population and its own MT19937 stream; the driver runs it through the
                // host-agnostic IPopulationHost seam (GN-only, since DE reports no proposal covariance).
                var host = new Janus.DePopulationHost(trans, _popsize, _seed + threadId, _f, _cr, _strategy);
                optimizer = new Janus.JanusOptimizer(
                    trans, host, completed, study.Direction, _seed + threadId,
                    _fuseEvery, _fuseStart, _kappa, _gnWindow, _gnLambda, _gnRidge,
                    // CMIX is inert on a covariance-free host — pass the GN-only ("none") configuration explicitly.
                    cmixW: null, cmixLambda: 1.0, cmixWBase: 0.5, cmixWMax: 0.20,
                    _gnPseudoMode, _gnPseudoMode, _pseudoTarget, _pcaK,
                    cmixMode: "none", cmixCandidateFrac: 0.0, cmixCandidateCenter: "mean");
                lanes.Optimizers[threadId] = optimizer;
            }
        }
        if (!created)
            optimizer!.SetCompletedTrials(completed);

        if (optimizer.Dim != searchSpace.Count)
        {
            if (_warnIndependentSampling)
            {
                Logger.Warning(
                    "`JanusDeSampler` does not support dynamic search space. " +
                    $"`{_independentSampler.GetType().Name}` is used instead of `JanusDeSampler`.");
                _warnIndependentSampling = false;
            }
            return new OrderedDict<string, object?>();
        }

        optimizer.UpdateEs();
        return optimizer.Ask(trial.Number);
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">The study has multiple objectives, but this sampler requires a scalar objective.</exception>
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        RaiseErrorIfMultiObjective(study);
        return _independentSampler.SampleIndependent(study, trial, paramName, paramDistribution);
    }

    /// <inheritdoc />
    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _independentSampler.BeforeTrial(study, trial);

    /// <inheritdoc />
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
        => _independentSampler.AfterTrial(study, trial, state, values);

    /// <summary>Collects historical trials eligible for this sampler, applying its completed/pruned-trial policy.</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <returns>Trial snapshots eligible for the sampler history.</returns>
    private List<FrozenTrial> GetTrials(Study study)
    {
        var result = new List<FrozenTrial>();
        foreach (var t in study.GetTrialsInternal(deepcopy: false, states: null, useCache: true))
            if (t.State == TrialState.Complete)
                result.Add(t);
        return result;
    }
}
