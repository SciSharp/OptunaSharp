// JANUS sampler — Jacobian-Aligned Newton-Unified Search (single-objective, CMA-ES host).
//
// Reference paper:
//   Hongyuan Yu, Pufan Xu, Jiaojiao Yi, Yiding Tian, Mingrui Sun, Jiayuan Lu, Changyuan Wen.
//   "JANUS: Online Jacobian-Aligned Infill for Black-Box Optimization." arXiv:2608.22862 (2026).
// Reference code: https://github.com/hongyuanyu/JANUS  (Apache-2.0). This is a native, paper-faithful
// port of the authors' `janus_optuna.JanusSampler`, hosted on OptunaSharp's own `Cma` (the reference
// drives an evotorch/torch CMA-ES). JANUS is NOT part of the Optuna ecosystem — like `LmCmaSampler`,
// it is validated by convergence, with no upstream trajectory to bit-match. See JanusOptimizer.cs for
// the algorithm and the documented, principled divergences (MT19937 RNG, SearchSpaceTransform box,
// square-root-invariant bounded injection, scalar-window extension).

using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     A plug-and-play, training-free JANUS sampler. JANUS estimates a local Jacobian from the recent
///     evaluation trace and uses the <i>same</i> Jacobian for both a damped Gauss–Newton exploitation
///     candidate (bounded-injected into the population) and a trace-preserving inverse-curvature
///     exploration metric (a fraction of the offspring), while a CMA-ES host keeps full control of
///     selection, covariance adaptation, and step-size control. Port of the authors' reference
///     <c>janus_optuna.JanusSampler</c> (arXiv:2608.22862; https://github.com/hongyuanyu/JANUS).
///
///     <para><b>JANUS-GN vs full JANUS.</b> Passing <c>cmixCandidateFrac: 0</c> (or <c>cmixMode:
///     "none"</c>) disables the CMIX exploration candidates, leaving only the Gauss–Newton
///     exploitation injection — the paper's conservative <c>JANUS-GN</c> variant. The default is full
///     JANUS (GN + CMIX).</para>
///
///     <para><b>Residual coordinates (optional but recommended).</b> If the objective decomposes into
///     components <c>f(x) = mean_i r_i(x)</c>, expose the residual vector on each trial via
///     <c>trial.SetUserAttr("component_losses", new double[]{ r1, r2, … })</c>; JANUS then estimates
///     the local Jacobian directly from those residuals, where its geometric signal is sharpest.
///     Without it, JANUS falls back to a local linear/quadratic fit on the recent window of scalar
///     objective values (the paper's BBOB "scalar residual coordinate" mode — see <c>pseudoMode</c>).</para>
///
///     <para><b>Beyond-core, in-memory state.</b> Unlike <see cref="CmaEsSampler" />, which serializes
///     its optimizer into trial system attrs (for RDB / distributed resume), JANUS holds its optimizer
///     in memory for the study's duration — matching the reference implementation. A JANUS study
///     therefore does not resume its infill state across process boundaries. Categorical and
///     single-value distributions are handled by the independent sampler, as in <see cref="CmaEsSampler" />.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(sampler: new JanusSampler(seed: 0, nStartupTrials: 16));
///     study.Optimize(t =>
///     {
///         double s = 0;
///         for (int i = 0; i &lt; 10; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); double r = (x - 0.5) * (x - 0.5); s += r; }
///         return s;
///     }, nTrials: 300);
///     </code>
/// </example>
public sealed class JanusSampler : BaseSampler
{
    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers.janus");

    private readonly double? _sigma0;
    private readonly int _seed;
    private readonly int? _popsize;
    private readonly int _nStartupTrials;
    private readonly BaseSampler _independentSampler;
    private bool _warnIndependentSampling;
    private readonly int _nJobs;
    private readonly IntersectionSearchSpace _searchSpace;

    private readonly int _fuseEvery;
    private readonly int? _fuseStart;
    private readonly double _kappa;
    private readonly int? _gnWindow;
    private readonly double _gnLambda;
    private readonly double _gnRidge;
    private readonly double? _cmixW;
    private readonly double _cmixLambda;
    private readonly double _cmixWBase;
    private readonly double _cmixWMax;
    private readonly string _pseudoMode;
    private readonly string _gnPseudoMode;
    private readonly string _cmixPseudoMode;
    private readonly string _pseudoTarget;
    private readonly string _pcaK;
    private readonly string _cmixMode;
    private readonly double _cmixCandidateFrac;
    private readonly string _cmixCandidateCenter;

    private readonly Dictionary<int, Janus.JanusOptimizer> _optimizers = new();

    /// <param name="sigma0">Initial CMA-ES step size (accepted for signature parity; the host's initial step is <c>min_range/6</c> in the transformed box, as in the reference).</param>
    /// <param name="seed">Random seed. When <c>null</c>, a random seed is drawn.</param>
    /// <param name="popsize">Population size λ (default <c>4 + ⌊3 ln d⌋</c>).</param>
    /// <param name="nStartupTrials">Independent sampling is used until this many trials finish.</param>
    /// <param name="independentSampler">Sampler for parameters outside the relative space (default <see cref="RandomSampler" />).</param>
    /// <param name="warnIndependentSampling">Warn when a parameter is sampled independently.</param>
    /// <param name="nJobs">Number of parallel workers; a separate optimizer is kept per <c>trial.Number % nJobs</c> (as in the reference).</param>
    /// <param name="fuseEvery">Generations between Jacobian re-estimation (assembly cadence).</param>
    /// <param name="fuseStart">Window size before the first Jacobian assembly (<c>null</c> = auto).</param>
    /// <param name="kappa">Bounded-injection clip factor (whitened step ‖z‖ ≤ κ√d).</param>
    /// <param name="gnWindow">Sliding window of recent <c>(x, f)</c> pairs for the Jacobian regression (<c>null</c> = auto).</param>
    /// <param name="gnLambda">Levenberg–Marquardt damping for the Gauss–Newton step.</param>
    /// <param name="gnRidge">Ridge regularization for the residual-gradient regression.</param>
    /// <param name="cmixW">CMIX absolute mixing weight for the reshape mode (<c>null</c> = auto); unused by the default candidate/portfolio mode.</param>
    /// <param name="cmixLambda">Levenberg–Marquardt damping for the CMIX inverse-curvature metric.</param>
    /// <param name="cmixWBase">Base CMIX mixing weight before dimension scaling.</param>
    /// <param name="cmixWMax">Upper clip for the auto CMIX mixing weight.</param>
    /// <param name="pseudoMode">Scalar residual-coordinate fit: <c>linear</c> (default), <c>quadratic</c>, <c>diag_quadratic</c>, <c>pca_quadratic</c>, <c>diag_pca_quadratic</c>, or <c>none</c>.</param>
    /// <param name="gnPseudoMode">Override <paramref name="pseudoMode" /> for the GN direction (<c>null</c> = same).</param>
    /// <param name="cmixPseudoMode">Override <paramref name="pseudoMode" /> for the CMIX metric (<c>null</c> = same).</param>
    /// <param name="pseudoTarget">Scalar-value transform for the fit: <c>raw</c> (default), <c>rank</c>, or <c>log</c>.</param>
    /// <param name="pcaK">PCA rank for the <c>pca_quadratic</c> fit (<c>auto</c> or an integer string).</param>
    /// <param name="cmixMode"><c>portfolio</c> (default) / <c>candidates</c> draw CMIX samples into tail slots; <c>reshape</c> mixes the inverse-curvature metric into the host covariance in place (component mode; no-op on a covariance-free host); <c>none</c> disables CMIX (JANUS-GN).</param>
    /// <param name="cmixCandidateFrac">Fraction of offspring drawn from the JANUS exploration metric (default 0.4). Set to 0 for JANUS-GN.</param>
    /// <param name="cmixCandidateCenter">Center of the CMIX samples: <c>mean</c> (default), <c>best</c>, <c>elite</c>, or <c>archive</c> (a diverse Pareto-archive of residual-window points, round-robined).</param>
    public JanusSampler(
        double? sigma0 = null,
        int? seed = null,
        int? popsize = null,
        int nStartupTrials = 1,
        BaseSampler? independentSampler = null,
        bool warnIndependentSampling = true,
        int nJobs = 1,
        int fuseEvery = 4,
        int? fuseStart = null,
        double kappa = 1.0,
        int? gnWindow = null,
        double gnLambda = 1.0,
        double gnRidge = 1.0,
        double? cmixW = null,
        double cmixLambda = 1.0,
        double cmixWBase = 0.5,
        double cmixWMin = 0.01,
        double cmixWMax = 0.20,
        string pseudoMode = "linear",
        string? gnPseudoMode = null,
        string? cmixPseudoMode = null,
        string pseudoTarget = "raw",
        string pcaK = "auto",
        string cmixMode = "portfolio",
        double cmixCandidateFrac = 0.4,
        string cmixCandidateCenter = "mean")
    {
        _sigma0 = sigma0;
        // A missing seed is drawn from NumSharp's MT19937 (never System.Random — the project's RNG
        // discipline); the reference draws it from Python's random.randint for the same purpose.
        _seed = seed ?? (int)new LazyRandomState().Rng.randint(1, int.MaxValue);
        _popsize = popsize;
        _nStartupTrials = nStartupTrials;
        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _warnIndependentSampling = warnIndependentSampling;
        _nJobs = Math.Max(1, nJobs);
        _searchSpace = new IntersectionSearchSpace();

        _fuseEvery = fuseEvery;
        _fuseStart = fuseStart;
        _kappa = kappa;
        _gnWindow = gnWindow;
        _gnLambda = gnLambda;
        _gnRidge = gnRidge;
        _cmixW = cmixW;
        _cmixLambda = cmixLambda;
        _cmixWBase = cmixWBase;
        _cmixWMax = cmixWMax;
        _ = cmixWMin;   // accepted for signature parity with the reference (auto-min uses w_base/d)
        _pseudoMode = pseudoMode;
        _gnPseudoMode = gnPseudoMode ?? pseudoMode;
        _cmixPseudoMode = cmixPseudoMode ?? pseudoMode;
        _pseudoTarget = pseudoTarget;
        _pcaK = pcaK;
        _cmixMode = cmixMode;
        _cmixCandidateFrac = cmixCandidateFrac;
        _cmixCandidateCenter = cmixCandidateCenter;
    }

    public override void ReseedRng()
    {
        _independentSampler.ReseedRng();
    }

    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _searchSpace.Calculate(study))
        {
            // A single-valued distribution is sampled inside Trial; categoricals are handled
            // independently (the CMA-ES host works on Float/Int in the transformed box).
            if (distribution.Single()) continue;
            if (distribution is not FloatDistribution && distribution is not IntDistribution) continue;
            searchSpace[name] = distribution;
        }
        return searchSpace;
    }

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

        if (!_optimizers.TryGetValue(threadId, out var optimizer))
        {
            var trans = new SearchSpaceTransform(searchSpace, transformStep: true, transform01: true);
            // The CMA-ES host owns the proposal (initial mean/step in the transformed box) and its own
            // MT19937 stream; the driver drives it through the host-agnostic IPopulationHost seam and
            // keeps a separate CMIX-noise stream seeded from the same value (Plan 03).
            var host = new Janus.CmaPopulationHost(trans, _sigma0, _popsize, _seed + threadId);
            optimizer = new Janus.JanusOptimizer(
                trans, host, completed, study.Direction, _seed + threadId,
                _fuseEvery, _fuseStart, _kappa, _gnWindow, _gnLambda, _gnRidge,
                _cmixW, _cmixLambda, _cmixWBase, _cmixWMax,
                _gnPseudoMode, _cmixPseudoMode, _pseudoTarget, _pcaK,
                _cmixMode, _cmixCandidateFrac, _cmixCandidateCenter);
            _optimizers[threadId] = optimizer;
        }
        else
        {
            optimizer.SetCompletedTrials(completed);
        }

        if (optimizer.Dim != CountEncodedDims(searchSpace))
        {
            if (_warnIndependentSampling)
            {
                Logger.Warning(
                    "`JanusSampler` does not support dynamic search space. " +
                    $"`{_independentSampler.GetType().Name}` is used instead of `JanusSampler`.");
                _warnIndependentSampling = false;
            }
            return new OrderedDict<string, object?>();
        }

        optimizer.UpdateEs();
        return optimizer.Ask(trial.Number);
    }

    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        RaiseErrorIfMultiObjective(study);
        return _independentSampler.SampleIndependent(study, trial, paramName, paramDistribution);
    }

    public override void BeforeTrial(Study study, FrozenTrial trial)
    {
        _independentSampler.BeforeTrial(study, trial);
    }

    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        _independentSampler.AfterTrial(study, trial, state, values);
    }

    private List<FrozenTrial> GetTrials(Study study)
    {
        var result = new List<FrozenTrial>();
        foreach (var t in study.GetTrialsInternal(deepcopy: false, states: null, useCache: true))
            if (t.State == TrialState.Complete)
                result.Add(t);
        return result;
    }

    /// <summary>Number of encoded (transformed) dimensions — Float/Int each encode to one column.</summary>
    private static int CountEncodedDims(OrderedDict<string, BaseDistribution> searchSpace) => searchSpace.Count;
}
