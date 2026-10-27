// Sampler for REMBO — Random EMbedding Bayesian Optimization (Wang, Hutter, Zoghi, Matheson & de Freitas,
// JAIR 55 (2016) 361-387, doi:10.1613/jair.4806). REMBO is a *meta*-sampler: it draws a random Gaussian
// embedding A ∈ ℝ^{D×d} once, runs an ordinary Bayesian optimizer in the low-dimensional latent box
// Y = [−√d, √d]^d, and maps each proposed latent point y back to the study's real D parameters via
// x = p_X(A·y). It only helps — dramatically — when the objective has low *effective* dimensionality
// (Definition 1), which is common in hyperparameter tuning where a few knobs dominate. REMBO has no
// Optuna counterpart, so — like TuRBO / LM-CMA / the NES family — this is a native, convergence-validated
// port on NumSharp's MT19937 (never System.Random).

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers.RandomEmbedding;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     Which of REMBO's two kernels (Wang et al. 2016, §3.2) the inner Bayesian optimizer uses.
/// </summary>
public enum RandomEmbeddingKernel
{
    /// <summary>
    ///     The low-dimensional kernel: the inner sampler models the objective as a function of the latent
    ///     point <c>y</c> directly (a GP in <c>d</c> dimensions). Cheap and the paper's choice for very high
    ///     extrinsic dimension, but it can waste effort on the large part of the box <c>Y</c> that projects
    ///     onto the boundary of the real box (two far-apart <c>y</c>'s can map to nearly the same point).
    /// </summary>
    LowDimensional = 0,

    /// <summary>
    ///     The high-dimensional kernel: the GP measures similarity in the projected real parameter space
    ///     <c>p_X(A·y)</c>, so <c>y</c>'s that map to the same clipped point are treated as identical —
    ///     removing the boundary over-exploration and making a single embedding far more reliable. It fits a
    ///     GP in the real (embedded) parameter space and optimizes expected improvement over the reachable
    ///     subspace, so it is best suited to moderate extrinsic dimension, and it also embeds categorical
    ///     parameters (the §3.2 categorical/Hamming kernel).
    /// </summary>
    HighDimensional = 1
}

/// <summary>
///     A sampler implementing REMBO — Random EMbedding Bayesian Optimization (Wang et al. 2016). It
///     draws a fixed random Gaussian matrix <c>A ∈ ℝ^{D×d}</c>, runs the paper's inner Bayesian optimizer
///     (by default a squared-exponential GP with expected improvement maximized by DIRECT + CMA-ES and
///     the Algorithm-3 adaptive length scale) in the small box <c>Y = [−√d, √d]^d</c>,
///     and maps each latent point <c>y</c> to the study's real parameters through <c>x = p_X(A·y)</c>
///     (a box-constrained projection). This lets BO scale to a search space of (in principle) enormous
///     dimension <c>D</c> as long as the objective's <b>effective</b> dimensionality <c>d_e</c> is low —
///     the invariance the paper proves and demonstrates up to a billion dimensions.
///
///     <para><b>The one knob that matters: <c>nEmbeddingDims</c> (d).</b> It must be at least the
///     objective's (guessed) effective dimensionality for the embedding to contain the optimum; larger
///     <c>d</c> raises the chance the box captures the optimum but slows the inner optimizer (Section 3.1).
///     There is no universal default — it reflects a belief about the problem — so it is required.</para>
///
///     <para><b>Interleaved runs (<c>nEmbeddings</c>, k).</b> A single embedding's box misses the optimum
///     with some probability <c>δ</c>; running <c>k</c> independent embeddings drops that to <c>δ^k</c>
///     (Section 3.1). With <c>k &gt; 1</c> trials are assigned to embeddings round-robin, each embedding
///     keeping its own latent history.</para>
///
///     <para><b>How it fits the Optuna protocol.</b> Every real parameter is produced through
///     <see cref="SampleIndependent" /> from a single latent point sampled once per trial — so all real
///     parameters of a trial come from one coherent <c>y</c>, and even the very first trial is embedded
///     (its observation feeds the inner optimizer). By default the latent point is chosen by REMBO's
///     paper-faithful optimizer — a squared-exponential GP with expected improvement maximized by
///     DIRECT + CMA-ES and the Algorithm-3 adaptive length scale — fit on the finished trials' latent
///     points; an overridden inner sampler is driven over a private in-memory <i>shadow study</i> of them
///     instead. Either way the latent point is persisted in the real trial's system attributes so the run
///     can be rebuilt after resumption. <b>Float and integer</b> parameters are embedded (integers/steps/log honored via
///     the standard de-normalization); <b>categorical and single-valued</b> parameters are delegated to an
///     independent sampler, as in <see cref="CmaEsSampler" /> / <see cref="TurboSampler" />. Single-objective
///     only.</para>
///
///     <para><b>Beyond-core, convergence-validated.</b> REMBO has no Optuna counterpart and its inner GP is
///     stochastic, so — like <see cref="TurboSampler" /> and <see cref="LmCmaSampler" /> — it is validated
///     by convergence (embedding recovery on a low-effective-dimension objective) and same-seed determinism
///     under sequential optimization, not by byte-parity.</para>
/// </summary>
/// <example>
///     <code>
///     // A 2-effective-dimension objective hidden inside 25 real parameters: REMBO with d = 4 finds it.
///     var study = Optuna.CreateStudy(sampler: new RandomEmbeddingSampler(nEmbeddingDims: 4, seed: 42));
///     study.Optimize(t =>
///     {
///         double sum = 0;
///         for (int i = 0; i &lt; 25; i++)
///         {
///             double x = t.SuggestFloat($"x{i}", -5, 5);
///             if (i &lt; 2) sum += (x - 2) * (x - 2);   // only x0, x1 matter
///         }
///         return sum;
///     }, nTrials: 200);
///     </code>
/// </example>
public sealed class RandomEmbeddingSampler : BaseSampler
{
    /// <summary>Trial system-attribute key holding the trial's latent coordinates (a list of <c>d</c> doubles).</summary>
    private const string LatentAttrKey = "rembo:latent";

    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers.rembo");

    private readonly RandomEmbeddingProjection _embedding;
    private readonly RandomEmbeddingKernel _kernel;

    // The inner Bayesian optimizer. When null (the default) REMBO uses the paper-faithful proposer
    // (squared-exponential GP + DIRECT/CMA-ES + Algorithm 3); a supplied one is driven over a private shadow
    // study instead — the beyond-paper "bring your own inner BO" override.
    private readonly BaseSampler? _innerSampler;
    private readonly BaseSampler _independentSampler;
    private readonly bool _warnIndependentSampling;

    // The paper-faithful inner optimizer (Wang et al. 2016, §3.2-3.3) runs when `_innerSampler` is null; it is
    // built per study (StudyEmbeddingState.Proposer) because its Algorithm-3 state advances with that study's
    // history. `_searchSpace` supplies the embedded real search space the high-dimensional kernel projects
    // into; `_baseSeed` anchors both the embedding and the proposer's deterministic streams.
    private readonly int _nStartupTrials;
    private readonly IntersectionSearchSpace _searchSpace;
    private readonly uint _baseSeed;

    // All shadow-study bookkeeping and latent-point derivation happen under this lock: the inner sampler
    // instance is shared across the k shadow studies, and the whole latent sampling is fast relative to a
    // (typically expensive) objective evaluation, which runs outside the lock. That bookkeeping is kept per
    // study (storage object and study id): instance-wide maps keyed by trial id and run handed study A's latent
    // points, shadow studies and GP state to study B when the sampler was reused (pre-release review item H3).
    private readonly object _gate = new();
    private readonly PerStudyState<StudyEmbeddingState> _studies = new();
    private OrderedDict<string, BaseDistribution>? _latentSpace;

    /// <summary>Everything REMBO accumulates for one study: latent points, shadow studies and the GP proposer.</summary>
    private sealed class StudyEmbeddingState
    {
        /// <summary>
        ///     Each sampled trial's latent point by real trial id, with the trial's identity: a hit only counts when the
        ///     trial asking is the one that was cached (trial ids repeat when a storage reuses them).
        /// </summary>
        public readonly Dictionary<int, (TrialAnchor Trial, LatentPoint Point)> LatentCache = new();

        /// <summary>The shadow study of each embedding (run) index, for an overridden inner sampler.</summary>
        public readonly Dictionary<int, Study> ShadowStudies = new();

        /// <summary>By run: the real trial numbers already mirrored into its shadow study.</summary>
        public readonly Dictionary<int, HashSet<int>> SyncedNumbers = new();

        /// <summary>The paper-faithful proposer of this study, created on first use when no inner sampler is set.</summary>
        public RemboGpProposer? Proposer;

        /// <summary>
        ///     The first trial this state served: if the study now stored under this id no longer has it, the study
        ///     was recreated and this state is stale. <c>null</c> until a latent point is computed.
        /// </summary>
        public TrialAnchor? Anchor;
    }

    /// <summary>A trial's assignment: which embedding it used and the latent point that produced its parameters.</summary>
    /// <param name="Run">The embedding (run) index in <c>[0, k)</c> this trial belongs to.</param>
    /// <param name="Y">The latent coordinates <c>y ∈ Y</c> that were mapped to the real parameters.</param>
    private readonly record struct LatentPoint(int Run, double[] Y);

    /// <param name="nEmbeddingDims">The internal embedding dimension <c>d</c> the inner optimizer searches
    ///     over — the core REMBO knob. Set it to at least the guessed effective dimensionality of the
    ///     objective; larger values raise the chance the box captures the optimum but slow convergence.</param>
    /// <param name="innerSampler">Optional override for the inner Bayesian optimizer. <c>null</c> (the
    ///     default) runs REMBO's paper-faithful optimizer directly (a squared-exponential GP with expected
    ///     improvement maximized by DIRECT + CMA-ES and the Algorithm-3 adaptive length scale). Supplying one
    ///     instead drives it over a private shadow study of the latent points — a beyond-paper generalization
    ///     for bringing another history-reconstructing sampler (e.g. <see cref="TPESampler" />); samplers that
    ///     persist an optimizer blob across trials (CMA-ES / ES) are not intended here, and the
    ///     <paramref name="kernel" /> choice does not apply to an overridden sampler.</param>
    /// <param name="nEmbeddings">The number of interleaved independent embeddings <c>k</c> (Section 3.1),
    ///     default 1; trials are assigned to embeddings round-robin.</param>
    /// <param name="kernel">Which REMBO kernel (§3.2) the paper-faithful optimizer uses. The default
    ///     <see cref="RandomEmbeddingKernel.LowDimensional" /> models the objective over the latent <c>y</c>;
    ///     <see cref="RandomEmbeddingKernel.HighDimensional" /> fits the GP in the projected real parameter
    ///     space (more reliable for a single embedding, and it also embeds categorical parameters). Applies
    ///     only to the default optimizer, not to an overridden <paramref name="innerSampler" />.</param>
    /// <param name="independentSampler">Sampler for parameters REMBO does not embed — single-valued, and
    ///     (under the low-dimensional kernel) categorical; <c>null</c> uses <see cref="RandomSampler" />.</param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently rather
    ///     than embedded.</param>
    /// <param name="nStartupTrials">Completed trials of an embedding required before its GP takes over from
    ///     uniform-random latent sampling (default 10). Used by the paper-faithful default optimizer.</param>
    /// <param name="seed">Random seed anchoring the embedding matrix and the default inner/independent
    ///     samplers; a fixed seed reproduces the trajectory under sequential (single-worker) optimization.</param>
    /// <summary>Configures the random embedding, its inner Bayesian optimizer (low- or high-dimensional
    ///     kernel), and the independent fallback.</summary>
    /// <exception cref="System.ArgumentException"><paramref name="nEmbeddingDims" /> or <paramref name="nEmbeddings" /> is below one.</exception>
    public RandomEmbeddingSampler(
        int nEmbeddingDims,
        BaseSampler? innerSampler = null,
        int nEmbeddings = 1,
        RandomEmbeddingKernel kernel = RandomEmbeddingKernel.LowDimensional,
        BaseSampler? independentSampler = null,
        bool warnIndependentSampling = true,
        int nStartupTrials = 10,
        int? seed = null)
    {
        if (nEmbeddingDims < 1)
            throw new ArgumentException("nEmbeddingDims must be at least 1.", nameof(nEmbeddingDims));
        if (nEmbeddings < 1)
            throw new ArgumentException("nEmbeddings must be at least 1.", nameof(nEmbeddings));
        if (nStartupTrials < 1)
            throw new ArgumentException("nStartupTrials must be at least 1.", nameof(nStartupTrials));

        // Anchor the embedding matrix through a LazyRandomState so a null seed draws from OS entropy without
        // ever touching System.Random, and a fixed seed makes the matrix (and thus the whole run) reproducible.
        var seedRng = new LazyRandomState(seed);
        _baseSeed = (uint)seedRng.Rng.randint(0, int.MaxValue);
        _embedding = new RandomEmbeddingProjection(_baseSeed, nEmbeddingDims, nEmbeddings);
        _kernel = kernel;
        _searchSpace = new IntersectionSearchSpace();

        // A null inner sampler selects the paper-faithful default (SE-GP + DIRECT/CMA-ES + Algorithm 3); a
        // supplied one is kept as the shadow-study override. The independent fallback handles the parameters
        // REMBO does not embed.
        _innerSampler = innerSampler;
        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _warnIndependentSampling = warnIndependentSampling;

        // The paper-faithful proposer (default path only) is built per study from these settings; see
        // StudyEmbeddingState.Proposer.
        _nStartupTrials = nStartupTrials;
    }

    /// <inheritdoc />
    public override void ReseedRng()
    {
        // Reseed the delegated samplers' streams for parallel workers; the embedding matrix stays fixed on
        // purpose (reseeding it would silently reinterpret every already-stored latent point).
        _innerSampler.ReseedRng();
        _independentSampler.ReseedRng();
    }

    /// <inheritdoc />
    /// <remarks>REMBO produces every parameter through <see cref="SampleIndependent" /> from one shared latent
    ///     point, so it declares no relative search space (like <see cref="RandomSampler" />).</remarks>
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
        => new();

    /// <inheritdoc />
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
        => new();

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">The study has multiple objectives, but REMBO requires a scalar objective.</exception>
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        RaiseErrorIfMultiObjective(study);

        // Parameters REMBO cannot embed go to the independent sampler (as in CmaEsSampler / TurboSampler):
        // single-valued always, and — under the low-dimensional kernel — categoricals, which cannot be
        // recovered from a single projected scalar. The high-dimensional kernel does embed categoricals
        // (the §3.2 categorical/Hamming kernel, via scale-and-round).
        if (!IsEmbedded(paramDistribution))
        {
            if (_warnIndependentSampling)
                LogIndependentSampling(trial, paramName);
            return _independentSampler.SampleIndependent(study, trial, paramName, paramDistribution);
        }

        LatentPoint latent = GetOrComputeLatent(study, trial);
        return _embedding.ToExternal(latent.Run, paramName, paramDistribution, latent.Y);
    }

    /// <summary>Whether a parameter is embedded by REMBO (versus delegated to the independent sampler).</summary>
    /// <param name="distribution">The parameter's distribution.</param>
    /// <returns><c>true</c> for float and integer parameters, and (under the high-dimensional kernel) for
    ///     categoricals; <c>false</c> for single-valued parameters and (under the low-dimensional kernel)
    ///     categoricals.</returns>
    private bool IsEmbedded(BaseDistribution distribution)
    {
        if (distribution.Single())
            return false;
        if (distribution is FloatDistribution or IntDistribution)
            return true;
        return distribution is CategoricalDistribution && _kernel == RandomEmbeddingKernel.HighDimensional;
    }

    /// <summary>
    ///     Returns this trial's latent point, sampling it once (via the low- or high-dimensional kernel) and
    ///     caching it so every real parameter of the trial maps from the same <c>y</c>.
    /// </summary>
    /// <param name="study">The real study whose finished trials seed the inner optimizer.</param>
    /// <param name="trial">The running trial whose latent point is needed.</param>
    /// <returns>The trial's embedding (run) index and its latent coordinates.</returns>
    private LatentPoint GetOrComputeLatent(Study study, FrozenTrial trial)
    {
        lock (_gate)
        {
            StudyEmbeddingState state = _studies.Resolve(study, static () => new StudyEmbeddingState());

            // Cached within this process for the trial's whole lifetime (all its parameter suggestions). The
            // identity check keeps a reused trial id (a recreated study) from picking up another trial's point.
            if (state.LatentCache.TryGetValue(trial.TrialId, out var cached) && cached.Trial.Matches(trial))
                return cached.Point;

            // A miss is the once-per-trial path, where reading the study's trial list is affordable: drop a state
            // built for a study since deleted and recreated under this id before its GP or shadow studies are used.
            if (state.Anchor is { } anchor && !anchor.IsIn(study.GetTrialsView()))
                state = _studies.Resolve(study, static () => new StudyEmbeddingState(), static _ => false);
            state.Anchor ??= TrialAnchor.Of(trial);

            int run = _embedding.RunForTrial(trial.Number);

            // Resumption safety: if this trial was already partially sampled (in a previous process), reuse
            // the persisted latent point so the trial's parameters stay mutually consistent.
            if (TryReadStoredLatent(trial, out double[] storedY))
            {
                var restored = new LatentPoint(run, storedY);
                state.LatentCache[trial.TrialId] = (TrialAnchor.Of(trial), restored);
                return restored;
            }

            // Default path → the paper-faithful proposer (both §3.2 kernels + Algorithm 3); an overridden
            // inner sampler → the shadow-study path over the latent y.
            double[] y = _innerSampler is null
                ? ComputeLatentPaper(study, state, run)
                : ComputeLatentLowDim(study, state, run);

            // Persist the latent point so future rebuilds (and resumption) can recover this trial.
            study.Storage.SetTrialSystemAttr(trial.TrialId, LatentAttrKey, y.Cast<object?>().ToList());

            var result = new LatentPoint(run, y);
            state.LatentCache[trial.TrialId] = (TrialAnchor.Of(trial), result);
            return result;
        }
    }

    /// <summary>Proposes a latent point with the low-dimensional kernel: the inner sampler over the shadow study.</summary>
    /// <param name="study">The real study whose finished trials seed the shadow study.</param>
    /// <param name="state">The real study's REMBO state, holding its shadow studies; caller holds <c>_gate</c>.</param>
    /// <param name="run">The embedding (run) index the shadow study serves.</param>
    /// <returns>The proposed latent coordinates.</returns>
    private double[] ComputeLatentLowDim(Study study, StudyEmbeddingState state, int run)
    {
        Study shadow = GetOrCreateShadow(study, state, run);
        SyncShadow(study, state, shadow, run);

        _latentSpace ??= _embedding.LatentSearchSpace();

        // Ask the inner optimizer for a latent point, then fail the ask-trial: leaving it RUNNING would
        // make the GP treat it as a pending evaluation on every subsequent ask (it counts running trials
        // that carry all search-space keys), while completing it would inject a bogus observation. A
        // FAILED trial is invisible to the inner sampler; the real observation is mirrored later by SyncShadow.
        Trial ask = shadow.Ask(_latentSpace);
        double[] y = ReadLatent(ask);
        shadow.Tell(ask, values: null, state: TrialState.Fail);
        return y;
    }

    /// <summary>
    ///     Proposes a latent point with REMBO's paper-faithful optimizer (Wang et al. 2016, §3.2-3.3): a
    ///     squared-exponential GP whose expected improvement is maximized over <c>Y</c> by DIRECT + CMA-ES
    ///     with the Algorithm-3 adaptive length scale, fit on this embedding's completed-trial history
    ///     (the latent <c>y</c> for the low-dimensional kernel, or the projected real parameters for the
    ///     high-dimensional one).
    /// </summary>
    /// <param name="study">The real study supplying this embedding's completed trials and their values.</param>
    /// <param name="state">The real study's REMBO state, holding its proposer; caller holds <c>_gate</c>.</param>
    /// <param name="run">The embedding (run) index whose history trains the GP.</param>
    /// <returns>The proposed latent coordinates.</returns>
    private double[] ComputeLatentPaper(Study study, StudyEmbeddingState state, int run)
    {
        OrderedDict<string, BaseDistribution> embeddedSpace = EmbeddedSpace(study);
        double sign = study.Direction == StudyDirection.Minimize ? -1.0 : 1.0;

        // This embedding's completed trials in evaluation order, as (stored latent point, maximization value).
        // A trial without a finite value or a recoverable latent point (e.g. all-categorical under the
        // low-dimensional kernel) carries no observation the GP can use, so it is skipped.
        var latents = new List<double[]>();
        var values = new List<double>();
        foreach (var t in study.GetTrialsInternal(deepcopy: false, states: new[] { TrialState.Complete }, useCache: true)
                     .OrderBy(t => t.Number))
        {
            if (_embedding.RunForTrial(t.Number) != run) continue;
            if (t.Value is not double v || double.IsNaN(v) || double.IsInfinity(v)) continue;
            if (!TryReadStoredLatent(t, out double[] y)) continue;
            latents.Add(y);
            values.Add(sign * v);
        }

        // Built on the study's first GP proposal. Its high-dimensional kernel makes it fit the GP in the projected
        // real parameter space (and embed categoricals) rather than over y.
        state.Proposer ??= new RemboGpProposer(
            _embedding, _kernel == RandomEmbeddingKernel.HighDimensional, _nStartupTrials, _baseSeed);
        return state.Proposer.Propose(run, latents, values, embeddedSpace);
    }

    /// <summary>The embedded real search space (float / integer / categorical) shared by all embeddings.</summary>
    /// <param name="study">The study whose intersection search space is filtered to embedded parameters.</param>
    /// <returns>The embedded distributions in intersection order; empty before any trial completes.</returns>
    private OrderedDict<string, BaseDistribution> EmbeddedSpace(Study study)
    {
        var space = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _searchSpace.Calculate(study))
            if (IsEmbedded(distribution))
                space[name] = distribution;
        return space;
    }

    /// <summary>Returns the shadow study for one embedding, creating an in-memory one (same direction) on first use.</summary>
    /// <param name="study">The real study whose single objective direction the shadow mirrors.</param>
    /// <param name="state">The real study's REMBO state; its shadow studies serve that study only.</param>
    /// <param name="run">The embedding (run) index the shadow study serves.</param>
    /// <returns>The cached shadow study driving the inner optimizer in the latent box for this run.</returns>
    private Study GetOrCreateShadow(Study study, StudyEmbeddingState state, int run)
    {
        if (state.ShadowStudies.TryGetValue(run, out Study? shadow))
            return shadow;

        // One in-memory study per embedding, reused across trials (so its creation log fires k times, not
        // once per trial). Direction is safe to read: the caller already rejected multi-objective studies.
        shadow = Optuna.CreateStudy(sampler: _innerSampler, direction: study.Direction);
        state.ShadowStudies[run] = shadow;
        state.SyncedNumbers[run] = new HashSet<int>();
        return shadow;
    }

    /// <summary>
    ///     Mirrors the real study's newly-finished trials of one embedding into its shadow study as latent
    ///     observations, so the inner optimizer sees every completed real evaluation exactly once.
    /// </summary>
    /// <param name="realStudy">The real study supplying finished trials and their objective values.</param>
    /// <param name="state">The real study's REMBO state, recording which trials each shadow study already mirrors.</param>
    /// <param name="shadow">The shadow study receiving the mirrored latent observations.</param>
    /// <param name="run">The embedding (run) index whose trials are mirrored.</param>
    private void SyncShadow(Study realStudy, StudyEmbeddingState state, Study shadow, int run)
    {
        HashSet<int> synced = state.SyncedNumbers[run];
        _latentSpace ??= _embedding.LatentSearchSpace();

        // Completed trials only: an observation needs a value, and the latent trials the GP fits must be
        // COMPLETE. Ordered by number so the shadow accrues observations in the same order they were made.
        foreach (var t in realStudy.GetTrialsInternal(deepcopy: false, states: new[] { TrialState.Complete }, useCache: true)
                     .OrderBy(t => t.Number))
        {
            if (synced.Contains(t.Number))
                continue;
            if (_embedding.RunForTrial(t.Number) != run)
                continue;

            // Mark unusable trials synced so they are not re-examined: a NaN/absent value cannot train a GP,
            // and a trial with only categorical parameters never received a latent point to mirror.
            if (t.Value is not double v || double.IsNaN(v) || !TryReadStoredLatent(t, out double[] y))
            {
                synced.Add(t.Number);
                continue;
            }

            var mirror = FrozenTrial.CreateTrial(
                state: TrialState.Complete, value: v, @params: LatentParams(y), distributions: _latentSpace);
            shadow.AddTrial(mirror);
            synced.Add(t.Number);
        }
    }

    /// <summary>Reads the latent coordinates an ask-trial was assigned, in coordinate order.</summary>
    /// <param name="ask">The shadow trial returned by <c>Ask</c> with all latent parameters suggested.</param>
    /// <returns>The <c>d</c>-vector of latent coordinates.</returns>
    private double[] ReadLatent(Trial ask)
    {
        var y = new double[_embedding.Dim];
        var p = ask.Params;
        for (int i = 0; i < _embedding.Dim; i++)
            y[i] = Convert.ToDouble(p[RandomEmbeddingProjection.LatentName(i)]);
        return y;
    }

    /// <summary>Builds the latent parameter map (<c>z0..z_{d-1}</c> → value) for a mirrored shadow trial.</summary>
    /// <param name="y">The latent coordinates to expose as the mirrored trial's parameters.</param>
    /// <returns>An ordered map keyed by latent coordinate name in coordinate order.</returns>
    private OrderedDict<string, object?> LatentParams(double[] y)
    {
        var p = new OrderedDict<string, object?>(y.Length);
        for (int i = 0; i < y.Length; i++)
            p[RandomEmbeddingProjection.LatentName(i)] = y[i];
        return p;
    }

    /// <summary>Reads a trial's persisted latent point, if present and dimensionally compatible.</summary>
    /// <param name="trial">The trial whose system attributes may hold a latent point.</param>
    /// <param name="y">Receives the latent coordinates on success; an empty array otherwise.</param>
    /// <returns><c>true</c> if a latent point of the expected dimension was recovered; otherwise <c>false</c>.</returns>
    private bool TryReadStoredLatent(FrozenTrial trial, out double[] y)
    {
        y = Array.Empty<double>();
        if (!trial.SystemAttrsDict.TryGetValue(LatentAttrKey, out object? raw) || raw is not IEnumerable seq)
            return false;

        var values = new List<double>(_embedding.Dim);
        foreach (object? o in seq)
            values.Add(Convert.ToDouble(o));

        // A dimension mismatch means the stored point came from a different d (e.g. the study was reused with
        // another RandomEmbeddingSampler): treat it as absent rather than misinterpreting the coordinates.
        if (values.Count != _embedding.Dim)
            return false;

        y = values.ToArray();
        return true;
    }

    /// <summary>Warns that a parameter is sampled independently because REMBO cannot embed it.</summary>
    /// <param name="trial">The trial the parameter belongs to.</param>
    /// <param name="paramName">The categorical or single-valued parameter being delegated.</param>
    private void LogIndependentSampling(FrozenTrial trial, string paramName)
    {
        Logger.Warning(string.Format(
            IndependentSamplingWarningTemplate,
            paramName,
            trial.Number,
            _independentSampler.GetType().Name,
            GetType().Name,
            "categorical and single-valued parameters are not embedded by `RandomEmbeddingSampler`"));
    }

    /// <inheritdoc />
    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _independentSampler.BeforeTrial(study, trial);

    /// <inheritdoc />
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        // Mirror TurboSampler: REMBO's inner GP does not consume trial constraints, so flag it if any were set.
        if (trial.Constraints.Count > 0)
            OptunaWarnings.Warn(
                "Constraints have been set, but RandomEmbeddingSampler will not use them for optimization.",
                WarningCategory.UserWarning);
        _independentSampler.AfterTrial(study, trial, state, values);
    }
}
