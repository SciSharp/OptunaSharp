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

namespace OptunaSharp.Samplers;

/// <summary>
///     A sampler implementing REMBO — Random EMbedding Bayesian Optimization (Wang et al. 2016). It
///     draws a fixed random Gaussian matrix <c>A ∈ ℝ^{D×d}</c>, runs an inner Bayesian optimizer (a
///     <see cref="GPSampler" /> by default, matching the paper) in the small box <c>Y = [−√d, √d]^d</c>,
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
///     (its observation feeds the inner optimizer). The latent point is chosen by the inner sampler over
///     a private in-memory <i>shadow study</i> whose trials are the latent coordinates of the real study's
///     finished trials; it is persisted in the real trial's system attributes so the shadow can be rebuilt
///     after resumption. <b>Float and integer</b> parameters are embedded (integers/steps/log honored via
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
    private readonly BaseSampler _innerSampler;
    private readonly BaseSampler _independentSampler;
    private readonly bool _warnIndependentSampling;

    // All shadow-study bookkeeping and latent-point derivation happen under this lock: the inner sampler
    // instance is shared across the k shadow studies, and the whole latent sampling is fast relative to a
    // (typically expensive) objective evaluation, which runs outside the lock.
    private readonly object _gate = new();
    private readonly Dictionary<int, LatentPoint> _latentCache = new();      // by real trial id
    private readonly Dictionary<int, Study> _shadowStudies = new();          // by embedding (run) index
    private readonly Dictionary<int, HashSet<int>> _syncedNumbers = new();   // by run: real trial numbers already mirrored
    private OrderedDict<string, BaseDistribution>? _latentSpace;

    /// <summary>A trial's assignment: which embedding it used and the latent point that produced its parameters.</summary>
    /// <param name="Run">The embedding (run) index in <c>[0, k)</c> this trial belongs to.</param>
    /// <param name="Y">The latent coordinates <c>y ∈ Y</c> that were mapped to the real parameters.</param>
    private readonly record struct LatentPoint(int Run, double[] Y);

    /// <param name="nEmbeddingDims">The internal embedding dimension <c>d</c> the inner optimizer searches
    ///     over — the core REMBO knob. Set it to at least the guessed effective dimensionality of the
    ///     objective; larger values raise the chance the box captures the optimum but slow convergence.</param>
    /// <param name="innerSampler">The Bayesian optimizer run in the latent box; <c>null</c> uses a
    ///     <see cref="GPSampler" /> (the paper's choice). Pass any history-reconstructing sampler
    ///     (e.g. <see cref="TPESampler" /> for a cheaper pure-managed variant); samplers that persist an
    ///     optimizer blob across trials (CMA-ES / ES) are not intended here.</param>
    /// <param name="nEmbeddings">The number of interleaved independent embeddings <c>k</c> (Section 3.1),
    ///     default 1; trials are assigned to embeddings round-robin.</param>
    /// <param name="independentSampler">Sampler for parameters REMBO does not embed — categorical and
    ///     single-valued; <c>null</c> uses <see cref="RandomSampler" />.</param>
    /// <param name="warnIndependentSampling">Emit a warning when a categorical or single-valued parameter is
    ///     sampled independently rather than embedded.</param>
    /// <param name="seed">Random seed anchoring the embedding matrix and the default inner/independent
    ///     samplers; a fixed seed reproduces the trajectory under sequential (single-worker) optimization.</param>
    /// <summary>Configures the random embedding, its inner Bayesian optimizer, and the independent fallback.</summary>
    /// <exception cref="System.ArgumentException"><paramref name="nEmbeddingDims" /> or <paramref name="nEmbeddings" /> is below one.</exception>
    public RandomEmbeddingSampler(
        int nEmbeddingDims,
        BaseSampler? innerSampler = null,
        int nEmbeddings = 1,
        BaseSampler? independentSampler = null,
        bool warnIndependentSampling = true,
        int? seed = null)
    {
        if (nEmbeddingDims < 1)
            throw new ArgumentException("nEmbeddingDims must be at least 1.", nameof(nEmbeddingDims));
        if (nEmbeddings < 1)
            throw new ArgumentException("nEmbeddings must be at least 1.", nameof(nEmbeddings));

        // Anchor the embedding matrix through a LazyRandomState so a null seed draws from OS entropy without
        // ever touching System.Random, and a fixed seed makes the matrix (and thus the whole run) reproducible.
        var seedRng = new LazyRandomState(seed);
        uint baseSeed = (uint)seedRng.Rng.randint(0, int.MaxValue);
        _embedding = new RandomEmbeddingProjection(baseSeed, nEmbeddingDims, nEmbeddings);

        // The default inner GP runs on the private shadow study; silence its independent-sampling warnings so
        // the shadow's startup randomness never surfaces to the user (their own inner sampler keeps its setting).
        _innerSampler = innerSampler ?? new GPSampler(seed: seed, warnIndependentSampling: false);
        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _warnIndependentSampling = warnIndependentSampling;
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

        // Only continuous/integer parameters ride the linear embedding; a categorical cannot be recovered
        // from a single projected scalar, and a single-valued parameter has nothing to optimize — both go to
        // the independent sampler, exactly as in CmaEsSampler / TurboSampler.
        if (paramDistribution is CategoricalDistribution || paramDistribution.Single())
        {
            if (_warnIndependentSampling)
                LogIndependentSampling(trial, paramName);
            return _independentSampler.SampleIndependent(study, trial, paramName, paramDistribution);
        }

        LatentPoint latent = GetOrComputeLatent(study, trial);
        return _embedding.ToExternal(latent.Run, paramName, paramDistribution, latent.Y);
    }

    /// <summary>
    ///     Returns this trial's latent point, sampling it once (from the inner optimizer over the shadow
    ///     study) and caching it so every real parameter of the trial maps from the same <c>y</c>.
    /// </summary>
    /// <param name="study">The real study whose finished trials seed the shadow study.</param>
    /// <param name="trial">The running trial whose latent point is needed.</param>
    /// <returns>The trial's embedding (run) index and its latent coordinates.</returns>
    private LatentPoint GetOrComputeLatent(Study study, FrozenTrial trial)
    {
        lock (_gate)
        {
            // Cached within this process for the trial's whole lifetime (all its parameter suggestions).
            if (_latentCache.TryGetValue(trial.TrialId, out LatentPoint cached))
                return cached;

            int run = _embedding.RunForTrial(trial.Number);

            // Resumption safety: if this trial was already partially sampled (in a previous process), reuse
            // the persisted latent point so the trial's parameters stay mutually consistent.
            if (TryReadStoredLatent(trial, out double[] storedY))
            {
                var restored = new LatentPoint(run, storedY);
                _latentCache[trial.TrialId] = restored;
                return restored;
            }

            Study shadow = GetOrCreateShadow(study, run);
            SyncShadow(study, shadow, run);

            _latentSpace ??= _embedding.LatentSearchSpace();

            // Ask the inner optimizer for a latent point, then fail the ask-trial: leaving it RUNNING would
            // make the GP treat it as a pending evaluation on every subsequent ask (it counts running trials
            // that carry all search-space keys), while completing it would inject a bogus observation. A
            // FAILED trial is invisible to the inner sampler; the real observation is mirrored later by SyncShadow.
            Trial ask = shadow.Ask(_latentSpace);
            double[] y = ReadLatent(ask);
            shadow.Tell(ask, values: null, state: TrialState.Fail);

            // Persist the latent point so future shadow rebuilds (and resumption) can recover this trial.
            study.Storage.SetTrialSystemAttr(trial.TrialId, LatentAttrKey, y.Cast<object?>().ToList());

            var result = new LatentPoint(run, y);
            _latentCache[trial.TrialId] = result;
            return result;
        }
    }

    /// <summary>Returns the shadow study for one embedding, creating an in-memory one (same direction) on first use.</summary>
    /// <param name="study">The real study whose single objective direction the shadow mirrors.</param>
    /// <param name="run">The embedding (run) index the shadow study serves.</param>
    /// <returns>The cached shadow study driving the inner optimizer in the latent box for this run.</returns>
    private Study GetOrCreateShadow(Study study, int run)
    {
        if (_shadowStudies.TryGetValue(run, out Study? shadow))
            return shadow;

        // One in-memory study per embedding, reused across trials (so its creation log fires k times, not
        // once per trial). Direction is safe to read: the caller already rejected multi-objective studies.
        shadow = Optuna.CreateStudy(sampler: _innerSampler, direction: study.Direction);
        _shadowStudies[run] = shadow;
        _syncedNumbers[run] = new HashSet<int>();
        return shadow;
    }

    /// <summary>
    ///     Mirrors the real study's newly-finished trials of one embedding into its shadow study as latent
    ///     observations, so the inner optimizer sees every completed real evaluation exactly once.
    /// </summary>
    /// <param name="realStudy">The real study supplying finished trials and their objective values.</param>
    /// <param name="shadow">The shadow study receiving the mirrored latent observations.</param>
    /// <param name="run">The embedding (run) index whose trials are mirrored.</param>
    private void SyncShadow(Study realStudy, Study shadow, int run)
    {
        HashSet<int> synced = _syncedNumbers[run];
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
