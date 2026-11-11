using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers.CmaEs;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     A sampler using <a href="https://github.com/CyberAgentAILab/cmaes">cmaes</a> as the backend.
///     Port of <c>optuna/samplers/_cmaes.py: CmaEsSampler</c> driving the <see cref="Cma" /> optimizer.
///
///     <para>This sampler does not support <see cref="CategoricalDistribution" />. If the search
///     space includes categorical parameters they are sampled by the independent sampler.</para>
///
///     <para><b>Optimizer state is Optuna's pickle.</b> Like Optuna, the sampler stores
///     <c>pickle.dumps(optimizer).hex()</c> of a cmaes <c>CMA</c>, <c>SepCMA</c> or <c>CMAwM</c> under the
///     chunked <c>cma:optimizer:{i}</c> (<c>sepcma:</c>, <c>cmawm:</c>) system attrs, written and read by
///     <see cref="CmaPickle" />, so C# and Python workers can share one CMA-ES study in any order. A
///     stored state is byte-identical to the one cmaes 0.13.1 stores for the same optimizer, except
///     the <c>_funhist_values</c> entries cmaes leaves uninitialized (<c>np.empty</c>) until its first
///     generations fill them. States written by OptunaSharp before 2026-10-29 (a <c>cma\n</c>-tagged
///     JSON) are still read; Python cannot read those, so a study gains Python workers once a current
///     C# worker has stored the next state. With <c>with_margin=True</c> the Python workers need
///     cmaes 0.13.1, whose <c>CMAwM</c> state OptunaSharp reads.</para>
///
///     <para><b>Faster than Optuna's structure, same results.</b> Optuna restores the optimizer from its pickle
///     and rescans every trial on every ask, which makes a run quadratic in its trial count. This sampler keeps
///     the restored optimizer for the rest of its generation and folds the history incrementally
///     (<c>CmaTrialHistory</c>), reading an in-memory study through its zero-copy view; seeded trials and every
///     stored byte stay identical to Optuna's (Rustuna plan 13, phase 1; the measured gain is in
///     <c>docs/plans/13-rustuna-derived-performance.md</c>).</para>
/// </summary>
/// <example>
///     <code>
///     var sampler = new CmaEsSampler();
///     var study = Optuna.CreateStudy(sampler: sampler);
///     study.Optimize(t => Math.Pow(t.SuggestFloat("x", -1, 1), 2) + t.SuggestInt("y", -1, 1), nTrials: 20);
///     </code>
/// </example>
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html">Official Optuna CmaEsSampler reference.</seealso>
public class CmaEsSampler : BaseSampler
{
    // The value of system_attrs must be less than 2046 characters on RDBStorage.
    private const int SystemAttrMaxLength = 2045;
    private const double Eps = 1e-10;

    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.samplers._cmaes");

    private readonly IReadOnlyDictionary<string, object?>? _x0;
    private readonly double? _sigma0;
    private readonly BaseSampler _independentSampler;
    private readonly int _nStartupTrials;
    private bool _warnIndependentSampling;
    private readonly LazyRandomState _cmaRng;
    private readonly IntersectionSearchSpace _searchSpace;
    private readonly bool _considerPrunedTrials;
    private readonly int? _popsize;
    private readonly bool _useSeparableCma;
    private readonly bool _withMargin;
    private readonly bool _lrAdapt;
    private readonly List<FrozenTrial>? _sourceTrials;
    private readonly string _attrPrefix;

    /// <summary>
    ///     The history this sampler reads on every ask (Optuna's <c>_get_trials</c>, the latest stored state, a
    ///     generation's trials), kept up to date incrementally instead of rebuilt from a full copy each time. Same
    ///     trials, same order (Rustuna plan 13, item S2). Touched only under the sampler's lock.
    /// </summary>
    private readonly CmaTrialHistory _history;

    /// <summary>
    ///     The optimizer deserialized from the latest stored state, kept for the remaining asks of its generation so
    ///     each ask skips the unpickling and reuses the eigendecomposition the first ask computed (plan 13, item S1).
    ///     Null when nothing is cached. Touched only under the sampler's lock.
    /// </summary>
    /// <remarks>
    ///     It answers only for <see cref="_cachedHolderTrialId" />'s state, and every ask that reuses it would have
    ///     restored an identical object: asking caches B and D and rewrites C as B·D²·Bᵀ, which a fresh restore of the
    ///     same stored C recomputes to the same bits, and the per-ask reseed resets the random stream and its Gaussian
    ///     cache. It is dropped after every tell: the told object stands neither for the state it came from (restored
    ///     again if the telling trial fails) nor for the state just stored, because Optuna's pickle keeps only C's
    ///     upper triangle and a restore mirrors it, while the told C keeps the update's floating-point asymmetry.
    ///     Keeping it moves seeded results (measured: from generation 3 on a 40-D sphere).
    /// </remarks>
    private ICmaOptimizer? _cachedOptimizer;

    /// <summary>
    ///     The trial whose stored state <see cref="_cachedOptimizer" /> was restored from; -1 when nothing is cached.
    ///     A holder is a finished trial, whose system attributes cannot change, so its id names its state; a state
    ///     another worker (C# or Python) stores lands on another trial and misses.
    /// </summary>
    private int _cachedHolderTrialId = -1;

    /// <summary>
    ///     How many stored optimizer states this instance deserialized. Lets the tests assert the cache's contract
    ///     (one restore per generation in a sequential run) without timing anything.
    /// </summary>
    internal int RestoreCount { get; private set; }

    /// <summary>
    ///     The opt-in <c>reuseToldOptimizer</c> (Rustuna plan 13, item D1; the owner chose opt-in with Optuna's path as
    ///     the default): after a tell, the told optimizer becomes the cache for the state it just stored instead of
    ///     being dropped. Faster, and no longer Optuna's samples (see <see cref="_cachedOptimizer" /> for why).
    /// </summary>
    private readonly bool _reuseToldOptimizer;

    /// <param name="x0">
    ///     Initial parameter values for CMA-ES (external representation). By default the mean of the
    ///     low and high of each distribution. <b>Deprecated in v4.9.0.</b>
    /// </param>
    /// <param name="sigma0">
    ///     Initial standard deviation of CMA-ES. By default <c>min_range / 6</c>. <b>Deprecated in v4.9.0.</b>
    /// </param>
    /// <param name="nStartupTrials">Independent sampling is used until this many trials finish.</param>
    /// <param name="independentSampler">Sampler for parameters outside the relative search space (default <see cref="RandomSampler" />).</param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently.</param>
    /// <param name="seed">Random seed for CMA-ES.</param>
    /// <param name="considerPrunedTrials">If <c>true</c>, PRUNED trials are considered for sampling (experimental).</param>
    /// <param name="restartStrategy"><b>Deprecated in v4.4.0</b> — falls back to <c>null</c>.</param>
    /// <param name="popsize">Population size of CMA-ES.</param>
    /// <param name="incPopsize"><b>Deprecated in v4.4.0</b> — no longer used.</param>
    /// <param name="useSeparableCma">Constrain the covariance matrix to be diagonal (experimental).</param>
    /// <param name="withMargin">Use CMA-ES with margin for discrete distributions (experimental).</param>
    /// <param name="lrAdapt">Use CMA-ES with learning-rate adaptation (experimental).</param>
    /// <param name="sourceTrials">Warm Starting CMA-ES source trials (experimental).</param>
    /// <param name="reuseToldOptimizer">
    ///     <b>Off by default; turning it on makes seeded runs stop matching Python Optuna</b> (from about the third
    ///     generation; same solution quality, a different run). When on, the optimizer that just told a generation keeps
    ///     answering the asks that follow, as Rustuna keeps its live optimizer, instead of being restored from the state
    ///     it stored: about 15 % faster on a 40-D study. The state is still stored every generation exactly as Optuna
    ///     stores it, so resuming a study and Python workers on the same storage keep working; they just do not
    ///     reproduce this worker's samples. An OptunaSharp extension with no Optuna counterpart, so it comes after every
    ///     Optuna parameter.
    /// </param>
    /// <summary>Configures covariance-matrix adaptation and its independent startup and unsupported-parameter fallback.</summary>
    /// <exception cref="NumSharp.ValueError">Source-trial initialization, explicit initial parameters, separable adaptation, margin correction, or learning-rate adaptation are combined incompatibly.</exception>
    public CmaEsSampler(
        IReadOnlyDictionary<string, object?>? x0 = null,
        double? sigma0 = null,
        int nStartupTrials = 1,
        BaseSampler? independentSampler = null,
        bool warnIndependentSampling = true,
        int? seed = null,
        bool considerPrunedTrials = false,
        string? restartStrategy = null,
        int? popsize = null,
        int incPopsize = -1,
        bool useSeparableCma = false,
        bool withMargin = false,
        bool lrAdapt = false,
        List<FrozenTrial>? sourceTrials = null,
        bool reuseToldOptimizer = false)
    {
        if (restartStrategy != null || incPopsize != -1)
        {
            string msg = string.Format(Deprecations.Template, "`restart_strategy`", "4.4.0", "6.0.0");
            OptunaWarnings.Warn(
                msg + " From v4.4.0 onward, `restart_strategy` automatically falls back to " +
                "`None`. `restart_strategy` will be supported in OptunaHub.",
                WarningCategory.FutureWarning);
        }
        if (x0 != null)
            OptunaWarnings.Warn(string.Format(Deprecations.Template, "`x0`", "4.9.0", "6.0.0"),
                WarningCategory.FutureWarning);
        if (sigma0 != null)
            OptunaWarnings.Warn(string.Format(Deprecations.Template, "`sigma0`", "4.9.0", "6.0.0"),
                WarningCategory.FutureWarning);

        _x0 = x0;
        _sigma0 = sigma0;
        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _nStartupTrials = nStartupTrials;
        _warnIndependentSampling = warnIndependentSampling;
        _cmaRng = new LazyRandomState(seed);
        _searchSpace = new IntersectionSearchSpace();
        _considerPrunedTrials = considerPrunedTrials;
        _popsize = popsize;
        _useSeparableCma = useSeparableCma;
        _withMargin = withMargin;
        _lrAdapt = lrAdapt;
        _sourceTrials = sourceTrials;
        _reuseToldOptimizer = reuseToldOptimizer;

        _attrPrefix = useSeparableCma ? "sepcma:" : withMargin ? "cmawm:" : "cma:";
        _history = new CmaTrialHistory(AsSamplerTrial, AttrKeyGeneration, AttrKeyOptimizer);

        // Optuna raises an ExperimentalWarning (warn_experimental_argument) for consider_pruned_trials,
        // use_separable_cma, source_trials, with_margin and lr_adapt. OptunaSharp targets the Optuna 5.x
        // surface and treats these as stable, so those warnings are intentionally omitted.

        if (sourceTrials != null && (x0 != null || sigma0 != null))
            throw new ValueError(
                "It is prohibited to pass `source_trials` argument when x0 or sigma0 is specified.");
        if (sourceTrials != null && useSeparableCma)
            throw new ValueError(
                "It is prohibited to pass `source_trials` argument when using separable CMA-ES.");
        if (lrAdapt && (useSeparableCma || withMargin))
            throw new ValueError(
                "It is prohibited to pass `use_separable_cma` or `with_margin` argument when " +
                "using `lr_adapt`.");
        if (useSeparableCma && withMargin)
            throw new ValueError(
                "Currently, we do not support `use_separable_cma=True` and `with_margin=True`.");
    }

    /// <summary>
    ///     A worker instance for <see cref="SamplerConcurrency.PerWorker" />: <paramref name="prototype" />'s configuration
    ///     with a CMA generator, a search-space cache and an independent sampler of its own.
    /// </summary>
    /// <param name="prototype">The study's sampler whose configuration the worker instance takes.</param>
    /// <param name="independentSampler">A worker instance of the prototype's independent sampler.</param>
    /// <remarks>
    ///     Not the public constructor: that one re-emits the deprecation warnings for <c>restart_strategy</c>, <c>x0</c>
    ///     and <c>sigma0</c>, which every worker would otherwise print again. The optimizer state itself lives in the
    ///     storage's system attributes, so the instances continue one CMA-ES run between them, as processes do.
    /// </remarks>
    private CmaEsSampler(CmaEsSampler prototype, BaseSampler independentSampler)
    {
        _x0 = prototype._x0;
        _sigma0 = prototype._sigma0;
        _independentSampler = independentSampler;
        _nStartupTrials = prototype._nStartupTrials;
        _warnIndependentSampling = prototype._warnIndependentSampling;
        // OS entropy: each trial's optimizer seed is this generator's draw plus the trial number (Optuna's rule), so
        // instances seeded alike would hand trials correlated seeds.
        _cmaRng = new LazyRandomState(null);
        _searchSpace = new IntersectionSearchSpace();
        _considerPrunedTrials = prototype._considerPrunedTrials;
        _popsize = prototype._popsize;
        _useSeparableCma = prototype._useSeparableCma;
        _withMargin = prototype._withMargin;
        _lrAdapt = prototype._lrAdapt;
        _sourceTrials = prototype._sourceTrials;
        _attrPrefix = prototype._attrPrefix;
        _reuseToldOptimizer = prototype._reuseToldOptimizer;
        // A history and an optimizer cache of its own: each worker folds the shared study by itself.
        _history = new CmaTrialHistory(AsSamplerTrial, AttrKeyGeneration, AttrKeyOptimizer);
    }

    /// <summary>
    ///     A new <see cref="CmaEsSampler" /> with this one's configuration, for a <see cref="SamplerConcurrency.PerWorker" />
    ///     worker. Each trial reseeds the restored optimizer from this instance's generator plus the trial number, so
    ///     separate instances draw distinct candidates (0 duplicate samples in the measured eight-instance runs).
    /// </summary>
    /// <returns>
    ///     A fresh instance, or <see langword="null" /> (shared access) for a subclass or when the independent sampler has
    ///     no worker instance: it is called without its own monitor, so a shared one must stay behind this sampler's.
    /// </returns>
    internal override BaseSampler? CreateWorkerInstance()
    {
        if (GetType() != typeof(CmaEsSampler))
            return null;
        BaseSampler? independent = _independentSampler.CreateWorkerInstance();
        return independent is null ? null : new CmaEsSampler(this, independent);
    }

    /// <inheritdoc />
    public override void ReseedRng()
    {
        // _cmaRng doesn't require reseeding because the relative sampling reseeds in each trial.
        _independentSampler.ReseedRng();
    }

    /// <inheritdoc />
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _searchSpace.Calculate(study))
        {
            if (distribution.Single())
                // `cma` cannot handle distributions that contain just a single value, so we skip
                // them. Note that the parameter values for such distributions are sampled in `Trial`.
                continue;

            if (distribution is not FloatDistribution && distribution is not IntDistribution)
                // Categorical distribution is unsupported.
                continue;

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

        if (searchSpace.Count == 0)
            return new OrderedDict<string, object?>();

        // Optuna's `completed_trials = self._get_trials(study)`, read incrementally: the history folds each finished
        // trial once instead of copying and filtering every trial on every ask (CmaTrialHistory).
        RefreshHistory(study);
        if (_history.CompletedCount < _nStartupTrials)
            return new OrderedDict<string, object?>();

        // When `with_margin=true`, bounds in discrete dimensions are handled inside `CMAwM`.
        var trans = new SearchSpaceTransform(searchSpace, transformStep: !_withMargin, transform01: true);

        var restoredOptimizer = RestoreOptimizer();
        var optimizer = restoredOptimizer ?? InitOptimizer(trans, study.Direction, searchSpace);

        // trans.bounds has one row per encoded column; its row count is read from the raw bounds instead of building the
        // NumSharp bounds array only for its shape (plan 14).
        int nDim = trans.RawBounds.GetLength(0);
        if (optimizer.Dim != nDim)
        {
            if (_warnIndependentSampling)
            {
                string indName = _independentSampler.GetType().Name;
                Logger.Warning(
                    "`CmaEsSampler` does not support dynamic search space. " +
                    $"`{indName}` is used instead of `CmaEsSampler`.");
                _warnIndependentSampling = false;
            }
            return new OrderedDict<string, object?>();
        }

        // Optuna's `_get_solution_trials(completed_trials, optimizer.generation)`, from the generation index.
        var solutionTrials = _history.SolutionTrials(optimizer.Generation);

        if (solutionTrials.Count >= optimizer.PopulationSize)
        {
            var solutions = new List<(double[] X, double Value)>();
            foreach (var t in solutionTrials.Take(optimizer.PopulationSize))
            {
                // CMAwM updates its distribution from the RAW x it stashed as `x_for_tell`, not from
                // the trial's (encoded, grid-snapped) params. Every other backend re-transforms params.
                double[] x = optimizer is CmaWm
                    ? ((System.Collections.IEnumerable)t.SystemAttrs["x_for_tell"]!)
                        .Cast<object?>().Select(v => Convert.ToDouble(v)).ToArray()
                    : trans.TransformValues(t.Params);
                double y = study.Direction == StudyDirection.Minimize ? t.Value!.Value : -t.Value!.Value;
                solutions.Add((x, y));
            }

            optimizer.Tell(solutions);

            // Store optimizer. Python pickles a restored optimizer differently from a fresh one (see CmaPickle.ToHex).
            string optimizerStr = SerializeOptimizer(optimizer, restored: restoredOptimizer is not null);
            foreach (var (key, value) in SplitOptimizerStr(optimizerStr))
                study.Storage.SetTrialSystemAttr(trial.TrialId, key, value);

            if (_reuseToldOptimizer && restoredOptimizer is not null)
            {
                // Opt-in Rustuna path: the told object answers for the state this trial just stored, from the moment
                // the trial completes and becomes the latest state holder. If it fails instead, the previous holder is
                // the latest again, its id misses the cache, and that state is restored, as Optuna does. A fresh
                // optimizer (no state was restored before this first tell) is not kept, so the cached object always
                // carries the restored layout CmaPickle writes for every later tell.
                _cachedOptimizer = optimizer;
                _cachedHolderTrialId = trial.TrialId;
            }
            else
            {
                // The told object no longer equals any stored state (see _cachedOptimizer), so the next ask restores the
                // state just stored, as Optuna does: one restore per generation instead of one per ask.
                _cachedOptimizer = null;
                _cachedHolderTrialId = -1;
            }
        }

        // Caution: optimizer should update its seed value.
        int seed = (int)NumPyDraws.RandInt(_cmaRng.Rng, 1, 65536) + trial.Number;
        optimizer.ReseedRng(seed);
        double[] cmaParams;
        if (optimizer is CmaWm cmaWm)
        {
            // CMAwM's ask() yields (encoded, raw): evaluate/untransform the encoded x, but stash the
            // raw x as `x_for_tell` so the next tell updates the distribution from it (see above).
            var (encoded, xForTell) = cmaWm.AskEncoded();
            study.Storage.SetTrialSystemAttr(
                trial.TrialId, "x_for_tell", xForTell.Select(v => (object?)v).ToList());
            cmaParams = encoded;
        }
        else
        {
            cmaParams = optimizer.Ask();
        }

        study.Storage.SetTrialSystemAttr(trial.TrialId, AttrKeyGeneration, optimizer.Generation);

        // Untransform the managed candidate as it is: Optuna's trans.untransform(params), without a NumSharp copy.
        return trans.UntransformValues(cmaParams);
    }

    private string AttrKeyGeneration => _attrPrefix + "generation";
    private string AttrKeyOptimizer => _attrPrefix + "optimizer";

    /// <summary>Reassembles serialized optimizer chunks in numeric chunk order.</summary>
    /// <param name="optimizerAttrs">Stored serialized optimizer chunks keyed by chunk identifier.</param>
    /// <returns>The full serialized optimizer string in chunk-index order.</returns>
    private string ConcatOptimizerAttrs(IReadOnlyDictionary<string, string> optimizerAttrs)
        => string.Concat(Enumerable.Range(0, optimizerAttrs.Count)
            .Select(i => optimizerAttrs[$"{AttrKeyOptimizer}:{i}"]));

    /// <summary>Splits a serialized optimizer into storage-sized system-attribute chunks.</summary>
    /// <param name="optimizerStr">Tagged serialized optimizer state, after any chunk reassembly.</param>
    /// <returns>Chunk-indexed system attributes within the storage string-length limit.</returns>
    private Dictionary<string, string> SplitOptimizerStr(string optimizerStr)
    {
        int optimizerLen = optimizerStr.Length;
        var attrs = new Dictionary<string, string>();
        int chunks = (int)Math.Ceiling(optimizerLen / (double)SystemAttrMaxLength);
        for (int i = 0; i < chunks; i++)
        {
            int start = i * SystemAttrMaxLength;
            int end = Math.Min((i + 1) * SystemAttrMaxLength, optimizerLen);
            attrs[$"{AttrKeyOptimizer}:{i}"] = optimizerStr[start..end];
        }
        return attrs;
    }

    // The stored state is Optuna's: the hex of the optimizer's pickle, which names its own class, so restore builds
    // the right type regardless of the sampler's current config (a single-dimensional `use_separable_cma=True` study
    // stores a fallback CMA under the `sepcma:` prefix and must restore it as a Cma, not a SepCma). OptunaSharp
    // before 2026-10-29 stored its own JSON instead, self-describing through a `<tag>\n` prefix and written by
    // OptimizerStateJson, which emits no newlines and encodes ±inf/NaN as named literals. Hex never contains a
    // newline, so the newline tells the two apart. Both keep a non-finite funhist value (one generation's best or
    // worst), and so does pickle-based Optuna.

    /// <summary>Serializes the optimizer as Optuna stores it: the lowercase hex of its cmaes pickle.</summary>
    /// <param name="optimizer">Optimizer whose proposal and adaptation state are persisted.</param>
    /// <param name="restored">Whether the optimizer was restored from a stored state before this tell.</param>
    /// <returns><c>pickle.dumps(optimizer).hex()</c> of the equivalent cmaes optimizer.</returns>
    /// <exception cref="System.NotSupportedException">The optimizer implementation has no cmaes counterpart.</exception>
    private static string SerializeOptimizer(ICmaOptimizer optimizer, bool restored)
        => CmaPickle.ToHex(optimizer, restored);

    /// <summary>
    ///     Reconstructs a stored optimizer: Optuna's hex pickle (written by Python or by OptunaSharp), or the
    ///     <c>&lt;tag&gt;\n</c> JSON OptunaSharp wrote before 2026-10-29.
    /// </summary>
    /// <param name="optimizerStr">The stored state, after chunk reassembly.</param>
    /// <returns>The optimizer reconstructed from the state.</returns>
    /// <exception cref="NumSharp.ValueError">
    ///     The text is not a legacy JSON state and not hexadecimal (Python's <c>bytes.fromhex</c> message), or its
    ///     pickle is not a cmaes optimizer state this version reads (for example a with-margin state from cmaes 0.13.0
    ///     or older).
    /// </exception>
    /// <exception cref="System.Text.Json.JsonException">A legacy JSON payload is not valid JSON for its state type, or is the literal null.</exception>
    /// <exception cref="System.ArgumentException">A legacy JSON state contains invalid optimizer constructor settings.</exception>
    private static ICmaOptimizer DeserializeOptimizer(string optimizerStr)
    {
        int nl = optimizerStr.IndexOf('\n');
        string tag = nl < 0 ? "" : optimizerStr[..nl];
        string json = nl < 0 ? "" : optimizerStr[(nl + 1)..];
        return tag switch
        {
            "cma" => Cma.FromState(OptimizerStateJson.Deserialize<CmaState>(json)),
            "sepcma" => SepCma.FromState(OptimizerStateJson.Deserialize<SepCmaState>(json)),
            "cmawm" => CmaWm.FromState(OptimizerStateJson.Deserialize<CmaWmState>(json)),
            // Anything else is Optuna's hex pickle, read as Python reads it (bytes.fromhex skips whitespace, so a
            // newline alone does not make a legacy state).
            _ => CmaPickle.FromHex(optimizerStr),
        };
    }

    /// <summary>
    ///     Optuna's <c>_restore_optimizer</c>: the optimizer stored on the latest completed trial that holds one, from
    ///     <see cref="_cachedOptimizer" /> when that trial's state is the one cached.
    /// </summary>
    /// <returns>The newest available optimizer, or null when no completed trial stores one.</returns>
    /// <exception cref="NumSharp.ValueError">The stored state is neither a legacy JSON state nor a readable cmaes pickle.</exception>
    /// <exception cref="System.Text.Json.JsonException">A legacy JSON payload is not valid JSON for its state type.</exception>
    /// <exception cref="ArgumentException">A legacy JSON state contains invalid optimizer constructor settings.</exception>
    /// <remarks>
    ///     The holder is <see cref="CmaTrialHistory.LatestStateHolder" />, the trial Optuna's backward scan over the
    ///     completed trials stops at. A cache hit returns the object an earlier ask of this generation restored and asked
    ///     with, which produces the same samples as a fresh restore (see <see cref="_cachedOptimizer" />).
    /// </remarks>
    private ICmaOptimizer? RestoreOptimizer()
    {
        FrozenTrial? holder = _history.LatestStateHolder;
        if (holder is null)
            return null;
        if (_cachedOptimizer is not null && _cachedHolderTrialId == holder.TrialId)
            return _cachedOptimizer;

        // Restore a previous CMA object.
        var optimizerAttrs = holder.SystemAttrs
            .Where(kv => kv.Key.StartsWith(AttrKeyOptimizer, StringComparison.Ordinal))
            .ToDictionary(kv => kv.Key, kv => (string)kv.Value!);
        var restored = DeserializeOptimizer(ConcatOptimizerAttrs(optimizerAttrs));
        RestoreCount++;
        _cachedOptimizer = restored;
        _cachedHolderTrialId = holder.TrialId;
        return restored;
    }

    /// <summary>
    ///     Brings <see cref="_history" /> up to date with the study, and forgets the optimizer cache when the history
    ///     started over (another study or storage, whose trial ids may repeat this one's).
    /// </summary>
    /// <param name="study">The study being sampled.</param>
    /// <exception cref="NumSharp.KeyError">The study no longer exists in storage.</exception>
    /// <remarks>
    ///     The read is <c>Study.GetTrialsForSampler(useCache: true)</c>: the zero-copy view on an in-memory storage,
    ///     otherwise the per-trial cached list Optuna's <c>_get_trials(deepcopy=False, use_cache=True)</c> reads, so a
    ///     database-backed storage makes no more round trips than before.
    /// </remarks>
    private void RefreshHistory(Study study)
    {
        _history.Refresh(study.GetTrialsForSampler(useCache: true), study.Storage, study.StudyId);
        if (_history.WasReset)
        {
            _cachedOptimizer = null;
            _cachedHolderTrialId = -1;
        }
    }

    /// <summary>Creates the initial optimizer in the transformed numerical box.</summary>
    /// <param name="trans">Search-space transform between external parameter values and the numerical box.</param>
    /// <param name="direction">Whether objective values are minimized or maximized.</param>
    /// <param name="searchSpace">Ordered parameter distributions defining the coordinates used by this operation.</param>
    /// <returns>The initialized proposal optimizer.</returns>
    /// <exception cref="NumSharp.ValueError">No source trial has a compatible search space.</exception>
    /// <exception cref="System.ArgumentException">A distribution cannot supply a margin step, or the optimizer rejects its initial mean, scale, covariance, or population settings.</exception>
    private ICmaOptimizer InitOptimizer(
        SearchSpaceTransform trans, StudyDirection direction, OrderedDict<string, BaseDistribution> searchSpace)
    {
        double[] boundsFlat = trans.Bounds.ToArray<double>();
        int nDim = boundsFlat.Length / 2;
        var lower = new double[nDim];
        var upper = new double[nDim];
        var boundsArr = new double[nDim, 2];
        for (int i = 0; i < nDim; i++)
        {
            lower[i] = boundsFlat[i * 2];
            upper[i] = boundsFlat[i * 2 + 1];
            boundsArr[i, 0] = lower[i];
            boundsArr[i, 1] = upper[i];
        }

        double[,]? cov = null;
        double[] mean;
        double sigma0;

        if (_sourceTrials == null)
        {
            if (_x0 == null)
            {
                mean = new double[nDim];
                for (int i = 0; i < nDim; i++) mean[i] = lower[i] + (upper[i] - lower[i]) / 2;
            }
            else
            {
                // `_x0` is external representations.
                mean = trans.Transform(_x0).ToArray<double>();
            }

            if (_sigma0 == null)
            {
                // np.min((upper_bounds - lower_bounds) / 6)
                sigma0 = double.PositiveInfinity;
                for (int i = 0; i < nDim; i++)
                {
                    double v = (upper[i] - lower[i]) / 6;
                    if (v < sigma0) sigma0 = v;
                }
            }
            else
            {
                sigma0 = _sigma0.Value;
            }
        }
        else
        {
            // Warm Starting CMA-ES: estimate the prior distribution from the source task's trials.
            var expectedStates = new List<TrialState> { TrialState.Complete };
            if (_considerPrunedTrials) expectedStates.Add(TrialState.Pruned);
            int sign = direction == StudyDirection.Minimize ? 1 : -1;

            var sourceSolutions = new List<(double[] Params, double Value)>();
            foreach (var t in _sourceTrials)
            {
                if (!expectedStates.Contains(t.State)) continue;
                if (!IsCompatibleSearchSpace(searchSpace, t.Distributions)) continue;
                double[] x = trans.Transform(t.Params).ToArray<double>();
                sourceSolutions.Add((x, sign * t.Value!.Value));
            }
            if (sourceSolutions.Count == 0)
                throw new ValueError("No compatible source_trials");

            (mean, sigma0, cov) = WarmStart.GetWarmStartMgd(sourceSolutions);
        }

        // Python's sigma0 is a numpy.float64 only when it came from `np.min(...)` and survived
        // `max(sigma0, _EPS)`, which returns the float _EPS when that is strictly larger; a user sigma0 and a warm
        // start's `math.pow` are floats. The type changes no value; it only shapes the pickled state (see Cma).
        bool sigmaIsNumpy = _sourceTrials == null && _sigma0 == null && !(Eps > sigma0);

        // Avoid ZeroDivisionError in cmaes.
        sigma0 = Math.Max(sigma0, Eps);

        if (_useSeparableCma)
        {
            if (boundsArr.GetLength(0) == 1)
            {
                OptunaWarnings.Warn(
                    "Separable CMA-ES does not operate meaningfully on single-dimensional " +
                    "search spaces. The setting `use_separable_cma=True` will be ignored.",
                    WarningCategory.UserWarning);
            }
            else
            {
                return new SepCma(
                    mean: mean,
                    sigma: sigma0,
                    bounds: boundsArr,
                    nMaxResampling: 10 * nDim,
                    seed: (int)NumPyDraws.RandInt(_cmaRng.Rng, 1, 2147483646),
                    populationSize: _popsize,
                    sigmaIsNumpyScalar: sigmaIsNumpy);
            }
        }

        if (_withMargin)
        {
            // Per-dimension discretization step (in the [0,1] transformed space). Continuous dims and
            // log dims get 0.0; a discrete dim gets step/(high-low). `with_margin` handles the bounds
            // of discrete dims inside CMAwM, so `trans` was built with transformStep=false.
            var steps = new double[searchSpace.Count];
            int si = 0;
            foreach (var dist in searchSpace.Values)
            {
                steps[si++] = dist switch
                {
                    FloatDistribution fd => fd.Step is null || fd.Log ? 0.0
                        : fd.Low == fd.High ? 1.0
                        : fd.Step.Value / (fd.High - fd.Low),
                    IntDistribution id => id.Log ? 0.0
                        : id.Low == id.High ? 1.0
                        : (double)id.Step / (id.High - id.Low),
                    _ => throw new ArgumentException(
                        "CmaEsSampler with_margin supports only Float/Int distributions."),
                };
            }

            return new CmaWm(
                mean: mean,
                sigma: sigma0,
                bounds: boundsArr,
                steps: steps,
                nMaxResampling: 10 * nDim,
                seed: (int)NumPyDraws.RandInt(_cmaRng.Rng, 1, 2147483646),
                populationSize: _popsize,
                cov: cov,
                sigmaIsNumpyScalar: sigmaIsNumpy);
        }

        int seed = (int)NumPyDraws.RandInt(_cmaRng.Rng, 1, 2147483646);
        return new Cma(
            mean: mean,
            sigma: sigma0,
            bounds: boundsArr,
            nMaxResampling: 10 * nDim,
            seed: seed,
            populationSize: _popsize,
            cov: cov,
            lrAdapt: _lrAdapt,
            sigmaIsNumpyScalar: sigmaIsNumpy);
    }

    /// <summary>
    ///     Python <c>_is_compatible_search_space</c>: the relative space and the source trial's
    ///     distributions must have identical key sets.
    /// </summary>
    /// <param name="relativeSpace">Distributions selected for joint numerical sampling.</param>
    /// <param name="distributions">Parameter distributions used to interpret the numerical coordinates.</param>
    /// <returns>True if all source distributions are compatible with the current relative space.</returns>
    private static bool IsCompatibleSearchSpace(
        OrderedDict<string, BaseDistribution> relativeSpace,
        IReadOnlyDictionary<string, BaseDistribution> distributions)
    {
        int intersection = relativeSpace.Keys.Count(distributions.ContainsKey);
        return intersection == relativeSpace.Count && intersection == distributions.Count;
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">The study has multiple objectives, but this sampler requires a scalar objective.</exception>
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        RaiseErrorIfMultiObjective(study);

        if (_warnIndependentSampling)
        {
            // Optuna's `len(self._get_trials(study)) >= self._n_startup_trials`, from the incremental history, so a
            // categorical parameter sampled every trial no longer costs a full copy of the history each time.
            RefreshHistory(study);
            if (_history.CompletedCount >= _nStartupTrials)
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
            "dynamic search space and `CategoricalDistribution` are not supported by `CmaEsSampler`"));
    }

    /// <summary>
    ///     The per-trial filter of Optuna's <c>_get_trials</c>: a COMPLETE trial as it is; with
    ///     <c>consider_pruned_trials</c>, a PRUNED trial that reported values, copied with its value rewritten to the
    ///     value at its largest step; anything else is ignored.
    /// </summary>
    /// <param name="t">A trial snapshot from the history.</param>
    /// <returns>The snapshot the sampler treats as completed, or null when it ignores the trial.</returns>
    /// <remarks>
    ///     <see cref="CmaTrialHistory" /> calls this once per trial when the trial is folded and keeps the result, so a
    ///     pruned trial's copy is made once per run rather than once per ask; a finished trial cannot change, so the copy
    ///     stays exact.
    /// </remarks>
    private FrozenTrial? AsSamplerTrial(FrozenTrial t)
    {
        if (t.State == TrialState.Complete)
            return t;
        if (t.State == TrialState.Pruned && t.IntermediateValuesForRead.Count > 0 && _considerPrunedTrials)
        {
            // max(t.intermediate_values.items()) — the value at the largest step key.
            long maxStep = t.IntermediateValuesForRead.Keys.Max();
            double value = t.IntermediateValuesForRead[maxStep];
            // We rewrite the value of the trial `t` for sampling, so we need a copy.
            var copied = t.DeepCopy();
            copied.Value = value;
            return copied;
        }
        return null;
    }

    /// <inheritdoc />
    public override void BeforeTrial(Study study, FrozenTrial trial)
    {
        _independentSampler.BeforeTrial(study, trial);
    }

    /// <inheritdoc />
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        // Python `if trial.constraints:` — FrozenTrial.Constraints is never null (an empty dict when
        // unset), so warn only when at least one constraint was actually set on the trial.
        if (trial.Constraints.Count > 0)
            OptunaWarnings.Warn(
                "Constraints have been set, but CmaEsSampler will not use them for optimization.",
                WarningCategory.UserWarning);
        _independentSampler.AfterTrial(study, trial, state, values);
    }
}
