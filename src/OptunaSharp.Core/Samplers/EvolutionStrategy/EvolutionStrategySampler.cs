// Shared Optuna-protocol plumbing for the Natural-Evolution-Strategy samplers (OpenAI-ES / SNES / xNES).
//
// The three ES samplers differ only in their numeric core (the IEvolutionStrategyOptimizer) and their
// serialization tag; everything else — inferring the relative search space, restoring/telling/asking the
// optimizer across trials, chunking optimizer state into system attrs, the independent-sampling fallback —
// is identical. This base captures that common structure once, following the `CmaEsSampler` idiom exactly:
// each trial restores the optimizer from history, completes a generation once `popsize` solutions exist
// (reconstructing the population from stored trial params, never from persisted noise), and asks one
// candidate clamped to the transformed [0,1] box.

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers.EvolutionStrategy;

/// <summary>
///     Base class for the Evolution-Strategy samplers. Drives an <see cref="IEvolutionStrategyOptimizer" />
///     through the Optuna study protocol the way <c>CmaEsSampler</c> drives <c>Cma</c>: continuous
///     Float/Int parameters form a relative search space; the population is spread across consecutive
///     trials; and the optimizer state round-trips through trial system attrs (own JSON, chunked to the
///     storage string-length limit) so a study resumes across process boundaries and distributed workers.
///
///     <para>Categorical parameters, single-valued distributions, and the first <c>nStartupTrials</c> trials
///     are handled by the independent sampler, exactly as in <c>CmaEsSampler</c>. Dynamic search spaces are
///     unsupported (the sampler falls back to independent sampling with a warning).</para>
/// </summary>
public abstract class EvolutionStrategySampler : BaseSampler
{
    // The value of a system_attr must be under 2046 characters on RDBStorage (same as CmaEsSampler).
    private const int SystemAttrMaxLength = 2045;

    private readonly BaseSampler _independentSampler;
    private readonly int _nStartupTrials;
    private bool _warnIndependentSampling;
    private readonly LazyRandomState _esRng;
    private readonly IntersectionSearchSpace _searchSpace;

    /// <param name="nStartupTrials">Independent sampling is used until this many trials complete; the ES
    ///     optimizer starts from its box-center mean once the threshold is reached.</param>
    /// <param name="independentSampler">Sampler for parameters outside the relative search space (default <see cref="RandomSampler" />).</param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently.</param>
    /// <param name="seed">Random seed selecting the optimizer's base noise seed (persisted per study).</param>
    /// <summary>Initializes the shared startup, RNG, and search-space state for an ES sampler.</summary>
    protected EvolutionStrategySampler(
        int nStartupTrials, BaseSampler? independentSampler, bool warnIndependentSampling, int? seed)
    {
        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _nStartupTrials = nStartupTrials;
        _warnIndependentSampling = warnIndependentSampling;
        _esRng = new LazyRandomState(seed);
        _searchSpace = new IntersectionSearchSpace();
    }

    /// <summary>The system-attr key prefix distinguishing this sampler's stored state (e.g. <c>"openaies:"</c>).</summary>
    protected abstract string AttrPrefix { get; }

    /// <summary>The logger namespace for this sampler (e.g. <c>"optuna.samplers.openai_es"</c>).</summary>
    protected abstract Logger SamplerLogger { get; }

    /// <summary>
    ///     Builds the initial optimizer for a fresh study: the caller supplies the transformed box center
    ///     (the initial mean) and a base noise seed; the subclass fills in its algorithm-specific initial
    ///     scale and hyperparameters.
    /// </summary>
    /// <param name="boxCenter">The transformed <c>[0,1]^d</c> box center, used as the initial distribution mean.</param>
    /// <param name="nDim">Number of numerical decision coordinates.</param>
    /// <param name="seed">The optimizer's persisted base seed for its per-generation noise streams.</param>
    /// <returns>A freshly initialized optimizer.</returns>
    /// <remarks><c>private protected</c> because <see cref="IEvolutionStrategyOptimizer" /> is an internal
    ///     implementation type: only same-assembly ES samplers may derive and override this hook.</remarks>
    private protected abstract IEvolutionStrategyOptimizer CreateOptimizer(double[] boxCenter, int nDim, uint seed);

    /// <summary>Serializes the optimizer to a tagged string (<c>"&lt;tag&gt;\n" + JSON</c>) for storage.</summary>
    /// <param name="optimizer">The optimizer whose evolving state is persisted.</param>
    /// <returns>The tagged serialized optimizer state.</returns>
    private protected abstract string SerializeOptimizer(IEvolutionStrategyOptimizer optimizer);

    /// <summary>Reconstructs the optimizer from a tagged string produced by <see cref="SerializeOptimizer" />.</summary>
    /// <param name="optimizerStr">The tagged serialized optimizer state (after chunk reassembly).</param>
    /// <returns>The optimizer reconstructed from its state.</returns>
    private protected abstract IEvolutionStrategyOptimizer DeserializeOptimizer(string optimizerStr);

    /// <inheritdoc />
    public override void ReseedRng()
    {
        // _esRng needs no reseed here: the optimizer noise is a pure function of its persisted seed and
        // generation, and relative sampling never advances _esRng after the optimizer is created.
        _independentSampler.ReseedRng();
    }

    /// <inheritdoc />
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _searchSpace.Calculate(study))
        {
            // A single-valued distribution is sampled inside Trial; categorical distributions are
            // unsupported (as in CmaEsSampler) and delegated to the independent sampler.
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

        if (searchSpace.Count == 0)
            return new OrderedDict<string, object?>();

        var completedTrials = GetTrials(study);
        if (completedTrials.Count < _nStartupTrials)
            return new OrderedDict<string, object?>();

        var trans = new SearchSpaceTransform(searchSpace, transformStep: true, transform01: true);
        double[] boundsFlat = trans.Bounds.ToArray<double>();
        int nDim = boundsFlat.Length / 2;
        var lower = new double[nDim];
        var upper = new double[nDim];
        var boxCenter = new double[nDim];
        for (int i = 0; i < nDim; i++)
        {
            lower[i] = boundsFlat[i * 2];
            upper[i] = boundsFlat[i * 2 + 1];
            boxCenter[i] = lower[i] + (upper[i] - lower[i]) / 2;   // [0,1] center = 0.5
        }

        var optimizer = RestoreOptimizer(completedTrials)
                        ?? CreateOptimizer(boxCenter, nDim, (uint)(int)NumPyDraws.RandInt(_esRng.Rng, 1, 2147483646));

        if (optimizer.Dim != nDim)
        {
            // A dynamic search space would make the persisted mean/covariance meaningless; fall back to
            // independent sampling (warn once), exactly as CmaEsSampler does.
            if (_warnIndependentSampling)
            {
                SamplerLogger.Warning(
                    $"`{GetType().Name}` does not support dynamic search space. " +
                    $"`{_independentSampler.GetType().Name}` is used instead of `{GetType().Name}`.");
                _warnIndependentSampling = false;
            }
            return new OrderedDict<string, object?>();
        }

        // Complete a generation once all `popsize` of its members have finished: reconstruct the population
        // from the stored (clamped, step-snapped) params, tell the optimizer, and re-serialize its state.
        var solutionTrials = GetSolutionTrials(completedTrials, optimizer.Generation);
        if (solutionTrials.Count >= optimizer.PopulationSize)
        {
            var solutions = new List<(double[] X, double Value)>(optimizer.PopulationSize);
            foreach (var t in solutionTrials.Take(optimizer.PopulationSize))
            {
                double[] x = trans.Transform(t.Params).ToArray<double>();
                double y = study.Direction == StudyDirection.Minimize ? t.Value!.Value : -t.Value!.Value;
                solutions.Add((x, y));
            }

            optimizer.Tell(solutions);

            string optimizerStr = SerializeOptimizer(optimizer);
            foreach (var (key, value) in SplitOptimizerStr(optimizerStr))
                study.Storage.SetTrialSystemAttr(trial.TrialId, key, value);

            // Refresh: the new generation has no stamped trials yet, so the next ask is its index 0.
            solutionTrials = GetSolutionTrials(completedTrials, optimizer.Generation);
        }

        // Position within the current generation (0-based): every candidate already handed out, finished or still
        // running. Counting only the finished ones gave all concurrent workers of a generation the same member, so
        // Optimize(nJobs: 16) evaluated 49-54 of 120 trials at a point another trial already had. Past the
        // population (workers asking at a generation's tail, before its last members finish) the index names an
        // extra candidate from the same noise stream instead of clamping onto the last member, as Optuna's CMA-ES
        // draws a fresh point there; Tell takes the first `popsize` finished candidates either way. A sequential run
        // has no other running candidate and never passes the population, so its indices, and seeded results, are
        // unchanged.
        int index = solutionTrials.Count + CountRunningMembers(study, trial, optimizer.Generation);

        double[] esParams = optimizer.Ask(index);
        for (int i = 0; i < nDim; i++)
            esParams[i] = Math.Min(Math.Max(esParams[i], lower[i]), upper[i]);   // clamp to the [0,1] box

        study.Storage.SetTrialSystemAttr(trial.TrialId, AttrKeyGeneration, optimizer.Generation);

        // View the candidate zero-copy; Untransform copies it into a managed dict, so dispose here.
        using NDArray nd = np.array(esParams, copy: false);
        return trans.Untransform(nd);
    }

    // Built once per sampler (plan 15): GetSolutionTrials and CountRunningMembers read the generation key for every
    // trial of the history on every ask, and concatenating it there allocated a new string per trial per scan, most of
    // the sampler's per-trial allocation. AttrPrefix is a constant per sampler type, so the cached keys are the strings
    // the concatenation produced; a race between two first readers only builds an equal string twice.
    private string? _attrKeyGeneration;
    private string? _attrKeyOptimizer;

    /// <summary>The system-attribute key that stamps a trial with the generation it was asked in.</summary>
    private string AttrKeyGeneration => _attrKeyGeneration ??= AttrPrefix + "generation";

    /// <summary>The system-attribute key prefix of the serialized optimizer state's chunks.</summary>
    private string AttrKeyOptimizer => _attrKeyOptimizer ??= AttrPrefix + "optimizer";

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

    /// <summary>Finds the newest stored optimizer snapshot in completed trial history.</summary>
    /// <param name="completedTrials">Completed trial snapshots used as historical observations.</param>
    /// <returns>The newest available optimizer, or null when no stored snapshot exists.</returns>
    private IEvolutionStrategyOptimizer? RestoreOptimizer(List<FrozenTrial> completedTrials)
    {
        for (int idx = completedTrials.Count - 1; idx >= 0; idx--)
        {
            var t = completedTrials[idx];
            var optimizerAttrs = t.SystemAttrs
                .Where(kv => kv.Key.StartsWith(AttrKeyOptimizer, StringComparison.Ordinal))
                .ToDictionary(kv => kv.Key, kv => (string)kv.Value!);
            if (optimizerAttrs.Count == 0) continue;
            return DeserializeOptimizer(ConcatOptimizerAttrs(optimizerAttrs));
        }
        return null;
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">The study has multiple objectives, but this sampler requires a scalar objective.</exception>
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        RaiseErrorIfMultiObjective(study);

        if (_warnIndependentSampling)
        {
            var completeTrials = GetTrials(study);
            if (completeTrials.Count >= _nStartupTrials)
                LogIndependentSampling(trial, paramName);
        }

        return _independentSampler.SampleIndependent(study, trial, paramName, paramDistribution);
    }

    /// <summary>Explains why the current parameter is delegated to the independent sampler.</summary>
    /// <param name="trial">Trial snapshot being examined; copy its contents before modifying them.</param>
    /// <param name="paramName">Name of the parameter being suggested or looked up.</param>
    private void LogIndependentSampling(FrozenTrial trial, string paramName)
    {
        SamplerLogger.Warning(string.Format(
            IndependentSamplingWarningTemplate,
            paramName,
            trial.Number,
            _independentSampler.GetType().Name,
            GetType().Name,
            $"dynamic search space and `CategoricalDistribution` are not supported by `{GetType().Name}`"));
    }

    /// <summary>Collects completed trials — the history this sampler adapts from.</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <returns>Completed trial snapshots.</returns>
    private static List<FrozenTrial> GetTrials(Study study)
    {
        List<FrozenTrial> all = study.GetTrialsInternal(deepcopy: false, states: null, useCache: true);
        // Sized for the whole history (plan 15): nearly every trial is complete, so the list never regrows on an ask.
        var completeTrials = new List<FrozenTrial>(all.Count);
        foreach (var t in all)
            if (t.State == TrialState.Complete)
                completeTrials.Add(t);
        return completeTrials;
    }

    /// <summary>
    ///     Counts the other trials still running a member of <paramref name="generation" />: members handed out but
    ///     not yet evaluated, which the next ask must skip.
    /// </summary>
    /// <param name="study">Study supplying trial history; its per-trial snapshot is read, as for the finished members.</param>
    /// <param name="trial">The trial being sampled; it carries no generation stamp yet and is excluded regardless.</param>
    /// <param name="generation">Zero-based generation whose running members are counted.</param>
    /// <returns>The number of running trials, other than <paramref name="trial" />, stamped with the generation.</returns>
    /// <exception cref="System.FormatException">A stored generation stamp cannot be parsed as an integer.</exception>
    /// <exception cref="System.InvalidCastException">A stored generation stamp does not support integer conversion.</exception>
    /// <exception cref="System.OverflowException">A stored generation stamp exceeds the range of a 64-bit integer.</exception>
    /// <remarks>
    ///     The snapshot is taken inside this sampler's monitor (see <see cref="BaseSampler" />), after every
    ///     earlier ask of this process stamped its trial, so in-process workers never share a member. Workers in other
    ///     processes stamp concurrently and can still collide, as Optuna's own samplers do across processes. A trial
    ///     whose worker died stays RUNNING and keeps its member reserved until the generation is told.
    /// </remarks>
    private int CountRunningMembers(Study study, FrozenTrial trial, int generation)
    {
        int running = 0;
        foreach (var t in study.GetTrialsInternal(deepcopy: false, states: null, useCache: true))
        {
            if (t.State != TrialState.Running || t.TrialId == trial.TrialId)
                continue;
            if (t.SystemAttrs.TryGetValue(AttrKeyGeneration, out var v) && Convert.ToInt64(v) == generation)
                running++;
        }
        return running;
    }

    /// <summary>Selects trials belonging to one optimizer generation so adaptation uses a coherent population.</summary>
    /// <param name="trials">Trial history available to the operation, in its existing order.</param>
    /// <param name="generation">Zero-based generation whose individuals are requested.</param>
    /// <returns>Trials stamped with the requested optimizer generation.</returns>
    private List<FrozenTrial> GetSolutionTrials(List<FrozenTrial> trials, int generation)
    {
        var result = new List<FrozenTrial>();
        foreach (var t in trials)
        {
            long g = t.SystemAttrs.TryGetValue(AttrKeyGeneration, out var v) ? Convert.ToInt64(v) : -1;
            if (generation == g) result.Add(t);
        }
        return result;
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
                $"Constraints have been set, but {GetType().Name} will not use them for optimization.",
                WarningCategory.UserWarning);
        _independentSampler.AfterTrial(study, trial, state, values);
    }
}
