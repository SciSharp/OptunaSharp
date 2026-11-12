using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Samplers;

/// <summary>
///     Sampler that performs exhaustive search over the define-and-run user-specified grids.
///     Port of <c>optuna/samplers/_grid.py: GridSampler</c>.
///
///     With <see cref="GridSampler" />, the trials suggest all combinations of parameters in the
///     given search space during the study.
///
///     <para>The C# analogue of Optuna's <c>GridValueType = str | float | int | bool | None</c>
///     is <see cref="CategoricalChoice" />, so the grid values carry the same Python-equality
///     semantics (and the same type preservation — int vs float — that the journal format
///     depends on).</para>
/// </summary>
/// <example>
///     <code>
///     var searchSpace = new Dictionary&lt;string, IReadOnlyList&lt;CategoricalChoice&gt;&gt;
///     {
///         ["x"] = new CategoricalChoice[] { -50, 0, 50 },
///         ["y"] = new CategoricalChoice[] { -99, 0, 99 },
///     };
///     var study = Optuna.CreateStudy(sampler: new GridSampler(searchSpace));
///     study.Optimize(Objective);
///     </code>
/// </example>
/// <remarks>
///     This sampler with ask-and-tell stops the optimization (internally invoking
///     <see cref="Study.Stop" />) just after evaluating the final grid; handle that case manually.
///     It does not take care of a parameter's quantization specified by discrete suggest methods
///     but just samples one of the values specified in the search space. During distributed
///     optimization different workers may occasionally suggest the same configuration, so the
///     total number of actual trials may exceed the size of the grid.
/// </remarks>
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GridSampler.html">Official Optuna GridSampler reference.</seealso>
public class GridSampler : BaseSampler
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.samplers._grid");

    // _search_space: sorted-by-key {name -> list(values)}, driving the grid product and same-space check.
    private readonly OrderedDict<string, List<CategoricalChoice>> _searchSpace;

    // The JSON-model image of _search_space stored as the `search_space` trial system attr — a
    // plain dict of primitive lists, byte-compatible with Python's json.dumps.
    private readonly OrderedDict<string, object?> _searchSpaceAttr;

    private readonly List<CategoricalChoice[]> _allGrids;
    private readonly string[] _paramNames;
    private readonly int _nMinTrials;
    private readonly LazyRandomState _rng;

    /// <param name="searchSpace">
    ///     A dictionary whose key and value are a parameter name and the corresponding candidates
    ///     of values, respectively.
    /// </param>
    /// <param name="seed">
    ///     A seed to fix the order of trials as the grid is randomly shuffled. Fixing the seed for
    ///     each process is strongly recommended in distributed optimization to avoid duplicated
    ///     suggestions.
    /// </param>
    /// <summary>Materializes and shuffles the Cartesian product of the supplied candidate values.</summary>
    public GridSampler(IReadOnlyDictionary<string, IReadOnlyList<CategoricalChoice>> searchSpace, int? seed = null)
    {
        // Python calls _check_value on every value; the CategoricalChoice value type makes any
        // non-{None,bool,int,float,str} value unrepresentable, so that check is subsumed here.

        // sorted(search_space.items()) — deterministic, Python-string (code-point) ordering.
        _searchSpace = new OrderedDict<string, List<CategoricalChoice>>();
        foreach (var paramName in searchSpace.Keys.OrderBy(k => k, StringComparer.Ordinal))
            _searchSpace[paramName] = searchSpace[paramName].ToList();

        var originalGrids = CartesianProduct(_searchSpace.Values.ToList());
        _paramNames = _searchSpace.Keys.ToArray();  // == sorted(search_space.keys())
        _nMinTrials = originalGrids.Count;
        // Python: LazyRandomState(seed or 0) — a None/0 seed both mean 0.
        _rng = new LazyRandomState(seed ?? 0);

        // Python shuffles the list `_all_grids` in place. NumPy's list shuffle draws the SAME
        // Fisher-Yates j-sequence as shuffling an equal-length ndarray, so we shuffle an index
        // permutation (bit-exact, verified against numpy) and reorder the grids by it.
        var permutation = np.arange(_nMinTrials);
        _rng.Rng.shuffle(permutation);
        var perm = permutation.ToArray<long>();
        _allGrids = new List<CategoricalChoice[]>(_nMinTrials);
        foreach (long index in perm)
            _allGrids.Add(originalGrids[(int)index]);

        _searchSpaceAttr = new OrderedDict<string, object?>();
        foreach (var (paramName, values) in _searchSpace)
            _searchSpaceAttr[paramName] = values.Select(v => v.Value).ToList();
    }

    /// <inheritdoc />
    public override void ReseedRng()
    {
        // Python: self._rng.rng.seed(), fresh OS entropy: 128 bits through numpy's own key construction,
        // so concurrent workers never share a stream (THREADING.md §6.3).
        OsEntropySeed.Reseed(_rng.Rng);
    }

    /// <summary>
    ///     A worker instance for <see cref="SamplerConcurrency.PerWorker" />: <paramref name="prototype" />'s grid (shared:
    ///     built and shuffled once, read only afterwards) with a generator of its own.
    /// </summary>
    /// <param name="prototype">The study's sampler whose grid the worker instance takes.</param>
    /// <remarks>Not the public constructor, which would rebuild and reshuffle the whole Cartesian product per worker.</remarks>
    private GridSampler(GridSampler prototype)
    {
        _searchSpace = prototype._searchSpace;
        _searchSpaceAttr = prototype._searchSpaceAttr;
        _allGrids = prototype._allGrids;
        _paramNames = prototype._paramNames;
        _nMinTrials = prototype._nMinTrials;
        // The executor reseeds it from OS entropy before each trial, as a worker reseeds the shared one.
        _rng = new LazyRandomState(0);
    }

    /// <summary>
    ///     A new <see cref="GridSampler" /> over the same grid, for a <see cref="SamplerConcurrency.PerWorker" /> worker.
    ///     The first trials, as many as there are grid points, take the point numbered like the trial, which no two
    ///     trials share; later trials claim a point no stored trial holds, chosen at random from the instance's own
    ///     generator (0 duplicate samples in the measured eight-instance runs). A later point can still be taken twice
    ///     while both claims are in flight, which Optuna documents for its shared sampler and its processes too.
    /// </summary>
    /// <returns>A fresh instance, or <see langword="null" /> for a subclass, which keeps shared access.</returns>
    internal override BaseSampler? CreateWorkerInstance()
        => GetType() == typeof(GridSampler) ? new GridSampler(this) : null;

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">
    ///     The grid is empty (a parameter's value list is empty) and the trial needs a grid point: NumPy's
    ///     <c>choice</c> error, <c>'a' cannot be empty unless no samples are taken</c>, as in Optuna.
    /// </exception>
    public override void BeforeTrial(Study study, FrozenTrial trial)
    {
        // Instead of returning param values, GridSampler puts the target grid id as a system
        // attr, and the values are returned from SampleIndependent. This is because the
        // distribution object is hard to get at the beginning of a trial, while we need access to
        // it to validate the sampled value.

        // When the trial is created by RetryFailedTrialCallback or enqueue_trial, we should not
        // assign a new grid_id.
        if (trial.SystemAttrs.ContainsKey("grid_id") || trial.SystemAttrs.ContainsKey("fixed_params"))
            return;

        if (0 <= trial.Number && trial.Number < _nMinTrials)
        {
            study.Storage.SetTrialSystemAttr(trial.TrialId, "search_space", _searchSpaceAttr);
            study.Storage.SetTrialSystemAttr(trial.TrialId, "grid_id", trial.Number);
            return;
        }

        var targetGrids = GetUnvisitedGridIds(study);

        if (targetGrids.Count == 0)
        {
            // This case may occur with distributed optimization or trial queue. If there is no
            // target grid, GridSampler evaluates a visited, duplicated point with the current
            // trial. After that, the optimization stops.
            _logger.Warning(
                "`GridSampler` is re-evaluating a configuration because the grid has been " +
                "exhausted. This may happen due to a timing issue during distributed optimization " +
                "or when re-running optimizations on already finished studies.");

            // One of all grids is randomly picked up in this case.
            targetGrids = Enumerable.Range(0, _allGrids.Count).ToList();
        }

        // In distributed optimization, multiple workers may simultaneously pick up the same grid.
        // To make the conflict less frequent, the grid is chosen randomly:
        // grid_id = int(self._rng.rng.choice(target_grids)), through NumSharp's port of the legacy RandomState.choice.
        // Without size, replace=False or p it draws target_grids[randint(0, len)], the stream the former shortcut used,
        // and an empty list (a search space with an empty value list, whose grid is empty) raises NumPy's own
        // ValueError, "'a' cannot be empty unless no samples are taken" (QA-55), where the shortcut's randint(0, 0)
        // raised "low >= high". One call per trial past the grid's first pass.
        int gridId;
        using (NDArray candidates = np.array(targetGrids.Select(g => (long)g).ToArray()))
        using (NDArray picked = _rng.Rng.choice(candidates))
            gridId = (int)(long)picked;

        study.Storage.SetTrialSystemAttr(trial.TrialId, "search_space", _searchSpaceAttr);
        study.Storage.SetTrialSystemAttr(trial.TrialId, "grid_id", gridId);
    }

    /// <inheritdoc />
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
        => new();

    /// <inheritdoc />
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
        => new();

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">The trial has no grid identifier or the parameter name is absent from the configured grid.</exception>
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        if (!trial.SystemAttrs.ContainsKey("grid_id"))
            throw new ValueError("All parameters must be specified when using GridSampler with enqueue_trial.");

        if (!_searchSpace.ContainsKey(paramName))
            throw new ValueError($"The parameter name, {paramName}, is not found in the given grid.");

        int gridId = Convert.ToInt32(trial.SystemAttrs["grid_id"], CultureInfo.InvariantCulture);
        var paramValue = _allGrids[gridId][Array.IndexOf(_paramNames, paramName)];
        bool contains = paramDistribution.Contains(paramDistribution.ToInternalRepr(paramValue.Value));
        if (!contains)
        {
            OptunaWarnings.Warn(
                $"The value `{PythonRepr.Str(paramValue.Value)}` is out of range of the parameter `{paramName}`. " +
                $"The value will be used but the actual distribution is: `{paramDistribution}`.");
        }

        return paramValue.Value;
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.RuntimeError">The grid is exhausted and Study.Stop is invoked outside Study.Optimize, as in manual ask-and-tell usage.</exception>
    /// <exception cref="NumSharp.KeyError">One unvisited grid is left and the finished trial carries no <c>grid_id</c> (an enqueued trial), as in Optuna.</exception>
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        var targetGrids = GetUnvisitedGridIds(study);

        if (targetGrids.Count == 0)
        {
            study.Stop();
        }
        else if (targetGrids.Count == 1)
        {
            // Python's `get_trial_system_attrs(...)["grid_id"]`: an enqueued trial carries no grid id, and Optuna raises
            // KeyError('grid_id') here, not .NET's KeyNotFoundException (QA-66, found by the batched fuzz E61).
            if (!study.Storage.GetTrialSystemAttrs(trial.TrialId).TryGetValue("grid_id", out object? stamped))
                throw new KeyError("grid_id");
            int gridId = Convert.ToInt32(stamped, CultureInfo.InvariantCulture);
            if (gridId == targetGrids[0])
                study.Stop();
        }
    }

    /// <summary>Finds unfinished grids, preferring those not currently assigned to running trials.</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <returns>
    ///     Grid indices still requiring completion, preferring grids without running evaluations, in CPython's
    ///     set-iteration order (the order <c>rng.choice</c> indexes).
    /// </returns>
    private List<int> GetUnvisitedGridIds(Study study)
    {
        // List up unvisited grids based on already finished ones.
        var visitedGrids = new HashSet<int>();
        var runningGrids = new HashSet<int>();

        // We directly query the storage to get trials here instead of study.GetTrials, since some
        // pruners such as HyperbandPruner use the study transformed to filter trials.
        var trials = study.Storage.GetAllTrials(study.StudyId, deepcopy: false);

        foreach (var t in trials)
        {
            if (t.SystemAttrs.TryGetValue("grid_id", out var gridIdValue)
                && t.SystemAttrs.TryGetValue("search_space", out var searchSpaceValue)
                && SameSearchSpace(searchSpaceValue))
            {
                int gridId = Convert.ToInt32(gridIdValue, CultureInfo.InvariantCulture);
                if (t.State.IsFinished())
                    visitedGrids.Add(gridId);
                else if (t.State == TrialState.Running)
                    runningGrids.Add(gridId);
            }
        }

        // `list(set(range(n)) - set(visited_grids) - set(running_grids))`, in CPython's set-iteration order: before_trial
        // indexes this list with `rng.choice`, so ascending order picked another grid for the same draw. That branch is
        // not only concurrency's: a single-threaded optimize reaches it whenever the study's numbering is shifted
        // (trials from add_trial / enqueue_trial, another sampler, a resumed study) and >= 2 grids remain, e.g. 12 added
        // trials then a 16-point grid, where Optuna's 13th grid trial is x=11 and ascending order gave x=4 (QA-68).
        var unvisitedGrids = PyIntSet.Range(_nMinTrials).Difference(visitedGrids).Difference(runningGrids).ToList();

        // If evaluations for all grids have been started, return grids that have not yet finished
        // because all grids should be evaluated before stopping the optimization.
        if (unvisitedGrids.Count == 0)
            unvisitedGrids = PyIntSet.Range(_nMinTrials).Difference(visitedGrids).ToList();

        return unvisitedGrids;
    }

    /// <summary>Python <c>_grid_value_equal</c>: <c>==</c> except that two NaNs compare equal.</summary>
    /// <param name="value1">First categorical grid choice.</param>
    /// <param name="value2">Second categorical grid choice.</param>
    /// <returns>True for Python-equivalent choices, with NaN choices treated as equal.</returns>
    private static bool GridValueEqual(CategoricalChoice value1, CategoricalChoice value2)
        => CategoricalChoice.ChoicesEqual(value1, value2);

    /// <summary>Checks whether stored grid metadata describes the same ordered candidate lists.</summary>
    /// <param name="searchSpace">Deserialized search-space system attribute, expected to map parameter names to candidate lists.</param>
    /// <returns>True if parameter names and candidate values match the configured grid.</returns>
    private bool SameSearchSpace(object? searchSpace)
    {
        if (searchSpace is not IReadOnlyDictionary<string, object?> stored)
            return false;

        if (!new HashSet<string>(stored.Keys).SetEquals(_searchSpace.Keys))
            return false;

        foreach (var paramName in stored.Keys)
        {
            if (stored[paramName] is not IReadOnlyList<object?> storedValues)
                return false;

            var ownValues = _searchSpace[paramName];
            if (storedValues.Count != ownValues.Count)
                return false;

            for (int i = 0; i < storedValues.Count; i++)
            {
                if (!GridValueEqual(CategoricalChoice.From(storedValues[i]), ownValues[i]))
                    return false;
            }
        }

        return true;
    }

    /// <summary>Return <c>true</c> if all the possible params are evaluated, otherwise <c>false</c>.</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <returns>True if every grid configuration has a finished evaluation.</returns>
    public bool IsExhausted(Study study) => GetUnvisitedGridIds(study).Count == 0;

    /// <summary>Port of <c>itertools.product(*axes)</c> — the last axis varies fastest.</summary>
    /// <param name="axes">Candidate lists in parameter order.</param>
    /// <returns>All combinations with the last input axis varying fastest.</returns>
    private static List<CategoricalChoice[]> CartesianProduct(List<List<CategoricalChoice>> axes)
    {
        var result = new List<CategoricalChoice[]> { Array.Empty<CategoricalChoice>() };
        foreach (var pool in axes)
        {
            var extended = new List<CategoricalChoice[]>(result.Count * pool.Count);
            foreach (var prefix in result)
            {
                foreach (var item in pool)
                {
                    var row = new CategoricalChoice[prefix.Length + 1];
                    Array.Copy(prefix, row, prefix.Length);
                    row[prefix.Length] = item;
                    extended.Add(row);
                }
            }
            result = extended;
        }
        return result;
    }
}
