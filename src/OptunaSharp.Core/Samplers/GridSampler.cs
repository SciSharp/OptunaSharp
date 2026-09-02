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

    public override void ReseedRng()
    {
        // Python: self._rng.rng.seed() — reseed from OS entropy.
        Span<byte> buffer = stackalloc byte[4];
        System.Security.Cryptography.RandomNumberGenerator.Fill(buffer);
        _rng.Rng.seed(BitConverter.ToUInt32(buffer));
    }

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
        // To make the conflict less frequent, the grid is chosen randomly. numpy's
        // choice(target_grids) with no size/replace/p reduces to target_grids[randint(0, len)].
        int gridId = targetGrids[(int)_rng.Rng.randint(0, targetGrids.Count)];

        study.Storage.SetTrialSystemAttr(trial.TrialId, "search_space", _searchSpaceAttr);
        study.Storage.SetTrialSystemAttr(trial.TrialId, "grid_id", gridId);
    }

    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
        => new();

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
        => new();

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

    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        var targetGrids = GetUnvisitedGridIds(study);

        if (targetGrids.Count == 0)
        {
            study.Stop();
        }
        else if (targetGrids.Count == 1)
        {
            int gridId = Convert.ToInt32(
                study.Storage.GetTrialSystemAttrs(trial.TrialId)["grid_id"], CultureInfo.InvariantCulture);
            if (gridId == targetGrids[0])
                study.Stop();
        }
    }

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

        // [Misaligned] Python builds `list(set(range(n)) - visited - running)`, whose order is
        // CPython's set-iteration order; we return ascending order instead. This is observable
        // ONLY through the concurrency `choice(target_grids)` branch of before_trial (≥2 unvisited
        // grids while trial.number >= n_min_trials), which single-threaded optimize never reaches
        // — each trial finishes before the next, so the grid is exhausted (and the study stopped)
        // by the fast path. Under real concurrency the chosen grid is timing-dependent in Optuna
        // too, so bit-exact reproduction there is not a meaningful parity target.
        var unvisitedGrids = Enumerable.Range(0, _nMinTrials)
            .Where(g => !visitedGrids.Contains(g) && !runningGrids.Contains(g)).ToList();

        // If evaluations for all grids have been started, return grids that have not yet finished
        // because all grids should be evaluated before stopping the optimization.
        if (unvisitedGrids.Count == 0)
            unvisitedGrids = Enumerable.Range(0, _nMinTrials).Where(g => !visitedGrids.Contains(g)).ToList();

        return unvisitedGrids;
    }

    /// <summary>Python <c>_grid_value_equal</c>: <c>==</c> except that two NaNs compare equal.</summary>
    private static bool GridValueEqual(CategoricalChoice value1, CategoricalChoice value2)
        => CategoricalChoice.ChoicesEqual(value1, value2);

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
    public bool IsExhausted(Study study) => GetUnvisitedGridIds(study).Count == 0;

    /// <summary>Port of <c>itertools.product(*axes)</c> — the last axis varies fastest.</summary>
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
