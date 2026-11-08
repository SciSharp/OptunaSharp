using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp.SearchSpace;

/// <summary>
///     Port of <c>optuna/search_space/intersection.py</c>.
/// </summary>
public static class SearchSpaceFunctions
{
    /// <summary>Python <c>optuna.search_space.intersection._calculate</c>.</summary>
    /// <param name="trials">Trial snapshots sorted by ascending study-local number; the incremental fold scans them in reverse.</param>
    /// <param name="includePruned">Whether pruned trials contribute their recorded parameter distributions.</param>
    /// <param name="searchSpace">The previously intersected distributions, or null when no finished trial has initialized the cache.</param>
    /// <param name="cachedTrialNumber">The earliest trial number to revisit; minus one performs a complete scan.</param>
    /// <returns>The intersected distributions and the earliest trial number that must be revisited next time.</returns>
    internal static (OrderedDict<string, BaseDistribution>? SearchSpace, int NextCachedTrialNumber) Calculate(
        IReadOnlyList<FrozenTrial> trials,
        bool includePruned = false,
        OrderedDict<string, BaseDistribution>? searchSpace = null,
        int cachedTrialNumber = -1)
    {
        var statesOfInterest = new List<TrialState>
        {
            TrialState.Complete,
            TrialState.Waiting,
            TrialState.Running,
        };

        if (includePruned)
            statesOfInterest.Add(TrialState.Pruned);

        int nextCachedTrialNumber = -1;

        for (int i = trials.Count - 1; i >= 0; i--)
        {
            var trial = trials[i];
            if (!statesOfInterest.Contains(trial.State))
                continue;

            if (nextCachedTrialNumber == -1)
                nextCachedTrialNumber = trial.Number + 1;

            if (cachedTrialNumber > trial.Number)
                break;

            if (!trial.State.IsFinished())
            {
                // An unfinished trial may later contribute distributions. Keep the cache boundary
                // before it even while folding newer finished trials, so completion is not skipped.
                nextCachedTrialNumber = trial.Number;
                continue;
            }

            if (searchSpace is null)
            {
                searchSpace = new OrderedDict<string, BaseDistribution>(trial.DistributionsDict);
                continue;
            }

            var next = new OrderedDict<string, BaseDistribution>();
            foreach (var (name, distribution) in searchSpace)
            {
                if (trial.DistributionsDict.TryGetValue(name, out var trialDist) && trialDist.Equals(distribution))
                    next[name] = distribution;
            }
            searchSpace = next;
        }

        return (searchSpace, nextCachedTrialNumber);
    }

    /// <summary>
    ///     Return the intersection search space of the given trials.
    ///
    ///     Intersection search space contains the intersection of parameter distributions that
    ///     have been suggested in the completed trials of the study so far. If there are multiple
    ///     parameters that have the same name but different distributions, neither is included in
    ///     the resulting search space (i.e., the parameters with dynamic value ranges are
    ///     excluded).
    /// </summary>
    /// <returns>
    ///     A dictionary containing the parameter names and parameter's distributions sorted by
    ///     parameter names.
    /// </returns>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="includePruned">Whether pruned trials contribute their recorded parameter distributions.</param>
    public static OrderedDict<string, BaseDistribution> IntersectionSearchSpaceOf(
        IReadOnlyList<FrozenTrial> trials, bool includePruned = false)
    {
        var (searchSpace, _) = Calculate(trials, includePruned);
        return Sorted(searchSpace ?? new OrderedDict<string, BaseDistribution>());
    }

    /// <summary>Returns a new insertion-ordered dictionary sorted by ordinal parameter name.</summary>
    /// <param name="searchSpace">The parameter distributions in the column order used by the transform or model.</param>
    /// <returns>A new insertion-ordered dictionary sorted by ordinal parameter name.</returns>
    internal static OrderedDict<string, BaseDistribution> Sorted(OrderedDict<string, BaseDistribution> searchSpace)
    {
        var result = new OrderedDict<string, BaseDistribution>(searchSpace.Count);
        foreach (var kv in searchSpace.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            result[kv.Key] = kv.Value;
        return result;
    }
}

/// <summary>
///     A class to calculate the intersection search space of a <see cref="Study" />.
///     Port of <c>optuna/search_space/intersection.py: IntersectionSearchSpace</c>.
///
///     Intersection search space contains the intersection of parameter distributions that have
///     been suggested in the completed trials of the study so far. If there are multiple
///     parameters that have the same name but different distributions, neither is included in the
///     resulting search space (i.e., the parameters with dynamic value ranges are excluded).
///
///     Note that an instance of this class is supposed to be used for only one study. If
///     different studies are passed to <see cref="Calculate(Study, bool)" />, a
///     <see cref="ValueError" /> is raised.
/// </summary>
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.search_space.IntersectionSearchSpace.html" />
public class IntersectionSearchSpace
{
    private int _cachedTrialNumber = -1;
    private OrderedDict<string, BaseDistribution>? _searchSpace;
    private int? _studyId;

    private readonly bool _includePruned;

    /// <param name="includePruned">Whether pruned trials should be included in the search space.</param>
    /// <summary>Creates a per-study incremental intersection cache with optional pruned-trial handling.</summary>
    public IntersectionSearchSpace(bool includePruned = false)
    {
        _includePruned = includePruned;
    }

    /// <summary>
    ///     Returns the intersection search space of the <see cref="Study" />.
    /// </summary>
    /// <param name="study">
    ///     A study with completed trials. The same study must be passed for one instance of this
    ///     class through its lifetime.
    /// </param>
    /// <param name="useCache">An option to use cached trials for each trial.</param>
    /// <returns>
    ///     A dictionary containing the parameter names and parameter's distributions sorted by
    ///     parameter names.
    /// </returns>
    /// <exception cref="NumSharp.ValueError">This cache is reused with a different study identifier.</exception>
    /// <remarks>
    ///     <para>The returned dictionary is a new sorted container. Distribution objects are shared with trial snapshots and are treated as immutable.</para>
    ///     <para>
    ///         The trials are read through <c>Study.GetTrialsForSampler</c>: on an in-memory storage that is the
    ///         zero-copy view, so a call costs only the trials since the cached trial number, where Optuna copies the
    ///         whole history first. The walk below already stops at the cached number; the copy was the part that grew
    ///         with the study. Elsewhere the read is Optuna's, cache included.
    ///     </para>
    /// </remarks>
    public OrderedDict<string, BaseDistribution> Calculate(Study study, bool useCache = false)
    {
        if (_studyId is null)
        {
            _studyId = study.StudyId;
        }
        else
        {
            // Note that the check below is meaningless when InMemoryStorage is used because
            // InMemoryStorage.CreateNewStudy always returns the same study ID.
            if (_studyId != study.StudyId)
                throw new ValueError("`IntersectionSearchSpace` cannot handle multiple studies.");
        }

        (_searchSpace, _cachedTrialNumber) = SearchSpaceFunctions.Calculate(
            study.GetTrialsForSampler(useCache),
            _includePruned,
            _searchSpace,
            _cachedTrialNumber);
        var searchSpace = _searchSpace ?? new OrderedDict<string, BaseDistribution>();
        return SearchSpaceFunctions.Sorted(searchSpace);
    }
}
