using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.SearchSpace;

/// <summary>
///     Port of <c>optuna/search_space/intersection.py</c>.
/// </summary>
public static class SearchSpaceFunctions
{
    /// <summary>Python <c>optuna.search_space.intersection._calculate</c>.</summary>
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
    public static OrderedDict<string, BaseDistribution> IntersectionSearchSpaceOf(
        IReadOnlyList<FrozenTrial> trials, bool includePruned = false)
    {
        var (searchSpace, _) = Calculate(trials, includePruned);
        return Sorted(searchSpace ?? new OrderedDict<string, BaseDistribution>());
    }

    /// <summary>
    ///     OptunaSharp extension (no Optuna analog): the <see cref="Calculate" /> fold for a study whose
    ///     objective has declared <see cref="ConditionalParams" />. Unconditional parameters intersect
    ///     exactly as in <see cref="Calculate" />. A conditional parameter is kept when a trial lacks it
    ///     (it was inactive there), joins the space the first time any trial carries it, and is excluded
    ///     for good (recorded in <paramref name="excluded" />) once two trials carry it with different
    ///     distributions — so, like the intersection, the result is order-independent: a conditional
    ///     parameter is in it iff some trial suggested it and every trial that did agrees on its
    ///     distribution. Completed trials whose parameter set contradicts the rules are reported through
    ///     <see cref="OptunaWarnings" /> once per parameter (tracked in <paramref name="warned" />; pass
    ///     <c>null</c> to stay silent). Same incremental-cache contract as <see cref="Calculate" />.
    /// </summary>
    internal static (OrderedDict<string, BaseDistribution>? SearchSpace, int NextCachedTrialNumber) CalculateConditional(
        IReadOnlyList<FrozenTrial> trials,
        bool includePruned,
        OrderedDict<string, BaseDistribution>? searchSpace,
        int cachedTrialNumber,
        ConditionalParams conditionalParams,
        HashSet<string> excluded,
        HashSet<string>? warned)
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
                nextCachedTrialNumber = trial.Number;
                continue;
            }

            // Each finished trial is folded exactly once (the cache skips it afterwards), so this is the
            // one place to check it against the rules. Pruned trials stop mid-objective and legitimately
            // lack parameters they never reached, so only complete trials are judged.
            if (warned is not null && trial.State == TrialState.Complete)
                WarnContradictions(conditionalParams, trial, warned);

            if (searchSpace is null)
            {
                searchSpace = new OrderedDict<string, BaseDistribution>(trial.DistributionsDict);
                continue;
            }

            var next = new OrderedDict<string, BaseDistribution>();
            foreach (var (name, distribution) in searchSpace)
            {
                if (trial.DistributionsDict.TryGetValue(name, out var trialDist))
                {
                    if (trialDist.Equals(distribution))
                        next[name] = distribution;
                    else if (conditionalParams.IsConditional(name))
                        excluded.Add(name);  // A dynamic distribution: never evolve it relatively.
                }
                else if (conditionalParams.IsConditional(name))
                {
                    next[name] = distribution;  // Absent because inactive — the expected case.
                }
            }
            foreach (var (name, distribution) in trial.DistributionsDict)
            {
                if (!conditionalParams.IsConditional(name) || next.ContainsKey(name) || excluded.Contains(name))
                    continue;
                next[name] = distribution;  // First sighting of a conditional parameter.
            }
            searchSpace = next;
        }

        return (searchSpace, nextCachedTrialNumber);
    }

    private static void WarnContradictions(ConditionalParams conditionalParams, FrozenTrial trial, HashSet<string> warned)
    {
        foreach (var (name, message) in conditionalParams.Contradictions(trial))
        {
            if (!warned.Add(name))
                continue;
            OptunaWarnings.Warn(
                $"{message}. The `ConditionalParams` rules and the objective's suggest calls disagree about " +
                $"`{name}`; the sampler keeps working, but where they disagree that parameter is evolved less " +
                "effectively. This warning is shown once per parameter.");
        }
    }

    /// <summary>
    ///     OptunaSharp extension (no Optuna analog): the search space of the given trials with declared
    ///     <see cref="ConditionalParams" /> — the intersection over unconditional parameters, plus every
    ///     conditional parameter some trial suggested whose distribution never changed. See
    ///     <see cref="CalculateConditional" />.
    /// </summary>
    /// <returns>
    ///     A dictionary containing the parameter names and parameter's distributions sorted by
    ///     parameter names.
    /// </returns>
    public static OrderedDict<string, BaseDistribution> IntersectionSearchSpaceOf(
        IReadOnlyList<FrozenTrial> trials, ConditionalParams conditionalParams, bool includePruned = false)
    {
        if (conditionalParams is null)
            throw new ArgumentNullException(nameof(conditionalParams));
        var (searchSpace, _) = CalculateConditional(
            trials, includePruned, searchSpace: null, cachedTrialNumber: -1, conditionalParams,
            excluded: new HashSet<string>(StringComparer.Ordinal), warned: null);
        return Sorted(searchSpace ?? new OrderedDict<string, BaseDistribution>());
    }

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
public class IntersectionSearchSpace
{
    private int _cachedTrialNumber = -1;
    private OrderedDict<string, BaseDistribution>? _searchSpace;
    private int? _studyId;

    private readonly bool _includePruned;
    private readonly ConditionalParams? _conditionalParams;
    // Conditional parameters seen with conflicting distributions (tombstoned for the study's lifetime)
    // and those already reported as contradicting the rules — both part of the incremental cache state.
    private readonly HashSet<string> _excludedConditional = new(StringComparer.Ordinal);
    private readonly HashSet<string> _warnedConditional = new(StringComparer.Ordinal);

    /// <param name="includePruned">Whether pruned trials should be included in the search space.</param>
    /// <param name="conditionalParams">
    ///     OptunaSharp extension: parameters declared conditional stay in the space when a trial lacks
    ///     them (they were inactive there) instead of falling out of the intersection — see
    ///     <see cref="ConditionalParams" /> and <see cref="SearchSpaceFunctions.CalculateConditional" />.
    ///     <c>null</c> (default) is Optuna's plain intersection.
    /// </param>
    public IntersectionSearchSpace(bool includePruned = false, ConditionalParams? conditionalParams = null)
    {
        _includePruned = includePruned;
        _conditionalParams = conditionalParams;
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

        if (_conditionalParams is null)
        {
            (_searchSpace, _cachedTrialNumber) = SearchSpaceFunctions.Calculate(
                study.GetTrialsInternal(deepcopy: false, useCache: useCache),
                _includePruned,
                _searchSpace,
                _cachedTrialNumber);
        }
        else
        {
            (_searchSpace, _cachedTrialNumber) = SearchSpaceFunctions.CalculateConditional(
                study.GetTrialsInternal(deepcopy: false, useCache: useCache),
                _includePruned,
                _searchSpace,
                _cachedTrialNumber,
                _conditionalParams,
                _excludedConditional,
                _warnedConditional);
        }
        var searchSpace = _searchSpace ?? new OrderedDict<string, BaseDistribution>();
        return SearchSpaceFunctions.Sorted(searchSpace);
    }
}
