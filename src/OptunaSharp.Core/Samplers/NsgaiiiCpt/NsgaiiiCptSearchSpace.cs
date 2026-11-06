using System;
using System.Collections.Generic;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers.NsgaiiiCpt;

/// <summary>
///     The conditional-parameter search space of <see cref="NSGAIIICptSampler" /> — an OptunaSharp
///     extension with no Optuna analog, extracted here so that <see cref="IntersectionSearchSpace" /> and
///     every other sampler stay a bit-exact Optuna port. It is the sole home of the "Categorical Parameter
///     Trees" (CPT) search-space fold.
///
///     <para>Unconditional parameters intersect exactly as Optuna's <see cref="IntersectionSearchSpace" />
///     does (this class delegates to the shared parity fold <see cref="SearchSpaceFunctions.Calculate" />
///     for that). A parameter declared conditional in the supplied <see cref="ConditionalParams" /> is
///     instead <b>kept</b> when a trial lacks it (it was inactive there), joins the space the first time any
///     trial carries it, and is excluded for good once two trials carry it with conflicting distributions —
///     so, like the intersection, the result is order-independent. Completed trials whose parameter set
///     contradicts the declared rules are reported once per parameter through <see cref="OptunaWarnings" />.</para>
///
///     <para><b>Parity contract.</b> Constructed with a <c>null</c> rule set this class is a byte-for-byte
///     pass-through to the shared intersection fold, so an <see cref="NSGAIIICptSampler" /> without rules is
///     identical to <see cref="NSGAIIISampler" />. The customized behavior only runs when rules are present.</para>
/// </summary>
public sealed class NsgaiiiCptSearchSpace
{
    private int _cachedTrialNumber = -1;
    private OrderedDict<string, BaseDistribution>? _searchSpace;
    private int? _studyId;
    // The sorted result of the last Calculate, valid while the fold advanced nothing: same folded space
    // instance AND same next-revisit number ⇒ the sorted view cannot differ. Saves the O(P log P) sort and
    // the O(P) container build on every ask of an unchanged space (P = parameter count, hundreds for a
    // conditional genome), which is the common case — the space only changes when a new trial finishes.
    private OrderedDict<string, BaseDistribution>? _sortedCache;
    private OrderedDict<string, BaseDistribution>? _sortedCacheSource;
    private int _sortedCacheTrialNumber = -2;

    private readonly bool _includePruned;
    private readonly ConditionalParams? _conditionalParams;
    // Conditional parameters seen with conflicting distributions (tombstoned for the study's lifetime)
    // and those already reported as contradicting the rules — both part of the incremental cache state.
    private readonly HashSet<string> _excludedConditional = new(StringComparer.Ordinal);
    private readonly HashSet<string> _warnedConditional = new(StringComparer.Ordinal);

    /// <summary>Creates a per-study incremental search-space cache aware of the declared conditional parameters.</summary>
    /// <param name="conditionalParams">
    ///     The activation rules that tell a conditional parameter's <em>inactive absence</em> from a trial apart
    ///     from a genuinely missing observation. <c>null</c> makes this a plain intersection (Optuna parity).
    /// </param>
    /// <param name="includePruned">Whether pruned trials should contribute their recorded parameter distributions.</param>
    public NsgaiiiCptSearchSpace(ConditionalParams? conditionalParams, bool includePruned = false)
    {
        _conditionalParams = conditionalParams;
        _includePruned = includePruned;
    }

    /// <summary>
    ///     Returns the (conditional-aware) search space of the <see cref="Study" />, mirroring
    ///     <see cref="IntersectionSearchSpace.Calculate" />'s per-study incremental cache contract.
    /// </summary>
    /// <param name="study">
    ///     A study with completed trials. The same study must be passed for one instance of this class through
    ///     its lifetime.
    /// </param>
    /// <param name="useCache">Whether to read the study's cached trial snapshots.</param>
    /// <returns>The parameter names and distributions, sorted by ordinal parameter name.</returns>
    /// <exception cref="NumSharp.ValueError">This cache is reused with a different study identifier.</exception>
    /// <remarks>The returned dictionary is a new sorted container. Distribution objects are shared with trial snapshots and treated as immutable.</remarks>
    public OrderedDict<string, BaseDistribution> Calculate(Study study, bool useCache = false)
    {
        if (_studyId is null)
        {
            _studyId = study.StudyId;
        }
        else
        {
            // Same guard as IntersectionSearchSpace: one incremental cache instance is bound to one study.
            // (Meaningless under InMemoryStorage, whose study id is constant, but real for RDB/journal storages.)
            if (_studyId != study.StudyId)
                throw new NumSharp.ValueError("`NsgaiiiCptSearchSpace` cannot handle multiple studies.");
        }

        // The zero-copy prefix view replaces the per-call list copy of GetTrialsInternal: both folds only
        // READ the trials (reverse walk, early break at the cache watermark), and the fold's incremental
        // contract already tolerates the view's live instances — a finished trial is folded exactly once,
        // and finished trials never change their params/distributions afterwards.
        IReadOnlyList<FrozenTrial> trials = study.GetTrialsView();
        if (_conditionalParams is null)
        {
            // No rules ⇒ the shared, Optuna-parity intersection fold, byte-for-byte.
            (_searchSpace, _cachedTrialNumber) = SearchSpaceFunctions.Calculate(
                trials,
                _includePruned,
                _searchSpace,
                _cachedTrialNumber);
        }
        else
        {
            (_searchSpace, _cachedTrialNumber) = CalculateConditional(
                trials,
                _includePruned,
                _searchSpace,
                _cachedTrialNumber,
                _conditionalParams,
                _excludedConditional,
                _warnedConditional);
        }
        var searchSpace = _searchSpace ?? new OrderedDict<string, BaseDistribution>();
        // Cache hit: the fold neither replaced the space nor moved its watermark, so the sorted view is the
        // one already built. Treat the returned dictionary as read-only — every in-repo caller only iterates it.
        if (ReferenceEquals(searchSpace, _sortedCacheSource) && _cachedTrialNumber == _sortedCacheTrialNumber && _sortedCache is not null)
            return _sortedCache;
        _sortedCache = SearchSpaceFunctions.Sorted(searchSpace);
        _sortedCacheSource = searchSpace;
        _sortedCacheTrialNumber = _cachedTrialNumber;
        return _sortedCache;
    }

    /// <summary>
    ///     The <see cref="SearchSpaceFunctions.Calculate" /> fold for a study whose objective declared
    ///     <see cref="ConditionalParams" />. Unconditional parameters intersect exactly as in the parity fold;
    ///     a conditional parameter is kept when a trial lacks it, joins on first sighting, and is tombstoned in
    ///     <paramref name="excluded" /> once observed with conflicting distributions. Completed trials that
    ///     contradict the rules are reported once per parameter (tracked in <paramref name="warned" />; pass
    ///     <c>null</c> to stay silent). Same incremental-cache contract as the parity fold.
    /// </summary>
    /// <param name="trials">Trial snapshots sorted by ascending study-local number; the incremental fold scans them in reverse.</param>
    /// <param name="includePruned">Whether pruned trials contribute their recorded parameter distributions.</param>
    /// <param name="searchSpace">The cached conditional-aware distributions, or null before the first finished trial is folded in.</param>
    /// <param name="cachedTrialNumber">The earliest trial number to revisit; minus one performs a complete scan.</param>
    /// <param name="conditionalParams">The declared activation rules used to distinguish inactive parameters from missing observations.</param>
    /// <param name="excluded">Conditional parameters previously observed with conflicting distributions; updated in place to prevent their reintroduction.</param>
    /// <param name="warned">Parameter names already reported as contradicting activation rules; null suppresses these warnings.</param>
    /// <returns>The conditional-aware distributions and the earliest trial number that must be revisited next time.</returns>
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

    /// <summary>Reports each parameter whose recorded presence contradicts its declared activation rules at most once.</summary>
    /// <param name="conditionalParams">The declared activation rules used to distinguish inactive parameters from missing observations.</param>
    /// <param name="trial">The trial whose recorded state and values are used.</param>
    /// <param name="warned">The non-null set of parameter names already reported; updated so each parameter warns at most once.</param>
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
    ///     The conditional-aware search space of the given trials in one shot (no incremental cache) — the
    ///     intersection over unconditional parameters plus every conditional parameter some trial suggested
    ///     whose distribution never changed. See <see cref="CalculateConditional" />.
    /// </summary>
    /// <param name="trials">Trial snapshots to inspect.</param>
    /// <param name="conditionalParams">The declared activation rules used to distinguish inactive parameters from missing observations.</param>
    /// <param name="includePruned">Whether pruned trials contribute their recorded parameter distributions.</param>
    /// <returns>The parameter names and distributions, sorted by ordinal parameter name.</returns>
    /// <exception cref="System.ArgumentNullException">The conditional-parameter rule collection is null.</exception>
    public static OrderedDict<string, BaseDistribution> IntersectionSearchSpaceOf(
        IReadOnlyList<FrozenTrial> trials, ConditionalParams conditionalParams, bool includePruned = false)
    {
        if (conditionalParams is null)
            throw new ArgumentNullException(nameof(conditionalParams));
        var (searchSpace, _) = CalculateConditional(
            trials, includePruned, searchSpace: null, cachedTrialNumber: -1, conditionalParams,
            excluded: new HashSet<string>(StringComparer.Ordinal), warned: null);
        return SearchSpaceFunctions.Sorted(searchSpace ?? new OrderedDict<string, BaseDistribution>());
    }
}
