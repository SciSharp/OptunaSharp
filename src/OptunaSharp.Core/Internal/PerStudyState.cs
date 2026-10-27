using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using OptunaSharp.Storages;

namespace OptunaSharp.Internal;

/// <summary>
///     State a sampler instance keeps between asks for one study, stored per <b>storage object and study id</b> so
///     that one sampler instance can serve several studies one after another, as every Optuna sampler can.
/// </summary>
/// <typeparam name="TState">The mutable per-study state; created on first use and never shared between studies.</typeparam>
/// <remarks>
///     <para><b>Why not the study id alone.</b> Study ids are only unique inside one storage. Every fresh
///     <see cref="InMemoryStorage" /> and journal starts at id 0, so a cache keyed by the id alone hands study A's
///     population, generation counts or optimizer to study B as soon as a sampler is reused: a DI-registered
///     sampler, a benchmark loop, or a notebook that creates one study after another. Optuna's samplers keep no
///     such state (the GA base class rescans storage on every ask), so a reused Python sampler starts the second
///     study at generation 0. Pre-release review item H3 found OptunaSharp's incremental caches crashing
///     (<c>ArgumentOutOfRangeException: The view holds 1 trials</c>) or silently breeding study B from study A.</para>
///     <para><b>The key is <see cref="Study.Storage" /> by reference.</b> For in-memory and journal storages that is
///     the storage object itself, shared by every <see cref="Study" /> opened on it. An RDB storage is wrapped in a
///     <see cref="CachedStorage" /> per <see cref="Study" /> object, so two <see cref="Study" /> objects on the same
///     RDB study get separate states. That is deliberate: each wrapper caches trials on its own schedule, and a
///     state built from one wrapper's view must not be read against another's.</para>
///     <para><b>Lifetime.</b> Storages are held weakly (a <see cref="ConditionalWeakTable{TKey,TValue}" />): when a
///     storage becomes unreachable, the states of its studies go with it, so a long-lived sampler does not keep
///     every study it ever served alive.</para>
///     <para><b>Recreated studies.</b> A storage that reuses ids (SQLite reuses the highest deleted study and trial
///     ids; a gRPC proxy or a custom storage can pass that through unwrapped) can present a new study under an id
///     whose state is still here. Callers detect that through <c>isCurrent</c>, usually a <see cref="TrialAnchor" />
///     check, and get a fresh state.</para>
///     <para>Not synchronized: callers hold their own lock around <see cref="Resolve" /> and around every use of the
///     returned state.</para>
/// </remarks>
internal sealed class PerStudyState<TState> where TState : class
{
    private readonly ConditionalWeakTable<BaseStorage, Dictionary<int, TState>> _byStorage = new();

    /// <summary>
    ///     Returns the state of <paramref name="study" />, creating it on first use and replacing it when
    ///     <paramref name="isCurrent" /> reports that it was built for a study that no longer exists under this id.
    /// </summary>
    /// <param name="study">The study whose state is requested; keyed by its storage object and study id.</param>
    /// <param name="create">Builds an empty state. Called on first use and after a stale state is dropped.</param>
    /// <param name="isCurrent">
    ///     Whether a stored state still belongs to the study now stored under this id; <c>null</c> trusts every stored
    ///     state. Must only answer <c>false</c> when the study really changed: a replaced state loses everything it
    ///     accumulated.
    /// </param>
    /// <returns>The current state of the study; the caller may mutate it under its own lock.</returns>
    /// <exception cref="System.Exception">A <paramref name="create" /> or <paramref name="isCurrent" /> callback fails; its exception propagates and nothing is stored.</exception>
    internal TState Resolve(Study study, Func<TState> create, Func<TState, bool>? isCurrent = null)
    {
        // GetValue is atomic on its own; the inner dictionary is guarded by the caller's lock.
        Dictionary<int, TState> states = _byStorage.GetValue(study.Storage, static _ => new Dictionary<int, TState>());
        if (states.TryGetValue(study.StudyId, out TState? state) && (isCurrent is null || isCurrent(state)))
            return state;

        state = create();
        states[study.StudyId] = state;
        return state;
    }
}

/// <summary>
///     The identity of one trial (number, storage id, start time), kept by a per-study cache to notice that the
///     study it was built for has been deleted and another study now holds the same id.
/// </summary>
/// <remarks>
///     <para>A live study never loses a trial and never renumbers one, so the anchored trial is always found again
///     with the same identity. A recreated study under a reused id has a different trial at that number: a
///     different storage id, or, where SQLite reuses the deleted trial ids too, a later start time.</para>
///     <para>The start time is compared only when both sides have one. A waiting (enqueued) trial gets its start
///     time when it starts running, and that must not count as a different trial. Anchor a finished trial where
///     possible: its start time is always set.</para>
/// </remarks>
/// <param name="Number">The anchored trial's number within its study.</param>
/// <param name="TrialId">The anchored trial's storage identifier.</param>
/// <param name="DatetimeStart">The anchored trial's start time, or <c>null</c> while it was still waiting.</param>
internal readonly record struct TrialAnchor(int Number, int TrialId, DateTime? DatetimeStart)
{
    /// <summary>Captures the identity of <paramref name="trial" />.</summary>
    /// <param name="trial">The trial to anchor; only its number, storage id and start time are kept.</param>
    /// <returns>The anchor of the trial.</returns>
    internal static TrialAnchor Of(FrozenTrial trial) => new(trial.Number, trial.TrialId, trial.DatetimeStart);

    /// <summary>Whether <paramref name="trial" /> is the anchored trial (same storage id and, when both are known, start time).</summary>
    /// <param name="trial">A trial carrying the anchored number in the study now stored under the anchored study id.</param>
    /// <returns><c>true</c> when nothing distinguishes the trial from the anchored one.</returns>
    internal bool Matches(FrozenTrial trial)
        => trial.TrialId == TrialId
           && (DatetimeStart is null || trial.DatetimeStart is null || trial.DatetimeStart == DatetimeStart);

    /// <summary>Whether <paramref name="trials" /> still contain the anchored trial.</summary>
    /// <param name="trials">
    ///     The study's trials, or a subset of them such as the completed ones. A full trial list (where position equals
    ///     number) is checked in O(1); any other list is searched linearly for the anchored number.
    /// </param>
    /// <returns><c>true</c> when a trial with the anchored number is present and <see cref="Matches" /> it.</returns>
    internal bool IsIn(IReadOnlyList<FrozenTrial> trials)
    {
        // Fast path for a full trial list, where the trial numbered n sits at position n.
        if (Number < trials.Count && trials[Number].Number == Number)
            return Matches(trials[Number]);

        for (int i = 0; i < trials.Count; i++)
        {
            if (trials[i].Number == Number)
                return Matches(trials[i]);
        }
        return false;
    }
}
