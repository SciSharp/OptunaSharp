using System;
using System.Collections.Generic;
using OptunaSharp.Storages;

namespace OptunaSharp.Samplers.CmaEs;

/// <summary>
///     The parts of a study's history <see cref="CmaEsSampler" /> reads on every ask, kept up to date incrementally:
///     how many trials Optuna's <c>_get_trials</c> would return, the latest of them holding a stored optimizer state
///     (<c>_restore_optimizer</c>'s backward scan), and the ones stamped with a given generation
///     (<c>_get_solution_trials</c>). Optuna rebuilds all three from a full copy of the history on every ask, which
///     makes a run quadratic in its trial count; this class gives the same trials in the same order at the cost of the
///     trials that changed since the last ask (Rustuna plan 13, item S2).
/// </summary>
/// <remarks>
///     <para>
///         <b>Why the results are identical.</b> A finished trial cannot change (storages refuse updates to it), so the
///         leading run of finished trials, in trial-number order, is folded in once: each completed one is counted,
///         filed under its generation stamp, and remembered when it holds an optimizer state. Every refresh classifies
///         only the trials after the first unfinished one (the "tail"), because those may still change. Folded trials
///         all precede the tail's, so "folded, then tail" is Optuna's scan order. A trial that finishes out of order
///         (another worker's) waits in the tail until every earlier trial has finished, which keeps that order.
///     </para>
///     <para>
///         <b>Not thread-safe.</b> Its single owner calls it inside <c>SampleRelative</c> / <c>SampleIndependent</c>,
///         which <c>Trial</c> runs under the sampler's lock (<c>BaseSampler.Locked</c>); a per-worker sampler instance
///         owns its own history.
///     </para>
///     <para>
///         <b>Cost of a stuck trial.</b> A trial that stays RUNNING (a dead worker without heartbeats) stops the fold, so
///         every later refresh rescans the tail from it: speed degrades toward Optuna's per-ask cost, results stay
///         the same. Failing stale trials (<c>fail_stale_trials</c>) lets the fold move on.
///     </para>
/// </remarks>
internal sealed class CmaTrialHistory
{
    /// <summary>
    ///     Maps a finished or running trial to the snapshot the sampler treats as completed (itself, or a pruned trial
    ///     rewritten with its last intermediate value), or null when the sampler ignores it. Mirrors the filter of
    ///     Optuna's <c>CmaEsSampler._get_trials</c>.
    /// </summary>
    private readonly Func<FrozenTrial, FrozenTrial?> _asSamplerTrial;

    /// <summary>The system-attribute key of the generation stamp, e.g. <c>cma:generation</c>.</summary>
    private readonly string _generationKey;

    /// <summary>The prefix of the stored optimizer-state chunks, e.g. <c>cma:optimizer</c>.</summary>
    private readonly string _optimizerKeyPrefix;

    /// <summary>The storage the fold describes; trial ids and positions mean nothing in another one.</summary>
    private BaseStorage? _storage;

    /// <summary>The study the fold describes.</summary>
    private int _studyId = -1;

    /// <summary>How many leading trials (by number) are finished and folded in.</summary>
    private int _foldedCount;

    /// <summary>
    ///     The id of the last folded trial: a history whose trial at that position has another id is not the one
    ///     folded (a new study under a reused id), so the fold starts over.
    /// </summary>
    private int _lastFoldedTrialId = -1;

    /// <summary>How many folded trials the sampler treats as completed.</summary>
    private int _foldedCompletedCount;

    /// <summary>The folded completed trials by generation stamp (-1 when unstamped), each list in trial order.</summary>
    private readonly Dictionary<long, List<FrozenTrial>> _foldedByGeneration = new();

    /// <summary>The latest folded completed trial that holds an optimizer state; null when none does.</summary>
    private FrozenTrial? _foldedLatestStateHolder;

    /// <summary>The completed trials after the first unfinished one, from the last refresh, in trial order.</summary>
    private readonly List<FrozenTrial> _tail = new();

    /// <summary>Creates an empty history; the first <see cref="Refresh" /> folds the study.</summary>
    /// <param name="asSamplerTrial">
    ///     The sampler's completed-trial filter (see <see cref="_asSamplerTrial" />). It is called once per trial when the
    ///     trial is folded, and on every refresh for the trials in the tail, so it must not depend on anything but the
    ///     trial; the sampler's pruned-trial copies are made once and reused for the rest of the run.
    /// </param>
    /// <param name="generationKey">The generation stamp's system-attribute key.</param>
    /// <param name="optimizerKeyPrefix">The optimizer-state chunks' key prefix.</param>
    /// <exception cref="ArgumentNullException">An argument is <see langword="null" />.</exception>
    internal CmaTrialHistory(Func<FrozenTrial, FrozenTrial?> asSamplerTrial, string generationKey, string optimizerKeyPrefix)
    {
        _asSamplerTrial = asSamplerTrial ?? throw new ArgumentNullException(nameof(asSamplerTrial));
        _generationKey = generationKey ?? throw new ArgumentNullException(nameof(generationKey));
        _optimizerKeyPrefix = optimizerKeyPrefix ?? throw new ArgumentNullException(nameof(optimizerKeyPrefix));
    }

    /// <summary>The number of trials Optuna's <c>_get_trials</c> would return at the last refresh.</summary>
    internal int CompletedCount => _foldedCompletedCount + _tail.Count;

    /// <summary>
    ///     The latest completed trial, in trial order, that holds a stored optimizer state: the trial Optuna's
    ///     <c>_restore_optimizer</c> restores from. Null when no completed trial holds one.
    /// </summary>
    internal FrozenTrial? LatestStateHolder
    {
        get
        {
            // The tail's trials come after every folded one, so a holder there is the latest.
            for (int i = _tail.Count - 1; i >= 0; i--)
                if (HoldsOptimizerState(_tail[i]))
                    return _tail[i];
            return _foldedLatestStateHolder;
        }
    }

    /// <summary>
    ///     Whether the last <see cref="Refresh" /> started over (the first refresh, another storage or study, or a
    ///     history that no longer starts with the folded trials). The owner must then drop anything it keyed by trial
    ///     id, because ids from another storage can repeat.
    /// </summary>
    internal bool WasReset { get; private set; }

    /// <summary>
    ///     Brings the fold up to date with <paramref name="history" /> and classifies the trials after its first
    ///     unfinished one.
    /// </summary>
    /// <param name="history">
    ///     Every trial of the study in trial-number order, as <c>Study.GetTrialsForSampler</c> returns it. Read once per
    ///     position; not kept.
    /// </param>
    /// <param name="storage">The storage the history came from.</param>
    /// <param name="studyId">The study the history belongs to.</param>
    /// <exception cref="ArgumentNullException"><paramref name="history" /> or <paramref name="storage" /> is <see langword="null" />.</exception>
    internal void Refresh(IReadOnlyList<FrozenTrial> history, BaseStorage storage, int studyId)
    {
        ArgumentNullException.ThrowIfNull(history);
        ArgumentNullException.ThrowIfNull(storage);

        // The fold is valid only for the history it was built from: same storage, same study, and the folded
        // positions still hold the same trials. Anything else (a new study under a reused id, a fresh storage) starts
        // over; trial ids are unique within one storage, so the last folded id identifies the history.
        bool sameHistory = ReferenceEquals(_storage, storage) && _studyId == studyId
            && history.Count >= _foldedCount
            && (_foldedCount == 0 || history[_foldedCount - 1].TrialId == _lastFoldedTrialId);
        WasReset = !sameHistory;
        if (!sameHistory)
        {
            _storage = storage;
            _studyId = studyId;
            _foldedCount = 0;
            _lastFoldedTrialId = -1;
            _foldedCompletedCount = 0;
            _foldedByGeneration.Clear();
            _foldedLatestStateHolder = null;
        }

        // Fold the leading finished trials. Each history read happens once; the snapshot read is the one folded.
        while (_foldedCount < history.Count)
        {
            FrozenTrial source = history[_foldedCount];
            if (!source.State.IsFinished())
                break;
            FrozenTrial? completed = _asSamplerTrial(source);
            if (completed is not null)
            {
                _foldedCompletedCount++;
                long generation = GenerationOf(completed);
                if (!_foldedByGeneration.TryGetValue(generation, out List<FrozenTrial>? sameGeneration))
                    _foldedByGeneration[generation] = sameGeneration = new List<FrozenTrial>();
                sameGeneration.Add(completed);
                if (HoldsOptimizerState(completed))
                    _foldedLatestStateHolder = completed;
            }
            _lastFoldedTrialId = source.TrialId;
            _foldedCount++;
        }

        // The tail: trials after the first unfinished one, classified afresh each refresh because they may change.
        _tail.Clear();
        for (int i = _foldedCount; i < history.Count; i++)
        {
            FrozenTrial? completed = _asSamplerTrial(history[i]);
            if (completed is not null)
                _tail.Add(completed);
        }
    }

    /// <summary>
    ///     The completed trials stamped with <paramref name="generation" />, in trial order: what Optuna's
    ///     <c>_get_solution_trials</c> returns from the trials of the last refresh.
    /// </summary>
    /// <param name="generation">The optimizer generation whose trials the next tell needs.</param>
    /// <returns>A new list the caller may keep; folded trials first, then the tail's.</returns>
    internal List<FrozenTrial> SolutionTrials(int generation)
    {
        var result = _foldedByGeneration.TryGetValue(generation, out List<FrozenTrial>? folded)
            ? new List<FrozenTrial>(folded)
            : new List<FrozenTrial>();
        foreach (FrozenTrial trial in _tail)
            if (GenerationOf(trial) == generation)
                result.Add(trial);
        return result;
    }

    /// <summary>The trial's generation stamp, or -1 (Optuna's default) without one.</summary>
    /// <param name="trial">A trial the sampler treats as completed.</param>
    /// <returns>The stamp the sampler wrote when the trial asked.</returns>
    /// <exception cref="InvalidCastException">The stamp is not a number (a storage corrupted by hand).</exception>
    private long GenerationOf(FrozenTrial trial)
        => trial.SystemAttrs.TryGetValue(_generationKey, out object? stamp) ? Convert.ToInt64(stamp) : -1;

    /// <summary>
    ///     Whether the trial stores an optimizer state: any system attribute under the chunk prefix, the test of
    ///     Optuna's <c>_restore_optimizer</c>.
    /// </summary>
    /// <param name="trial">A trial the sampler treats as completed.</param>
    /// <returns>True when at least one optimizer-state chunk is present.</returns>
    private bool HoldsOptimizerState(FrozenTrial trial)
    {
        foreach (string key in trial.SystemAttrs.Keys)
            if (key.StartsWith(_optimizerKeyPrefix, StringComparison.Ordinal))
                return true;
        return false;
    }
}
