using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Distributions;

namespace OptunaSharp.Storages;

/// <summary>
///     Base class for storages. Port of <c>optuna/storages/_base.py: BaseStorage</c>.
///
///     This class is not supposed to be directly accessed by library users. It abstracts a
///     backend database and provides internal interfaces to read/write histories of studies and
///     trials.
///
///     <para><b>Thread safety</b> — a storage class instance can be shared among multiple
///     threads, and must therefore be thread-safe. It must guarantee that a data instance read
///     from the storage must not be modified by subsequent writes: a <see cref="FrozenTrial" />
///     returned by <see cref="GetTrial" /> should not be updated by subsequent
///     <c>SetTrial...</c> calls. This is usually achieved by replacing the old data with a copy
///     on write.</para>
///
///     <para><b>Ownership of RUNNING trials</b> — trials in finished states are not allowed to
///     be modified; trials in the WAITING state are not allowed to be modified except for the
///     state field.</para>
/// </summary>
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/storages.html">Official Optuna storage API.</seealso>
public abstract class BaseStorage
{
    /// <summary>Prefix prepended to automatically generated study names.</summary>
    public const string DefaultStudyNamePrefix = "no-name-";

    // How many monitors TellLockFor spreads trials over: enough that unrelated tells rarely share one, few enough that
    // the set is a fixed, small allocation per storage instance however many trials the storage ever holds.
    private const int TellLockStripes = 64;

    private readonly object[] _tellLocks = Enumerable.Range(0, TellLockStripes).Select(_ => new object()).ToArray();

    /// <summary>
    ///     The monitor under which a tell of one trial through this storage instance runs, by <c>Study.Tell</c> and by
    ///     the optimize loop alike: the check that the trial is still running, the sampler's <c>AfterTrial</c>, and the
    ///     final state write. Two threads telling one trial therefore act in turn, and the second sees the first one's
    ///     result: it is refused with <c>Cannot tell a COMPLETE trial.</c>, or skipped with <c>skipIfFinished</c>.
    /// </summary>
    /// <param name="trialId">The trial's storage identifier.</param>
    /// <returns>
    ///     The trial's stripe of a fixed set of monitors. Trials that share a stripe only wait for each other's tell,
    ///     which is short; the monitor is reentrant, so a sampler hook telling another trial of a shared stripe on the
    ///     same thread does not deadlock.
    /// </returns>
    /// <remarks>
    ///     <para>
    ///         Python runs that check and that write as separate storage calls too. Under the GIL two threads
    ///         practically never pass the check together: on Optuna 5.0, 300 of 300 rounds of two threads telling one
    ///         trial stored one tell and refused the other. Here the threads run in parallel, and without the monitor
    ///         both passed the check in about one round of 150. The loser then ran the sampler's <c>AfterTrial</c> with
    ///         its own values, which were never stored, and failed in the storage with
    ///         <see cref="UpdateFinishedTrialError" />, even with <c>skipIfFinished</c>.
    ///     </para>
    ///     <para>
    ///         Lock order: this monitor is taken before the sampler's monitor and the storage's own locks, and no code
    ///         holding those takes it. Footgun: a Python-bridged sampler takes the GIL inside its <c>AfterTrial</c>, so
    ///         that order continues to the GIL, and code that holds the GIL while it calls <c>Tell</c> can deadlock
    ///         against a concurrent tell, as it already can around <c>Suggest</c> against the sampler's monitor. Tells
    ///         through two storage instances, or from two processes, are not covered: the storage arbitrates them, as
    ///         Python's does, and the loser gets <see cref="UpdateFinishedTrialError" />.
    ///     </para>
    /// </remarks>
    internal object TellLockFor(int trialId) => _tellLocks[(int)((uint)trialId % TellLockStripes)];

    // ------------------------------------------------------------------ study manipulation

    /// <summary>
    ///     Create a new study from a name. If no name is specified, the storage class generates
    ///     a name. The returned study ID is unique among all current and deleted studies.
    /// </summary>
    /// <exception cref="DuplicatedStudyError">If a study with the same name already exists.</exception>
    /// <param name="directions">Objective directions in the same order as the objective values.</param>
    /// <param name="studyName">Name identifying the study in the selected storage.</param>
    /// <returns>Unique storage identifier assigned to the created study.</returns>
    public abstract int CreateNewStudy(IReadOnlyList<StudyDirection> directions, string? studyName = null);

    /// <summary>Delete a study.</summary>
    /// <exception cref="KeyError">If no study with the matching ID exists.</exception>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    public abstract void DeleteStudy(int studyId);

    /// <summary>Register a user-defined attribute to a study, overwriting any existing value.</summary>
    /// <exception cref="KeyError">If no study with the matching ID exists.</exception>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="key">Attribute name; a later write replaces its stored value.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    public abstract void SetStudyUserAttr(int studyId, string key, object? value);

    /// <summary>Register an optuna-internal attribute to a study, overwriting any existing value.</summary>
    /// <exception cref="KeyError">If no study with the matching ID exists.</exception>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="key">Attribute name; a later write replaces its stored value.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    public abstract void SetStudySystemAttr(int studyId, string key, object? value);

    // ------------------------------------------------------------------ study access

    /// <summary>Read the ID of a study.</summary>
    /// <exception cref="KeyError">If no study with the matching name exists.</exception>
    /// <param name="studyName">Name identifying the study in the selected storage.</param>
    /// <returns>Storage identifier of the named study.</returns>
    public abstract int GetStudyIdFromName(string studyName);

    /// <summary>Read the study name of a study.</summary>
    /// <exception cref="KeyError">If no study with the matching ID exists.</exception>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <returns>Registered study name.</returns>
    public abstract string GetStudyNameFromId(int studyId);

    /// <summary>Read whether a study maximizes or minimizes an objective.</summary>
    /// <exception cref="KeyError">If no study with the matching ID exists.</exception>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <returns>Objective directions in objective-value order; callers must not mutate shared storage records.</returns>
    public abstract List<StudyDirection> GetStudyDirections(int studyId);

    /// <summary>Read the user-defined attributes of a study.</summary>
    /// <exception cref="KeyError">If no study with the matching ID exists.</exception>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <returns>User metadata snapshot; callers must not mutate shared storage records.</returns>
    public abstract Collections.OrderedDict<string, object?> GetStudyUserAttrs(int studyId);

    /// <summary>Read the optuna-internal attributes of a study.</summary>
    /// <exception cref="KeyError">If no study with the matching ID exists.</exception>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <returns>Internal metadata snapshot; callers must not mutate shared storage records.</returns>
    public abstract Collections.OrderedDict<string, object?> GetStudySystemAttrs(int studyId);

    /// <summary>Read a list of <see cref="FrozenStudy" /> objects, sorted by study ID.</summary>
    /// <returns>Study metadata records ordered by storage identifier.</returns>
    public abstract List<FrozenStudy> GetAllStudies();

    // ------------------------------------------------------------------ trial manipulation

    /// <summary>
    ///     Create and add a new trial to a study. The returned trial ID is unique among all
    ///     current and deleted trials.
    /// </summary>
    /// <param name="studyId">ID of the study.</param>
    /// <param name="templateTrial">
    ///     Template <see cref="FrozenTrial" /> with default user-attributes, system-attributes,
    ///     intermediate-values, and a state.
    /// </param>
    /// <exception cref="KeyError">If no study with the matching ID exists.</exception>
    /// <returns>Unique storage identifier of the created trial.</returns>
    public abstract int CreateNewTrial(int studyId, FrozenTrial? templateTrial = null);

    /// <summary>Set a parameter to a trial.</summary>
    /// <exception cref="KeyError">If no trial with the matching ID exists.</exception>
    /// <exception cref="UpdateFinishedTrialError">If the trial is already finished.</exception>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="paramName">Parameter name used as the storage and distribution lookup key.</param>
    /// <param name="distribution">Distribution defining the parameter representation and allowed values.</param>
    /// <param name="paramValueInternal">Parameter value in the distribution's internal numeric representation.</param>
    /// <exception cref="NumSharp.ValueError">The distribution is incompatible with an existing parameter distribution.</exception>
    public abstract void SetTrialParam(
        int trialId, string paramName, double paramValueInternal, BaseDistribution distribution);

    /// <summary>Read the trial ID of a trial.</summary>
    /// <exception cref="KeyError">If no trial with the matching study ID and trial number exists.</exception>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="trialNumber">Zero-based trial number within the study, not a storage identifier.</param>
    /// <returns>Storage identifier corresponding to the per-study trial number.</returns>
    /// <exception cref="System.ArgumentOutOfRangeException">The trial number is negative.</exception>
    public virtual int GetTrialIdFromStudyIdTrialNumber(int studyId, int trialNumber)
    {
        var trials = GetAllTrials(studyId, deepcopy: false);
        if (trials.Count <= trialNumber)
            throw new KeyError($"No trial with trial_number={trialNumber} exists in study with study_id={studyId}.");
        return trials[trialNumber].TrialId;
    }

    /// <summary>Read the trial number of a trial (unique within a study, and sequential).</summary>
    /// <exception cref="KeyError">If no trial with the matching ID exists.</exception>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <returns>Zero-based sequential trial number within its study.</returns>
    public virtual int GetTrialNumberFromId(int trialId) => GetTrial(trialId).Number;

    /// <summary>Read the internal representation of a trial parameter.</summary>
    /// <exception cref="KeyError">If no trial with the matching ID exists, or no such parameter exists.</exception>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="paramName">Parameter name used as the storage and distribution lookup key.</param>
    /// <returns>Parameter value converted into its distribution's internal representation.</returns>
    public virtual double GetTrialParam(int trialId, string paramName)
    {
        var trial = GetTrial(trialId);
        return trial.Distributions[paramName].ToInternalRepr(trial.Params[paramName]);
    }

    /// <summary>
    ///     Update the state and values of a trial. If <paramref name="values" /> is not
    ///     <c>null</c>, this method overwrites any existing trial values.
    /// </summary>
    /// <returns>
    ///     <c>true</c> if the state is successfully updated; <c>false</c> if the state is kept
    ///     the same (which happens when updating a RUNNING trial to RUNNING).
    /// </returns>
    /// <exception cref="KeyError">If no trial with the matching ID exists.</exception>
    /// <exception cref="UpdateFinishedTrialError">If the trial is already finished.</exception>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="state">Trial state to record or inspect.</param>
    /// <param name="values">Objective values in study direction order, or null when no objective values are recorded.</param>
    public abstract bool SetTrialStateValues(int trialId, TrialState state, IReadOnlyList<double>? values = null);

    /// <summary>
    ///     Report an intermediate value of an objective function, overwriting any existing value
    ///     associated with the step.
    /// </summary>
    /// <exception cref="KeyError">If no trial with the matching ID exists.</exception>
    /// <exception cref="UpdateFinishedTrialError">If the trial is already finished.</exception>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="step">
    ///     Reporting step whose intermediate objective observation is stored. 64-bit, like Python's unbounded
    ///     <c>int</c>; a backend whose schema column is narrower (Optuna's PostgreSQL/MySQL <c>INTEGER</c>) refuses
    ///     a step it cannot hold instead of truncating it.
    /// </param>
    /// <param name="intermediateValue">Intermediate objective observation to store at the reporting step.</param>
    public abstract void SetTrialIntermediateValue(int trialId, long step, double intermediateValue);

    /// <summary>Set a user-defined attribute to a trial, overwriting any existing value.</summary>
    /// <exception cref="KeyError">If no trial with the matching ID exists.</exception>
    /// <exception cref="UpdateFinishedTrialError">If the trial is already finished.</exception>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="key">Attribute name; a later write replaces its stored value.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    public abstract void SetTrialUserAttr(int trialId, string key, object? value);

    /// <summary>Set an optuna-internal attribute to a trial, overwriting any existing value.</summary>
    /// <exception cref="KeyError">If no trial with the matching ID exists.</exception>
    /// <exception cref="UpdateFinishedTrialError">If the trial is already finished.</exception>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="key">Attribute name; a later write replaces its stored value.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    public abstract void SetTrialSystemAttr(int trialId, string key, object? value);

    // ------------------------------------------------------------------ trial access

    /// <summary>Read a trial.</summary>
    /// <exception cref="KeyError">If no trial with the matching ID exists.</exception>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <returns>Stored trial record; treat it as read-only unless copied.</returns>
    public abstract FrozenTrial GetTrial(int trialId);

    /// <summary>
    ///     Read all trials in a study, sorted by trial ID.
    /// </summary>
    /// <param name="studyId">ID of the study.</param>
    /// <param name="deepcopy">
    ///     Whether to copy the list of trials before returning. Set to <c>true</c> if you intend
    ///     to update the list or elements of the list.
    /// </param>
    /// <param name="states">Trial states to filter on. If <c>null</c>, include all states.</param>
    /// <exception cref="KeyError">If no study with the matching ID exists.</exception>
    /// <returns>Trials in identifier order, optionally filtered by state and copied.</returns>
    public abstract List<FrozenTrial> GetAllTrials(
        int studyId, bool deepcopy = true, IReadOnlyCollection<TrialState>? states = null);

    /// <summary>
    ///     Read all trials of a study as a fixed-membership, zero-copy window when the backend can afford
    ///     one — the O(1) sibling of <c>GetAllTrials(deepcopy: false)</c> for samplers that consult history
    ///     on every ask (a per-ask list copy makes the optimization loop quadratic in trial count). The
    ///     window's membership is frozen at the call; each read returns the trial's current snapshot.
    /// </summary>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <returns>
    ///     Trials in identifier order; treat the collection and its records as read-only. The base
    ///     implementation falls back to <c>GetAllTrials(deepcopy: false)</c>, so backends without a live
    ///     in-memory list keep their existing (copying) behavior and only <see cref="InMemoryStorage" />-like
    ///     backends get the O(1) path.
    /// </returns>
    /// <exception cref="KeyError">If no study with the matching ID exists.</exception>
    public virtual IReadOnlyList<FrozenTrial> GetTrialsPrefixView(int studyId)
        => GetAllTrials(studyId, deepcopy: false);

    /// <summary>Count the number of trials in a study.</summary>
    /// <exception cref="KeyError">If no study with the matching ID exists.</exception>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="states">States to include; null includes every state.</param>
    /// <returns>Number of trials matching the requested state filter.</returns>
    public virtual int GetNTrials(int studyId, IReadOnlyCollection<TrialState>? states = null)
        => GetAllTrials(studyId, deepcopy: false, states: states).Count;

    /// <summary>
    ///     Return the trial with the best value in a study. Valid only during single-objective
    ///     optimization.
    /// </summary>
    /// <exception cref="KeyError">If no study with the matching ID exists.</exception>
    /// <exception cref="RuntimeError">If the study has more than one direction.</exception>
    /// <exception cref="ValueError">If no trials have been completed.</exception>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <returns>Best completed single-objective trial by objective value; feasibility is handled by Study.</returns>
    public virtual FrozenTrial GetBestTrial(int studyId)
    {
        var allTrials = GetAllTrials(studyId, deepcopy: false, states: new[] { TrialState.Complete });

        if (allTrials.Count == 0)
            throw new ValueError("No trials are completed yet.");

        var directions = GetStudyDirections(studyId);
        if (directions.Count > 1)
            throw new RuntimeError("Best trial can be obtained only for single-objective optimization.");
        var direction = directions[0];

        return direction == StudyDirection.Maximize
            ? allTrials.Aggregate((best, t) => t.Value! > best.Value! ? t : best)
            : allTrials.Aggregate((best, t) => t.Value! < best.Value! ? t : best);
    }

    /// <summary>Read the parameter dictionary of a trial (external representations).</summary>
    /// <exception cref="KeyError">If no trial with the matching ID exists.</exception>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <returns>Parameter names mapped to external values; treat shared data as read-only.</returns>
    public virtual IReadOnlyDictionary<string, object?> GetTrialParams(int trialId) => GetTrial(trialId).Params;

    /// <summary>Read the user-defined attributes of a trial.</summary>
    /// <exception cref="KeyError">If no trial with the matching ID exists.</exception>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <returns>User attributes of the stored trial.</returns>
    public virtual IReadOnlyDictionary<string, object?> GetTrialUserAttrs(int trialId) => GetTrial(trialId).UserAttrs;

    /// <summary>Read the optuna-internal attributes of a trial.</summary>
    /// <exception cref="KeyError">If no trial with the matching ID exists.</exception>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <returns>Internal attributes of the stored trial.</returns>
    public virtual IReadOnlyDictionary<string, object?> GetTrialSystemAttrs(int trialId) => GetTrial(trialId).SystemAttrs;

    /// <summary>Clean up all connections to a database.</summary>
    public virtual void RemoveSession()
    {
    }

    /// <summary>Check whether a trial state is updatable.</summary>
    /// <param name="trialId">ID of the trial (only used for an error message).</param>
    /// <param name="trialState">Trial state to check.</param>
    /// <exception cref="UpdateFinishedTrialError">If the trial is already finished.</exception>
    /// <exception cref="NumSharp.KeyError">A finished-state check attempts to load a nonexistent trial.</exception>
    public virtual void CheckTrialIsUpdatable(int trialId, TrialState trialState)
    {
        if (trialState.IsFinished())
        {
            var trial = GetTrial(trialId);
            throw new UpdateFinishedTrialError(
                $"Trial#{trial.Number} has already finished and can not be updated.");
        }
    }
}
