using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;

namespace OptunaSharp;

/// <summary>
///     Basic attributes and aggregated results of a <see cref="Study" />.
///     Port of <c>optuna/study/_study_summary.py: StudySummary</c>.
///
///     See also <see cref="Optuna.GetAllStudySummaries" />.
/// </summary>
public class StudySummary : IEquatable<StudySummary>, IComparable<StudySummary>
{
    private readonly List<StudyDirection> _directions;
    /// <summary>Storage identifier of this study, also used as its ordering key.</summary>
    internal int StudyId;

    /// <param name="studyName">Name identifying the study in the selected storage.</param>
    /// <param name="direction">Single objective direction; ignored when a directions sequence is supplied.</param>
    /// <param name="bestTrial">Best single-objective trial, or null when unavailable or deliberately omitted.</param>
    /// <param name="userAttrs">User metadata; values should be JSON serializable for persistent storage.</param>
    /// <param name="nTrials">Number of trials represented by this summary.</param>
    /// <param name="datetimeStart">Local start time, or null when the trial or study has not started.</param>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="directions">Objective directions in the same order as the objective values.</param>
    /// <summary>Builds a study record from persisted metadata and objective directions.</summary>
    /// <exception cref="NumSharp.ValueError">Neither direction nor directions is supplied.</exception>
    /// <remarks>Directions are materialized; supplied attribute dictionaries and any best-trial object are retained by reference.</remarks>
    public StudySummary(
        string studyName,
        StudyDirection? direction,
        FrozenTrial? bestTrial,
        OrderedDict<string, object?> userAttrs,
        int nTrials,
        DateTime? datetimeStart,
        int studyId,
        IEnumerable<StudyDirection>? directions = null)
    {
        StudyName = studyName;
        if (direction is null && directions is null)
            throw new ValueError("Specify one of `direction` and `directions`.");
        if (directions is not null)
            _directions = directions.ToList();
        else
            _directions = new List<StudyDirection> { direction!.Value };
        BestTrial = bestTrial;
        UserAttrs = userAttrs;
        NTrials = nTrials;
        DatetimeStart = datetimeStart;
        StudyId = studyId;
    }

    /// <summary>Name of the <see cref="Study" />.</summary>
    public string StudyName { get; }

    /// <summary><see cref="FrozenTrial" /> with best objective value in the <see cref="Study" />.</summary>
    public FrozenTrial? BestTrial { get; }

    /// <summary>Dictionary that contains the attributes of the study set with <see cref="Study.SetUserAttr(string, object?)" />.</summary>
    public OrderedDict<string, object?> UserAttrs { get; }

    /// <summary>The number of trials ran in the <see cref="Study" />.</summary>
    public int NTrials { get; }

    /// <summary>Datetime where the <see cref="Study" /> started.</summary>
    public DateTime? DatetimeStart { get; }

    /// <summary>
    ///     <see cref="StudyDirection" /> of the study. Only available during single-objective
    ///     optimization.
    /// </summary>
    /// <exception cref="NumSharp.RuntimeError">There is more than one objective direction.</exception>
    /// <exception cref="System.ArgumentOutOfRangeException">The directions collection is empty.</exception>
    public StudyDirection Direction
    {
        get
        {
            if (_directions.Count > 1)
                throw new RuntimeError("This attribute is not available during multi-objective optimization.");
            return _directions[0];
        }
    }

    /// <summary>A sequence of <see cref="StudyDirection" /> objects.</summary>
    public IReadOnlyList<StudyDirection> Directions => _directions;

    // ------------------------------------------------------------------ equality / ordering

    /// <summary>
    ///     Python <c>__eq__</c>: value equality over every field (Optuna compares
    ///     <c>other.__dict__ == self.__dict__</c>) — study name, directions, best trial, user
    ///     attributes, trial count, start datetime and study id.
    /// </summary>
    /// <param name="other">Value to compare with this instance; null compares unequal.</param>
    /// <returns>True when the other value has the same type and equal recorded fields.</returns>
    public bool Equals(StudySummary? other)
    {
        if (other is null)
            return false;
        if (ReferenceEquals(this, other))
            return true;
        return StudyName == other.StudyName
            && _directions.SequenceEqual(other._directions)
            && (BestTrial is null ? other.BestTrial is null : BestTrial.Equals(other.BestTrial))
            && Internal.PyJson.ValueEquals(UserAttrs, other.UserAttrs)
            && NTrials == other.NTrials
            && Nullable.Equals(DatetimeStart, other.DatetimeStart)
            && StudyId == other.StudyId;
    }

    /// <summary>Compares the fields participating in value equality.</summary>
    /// <param name="obj">Object to compare with this instance.</param>
    /// <returns>True when the other value has the same type and equal recorded fields.</returns>
    public override bool Equals(object? obj) => obj is StudySummary other && Equals(other);

    /// <summary>
    ///     A hash consistent with <see cref="Equals(StudySummary)" />. Optuna's <c>StudySummary</c>
    ///     is actually unhashable (it defines <c>__eq__</c> without <c>__hash__</c>), but — as with
    ///     the <see cref="FrozenTrial" /> port — we expose a usable hash rather than replicate that.
    /// </summary>
    /// <returns>Hash code consistent with this type's equality contract.</returns>
    public override int GetHashCode() => HashCode.Combine(StudyId, StudyName, NTrials);

    /// <summary>Python <c>__lt__</c> / <c>__le__</c>: <see cref="StudySummary" /> is ordered by study id.</summary>
    /// <param name="other">Study record to compare by identifier; null sorts before this instance.</param>
    /// <returns>Negative, zero, or positive according to study identifier order; null sorts first.</returns>
    public int CompareTo(StudySummary? other) => other is null ? 1 : StudyId.CompareTo(other.StudyId);

    /// <summary>Applies the <c>==</c> comparison using the type's value equality.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator ==(StudySummary? left, StudySummary? right)
        => left is null ? right is null : left.Equals(right);

    /// <summary>Applies the <c>!=</c> comparison using the type's value equality.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator !=(StudySummary? left, StudySummary? right) => !(left == right);

    // Ordering mirrors Python's __lt__/__le__ (by study id); >/>= are the reflected forms.
    /// <summary>Applies the <c>&lt;</c> comparison by study identifier.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator <(StudySummary left, StudySummary right) => left.StudyId < right.StudyId;

    /// <summary>Applies the <c>&gt;</c> comparison by study identifier.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator >(StudySummary left, StudySummary right) => left.StudyId > right.StudyId;

    /// <summary>Applies the <c>&lt;=</c> comparison by study identifier.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator <=(StudySummary left, StudySummary right) => left.StudyId <= right.StudyId;

    /// <summary>Applies the <c>&gt;=</c> comparison by study identifier.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator >=(StudySummary left, StudySummary right) => left.StudyId >= right.StudyId;
}
