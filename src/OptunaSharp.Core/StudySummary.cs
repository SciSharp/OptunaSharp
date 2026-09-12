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
    internal int StudyId;

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

    /// <summary>Dictionary that contains the attributes of the study set with <see cref="Study.SetUserAttr" />.</summary>
    public OrderedDict<string, object?> UserAttrs { get; }

    /// <summary>The number of trials ran in the <see cref="Study" />.</summary>
    public int NTrials { get; }

    /// <summary>Datetime where the <see cref="Study" /> started.</summary>
    public DateTime? DatetimeStart { get; }

    /// <summary>
    ///     <see cref="StudyDirection" /> of the study. Only available during single-objective
    ///     optimization.
    /// </summary>
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

    public override bool Equals(object? obj) => obj is StudySummary other && Equals(other);

    /// <summary>
    ///     A hash consistent with <see cref="Equals(StudySummary)" />. Optuna's <c>StudySummary</c>
    ///     is actually unhashable (it defines <c>__eq__</c> without <c>__hash__</c>), but — as with
    ///     the <see cref="FrozenTrial" /> port — we expose a usable hash rather than replicate that.
    /// </summary>
    public override int GetHashCode() => HashCode.Combine(StudyId, StudyName, NTrials);

    /// <summary>Python <c>__lt__</c> / <c>__le__</c>: <see cref="StudySummary" /> is ordered by study id.</summary>
    public int CompareTo(StudySummary? other) => other is null ? 1 : StudyId.CompareTo(other.StudyId);

    public static bool operator ==(StudySummary? left, StudySummary? right)
        => left is null ? right is null : left.Equals(right);

    public static bool operator !=(StudySummary? left, StudySummary? right) => !(left == right);

    // Ordering mirrors Python's __lt__/__le__ (by study id); >/>= are the reflected forms.
    public static bool operator <(StudySummary left, StudySummary right) => left.StudyId < right.StudyId;

    public static bool operator >(StudySummary left, StudySummary right) => left.StudyId > right.StudyId;

    public static bool operator <=(StudySummary left, StudySummary right) => left.StudyId <= right.StudyId;

    public static bool operator >=(StudySummary left, StudySummary right) => left.StudyId >= right.StudyId;
}
