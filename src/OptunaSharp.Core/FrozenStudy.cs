using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;

namespace OptunaSharp;

/// <summary>
///     Basic attributes of a <see cref="Study" />. Port of <c>optuna/study/_frozen.py: FrozenStudy</c>.
///
///     This class is private in Optuna and not referenced by Optuna users.
/// </summary>
public class FrozenStudy : IEquatable<FrozenStudy>, IComparable<FrozenStudy>
{
    private readonly List<StudyDirection> _directions;
    internal int StudyId;

    public FrozenStudy(
        string studyName,
        StudyDirection? direction,
        OrderedDict<string, object?> userAttrs,
        OrderedDict<string, object?> systemAttrs,
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
        UserAttrs = userAttrs;
        SystemAttrs = systemAttrs;
        StudyId = studyId;
    }

    /// <summary>Name of the <see cref="Study" />.</summary>
    public string StudyName { get; }

    /// <summary>
    ///     Dictionary that contains the attributes of the <see cref="Study" /> set with
    ///     <see cref="Study.SetUserAttr" />.
    /// </summary>
    public OrderedDict<string, object?> UserAttrs { get; internal set; }

    /// <summary>Dictionary that contains the attributes internally set by Optuna.</summary>
    public OrderedDict<string, object?> SystemAttrs { get; internal set; }

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

    /// <summary>A list of <see cref="StudyDirection" /> objects.</summary>
    public List<StudyDirection> Directions => _directions;

    internal FrozenStudy DeepCopy() => new(
        studyName: StudyName,
        direction: null,
        userAttrs: new OrderedDict<string, object?>(
            UserAttrs.Select(kv => new KeyValuePair<string, object?>(kv.Key, Internal.PyJson.DeepCopy(kv.Value)))),
        systemAttrs: new OrderedDict<string, object?>(
            SystemAttrs.Select(kv => new KeyValuePair<string, object?>(kv.Key, Internal.PyJson.DeepCopy(kv.Value)))),
        studyId: StudyId,
        directions: _directions.ToList());

    // ------------------------------------------------------------------ equality / ordering

    /// <summary>
    ///     Python <c>__eq__</c>: value equality over every field (<c>other.__dict__ == self.__dict__</c>)
    ///     — study name, directions, user attributes, system attributes and study id.
    /// </summary>
    public bool Equals(FrozenStudy? other)
    {
        if (other is null)
            return false;
        if (ReferenceEquals(this, other))
            return true;
        return StudyName == other.StudyName
            && _directions.SequenceEqual(other._directions)
            && Internal.PyJson.ValueEquals(UserAttrs, other.UserAttrs)
            && Internal.PyJson.ValueEquals(SystemAttrs, other.SystemAttrs)
            && StudyId == other.StudyId;
    }

    public override bool Equals(object? obj) => obj is FrozenStudy other && Equals(other);

    /// <summary>
    ///     A hash consistent with <see cref="Equals(FrozenStudy)" />. Optuna's <c>FrozenStudy</c> is
    ///     unhashable (<c>__eq__</c> without <c>__hash__</c>); as elsewhere in the port we expose a
    ///     usable hash instead.
    /// </summary>
    public override int GetHashCode() => HashCode.Combine(StudyId, StudyName);

    /// <summary>Python <c>__lt__</c> / <c>__le__</c>: <see cref="FrozenStudy" /> is ordered by study id.</summary>
    public int CompareTo(FrozenStudy? other) => other is null ? 1 : StudyId.CompareTo(other.StudyId);

    public static bool operator ==(FrozenStudy? left, FrozenStudy? right)
        => left is null ? right is null : left.Equals(right);

    public static bool operator !=(FrozenStudy? left, FrozenStudy? right) => !(left == right);

    // Ordering mirrors Python's __lt__/__le__ (by study id); >/>= are the reflected forms.
    public static bool operator <(FrozenStudy left, FrozenStudy right) => left.StudyId < right.StudyId;

    public static bool operator >(FrozenStudy left, FrozenStudy right) => left.StudyId > right.StudyId;

    public static bool operator <=(FrozenStudy left, FrozenStudy right) => left.StudyId <= right.StudyId;

    public static bool operator >=(FrozenStudy left, FrozenStudy right) => left.StudyId >= right.StudyId;
}
