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
    /// <summary>Storage identifier of this study, also used as its ordering key.</summary>
    internal int StudyId;

    /// <param name="studyName">Name identifying the study in the selected storage.</param>
    /// <param name="direction">Single objective direction; ignored when a directions sequence is supplied.</param>
    /// <param name="userAttrs">User metadata; values should be JSON serializable for persistent storage.</param>
    /// <param name="systemAttrs">Internal metadata used by samplers and storage; preserve reserved keys.</param>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <param name="directions">Objective directions in the same order as the objective values.</param>
    /// <summary>Builds a study record from persisted metadata and objective directions.</summary>
    /// <exception cref="NumSharp.ValueError">Neither direction nor directions is supplied.</exception>
    /// <remarks>Directions are materialized; supplied attribute dictionaries and any best-trial object are retained by reference.</remarks>
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
    ///     <see cref="Study.SetUserAttr(string, object?)" />.
    /// </summary>
    public OrderedDict<string, object?> UserAttrs { get; internal set; }

    /// <summary>Dictionary that contains the attributes internally set by Optuna.</summary>
    public OrderedDict<string, object?> SystemAttrs { get; internal set; }

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

    /// <summary>A list of <see cref="StudyDirection" /> objects.</summary>
    public List<StudyDirection> Directions => _directions;

    /// <summary>Copies study directions and supported nested attribute containers.</summary>
    /// <returns>A detached study record with the same storage identifier.</returns>
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
    /// <param name="other">Value to compare with this instance; null compares unequal.</param>
    /// <returns>True when the other value has the same type and equal recorded fields.</returns>
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

    /// <summary>Compares the fields participating in value equality.</summary>
    /// <param name="obj">Object to compare with this instance.</param>
    /// <returns>True when the other value has the same type and equal recorded fields.</returns>
    public override bool Equals(object? obj) => obj is FrozenStudy other && Equals(other);

    /// <summary>
    ///     A hash consistent with <see cref="Equals(FrozenStudy)" />. Optuna's <c>FrozenStudy</c> is
    ///     unhashable (<c>__eq__</c> without <c>__hash__</c>); as elsewhere in the port we expose a
    ///     usable hash instead.
    /// </summary>
    /// <returns>Hash code consistent with this type's equality contract.</returns>
    public override int GetHashCode() => HashCode.Combine(StudyId, StudyName);

    /// <summary>Python <c>__lt__</c> / <c>__le__</c>: <see cref="FrozenStudy" /> is ordered by study id.</summary>
    /// <param name="other">Study record to compare by identifier; null sorts before this instance.</param>
    /// <returns>Negative, zero, or positive according to study identifier order; null sorts first.</returns>
    public int CompareTo(FrozenStudy? other) => other is null ? 1 : StudyId.CompareTo(other.StudyId);

    /// <summary>Applies the <c>==</c> comparison using the type's value equality.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator ==(FrozenStudy? left, FrozenStudy? right)
        => left is null ? right is null : left.Equals(right);

    /// <summary>Applies the <c>!=</c> comparison using the type's value equality.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator !=(FrozenStudy? left, FrozenStudy? right) => !(left == right);

    // Ordering mirrors Python's __lt__/__le__ (by study id); >/>= are the reflected forms.
    /// <summary>Applies the <c>&lt;</c> comparison by study identifier.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator <(FrozenStudy left, FrozenStudy right) => left.StudyId < right.StudyId;

    /// <summary>Applies the <c>&gt;</c> comparison by study identifier.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator >(FrozenStudy left, FrozenStudy right) => left.StudyId > right.StudyId;

    /// <summary>Applies the <c>&lt;=</c> comparison by study identifier.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator <=(FrozenStudy left, FrozenStudy right) => left.StudyId <= right.StudyId;

    /// <summary>Applies the <c>&gt;=</c> comparison by study identifier.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator >=(FrozenStudy left, FrozenStudy right) => left.StudyId >= right.StudyId;
}
