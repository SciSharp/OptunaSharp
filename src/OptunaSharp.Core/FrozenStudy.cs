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
public class FrozenStudy
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
}
