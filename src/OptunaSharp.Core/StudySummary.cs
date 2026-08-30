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
public class StudySummary
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
}
