namespace OptunaSharp;

/// <summary>
///     Direction of a <see cref="Study" />. Port of <c>optuna/study/_study_direction.py</c>;
///     the numeric values are Optuna's and are persisted by storages.
/// </summary>
public enum StudyDirection
{
    /// <summary>Direction has not been set.</summary>
    NotSet = 0,

    /// <summary>The <see cref="Study" /> minimizes the objective function.</summary>
    Minimize = 1,

    /// <summary>The <see cref="Study" /> maximizes the objective function.</summary>
    Maximize = 2,
}

public static class StudyDirectionExtensions
{
    /// <summary>Python <c>direction.name</c> (e.g. <c>"MINIMIZE"</c>).</summary>
    internal static string PythonName(this StudyDirection direction) => direction switch
    {
        StudyDirection.Minimize => "MINIMIZE",
        StudyDirection.Maximize => "MAXIMIZE",
        _ => "NOT_SET",
    };
}
