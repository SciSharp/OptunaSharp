using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Visualization;

/// <summary>
///     One study's empirical-distribution-function curve (y for each shared x). Port of
///     <c>_EDFLineInfo</c> in <c>optuna/visualization/_edf.py</c>.
/// </summary>
public sealed record EdfLineInfo(string StudyName, double[] YValues);

/// <summary>
///     The data backing an EDF plot: one line per study over a shared x grid. Port of <c>_EDFInfo</c>.
/// </summary>
public sealed record EdfInfo(IReadOnlyList<EdfLineInfo> Lines, double[] XValues);

/// <summary>
///     Data layer for <c>optuna.visualization.plot_edf</c> — the pure <c>_get_edf_info</c> half of
///     <c>_edf.py</c>. Only complete trials are considered.
/// </summary>
public static class Edf
{
    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.visualization._edf");

    private const int NumSamplesXAxis = 100;

    /// <inheritdoc cref="GetInfo(System.Collections.Generic.IReadOnlyList{Study}, System.Func{FrozenTrial, double}, string)" />
    public static EdfInfo GetInfo(
        Study study,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => GetInfo(new[] { study }, target, targetName);

    /// <summary>Python <c>_get_edf_info(study, target, target_name)</c>.</summary>
    public static EdfInfo GetInfo(
        IReadOnlyList<Study> studies,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
    {
        VisualizationUtils.CheckPlotArgs(studies, target, targetName);

        if (studies.Count == 0)
        {
            Logger.Warning("There are no studies.");
            return new EdfInfo(new List<EdfLineInfo>(), Array.Empty<double>());
        }

        target ??= t => t.Value ?? double.NaN;

        var studyNames = new List<string>();
        var allValues = new List<double[]>();
        foreach (var study in studies)
        {
            var trials = VisualizationUtils.FilterNonfinite(
                study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete }), target: target);
            allValues.Add(trials.Select(target).ToArray());
            studyNames.Add(study.StudyName);
        }

        if (allValues.All(values => values.Length == 0))
        {
            Logger.Warning("There are no complete trials.");
            return new EdfInfo(new List<EdfLineInfo>(), Array.Empty<double>());
        }

        var concatenated = allValues.SelectMany(v => v).ToArray();
        double minXValue = concatenated.Min();
        double maxXValue = concatenated.Max();
        double[] xValues = MathUtil.Linspace(minXValue, maxXValue, NumSamplesXAxis);

        var edfLineInfoList = new List<EdfLineInfo>();
        foreach (var (studyName, values) in studyNames.Zip(allValues, (n, v) => (n, v)))
        {
            // y_values = np.sum(values[:, None] <= x_values, axis=0) / values.size
            var yValues = new double[xValues.Length];
            for (int j = 0; j < xValues.Length; j++)
            {
                int count = 0;
                foreach (double v in values)
                    if (v <= xValues[j])
                        count++;
                yValues[j] = (double)count / values.Length;  // 0/0 → NaN, matching NumPy on an empty study.
            }
            edfLineInfoList.Add(new EdfLineInfo(studyName, yValues));
        }

        return new EdfInfo(edfLineInfoList, xValues);
    }
}
