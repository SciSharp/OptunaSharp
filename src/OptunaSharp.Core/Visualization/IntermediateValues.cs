using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Logging;

namespace OptunaSharp.Visualization;

/// <summary>
///     One trial's intermediate-value curve. Port of <c>_TrialInfo</c> in
///     <c>optuna/visualization/_intermediate_values.py</c>.
/// </summary>
public sealed record TrialInfo(
    int TrialNumber,
    IReadOnlyList<(int Step, double Value)> SortedIntermediateValues,
    bool Feasible);

/// <summary>
///     The data backing an intermediate-values plot. Port of <c>_IntermediatePlotInfo</c>.
/// </summary>
public sealed record IntermediatePlotInfo(IReadOnlyList<TrialInfo> TrialInfos);

/// <summary>
///     Data layer for <c>optuna.visualization.plot_intermediate_values</c> — the pure
///     <c>_get_intermediate_plot_info</c> half of <c>_intermediate_values.py</c> (the plotly
///     rendering is a Tier X concern).
/// </summary>
public static class IntermediateValues
{
    private static readonly Logger Logger =
        OptunaLogging.GetLogger("optuna.visualization._intermediate_values");

    /// <summary>Python <c>_get_intermediate_plot_info(study)</c>.</summary>
    public static IntermediatePlotInfo GetPlotInfo(Study study)
    {
        var trials = study.GetTrials(
            deepcopy: false,
            states: new[] { TrialState.Pruned, TrialState.Complete, TrialState.Running });

        static bool SatisfiesConstraints(FrozenTrial trial)
            => trial.Constraints.Values.All(x => x <= 0.0);

        var trialInfos = trials
            .Where(trial => trial.IntermediateValues.Count > 0)
            .Select(trial => new TrialInfo(
                trial.Number,
                trial.IntermediateValues
                    .OrderBy(kv => kv.Key)
                    .Select(kv => (kv.Key, kv.Value))
                    .ToList(),
                SatisfiesConstraints(trial)))
            .ToList();

        if (trials.Count == 0)
            Logger.Warning("Study instance does not contain trials.");
        else if (trialInfos.Count == 0)
            Logger.Warning(
                "You need to set up the pruning feature to utilize `plot_intermediate_values()`");

        return new IntermediatePlotInfo(trialInfos);
    }
}
