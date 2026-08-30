using System;
using System.Linq;
using NumSharp;

namespace OptunaSharp.Pruners;

/// <summary>
///     Pruner which wraps another pruner with tolerance.
///     Port of <c>optuna/pruners/_patient.py: PatientPruner</c> (experimental since 2.8.0).
///
///     This pruner monitors intermediate values in a trial and prunes the trial if the
///     improvement in the intermediate values after a patience period is less than a threshold.
///
///     The pruner handles NaN values in the following manner:
///     1. If all intermediate values before or during the patient period are NaN, the trial will
///        not be pruned.
///     2. During the pruning calculations, NaN values are ignored. Only valid numeric values are
///        considered.
/// </summary>
public class PatientPruner : BasePruner
{
    private readonly BasePruner? _wrappedPruner;
    private readonly int _patience;
    private readonly double _minDelta;

    /// <param name="wrappedPruner">
    ///     Wrapped pruner to perform pruning when <see cref="PatientPruner" /> allows a trial to
    ///     be pruned. If it is <c>null</c>, this pruner is equivalent to early-stopping taken
    ///     the intermediate values in the individual trial.
    /// </param>
    /// <param name="patience">
    ///     Pruning is disabled until the objective doesn't improve for <c>patience</c>
    ///     consecutive steps.
    /// </param>
    /// <param name="minDelta">
    ///     Tolerance value to check whether or not the objective improves. Must be non-negative.
    /// </param>
    public PatientPruner(BasePruner? wrappedPruner, int patience, double minDelta = 0.0)
    {
        if (patience < 0)
            throw new ValueError($"patience cannot be negative but got {patience}.");

        if (minDelta < 0)
            throw new ValueError($"min_delta cannot be negative but got {Internal.PythonRepr.FloatRepr(minDelta)}.");

        _wrappedPruner = wrappedPruner;
        _patience = patience;
        _minDelta = minDelta;
    }

    public override bool Prune(Study study, FrozenTrial trial)
    {
        int? stepNullable = trial.LastStep;
        if (stepNullable is null)
            return false;

        var intermediateValues = trial.IntermediateValues;
        var steps = intermediateValues.Keys.ToArray();

        // Do not prune if number of step to determine are insufficient.
        if (steps.Length <= _patience + 1)
            return false;

        Array.Sort(steps);
        // This is the score patience steps ago:
        var stepsBeforePatience = steps.Take(steps.Length - _patience - 1);
        var scoresBeforePatience = stepsBeforePatience.Select(s => intermediateValues[s]).ToArray();
        // And these are the scores after that:
        var stepsAfterPatience = steps.Skip(steps.Length - _patience - 1);
        var scoresAfterPatience = stepsAfterPatience.Select(s => intermediateValues[s]).ToArray();

        var direction = study.Direction;
        bool maybePrune;
        if (direction == StudyDirection.Minimize)
            maybePrune = NanMin(scoresBeforePatience) + _minDelta < NanMin(scoresAfterPatience);
        else
            maybePrune = NanMax(scoresBeforePatience) - _minDelta > NanMax(scoresAfterPatience);

        if (maybePrune)
        {
            if (_wrappedPruner is not null)
                return _wrappedPruner.Prune(study, trial);
            return true;
        }
        return false;
    }

    private static double NanMin(double[] values)
    {
        double result = double.NaN;
        foreach (double v in values)
        {
            if (double.IsNaN(v))
                continue;
            result = double.IsNaN(result) ? v : Math.Min(result, v);
        }
        return result;
    }

    private static double NanMax(double[] values)
    {
        double result = double.NaN;
        foreach (double v in values)
        {
            if (double.IsNaN(v))
                continue;
            result = double.IsNaN(result) ? v : Math.Max(result, v);
        }
        return result;
    }
}
