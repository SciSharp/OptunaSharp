using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Logging;
using OptunaSharp.Pruners;

namespace OptunaSharp.Internal;

/// <summary>Port of <c>optuna/study/_tell.py</c>.</summary>
internal static class TellHelpers
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.study._tell");

    /// <summary>Python <c>_get_frozen_trial(study, trial)</c>.</summary>
    internal static FrozenTrial GetFrozenTrial(Study study, Trial trial)
        => study.Storage.GetTrial(trial.TrialId);

    internal static FrozenTrial GetFrozenTrial(Study study, int trialNumber)
    {
        int trialId;
        try
        {
            trialId = study.Storage.GetTrialIdFromStudyIdTrialNumber(study.StudyId, trialNumber);
        }
        catch (KeyError e)
        {
            throw new ValueError(
                $"Cannot tell for trial with number {trialNumber} since it has not been created.", e);
        }

        return study.Storage.GetTrial(trialId);
    }

    /// <summary>Python <c>_check_state_and_values</c>.</summary>
    internal static void CheckStateAndValues(TrialState? state, IReadOnlyList<double>? values)
    {
        if (state == TrialState.Complete)
        {
            if (values is null)
                throw new ValueError(
                    "No values were told. Values are required when state is TrialState.COMPLETE.");
        }
        else if (state is TrialState.Pruned or TrialState.Fail)
        {
            if (values is not null)
                throw new ValueError(
                    "Values were told. Values cannot be specified when state is " +
                    "TrialState.PRUNED or TrialState.FAIL.");
        }
        else if (state is not null)
        {
            throw new ValueError($"Cannot tell with state {state.Value.PythonStr()}.");
        }
    }

    /// <summary>Python <c>_check_values_are_feasible</c>. Returns an error message or <c>null</c>.</summary>
    internal static string? CheckValuesAreFeasible(Study study, IReadOnlyList<double> values)
    {
        var errors = new List<string>();
        foreach (double v in values)
        {
            if (double.IsNaN(v))
                errors.Add($"The value {PythonRepr.FloatRepr(v)} is not acceptable");
        }
        if (errors.Count > 0)
            return string.Join("; ", errors);
        if (study.Directions.Count != values.Count)
            return
                $"The number of the values {values.Count} did not match the number of the objectives " +
                $"{study.Directions.Count}";
        return null;
    }

    /// <summary>
    ///     Python <c>_tell_with_warning</c> — the internal method of <see cref="Study.Tell(Trial, IReadOnlyList{double}, TrialState?, bool)" />.
    /// </summary>
    /// <returns>(state, values, valuesConversionFailureMessage)</returns>
    internal static (TrialState State, List<double>? Values, string? WarningMessage) TellWithWarning(
        Study study,
        FrozenTrial frozenTrial,
        IReadOnlyList<double>? valueOrValues = null,
        TrialState? state = null,
        bool skipIfFinished = false,
        bool suppressWarning = false)
    {
        // We must invalidate all trials cache here as it is only valid within a trial.
        study.ThreadLocal.CachedAllTrials = null;

        // Validate the trial argument.
        if (frozenTrial.State.IsFinished() && skipIfFinished)
        {
            _logger.Info(
                $"Skipped telling trial {frozenTrial.Number} with values " +
                $"{FormatValues(valueOrValues)} and state {state?.PythonStr() ?? "None"} since trial was already finished. " +
                $"Finished trial has values {FormatValues(frozenTrial.Values)} and state {frozenTrial.State.PythonStr()}.");
            return (frozenTrial.State, frozenTrial.Values, null);
        }
        if (frozenTrial.State != TrialState.Running)
            throw new ValueError($"Cannot tell a {frozenTrial.State.PythonName()} trial.");

        // Validate the state and values arguments.
        IReadOnlyList<double>? values = valueOrValues;

        CheckStateAndValues(state, values);

        string? valuesConversionFailureMessage = null;

        List<double>? finalValues = values?.ToList();

        if (state == TrialState.Complete)
        {
            System.Diagnostics.Debug.Assert(values is not null);

            valuesConversionFailureMessage = CheckValuesAreFeasible(study, values!);
            if (valuesConversionFailureMessage is not null)
                throw new ValueError(valuesConversionFailureMessage);
        }
        else if (state == TrialState.Pruned)
        {
            // Register the last intermediate value if present as the value of the trial.
            // TODO(hvy): Whether a pruned trials should have an actual value can be discussed.
            System.Diagnostics.Debug.Assert(values is null);

            int? lastStep = frozenTrial.LastStep;
            if (lastStep is not null)
            {
                double lastIntermediateValue = frozenTrial.IntermediateValues[lastStep.Value];
                // intermediate_values can be unacceptable value, i.e., NaN.
                if (CheckValuesAreFeasible(study, new[] { lastIntermediateValue }) is null)
                    finalValues = new List<double> { lastIntermediateValue };
            }
        }
        else if (state is null)
        {
            if (values is null)
            {
                valuesConversionFailureMessage = "The value None could not be cast to float.";
            }
            else
            {
                valuesConversionFailureMessage = CheckValuesAreFeasible(study, values);
            }

            if (valuesConversionFailureMessage is null)
            {
                state = TrialState.Complete;
            }
            else
            {
                state = TrialState.Fail;
                finalValues = null;
                if (!suppressWarning)
                {
                    OptunaWarnings.Warn(valuesConversionFailureMessage);
                    valuesConversionFailureMessage = null;
                }
            }
        }

        System.Diagnostics.Debug.Assert(state is not null);

        // Post-processing and storing the trial.
        try
        {
            // Sampler defined trial post-processing.
            var filteredStudy = PrunerHelpers.FilterStudy(study, frozenTrial);
            study.Sampler.AfterTrial(filteredStudy, frozenTrial, state!.Value, finalValues);
        }
        finally
        {
            study.Storage.SetTrialStateValues(frozenTrial.TrialId, state!.Value, finalValues);
        }

        return (state.Value, finalValues, valuesConversionFailureMessage);
    }

    private static string FormatValues(IReadOnlyList<double>? values)
        => values is null ? "None" : PythonRepr.ListRepr(values);
}
