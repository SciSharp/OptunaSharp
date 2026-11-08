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
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="trial">The trial whose recorded state and values are used.</param>
    /// <returns>The snapshot read from study storage using the trial identifier or study-local trial number.</returns>
    /// <exception cref="NumSharp.KeyError">The trial identifier has no corresponding record in the study storage.</exception>
    internal static FrozenTrial GetFrozenTrial(Study study, Trial trial)
        => study.Storage.GetTrial(trial.TrialId);

    /// <summary>Returns the snapshot read from study storage using the trial identifier or study-local trial number.</summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="trialNumber">The study-local trial number, rather than the storage's internal trial identifier.</param>
    /// <returns>The snapshot read from study storage using the trial identifier or study-local trial number.</returns>
    /// <exception cref="NumSharp.KeyError">The trial identifier has no corresponding record in the study storage.</exception>
    /// <exception cref="NumSharp.ValueError">The study-local trial number has not been created.</exception>
    internal static FrozenTrial GetFrozenTrial(Study study, int trialNumber)
        => study.Storage.GetTrial(GetTrialId(study, trialNumber));

    /// <summary>
    ///     The first half of Python's <c>_get_frozen_trial(study, trial_number)</c>: resolves a study-local trial number
    ///     to its storage identifier. A tell by number needs the identifier before reading the trial, to take the
    ///     trial's tell monitor (<see cref="Storages.BaseStorage.TellLockFor" />) first.
    /// </summary>
    /// <param name="study">The study whose storage resolves the number.</param>
    /// <param name="trialNumber">The study-local trial number, rather than the storage's internal trial identifier.</param>
    /// <returns>The storage identifier of the trial with that number.</returns>
    /// <exception cref="NumSharp.ValueError">The study-local trial number has not been created.</exception>
    internal static int GetTrialId(Study study, int trialNumber)
    {
        try
        {
            return study.Storage.GetTrialIdFromStudyIdTrialNumber(study.StudyId, trialNumber);
        }
        catch (KeyError e)
        {
            throw new ValueError(
                $"Cannot tell for trial with number {trialNumber} since it has not been created.", e);
        }
    }

    /// <summary>Python <c>_check_state_and_values</c>.</summary>
    /// <param name="state">The requested or recorded trial completion state.</param>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <exception cref="NumSharp.ValueError">The requested completion state and presence of objective values are inconsistent.</exception>
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
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <returns>Null when the objective values are valid, otherwise a diagnostic explaining the mismatch.</returns>
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
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="frozenTrial">The trial snapshot to validate and complete in storage.</param>
    /// <param name="valueOrValues">The values returned by the objective before completion-state validation.</param>
    /// <param name="state">The requested or recorded trial completion state.</param>
    /// <param name="skipIfFinished">Whether an already finished trial is returned unchanged instead of rejected.</param>
    /// <param name="suppressWarning">Whether value-validation warnings are returned without being emitted.</param>
    /// <exception cref="NumSharp.ValueError">The trial is not running and cannot be skipped, the requested state/value combination is invalid, or explicit complete-state values fail validation.</exception>
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

            long? lastStep = frozenTrial.LastStep;
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
            // Sampler defined trial post-processing, serialized with the sampler's other calls from concurrent
            // trials (see BaseSampler.SyncRoot). A worker of a SamplerConcurrency.PerWorker call tells its trial on its
            // own thread, so its own instance, the one that sampled the trial, gets AfterTrial.
            var filteredStudy = PrunerHelpers.FilterStudy(study, frozenTrial);
            var sampler = study.SamplerForCurrentThread;
            sampler.Locked(() => sampler.AfterTrial(filteredStudy, frozenTrial, state!.Value, finalValues));
        }
        finally
        {
            study.Storage.SetTrialStateValues(frozenTrial.TrialId, state!.Value, finalValues);
        }

        return (state.Value, finalValues, valuesConversionFailureMessage);
    }

    /// <summary>Returns the Python-style representation of the supplied objective-value sequence.</summary>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <returns>The Python-style representation of the supplied objective-value sequence.</returns>
    private static string FormatValues(IReadOnlyList<double>? values)
        => values is null ? "None" : PythonRepr.ListRepr(values);
}
