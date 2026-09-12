using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Terminator;

/// <summary>
///     Base class for error evaluators. Port of
///     <c>optuna/terminator/erroreval.py: BaseErrorEvaluator</c>.
/// </summary>
public abstract class BaseErrorEvaluator
{
    /// <summary>Evaluate the statistical error of the objective function.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="studyDirection">Whether larger or smaller objective values are preferred.</param>
    /// <returns>The estimated statistical error in objective-value units.</returns>
    /// <remarks>Concrete evaluators define their data prerequisites and may propagate failures from user-supplied evaluators or trial storage.</remarks>
    public abstract double Evaluate(IReadOnlyList<FrozenTrial> trials, StudyDirection studyDirection);
}

/// <summary>
///     An error evaluator for objective functions based on cross-validation.
///     Port of <c>optuna/terminator/erroreval.py: CrossValidationErrorEvaluator</c>
///     (experimental since 3.2.0).
///
///     This evaluator evaluates the objective function's statistical error, which comes from the
///     randomness of dataset. This evaluator assumes that the objective function is the average of
///     the cross-validation and uses the scaled variance of the cross-validation scores in the best
///     trial at the moment as the statistical error.
/// </summary>
public class CrossValidationErrorEvaluator : BaseErrorEvaluator
{
    /// <summary>Python <c>_CROSS_VALIDATION_SCORES_KEY</c>.</summary>
    internal const string CrossValidationScoresKey = "terminator:cv_scores";

    /// <summary>
    ///     Evaluate the statistical error of the objective function based on cross-validation.
    ///
    ///     The best trial in <paramref name="trials" /> is used to compute the statistical error.
    /// </summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="studyDirection">Whether larger or smaller objective values are preferred.</param>
    /// <returns>The corrected standard error calculated from the best trial's cross-validation scores.</returns>
    /// <exception cref="NumSharp.ValueError">The best completed trial does not contain reported cross-validation scores.</exception>
    /// <exception cref="NumSharp.TypeError">A stored cross-validation score is not numeric.</exception>
    public override double Evaluate(IReadOnlyList<FrozenTrial> trials, StudyDirection studyDirection)
    {
        var completeTrials = trials.Where(t => t.State == TrialState.Complete).ToList();
        Debug.Assert(completeTrials.Count > 0);

        // Python `max`/`min` return the FIRST extremal element on ties.
        FrozenTrial bestTrial = completeTrials[0];
        for (int i = 1; i < completeTrials.Count; i++)
        {
            double bestValue = bestTrial.Value!.Value;
            double currentValue = completeTrials[i].Value!.Value;
            bool better = studyDirection == StudyDirection.Maximize
                ? currentValue > bestValue
                : currentValue < bestValue;
            if (better)
                bestTrial = completeTrials[i];
        }

        var bestTrialAttrs = bestTrial.SystemAttrs;
        if (!bestTrialAttrs.TryGetValue(CrossValidationScoresKey, out var cvScoresObj))
            throw new ValueError(
                "Cross-validation scores have not been reported. Please call " +
                "`report_cross_validation_scores(trial, scores)` during a trial and pass the " +
                "list of scores as `scores`.");

        var cvScores = ToDoubleList(cvScoresObj);

        int k = cvScores.Count;
        Debug.Assert(k > 1, "Should be guaranteed by `report_cross_validation_scores`.");
        double scale = 1.0 / k + 1.0 / (k - 1);

        double var = scale * NumpyVar(cvScores);
        double std = Math.Sqrt(var);

        return std;
    }

    /// <summary>
    ///     NumPy <c>np.var(x)</c> (population variance, <c>ddof=0</c>). For the small arrays a
    ///     cross-validation produces (&lt; 128 elements), NumPy sums sequentially, so this naive
    ///     two-pass sum is bit-identical.
    /// </summary>
    /// <param name="x">The numerical input values, in the order used by the operation.</param>
    /// <returns>The population variance using the port's specified accumulation order.</returns>
    private static double NumpyVar(IReadOnlyList<double> x)
    {
        int n = x.Count;
        double sum = 0.0;
        for (int i = 0; i < n; i++)
            sum += x[i];
        double mean = sum / n;
        double squared = 0.0;
        for (int i = 0; i < n; i++)
        {
            double d = x[i] - mean;
            squared += d * d;
        }
        return squared / n;
    }

    /// <summary>Returns the numeric items of the supplied score sequence, or an empty list for a non-sequence value.</summary>
    /// <param name="value">The value to convert, format, or compare using the operation's stated convention.</param>
    /// <returns>The numeric items of the supplied score sequence, or an empty list for a non-sequence value.</returns>
    /// <exception cref="NumSharp.TypeError">A score sequence contains a nonnumeric item.</exception>
    private static List<double> ToDoubleList(object? value)
    {
        var result = new List<double>();
        if (value is IEnumerable sequence and not string)
        {
            foreach (var item in sequence)
            {
                if (!PyFloat.TryConvert(item, out double v))
                    throw new TypeError("Cross-validation scores must be numeric.");
                result.Add(v);
            }
        }
        return result;
    }
}

/// <summary>
///     An error evaluator that always returns a constant value.
///     Port of <c>optuna/terminator/erroreval.py: StaticErrorEvaluator</c>
///     (experimental since 3.2.0).
///
///     This evaluator can be used to terminate the optimization when the evaluated improvement
///     potential is below the fixed threshold.
/// </summary>
public class StaticErrorEvaluator : BaseErrorEvaluator
{
    private readonly double _constant;

    /// <param name="constant">A user-specified constant value to always return as an error estimate.</param>
    /// <summary>Stores a fixed error threshold for comparison with an improvement estimate.</summary>
    public StaticErrorEvaluator(double constant)
    {
        _constant = constant;
    }

    /// <summary>Returns the constant error threshold supplied to the constructor.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="studyDirection">Whether larger or smaller objective values are preferred.</param>
    /// <returns>The constant error threshold supplied to the constructor.</returns>
    public override double Evaluate(IReadOnlyList<FrozenTrial> trials, StudyDirection studyDirection)
        => _constant;
}

/// <summary>
///     Module-level functions of <c>optuna.terminator</c> (<c>report_cross_validation_scores</c>).
/// </summary>
public static class TerminatorFunctions
{
    /// <summary>
    ///     Report cross-validation scores of a trial. Port of
    ///     <c>optuna/terminator/erroreval.py: report_cross_validation_scores</c>.
    ///
    ///     This function should be called within the objective function to report the
    ///     cross-validation scores. The reported scores are used to evaluate the statistical error
    ///     for termination judgement.
    /// </summary>
    /// <param name="trial">A <see cref="Trial" /> object to report the cross-validation scores.</param>
    /// <param name="scores">The cross-validation scores of the trial.</param>
    /// <exception cref="NumSharp.ValueError">Fewer than two cross-validation scores are supplied.</exception>
    public static void ReportCrossValidationScores(Trial trial, IReadOnlyList<double> scores)
    {
        OptunaWarnings.Warn(TerminatorModule.DeprecationWarningMessage, WarningCategory.FutureWarning);

        if (scores.Count <= 1)
            throw new ValueError("The length of `scores` is expected to be greater than one.");

        trial.Storage.SetTrialSystemAttr(
            trial.TrialId,
            CrossValidationErrorEvaluator.CrossValidationScoresKey,
            scores.Select(s => (object?)s).ToList());
    }
}
