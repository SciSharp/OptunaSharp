using System.Collections.Generic;

namespace OptunaSharp.Integration;

/// <summary>
///     The per-fold outcome of one cross-validation split, the C# analogue of one column of scikit-learn's
///     <c>cross_validate</c> result (<c>test_score</c> / <c>train_score</c> / <c>fit_time</c> /
///     <c>score_time</c> at a single fold index).
/// </summary>
/// <param name="TestScore">
///     The validation score for this fold. Following the scikit-learn scorer convention (which
///     <see cref="OptunaSearchCV" /> mirrors), <b>higher is better</b>. For a loss, negate it.
/// </param>
/// <param name="TrainScore">
///     The training score for this fold, or <c>null</c> when training scores were not requested
///     (<c>returnTrainScore = false</c>).
/// </param>
/// <param name="FitTimeSeconds">Seconds spent fitting the estimator on this fold's training split.</param>
/// <param name="ScoreTimeSeconds">Seconds spent scoring the estimator on this fold's validation split.</param>
public readonly record struct CrossValidationFoldResult(
    double TestScore, double? TrainScore, double FitTimeSeconds, double ScoreTimeSeconds);

/// <summary>
///     The estimator + cross-validation + scorer seam that <see cref="OptunaSearchCV" /> drives, the
///     .NET-idiomatic analogue of the scikit-learn <c>(estimator, cv, scorer)</c> bundle that Optuna's
///     Python <c>OptunaSearchCV</c> composes internally.
///
///     <para>
///     An implementation owns the training data, the fold strategy, and the scoring metric.
///     <see cref="OptunaSearchCV" /> owns the Optuna search: it samples a parameter set per trial, calls
///     <see cref="CrossValidate" /> to score it, records the per-fold / mean / std user attributes, reports
///     the cross-validation scores to the terminator, and maximizes the mean validation score — exactly as
///     the Python <c>_Objective</c> does.
///     </para>
///
///     <para>
///     Two backends ship with OptunaSharp. The <b>native scikit-learn port</b> —
///     <c>OptunaSharp.Integration.Sklearn.SklearnEstimator</c> (a pure-managed, byte-exact port of
///     <c>RandomForestRegressor</c> + <c>KFold</c> + <c>cross_validate</c> + the <c>r2</c> scorer) — makes
///     the whole search <b>end-to-end byte-exact</b> against Python's <c>optuna_integration.OptunaSearchCV</c>.
///     The <b>ML.NET adapter</b> — <c>OptunaSharp.Integration.MLNetEstimator</c> — drives an arbitrary
///     ML.NET pipeline (validated by usage, not bit-parity, since ML.NET is not scikit-learn). A custom /
///     in-memory implementation (e.g. for tests) only needs to produce fold scores.
///     </para>
/// </summary>
public interface ICrossValidationEstimator
{
    /// <summary>
    ///     The number of cross-validation splits — scikit-learn's <c>cv.get_n_splits(...)</c>, surfaced by
    ///     <see cref="OptunaSearchCV.NSplits" />.
    /// </summary>
    int GetNumSplits();

    /// <summary>
    ///     Fit + score the estimator, configured with <paramref name="parameters" />, across every
    ///     cross-validation fold, returning one <see cref="CrossValidationFoldResult" /> per fold. The
    ///     analogue of cloning the estimator, applying the sampled parameters, and running scikit-learn's
    ///     <c>cross_validate</c>.
    /// </summary>
    /// <param name="parameters">
    ///     The hyperparameters sampled for this trial (Optuna parameter name → suggested value), to apply to
    ///     the estimator before fitting.
    /// </param>
    /// <param name="returnTrainScore">
    ///     When <c>true</c>, each fold result should also carry a <see cref="CrossValidationFoldResult.TrainScore" />.
    /// </param>
    /// <returns>One result per fold; the list length must equal <see cref="GetNumSplits" />.</returns>
    IReadOnlyList<CrossValidationFoldResult> CrossValidate(
        IReadOnlyDictionary<string, object?> parameters, bool returnTrainScore);

    /// <summary>
    ///     Refit the estimator on the full training data with the best parameters found, retaining the
    ///     fitted model for prediction. The analogue of scikit-learn's refit step
    ///     (<c>best_estimator_.fit(X, y)</c>); invoked by <see cref="OptunaSearchCV" /> when
    ///     <c>refit = true</c>.
    /// </summary>
    /// <param name="parameters">The best hyperparameters found by the search.</param>
    void Refit(IReadOnlyDictionary<string, object?> parameters);
}
