using NumSharp;

namespace OptunaSharp.Integration.Sklearn;

/// <summary>
///     A scikit-learn scorer: evaluates a fitted estimator on <c>(X, y)</c>, <b>higher is better</b> — the
///     analogue of the callable <c>scorer(estimator, X, y)</c> that <c>cross_validate</c> and
///     <see cref="OptunaSearchCV" /> use.
/// </summary>
/// <param name="estimator">The regressor configuration cloned for each fold or parameter assignment.</param>
/// <param name="x">Feature values with one sample per row and one feature per column.</param>
/// <param name="y">Target values aligned with the feature rows.</param>
/// <returns>The scalar validation score, with higher values indicating better predictions.</returns>
public delegate double Scorer(IRegressor estimator, double[,] x, double[] y);

/// <summary>
///     Port of scikit-learn's <c>check_scoring</c> for the regressor path — resolves a scoring name (or
///     <c>null</c>) to a <see cref="Scorer" />. <c>null</c> uses the estimator's own <c>score</c> (for a
///     regressor, <c>r2</c>), exactly as <c>check_scoring(estimator, scoring=None)</c> does.
/// </summary>
public static class Scorers
{
    /// <summary>The default scorer (<c>scoring=None</c>): the estimator's own <c>score</c> — <c>r2</c> for a regressor.</summary>
    public static Scorer Default { get; } = (est, x, y) => est.Score(x, y);

    /// <summary>The <c>"r2"</c> scorer — <c>r2_score(y, estimator.predict(X))</c>.</summary>
    public static Scorer R2 { get; } = (est, x, y) => Metrics.R2Score(y, est.Predict(x));

    /// <summary>The <c>"neg_mean_squared_error"</c> scorer — <c>-mean_squared_error(y, estimator.predict(X))</c>.</summary>
    public static Scorer NegMeanSquaredError { get; } = (est, x, y) => -Metrics.MeanSquaredError(y, est.Predict(x));

    /// <summary>
    ///     Resolve a scikit-learn scoring name to a <see cref="Scorer" />. Supported: <c>null</c> (the
    ///     estimator's default <c>score</c>), <c>"r2"</c>, and <c>"neg_mean_squared_error"</c>.
    /// </summary>
    /// <param name="scoring">The registered scorer name, or null to use the estimator's own score.</param>
    /// <returns>The registered scorer corresponding to the supplied name.</returns>
    /// <exception cref="NumSharp.ValueError">The scorer name is neither r2 nor neg_mean_squared_error and is not null.</exception>
    public static Scorer CheckScoring(string? scoring = null) => scoring switch
    {
        null => Default,
        "r2" => R2,
        "neg_mean_squared_error" => NegMeanSquaredError,
        _ => throw new ValueError(
            $"scoring='{scoring}' is not supported by OptunaSharp.Integration.Sklearn " +
            "(supported: null, 'r2', 'neg_mean_squared_error')."),
    };
}
