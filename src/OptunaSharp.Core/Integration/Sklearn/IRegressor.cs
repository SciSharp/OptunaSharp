using System.Collections.Generic;

namespace OptunaSharp.Integration.Sklearn;

/// <summary>
///     The minimal scikit-learn regressor surface that <see cref="SklearnEstimator" /> and the
///     <see cref="Scorer" /> drive — the .NET analogue of the <c>fit</c> / <c>predict</c> / <c>score</c>
///     methods plus <c>clone(estimator).set_params(**params)</c>.
///
///     <para>
///     <see cref="RandomForestRegressor" /> is the byte-exact implementation. The interface keeps the
///     cross-validation machinery (folds, scorer, <see cref="OptunaSearchCV" />) independent of the
///     concrete estimator, so a future byte-exact regressor can slot in unchanged.
///     </para>
/// </summary>
public interface IRegressor
{
    /// <summary>
    ///     Return a fresh, <b>unfitted</b> clone of this estimator with <paramref name="parameters" />
    ///     applied — the analogue of <c>clone(estimator).set_params(**parameters)</c>. Keys are
    ///     scikit-learn parameter names (e.g. <c>"n_estimators"</c>, <c>"max_depth"</c>,
    ///     <c>"min_samples_split"</c>, <c>"max_features"</c>).
    /// </summary>
    /// <param name="parameters">The estimator hyperparameter names and values to apply.</param>
    /// <returns>An unfitted regressor with the requested parameter overrides applied.</returns>
    IRegressor WithParams(IReadOnlyDictionary<string, object?> parameters);

    /// <summary>Fit the estimator on <paramref name="x" /> (row-major <c>(n_samples, n_features)</c>) and targets <paramref name="y" />.</summary>
    /// <param name="x">Feature values with one sample per row and one feature per column.</param>
    /// <param name="y">Target values aligned with the feature rows.</param>
    void Fit(double[,] x, double[] y);

    /// <summary>Predict targets for <paramref name="x" /> (row-major <c>(n_samples, n_features)</c>). Requires a prior <see cref="Fit" />.</summary>
    /// <param name="x">Feature values with one sample per row and one feature per column.</param>
    /// <returns>One predicted target value per feature row.</returns>
    double[] Predict(double[,] x);

    /// <summary>The estimator's default score on <paramref name="x" />/<paramref name="y" /> (for a regressor, <c>r2_score</c>).</summary>
    /// <param name="x">Feature values with one sample per row and one feature per column.</param>
    /// <param name="y">Target values aligned with the feature rows.</param>
    /// <returns>The coefficient of determination, where higher values indicate better predictions.</returns>
    double Score(double[,] x, double[] y);
}
