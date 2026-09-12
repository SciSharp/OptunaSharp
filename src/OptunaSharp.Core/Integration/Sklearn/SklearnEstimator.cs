using System;
using System.Collections.Generic;
using NumSharp;

namespace OptunaSharp.Integration.Sklearn;

/// <summary>
///     The native, byte-exact <see cref="ICrossValidationEstimator" /> backend for
///     <see cref="OptunaSearchCV" /> — the .NET analogue of the <c>(estimator, cv, scorer)</c> triple that
///     Optuna's Python <c>OptunaSearchCV</c> composes internally, and the piece that makes the whole search
///     <b>end-to-end byte-exact</b> against <c>optuna_integration.OptunaSearchCV</c>.
///
///     <para>
///     It owns the training data (<c>X</c>, <c>y</c>), a base <see cref="IRegressor" /> (typically a
///     <see cref="RandomForestRegressor" />), the fold strategy (<see cref="BaseCrossValidator" />, default
///     5-fold <see cref="KFold" />), and the scoring metric (<see cref="Scorer" />, default the estimator's
///     <c>r2</c>). Per trial, <see cref="CrossValidate" /> applies the sampled hyperparameters
///     (<c>clone(estimator).set_params(**params)</c>) and runs the ported
///     <c>cross_validate</c> — each fold byte-exact vs scikit-learn. After a refit the best model is exposed
///     through <see cref="Predict" /> / <see cref="Score" />.
///     </para>
/// </summary>
public sealed class SklearnEstimator : ICrossValidationEstimator
{
    private readonly IRegressor _baseEstimator;
    private readonly double[,] _x;
    private readonly double[] _y;
    private readonly BaseCrossValidator _cv;
    private readonly Scorer _scorer;
    private readonly double _errorScore;

    private IRegressor? _bestEstimator;

    /// <param name="estimator">
    ///     The base estimator to search over — its hyperparameters are overridden per trial with the
    ///     sampled values (e.g. a <see cref="RandomForestRegressor" />).
    /// </param>
    /// <param name="x">Training data, row-major <c>(n_samples, n_features)</c>.</param>
    /// <param name="y">Target values, length <c>n_samples</c>.</param>
    /// <param name="cv">
    ///     The cross-validation strategy. <c>null</c> uses 5-fold <see cref="KFold" /> (scikit-learn's
    ///     <c>check_cv</c> default for a regressor). Pass a <see cref="KFold" /> to control folds/shuffle, or
    ///     any <see cref="BaseCrossValidator" />.
    /// </param>
    /// <param name="scorer">
    ///     The scoring metric (higher is better). <c>null</c> uses the estimator's own <c>score</c>
    ///     (<c>r2</c> for a regressor) — see <see cref="Scorers.CheckScoring" />.
    /// </param>
    /// <param name="errorScore">
    ///     The score assigned to a fold whose fit or scoring throws. Defaults to <c>NaN</c>
    ///     (scikit-learn's <c>error_score=np.nan</c>).
    /// </param>
    /// <summary>Combines a regressor, owned training data, fold strategy, and scoring function for OptunaSearchCV.</summary>
    /// <exception cref="System.ArgumentNullException">The estimator, feature matrix, or target vector is null.</exception>
    /// <exception cref="NumSharp.ValueError">The feature-row and target counts differ, or the supplied fold/scorer specification is unsupported.</exception>
    public SklearnEstimator(
        IRegressor estimator,
        double[,] x,
        double[] y,
        BaseCrossValidator? cv = null,
        Scorer? scorer = null,
        double errorScore = double.NaN)
    {
        _baseEstimator = estimator ?? throw new ArgumentNullException(nameof(estimator));
        _x = x ?? throw new ArgumentNullException(nameof(x));
        _y = y ?? throw new ArgumentNullException(nameof(y));
        if (_x.GetLength(0) != _y.Length)
            throw new ValueError(
                $"X has {_x.GetLength(0)} rows but y has {_y.Length} entries; they must match.");
        _cv = cv ?? new KFold(5);
        _scorer = scorer ?? Scorers.Default;
        _errorScore = errorScore;
    }

    /// <summary>The best estimator refit on the full data (after <see cref="Refit" />), or <c>null</c> before one.</summary>
    public IRegressor? BestEstimator => _bestEstimator;

    /// <inheritdoc />
    public int GetNumSplits() => _cv.GetNumSplits(_y.Length);

    /// <inheritdoc />
    public IReadOnlyList<CrossValidationFoldResult> CrossValidate(
        IReadOnlyDictionary<string, object?> parameters, bool returnTrainScore)
    {
        // clone(estimator).set_params(**params), then cross_validate over the folds.
        var configured = _baseEstimator.WithParams(parameters);
        return CrossValidation.CrossValidate(
            configured, _x, _y, _cv, _scorer, returnTrainScore, _errorScore);
    }

    /// <inheritdoc />
    public void Refit(IReadOnlyDictionary<string, object?> parameters)
    {
        var best = _baseEstimator.WithParams(parameters);
        best.Fit(_x, _y);
        _bestEstimator = best;
    }

    /// <summary>Predict with the refit best estimator. Available only after a refit.</summary>
    /// <param name="x">Feature values with one sample per row and one feature per column.</param>
    /// <returns>One predicted target value per feature row.</returns>
    /// <exception cref="System.InvalidOperationException">The estimator has not been refit on the complete training data.</exception>
    public double[] Predict(double[,] x)
    {
        if (_bestEstimator is null)
            throw new InvalidOperationException(
                "The estimator has not been refit. Call OptunaSearchCV.Fit with refit = true first.");
        return _bestEstimator.Predict(x);
    }

    /// <summary>Score with the refit best estimator, using the configured scorer (higher is better). Available only after a refit.</summary>
    /// <param name="x">Feature values with one sample per row and one feature per column.</param>
    /// <param name="y">Target values aligned with the feature rows.</param>
    /// <returns>The coefficient of determination, where higher values indicate better predictions.</returns>
    /// <exception cref="System.InvalidOperationException">The estimator has not been refit on the complete training data.</exception>
    public double Score(double[,] x, double[] y)
    {
        if (_bestEstimator is null)
            throw new InvalidOperationException(
                "The estimator has not been refit. Call OptunaSearchCV.Fit with refit = true first.");
        return _scorer(_bestEstimator, x, y);
    }
}
