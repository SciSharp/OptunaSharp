using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace OptunaSharp.Integration.Sklearn;

/// <summary>
///     Port of the essence of <c>sklearn.model_selection.cross_validate</c> + <c>_fit_and_score</c> for the
///     native <see cref="IRegressor" />: for each cross-validation split it clones the estimator, fits it on
///     the training rows, and scores it on the validation rows (and the training rows when
///     <c>returnTrainScore</c> is set), returning one <see cref="CrossValidationFoldResult" /> per fold.
///
///     <para>
///     The <b>scores</b> are byte-exact against scikit-learn (identical folds, forest, and scorer); the
///     <b>times</b> are wall-clock and only informational (scikit-learn does not compare them either). A
///     fold whose fit or score throws is assigned <c>errorScore</c> (scikit-learn's
///     <c>error_score</c>, default <c>NaN</c>). Fold generation and row-selection failures still
///     propagate because they prevent construction of a valid train/test partition.
///     </para>
/// </summary>
public static partial class CrossValidation
{
    /// <summary>Returns one validation result per fold, including timing and requested training scores.</summary>
    /// <param name="estimator">The regressor configuration cloned for each fold or parameter assignment.</param>
    /// <param name="x">Feature values with one sample per row and one feature per column.</param>
    /// <param name="y">Target values aligned with the feature rows.</param>
    /// <param name="cv">The prepared fold generator used to select training and test rows.</param>
    /// <param name="scorer">The scoring delegate; greater scores must indicate better predictions.</param>
    /// <param name="returnTrainScore">Whether each fold also computes and returns its training score.</param>
    /// <param name="errorScore">The numeric score assigned to a fold whose fitting or scoring raises an exception.</param>
    /// <returns>One validation result per fold, including timing and requested training scores.</returns>
    /// <exception cref="System.IndexOutOfRangeException">A split contains a row index outside the supplied features or targets.</exception>
    /// <exception cref="System.Exception">The cross-validator throws while generating folds; estimator and scorer failures inside an established fold are converted to errorScore.</exception>
    internal static List<CrossValidationFoldResult> CrossValidate(
        IRegressor estimator,
        double[,] x,
        double[] y,
        BaseCrossValidator cv,
        Scorer scorer,
        bool returnTrainScore,
        double errorScore)
    {
        int nSamples = y.Length;
        var results = new List<CrossValidationFoldResult>();

        foreach (var (train, test) in cv.Split(nSamples))
        {
            // Validate the partition by selecting rows before entering fold-level error handling.
            // An invalid split is a cross-validation setup error, not a failed estimator fit.
            var (xTrain, yTrain) = Take(x, y, train);
            var (xTest, yTest) = Take(x, y, test);

            var stopwatch = Stopwatch.StartNew();
            double testScore;
            double? trainScore = null;
            double fitTime;
            double scoreTime;

            try
            {
                // clone(estimator) per fold — an unfitted copy carrying the configured hyperparameters.
                var fold = estimator.WithParams(EmptyParams);
                fold.Fit(xTrain, yTrain);
                fitTime = stopwatch.Elapsed.TotalSeconds;

                double afterFit = stopwatch.Elapsed.TotalSeconds;
                testScore = scorer(fold, xTest, yTest);
                scoreTime = stopwatch.Elapsed.TotalSeconds - afterFit;

                if (returnTrainScore)
                    trainScore = scorer(fold, xTrain, yTrain);
            }
            catch
            {
                // scikit-learn's error_score: assign the fold the configured score instead of failing.
                fitTime = stopwatch.Elapsed.TotalSeconds;
                testScore = errorScore;
                scoreTime = 0.0;
                if (returnTrainScore)
                    trainScore = errorScore;
            }

            results.Add(new CrossValidationFoldResult(testScore, trainScore, fitTime, scoreTime));
        }

        return results;
    }

    private static readonly IReadOnlyDictionary<string, object?> EmptyParams =
        new Dictionary<string, object?>();

    // _safe_indexing(X, indices) / y[indices] — rows in the (ascending) index order the splitter yields.
    /// <summary>Returns copies of the selected feature rows and matching target values.</summary>
    /// <param name="x">Feature values with one sample per row and one feature per column.</param>
    /// <param name="y">Target values aligned with the feature rows.</param>
    /// <param name="indices">The ordered positions or sample-row indices to use.</param>
    /// <returns>Copies of the selected feature rows and matching target values.</returns>
    private static (double[,] X, double[] Y) Take(double[,] x, double[] y, int[] indices)
    {
        int cols = x.GetLength(1);
        var subX = new double[indices.Length, cols];
        var subY = new double[indices.Length];
        for (int r = 0; r < indices.Length; r++)
        {
            int src = indices[r];
            for (int c = 0; c < cols; c++)
                subX[r, c] = x[src, c];
            subY[r] = y[src];
        }
        return (subX, subY);
    }
}
