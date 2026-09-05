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
///     fold whose fit or score throws is assigned <paramref name="errorScore" /> (scikit-learn's
///     <c>error_score</c>, default <c>NaN</c>), never failing the whole search.
///     </para>
/// </summary>
public static partial class CrossValidation
{
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
