using OptunaSharp.Internal;

namespace OptunaSharp.Integration.Sklearn;

/// <summary>
///     Byte-exact ports of the scikit-learn regression metrics that <see cref="OptunaSearchCV" /> scores
///     with — <c>sklearn.metrics.r2_score</c> and <c>sklearn.metrics.mean_squared_error</c> — for the
///     single-output (<c>n_outputs == 1</c>), no-<c>sample_weight</c> case.
///
///     <para>
///     Every reduction rides <c>MathUtil.PairwiseSum</c> (NumPy's <c>add.reduce</c> order) and the
///     residual squares use <c>x * x</c> — NumPy's array <c>** 2</c> is the <c>square</c> ufunc, a
///     multiply, <b>not</b> <c>pow(x, 2.0)</c> (verified against NumPy 2.4.2: <c>arr ** 2 == arr * arr</c>
///     but <c>!= pow</c>). So on identical <c>(y_true, y_pred)</c> the result is bit-for-bit scikit-learn.
///     </para>
/// </summary>
internal static class Metrics
{
    /// <summary>
    ///     Port of <c>sklearn.metrics.r2_score(y_true, y_pred, multioutput="uniform_average",
    ///     force_finite=True)</c> for a single output. Returns <c>NaN</c> for fewer than two samples
    ///     (matching scikit-learn's <c>UndefinedMetricWarning</c> path). With <paramref name="forceFinite" />
    ///     (the default) a constant <c>y_true</c> yields <c>1.0</c> for perfect predictions (zero numerator)
    ///     or <c>0.0</c> otherwise, instead of <c>NaN</c>/<c>-Inf</c>.
    /// </summary>
    /// <param name="yTrue">The observed target values, aligned with the predictions.</param>
    /// <param name="yPred">The predicted target values, aligned with the observations.</param>
    /// <param name="forceFinite">Whether perfect/imperfect predictions for a constant target return 1/0 instead of nonfinite R-squared values.</param>
    /// <returns>The coefficient of determination, with the requested treatment of constant-target nonfinite values.</returns>
    internal static double R2Score(double[] yTrue, double[] yPred, bool forceFinite = true)
    {
        int n = yTrue.Length;
        if (n < 2)
            return double.NaN;

        // numerator = sum((y_true - y_pred) ** 2, axis=0)
        var num = new double[n];
        for (int i = 0; i < n; i++)
        {
            double r = yTrue[i] - yPred[i];
            num[i] = r * r;
        }
        double numerator = MathUtil.PairwiseSum(num);

        // denominator = sum((y_true - average(y_true)) ** 2, axis=0); _average(weights=None) == mean.
        double meanY = MathUtil.PairwiseSum(yTrue) / n;
        var den = new double[n];
        for (int i = 0; i < n; i++)
        {
            double d = yTrue[i] - meanY;
            den[i] = d * d;
        }
        double denominator = MathUtil.PairwiseSum(den);

        // _assemble_fraction_of_explained_deviance (single output).
        if (!forceFinite)
            return 1.0 - numerator / denominator;

        if (numerator == 0.0)
            return 1.0;                          // zero numerator = perfect, even if denominator == 0.
        if (denominator == 0.0)
            return 0.0;                          // nonzero numerator, zero denominator.
        return 1.0 - numerator / denominator;
    }

    /// <summary>
    ///     Port of <c>sklearn.metrics.mean_squared_error(y_true, y_pred)</c> for a single output —
    ///     <c>average((y_true - y_pred) ** 2)</c> = <c>MathUtil.PairwiseSum</c> of the squared
    ///     residuals divided by <c>n</c>. The scikit-learn scorer <c>neg_mean_squared_error</c> is its
    ///     negation (higher-is-better).
    /// </summary>
    /// <param name="yTrue">The observed target values, aligned with the predictions.</param>
    /// <param name="yPred">The predicted target values, aligned with the observations.</param>
    /// <returns>The mean squared residual across the aligned observations and predictions.</returns>
    internal static double MeanSquaredError(double[] yTrue, double[] yPred)
    {
        int n = yTrue.Length;
        var sq = new double[n];
        for (int i = 0; i < n; i++)
        {
            double r = yTrue[i] - yPred[i];
            sq[i] = r * r;
        }
        return MathUtil.PairwiseSum(sq) / n;
    }
}
