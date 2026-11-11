using AwesomeAssertions;
using OptunaSharp.Internal;
using OptunaSharp.Terminator;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs in the terminator, found by the 2026-11-02 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-02.md</c>). Expected values are Optuna 5.0.0 / numpy 2.4.2 results; each
///     test fails today and passes once fixed, at which point its <c>[OpenBugs]</c> attribute should be removed.
/// </summary>
[TestClass]
public class TerminatorParityBugTests
{
    /// <summary>
    ///     QA-22 (fixed 2026-11-11): <see cref="CrossValidationErrorEvaluator" /> must compute <c>np.var</c> with numpy's
    ///     pairwise summation; the port summed sequentially, which differs from 8 scores on (10-fold CV is the usual case).
    /// </summary>
    /// <remarks>
    ///     numpy's <c>add.reduce</c> on a contiguous float64 array uses 8 interleaved accumulators once
    ///     <c>n &gt;= 8</c> (<c>pairwise_sum</c>), both for the mean and for the sum of squared deviations; the
    ///     port's <c>NumpyVar</c> looped once. For these ten scores Optuna returns <c>0.03772062328635506</c> and
    ///     the old port <c>0.03772062328635505</c>; 1,584 of 5,000 random 10-fold score sets differed. Fixed by computing
    ///     the variance with NumSharp's <c>np.var</c>, which runs NumPy's <c>_var</c> and its pairwise sums.
    /// </remarks>
    [TestMethod]
    public void CrossValidationError_TenScores_MatchesNumpyPairwiseVar()
    {
        var scores = new[] { 0.942, 0.9189, 0.7766, 0.9146, 0.7776, 0.9348, 0.886, 0.804, 0.7631, 0.7021 };
        var study = Optuna.CreateStudy();
        var trial = study.Ask();
        TerminatorFunctions.ReportCrossValidationScores(trial, scores);
        study.Tell(trial, 0.5);

        double error = new CrossValidationErrorEvaluator().Evaluate(study.Trials, StudyDirection.Minimize);

        PythonRepr.FloatRepr(error).Should().Be("0.03772062328635506");
    }
}
