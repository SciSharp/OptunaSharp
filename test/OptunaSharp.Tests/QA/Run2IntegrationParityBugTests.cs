using System.Collections.Generic;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Integration;
using OptunaSharp.Integration.Sklearn;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs in the integrations found by the 2026-11-10 parity QA run (<c>docs/qa/PARITY_QA_2026-11-10.md</c>).
///     Each test fails today and passes once fixed, at which point its <c>OpenBugs</c> category should be removed.
/// </summary>
[TestClass]
public class Run2IntegrationParityBugTests
{
    /// <summary>
    ///     QA-81: optuna-integration's <c>OptunaSearchCV.__init__</c> stores <c>self.n_jobs = n_jobs if n_jobs else 1</c>,
    ///     so <c>n_jobs=0</c> (falsy) runs sequentially like <c>None</c>. The port maps only <c>null</c> to 1
    ///     (<c>nJobs ?? 1</c>) and hands 0 to <c>Study.Optimize</c>, which refuses it like <c>ThreadPoolExecutor</c>:
    ///     <c>ValueError("max_workers must be greater than 0")</c>.
    /// </summary>
    /// <remarks>
    ///     Measured with optuna-integration (vendored) and scikit-learn: <c>OptunaSearchCV(RandomForestRegressor(
    ///     n_estimators=5, random_state=0), {"max_depth": IntDistribution(1, 4)}, n_trials=2, n_jobs=0,
    ///     random_state=0).fit(X, y)</c> completes two trials and reports <c>n_jobs == 1</c>, the same as
    ///     <c>n_jobs=None</c> and <c>n_jobs=1</c>; the port's <c>Fit()</c> throws.
    ///     <b>Fixed 2026-11-12:</b> <c>nJobs: 0</c> maps to 1, optuna-integration's <c>n_jobs if n_jobs else 1</c>.
    /// </remarks>
    [TestMethod]
    public void OptunaSearchCV_NJobsZero_RunsSequentiallyLikeOptunaIntegration()
    {
        // Any regression data will do: the defect is in argument handling, before the first fold is fitted.
        var x = new double[30, 3];
        var y = new double[30];
        for (int i = 0; i < 30; i++)
        {
            for (int j = 0; j < 3; j++)
                x[i, j] = ((i * 7 + j * 3) % 11) / 10.0;
            y[i] = 2 * x[i, 0] + 0.1 * x[i, 2];
        }
        var estimator = new SklearnEstimator(new RandomForestRegressor(nEstimators: 5, randomState: 0), x, y);
        var search = new OptunaSearchCV(estimator,
            new Dictionary<string, BaseDistribution> { ["max_depth"] = new IntDistribution(1, 4) },
            nTrials: 2, nJobs: 0, randomState: 0);

        search.Fit();

        search.NTrials.Should().Be(2);
    }
}
