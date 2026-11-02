using System;
using System.Collections.Generic;
using System.IO;
using AwesomeAssertions;
using OptunaSharp.Artifacts;
using OptunaSharp.Importance;
using OptunaSharp.Importance.PedAnova;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs in error contracts and argument validation, found by the 2026-11-02 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-02.md</c>). Python raises where the port does not, or raises a different
///     type or text; each test pins Optuna 5.0.0's behavior, fails today, and passes once fixed, at which point
///     its <c>[OpenBugs]</c> attribute should be removed.
/// </summary>
[TestClass]
public class ErrorContractParityBugTests
{
    /// <summary>
    ///     QA-15: fANOVA on a study whose objective never varies must raise Python's <c>RuntimeError</c>
    ///     (NumSharp's <see cref="NumSharp.RuntimeError" />), not <see cref="InvalidOperationException" />.
    /// </summary>
    /// <remarks>
    ///     Optuna's <c>_Fanova.fit</c> raises <c>RuntimeError("Encountered zero total variance in all
    ///     trees.")</c>; the message matches but the type does not, so a <c>catch (RuntimeError)</c> written
    ///     from the Python docs misses it.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void Fanova_ZeroVariance_RaisesRuntimeError()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 42));
        study.Optimize(t =>
        {
            t.SuggestFloat("x", -1, 1);
            t.SuggestFloat("y", -1, 1);
            return 1.0;
        }, nTrials: 30);

        ((Action)(() => ImportanceFunctions.GetParamImportances(study, new FanovaImportanceEvaluator(seed: 0))))
            .Should().Throw<NumSharp.RuntimeError>().WithMessage("Encountered zero total variance in all trees.");
    }

    /// <summary>
    ///     QA-16: PED-ANOVA's quantile precondition must reject bad arguments in every build, as Python's
    ///     <c>assert</c> does; the port uses <c>Debug.Assert</c>, which the Release package compiles out.
    /// </summary>
    /// <remarks>
    ///     Python: <c>PedAnovaImportanceEvaluator(target_quantile=0.5, region_quantile=0.2)</c> raises
    ///     <c>AssertionError: condition 0.0 &lt; `target_quantile` &lt; `region_quantile` &lt;= 1.0 must be
    ///     satisfied</c>. The shipped (Release) OptunaSharp accepts it and computes importances from an empty or
    ///     inverted region. Only meaningful when this test runs against a Release build, as CI does.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void PedAnova_InvalidQuantiles_RejectedInReleaseBuilds()
    {
        ((Action)(() => new PedAnovaImportanceEvaluator(targetQuantile: 0.5, regionQuantile: 0.2)))
            .Should().Throw<Exception>()
            .WithMessage("*condition 0.0 < `target_quantile` < `region_quantile` <= 1.0 must be satisfied*");
    }

    /// <summary>
    ///     QA-16 (artifacts): <see cref="Backoff" /> must reject a non-positive retry count at construction, as
    ///     Python's <c>assert max_retries &gt; 0</c> does; in Release the port accepts it and then silently skips
    ///     every upload.
    /// </summary>
    /// <remarks>
    ///     With <c>maxRetries: 0</c> the retry loop of <see cref="Backoff.Write" /> never runs, so the call returns
    ///     without writing anything (silent data loss), and <see cref="Backoff.OpenReader" /> throws "must not
    ///     reach here". Python fails fast with <c>AssertionError</c> in <c>Backoff.__init__</c>. The remarks on
    ///     the constructor already call these "debug assertions".
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void Backoff_ZeroRetries_RejectedAtConstruction()
    {
        string dir = Path.Combine(Path.GetTempPath(), "optunasharp-qa-backoff-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var store = new FileSystemArtifactStore(dir);
            ((Action)(() => new Backoff(store, maxRetries: 0))).Should().Throw<Exception>();
        }
        finally
        {
            // The store may have written nothing; delete the scratch folder either way.
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    ///     QA-17: a CMA-ES <c>x0</c> that misses a parameter of the search space must fail with Python's
    ///     message; the port throws a <see cref="KeyNotFoundException" /> naming the key.
    /// </summary>
    /// <remarks>
    ///     Optuna 5.0.0 with <c>x0={"x": 1.0, "y": -1.0, "z": 1.0}</c> on a space that also has <c>li</c>
    ///     fails the second trial with <c>AssertionError: Parameter configuration must contain all
    ///     distributions.</c> (from <c>_SearchSpaceTransform.transform</c>); the port fails the same trial with
    ///     <c>KeyNotFoundException: The given key 'li' was not present in the dictionary.</c> Trajectories up to
    ///     the failure are identical. NIT: the C# text is arguably more helpful, but it is not Optuna's contract.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void CmaEs_X0MissingParameter_FailsWithPythonsMessage()
    {
        var study = Optuna.CreateStudy(sampler: new CmaEsSampler(
            seed: 3, x0: new Dictionary<string, object?> { ["x"] = 1.0, ["y"] = -1.0 }, sigma0: 0.5));

        Action run = () => study.Optimize(t => t.SuggestFloat("x", -5, 5) + t.SuggestFloat("y", -5, 5) + t.SuggestInt("li", 1, 1000, log: true), nTrials: 3);

        run.Should().Throw<Exception>().WithMessage("Parameter configuration must contain all distributions.");
    }
}
