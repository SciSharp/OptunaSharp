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
    ///     QA-16 (third instance): multi-objective TPE must fail a trial whose <c>gamma</c> asks for a negative number
    ///     of "below" trials, as Optuna's <c>assert 0 &lt; n_below &lt; len(trials)</c> does; the port's
    ///     <c>Debug.Assert</c> is compiled out and the split carries on with <c>nBelow = -1</c>.
    /// </summary>
    /// <remarks>
    ///     <c>_split_complete_trials_multi_objective</c> returns early for <c>n_below == 0</c> and
    ///     <c>n_below == len(trials)</c> and asserts the open interval otherwise. With
    ///     <c>TPESampler(seed=1, n_startup_trials=2, gamma=lambda n: -1)</c> on a two-objective study, Optuna fails
    ///     trials 2-4 with <c>AssertionError()</c> (logged as "Trial 2 failed with parameters: {} because of the
    ///     following error: AssertionError()."); the port completes them. <c>gamma</c> is deprecated since 4.9, so this
    ///     only bites custom callables. Same class: <c>CrossValidationErrorEvaluator.evaluate([])</c> raises
    ///     <c>AssertionError</c> in Python and <see cref="ArgumentOutOfRangeException" /> here.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void TpeMultiObjective_NegativeGamma_FailsTrialsLikePythonAssert()
    {
#pragma warning disable CS0618 // gamma is deprecated in Optuna 4.9, as in the port; the test needs it.
        var sampler = new TPESampler(seed: 1, nStartupTrials: 2, gamma: n => -1);
#pragma warning restore CS0618
        var study = Optuna.CreateStudy(sampler: sampler, directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1), y = t.SuggestFloat("y", 0, 1);
            return (IReadOnlyList<double>)new[] { x, 1 - x + y };
        }, nTrials: 5, @catch: new[] { typeof(Exception) });

        study.Trials.ConvertAll(t => t.State).Should().Equal(
            TrialState.Complete, TrialState.Complete, TrialState.Fail, TrialState.Fail, TrialState.Fail);
    }

    /// <summary>
    ///     QA-39: constructing a deprecated class must emit Optuna's runtime <c>FutureWarning</c>; for
    ///     <c>RetryFailedTrialCallback</c> and <c>JournalFileStorage</c> the port only carries a compile-time
    ///     <see cref="ObsoleteAttribute" />, so code that is not recompiled, or that suppresses CS0618, is never told.
    /// </summary>
    /// <remarks>
    ///     Optuna's <c>@deprecated_class</c> warns on every construction:
    ///     "RetryFailedTrialCallback has been deprecated in v4.9.0. This feature will be removed in v6.0.0. See
    ///     https://github.com/optuna/optuna/releases/tag/v4.9.0. Use `RetryHeartbeatStaleTrialCallback` instead." and
    ///     "JournalFileStorage has been deprecated in v4.0.0. ... Use :class:`~optuna.storages.journal.JournalFileBackend`
    ///     instead.". The port already does this at run time for the deprecated distribution classes, the
    ///     <c>suggest_uniform</c> family, <c>get_failed_trial_callback</c> and the deprecated sampler arguments, so these
    ///     two are the odd ones out.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void DeprecatedClasses_EmitRuntimeFutureWarningLikeOptuna()
    {
        var warnings = new List<string>();
        void Handler(string message, global::OptunaSharp.Internal.WarningCategory category) => warnings.Add(message);
        global::OptunaSharp.Internal.OptunaWarnings.WarningRaised += Handler;
        string dir = Path.Combine(Path.GetTempPath(), "optunasharp-qa-deprecated-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
#pragma warning disable CS0618 // The point of the test is constructing the deprecated classes.
            _ = new global::OptunaSharp.Storages.RetryFailedTrialCallback(maxRetry: 1);
            _ = new global::OptunaSharp.Storages.Journal.JournalFileStorage(Path.Combine(dir, "a.log"));
#pragma warning restore CS0618
        }
        finally
        {
            global::OptunaSharp.Internal.OptunaWarnings.WarningRaised -= Handler;
            Directory.Delete(dir, recursive: true);
        }

        warnings.Should().Equal(
            "RetryFailedTrialCallback has been deprecated in v4.9.0. This feature will be removed in v6.0.0. "
            + "See https://github.com/optuna/optuna/releases/tag/v4.9.0. Use `RetryHeartbeatStaleTrialCallback` instead.",
            "JournalFileStorage has been deprecated in v4.0.0. This feature will be removed in v6.0.0. "
            + "See https://github.com/optuna/optuna/releases/tag/v4.0.0. "
            + "Use :class:`~optuna.storages.journal.JournalFileBackend` instead.");
    }

    /// <summary>
    ///     QA-17 (fixed 2026-11-12 by <c>1e8aba6</c>, with run 2's QA-83): a CMA-ES <c>x0</c> that misses a parameter of
    ///     the search space must fail with Python's message; the port threw a <see cref="KeyNotFoundException" /> naming
    ///     the key.
    /// </summary>
    /// <remarks>
    ///     Optuna 5.0.0 with <c>x0={"x": 1.0, "y": -1.0, "z": 1.0}</c> on a space that also has <c>li</c>
    ///     fails the second trial with <c>AssertionError: Parameter configuration must contain all
    ///     distributions.</c> (from <c>_SearchSpaceTransform.transform</c>); the port fails the same trial with
    ///     <c>KeyNotFoundException: The given key 'li' was not present in the dictionary.</c> Trajectories up to
    ///     the failure are identical. NIT: the C# text is arguably more helpful, but it is not Optuna's contract.
    /// </remarks>
    [TestMethod]
    public void CmaEs_X0MissingParameter_FailsWithPythonsMessage()
    {
        var study = Optuna.CreateStudy(sampler: new CmaEsSampler(
            seed: 3, x0: new Dictionary<string, object?> { ["x"] = 1.0, ["y"] = -1.0 }, sigma0: 0.5));

        Action run = () => study.Optimize(t => t.SuggestFloat("x", -5, 5) + t.SuggestFloat("y", -5, 5) + t.SuggestInt("li", 1, 1000, log: true), nTrials: 3);

        run.Should().Throw<Exception>().WithMessage("Parameter configuration must contain all distributions.");
    }
}
