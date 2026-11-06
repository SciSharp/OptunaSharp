using System;
using AwesomeAssertions;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests;

/// <summary>
///     Pins the first-study example that the public entry points print: the root <c>README.md</c> quick start,
///     <c>docs/website-src/docs/getting-started.md</c> ("Your first study") and the DocFX home page. Each shows the
///     same seeded TPE study together with the exact values it prints, and says Python Optuna prints the same.
///     <para>The expected values were produced by live Optuna 5.0.0 (the vendored <c>refs/optuna</c>) on
///     2026-11-06: <c>optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))</c>, optimizing
///     <c>(trial.suggest_float("x", -10, 10) - 2) ** 2</c> for 20 trials, gives
///     <c>best_value == 0.0004973004749973951</c> at <c>x == 2.0223002348641756</c>. A change that breaks this
///     test also makes the published quick start wrong, so the pages must be updated with it.</para>
/// </summary>
[TestClass]
public class GettingStartedDocsTests
{
    /// <summary>
    ///     The documented study reproduces the documented values bit for bit: the best value and its <c>x</c>
    ///     compare with <c>Be</c>, not an approximation, because the pages print them to the last digit.
    /// </summary>
    [TestMethod]
    public void FirstStudy_PrintsTheDocumentedValues_ThatPythonOptunaPrints()
    {
        var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 42));

        study.Optimize(trial =>
        {
            double x = trial.SuggestFloat("x", -10, 10);
            return (x - 2) * (x - 2);
        }, nTrials: 20);

        study.BestValue.Should().Be(0.0004973004749973951, "Optuna 5.0.0 prints this study.best_value");
        Convert.ToDouble(study.BestParams["x"]).Should().Be(2.0223002348641756, "Optuna 5.0.0 prints this study.best_params['x']");
        study.Trials.Should().HaveCount(20);
    }
}
