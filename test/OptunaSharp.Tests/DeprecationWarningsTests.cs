using System;
using System.Collections.Generic;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests;

/// <summary>
///     Pins the runtime deprecation warnings of the ported <c>optuna.distributions</c> /
///     <c>optuna.trial</c> / <c>optuna.study</c> surface. Optuna's <c>@deprecated_class</c> /
///     <c>@deprecated_func</c> decorators raise a <c>FutureWarning</c> with a fixed message when a
///     deprecated class is constructed or a deprecated function is called; OptunaSharp mirrors that
///     through <see cref="OptunaWarnings.Warn(string, WarningCategory)" /> (the compile-time
///     <c>[Obsolete]</c> attribute alone has no runtime effect).
///
///     The expected strings below are the <b>verbatim</b> messages captured from live Optuna 5.x —
///     including Optuna's Python member names (<c>suggest_uniform</c>, <c>system_attrs</c>) and its
///     raw reStructuredText role markup — so this guards byte-for-byte parity and catches any drift
///     in <see cref="DeprecationMessages" />.
/// </summary>
[TestClass]
public class DeprecationWarningsTests
{
    /// <summary>Run <paramref name="action" /> and return the single warning it raises through the OptunaSharp seam.</summary>
    private static (string Message, WarningCategory Category) CaptureSingleWarning(Action action)
    {
        var captured = new List<(string Message, WarningCategory Category)>();
        void Handler(string message, WarningCategory category) => captured.Add((message, category));
        OptunaWarnings.WarningRaised += Handler;
        try
        {
            action();
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }

        captured.Should().ContainSingle("a deprecated member emits exactly one FutureWarning");
        return captured[0];
    }

    /// <summary>Assert <paramref name="action" /> raises a single <c>FutureWarning</c> equal to <paramref name="expected" />.</summary>
    private static void AssertWarns(Action action, string expected)
    {
        var (message, category) = CaptureSingleWarning(action);
        category.Should().Be(WarningCategory.FutureWarning);
        message.Should().Be(expected);
    }

#pragma warning disable CS0618  // These tests exercise the deprecated surface on purpose.
    [TestMethod]
    public void DeprecatedFloatDistributionAliases_Warn_Verbatim()
    {
        AssertWarns(() => new UniformDistribution(0, 1),
            "UniformDistribution has been deprecated in v3.0.0. This feature will be removed in v6.0.0. " +
            "See https://github.com/optuna/optuna/releases/tag/v3.0.0. " +
            "Use :class:`~optuna.distributions.FloatDistribution` instead.");
        AssertWarns(() => new LogUniformDistribution(1, 2),
            "LogUniformDistribution has been deprecated in v3.0.0. This feature will be removed in v6.0.0. " +
            "See https://github.com/optuna/optuna/releases/tag/v3.0.0. " +
            "Use :class:`~optuna.distributions.FloatDistribution` instead.");
        AssertWarns(() => new DiscreteUniformDistribution(0, 1, 0.1),
            "DiscreteUniformDistribution has been deprecated in v3.0.0. This feature will be removed in v6.0.0. " +
            "See https://github.com/optuna/optuna/releases/tag/v3.0.0. " +
            "Use :class:`~optuna.distributions.FloatDistribution` instead.");
    }

    [TestMethod]
    public void DeprecatedIntDistributionAliases_Warn_Verbatim()
    {
        AssertWarns(() => new IntUniformDistribution(0, 10),
            "IntUniformDistribution has been deprecated in v3.0.0. This feature will be removed in v6.0.0. " +
            "See https://github.com/optuna/optuna/releases/tag/v3.0.0. " +
            "Use :class:`~optuna.distributions.IntDistribution` instead.");
        AssertWarns(() => new IntLogUniformDistribution(1, 10),
            "IntLogUniformDistribution has been deprecated in v3.0.0. This feature will be removed in v6.0.0. " +
            "See https://github.com/optuna/optuna/releases/tag/v3.0.0. " +
            "Use :class:`~optuna.distributions.IntDistribution` instead.");
    }

    [TestMethod]
    public void DeprecatedSuggestAliases_Warn_Verbatim()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(0));
        var trial = study.Ask();

        AssertWarns(() => trial.SuggestUniform("a", 0, 1),
            "suggest_uniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. " +
            "See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float instead.");
        AssertWarns(() => trial.SuggestLogUniform("b", 1, 2),
            "suggest_loguniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. " +
            "See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float(..., log=True) instead.");
        AssertWarns(() => trial.SuggestDiscreteUniform("c", 0, 1, 0.1),
            "suggest_discrete_uniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. " +
            "See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float(..., step=...) instead.");
    }

    [TestMethod]
    public void DeprecatedSetSystemAttr_Warns_Verbatim_On_Every_Trial_Kind_And_Study()
    {
        const string expected =
            "set_system_attr has been deprecated in v3.1.0. This feature will be removed in v5.0.0. " +
            "See https://github.com/optuna/optuna/releases/tag/v3.1.0.";

        var study = Optuna.CreateStudy(sampler: new RandomSampler(0));
        var trial = study.Ask();
        var frozen = FrozenTrial.CreateTrial(value: 1.0);
        var fixedTrial = new FixedTrial(new OrderedDict<string, object?>());

        AssertWarns(() => trial.SetSystemAttr("k", "v"), expected);
        AssertWarns(() => frozen.SetSystemAttr("k", "v"), expected);
        AssertWarns(() => fixedTrial.SetSystemAttr("k", "v"), expected);
        AssertWarns(() => study.SetSystemAttr("k", "v"), expected);
    }

    [TestMethod]
    public void DeprecatedSystemAttrs_Warns_Verbatim_On_Trial_And_Study_Only()
    {
        const string expected =
            "system_attrs has been deprecated in v3.1.0. This feature will be removed in v5.0.0. " +
            "See https://github.com/optuna/optuna/releases/tag/v3.1.0.";

        var study = Optuna.CreateStudy(sampler: new RandomSampler(0));
        var trial = study.Ask();

        // Optuna deprecates `Trial.system_attrs` and `Study.system_attrs` specifically.
        AssertWarns(() => { _ = trial.SystemAttrs; }, expected);
        AssertWarns(() => { _ = study.SystemAttrs; }, expected);

        // FrozenTrial.system_attrs and FixedTrial.system_attrs are NOT deprecated in Optuna, so they
        // must stay silent.
        var frozen = FrozenTrial.CreateTrial(value: 1.0);
        var fixedTrial = new FixedTrial(new OrderedDict<string, object?>());
        var stray = new List<string>();
        void Handler(string message, WarningCategory category) => stray.Add(message);
        OptunaWarnings.WarningRaised += Handler;
        try
        {
            _ = frozen.SystemAttrs;
            _ = fixedTrial.SystemAttrs;
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }
        stray.Should().BeEmpty("FrozenTrial/FixedTrial.system_attrs are not deprecated in Optuna");
    }
#pragma warning restore CS0618
}
