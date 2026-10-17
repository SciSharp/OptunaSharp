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
///     including Optuna's Python member names (<c>suggest_uniform</c>, …) and its raw
///     reStructuredText role markup — so this guards byte-for-byte parity and catches any drift in
///     <see cref="DeprecationMessages" />. The <c>system_attrs</c>/<c>set_system_attr</c> members reached
///     their removal version in Optuna 5.0.0 (#6834); their surface is pinned by
///     <see cref="SystemAttrMembers_Match_Optuna_5_0_0_Surface" /> instead.
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

    /// <summary>
    ///     Optuna 5.0.0 (#6834) deleted the v3.1.0-deprecated <c>set_system_attr</c>/<c>system_attrs</c> from
    ///     <c>Study</c>, <c>Trial</c> and the abstract <c>BaseTrial</c>, and <c>set_system_attr</c> from
    ///     <c>FrozenTrial</c>/<c>FixedTrial</c>, while keeping the read-only <c>system_attrs</c> of the latter
    ///     two. Pins that exact public surface, so neither a resurrected member nor a lost getter goes unnoticed.
    /// </summary>
    [TestMethod]
    public void SystemAttrMembers_Match_Optuna_5_0_0_Surface()
    {
        const System.Reflection.BindingFlags Public =
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance;

        foreach (Type removedBoth in new[] { typeof(Study), typeof(Trial), typeof(BaseTrial) })
        {
            removedBoth.GetProperty("SystemAttrs", Public).Should().BeNull($"{removedBoth.Name}.system_attrs was removed");
            removedBoth.GetMethod("SetSystemAttr", Public).Should().BeNull($"{removedBoth.Name}.set_system_attr was removed");
        }
        foreach (Type keptGetter in new[] { typeof(FrozenTrial), typeof(FixedTrial) })
        {
            keptGetter.GetProperty("SystemAttrs", Public).Should().NotBeNull($"{keptGetter.Name}.system_attrs is kept");
            keptGetter.GetMethod("SetSystemAttr", Public).Should().BeNull($"{keptGetter.Name}.set_system_attr was removed");
        }
    }

    /// <summary>
    ///     The <c>system_attrs</c> getters Optuna 5.0.0 keeps (<c>FrozenTrial</c>, <c>FixedTrial</c>) were never
    ///     deprecated, so reading them must raise no warning at all.
    /// </summary>
    [TestMethod]
    public void SystemAttrs_On_FrozenTrial_And_FixedTrial_Stays_Silent()
    {
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
