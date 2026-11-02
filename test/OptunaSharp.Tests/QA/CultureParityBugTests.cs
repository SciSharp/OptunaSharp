using System;
using System.Collections.Generic;
using System.Globalization;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Pruners;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs that depend on the current culture, found by the 2026-11-02 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-02.md</c>, D33: every probe re-run under tr-TR, de-DE, ar-SA, he-IL, sv-SE and
///     fa-IR). Python's messages never depend on the locale, so each expected text is Optuna 5.0.0's on any
///     machine; each test fails today and passes once fixed, at which point its <c>[OpenBugs]</c> attribute should be
///     removed.
/// </summary>
[TestClass]
public class CultureParityBugTests
{
    /// <summary>
    ///     Runs <paramref name="action" /> with the current thread's culture set to <paramref name="cultureName" /> and
    ///     restores the previous cultures afterwards, so one test cannot leak its culture into another.
    /// </summary>
    /// <typeparam name="T">The result type of the action.</typeparam>
    /// <param name="cultureName">An ICU culture name such as <c>sv-SE</c>.</param>
    /// <param name="action">The code to run under that culture.</param>
    /// <returns>Whatever <paramref name="action" /> returns.</returns>
    /// <exception cref="CultureNotFoundException">The culture is unknown (for example under invariant globalization).</exception>
    private static T WithCulture<T>(string cultureName, Func<T> action)
    {
        var saved = CultureInfo.CurrentCulture;
        var savedUi = CultureInfo.CurrentUICulture;
        var culture = CultureInfo.GetCultureInfo(cultureName);
        CultureInfo.CurrentCulture = culture;
        CultureInfo.CurrentUICulture = culture;
        try
        {
            return action();
        }
        finally
        {
            CultureInfo.CurrentCulture = saved;
            CultureInfo.CurrentUICulture = savedUi;
        }
    }

    /// <summary>
    ///     QA-33: integers in Core messages must print with an ASCII minus sign, as Python's f-strings do; the port
    ///     interpolates <see cref="long" /> and <see cref="int" /> values with the current culture, whose negative sign
    ///     is <c>U+2212</c> under sv-SE, nb-NO and fi-FI, <c>U+200E</c> + <c>-</c> under he-IL and <c>U+061C</c> +
    ///     <c>-</c> under ar-SA.
    /// </summary>
    /// <remarks>
    ///     Found by re-running the probes of this QA run under six cultures: floats are formatted through
    ///     <c>PythonRepr</c> and stay invariant, but plain <c>$"{low}"</c> interpolations of integers do not. Examples:
    ///     the int-step adjustment warning (<c>[−3, 4]</c> under sv-SE), every "cannot be negative, but got ...=-1"
    ///     validation message, and <c>FrozenTrial</c>'s repr (<c>number=؜-1</c> under ar-SA). The invisible marks make
    ///     the text differ from Python's even where it looks the same, and break log scraping and exact-message tests
    ///     on those machines. The timeline warning's datetimes (QA-32) are the same class.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void NegativeIntegersInMessages_UseAsciiMinusUnderAnyCulture()
    {
        var warnings = new List<string>();
        void Handler(string message, WarningCategory category) => warnings.Add(message);
        OptunaWarnings.WarningRaised += Handler;
        try
        {
            WithCulture("sv-SE", () => new IntDistribution(-3, 4, step: 2));
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }
        warnings.Should().ContainSingle().Which.Should().Be(
            "The distribution is specified by [-3, 4] and step=2, but the range is not divisible by `step`. "
            + "It will be replaced with [-3, 3].");

        var error = WithCulture("ar-SA", () =>
        {
            try
            {
                _ = new MedianPruner(nStartupTrials: -1);
                return null;
            }
            catch (ValueError e)
            {
                return e.Message;
            }
        });
        error.Should().Be("Number of startup trials cannot be negative, but got n_startup_trials=-1.");
    }
}
