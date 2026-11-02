using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs in the samplers, found by the 2026-11-02 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-02.md</c>). Every expected value is what Optuna 5.0.0 returns on
///     CPython 3.12 for the same seed and objective; each test fails today and passes once the divergence
///     is fixed, at which point its <c>[OpenBugs]</c> attribute should be removed.
/// </summary>
[TestClass]
public class SamplerParityBugTests
{
    /// <summary>
    ///     QA-11: <see cref="BruteForceSampler" /> must enumerate a stepped float range with Python's correctly
    ///     rounded <c>float(Decimal)</c>; the port's <c>(double)decimal</c> puts 7% of candidate values one ULP
    ///     off whenever the lower bound is not a short decimal.
    /// </summary>
    /// <remarks>
    ///     Optuna's <c>_enumerate_candidates</c> walks <c>Decimal(str(low)) + k * Decimal(str(step))</c> and
    ///     converts each value with <c>float()</c>. Fed Python's own <c>(low, high, step)</c>, the port's
    ///     enumeration differs in 4,530 of 17,030 fuzzed ranges (60,839 of 856,670 values). Same root cause as
    ///     QA-2, different component: here it changes the parameter values a brute-force search tries. Python
    ///     5.0.0 with seed 0 tries exactly the nine values below; the port tries <c>7.607999463635719</c> in
    ///     place of <c>7.607999463635718</c>.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void BruteForce_SteppedFloatCandidates_MatchPythonDecimalEnumeration()
    {
        var study = Optuna.CreateStudy(sampler: new BruteForceSampler(seed: 0));
        study.Optimize(t => t.SuggestFloat("x", 3.6079994636357178, 11.607999463635718, step: 1.0), nTrials: 20);

        var tried = study.Trials.Select(t => PythonRepr.FloatRepr((double)t.Params["x"]!)).OrderBy(s => s, System.StringComparer.Ordinal).ToList();
        tried.Should().Equal(
            "10.607999463635718", "11.607999463635718", "3.6079994636357178", "4.607999463635718", "5.607999463635718",
            "6.607999463635718", "7.607999463635718", "8.607999463635718", "9.607999463635718");
    }
}
