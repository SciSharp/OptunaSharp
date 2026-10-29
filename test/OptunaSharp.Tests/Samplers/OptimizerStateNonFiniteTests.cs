using System;
using System.Linq;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.CmaEs;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Regression tests for pre-release review item H1 (2026-10-26): one ±inf objective value made every
///     later ask of a study fail for the samplers that persist their optimizer as JSON, because
///     System.Text.Json's default options refuse to write non-finite doubles. Optuna pickles the same
///     state and keeps going, so a diverged run that reports <c>inf</c> must not end the study here either.
///     (Since 2026-10-29 the CMA-ES family stores Optuna's pickle itself; it is still covered here, end to end.)
///     <para>Each sampler runs long enough for the infinite values to be told to its optimizer and for the
///     resulting state to be written to, and restored from, the trial system attributes several times.</para>
/// </summary>
[TestClass]
public class OptimizerStateNonFiniteTests
{
    /// <summary>
    ///     The persisted-state encoding itself: ±inf and NaN survive a write/read cycle with their exact
    ///     bits, every finite value (including <c>-0.0</c> and the subnormal minimum) is untouched, and the
    ///     non-finite values appear as named JSON literals.
    /// </summary>
    [TestMethod]
    public void OptimizerStateJson_RoundTripsNonFiniteDoubles_BitExactly()
    {
        double[] values =
        {
            double.PositiveInfinity, double.NegativeInfinity, double.NaN, 1.5, -0.0, double.Epsilon, double.MaxValue,
        };
        var state = new CmaState { NDim = 1, FunhistValues = values, Mean = new[] { double.NegativeInfinity } };

        string json = OptimizerStateJson.Serialize(state);
        var restored = OptimizerStateJson.Deserialize<CmaState>(json);

        json.Should().Contain("\"Infinity\"").And.Contain("\"-Infinity\"").And.Contain("\"NaN\"");
        json.Should().NotContain("\n", "the samplers split '<tag>\\n<json>' on the first newline");
        restored.FunhistValues.Select(BitConverter.DoubleToInt64Bits)
            .Should().Equal(values.Select(v => BitConverter.DoubleToInt64Bits(double.IsNaN(v) ? double.NaN : v)));
        restored.Mean.Should().Equal(double.NegativeInfinity);
    }

    /// <summary>
    ///     A state written before the fix (default System.Text.Json options, finite values only) is read
    ///     back unchanged, so studies persisted by earlier versions keep restoring their optimizer.
    /// </summary>
    [TestMethod]
    public void OptimizerStateJson_ReadsStatesWrittenWithDefaultOptions()
    {
        var state = new CmaState
        {
            NDim = 2, Sigma0 = 0.25, Popsize = 6, G = 3, Mean = new[] { 0.5, -0.125 }, Sigma = 0.3,
            C = new[] { new[] { 1.0, 0.1 }, new[] { 0.1, 2.0 } }, FunhistValues = new[] { 3.5, 7.25 },
        };

        string legacyJson = JsonSerializer.Serialize(state);
        var restored = OptimizerStateJson.Deserialize<CmaState>(legacyJson);

        OptimizerStateJson.Serialize(restored).Should().Be(legacyJson,
            "finite states encode identically with and without named floating-point literals");
    }

    /// <summary>A bare JSON <c>null</c> payload fails with a JSON error instead of yielding a null state.</summary>
    [TestMethod]
    public void OptimizerStateJson_NullPayload_ThrowsJsonException()
    {
        Action read = () => OptimizerStateJson.Deserialize<CmaState>("null");

        read.Should().Throw<JsonException>().WithMessage("*CmaState*null*");
    }

    /// <summary>
    ///     End to end per sampler: every odd-numbered trial reports +inf and trial 10 reports −inf (an unstable
    ///     objective on a minimized study). Every trial must complete, and the samplers whose persisted state
    ///     carries fitness values must actually have stored a non-finite value — proof that the write/restore
    ///     path ran with it instead of the infinite trials being silently skipped.
    ///     <para>Half the trials are infinite on purpose: DE and PSO never adopt a non-finite value as an
    ///     incumbent, so their state holds ±inf only while a population slot has seen nothing finite yet. With
    ///     every odd trial infinite, the even-indexed slots of their populations stay infinite across
    ///     generations, which is exactly the state that used to fail to serialize.</para>
    /// </summary>
    /// <param name="samplerName">Short name of the sampler variant under test.</param>
    /// <param name="stateHoldsFitness">Whether the sampler's optimizer state records objective values.</param>
    [TestMethod]
    [DataRow("cma", true)]
    [DataRow("sepcma", true)]
    [DataRow("cmawm", true)]
    [DataRow("lmcma", true)]
    [DataRow("vkdcma", false)]
    [DataRow("de", true)]
    [DataRow("pso", true)]
    [DataRow("openaies", false)]
    [DataRow("snes", false)]
    [DataRow("xnes", false)]
    public void InfiniteObjectiveValues_DoNotBreakLaterTrials(string samplerName, bool stateHoldsFitness)
    {
        var study = Optuna.CreateStudy(sampler: MakeSampler(samplerName));

        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            // The margin variant needs a discrete dimension; the others get a second continuous one.
            double y = samplerName == "cmawm" ? t.SuggestInt("y", -5, 5) : t.SuggestFloat("y", -5, 5);
            return t.Number switch
            {
                10 => double.NegativeInfinity,
                _ when t.Number % 2 == 1 => double.PositiveInfinity,
                _ => x * x + y * y,
            };
        }, nTrials: 60);

        study.Trials.Should().HaveCount(60);
        study.Trials.Should().OnlyContain(t => t.State == TrialState.Complete,
            "an infinite objective value must not make any later ask or tell fail");
        if (stateHoldsFitness)
        {
            study.Trials.Should().Contain(t => StoredStateHoldsInfinity(t, samplerName),
                "the optimizer state written after the infinite trials must carry them");
        }
    }

    /// <summary>
    ///     Whether a trial's stored optimizer state holds an infinite fitness value. The JSON states keep it as a named
    ///     literal; the CMA-ES family stores Optuna's pickle instead (review item P3-2), whose <c>_funhist_values</c>
    ///     array keeps it as an IEEE value, so those states are decoded and that array is inspected.
    /// </summary>
    /// <param name="trial">A finished trial.</param>
    /// <param name="samplerName">The DataRow sampler name.</param>
    /// <returns>True when the trial stored a state that holds +inf or -inf.</returns>
    private static bool StoredStateHoldsInfinity(FrozenTrial trial, string samplerName)
    {
        string? prefix = samplerName switch { "cma" => "cma:", "sepcma" => "sepcma:", "cmawm" => "cmawm:", _ => null };
        if (prefix is null)
            return trial.SystemAttrs.Values.OfType<string>().Any(s => s.Contains("Infinity"));
        int chunks = trial.SystemAttrs.Keys.Count(k => k.StartsWith(prefix + "optimizer:", StringComparison.Ordinal));
        if (chunks == 0)
            return false;
        string hex = string.Concat(Enumerable.Range(0, chunks).Select(i => (string)trial.SystemAttrs[$"{prefix}optimizer:{i}"]!));
        var state = (PyInstance)PythonPickle.Loads(CmaPickle.FromHexString(hex))!;
        // CMAwM keeps the funhist on its inner CMA.
        if (state.TryGet("_cma", out var inner))
            state = (PyInstance)inner!;
        return ((NpArray)state.Get("_funhist_values")!).Doubles!.Any(double.IsInfinity);
    }

    /// <summary>Builds the seeded sampler variant named by a test row.</summary>
    /// <param name="name">One of the DataRow sampler names.</param>
    /// <returns>A fresh, seeded sampler instance.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The name is not one of the covered samplers.</exception>
    private static BaseSampler MakeSampler(string name) => name switch
    {
        "cma" => new CmaEsSampler(seed: 7),
        "sepcma" => new CmaEsSampler(seed: 7, useSeparableCma: true),
        "cmawm" => new CmaEsSampler(seed: 7, withMargin: true),
        "lmcma" => new LmCmaSampler(seed: 7),
        "vkdcma" => new VkdCmaSampler(seed: 7),
        "de" => new DifferentialEvolutionSampler(seed: 7),
        "pso" => new PsoSampler(seed: 7),
        "openaies" => new OpenAiEsSampler(seed: 7),
        "snes" => new SnesSampler(seed: 7),
        "xnes" => new XnesSampler(seed: 7),
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "Unknown sampler variant."),
    };
}
