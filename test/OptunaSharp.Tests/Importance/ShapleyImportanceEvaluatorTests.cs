using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Importance.Shap;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Importance;

/// <summary>
///     Byte-exact regression tests for <see cref="ShapleyImportanceEvaluator" /> (Core,
///     <c>OptunaSharp.Importance.Shap</c>) that need NO Python at test time: they replay a committed
///     oracle (<c>oracle/shap_oracle.json</c>) produced by the real
///     <c>optuna_integration.shap.ShapleyImportanceEvaluator</c> (shap 0.52.0 + scikit-learn 1.8.0),
///     and assert the importances match <b>bit-for-bit</b> (raw IEEE-754). The live differential gate
///     lives in <c>OptunaSharp.Tests.Interop.ShapleyImportanceParityTests</c>.
///
///     <para>Bit-exact parity is host-pinned to x64/x86 (the surrogate's float32 forest accumulation
///     isn't arm64-portable); the replay reports Inconclusive on other architectures, matching the
///     interop suite's host-pin model.</para>
/// </summary>
[TestClass]
public class ShapleyImportanceEvaluatorTests
{
    private static bool BitExactHost =>
        RuntimeInformation.ProcessArchitecture is Architecture.X64 or Architecture.X86;

    // The oracle objective (must be identical to oracle/gen_shap_oracle.py's `objective`).
    private static double Objective(Trial t)
    {
        double x1 = t.SuggestFloat("x1", 0.1, 3);
        double x2 = t.SuggestFloat("x2", 0.1, 3, log: true);
        long x3 = t.SuggestInt("x3", 2, 4, log: true);
        double x4 = t.SuggestCategorical("x4", new[] { 0.1, 1.0, 10.0 });
        return x1 + x2 * x3 + x4;
    }

    private sealed record ExpectedParam(string Param, long Bits, double Value);
    private sealed record Scenario(
        string Name, int SamplerSeed, int NTrials, int EvalSeed, int NTrees, int MaxDepth,
        IReadOnlyList<ExpectedParam> Expected);

    private static IReadOnlyList<Scenario> LoadOracle()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "oracle", "shap_oracle.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.EnumerateArray().Select(sc => new Scenario(
            sc.GetProperty("name").GetString()!,
            sc.GetProperty("sampler_seed").GetInt32(),
            sc.GetProperty("n_trials").GetInt32(),
            sc.GetProperty("eval_seed").GetInt32(),
            sc.GetProperty("n_trees").GetInt32(),
            sc.GetProperty("max_depth").GetInt32(),
            sc.GetProperty("expected").EnumerateArray().Select(e => new ExpectedParam(
                e.GetProperty("param").GetString()!,
                e.GetProperty("bits").GetInt64(),
                e.GetProperty("value").GetDouble())).ToList())).ToList();
    }

    [TestMethod]
    public void CommittedOracle_IsByteExact()
    {
        if (!BitExactHost)
            Assert.Inconclusive(
                "SHAP bit-exact parity is host-pinned to x64/x86; " +
                $"ProcessArchitecture={RuntimeInformation.ProcessArchitecture}.");

        var scenarios = LoadOracle();
        scenarios.Should().NotBeEmpty("the committed oracle must contain scenarios");

        foreach (var sc in scenarios)
        {
            var study = Optuna.CreateStudy(sampler: new RandomSampler(sc.SamplerSeed));
            study.Optimize(Objective, nTrials: sc.NTrials);

            var importances = new ShapleyImportanceEvaluator(
                nTrees: sc.NTrees, maxDepth: sc.MaxDepth, seed: sc.EvalSeed).Evaluate(study);

            var actual = importances.ToList();
            actual.Should().HaveCount(sc.Expected.Count, $"[{sc.Name}] parameter count");

            for (int i = 0; i < sc.Expected.Count; i++)
            {
                var exp = sc.Expected[i];
                actual[i].Key.Should().Be(exp.Param,
                    $"[{sc.Name}] descending-importance ordering at position {i}");
                BitConverter.DoubleToInt64Bits(actual[i].Value).Should().Be(exp.Bits,
                    $"[{sc.Name}] importance of {exp.Param} must be byte-exact vs SHAP 0.52.0 " +
                    $"(expected {exp.Value:G17}, got {actual[i].Value:G17})");
            }
        }
    }

    [TestMethod]
    public void EmptyParams_ReturnsEmptyDict()
    {
        // Optuna: "If params is specified as an empty list, an empty dictionary is returned."
        var study = Optuna.CreateStudy(sampler: new RandomSampler(0));
        study.Optimize(Objective, nTrials: 5);

        var importances = new ShapleyImportanceEvaluator(seed: 0)
            .Evaluate(study, @params: new List<string>());

        importances.Should().BeEmpty();
    }

    // [Misaligned] Optuna decorates the class with @experimental_class("3.0.0"), which raises an
    // ExperimentalWarning on construction. OptunaSharp targets the Optuna 5.x surface and treats this
    // evaluator as stable, so no ExperimentalWarning is emitted — this guards that divergence.
    [TestMethod]
    [Misaligned]
    public void Constructor_DoesNotRaiseExperimentalWarning()
    {
        var raised = new List<(string Message, WarningCategory Category)>();
        void Handler(string m, WarningCategory c) => raised.Add((m, c));

        OptunaWarnings.WarningRaised += Handler;
        try
        {
            _ = new ShapleyImportanceEvaluator(seed: 0);
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }

        raised.Should().NotContain(w => w.Category == WarningCategory.ExperimentalWarning);
    }

    [TestMethod]
    public void MultiObjectiveWithoutTarget_ThrowsValueError()
    {
        var study = Optuna.CreateStudy(
            sampler: new RandomSampler(0),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.Optimize(t =>
        {
            double x1 = t.SuggestFloat("x1", 0.1, 3);
            double x2 = t.SuggestFloat("x2", 0.1, 3, log: true);
            long x3 = t.SuggestInt("x3", 2, 4, log: true);
            double x4 = t.SuggestCategorical("x4", new[] { 0.1, 1.0, 10.0 });
            return new[] { x1 + x2 * x3 + x4, x1 * x4 };
        }, nTrials: 5);

        Action act = () => new ShapleyImportanceEvaluator(seed: 0).Evaluate(study);

        act.Should().Throw<ValueError>()
            .WithMessage("If the `study` is being used for multi-objective*");
    }
}
