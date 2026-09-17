using System;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Tests for the native <see cref="RandomEmbeddingSampler" /> (REMBO, Wang et al. 2016). REMBO has no
///     <c>cmaes</c>/Optuna/OptunaHub counterpart, so — like <c>TurboSamplerTests</c> — these are behavioral
///     gates. They mirror the paper's own findings: a <b>single</b> random embedding is fragile (its box may
///     miss the optimum, §3.2), so the headline convergence tests use <b>interleaved</b> embeddings (§3.1),
///     which recover a low-effective-dimension optimum hidden in a much larger space and beat plain search.
///     Cheaper mechanics tests (bounds, mixed types, determinism, fallbacks) use small budgets, several with
///     an optimum at the box center — which the embedding reaches at the latent origin regardless of the
///     random matrix. Everything is seeded and therefore deterministic, so thresholds are firm.
/// </summary>
[TestClass]
public class RandomEmbeddingSamplerTests
{
    // An off-center, d_e-effective-dimension quadratic hidden inside D real parameters on [-5, 5]: only the
    // first `effective` coordinates move the objective, and their optimum is away from the box center — the
    // non-trivial case where the embedding must actually capture the optimum.
    private static double Embedded(Trial t, int D, int effective)
    {
        var target = new[] { 2.0, -3.0, 1.5, -1.0, 4.0, -2.5 };
        double sum = 0;
        for (int i = 0; i < D; i++)
        {
            double x = t.SuggestFloat($"x{i}", -5, 5);
            if (i < effective) { double d = x - target[i]; sum += d * d; }
        }
        return sum;
    }

    // A sphere whose optimum sits at the box center (x = 0): reachable at the latent origin, so a single
    // embedding solves it reliably — useful for fast mechanics tests that still exercise real convergence.
    private static double CenteredSphere(Trial t, int D, int effective)
    {
        double sum = 0;
        for (int i = 0; i < D; i++)
        {
            double x = t.SuggestFloat($"x{i}", -5, 5);
            if (i < effective) sum += x * x;
        }
        return sum;
    }

    [TestMethod]
    public void Sampler_RecoversEmbeddedOptimum_AndBeatsRandomSearch()
    {
        // 2 effective dimensions buried in 25, optimum off-center: interleaved embeddings (§3.1) reliably
        // capture it and the inner GP drives the gap near zero — beating undirected search on the full space.
        var rembo = Optuna.CreateStudy(sampler: new RandomEmbeddingSampler(nEmbeddingDims: 3, nEmbeddings: 3, seed: 42));
        rembo.Optimize(t => Embedded(t, D: 25, effective: 2), nTrials: 240);

        var random = Optuna.CreateStudy(sampler: new RandomSampler(seed: 42));
        random.Optimize(t => Embedded(t, D: 25, effective: 2), nTrials: 240);

        rembo.BestValue.Should().BeLessThan(0.1, "interleaved random embeddings recover the low-effective-dimension optimum");
        rembo.BestValue.Should().BeLessThan(random.BestValue, "REMBO's low-dimensional GP outperforms undirected search on the full space");
    }

    [TestMethod]
    public void Sampler_Maximize_Works()
    {
        var study = Optuna.CreateStudy(
            direction: StudyDirection.Maximize, sampler: new RandomEmbeddingSampler(nEmbeddingDims: 3, nEmbeddings: 3, seed: 42));
        study.Optimize(t => -Embedded(t, D: 20, effective: 2), nTrials: 240);
        study.BestValue.Should().BeGreaterThan(-0.2, "direction flipping lets REMBO maximize the negated objective");
    }

    [TestMethod]
    public void Sampler_SameSeed_IsReproducible()
    {
        double Run()
        {
            var study = Optuna.CreateStudy(sampler: new RandomEmbeddingSampler(nEmbeddingDims: 3, seed: 7));
            study.Optimize(t => Embedded(t, D: 15, effective: 2), nTrials: 50);
            return study.BestValue;
        }
        Run().Should().Be(Run(), "a fixed seed reproduces the trajectory under sequential optimization");
    }

    [TestMethod]
    public void Sampler_RespectsBounds()
    {
        var study = Optuna.CreateStudy(sampler: new RandomEmbeddingSampler(nEmbeddingDims: 3, seed: 3));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -2, 3);
            double y = t.SuggestFloat("y", 1, 4);
            return (x - 1) * (x - 1) + (y - 2) * (y - 2);
        }, nTrials: 60);

        foreach (var tr in study.Trials)
        {
            ((double)tr.Params["x"]!).Should().BeInRange(-2, 3);
            ((double)tr.Params["y"]!).Should().BeInRange(1, 4);
        }
    }

    [TestMethod]
    public void Sampler_MixedIntFloatStepLog_Works()
    {
        // Every parameter kind must be produced in range through the embedding's de-normalization: a float
        // domain, an integer domain, a stepped float grid, and a log-scaled float all map from the same
        // latent point. This checks the round-trip and bounds, not convergence (four coupled effective
        // dimensions through one latent vector is a harder search, exercised elsewhere).
        var study = Optuna.CreateStudy(sampler: new RandomEmbeddingSampler(nEmbeddingDims: 4, seed: 3));
        study.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            long b = t.SuggestInt("b", -8, 8);
            double c = t.SuggestFloat("c", -4, 4, step: 0.5);
            double d = t.SuggestFloat("d", 1e-3, 1e3, log: true);
            double logd = Math.Log10(d);
            return a * a + b * b + c * c + logd * logd;
        }, nTrials: 100);

        foreach (var tr in study.Trials)
        {
            ((double)tr.Params["a"]!).Should().BeInRange(-5, 5);
            ((long)tr.Params["b"]!).Should().BeInRange(-8, 8);
            ((double)tr.Params["c"]!).Should().BeInRange(-4, 4);
            (Math.IEEERemainder((double)tr.Params["c"]! * 2, 1.0)).Should().BeApproximately(0.0, 1e-9, "the step grid (0.5) is honored");
            ((double)tr.Params["d"]!).Should().BeInRange(1e-3, 1e3);
        }
    }

    [TestMethod]
    public void Sampler_CategoricalParameter_FallsBackToIndependent()
    {
        // A categorical cannot be recovered from a single projected scalar, so it is sampled independently
        // while the continuous parameters ride the embedding. The continuous optimum is the box center.
        var study = Optuna.CreateStudy(sampler: new RandomEmbeddingSampler(nEmbeddingDims: 3, seed: 3));
        study.Optimize(t =>
        {
            string pick = t.SuggestCategorical("mode", new[] { "u", "v", "w" });
            double bonus = pick == "v" ? 0.0 : 1.0;
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return x * x + y * y + bonus;
        }, nTrials: 120);

        study.BestValue.Should().BeLessThan(1.5, "continuous params converge while the categorical is independent");
    }

    [TestMethod]
    public void Sampler_TpeInnerSampler_Works()
    {
        // The inner optimizer is pluggable; TPE is the cheaper pure-managed alternative to the default GP.
        var study = Optuna.CreateStudy(sampler: new RandomEmbeddingSampler(
            nEmbeddingDims: 3, innerSampler: new TPESampler(seed: 3), seed: 3));
        study.Optimize(t => CenteredSphere(t, D: 15, effective: 2), nTrials: 120);
        study.BestValue.Should().BeLessThan(0.5, "REMBO drives an arbitrary inner sampler in the latent box");
    }

    [TestMethod]
    public void Sampler_PersistsLatentPointInSystemAttrs()
    {
        var study = Optuna.CreateStudy(sampler: new RandomEmbeddingSampler(nEmbeddingDims: 3, seed: 5));
        study.Optimize(t => CenteredSphere(t, D: 10, effective: 2), nTrials: 20);

        // Every embedded trial stores its d-dimensional latent point so the shadow study can be rebuilt.
        foreach (var tr in study.Trials)
        {
            tr.SystemAttrsDict.Should().ContainKey("rembo:latent");
            ((System.Collections.IEnumerable)tr.SystemAttrsDict["rembo:latent"]!)
                .Cast<object?>().Count().Should().Be(3, "the stored latent point has one coordinate per embedding dimension");
        }
    }

    [TestMethod]
    public void Sampler_MultiObjective_Raises()
    {
        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomEmbeddingSampler(nEmbeddingDims: 3, seed: 1));
        Action act = () => study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            return new[] { x * x, (x - 2) * (x - 2) };
        }, nTrials: 5);
        act.Should().Throw<ValueError>("REMBO is single-objective only");
    }

    [TestMethod]
    public void Constructor_RejectsInvalidDimensions()
    {
        Action badDim = () => new RandomEmbeddingSampler(nEmbeddingDims: 0);
        Action badK = () => new RandomEmbeddingSampler(nEmbeddingDims: 3, nEmbeddings: 0);
        badDim.Should().Throw<ArgumentException>("the embedding dimension must be at least 1");
        badK.Should().Throw<ArgumentException>("the number of embeddings must be at least 1");
    }
}
