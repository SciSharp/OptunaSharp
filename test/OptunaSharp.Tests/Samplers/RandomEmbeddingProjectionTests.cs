using System;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers.RandomEmbedding;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Unit tests for <see cref="RandomEmbeddingProjection" /> — the geometric core of REMBO (Wang et al.
///     2016): the random Gaussian embedding matrix, the latent box <c>Y = [−√d, √d]^d</c>, the round-robin
///     assignment of trials to interleaved embeddings, and the box-constrained projection
///     <c>x = p_X(A·y)</c>. These exercise the deterministic, RNG-driven geometry directly (the sampler's
///     end-to-end convergence is covered separately), so they can assert exact values without the inner GP.
/// </summary>
[TestClass]
public class RandomEmbeddingProjectionTests
{
    [TestMethod]
    public void LatentSearchSpace_IsDContinuousCoordinatesOnTheBox()
    {
        var emb = new RandomEmbeddingProjection(baseSeed: 1, dim: 4, nEmbeddings: 1);
        emb.Dim.Should().Be(4);
        emb.BoxHalfWidth.Should().Be(Math.Sqrt(4));  // √d

        var space = emb.LatentSearchSpace();
        space.Count.Should().Be(4);
        for (int i = 0; i < 4; i++)
        {
            space.Should().ContainKey($"z{i}");
            var f = space[$"z{i}"].Should().BeOfType<FloatDistribution>().Subject;
            f.Low.Should().Be(-2.0);   // −√4
            f.High.Should().Be(2.0);   //  √4
            f.Log.Should().BeFalse();
        }
    }

    [TestMethod]
    public void RunForTrial_IsRoundRobinAcrossEmbeddings()
    {
        var single = new RandomEmbeddingProjection(baseSeed: 1, dim: 2, nEmbeddings: 1);
        // A single embedding always maps to run 0.
        Enumerable.Range(0, 7).Select(single.RunForTrial).Should().OnlyContain(r => r == 0);

        var three = new RandomEmbeddingProjection(baseSeed: 1, dim: 2, nEmbeddings: 3);
        new[] { 0, 1, 2, 3, 4, 5, 6 }.Select(three.RunForTrial)
            .Should().Equal(0, 1, 2, 0, 1, 2, 0);
    }

    [TestMethod]
    public void ProjectionRow_IsDeterministicAndSeedStable()
    {
        var a = new RandomEmbeddingProjection(baseSeed: 42, dim: 5, nEmbeddings: 1);
        var b = new RandomEmbeddingProjection(baseSeed: 42, dim: 5, nEmbeddings: 1);

        double[] rowA = a.ProjectionRow(0, "x");
        rowA.Length.Should().Be(5);
        // Same seed + identity ⇒ identical row (the matrix is a pure function of (seed, run, name)).
        a.ProjectionRow(0, "x").Should().Equal(rowA);
        b.ProjectionRow(0, "x").Should().Equal(rowA);
    }

    [TestMethod]
    public void ProjectionRow_DiffersByName_Run_AndSeed()
    {
        var a = new RandomEmbeddingProjection(baseSeed: 42, dim: 6, nEmbeddings: 2);
        var other = new RandomEmbeddingProjection(baseSeed: 43, dim: 6, nEmbeddings: 2);

        double[] baseRow = a.ProjectionRow(0, "x");
        a.ProjectionRow(0, "y").Should().NotEqual(baseRow, "different parameter names get independent rows");
        a.ProjectionRow(1, "x").Should().NotEqual(baseRow, "a different embedding uses a different matrix");
        other.ProjectionRow(0, "x").Should().NotEqual(baseRow, "a different base seed draws a different matrix");
    }

    [TestMethod]
    public void ProjectionRow_EntriesAreStandardNormalInDistribution()
    {
        // A coarse sanity check that the entries are ~N(0,1): a large sample's mean is near 0 and its
        // standard deviation near 1. This guards against a scaling/typo in the row draw.
        var emb = new RandomEmbeddingProjection(baseSeed: 7, dim: 2000, nEmbeddings: 1);
        double[] row = emb.ProjectionRow(0, "p");
        double mean = row.Average();
        double std = Math.Sqrt(row.Select(v => (v - mean) * (v - mean)).Average());
        mean.Should().BeApproximately(0.0, 0.1);
        std.Should().BeApproximately(1.0, 0.1);
    }

    [TestMethod]
    public void ToExternal_ZeroLatent_MapsToBoxMidpoint()
    {
        var emb = new RandomEmbeddingProjection(baseSeed: 123, dim: 3, nEmbeddings: 1);
        var zero = new double[3];

        // y = 0 ⇒ A·y = 0 ⇒ clipped to the center of the unit box ⇒ the midpoint of [low, high], regardless
        // of the (random) projection row. This pins the affine mapping independently of the RNG.
        emb.ToExternal(0, "x", new FloatDistribution(-5, 5), zero).Should().Be(0.0);
        emb.ToExternal(0, "x", new FloatDistribution(0, 10), zero).Should().Be(5.0);
    }

    [TestMethod]
    public void ToExternal_AppliesBoxProjection_ClippingToBounds()
    {
        var emb = new RandomEmbeddingProjection(baseSeed: 55, dim: 4, nEmbeddings: 1);
        var dist = new FloatDistribution(-5, 5);
        double[] row = emb.ProjectionRow(0, "x");

        // y = +c·row ⇒ raw = c·‖row‖² ≫ 1 ⇒ p_X clips to +1 ⇒ the upper bound; y = −c·row ⇒ the lower bound.
        // The de-normalization can land one ULP short of the exact bound (e.g. 4.999999999999999), which is
        // within bounds and harmless, so compare with a tiny tolerance rather than for exact equality.
        double[] yHigh = row.Select(r => r * 10.0).ToArray();
        double[] yLow = row.Select(r => r * -10.0).ToArray();
        ((double)emb.ToExternal(0, "x", dist, yHigh)!).Should().BeApproximately(5.0, 1e-9);
        ((double)emb.ToExternal(0, "x", dist, yLow)!).Should().BeApproximately(-5.0, 1e-9);
    }

    [TestMethod]
    public void ToExternal_StaysInBounds_ForArbitraryLatentPoints()
    {
        var emb = new RandomEmbeddingProjection(baseSeed: 9, dim: 4, nEmbeddings: 1);
        var f = new FloatDistribution(-2, 3);
        var g = new FloatDistribution(1e-3, 1e3, log: true);
        var n = new IntDistribution(-8, 8);

        var rng = np.random.RandomState();
        rng.seed(1u);
        for (int t = 0; t < 200; t++)
        {
            double[] y = rng.uniform(-emb.BoxHalfWidth, emb.BoxHalfWidth, new Shape(4)).ToArray<double>();
            ((double)emb.ToExternal(0, "a", f, y)!).Should().BeInRange(-2, 3);
            ((double)emb.ToExternal(0, "b", g, y)!).Should().BeInRange(1e-3, 1e3);
            ((long)emb.ToExternal(0, "c", n, y)!).Should().BeInRange(-8, 8);
        }
    }

    [TestMethod]
    public void ToExternal_IntegerDistribution_ReturnsBoxedLong()
    {
        var emb = new RandomEmbeddingProjection(baseSeed: 3, dim: 2, nEmbeddings: 1);
        object? value = emb.ToExternal(0, "i", new IntDistribution(0, 10), new double[2]);
        value.Should().BeOfType<long>();
    }

    [TestMethod]
    public void ToExternal_Categorical_ScaleAndRoundsToAChoice()
    {
        // §3.2 s(x): the projected coordinate is scaled-and-rounded into the choice set. Every latent point
        // must yield a valid choice, and the three cells of [0,1] (via engineered ±c·row projections) select
        // the low / middle / high choice — a monotone map from the projected coordinate to the choice index.
        var emb = new RandomEmbeddingProjection(baseSeed: 77, dim: 3, nEmbeddings: 1);
        var cat = new CategoricalDistribution("a", "b", "c");   // ToExternalRepr yields the choice's value (a string)
        double[] row = emb.ProjectionRow(0, "cat");

        // y = 0 → unit 0.5 → middle cell "b"; y = −c·row → unit 0 → "a"; y = +c·row → unit 1 → last cell "c".
        emb.ToExternal(0, "cat", cat, new double[3]).Should().Be("b");
        emb.ToExternal(0, "cat", cat, row.Select(r => r * -10.0).ToArray()).Should().Be("a");
        emb.ToExternal(0, "cat", cat, row.Select(r => r * 10.0).ToArray()).Should().Be("c");

        // Arbitrary latent points always land on a valid choice.
        var rng = np.random.RandomState();
        rng.seed(5u);
        for (int t = 0; t < 100; t++)
        {
            double[] y = rng.uniform(-emb.BoxHalfWidth, emb.BoxHalfWidth, new Shape(3)).ToArray<double>();
            emb.ToExternal(0, "cat", cat, y).Should().BeOneOf("a", "b", "c");
        }
    }

    [TestMethod]
    public void ToExternal_RejectsWrongLatentDimension()
    {
        var emb = new RandomEmbeddingProjection(baseSeed: 1, dim: 3, nEmbeddings: 1);
        Action act = () => emb.ToExternal(0, "x", new FloatDistribution(-1, 1), new double[2]);
        act.Should().Throw<ArgumentException>("the latent point must match the embedding dimension");
    }

    [TestMethod]
    public void Constructor_RejectsNonPositiveDimensions()
    {
        Action badDim = () => new RandomEmbeddingProjection(baseSeed: 1, dim: 0, nEmbeddings: 1);
        Action badK = () => new RandomEmbeddingProjection(baseSeed: 1, dim: 2, nEmbeddings: 0);
        badDim.Should().Throw<ArgumentException>();
        badK.Should().Throw<ArgumentException>();
    }
}
