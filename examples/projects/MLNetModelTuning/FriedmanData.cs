using Microsoft.ML.Data;
using NumSharp;

namespace MLNetModelTuning;

/// <summary>One row of the Friedman #1 data, in the shape ML.NET loads: a feature vector and a label.</summary>
public sealed class FriedmanRow
{
    /// <summary>The ten features, each uniform on [0, 1); only the first five influence the label.</summary>
    [VectorType(FriedmanData.FeatureCount)]
    public float[] Features { get; set; } = [];

    /// <summary>The regression target.</summary>
    public float Label { get; set; }
}

/// <summary>
///     Friedman's #1 regression benchmark: <c>y = 10 sin(π x₀ x₁) + 20 (x₂ − 0.5)² + 10 x₃ + 5 x₄ + noise·ε</c>,
///     with ten uniform features, five of which are pure noise. It has a known structure (a non-linear
///     interaction, a quadratic, two linear terms), which a tree ensemble has to learn.
/// </summary>
/// <remarks>
///     The rows are drawn with NumSharp's <c>np.random.RandomState</c>, whose MT19937 stream is NumPy's, in the
///     order scikit-learn draws them. So <see cref="Generate" /> returns the same rows as
///     <c>sklearn.datasets.make_friedman1(n_samples, noise=noise, random_state=seed)</c>, rounded to the
///     single precision ML.NET trains in.
/// </remarks>
public static class FriedmanData
{
    /// <summary>The number of features per row, as in scikit-learn's default.</summary>
    public const int FeatureCount = 10;

    /// <summary>Draws the rows.</summary>
    /// <param name="samples">How many rows to draw.</param>
    /// <param name="noise">The standard deviation of the Gaussian noise added to each label.</param>
    /// <param name="seed">The seed of the random stream, scikit-learn's <c>random_state</c>.</param>
    /// <returns>The rows, in draw order.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="samples" /> is below 1, or <paramref name="noise" /> is negative.</exception>
    public static FriedmanRow[] Generate(int samples, double noise, int seed)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(samples, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(noise);

        NumPyRandom generator = np.random.RandomState(seed);

        // the same two draws, in the same order, as make_friedman1: all of X first, then one ε per row.
        // NumSharp pools NDArray buffers, so both arrays are disposed as soon as they are copied out
        double[] x;
        double[] epsilon;
        using (NDArray drawnX = generator.uniform(0.0, 1.0, new Shape(samples, FeatureCount)))
            x = drawnX.ToArray<double>();
        using (NDArray drawnEpsilon = generator.standard_normal(new Shape(samples)))
            epsilon = drawnEpsilon.ToArray<double>();

        var rows = new FriedmanRow[samples];
        for (int i = 0; i < samples; i++)
        {
            ReadOnlySpan<double> row = x.AsSpan(i * FeatureCount, FeatureCount);

            // term order as in scikit-learn, so the double result matches NumPy's before the cast to float
            double label = 10 * Math.Sin(Math.PI * row[0] * row[1])
                           + 20 * Math.Pow(row[2] - 0.5, 2.0)
                           + 10 * row[3]
                           + 5 * row[4]
                           + noise * epsilon[i];

            float[] features = new float[FeatureCount];
            for (int j = 0; j < FeatureCount; j++)
                features[j] = (float)row[j];
            rows[i] = new FriedmanRow { Features = features, Label = (float)label };
        }

        return rows;
    }
}
