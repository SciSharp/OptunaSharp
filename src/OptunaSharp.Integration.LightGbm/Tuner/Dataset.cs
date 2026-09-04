using System;
using System.Collections.Generic;
using NumSharp;

namespace OptunaSharp.Integration.LightGbm;

/// <summary>
///     The C# analog of <c>lightgbm.Dataset</c> — the feature matrix + optional label / weight a
///     <see cref="LightGBMTuner" /> / <see cref="LightGBMTunerCV" /> trains on. It mirrors the role of
///     <c>lgb.Dataset</c> in <c>optuna_integration.lightgbm</c>, but holds plain managed arrays (this
///     package trains through <b>ML.NET's</b> LightGBM, not the Python binding), so it is a thin data
///     holder rather than a wrapper around a native <c>Booster</c> handle.
///
///     <para>Rows are feature vectors of equal length. The matrix is stored as <see cref="float" /> to
///     match ML.NET's feature-column type; the <see cref="Label" /> stays <see cref="double" /> (the
///     tuner's objective values are doubles).</para>
/// </summary>
public sealed class Dataset
{
    /// <summary>The feature matrix, <c>n</c> rows × <see cref="NumFeatures" /> columns.</summary>
    public float[][] Data { get; }

    /// <summary>The per-row target, or <c>null</c> for an unlabeled set (e.g. a pure predict set).</summary>
    public double[]? Label { get; }

    /// <summary>Optional per-row weight (currently accepted for API parity; unused by the default trainer).</summary>
    public double[]? Weight { get; }

    /// <summary>Optional feature names (accepted for API parity; the default trainer indexes by position).</summary>
    public IReadOnlyList<string>? FeatureName { get; }

    /// <summary>The number of rows.</summary>
    public int NumRows => Data.Length;

    /// <summary>The number of feature columns (0 for an empty set).</summary>
    public int NumFeatures { get; }

    /// <summary>Construct from a jagged feature matrix (each inner array is one row's features).</summary>
    public Dataset(
        float[][] data, double[]? label = null, double[]? weight = null,
        IReadOnlyList<string>? featureName = null)
    {
        Data = data ?? throw new ArgumentNullException(nameof(data));
        NumFeatures = data.Length > 0 ? data[0].Length : 0;
        foreach (var row in data)
            if (row.Length != NumFeatures)
                throw new ValueError("All rows of `data` must have the same number of features.");
        if (label is not null && label.Length != data.Length)
            throw new ValueError(
                $"`label` length ({label.Length}) must equal the number of rows ({data.Length}).");
        if (weight is not null && weight.Length != data.Length)
            throw new ValueError(
                $"`weight` length ({weight.Length}) must equal the number of rows ({data.Length}).");

        Label = label;
        Weight = weight;
        FeatureName = featureName;
    }

    /// <summary>Convenience overload accepting a rectangular <c>double[,]</c> matrix (copied to floats).</summary>
    public Dataset(
        double[,] data, double[]? label = null, double[]? weight = null,
        IReadOnlyList<string>? featureName = null)
        : this(ToJagged(data), label, weight, featureName)
    {
    }

    /// <summary>The label, or a zero vector when this set is unlabeled (mirrors an all-zero target).</summary>
    internal double[] GetLabelOrZeros()
        => Label ?? new double[NumRows];

    private static float[][] ToJagged(double[,] data)
    {
        int n = data.GetLength(0), d = data.GetLength(1);
        var jagged = new float[n][];
        for (int i = 0; i < n; i++)
        {
            var row = new float[d];
            for (int j = 0; j < d; j++)
                row[j] = (float)data[i, j];
            jagged[i] = row;
        }
        return jagged;
    }
}
