using System;
using System.Collections.Generic;
using System.Linq;
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

    /// <summary>
    ///     Optional query-group sizes for <b>ranking</b> objectives (<c>lambdarank</c> / <c>rank_xendcg</c>),
    ///     the analog of LightGBM's <c>lgb.Dataset(..., group=[...])</c>. Each element is the number of
    ///     consecutive rows forming one query group, so the sizes must sum to <see cref="NumRows" />. Left
    ///     <c>null</c> for non-ranking datasets.
    /// </summary>
    public int[]? Group { get; }

    /// <summary>Optional feature names (accepted for API parity; the default trainer indexes by position).</summary>
    public IReadOnlyList<string>? FeatureName { get; }

    /// <summary>The number of rows.</summary>
    public int NumRows => Data.Length;

    /// <summary>The number of feature columns (0 for an empty set).</summary>
    public int NumFeatures { get; }

    /// <summary>The number of query groups (0 when <see cref="Group" /> is <c>null</c>).</summary>
    internal int NumGroups => Group?.Length ?? 0;

    /// <summary>Construct from a jagged feature matrix (each inner array is one row's features).</summary>
    /// <param name="data">The feature rows retained by the dataset; every row must have the same number of columns.</param>
    /// <param name="label">Optional target values, one per feature row; absent labels are treated as zeros by the default trainer.</param>
    /// <param name="weight">Optional per-row weights retained in the dataset.</param>
    /// <param name="featureName">Optional ordered feature names retained in the dataset.</param>
    /// <param name="group">Positive contiguous query-group sizes whose sum equals the row count; null when ranking groups are not supplied.</param>
    /// <exception cref="System.ArgumentNullException"><paramref name="data" /> is null.</exception>
    /// <exception cref="NumSharp.ValueError">Rows have inconsistent feature counts, labels or weights do not match the row count, or query-group sizes are nonpositive or do not sum to the row count.</exception>
    public Dataset(
        float[][] data, double[]? label = null, double[]? weight = null,
        IReadOnlyList<string>? featureName = null, int[]? group = null)
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
        if (group is not null)
        {
            foreach (var g in group)
                if (g <= 0)
                    throw new ValueError($"Every `group` size must be positive; got {g}.");
            int sum = group.Sum();
            if (sum != data.Length)
                throw new ValueError(
                    $"`group` sizes must sum to the number of rows ({data.Length}); got {sum}.");
        }

        Label = label;
        Weight = weight;
        Group = group;
        FeatureName = featureName;
    }

    /// <summary>Convenience overload accepting a rectangular <c>double[,]</c> matrix (copied to floats).</summary>
    /// <param name="data">The rectangular feature matrix copied into float32 row arrays.</param>
    /// <param name="label">Optional target values, one per feature row; absent labels are treated as zeros by the default trainer.</param>
    /// <param name="weight">Optional per-row weights retained in the dataset.</param>
    /// <param name="featureName">Optional ordered feature names retained in the dataset.</param>
    /// <param name="group">Positive contiguous query-group sizes whose sum equals the row count; null when ranking groups are not supplied.</param>
    /// <exception cref="NumSharp.ValueError">Rows have inconsistent feature counts, labels or weights do not match the row count, or query-group sizes are nonpositive or do not sum to the row count.</exception>
    public Dataset(
        double[,] data, double[]? label = null, double[]? weight = null,
        IReadOnlyList<string>? featureName = null, int[]? group = null)
        : this(ToJagged(data), label, weight, featureName, group)
    {
    }

    /// <summary>The label, or a zero vector when this set is unlabeled (mirrors an all-zero target).</summary>
    /// <returns>The stored label array, or a zero-filled array with one entry per row.</returns>
    internal double[] GetLabelOrZeros()
        => Label ?? new double[NumRows];

    /// <summary>Per-row group id (0-based), expanded from <see cref="Group" /> sizes — for ML.NET ranking.</summary>
    /// <returns>Zero-based query-group identifiers for every dataset row.</returns>
    /// <exception cref="NumSharp.ValueError">The dataset has no ranking query-group sizes.</exception>
    internal uint[] RowGroupIds()
    {
        if (Group is null)
            throw new ValueError(
                "A ranking objective (`lambdarank` / `rank_xendcg`) requires the dataset's `group` "
                + "sizes, but none were supplied.");
        var ids = new uint[NumRows];
        int row = 0;
        for (uint g = 0; g < Group.Length; g++)
            for (int k = 0; k < Group[g]; k++)
                ids[row++] = g;
        return ids;
    }

    /// <summary>A contiguous row slice <c>[offset, offset + count)</c>, with group sizes recomputed for it.</summary>
    /// <param name="offset">The zero-based first row included in the slice.</param>
    /// <param name="count">The number of consecutive rows to retain.</param>
    /// <returns>A dataset containing the requested consecutive rows and aligned metadata.</returns>
    internal Dataset Slice(int offset, int count)
    {
        var data = new float[count][];
        var label = Label is null ? null : new double[count];
        var weight = Weight is null ? null : new double[count];
        for (int i = 0; i < count; i++)
        {
            data[i] = Data[offset + i];
            if (label is not null) label[i] = Label![offset + i];
            if (weight is not null) weight[i] = Weight![offset + i];
        }

        int[]? group = null;
        if (Group is not null)
        {
            // Keep only the group cells overlapping [offset, offset + count), clipping the partial ends.
            var sizes = new List<int>();
            int start = 0;
            foreach (int gs in Group)
            {
                int lo = Math.Max(start, offset);
                int hi = Math.Min(start + gs, offset + count);
                if (hi > lo) sizes.Add(hi - lo);
                start += gs;
            }
            group = sizes.ToArray();
        }

        return new Dataset(data, label, weight, FeatureName, group);
    }

    /// <summary>
    ///     Select whole query groups by group index (order preserved), recomputing <see cref="Group" />
    ///     for the selection — the fold builder for ranking cross-validation, which must never split a group.
    /// </summary>
    /// <param name="groupIndices">The zero-based query-group indices to retain as complete groups.</param>
    /// <returns>A dataset containing complete selected query groups in the requested order.</returns>
    /// <exception cref="NumSharp.ValueError">The source dataset is not grouped.</exception>
    internal Dataset SelectGroups(IReadOnlyList<int> groupIndices)
    {
        if (Group is null)
            throw new ValueError("SelectGroups requires a grouped dataset.");

        // Precompute each group's row offset.
        var offsets = new int[Group.Length];
        for (int g = 1; g < Group.Length; g++)
            offsets[g] = offsets[g - 1] + Group[g - 1];

        int total = groupIndices.Sum(g => Group[g]);
        var data = new float[total][];
        var label = Label is null ? null : new double[total];
        var weight = Weight is null ? null : new double[total];
        var sizes = new int[groupIndices.Count];

        int row = 0;
        for (int gi = 0; gi < groupIndices.Count; gi++)
        {
            int g = groupIndices[gi];
            int off = offsets[g];
            sizes[gi] = Group[g];
            for (int k = 0; k < Group[g]; k++)
            {
                data[row] = Data[off + k];
                if (label is not null) label[row] = Label![off + k];
                if (weight is not null) weight[row] = Weight![off + k];
                row++;
            }
        }

        return new Dataset(data, label, weight, FeatureName, sizes);
    }

    /// <summary>Copies a rectangular double-precision feature matrix into float32 feature rows.</summary>
    /// <param name="data">The feature rows or ML.NET data view used by the operation.</param>
    /// <returns>Independent float32 row arrays containing the converted features.</returns>
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
