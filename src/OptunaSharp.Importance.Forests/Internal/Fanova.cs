using System;
using System.Collections.Generic;

namespace OptunaSharp.Importance.Forests.Internal;

/// <summary>
///     Port of <c>optuna/importance/_fanova/_fanova.py: _Fanova</c>. Fits a random forest (ML.NET
///     FastForest here, scikit-learn's <c>RandomForestRegressor</c> in Optuna) and computes fANOVA
///     importances from the resulting <see cref="FanovaTree" />s.
/// </summary>
internal sealed class Fanova
{
    private readonly RandomForestRegressor _forest;

    private FanovaTree[]? _trees;
    private IReadOnlyList<long[]>? _columnToEncodedColumns;
    private readonly Dictionary<int, double[]> _variances = new();

    internal Fanova(int nTrees, int maxDepth, int minSamplesSplit, int minSamplesLeaf, int? seed)
    {
        // min_samples_split has no direct FastForest equivalent (Optuna passes its sklearn default,
        // 2). min_samples_leaf maps to MinimumExampleCountPerLeaf.
        _ = minSamplesSplit;
        _forest = new RandomForestRegressor(nTrees, maxDepth, minSamplesLeaf, seed);
    }

    internal void Fit(double[][] x, double[] y, double[,] searchSpaces, IReadOnlyList<long[]> columnToEncodedColumns)
    {
        System.Diagnostics.Debug.Assert(x.Length == y.Length);
        System.Diagnostics.Debug.Assert(searchSpaces.GetLength(1) == 2);

        int nFeatures = searchSpaces.GetLength(0);
        _forest.Fit(x, y, nFeatures);

        var trees = new FanovaTree[_forest.Trees.Length];
        for (int i = 0; i < trees.Length; i++)
            trees[i] = new FanovaTree(_forest.Trees[i], searchSpaces);
        _trees = trees;
        _columnToEncodedColumns = columnToEncodedColumns;
        _variances.Clear();

        bool allZeroVariance = true;
        foreach (var tree in trees)
        {
            if (tree.Variance != 0.0)
            {
                allZeroVariance = false;
                break;
            }
        }

        if (allZeroVariance)
        {
            // Can occur if e.g. X.shape[0] == 1 — no importance is assessable.
            throw new InvalidOperationException("Encountered zero total variance in all trees.");
        }
    }

    internal (double Mean, double Std) GetImportance(int feature)
    {
        System.Diagnostics.Debug.Assert(_trees is not null);

        ComputeVariances(feature);

        var fractions = new List<double>();
        for (int treeIndex = 0; treeIndex < _trees!.Length; treeIndex++)
        {
            double treeVariance = _trees[treeIndex].Variance;
            if (treeVariance > 0.0)
                fractions.Add(_variances[feature][treeIndex] / treeVariance);
        }

        return (Mean(fractions), Std(fractions));
    }

    private void ComputeVariances(int feature)
    {
        System.Diagnostics.Debug.Assert(_trees is not null && _columnToEncodedColumns is not null);

        if (_variances.ContainsKey(feature))
            return;

        long[] rawFeatures = _columnToEncodedColumns![feature];
        var variances = new double[_trees!.Length];
        for (int treeIndex = 0; treeIndex < _trees.Length; treeIndex++)
        {
            double marginalVariance = _trees[treeIndex].GetMarginalVariance(rawFeatures);
            variances[treeIndex] = Math.Max(marginalVariance, 0.0);  // np.clip(., 0.0, None)
        }

        _variances[feature] = variances;
    }

    /// <summary><c>np.ndarray.mean()</c>.</summary>
    private static double Mean(IReadOnlyList<double> values)
    {
        double sum = 0.0;
        foreach (double v in values)
            sum += v;
        return sum / values.Count;
    }

    /// <summary><c>np.ndarray.std()</c> — population standard deviation (ddof=0).</summary>
    private static double Std(IReadOnlyList<double> values)
    {
        double mean = Mean(values);
        double sum = 0.0;
        foreach (double v in values)
        {
            double d = v - mean;
            sum += d * d;
        }
        return Math.Sqrt(sum / values.Count);
    }
}
