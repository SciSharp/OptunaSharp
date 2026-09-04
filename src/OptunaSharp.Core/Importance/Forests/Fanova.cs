using System;
using System.Collections.Generic;
using OptunaSharp.Internal;

namespace OptunaSharp.Importance.Forests.Internal;

/// <summary>
///     Port of <c>optuna/importance/_fanova/_fanova.py: _Fanova</c>, decoupled from the forest
///     backend. It computes fANOVA importances from a pre-fitted forest of <see cref="FanovaTree" />s
///     — the trees come from the native scikit-learn CART port (Tier N, bit-exact vs Optuna) or from
///     ML.NET's FastForest (the satellite package). The per-tree marginal-variance fractions are
///     averaged; the fit-time zero-total-variance guard is preserved.
/// </summary>
internal sealed class Fanova
{
    private readonly FanovaTree[] _trees;
    private readonly IReadOnlyList<long[]> _columnToEncodedColumns;
    private readonly Dictionary<int, double[]> _variances = new();

    /// <param name="trees">The fitted forest as sklearn-flat trees.</param>
    /// <param name="searchSpaces">Bounds per encoded column, shape <c>(n_encoded, 2)</c>.</param>
    /// <param name="columnToEncodedColumns">Mapping from parameter column to its encoded columns.</param>
    internal Fanova(SklearnLikeTree[] trees, double[,] searchSpaces, IReadOnlyList<long[]> columnToEncodedColumns)
    {
        System.Diagnostics.Debug.Assert(searchSpaces.GetLength(1) == 2);

        _trees = new FanovaTree[trees.Length];
        for (int i = 0; i < trees.Length; i++)
            _trees[i] = new FanovaTree(trees[i], searchSpaces);
        _columnToEncodedColumns = columnToEncodedColumns;

        bool allZeroVariance = true;
        foreach (var tree in _trees)
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
        ComputeVariances(feature);

        var fractions = new List<double>();
        for (int treeIndex = 0; treeIndex < _trees.Length; treeIndex++)
        {
            double treeVariance = _trees[treeIndex].Variance;
            if (treeVariance > 0.0)
                fractions.Add(_variances[feature][treeIndex] / treeVariance);
        }

        var arr = fractions.ToArray();
        return (Mean(arr), Std(arr));
    }

    private void ComputeVariances(int feature)
    {
        if (_variances.ContainsKey(feature))
            return;

        long[] rawFeatures = _columnToEncodedColumns[feature];
        var variances = new double[_trees.Length];
        for (int treeIndex = 0; treeIndex < _trees.Length; treeIndex++)
        {
            double marginalVariance = _trees[treeIndex].GetMarginalVariance(rawFeatures);
            variances[treeIndex] = Math.Max(marginalVariance, 0.0);  // np.clip(., 0.0, None)
        }

        _variances[feature] = variances;
    }

    /// <summary><c>np.ndarray.mean()</c> — pairwise sum / n, matching numpy bit-for-bit.</summary>
    private static double Mean(double[] values)
        => MathUtil.PairwiseSum(values) / values.Length;

    /// <summary><c>np.ndarray.std()</c> — population standard deviation (ddof=0), numpy pairwise.</summary>
    private static double Std(double[] values)
    {
        double mean = Mean(values);
        var sq = new double[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            double d = values[i] - mean;
            sq[i] = d * d;
        }
        return Math.Sqrt(MathUtil.PairwiseSum(sq) / values.Length);
    }
}
