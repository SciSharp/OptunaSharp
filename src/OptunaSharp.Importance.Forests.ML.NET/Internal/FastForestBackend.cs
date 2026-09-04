using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using OptunaSharp.Importance.Forests.Internal;

namespace OptunaSharp.Importance.Forests.MLNet;

/// <summary>
///     The ML.NET FastForest backend for the forest-based importance evaluators — the alternative to
///     Core's pure-managed, scikit-learn-bit-exact CART. It fits <c>FastForest</c> and adapts each
///     tree into Core's <see cref="SklearnLikeTree" /> (the same structure the fANOVA walk and MDI
///     read): a pre-order DFS gives the sklearn-flat node arrays, and the training set is routed
///     through each tree to fill in the per-node impurity (variance) and weighted counts that MDI
///     needs (ML.NET does not expose per-node bootstrap statistics).
///
///     <para><b>Documented divergence.</b> FastForest is not tree-identical to scikit-learn (binned
///     features, leaf-wise growth, float32 thresholds; routed impurities rather than per-tree
///     bootstrap), so importances differ numerically from Python Optuna — the ranking of a dominant
///     parameter is robust, exact values are not. The forest is fitted single-threaded from the seed
///     for reproducibility.</para>
/// </summary>
internal static class FastForestBackend
{
    internal static SklearnLikeTree[] Fit(float[,] x, double[] y, int nFeatures, int nTrees, int maxDepth, int? seed)
    {
        int n = y.Length;
        var mlContext = new MLContext(seed: seed);

        var rows = new InputRow[n];
        for (int i = 0; i < n; i++)
        {
            var features = new float[nFeatures];
            for (int j = 0; j < nFeatures; j++)
                features[j] = x[i, j];
            rows[i] = new InputRow { Features = features, Label = (float)y[i] };
        }

        var schema = SchemaDefinition.Create(typeof(InputRow));
        schema[nameof(InputRow.Features)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, nFeatures);
        IDataView dataView = mlContext.Data.LoadFromEnumerable(rows, schema);

        // A depth-d binary tree has at most 2^d leaves; a tree can never have more leaves than rows.
        int leafCap = Math.Max(2, n);
        int fromDepth = maxDepth >= 30 ? int.MaxValue : (1 << maxDepth);
        int numberOfLeaves = Math.Max(2, Math.Min(fromDepth, leafCap));

        var options = new FastForestRegressionTrainer.Options
        {
            LabelColumnName = nameof(InputRow.Label),
            FeatureColumnName = nameof(InputRow.Features),
            NumberOfTrees = nTrees,
            NumberOfLeaves = numberOfLeaves,
            MinimumExampleCountPerLeaf = 1,
            FeatureFraction = 1.0,       // sklearn RandomForestRegressor default (max_features=1.0)
            NumberOfThreads = 1,         // determinism + the NumSharp single-threaded-kernel rule
            AllowEmptyTrees = true,
            ShuffleLabels = false,
        };
        if (seed.HasValue)
        {
            options.Seed = seed.Value;
            options.FeatureSelectionSeed = seed.Value;
        }

        var model = mlContext.Regression.Trainers.FastForest(options).Fit(dataView);
        var ensemble = model.Model.TrainedTreeEnsemble;

        var trees = new List<SklearnLikeTree>(ensemble.Trees.Count);
        foreach (var tree in ensemble.Trees)
            trees.Add(Adapt(tree, nFeatures, x, y));
        return trees.ToArray();
    }

    /// <summary>
    ///     Adapt an ML.NET <c>RegressionTree</c> into a <see cref="SklearnLikeTree" /> by a pre-order
    ///     DFS (parents before children — sklearn's numbering), then route the training set to fill
    ///     in per-node impurity + weighted counts.
    /// </summary>
    private static SklearnLikeTree Adapt(RegressionTreeBase tree, int nFeatures, float[,] x, double[] y)
    {
        int internalNodes = tree.NumberOfNodes;
        int[] feature, childrenLeft, childrenRight;
        double[] threshold, value;

        if (internalNodes == 0)
        {
            double onlyLeaf = tree.LeafValues.Count > 0 ? tree.LeafValues[0] : 0.0;
            feature = new[] { -1 }; childrenLeft = new[] { -1 }; childrenRight = new[] { -1 };
            threshold = new[] { -2.0 }; value = new[] { onlyLeaf };
        }
        else
        {
            int total = internalNodes + tree.NumberOfLeaves;
            feature = new int[total]; threshold = new double[total];
            childrenLeft = new int[total]; childrenRight = new int[total]; value = new double[total];
            int next = 0;

            int VisitChild(int mlChild)
            {
                if (mlChild >= 0)
                    return VisitInternal(mlChild);
                int leafIndex = ~mlChild;  // ML.NET encodes a leaf child as leafIndex = ~child
                int sk = next++;
                feature[sk] = -1; threshold[sk] = -2.0; childrenLeft[sk] = -1; childrenRight[sk] = -1;
                value[sk] = tree.LeafValues[leafIndex];
                return sk;
            }
            int VisitInternal(int mlNode)
            {
                int sk = next++;
                feature[sk] = tree.NumericalSplitFeatureIndexes[mlNode];
                threshold[sk] = tree.NumericalSplitThresholds[mlNode];  // float32 -> double
                value[sk] = 0.0;
                childrenLeft[sk] = VisitChild(tree.LeftChild[mlNode]);
                childrenRight[sk] = VisitChild(tree.RightChild[mlNode]);
                return sk;
            }
            VisitInternal(0);
        }

        var (impurity, weightedNNodeSamples) = RouteStatistics(feature, threshold, childrenLeft, childrenRight, x, y);
        return new SklearnLikeTree(nFeatures, feature, childrenLeft, childrenRight, threshold, value,
            impurity, weightedNNodeSamples);
    }

    /// <summary>
    ///     Route the training set through the tree structure to compute each node's variance
    ///     (impurity) and sample count — the statistics MDI reads (approximating scikit-learn's stored
    ///     per-tree bootstrap statistics with the full-training-set variance).
    /// </summary>
    private static (double[] Impurity, double[] Wnns) RouteStatistics(
        int[] feature, double[] threshold, int[] childrenLeft, int[] childrenRight, float[,] x, double[] y)
    {
        int nNodes = feature.Length;
        var count = new double[nNodes];
        var sumY = new double[nNodes];
        var sumY2 = new double[nNodes];

        for (int r = 0; r < y.Length; r++)
        {
            int node = 0;
            while (true)
            {
                count[node] += 1.0;
                sumY[node] += y[r];
                sumY2[node] += y[r] * y[r];
                if (feature[node] < 0)  // leaf
                    break;
                node = x[r, feature[node]] <= threshold[node] ? childrenLeft[node] : childrenRight[node];
            }
        }

        var impurity = new double[nNodes];
        for (int nd = 0; nd < nNodes; nd++)
        {
            if (count[nd] > 0.0)
            {
                double mean = sumY[nd] / count[nd];
                double variance = sumY2[nd] / count[nd] - mean * mean;
                impurity[nd] = variance > 0.0 ? variance : 0.0;
            }
        }
        return (impurity, count);
    }

    private sealed class InputRow
    {
        public float[] Features { get; set; } = Array.Empty<float>();
        public float Label { get; set; }
    }
}
