using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace OptunaSharp.Importance.Forests.Internal;

/// <summary>
///     The random-forest backend for the forest-based importance evaluators — the analogue of
///     Optuna's <c>sklearn.ensemble.RandomForestRegressor</c>, fitted here by ML.NET's
///     <c>FastForest</c>. After <see cref="Fit" /> it exposes each fitted tree as a
///     <see cref="SklearnLikeTree" /> (for fANOVA's marginal-variance walk) and can compute
///     Mean-Decrease-Impurity feature importances (<see cref="FeatureImportances" />).
///
///     <para><b>Documented divergence.</b> ML.NET FastForest is not tree-identical to
///     scikit-learn's random forest — it bins features, grows leaf-wise (not depth-wise), and uses
///     float32 thresholds — so importances differ numerically from Python Optuna. The fANOVA
///     algorithm and the MDI formula themselves are faithful; only the underlying forest differs.
///     For reproducibility the forest is fitted single-threaded from the given seed.</para>
/// </summary>
internal sealed class RandomForestRegressor
{
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int _minSamplesLeaf;
    private readonly int? _seed;

    private SklearnLikeTree[]? _trees;
    private double[][]? _x;
    private double[]? _y;
    private int _nFeatures;

    internal RandomForestRegressor(int nTrees, int maxDepth, int minSamplesLeaf, int? seed)
    {
        _nTrees = nTrees;
        _maxDepth = maxDepth;
        _minSamplesLeaf = minSamplesLeaf;
        _seed = seed;
    }

    internal SklearnLikeTree[] Trees =>
        _trees ?? throw new InvalidOperationException("Fit must be called before accessing Trees.");

    internal void Fit(double[][] x, double[] y, int nFeatures)
    {
        System.Diagnostics.Debug.Assert(x.Length == y.Length);
        _x = x;
        _y = y;
        _nFeatures = nFeatures;

        // MLContext seed drives all randomness; null → nondeterministic (Optuna's random_state=None).
        var mlContext = new MLContext(seed: _seed);

        var rows = new InputRow[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            var features = new float[nFeatures];
            for (int j = 0; j < nFeatures; j++)
                features[j] = (float)x[i][j];
            rows[i] = new InputRow { Features = features, Label = (float)y[i] };
        }

        // The feature-vector width is only known at runtime, so set it on a SchemaDefinition rather
        // than a compile-time [VectorType(n)] attribute.
        var schema = SchemaDefinition.Create(typeof(InputRow));
        schema[nameof(InputRow.Features)].ColumnType =
            new VectorDataViewType(NumberDataViewType.Single, nFeatures);
        IDataView dataView = mlContext.Data.LoadFromEnumerable(rows, schema);

        // A depth-d binary tree has at most 2^d leaves; ML.NET caps tree size by leaf count, and a
        // tree can never have more leaves than training rows (min 1 sample per leaf). Clamp to avoid
        // absurd allocations for the default max_depth=64.
        int leafCap = Math.Max(2, x.Length);
        int fromDepth = _maxDepth >= 30 ? int.MaxValue : (1 << _maxDepth);
        int numberOfLeaves = Math.Max(2, Math.Min(fromDepth, leafCap));

        var options = new FastForestRegressionTrainer.Options
        {
            LabelColumnName = nameof(InputRow.Label),
            FeatureColumnName = nameof(InputRow.Features),
            NumberOfTrees = _nTrees,
            NumberOfLeaves = numberOfLeaves,
            MinimumExampleCountPerLeaf = _minSamplesLeaf,
            // scikit-learn's RandomForestRegressor default is max_features=1.0 (all features per
            // split); mirror it rather than FastForest's 0.7 default.
            FeatureFraction = 1.0,
            NumberOfThreads = 1,          // determinism + the NumSharp single-threaded-kernel rule
            AllowEmptyTrees = true,       // tolerate tiny / degenerate studies
            ShuffleLabels = false,
        };
        if (_seed.HasValue)
        {
            options.Seed = _seed.Value;
            options.FeatureSelectionSeed = _seed.Value;
        }

        var trainer = mlContext.Regression.Trainers.FastForest(options);
        var model = trainer.Fit(dataView);

        var ensemble = model.Model.TrainedTreeEnsemble;
        var trees = new List<SklearnLikeTree>(ensemble.Trees.Count);
        foreach (var tree in ensemble.Trees)
            trees.Add(Adapt(tree, nFeatures));
        _trees = trees.ToArray();
    }

    /// <summary>
    ///     Mean Decrease Impurity importances per encoded feature — the port of
    ///     <c>sklearn.ensemble.RandomForestRegressor.feature_importances_</c>: each tree's
    ///     normalized MDI, averaged across trees, then renormalized. Computed by routing the fitted
    ///     training set through each extracted tree (ML.NET does not expose per-node sample counts /
    ///     impurities), so the impurities are the training-set variance at each node rather than the
    ///     per-tree bootstrap impurity scikit-learn stores — a documented divergence.
    /// </summary>
    internal double[] FeatureImportances()
    {
        if (_trees is null || _x is null || _y is null)
            throw new InvalidOperationException("Fit must be called before FeatureImportances.");

        var all = new double[_nFeatures];
        foreach (var tree in _trees)
        {
            double[] treeImportances = ComputeTreeMdi(tree, _x, _y, _nFeatures);
            for (int j = 0; j < _nFeatures; j++)
                all[j] += treeImportances[j];
        }

        for (int j = 0; j < _nFeatures; j++)
            all[j] /= _trees.Length;  // np.mean over trees

        double normalizer = 0.0;
        for (int j = 0; j < _nFeatures; j++)
            normalizer += all[j];
        if (normalizer > 0.0)
        {
            for (int j = 0; j < _nFeatures; j++)
                all[j] /= normalizer;
        }

        return all;
    }

    /// <summary>
    ///     A single tree's normalized MDI, the port of
    ///     <c>sklearn.tree._tree.Tree.compute_feature_importances(normalize=True)</c>, evaluated on
    ///     the given data routed through the tree structure.
    /// </summary>
    private static double[] ComputeTreeMdi(SklearnLikeTree tree, double[][] x, double[] y, int nFeatures)
    {
        int n = tree.NodeCount;
        var count = new double[n];
        var sumY = new double[n];
        var sumY2 = new double[n];

        for (int r = 0; r < x.Length; r++)
        {
            int node = 0;
            while (true)
            {
                count[node] += 1.0;
                sumY[node] += y[r];
                sumY2[node] += y[r] * y[r];

                if (tree.Feature[node] < 0)  // leaf
                    break;

                int feature = tree.Feature[node];
                node = x[r][feature] <= tree.Threshold[node]
                    ? tree.ChildrenLeft[node]
                    : tree.ChildrenRight[node];
            }
        }

        var importances = new double[nFeatures];
        for (int node = 0; node < n; node++)
        {
            if (tree.Feature[node] < 0)  // leaf — no split contribution
                continue;

            int left = tree.ChildrenLeft[node];
            int right = tree.ChildrenRight[node];
            double decrease =
                count[node] * Impurity(node, count, sumY, sumY2)
                - count[left] * Impurity(left, count, sumY, sumY2)
                - count[right] * Impurity(right, count, sumY, sumY2);
            importances[tree.Feature[node]] += decrease;
        }

        double rootCount = count[0];
        if (rootCount > 0.0)
        {
            for (int j = 0; j < nFeatures; j++)
                importances[j] /= rootCount;
        }

        double normalizer = 0.0;
        for (int j = 0; j < nFeatures; j++)
            normalizer += importances[j];
        if (normalizer > 0.0)
        {
            for (int j = 0; j < nFeatures; j++)
                importances[j] /= normalizer;
        }
        // else: a tree with no impurity decrease contributes an all-zeros vector (sklearn behavior).

        return importances;
    }

    /// <summary>Node impurity = variance of the targets reaching it (regression MSE criterion).</summary>
    private static double Impurity(int node, double[] count, double[] sumY, double[] sumY2)
    {
        if (count[node] <= 0.0)
            return 0.0;
        double mean = sumY[node] / count[node];
        double variance = sumY2[node] / count[node] - mean * mean;
        return variance > 0.0 ? variance : 0.0;  // guard float cancellation
    }

    /// <summary>
    ///     Adapt an ML.NET <c>RegressionTree</c> into a <see cref="SklearnLikeTree" /> by a
    ///     pre-order DFS from the root, so parents get smaller node indices than their children (the
    ///     ordering the fANOVA precompute loops require) — matching scikit-learn's node numbering.
    ///
    ///     <para>ML.NET numbers internal nodes <c>0..NumberOfNodes-1</c> (root = 0) and encodes a
    ///     leaf child as a negative index (<c>leafIndex = ~child</c>), the leaf's prediction being
    ///     <c>LeafValues[leafIndex]</c>.</para>
    /// </summary>
    private static SklearnLikeTree Adapt(RegressionTreeBase tree, int nFeatures)
    {
        int internalNodes = tree.NumberOfNodes;

        if (internalNodes == 0)
        {
            // A tree that never split: sklearn represents it as a single leaf node.
            double onlyLeaf = tree.LeafValues.Count > 0 ? tree.LeafValues[0] : 0.0;
            return new SklearnLikeTree(
                nFeatures,
                feature: new[] { -1 },
                childrenLeft: new[] { -1 },
                childrenRight: new[] { -1 },
                threshold: new[] { -2.0 },
                value: new[] { onlyLeaf });
        }

        int totalNodes = internalNodes + tree.NumberOfLeaves;
        var feature = new int[totalNodes];
        var threshold = new double[totalNodes];
        var childrenLeft = new int[totalNodes];
        var childrenRight = new int[totalNodes];
        var value = new double[totalNodes];
        int next = 0;

        int VisitChild(int mlChild)
        {
            if (mlChild >= 0)
                return VisitInternal(mlChild);

            int leafIndex = ~mlChild;  // ML.NET encodes a leaf as leafIndex = ~child (= -child - 1)
            int sk = next++;
            feature[sk] = -1;
            threshold[sk] = -2.0;
            childrenLeft[sk] = -1;
            childrenRight[sk] = -1;
            value[sk] = tree.LeafValues[leafIndex];
            return sk;
        }

        int VisitInternal(int mlNode)
        {
            int sk = next++;
            feature[sk] = tree.NumericalSplitFeatureIndexes[mlNode];
            threshold[sk] = tree.NumericalSplitThresholds[mlNode];  // float32 → double
            value[sk] = 0.0;                                        // unused for internal nodes
            childrenLeft[sk] = VisitChild(tree.LeftChild[mlNode]);
            childrenRight[sk] = VisitChild(tree.RightChild[mlNode]);
            return sk;
        }

        VisitInternal(0);
        return new SklearnLikeTree(nFeatures, feature, childrenLeft, childrenRight, threshold, value);
    }

    private sealed class InputRow
    {
        public float[] Features { get; set; } = Array.Empty<float>();
        public float Label { get; set; }
    }
}
