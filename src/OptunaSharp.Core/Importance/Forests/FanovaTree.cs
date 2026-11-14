using System;
using System.Collections.Generic;
using OptunaSharp.Internal;

namespace OptunaSharp.Importance.Forests.Internal;

/// <summary>
///     Port of <c>optuna/importance/_fanova/_tree.py: _FanovaTree</c> — the per-tree fANOVA
///     marginal-variance machinery from "An Efficient Approach for Assessing Hyperparameter
///     Importance" (Hutter et al., ICML 2014).
///
///     <para>Reads the tree structure through <see cref="SklearnLikeTree" /> (the sklearn flat-node
///     representation), so the algorithm is identical whether the tree came from a hand-built mock
///     (the port of Optuna's <c>test_tree.py</c>) or from ML.NET's FastForest. All arithmetic is
///     plain <c>double</c>; the port targets Optuna's <c>math.isclose</c> tolerance, not
///     bit-for-bit (the forest already diverges from scikit-learn — see the package README).</para>
/// </summary>
internal sealed class FanovaTree
{
    private readonly SklearnLikeTree _tree;
    private readonly double[,] _searchSpaces;      // shape (n_features, 2)

    private readonly double[,] _statistics;        // shape (n_nodes, 2): weighted value, sum of weights
    private readonly double[][] _splitMidpoints;   // per feature
    private readonly double[][] _splitSizes;       // per feature
    private readonly bool[,] _subtreeActiveFeatures; // shape (n_nodes, n_features)

    private double? _variance;                     // computed lazily, requires _statistics

    /// <summary>Precomputes split intervals and volume-weighted node statistics for exact tree marginalization.</summary>
    /// <param name="tree">The fitted tree whose nodes and split statistics are inspected.</param>
    /// <param name="searchSpaces">The lower and upper bound of each encoded feature, with shape (features, 2).</param>
    internal FanovaTree(SklearnLikeTree tree, double[,] searchSpaces)
    {
        System.Diagnostics.Debug.Assert(searchSpaces.GetLength(0) == tree.NFeatures);
        System.Diagnostics.Debug.Assert(searchSpaces.GetLength(1) == 2);

        _tree = tree;
        _searchSpaces = searchSpaces;

        _statistics = PrecomputeStatistics();
        (_splitMidpoints, _splitSizes) = PrecomputeSplitMidpointsAndSizes();
        _subtreeActiveFeatures = PrecomputeSubtreeActiveFeatures();
    }

    private int NFeatures => _searchSpaces.GetLength(0);
    private int NNodes => _tree.NodeCount;

    // ---- exposed for the test_tree.py port -------------------------------------------------
    /// <summary>The cached prediction and integration weight of each tree node.</summary>
    internal double[,] Statistics => _statistics;
    /// <summary>Returns the cached midpoint of each interval induced by the feature's split thresholds.</summary>
    /// <param name="feature">The zero-based feature column whose splits or importance are requested.</param>
    /// <returns>The cached midpoint of each interval induced by the feature's split thresholds.</returns>
    internal double[] SplitMidpoints(int feature) => _splitMidpoints[feature];
    /// <summary>Returns the cached width of each interval induced by the feature's split thresholds.</summary>
    /// <param name="feature">The zero-based feature column whose splits or importance are requested.</param>
    /// <returns>The cached width of each interval induced by the feature's split thresholds.</returns>
    internal double[] SplitSizes(int feature) => _splitSizes[feature];
    /// <summary>A node-by-feature mask identifying the split features occurring in each subtree.</summary>
    internal bool[,] SubtreeActiveFeatures => _subtreeActiveFeatures;
    /// <summary>The number of encoded feature dimensions integrated by this tree.</summary>
    internal int NFeaturesInternal => NFeatures;
    /// <summary>The number of nodes in the flattened regression tree.</summary>
    internal int NNodesInternal => NNodes;

    /// <summary>The integrated total variance of the tree's prediction over the search space.</summary>
    internal double Variance
    {
        get
        {
            if (_variance is null)
            {
                var values = new List<double>();
                var weights = new List<double>();
                for (int node = 0; node < NNodes; node++)
                {
                    if (_tree.Feature[node] < 0)  // leaf
                    {
                        values.Add(_statistics[node, 0]);
                        weights.Add(_statistics[node, 1]);
                    }
                }

                double averageValue = WeightedAverage(values, weights);
                double variance = WeightedSquaredDeviation(values, weights, averageValue);
                _variance = variance;
            }

            return _variance.Value;
        }
    }

    /// <summary>Returns the variance of the marginalized prediction for the selected feature subset.</summary>
    /// <param name="features">The encoded feature indices retained while integrating out all other dimensions.</param>
    /// <returns>The variance of the marginalized prediction for the selected feature subset.</returns>
    internal double GetMarginalVariance(long[] features)
    {
        System.Diagnostics.Debug.Assert(features.Length > 0);

        // For each midpoint along the given dimensions, traverse this tree to compute the marginal
        // predictions.
        var selectedMidpoints = new double[features.Length][];
        var selectedSizes = new double[features.Length][];
        for (int k = 0; k < features.Length; k++)
        {
            selectedMidpoints[k] = _splitMidpoints[features[k]];
            selectedSizes[k] = _splitSizes[features[k]];
        }

        var sample = new double[NFeatures];
        for (int i = 0; i < sample.Length; i++)
            sample[i] = double.NaN;

        var values = new List<double>();
        var weights = new List<double>();

        // itertools.product over the selected midpoints (and, in lock-step, the selected sizes).
        var dims = new int[features.Length];
        for (int k = 0; k < features.Length; k++)
            dims[k] = selectedMidpoints[k].Length;

        foreach (var combo in CartesianProduct(dims))
        {
            for (int k = 0; k < features.Length; k++)
                sample[features[k]] = selectedMidpoints[k][combo[k]];

            var (value, weight) = GetMarginalizedStatistics(sample);

            double sizeProduct = 1.0;
            for (int k = 0; k < features.Length; k++)
                sizeProduct *= selectedSizes[k][combo[k]];
            weight *= sizeProduct;

            values.Add(value);
            weights.Add(weight);
        }

        double averageValue = WeightedAverage(values, weights);
        double variance = WeightedSquaredDeviation(values, weights, averageValue);

        System.Diagnostics.Debug.Assert(variance >= 0.0);
        return variance;
    }

    /// <summary>Returns the integrated prediction and search-space volume consistent with the fixed feature values.</summary>
    /// <param name="featureVector">The feature values to condition on; NaN marks a feature to integrate out.</param>
    /// <returns>The integrated prediction and search-space volume consistent with the fixed feature values.</returns>
    internal (double Value, double Weight) GetMarginalizedStatistics(double[] featureVector)
    {
        System.Diagnostics.Debug.Assert(featureVector.Length == NFeatures);

        var active = new bool[NFeatures];
        for (int i = 0; i < NFeatures; i++)
            active[i] = !double.IsNaN(featureVector[i]);

        // Reduce search-space cardinalities to 1 for non-active (marginalized) features.
        var rootSearchSpaces = (double[,])_searchSpaces.Clone();
        for (int i = 0; i < NFeatures; i++)
        {
            if (!active[i])
            {
                rootSearchSpaces[i, 0] = 0.0;
                rootSearchSpaces[i, 1] = 1.0;
            }
        }

        // Start from the root and traverse towards the leaves (an explicit LIFO stack, as Python's
        // list .append/.pop).
        var activeNodes = new Stack<(int Node, double[,] SearchSpaces)>();
        activeNodes.Push((0, rootSearchSpaces));

        var nodeIndices = new List<int>();
        var activeLeafSearchSpaces = new List<double[,]>();

        while (activeNodes.Count > 0)
        {
            var (nodeIndex, searchSpaces) = activeNodes.Pop();

            int feature = _tree.Feature[nodeIndex];
            if (feature >= 0)  // Not a leaf; avoid an extra leaf check.
            {
                double response = featureVector[feature];
                if (!double.IsNaN(response))
                {
                    int nextNodeIndex;
                    double[,] nextSubspace;
                    if (response <= _tree.Threshold[nodeIndex])
                    {
                        nextNodeIndex = _tree.ChildrenLeft[nodeIndex];
                        nextSubspace = GetSubspaces(searchSpaces, searchSpacesColumn: 1, nodeIndex);
                    }
                    else
                    {
                        nextNodeIndex = _tree.ChildrenRight[nodeIndex];
                        nextSubspace = GetSubspaces(searchSpaces, searchSpacesColumn: 0, nodeIndex);
                    }

                    activeNodes.Push((nextNodeIndex, nextSubspace));
                    continue;
                }

                // The subtree from this node splits on an active feature — push both children.
                if (SubtreeSplitsOnActiveFeature(nodeIndex, active))
                {
                    activeNodes.Push((_tree.ChildrenLeft[nodeIndex], searchSpaces));
                    activeNodes.Push((_tree.ChildrenRight[nodeIndex], searchSpaces));
                    continue;
                }
            }

            // Leaf, or the subtree does not split on any active feature.
            nodeIndices.Add(nodeIndex);
            activeLeafSearchSpaces.Add(searchSpaces);
        }

        var values = new double[nodeIndices.Count];
        var weights = new double[nodeIndices.Count];
        for (int i = 0; i < nodeIndices.Count; i++)
        {
            values[i] = _statistics[nodeIndices[i], 0];
            double cardinality = GetCardinality(activeLeafSearchSpaces[i]);
            weights[i] = _statistics[nodeIndices[i], 1] / cardinality;
        }

        double value = WeightedAverage(values, weights);
        double weight = MathUtil.PairwiseSum(weights);  // numpy weights.sum() (pairwise)

        return (value, weight);
    }

    /// <summary>Returns true when the subtree splits on at least one of the selected active features.</summary>
    /// <param name="nodeIndex">The zero-based node index in the flat tree arrays.</param>
    /// <param name="active">The feature mask specifying which coordinates remain conditioned on.</param>
    /// <returns>True when the subtree splits on at least one of the selected active features.</returns>
    private bool SubtreeSplitsOnActiveFeature(int nodeIndex, bool[] active)
    {
        for (int i = 0; i < NFeatures; i++)
        {
            if (active[i] && _subtreeActiveFeatures[nodeIndex, i])
                return true;
        }
        return false;
    }

    /// <summary>Returns the prediction and search-space weight associated with every tree node.</summary>
    /// <returns>The prediction and search-space weight associated with every tree node.</returns>
    private double[,] PrecomputeStatistics()
    {
        int nNodes = NNodes;
        var statistics = new double[nNodes, 2];

        // Per-node subspace; node 0 owns the whole search space. `double[][,]` is a 1-D array of
        // 2-D arrays (element subspaces[i] is a double[,]).
        var subspaces = new double[nNodes][,];
        subspaces[0] = _searchSpaces;

        // Compute marginals for leaf nodes; propagate child subspaces down.
        for (int nodeIndex = 0; nodeIndex < nNodes; nodeIndex++)
        {
            double[,] subspace = subspaces[nodeIndex];

            if (_tree.Feature[nodeIndex] < 0)  // leaf
            {
                double value = _tree.Value[nodeIndex];
                double weight = GetCardinality(subspace);
                statistics[nodeIndex, 0] = value;
                statistics[nodeIndex, 1] = weight;
            }
            else
            {
                int left = _tree.ChildrenLeft[nodeIndex];
                int right = _tree.ChildrenRight[nodeIndex];
                System.Diagnostics.Debug.Assert(subspaces[left] is null);
                subspaces[left] = GetSubspaces(subspace, searchSpacesColumn: 1, nodeIndex);
                System.Diagnostics.Debug.Assert(subspaces[right] is null);
                subspaces[right] = GetSubspaces(subspace, searchSpacesColumn: 0, nodeIndex);
            }
        }

        // Compute marginals for internal nodes (children already computed).
        for (int nodeIndex = nNodes - 1; nodeIndex >= 0; nodeIndex--)
        {
            if (_tree.Feature[nodeIndex] >= 0)  // internal
            {
                int left = _tree.ChildrenLeft[nodeIndex];
                int right = _tree.ChildrenRight[nodeIndex];
                var childValues = new[] { statistics[left, 0], statistics[right, 0] };
                var childWeights = new[] { statistics[left, 1], statistics[right, 1] };
                double value = WeightedAverage(childValues, childWeights);
                double weight = childWeights[0] + childWeights[1];
                statistics[nodeIndex, 0] = value;
                statistics[nodeIndex, 1] = weight;
            }
        }

        return statistics;
    }

    /// <summary>Returns the midpoint and width arrays of each feature's split-induced intervals.</summary>
    /// <returns>The midpoint and width arrays of each feature's split-induced intervals.</returns>
    private (double[][] Midpoints, double[][] Sizes) PrecomputeSplitMidpointsAndSizes()
    {
        var midpoints = new double[NFeatures][];
        var sizes = new double[NFeatures][];

        var featureSplitValues = ComputeFeaturesSplitValues();
        for (int feature = 0; feature < NFeatures; feature++)
        {
            double[] splits = featureSplitValues[feature];
            // concatenate([low], splits, [high])
            var values = new double[splits.Length + 2];
            values[0] = _searchSpaces[feature, 0];
            Array.Copy(splits, 0, values, 1, splits.Length);
            values[^1] = _searchSpaces[feature, 1];

            var midpoint = new double[values.Length - 1];
            var size = new double[values.Length - 1];
            for (int i = 0; i < midpoint.Length; i++)
            {
                midpoint[i] = 0.5 * (values[i + 1] + values[i]);
                size[i] = values[i + 1] - values[i];
            }

            midpoints[feature] = midpoint;
            sizes[feature] = size;
        }

        return (midpoints, sizes);
    }

    /// <summary>Returns the sorted distinct split thresholds for each feature.</summary>
    /// <returns>The sorted distinct split thresholds for each feature.</returns>
    private double[][] ComputeFeaturesSplitValues()
    {
        var allSplitValues = new SortedSet<double>[NFeatures];
        for (int i = 0; i < NFeatures; i++)
            allSplitValues[i] = new SortedSet<double>();

        for (int nodeIndex = 0; nodeIndex < NNodes; nodeIndex++)
        {
            int feature = _tree.Feature[nodeIndex];
            if (feature >= 0)  // internal
                allSplitValues[feature].Add(_tree.Threshold[nodeIndex]);
        }

        var sorted = new double[NFeatures][];
        for (int feature = 0; feature < NFeatures; feature++)
        {
            var arr = new double[allSplitValues[feature].Count];
            allSplitValues[feature].CopyTo(arr);  // SortedSet enumerates ascending
            sorted[feature] = arr;
        }

        return sorted;
    }

    /// <summary>Returns a node-by-feature mask identifying split features present in each subtree.</summary>
    /// <returns>A node-by-feature mask identifying split features present in each subtree.</returns>
    private bool[,] PrecomputeSubtreeActiveFeatures()
    {
        var subtreeActiveFeatures = new bool[NNodes, NFeatures];

        for (int nodeIndex = NNodes - 1; nodeIndex >= 0; nodeIndex--)
        {
            int feature = _tree.Feature[nodeIndex];
            if (feature >= 0)  // internal
            {
                subtreeActiveFeatures[nodeIndex, feature] = true;
                int left = _tree.ChildrenLeft[nodeIndex];
                int right = _tree.ChildrenRight[nodeIndex];
                for (int i = 0; i < NFeatures; i++)
                {
                    subtreeActiveFeatures[nodeIndex, i] =
                        subtreeActiveFeatures[nodeIndex, i]
                        || subtreeActiveFeatures[left, i]
                        || subtreeActiveFeatures[right, i];
                }
            }
        }

        return subtreeActiveFeatures;
    }

    // ---- node-subspace helpers (ports of the module-level _get_* functions) ----------------

    /// <summary>Port of <c>_get_subspaces(search_spaces, search_spaces_column, feature, threshold)</c>.</summary>
    /// <param name="searchSpaces">The lower and upper bound of each encoded feature, with shape (features, 2).</param>
    /// <param name="searchSpacesColumn">The interval endpoints for one feature.</param>
    /// <param name="nodeIndex">The zero-based node index in the flat tree arrays.</param>
    /// <returns>A copy of the feature bounds restricted to the selected child side.</returns>
    private double[,] GetSubspaces(double[,] searchSpaces, int searchSpacesColumn, int nodeIndex)
    {
        var subspace = (double[,])searchSpaces.Clone();
        subspace[_tree.Feature[nodeIndex], searchSpacesColumn] = _tree.Threshold[nodeIndex];
        return subspace;
    }

    // Exposed for the test_tree.py port (test_tree_get_node_subspaces).
    /// <summary>Returns the feature bounds after restricting the split feature to the left child.</summary>
    /// <param name="nodeIndex">The zero-based node index in the flat tree arrays.</param>
    /// <param name="searchSpaces">The lower and upper bound of each encoded feature, with shape (features, 2).</param>
    /// <returns>The feature bounds after restricting the split feature to the left child.</returns>
    internal double[,] GetNodeLeftChildSubspaces(int nodeIndex, double[,] searchSpaces)
        => GetSubspaces(searchSpaces, searchSpacesColumn: 1, nodeIndex);

    /// <summary>Returns the feature bounds after restricting the split feature to the right child.</summary>
    /// <param name="nodeIndex">The zero-based node index in the flat tree arrays.</param>
    /// <param name="searchSpaces">The lower and upper bound of each encoded feature, with shape (features, 2).</param>
    /// <returns>The feature bounds after restricting the split feature to the right child.</returns>
    internal double[,] GetNodeRightChildSubspaces(int nodeIndex, double[,] searchSpaces)
        => GetSubspaces(searchSpaces, searchSpacesColumn: 0, nodeIndex);

    /// <summary>
    ///     Port of <c>_get_cardinality(search_spaces) = np.prod(hi - lo)</c>: NumPy's sequential <c>multiply.reduce</c>,
    ///     which NumSharp 0.81.0's float <c>np.prod</c> also follows (pinned by <c>SequentialProductTests</c>); a loop because
    ///     it runs once per tree node.
    /// </summary>
    /// <param name="searchSpaces">The lower and upper bound of each encoded feature, with shape (features, 2).</param>
    /// <returns>The product of interval widths, representing the subspace volume.</returns>
    private static double GetCardinality(double[,] searchSpaces)
    {
        double product = 1.0;
        int n = searchSpaces.GetLength(0);
        for (int i = 0; i < n; i++)
            product *= searchSpaces[i, 1] - searchSpaces[i, 0];
        return product;
    }

    // ---- numeric helpers -------------------------------------------------------------------

    /// <summary>
    ///     <c>np.average(values, weights)</c> = <c>(values*weights).sum() / weights.sum()</c>. Both
    ///     reductions are numpy's pairwise summation (<see cref="MathUtil.PairwiseSum(double[])" />), so this is
    ///     bit-exact vs Optuna even for the large midpoint arrays a real forest produces (not just the
    ///     ≤3-element <c>test_tree.py</c> cases where pairwise == sequential).
    /// </summary>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <param name="weights">The nonnegative weight associated with each observation or value.</param>
    /// <returns>The sum of weighted values divided by total weight.</returns>
    private static double WeightedAverage(IReadOnlyList<double> values, IReadOnlyList<double> weights)
    {
        int n = values.Count;
        var products = new double[n];
        var w = new double[n];
        for (int i = 0; i < n; i++)
        {
            products[i] = values[i] * weights[i];
            w[i] = weights[i];
        }
        return MathUtil.PairwiseSum(products) / MathUtil.PairwiseSum(w);
    }

    /// <summary><c>np.average((values - avg)**2, weights)</c> (pairwise reductions).</summary>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <param name="weights">The nonnegative weight associated with each observation or value.</param>
    /// <param name="average">The weighted mean about which squared deviations are measured.</param>
    /// <returns>The weighted sum of squared deviations from the supplied average.</returns>
    private static double WeightedSquaredDeviation(
        IReadOnlyList<double> values, IReadOnlyList<double> weights, double average)
    {
        int n = values.Count;
        var products = new double[n];
        var w = new double[n];
        for (int i = 0; i < n; i++)
        {
            double d = values[i] - average;
            products[i] = d * d * weights[i];
            w[i] = weights[i];
        }
        return MathUtil.PairwiseSum(products) / MathUtil.PairwiseSum(w);
    }

    /// <summary><c>itertools.product</c> over ranges <c>[0, dims[k])</c>, yielding index tuples.</summary>
    /// <param name="dims">The coordinate indices to update.</param>
    /// <returns>Every index combination selecting one interval per requested feature.</returns>
    private static IEnumerable<int[]> CartesianProduct(int[] dims)
    {
        // If any dimension is empty, the product is empty (matches itertools.product).
        foreach (int d in dims)
        {
            if (d == 0)
                yield break;
        }

        var combo = new int[dims.Length];
        while (true)
        {
            yield return (int[])combo.Clone();

            // odometer increment from the rightmost index (itertools.product's order)
            int k = dims.Length - 1;
            while (k >= 0)
            {
                combo[k]++;
                if (combo[k] < dims[k])
                    break;
                combo[k] = 0;
                k--;
            }
            if (k < 0)
                yield break;
        }
    }
}
