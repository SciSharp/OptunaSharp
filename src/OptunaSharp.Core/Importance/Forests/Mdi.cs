using System.Collections.Generic;
using OptunaSharp.Internal;

namespace OptunaSharp.Importance.Forests.Internal;

/// <summary>
///     Mean Decrease Impurity feature importances over a forest of <see cref="SklearnLikeTree" />s —
///     the port of <c>sklearn.ensemble.RandomForestRegressor.feature_importances_</c> together with
///     the per-tree <c>sklearn.tree._tree.Tree.compute_feature_importances(normalize=True)</c>.
///
///     <para>It reads each node's stored <see cref="SklearnLikeTree.Impurity" /> and
///     <see cref="SklearnLikeTree.WeightedNNodeSamples" />. For the native CART those are the exact
///     criterion values (so this is bit-exact vs sklearn, matching numpy's pairwise
///     <c>np.mean</c>/<c>np.sum</c>); for the ML.NET satellite they are the routed training-set
///     variance (a documented approximation).</para>
/// </summary>
internal static class Mdi
{
    /// <summary>A single tree's normalized MDI (<c>compute_feature_importances(normalize=True)</c>).</summary>
    /// <param name="tree">The fitted tree whose nodes and split statistics are inspected.</param>
    /// <returns>The normalized impurity-reduction importance of each feature in this tree.</returns>
    private static double[] TreeImportances(SklearnLikeTree tree)
    {
        int nFeatures = tree.NFeatures;
        double[] impurity = tree.Impurity!;
        double[] wnns = tree.WeightedNNodeSamples!;
        var importances = new double[nFeatures];

        for (int node = 0; node < tree.NodeCount; node++)
        {
            if (tree.ChildrenLeft[node] != -1)  // != TREE_LEAF
            {
                int left = tree.ChildrenLeft[node];
                int right = tree.ChildrenRight[node];
                importances[tree.Feature[node]] +=
                    wnns[node] * impurity[node]
                    - wnns[left] * impurity[left]
                    - wnns[right] * impurity[right];
            }
        }

        for (int i = 0; i < nFeatures; i++)
            importances[i] /= wnns[0];  // nodes[0].weighted_n_node_samples

        double normalizer = MathUtil.PairwiseSum(importances);  // np.sum(importances)
        if (normalizer > 0.0)
            for (int i = 0; i < nFeatures; i++)
                importances[i] /= normalizer;

        return importances;
    }

    /// <summary>
    ///     Port of <c>BaseForest.feature_importances_</c>: mean of the per-tree importances over the
    ///     multi-node trees, then renormalize to sum 1 (or all zeros if every tree is a single node).
    /// </summary>
    /// <param name="trees">The fitted regression trees in the ensemble.</param>
    /// <param name="nFeatures">The number of feature columns in the training matrix.</param>
    /// <returns>The ensemble-averaged normalized impurity importance of each feature.</returns>
    internal static double[] FeatureImportances(SklearnLikeTree[] trees, int nFeatures)
    {
        var perTree = new List<double[]>();
        foreach (var tree in trees)
        {
            if (tree.NodeCount > 1)  // sklearn excludes single-node trees from the mean
                perTree.Add(TreeImportances(tree));
        }

        var all = new double[nFeatures];
        if (perTree.Count == 0)
            return all;

        // np.mean(all_importances, axis=0, dtype=float64): pairwise sum per column / n_trees.
        for (int j = 0; j < nFeatures; j++)
        {
            var column = new double[perTree.Count];
            for (int t = 0; t < perTree.Count; t++)
                column[t] = perTree[t][j];
            all[j] = MathUtil.PairwiseSum(column) / perTree.Count;
        }

        double sum = MathUtil.PairwiseSum(all);  // np.sum(all_importances)
        if (sum != 0.0)
            for (int j = 0; j < nFeatures; j++)
                all[j] /= sum;

        return all;
    }
}
