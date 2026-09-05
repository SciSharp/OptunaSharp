using System;
using OptunaSharp.Importance.Forests.Internal;

namespace OptunaSharp.Integration.Shap.Internal;

/// <summary>
///     Port of SHAP's <b>tree_path_dependent</b> Tree SHAP for the exact case
///     <see cref="ShapleyImportanceEvaluator" /> hits: <c>TreeExplainer(forest).shap_values(X)</c> on a
///     scikit-learn <c>RandomForestRegressor</c> with no background data. Two upstream files are ported:
///
///     <list type="bullet">
///       <item><c>shap/explainers/_tree.py</c> — <c>TreeEnsemble</c>/<c>SingleTree</c> ingestion of an
///         sklearn tree: <c>scaling = 1.0 / n_estimators</c>, <c>values = tree.value * scaling</c>,
///         <c>node_sample_weight = weighted_n_node_samples</c>, and the driver
///         <c>SingleTree.shap_values</c> that runs the C extension and returns <c>phi[:, :-1]</c>
///         (the per-feature contributions, dropping the bias column).</item>
///       <item><c>shap/cext/tree_shap.h</c> — <c>dense_tree_path_dependent</c> → <c>tree_shap</c> →
///         <c>tree_shap_recursive</c> with <c>extend_path</c> / <c>unwind_path</c> /
///         <c>unwound_path_sum</c> (Lundberg et al., https://arxiv.org/abs/1802.03888).</item>
///     </list>
///
///     <para><b>Faithful specialization.</b> Optuna only ever calls <c>shap_values</c> (never
///     <c>shap_interaction_values</c>), so <c>condition == 0</c> and <c>condition_fraction ≡ 1.0</c>
///     throughout; the surrogate is single-output (<c>num_outputs == 1</c>); Optuna's transformed
///     parameters are finite (no missing values, so the <c>children_default</c> branch is never taken);
///     and an sklearn tree has only numeric splits (<c>threshold_types</c> are all <c>0</c>, so the
///     categorical <c>type == 1</c> branch is never taken). Those branches are therefore elided. Every
///     surviving floating-point operation is transcribed in the C++'s exact order — multiplying by the
///     constant <c>condition_fraction = 1.0</c> is an exact no-op, so dropping it does not change a bit —
///     which is what makes the result byte-for-byte identical to running SHAP through pythonnet.</para>
///
///     <para><b>Only leaf values are read.</b> The per-feature output uses each tree's <em>leaf</em>
///     values and every node's <c>weighted_n_node_samples</c>; the tree's root/expected value feeds only
///     the bias column, which SHAP drops (<c>phi[:, :-1]</c>). So the output depends on nothing beyond
///     what the bit-exact CART port already reproduces node-for-node against scikit-learn.</para>
/// </summary>
internal static class TreeShap
{
    /// <summary>One element of the running "unique path" through the tree (a
    /// <c>PathElement</c> in <c>tree_shap.h</c>): the running permutation weights of the fraction of
    /// zero (feature absent) and one (feature present) paths.</summary>
    private struct PathElement
    {
        public int FeatureIndex;
        public double ZeroFraction;
        public double OneFraction;
        public double Pweight;
    }

    /// <summary>
    ///     The per-feature mean-model SHAP values, <c>explainer.shap_values(X)</c> — shape
    ///     <c>[nSamples][nFeatures]</c>. Each tree's contribution is pre-scaled by
    ///     <c>1 / nTrees</c> so summing the trees yields the forest mean directly (SHAP's linearity),
    ///     exactly as <c>dense_tree_path_dependent</c> accumulates into one buffer.
    /// </summary>
    internal static double[][] ShapValues(SklearnLikeTree[] forest, double[][] x, int nFeatures)
    {
        int nSamples = x.Length;

        // shap/explainers/_tree.py: scaling = 1.0 / len(model.estimators_); values *= scaling.
        double scaling = 1.0 / forest.Length;

        // The C extension is handed the ensemble max_depth (shared by every tree) to size the path
        // buffer; it does not affect the result. maxd = tree.max_depth + 2 (tree_shap()).
        int ensembleMaxDepth = 0;
        foreach (var tree in forest)
        {
            int d = MaxDepth(tree);
            if (d > ensembleMaxDepth) ensembleMaxDepth = d;
        }
        int maxd = ensembleMaxDepth + 2;
        var pathBuffer = new PathElement[maxd * (maxd + 1) / 2];

        var shap = new double[nSamples][];
        for (int i = 0; i < nSamples; i++)
        {
            // phi has one slot per feature. tree_shap.h keeps an extra bias slot at index M and adds
            // the tree's expected value to it; we omit it because the leaf loop only ever writes real
            // feature indices (path index 0 carries the sentinel feature -1 and is never read), and
            // SHAP discards that bias column from the returned values anyway.
            var phi = new double[nFeatures];
            double[] xi = x[i];
            foreach (var tree in forest)
                TreeShapPathDependent(tree, xi, phi, scaling, pathBuffer);
            shap[i] = phi;
        }

        return shap;
    }

    /// <summary><c>tree_shap()</c> for one tree and one sample (condition == 0): seed the recursion at
    /// the root with unit fractions and the sentinel parent feature -1.</summary>
    private static void TreeShapPathDependent(
        SklearnLikeTree tree, double[] x, double[] phi, double scaling, PathElement[] buffer)
    {
        Recurse(tree, x, phi, scaling, buffer,
            parentBase: 0, nodeIndex: 0, uniqueDepth: 0,
            parentZeroFraction: 1.0, parentOneFraction: 1.0, parentFeatureIndex: -1);
    }

    /// <summary>Port of <c>tree_shap_recursive</c> (condition == 0, single output, numeric splits, no
    /// missing values). <paramref name="parentBase" /> is the offset of the parent's path slice inside
    /// the shared triangular <paramref name="buffer" /> — the analogue of the pointer
    /// <c>parent_unique_path</c>.</summary>
    private static void Recurse(
        SklearnLikeTree tree, double[] x, double[] phi, double scaling, PathElement[] buffer,
        int parentBase, int nodeIndex, int uniqueDepth,
        double parentZeroFraction, double parentOneFraction, int parentFeatureIndex)
    {
        int[] childrenLeft = tree.ChildrenLeft;
        int[] childrenRight = tree.ChildrenRight;
        int[] features = tree.Feature;
        double[] thresholds = tree.Threshold;
        double[] values = tree.Value;
        double[] nodeSampleWeight = tree.WeightedNNodeSamples!;

        // unique_path = parent_unique_path + unique_depth + 1; copy the parent's [0..unique_depth].
        int b = parentBase + uniqueDepth + 1;
        Array.Copy(buffer, parentBase, buffer, b, uniqueDepth + 1);

        // condition == 0, so we always extend the path.
        ExtendPath(buffer, b, uniqueDepth, parentZeroFraction, parentOneFraction, parentFeatureIndex);

        int splitIndex = features[nodeIndex];

        // leaf node
        if (childrenRight[nodeIndex] < 0)
        {
            double leafValue = values[nodeIndex] * scaling;
            for (int i = 1; i <= uniqueDepth; i++)
            {
                double w = UnwoundPathSum(buffer, b, uniqueDepth, i);
                ref PathElement el = ref buffer[b + i];
                double scale = w * (el.OneFraction - el.ZeroFraction);
                phi[el.FeatureIndex] += scale * leafValue;
            }
            return;
        }

        // internal node: pick the branch x follows (numeric split, no missing values).
        int hotIndex = x[splitIndex] <= thresholds[nodeIndex] ? childrenLeft[nodeIndex] : childrenRight[nodeIndex];
        int coldIndex = hotIndex == childrenLeft[nodeIndex] ? childrenRight[nodeIndex] : childrenLeft[nodeIndex];
        double nodeWeight = nodeSampleWeight[nodeIndex];
        double hotZeroFraction = nodeSampleWeight[hotIndex] / nodeWeight;
        double coldZeroFraction = nodeSampleWeight[coldIndex] / nodeWeight;
        double incomingZeroFraction = 1.0;
        double incomingOneFraction = 1.0;

        // if we already split on this feature, undo that split so we can redo it for this node.
        int pathIndex = 0;
        for (; pathIndex <= uniqueDepth; ++pathIndex)
            if (buffer[b + pathIndex].FeatureIndex == splitIndex) break;
        if (pathIndex != uniqueDepth + 1)
        {
            incomingZeroFraction = buffer[b + pathIndex].ZeroFraction;
            incomingOneFraction = buffer[b + pathIndex].OneFraction;
            UnwindPath(buffer, b, uniqueDepth, pathIndex);
            uniqueDepth -= 1;
        }

        // condition == 0, so hot/cold condition fractions stay 1 and unique_depth is not adjusted.
        Recurse(tree, x, phi, scaling, buffer, b, hotIndex, uniqueDepth + 1,
            hotZeroFraction * incomingZeroFraction, incomingOneFraction, splitIndex);

        Recurse(tree, x, phi, scaling, buffer, b, coldIndex, uniqueDepth + 1,
            coldZeroFraction * incomingZeroFraction, 0.0, splitIndex);
    }

    /// <summary>Port of <c>extend_path</c>: grow the decision path with one more split, keeping the
    /// permutation weights of every subset size up to date.</summary>
    private static void ExtendPath(
        PathElement[] path, int b, int uniqueDepth, double zeroFraction, double oneFraction, int featureIndex)
    {
        path[b + uniqueDepth].FeatureIndex = featureIndex;
        path[b + uniqueDepth].ZeroFraction = zeroFraction;
        path[b + uniqueDepth].OneFraction = oneFraction;
        path[b + uniqueDepth].Pweight = uniqueDepth == 0 ? 1.0 : 0.0;
        for (int i = uniqueDepth - 1; i >= 0; i--)
        {
            path[b + i + 1].Pweight += oneFraction * path[b + i].Pweight * (i + 1) / (double)(uniqueDepth + 1);
            path[b + i].Pweight = zeroFraction * path[b + i].Pweight * (uniqueDepth - i) / (double)(uniqueDepth + 1);
        }
    }

    /// <summary>Port of <c>unwind_path</c>: undo a previous <see cref="ExtendPath" /> so the split can
    /// be reapplied at a deeper node.</summary>
    private static void UnwindPath(PathElement[] path, int b, int uniqueDepth, int pathIndex)
    {
        double oneFraction = path[b + pathIndex].OneFraction;
        double zeroFraction = path[b + pathIndex].ZeroFraction;
        double nextOnePortion = path[b + uniqueDepth].Pweight;

        for (int i = uniqueDepth - 1; i >= 0; --i)
        {
            if (oneFraction != 0)
            {
                double tmp = path[b + i].Pweight;
                path[b + i].Pweight = nextOnePortion * (uniqueDepth + 1) / (double)((i + 1) * oneFraction);
                nextOnePortion = tmp - path[b + i].Pweight * zeroFraction * (uniqueDepth - i) / (double)(uniqueDepth + 1);
            }
            else
            {
                path[b + i].Pweight = path[b + i].Pweight * (uniqueDepth + 1) / (zeroFraction * (uniqueDepth - i));
            }
        }

        for (int i = pathIndex; i < uniqueDepth; ++i)
        {
            path[b + i].FeatureIndex = path[b + i + 1].FeatureIndex;
            path[b + i].ZeroFraction = path[b + i + 1].ZeroFraction;
            path[b + i].OneFraction = path[b + i + 1].OneFraction;
        }
    }

    /// <summary>Port of <c>unwound_path_sum</c>: the total permutation weight that would result from
    /// unwinding the split at <paramref name="pathIndex" /> — used to weight each leaf's contribution.</summary>
    private static double UnwoundPathSum(PathElement[] path, int b, int uniqueDepth, int pathIndex)
    {
        double oneFraction = path[b + pathIndex].OneFraction;
        double zeroFraction = path[b + pathIndex].ZeroFraction;
        double nextOnePortion = path[b + uniqueDepth].Pweight;
        double total = 0;

        if (oneFraction != 0)
        {
            for (int i = uniqueDepth - 1; i >= 0; --i)
            {
                double tmp = nextOnePortion / (double)((i + 1) * oneFraction);
                total += tmp;
                nextOnePortion = path[b + i].Pweight - tmp * zeroFraction * (uniqueDepth - i);
            }
        }
        else
        {
            for (int i = uniqueDepth - 1; i >= 0; --i)
                total += path[b + i].Pweight / (zeroFraction * (uniqueDepth - i));
        }

        return total * (uniqueDepth + 1);
    }

    /// <summary>Depth of the deepest root-to-leaf path (root = 0), matching
    /// <c>sklearn.tree._tree.Tree.max_depth</c>; used only to size the path buffer.</summary>
    private static int MaxDepth(SklearnLikeTree tree)
    {
        int[] left = tree.ChildrenLeft;
        int[] right = tree.ChildrenRight;
        return Depth(0);

        int Depth(int node)
        {
            if (right[node] < 0) return 0;
            int dl = Depth(left[node]);
            int dr = Depth(right[node]);
            return 1 + (dl > dr ? dl : dr);
        }
    }
}
