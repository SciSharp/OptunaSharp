using Microsoft.ML.Trainers.FastTree;
using OptunaSharp.Importance.Forests.Internal;

namespace OptunaSharp.Importance.MLNet;

/// <summary>
///     Adapts an ML.NET tree ensemble into Core's <see cref="SklearnLikeTree" />s — the shared shape
///     read by the fANOVA marginal-variance walk and MDI. This is a <b>linked source file</b> compiled
///     into every ML.NET importance backend (<c>OptunaSharp.Integration.ML.NET</c> — the FastForest and
///     FastTree/FastTreeTweedie ensembles — and <c>OptunaSharp.Integration.LightGbm</c>) rather than a
///     shared runtime package, so no package takes a dependency on another and Core stays pure managed.
///     Each host assembly is a Core friend (<c>InternalsVisibleTo</c>), which is what grants access to
///     the internal <see cref="SklearnLikeTree" />.
///
///     <para><see cref="RegressionTreeBase" /> is the common base of <b>every</b> ML.NET tree ensemble:
///     FastForest (bagging), FastTree / FastTreeTweedie (gradient boosting) and LightGBM all expose
///     their trees as <c>RegressionTreeBase</c> with <c>LeftChild / RightChild /
///     NumericalSplitFeatureIndexes / NumericalSplitThresholds / LeafValues / NumberOfNodes /
///     NumberOfLeaves</c>, so one adapter serves them all.</para>
///
///     <para><b>Documented divergence.</b> An ML.NET ensemble is not tree-identical to scikit-learn
///     (binned features, leaf-wise growth, float32 thresholds; per-node statistics are recomputed by
///     routing the training set rather than stored per-tree bootstrap counts), so the resulting
///     importances differ numerically from Python Optuna — the ranking of a dominant parameter is
///     robust, exact values are not. For gradient boosters (FastTree / FastTreeTweedie / LightGBM) the
///     trees fit residuals, so importance over them is off-label for fANOVA/MDI (a different, non-Optuna
///     quantity); prefer Core's canonical, value-exact evaluators when Optuna parity matters.</para>
/// </summary>
internal static class MlNetTreeEnsembleAdapter
{
    /// <summary>
    ///     Adapt an ML.NET <c>RegressionTree</c> into a <see cref="SklearnLikeTree" /> by a pre-order
    ///     DFS (parents before children — sklearn's numbering), then route the training set to fill
    ///     in per-node impurity + weighted counts.
    /// </summary>
    internal static SklearnLikeTree ToSklearnLikeTree(RegressionTreeBase tree, int nFeatures, float[,] x, double[] y)
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
}
