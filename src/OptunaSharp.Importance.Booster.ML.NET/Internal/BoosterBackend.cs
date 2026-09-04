using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using OptunaSharp.Importance.Forests.Internal;
using OptunaSharp.Importance.MLNet;

namespace OptunaSharp.Importance.Booster;

/// <summary>
///     The ML.NET gradient-boosting backends (FastTree = MART, FastTreeTweedie = Tweedie-loss MART) for
///     the forest-based importance evaluators. Each fits a boosted-tree ensemble and adapts every tree
///     into Core's <see cref="SklearnLikeTree" /> via the shared <see cref="MlNetTreeEnsembleAdapter" />
///     (pre-order DFS for the flat node arrays, then the training set is routed through each tree to
///     recompute per-node impurity + counts). The tree structure is the same shape whether the trees
///     came from bagging or boosting, so the fANOVA walk and MDI read it uniformly.
///
///     <para><b>Experimental / off-label.</b> Optuna's importance uses only scikit-learn's
///     <c>RandomForestRegressor</c> (bagging, trees fit the objective); there is no boosting anywhere
///     in it. A boosting tree fits the ensemble's <i>residual</i>, so fANOVA/MDI over a boosted
///     ensemble measures a different, non-standard quantity with no value-parity to Optuna. Prefer
///     Core's canonical evaluators. For reproducibility the ensemble is fitted single-threaded from the
///     given seed.</para>
/// </summary>
internal static class BoosterBackend
{
    private const string LabelCol = "Label";
    private const string FeatureCol = "Features";

    /// <summary>Fit FastTree (MART — least-squares gradient boosting) and adapt its trees.</summary>
    internal static SklearnLikeTree[] FitFastTree(
        float[,] x, double[] y, int nFeatures, int nTrees, int maxDepth, int? seed)
    {
        var (ml, dataView, numberOfLeaves) = BuildData(x, y, nFeatures, maxDepth, seed);
        var options = new FastTreeRegressionTrainer.Options
        {
            LabelColumnName = LabelCol,
            FeatureColumnName = FeatureCol,
            NumberOfTrees = nTrees,
            NumberOfLeaves = numberOfLeaves,
            MinimumExampleCountPerLeaf = 1,
            FeatureFraction = 1.0,
            NumberOfThreads = 1,         // determinism + the NumSharp single-threaded-kernel rule
        };
        if (seed.HasValue)
        {
            options.Seed = seed.Value;
            options.FeatureSelectionSeed = seed.Value;
        }

        var model = ml.Regression.Trainers.FastTree(options).Fit(dataView);
        return Adapt(model.Model.TrainedTreeEnsemble.Trees, nFeatures, x, y);
    }

    /// <summary>Fit FastTreeTweedie (Tweedie-loss gradient boosting) and adapt its trees.</summary>
    internal static SklearnLikeTree[] FitFastTreeTweedie(
        float[,] x, double[] y, int nFeatures, int nTrees, int maxDepth, int? seed)
    {
        var (ml, dataView, numberOfLeaves) = BuildData(x, y, nFeatures, maxDepth, seed);
        var options = new FastTreeTweedieTrainer.Options
        {
            LabelColumnName = LabelCol,
            FeatureColumnName = FeatureCol,
            NumberOfTrees = nTrees,
            NumberOfLeaves = numberOfLeaves,
            MinimumExampleCountPerLeaf = 1,
            FeatureFraction = 1.0,
            NumberOfThreads = 1,
        };
        if (seed.HasValue)
        {
            options.Seed = seed.Value;
            options.FeatureSelectionSeed = seed.Value;
        }

        var model = ml.Regression.Trainers.FastTreeTweedie(options).Fit(dataView);
        return Adapt(model.Model.TrainedTreeEnsemble.Trees, nFeatures, x, y);
    }

    private static SklearnLikeTree[] Adapt(
        IReadOnlyList<RegressionTreeBase> trees, int nFeatures, float[,] x, double[] y)
        => trees.Select(tree => MlNetTreeEnsembleAdapter.ToSklearnLikeTree(tree, nFeatures, x, y)).ToArray();

    private static (MLContext Ml, IDataView DataView, int NumberOfLeaves) BuildData(
        float[,] x, double[] y, int nFeatures, int maxDepth, int? seed)
    {
        int n = y.Length;
        var ml = new MLContext(seed: seed);

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
        IDataView dataView = ml.Data.LoadFromEnumerable(rows, schema);

        // A depth-d binary tree has at most 2^d leaves; a tree can never have more leaves than rows.
        int leafCap = Math.Max(2, n);
        int fromDepth = maxDepth >= 30 ? int.MaxValue : (1 << maxDepth);
        int numberOfLeaves = Math.Max(2, Math.Min(fromDepth, leafCap));
        return (ml, dataView, numberOfLeaves);
    }

    private sealed class InputRow
    {
        public float[] Features { get; set; } = Array.Empty<float>();
        public float Label { get; set; }
    }
}
