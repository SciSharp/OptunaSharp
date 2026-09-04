using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using OptunaSharp.Importance.Forests.Internal;
using OptunaSharp.Importance.MLNet;

namespace OptunaSharp.Integration.LightGbm;

/// <summary>
///     The ML.NET LightGBM backend for the forest-based importance evaluators. LightGBM is a
///     histogram-based gradient-boosting machine; this fits a LightGBM regression ensemble and adapts
///     every tree into Core's <see cref="SklearnLikeTree" /> via the shared
///     <see cref="MlNetTreeEnsembleAdapter" /> (pre-order DFS for the flat node arrays, then the
///     training set is routed through each tree to recompute per-node impurity + counts). ML.NET
///     re-exposes LightGBM's trees as the same <c>RegressionTreeBase</c> shape as FastForest/FastTree,
///     so the fANOVA walk and MDI read them uniformly.
///
///     <para><b>Experimental / off-label.</b> Optuna's importance uses only scikit-learn's
///     <c>RandomForestRegressor</c> (bagging); LightGBM is a gradient booster whose trees fit the
///     ensemble's <i>residual</i>, not the objective, so fANOVA/MDI over a LightGBM ensemble measures a
///     different, non-standard quantity with no value-parity to Optuna. LightGBM's actual Optuna role is
///     a tuner (<c>optuna.integration.lightgbm</c>, Phase 5.7). For reproducibility the ensemble is
///     fitted single-threaded from the given seed.</para>
/// </summary>
internal static class LightGbmBackend
{
    private const string LabelCol = "Label";
    private const string FeatureCol = "Features";

    internal static SklearnLikeTree[] Fit(
        float[,] x, double[] y, int nFeatures, int nTrees, int maxDepth, int? seed)
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

        var options = new LightGbmRegressionTrainer.Options
        {
            LabelColumnName = LabelCol,
            FeatureColumnName = FeatureCol,
            NumberOfIterations = nTrees,
            NumberOfLeaves = numberOfLeaves,
            MinimumExampleCountPerLeaf = 1,
            NumberOfThreads = 1,         // determinism + the NumSharp single-threaded-kernel rule
        };
        if (seed.HasValue)
            options.Seed = seed.Value;

        var model = ml.Regression.Trainers.LightGbm(options).Fit(dataView);
        return Adapt(model.Model.TrainedTreeEnsemble.Trees, nFeatures, x, y);
    }

    private static SklearnLikeTree[] Adapt(
        IReadOnlyList<RegressionTreeBase> trees, int nFeatures, float[,] x, double[] y)
        => trees.Select(tree => MlNetTreeEnsembleAdapter.ToSklearnLikeTree(tree, nFeatures, x, y)).ToArray();

    private sealed class InputRow
    {
        public float[] Features { get; set; } = Array.Empty<float>();
        public float Label { get; set; }
    }
}
