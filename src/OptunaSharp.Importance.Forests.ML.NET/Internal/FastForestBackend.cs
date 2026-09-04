using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using OptunaSharp.Importance.Forests.Internal;
using OptunaSharp.Importance.MLNet;

namespace OptunaSharp.Importance.Forests.MLNet;

/// <summary>
///     The ML.NET FastForest backend for the forest-based importance evaluators — the alternative to
///     Core's pure-managed, scikit-learn-bit-exact CART. It fits <c>FastForest</c> and adapts each
///     tree into Core's <see cref="SklearnLikeTree" /> (the same structure the fANOVA walk and MDI
///     read) via the shared <see cref="MlNetTreeEnsembleAdapter" />: a pre-order DFS gives the
///     sklearn-flat node arrays, and the training set is routed through each tree to fill in the
///     per-node impurity (variance) and weighted counts that MDI needs (ML.NET does not expose
///     per-node bootstrap statistics).
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

        return ensemble.Trees
            .Select(tree => MlNetTreeEnsembleAdapter.ToSklearnLikeTree(tree, nFeatures, x, y))
            .ToArray();
    }

    private sealed class InputRow
    {
        public float[] Features { get; set; } = Array.Empty<float>();
        public float Label { get; set; }
    }
}
