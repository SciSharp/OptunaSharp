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
///     <para>Two fit overloads back the two evaluator constructors: a <b>simple</b> one that derives
///     reproducible default options from <c>(n_trees, max_depth, seed)</c>, and a <b>full-API</b> one
///     that takes a caller-supplied <see cref="LightGbmRegressionTrainer.Options" /> verbatim (only the
///     two schema column names are managed; the whole LightGBM surface — including the nested
///     <c>Booster</c> options — is the caller's). The <see cref="MLContext" /> seed is left unset: with
///     <c>options.Seed</c> fixed and a single thread the result is independent of it.</para>
///
///     <para><b>Experimental / off-label.</b> Optuna's importance uses only scikit-learn's
///     <c>RandomForestRegressor</c> (bagging); LightGBM is a gradient booster whose trees fit the
///     ensemble's <i>residual</i>, not the objective, so fANOVA/MDI over a LightGBM ensemble measures a
///     different, non-standard quantity with no value-parity to Optuna. LightGBM's actual Optuna role is
///     a tuner (<c>optuna.integration.lightgbm</c>) — see <see cref="LightGBMTuner" />.</para>
/// </summary>
internal static class LightGbmBackend
{
    private const string LabelCol = "Label";
    private const string FeatureCol = "Features";

    /// <summary>Simple path — reproducible defaults from <c>(n_trees, max_depth, seed)</c>, then the optional tweak.</summary>
    internal static SklearnLikeTree[] Fit(
        float[,] x, double[] y, int nFeatures, int nTrees, int maxDepth, int? seed,
        Action<LightGbmRegressionTrainer.Options>? configure = null)
    {
        var options = DefaultOptions(nTrees, maxDepth, seed, y.Length);
        configure?.Invoke(options);
        return Fit(x, y, nFeatures, options);
    }

    /// <summary>
    ///     Full-API path — fit LightGBM with a caller-supplied
    ///     <see cref="LightGbmRegressionTrainer.Options" /> (only the two schema column names are
    ///     overwritten; <c>NumberOfIterations</c>, <c>NumberOfLeaves</c>, <c>LearningRate</c>,
    ///     <c>MaximumBinCountPerFeature</c>, the nested <c>Booster</c>, <c>Seed</c>,
    ///     <c>NumberOfThreads</c>, … are the caller's).
    /// </summary>
    internal static SklearnLikeTree[] Fit(float[,] x, double[] y, int nFeatures, LightGbmRegressionTrainer.Options options)
    {
        RequireImportanceSchema(options.ExampleWeightColumnName, options.RowGroupColumnName);

        var ml = new MLContext();
        IDataView dataView = LoadData(ml, x, y, nFeatures);

        options.LabelColumnName = LabelCol;
        options.FeatureColumnName = FeatureCol;

        var model = ml.Regression.Trainers.LightGbm(options).Fit(dataView);
        return Adapt(model.Model.TrainedTreeEnsemble.Trees, nFeatures, x, y);
    }

    private static LightGbmRegressionTrainer.Options DefaultOptions(int nTrees, int maxDepth, int? seed, int n)
    {
        var options = new LightGbmRegressionTrainer.Options
        {
            NumberOfIterations = nTrees,
            NumberOfLeaves = LeavesFromDepth(maxDepth, n),
            MinimumExampleCountPerLeaf = 1,
            NumberOfThreads = 1,         // determinism + the NumSharp single-threaded-kernel rule
        };
        if (seed.HasValue)
            options.Seed = seed.Value;
        return options;
    }

    private static SklearnLikeTree[] Adapt(
        IReadOnlyList<RegressionTreeBase> trees, int nFeatures, float[,] x, double[] y)
        => trees.Select(tree => MlNetTreeEnsembleAdapter.ToSklearnLikeTree(tree, nFeatures, x, y)).ToArray();

    // The importance transform hands the trainer only a label + a feature vector — there is no
    // per-trial weight or group column — so these two of the trainer's four column-name options cannot
    // bind. Fail fast with a clear message instead of ML.NET's cryptic "column not found" at Fit time.
    private static void RequireImportanceSchema(string? exampleWeightColumnName, string? rowGroupColumnName)
    {
        if (exampleWeightColumnName is not null)
            throw new ArgumentException(
                "ExampleWeightColumnName is not supported: the importance transform provides only a label " +
                "and a feature vector. Leave it unset.", "options");
        if (rowGroupColumnName is not null)
            throw new ArgumentException(
                "RowGroupColumnName is not supported: the importance transform provides only a label and a " +
                "feature vector. Leave it unset.", "options");
    }

    // A depth-d binary tree has at most 2^d leaves; a tree can never have more leaves than rows.
    private static int LeavesFromDepth(int maxDepth, int n)
    {
        int leafCap = Math.Max(2, n);
        int fromDepth = maxDepth >= 30 ? int.MaxValue : (1 << maxDepth);
        return Math.Max(2, Math.Min(fromDepth, leafCap));
    }

    private static IDataView LoadData(MLContext ml, float[,] x, double[] y, int nFeatures)
    {
        int n = y.Length;
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
        return ml.Data.LoadFromEnumerable(rows, schema);
    }

    private sealed class InputRow
    {
        public float[] Features { get; set; } = Array.Empty<float>();
        public float Label { get; set; }
    }
}
