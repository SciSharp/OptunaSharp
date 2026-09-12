using System;
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
///     <para>Two fit overloads back the two evaluator constructors: a <b>simple</b> one that derives a
///     reproducible <see cref="FastForestRegressionTrainer.Options" /> from
///     <c>(n_trees, max_depth, seed)</c>, and a <b>full-API</b> one that takes a caller-supplied
///     <see cref="FastForestRegressionTrainer.Options" /> verbatim (only the two schema column names
///     are managed). The <see cref="MLContext" /> seed is left unset: with <c>options.Seed</c> fixed
///     and a single training thread the result is independent of it, so reproducibility is driven
///     entirely by the options — which is what puts the caller in full control on the options path.</para>
///
///     <para><b>Documented divergence.</b> FastForest is not tree-identical to scikit-learn (binned
///     features, leaf-wise growth, float32 thresholds; routed impurities rather than per-tree
///     bootstrap), so importances differ numerically from Python Optuna — the ranking of a dominant
///     parameter is robust, exact values are not.</para>
/// </summary>
internal static class FastForestBackend
{
    private const string LabelCol = "Label";
    private const string FeatureCol = "Features";

    /// <summary>
    ///     Simple path — build the reproducible default options from <c>(n_trees, max_depth, seed)</c>,
    ///     then apply the optional <paramref name="configure" /> tweak (start-from-defaults) before fitting.
    /// </summary>
    /// <param name="x">The float32 training matrix, with samples in rows and transformed features in columns.</param>
    /// <param name="y">One target or class-label value per sample, in training-row order.</param>
    /// <param name="nFeatures">The number of transformed feature columns in each training row.</param>
    /// <param name="nTrees">The number of ensemble trees or boosting iterations requested.</param>
    /// <param name="maxDepth">The maximum tree depth used to derive the bounded leaf count.</param>
    /// <param name="seed">The training or fold-shuffling seed, as specified by the operation.</param>
    /// <param name="configure">An optional callback customizing the derived trainer options before fitting.</param>
    /// <returns>The fitted ensemble adapted to the shared importance-evaluation tree representation.</returns>
    internal static SklearnLikeTree[] Fit(
        float[,] x, double[] y, int nFeatures, int nTrees, int maxDepth, int? seed,
        Action<FastForestRegressionTrainer.Options>? configure = null)
    {
        var options = DefaultOptions(nTrees, maxDepth, seed, y.Length);
        configure?.Invoke(options);
        return Fit(x, y, nFeatures, options);
    }

    /// <summary>
    ///     Full-API path — fit with a caller-supplied <see cref="FastForestRegressionTrainer.Options" />.
    ///     Only <c>LabelColumnName</c> /
    ///     <c>FeatureColumnName</c> are overwritten (the fixed schema the transformed trials live in);
    ///     every other knob — <c>NumberOfTrees</c>, <c>NumberOfLeaves</c>, <c>FeatureFraction</c>,
    ///     bagging, <c>Seed</c>, <c>NumberOfThreads</c>, … — is the caller's.
    /// </summary>
    /// <param name="x">The float32 training matrix, with samples in rows and transformed features in columns.</param>
    /// <param name="y">One target or class-label value per sample, in training-row order.</param>
    /// <param name="nFeatures">The number of transformed feature columns in each training row.</param>
    /// <param name="options">The native trainer options; label and feature column names are managed by this adapter.</param>
    /// <returns>The fitted ensemble adapted to the shared importance-evaluation tree representation.</returns>
    internal static SklearnLikeTree[] Fit(float[,] x, double[] y, int nFeatures, FastForestRegressionTrainer.Options options)
    {
        RequireImportanceSchema(options.ExampleWeightColumnName, options.RowGroupColumnName);

        var mlContext = new MLContext();
        IDataView dataView = LoadData(mlContext, x, y, nFeatures);

        options.LabelColumnName = LabelCol;
        options.FeatureColumnName = FeatureCol;

        var model = mlContext.Regression.Trainers.FastForest(options).Fit(dataView);
        return model.Model.TrainedTreeEnsemble.Trees
            .Select(tree => MlNetTreeEnsembleAdapter.ToSklearnLikeTree(tree, nFeatures, x, y))
            .ToArray();
    }

    /// <summary>The reproducible defaults the <c>(n_trees, max_depth, seed)</c> overload fits with.</summary>
    /// <param name="nTrees">The number of ensemble trees or boosting iterations requested.</param>
    /// <param name="maxDepth">The maximum tree depth used to derive the bounded leaf count.</param>
    /// <param name="seed">The training or fold-shuffling seed, as specified by the operation.</param>
    /// <param name="n">The number of training rows used to bound leaves or construct folds.</param>
    /// <returns>The configured native trainer options with deterministic single-thread settings.</returns>
    private static FastForestRegressionTrainer.Options DefaultOptions(int nTrees, int maxDepth, int? seed, int n)
    {
        var options = new FastForestRegressionTrainer.Options
        {
            NumberOfTrees = nTrees,
            NumberOfLeaves = LeavesFromDepth(maxDepth, n),
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
        return options;
    }

    // The importance transform hands the trainer only a label + a feature vector — there is no
    // per-trial weight or group column — so these two of the trainer's four column-name options cannot
    // bind. Fail fast with a clear message instead of ML.NET's cryptic "column not found" at Fit time.
    /// <summary>Rejects weight or group columns absent from the transformed-trial importance dataset.</summary>
    /// <param name="exampleWeightColumnName">An optional requested weight column; importance training has no such column and requires null.</param>
    /// <param name="rowGroupColumnName">An optional requested group column; importance training has no such column and requires null.</param>
    /// <exception cref="System.ArgumentException">A weight or group column is configured, but the importance data contains only features and labels.</exception>
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
    /// <summary>Bounds the binary-tree leaf count by both the requested depth and available sample count.</summary>
    /// <param name="maxDepth">The maximum tree depth used to derive the bounded leaf count.</param>
    /// <param name="n">The number of training rows used to bound leaves or construct folds.</param>
    /// <returns>A leaf count of at least two, capped by available rows and the depth-derived limit.</returns>
    private static int LeavesFromDepth(int maxDepth, int n)
    {
        int leafCap = Math.Max(2, n);
        int fromDepth = maxDepth >= 30 ? int.MaxValue : (1 << maxDepth);
        return Math.Max(2, Math.Min(fromDepth, leafCap));
    }

    /// <summary>Materializes float32 feature vectors and labels in the fixed ML.NET importance-training schema.</summary>
    /// <param name="mlContext">The ML.NET context used to split, train, or load the data.</param>
    /// <param name="x">The float32 training matrix, with samples in rows and transformed features in columns.</param>
    /// <param name="y">One target or class-label value per sample, in training-row order.</param>
    /// <param name="nFeatures">The number of transformed feature columns in each training row.</param>
    /// <returns>The typed feature/label data view used by the native trainer.</returns>
    private static IDataView LoadData(MLContext mlContext, float[,] x, double[] y, int nFeatures)
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
        return mlContext.Data.LoadFromEnumerable(rows, schema);
    }

    private sealed class InputRow
    {
        /// <summary>The row's ordered float32 feature vector exposed to the ML.NET schema.</summary>
        public float[] Features { get; set; } = Array.Empty<float>();
        /// <summary>The training label encoded in the trainer's required scalar type.</summary>
        public float Label { get; set; }
    }
}
