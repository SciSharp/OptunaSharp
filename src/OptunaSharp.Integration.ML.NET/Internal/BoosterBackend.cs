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
///     <para>Each trainer has two fit overloads: a <b>simple</b> one that derives reproducible default
///     options from <c>(n_trees, max_depth, seed)</c>, and a <b>full-API</b> one that takes the
///     trainer's real <c>Options</c> verbatim (only the two schema column names are managed). The
///     <see cref="MLContext" /> seed is left unset — with <c>options.Seed</c> fixed and a single thread
///     the result is independent of it, so the caller's options fully drive reproducibility.</para>
///
///     <para><b>Experimental / off-label.</b> Optuna's importance uses only scikit-learn's
///     <c>RandomForestRegressor</c> (bagging, trees fit the objective); there is no boosting anywhere
///     in it. A boosting tree fits the ensemble's <i>residual</i>, so fANOVA/MDI over a boosted
///     ensemble measures a different, non-standard quantity with no value-parity to Optuna. Prefer
///     Core's canonical evaluators.</para>
/// </summary>
internal static class BoosterBackend
{
    private const string LabelCol = "Label";
    private const string FeatureCol = "Features";

    // ---- FastTree (MART — least-squares gradient boosting) ----

    /// <summary>Simple path — reproducible defaults from <c>(n_trees, max_depth, seed)</c>, then the optional tweak.</summary>
    /// <param name="x">The float32 training matrix, with samples in rows and transformed features in columns.</param>
    /// <param name="y">One target or class-label value per sample, in training-row order.</param>
    /// <param name="nFeatures">The number of transformed feature columns in each training row.</param>
    /// <param name="nTrees">The number of ensemble trees or boosting iterations requested.</param>
    /// <param name="maxDepth">The maximum tree depth used to derive the bounded leaf count.</param>
    /// <param name="seed">The training or fold-shuffling seed, as specified by the operation.</param>
    /// <param name="configure">An optional callback customizing the derived trainer options before fitting.</param>
    /// <returns>The fitted FastTree ensemble adapted for importance evaluation.</returns>
    internal static SklearnLikeTree[] FitFastTree(
        float[,] x, double[] y, int nFeatures, int nTrees, int maxDepth, int? seed,
        Action<FastTreeRegressionTrainer.Options>? configure = null)
    {
        var options = DefaultFastTreeOptions(nTrees, maxDepth, seed, y.Length);
        configure?.Invoke(options);
        return FitFastTree(x, y, nFeatures, options);
    }

    /// <summary>
    ///     Full-API path — fit FastTree with a caller-supplied
    ///     <see cref="FastTreeRegressionTrainer.Options" /> (only the two schema column names are
    ///     overwritten; <c>LearningRate</c>, <c>Shrinkage</c>, bagging, <c>NumberOfLeaves</c>, pruning,
    ///     <c>Seed</c>, <c>NumberOfThreads</c>, … are the caller's).
    /// </summary>
    /// <param name="x">The float32 training matrix, with samples in rows and transformed features in columns.</param>
    /// <param name="y">One target or class-label value per sample, in training-row order.</param>
    /// <param name="nFeatures">The number of transformed feature columns in each training row.</param>
    /// <param name="options">The native trainer options; label and feature column names are managed by this adapter.</param>
    /// <returns>The fitted FastTree ensemble adapted for importance evaluation.</returns>
    internal static SklearnLikeTree[] FitFastTree(
        float[,] x, double[] y, int nFeatures, FastTreeRegressionTrainer.Options options)
    {
        RequireImportanceSchema(options.ExampleWeightColumnName, options.RowGroupColumnName);
        var ml = new MLContext();
        IDataView dataView = LoadData(ml, x, y, nFeatures);
        options.LabelColumnName = LabelCol;
        options.FeatureColumnName = FeatureCol;
        var model = ml.Regression.Trainers.FastTree(options).Fit(dataView);
        return Adapt(model.Model.TrainedTreeEnsemble.Trees, nFeatures, x, y);
    }

    /// <summary>Builds reproducible FastTree options from the requested tree count, depth, and seed.</summary>
    /// <param name="nTrees">The number of ensemble trees or boosting iterations requested.</param>
    /// <param name="maxDepth">The maximum tree depth used to derive the bounded leaf count.</param>
    /// <param name="seed">The training or fold-shuffling seed, as specified by the operation.</param>
    /// <param name="n">The number of training rows used to bound leaves or construct folds.</param>
    /// <returns>The configured native FastTree options.</returns>
    private static FastTreeRegressionTrainer.Options DefaultFastTreeOptions(int nTrees, int maxDepth, int? seed, int n)
    {
        var options = new FastTreeRegressionTrainer.Options
        {
            NumberOfTrees = nTrees,
            NumberOfLeaves = LeavesFromDepth(maxDepth, n),
            MinimumExampleCountPerLeaf = 1,
            FeatureFraction = 1.0,
            NumberOfThreads = 1,         // determinism + the NumSharp single-threaded-kernel rule
        };
        if (seed.HasValue)
        {
            options.Seed = seed.Value;
            options.FeatureSelectionSeed = seed.Value;
        }
        return options;
    }

    // ---- FastTreeTweedie (Tweedie-loss gradient boosting) ----

    /// <summary>Simple path — reproducible defaults from <c>(n_trees, max_depth, seed)</c>, then the optional tweak.</summary>
    /// <param name="x">The float32 training matrix, with samples in rows and transformed features in columns.</param>
    /// <param name="y">One target or class-label value per sample, in training-row order.</param>
    /// <param name="nFeatures">The number of transformed feature columns in each training row.</param>
    /// <param name="nTrees">The number of ensemble trees or boosting iterations requested.</param>
    /// <param name="maxDepth">The maximum tree depth used to derive the bounded leaf count.</param>
    /// <param name="seed">The training or fold-shuffling seed, as specified by the operation.</param>
    /// <param name="configure">An optional callback customizing the derived trainer options before fitting.</param>
    /// <returns>The fitted Tweedie ensemble adapted for importance evaluation.</returns>
    internal static SklearnLikeTree[] FitFastTreeTweedie(
        float[,] x, double[] y, int nFeatures, int nTrees, int maxDepth, int? seed,
        Action<FastTreeTweedieTrainer.Options>? configure = null)
    {
        var options = DefaultFastTreeTweedieOptions(nTrees, maxDepth, seed, y.Length);
        configure?.Invoke(options);
        return FitFastTreeTweedie(x, y, nFeatures, options);
    }

    /// <summary>
    ///     Full-API path — fit FastTreeTweedie with a caller-supplied
    ///     <see cref="FastTreeTweedieTrainer.Options" /> (only the two schema column names are
    ///     overwritten; every knob including the Tweedie <c>Index</c> is the caller's).
    /// </summary>
    /// <param name="x">The float32 training matrix, with samples in rows and transformed features in columns.</param>
    /// <param name="y">One target or class-label value per sample, in training-row order.</param>
    /// <param name="nFeatures">The number of transformed feature columns in each training row.</param>
    /// <param name="options">The native trainer options; label and feature column names are managed by this adapter.</param>
    /// <returns>The fitted Tweedie ensemble adapted for importance evaluation.</returns>
    internal static SklearnLikeTree[] FitFastTreeTweedie(
        float[,] x, double[] y, int nFeatures, FastTreeTweedieTrainer.Options options)
    {
        RequireImportanceSchema(options.ExampleWeightColumnName, options.RowGroupColumnName);
        var ml = new MLContext();
        IDataView dataView = LoadData(ml, x, y, nFeatures);
        options.LabelColumnName = LabelCol;
        options.FeatureColumnName = FeatureCol;
        var model = ml.Regression.Trainers.FastTreeTweedie(options).Fit(dataView);
        return Adapt(model.Model.TrainedTreeEnsemble.Trees, nFeatures, x, y);
    }

    /// <summary>Builds reproducible Tweedie boosting options from the requested tree count, depth, and seed.</summary>
    /// <param name="nTrees">The number of ensemble trees or boosting iterations requested.</param>
    /// <param name="maxDepth">The maximum tree depth used to derive the bounded leaf count.</param>
    /// <param name="seed">The training or fold-shuffling seed, as specified by the operation.</param>
    /// <param name="n">The number of training rows used to bound leaves or construct folds.</param>
    /// <returns>The configured native Tweedie options.</returns>
    private static FastTreeTweedieTrainer.Options DefaultFastTreeTweedieOptions(int nTrees, int maxDepth, int? seed, int n)
    {
        var options = new FastTreeTweedieTrainer.Options
        {
            NumberOfTrees = nTrees,
            NumberOfLeaves = LeavesFromDepth(maxDepth, n),
            MinimumExampleCountPerLeaf = 1,
            FeatureFraction = 1.0,
            NumberOfThreads = 1,
        };
        if (seed.HasValue)
        {
            options.Seed = seed.Value;
            options.FeatureSelectionSeed = seed.Value;
        }
        return options;
    }

    // ---- shared ----

    /// <summary>Converts fitted ML.NET trees into the shared importance-evaluation tree representation.</summary>
    /// <param name="trees">The fitted ensemble trees in their original order.</param>
    /// <param name="nFeatures">The number of transformed feature columns in each training row.</param>
    /// <param name="x">The float32 training matrix, with samples in rows and transformed features in columns.</param>
    /// <param name="y">One target or class-label value per sample, in training-row order.</param>
    /// <returns>The adapted tree array in ensemble order.</returns>
    private static SklearnLikeTree[] Adapt(
        IReadOnlyList<RegressionTreeBase> trees, int nFeatures, float[,] x, double[] y)
        => trees.Select(tree => MlNetTreeEnsembleAdapter.ToSklearnLikeTree(tree, nFeatures, x, y)).ToArray();

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
    /// <param name="ml">The ML.NET context used to construct the typed data view.</param>
    /// <param name="x">The float32 training matrix, with samples in rows and transformed features in columns.</param>
    /// <param name="y">One target or class-label value per sample, in training-row order.</param>
    /// <param name="nFeatures">The number of transformed feature columns in each training row.</param>
    /// <returns>The typed feature/label data view used by the native trainer.</returns>
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
        /// <summary>The row's ordered float32 feature vector exposed to the ML.NET schema.</summary>
        public float[] Features { get; set; } = Array.Empty<float>();
        /// <summary>The training label encoded in the trainer's required scalar type.</summary>
        public float Label { get; set; }
    }
}
