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
    internal static SklearnLikeTree[] FitFastTree(
        float[,] x, double[] y, int nFeatures, FastTreeRegressionTrainer.Options options)
    {
        var ml = new MLContext();
        IDataView dataView = LoadData(ml, x, y, nFeatures);
        options.LabelColumnName = LabelCol;
        options.FeatureColumnName = FeatureCol;
        var model = ml.Regression.Trainers.FastTree(options).Fit(dataView);
        return Adapt(model.Model.TrainedTreeEnsemble.Trees, nFeatures, x, y);
    }

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
    internal static SklearnLikeTree[] FitFastTreeTweedie(
        float[,] x, double[] y, int nFeatures, FastTreeTweedieTrainer.Options options)
    {
        var ml = new MLContext();
        IDataView dataView = LoadData(ml, x, y, nFeatures);
        options.LabelColumnName = LabelCol;
        options.FeatureColumnName = FeatureCol;
        var model = ml.Regression.Trainers.FastTreeTweedie(options).Fit(dataView);
        return Adapt(model.Model.TrainedTreeEnsemble.Trees, nFeatures, x, y);
    }

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

    private static SklearnLikeTree[] Adapt(
        IReadOnlyList<RegressionTreeBase> trees, int nFeatures, float[,] x, double[] y)
        => trees.Select(tree => MlNetTreeEnsembleAdapter.ToSklearnLikeTree(tree, nFeatures, x, y)).ToArray();

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
