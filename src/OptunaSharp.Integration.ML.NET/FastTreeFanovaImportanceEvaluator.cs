using System;
using System.Collections.Generic;
using Microsoft.ML.Trainers.FastTree;
using OptunaSharp.Collections;
using OptunaSharp.Importance.Forests.Internal;

namespace OptunaSharp.Importance.Booster;

/// <summary>
///     <b>Experimental / non-Optuna.</b> fANOVA importance evaluator backed by <b>ML.NET's FastTree</b>
///     (MART — least-squares gradient boosting). It reuses the shared fANOVA marginal-variance walk
///     (<c>optuna/importance/_fanova</c>), but fits a <i>gradient-boosted</i> ensemble instead of the
///     random forest fANOVA was designed for.
///
///     <para><b>Off-label — this is not what Optuna does.</b> Optuna's importance uses only
///     scikit-learn's <c>RandomForestRegressor</c> (bagging); there is no boosting anywhere in it.
///     Boosting trees fit the ensemble's <i>residual</i>, not the objective, so running fANOVA over
///     them measures a different, non-standard quantity. There is <b>no value-parity with Python
///     Optuna</b> and only weak ranking agreement. Prefer Core's canonical, value-exact
///     <see cref="FanovaImportanceEvaluator" />, or the faithful ML.NET FastForest backend
///     (<c>OptunaSharp.Importance.Forests.FastForestFanovaImportanceEvaluator</c>), when parity
///     matters. Provided for experimentation / consistency with an ML.NET boosting pipeline.</para>
/// </summary>
public sealed class FastTreeFanovaImportanceEvaluator : BaseImportanceEvaluator
{
    private readonly FastTreeRegressionTrainer.Options? _options;
    private readonly Action<FastTreeRegressionTrainer.Options>? _configure;
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int? _seed;

    /// <param name="nTrees">The number of boosting iterations (trees).</param>
    /// <param name="maxDepth">The maximum depth of each tree.</param>
    /// <param name="seed">Controls the randomness of the ensemble. Specify a value for determinism.</param>
    /// <summary>Creates the FastTree fANOVA evaluator with the requested tree count, depth, and random seed.</summary>
    public FastTreeFanovaImportanceEvaluator(int nTrees = 64, int maxDepth = 64, int? seed = null)
    {
        _nTrees = nTrees;
        _maxDepth = maxDepth;
        _seed = seed;
    }

    /// <summary>
    ///     Full-control overload: fit with a caller-supplied
    ///     <see cref="FastTreeRegressionTrainer.Options" />, exposing the <b>entire</b> ML.NET FastTree
    ///     API — <c>LearningRate</c>, <c>Shrinkage</c>, <c>NumberOfLeaves</c>, bagging, pruning,
    ///     <c>OptimizationAlgorithm</c>, early stopping, <c>Seed</c>, <c>NumberOfThreads</c>, … — rather
    ///     than only <c>(n_trees, max_depth, seed)</c>.
    /// </summary>
    /// <param name="options">
    ///     The FastTree trainer options. <c>LabelColumnName</c> and <c>FeatureColumnName</c> are managed
    ///     by the evaluator (bound to the transformed-trial schema) and are overwritten; every other
    ///     field is used as given. For a reproducible result set <c>Seed</c> and <c>NumberOfThreads = 1</c>.
    /// </param>
    /// <exception cref="System.ArgumentNullException"><paramref name="options" /> is null.</exception>
    public FastTreeFanovaImportanceEvaluator(FastTreeRegressionTrainer.Options options)
        => _options = options ?? throw new ArgumentNullException(nameof(options));

    /// <summary>
    ///     Convenience between the two ctors above: start from the reproducible default options for
    ///     <c>(nTrees, maxDepth, seed)</c> and <paramref name="configure" /> only the knobs you care
    ///     about (e.g. LearningRate) — without restating the defaults or the two managed column names.
    /// </summary>
    /// <param name="configure">Applied to the default options just before fitting.</param>
    /// <param name="nTrees">The number of boosting iterations (the options baseline).</param>
    /// <param name="maxDepth">The maximum depth of each tree (derives the baseline leaf count).</param>
    /// <param name="seed">Baseline seed; may be overridden inside <paramref name="configure" />.</param>
    /// <exception cref="System.ArgumentNullException"><paramref name="configure" /> is null.</exception>
    public FastTreeFanovaImportanceEvaluator(
        Action<FastTreeRegressionTrainer.Options> configure, int nTrees = 64, int maxDepth = 64, int? seed = null)
    {
        _configure = configure ?? throw new ArgumentNullException(nameof(configure));
        _nTrees = nTrees;
        _maxDepth = maxDepth;
        _seed = seed;
    }

    /// <inheritdoc />
    public override OrderedDict<string, double> Evaluate(
        Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null)
        => ForestImportance.EvaluateFanova(study, @params, target,
            (x, y, nFeatures) => _options is null
                ? BoosterBackend.FitFastTree(x, y, nFeatures, _nTrees, _maxDepth, _seed, _configure)
                : BoosterBackend.FitFastTree(x, y, nFeatures, _options));
}
