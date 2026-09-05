using System;
using System.Collections.Generic;
using Microsoft.ML.Trainers.FastTree;
using OptunaSharp.Collections;
using OptunaSharp.Importance.Forests.Internal;
using OptunaSharp.Importance.Forests.MLNet;

namespace OptunaSharp.Importance.Forests;

/// <summary>
///     fANOVA importance evaluator backed by <b>ML.NET's FastForest</b> — the alternative forest
///     backend to Core's pure-managed, scikit-learn-bit-exact
///     <see cref="FanovaImportanceEvaluator" />. Same fANOVA algorithm
///     (<c>optuna/importance/_fanova</c>, shared with Core), different random forest.
///
///     <para><b>Divergence from Python Optuna.</b> FastForest is not tree-identical to scikit-learn,
///     so importance <i>values</i> differ from Optuna (a dominant parameter's ranking is robust,
///     exact values are not) — a documented Tier X divergence. Prefer
///     <see cref="FanovaImportanceEvaluator" /> when Optuna parity matters; use this backend when you
///     want ML.NET's forest (e.g. a familiar/consistent model across an ML.NET pipeline).</para>
/// </summary>
public sealed class FastForestFanovaImportanceEvaluator : BaseImportanceEvaluator
{
    private readonly FastForestRegressionTrainer.Options? _options;
    private readonly Action<FastForestRegressionTrainer.Options>? _configure;
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int? _seed;

    /// <param name="nTrees">The number of trees in the forest.</param>
    /// <param name="maxDepth">The maximum depth of the trees in the forest.</param>
    /// <param name="seed">Controls the randomness of the forest. Specify a value for determinism.</param>
    public FastForestFanovaImportanceEvaluator(int nTrees = 64, int maxDepth = 64, int? seed = null)
    {
        _nTrees = nTrees;
        _maxDepth = maxDepth;
        _seed = seed;
    }

    /// <summary>
    ///     Full-control overload: fit the forest with a caller-supplied
    ///     <see cref="FastForestRegressionTrainer.Options" />, exposing the <b>entire</b> ML.NET
    ///     FastForest API (<c>NumberOfTrees</c>, <c>NumberOfLeaves</c>, <c>FeatureFraction</c>, bagging,
    ///     <c>MaximumBinCountPerFeature</c>, <c>Seed</c>, <c>NumberOfThreads</c>, …) rather than only
    ///     <c>(n_trees, max_depth, seed)</c>.
    /// </summary>
    /// <param name="options">
    ///     The FastForest trainer options. <c>LabelColumnName</c> and <c>FeatureColumnName</c> are
    ///     managed by the evaluator (bound to the transformed-trial schema) and are overwritten; every
    ///     other field is used as given. For a reproducible result set <c>Seed</c> and
    ///     <c>NumberOfThreads = 1</c>.
    /// </param>
    public FastForestFanovaImportanceEvaluator(FastForestRegressionTrainer.Options options)
        => _options = options ?? throw new ArgumentNullException(nameof(options));

    /// <summary>
    ///     Convenience between the two ctors above: start from the reproducible default options for
    ///     <c>(nTrees, maxDepth, seed)</c> and <paramref name="configure" /> only the knobs you care
    ///     about (e.g. FeatureFraction, or Seed for a reproducible result) — without restating the
    ///     defaults or the two managed column names. The whole FastForest options surface is reachable.
    /// </summary>
    /// <param name="configure">Applied to the default options just before fitting.</param>
    /// <param name="nTrees">The number of trees in the forest (the options baseline).</param>
    /// <param name="maxDepth">The maximum depth of the trees (derives the baseline leaf count).</param>
    /// <param name="seed">Baseline seed; may be overridden inside <paramref name="configure" />.</param>
    public FastForestFanovaImportanceEvaluator(
        Action<FastForestRegressionTrainer.Options> configure, int nTrees = 64, int maxDepth = 64, int? seed = null)
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
                ? FastForestBackend.Fit(x, y, nFeatures, _nTrees, _maxDepth, _seed, _configure)
                : FastForestBackend.Fit(x, y, nFeatures, _options));
}
