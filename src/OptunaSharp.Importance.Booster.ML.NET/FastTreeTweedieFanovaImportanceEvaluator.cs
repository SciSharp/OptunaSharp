using System;
using System.Collections.Generic;
using Microsoft.ML.Trainers.FastTree;
using OptunaSharp.Collections;
using OptunaSharp.Importance.Forests.Internal;

namespace OptunaSharp.Importance.Booster;

/// <summary>
///     <b>Experimental / non-Optuna.</b> fANOVA importance evaluator backed by <b>ML.NET's
///     FastTreeTweedie</b> (Tweedie-loss gradient boosting). It reuses the shared fANOVA
///     marginal-variance walk, but fits a <i>gradient-boosted</i> ensemble under a Tweedie loss (a
///     compound Poisson-gamma model normally used for non-negative, zero-inflated targets).
///
///     <para><b>Off-label — this is not what Optuna does.</b> Optuna's importance uses only
///     scikit-learn's <c>RandomForestRegressor</c> (bagging); there is no boosting, and certainly no
///     Tweedie loss, anywhere in it. Boosting trees fit the ensemble's <i>residual</i>, not the
///     objective, so fANOVA over them measures a different, non-standard quantity. There is <b>no
///     value-parity with Python Optuna</b> and only weak ranking agreement. Prefer Core's canonical,
///     value-exact <see cref="FanovaImportanceEvaluator" /> when parity matters. Provided purely for
///     experimentation.</para>
/// </summary>
public sealed class FastTreeTweedieFanovaImportanceEvaluator : BaseImportanceEvaluator
{
    private readonly FastTreeTweedieTrainer.Options? _options;
    private readonly Action<FastTreeTweedieTrainer.Options>? _configure;
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int? _seed;

    /// <param name="nTrees">The number of boosting iterations (trees).</param>
    /// <param name="maxDepth">The maximum depth of each tree.</param>
    /// <param name="seed">Controls the randomness of the ensemble. Specify a value for determinism.</param>
    public FastTreeTweedieFanovaImportanceEvaluator(int nTrees = 64, int maxDepth = 64, int? seed = null)
    {
        _nTrees = nTrees;
        _maxDepth = maxDepth;
        _seed = seed;
    }

    /// <summary>
    ///     Full-control overload: fit with a caller-supplied
    ///     <see cref="FastTreeTweedieTrainer.Options" />, exposing the <b>entire</b> ML.NET
    ///     FastTreeTweedie API — including the Tweedie <c>Index</c> (variance-power), <c>LearningRate</c>,
    ///     bagging, pruning, <c>Seed</c>, <c>NumberOfThreads</c>, … — rather than only
    ///     <c>(n_trees, max_depth, seed)</c>.
    /// </summary>
    /// <param name="options">
    ///     The FastTreeTweedie trainer options. <c>LabelColumnName</c> and <c>FeatureColumnName</c> are
    ///     managed by the evaluator (bound to the transformed-trial schema) and are overwritten; every
    ///     other field is used as given. For a reproducible result set <c>Seed</c> and
    ///     <c>NumberOfThreads = 1</c>.
    /// </param>
    public FastTreeTweedieFanovaImportanceEvaluator(FastTreeTweedieTrainer.Options options)
        => _options = options ?? throw new ArgumentNullException(nameof(options));

    /// <summary>
    ///     Convenience between the two ctors above: start from the reproducible default options for
    ///     <c>(nTrees, maxDepth, seed)</c> and <paramref name="configure" /> only the knobs you care
    ///     about (e.g. the Tweedie <c>Index</c> or LearningRate) — without restating the defaults or the
    ///     two managed column names.
    /// </summary>
    /// <param name="configure">Applied to the default options just before fitting.</param>
    /// <param name="nTrees">The number of boosting iterations (the options baseline).</param>
    /// <param name="maxDepth">The maximum depth of each tree (derives the baseline leaf count).</param>
    /// <param name="seed">Baseline seed; may be overridden inside <paramref name="configure" />.</param>
    public FastTreeTweedieFanovaImportanceEvaluator(
        Action<FastTreeTweedieTrainer.Options> configure, int nTrees = 64, int maxDepth = 64, int? seed = null)
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
                ? BoosterBackend.FitFastTreeTweedie(x, y, nFeatures, _nTrees, _maxDepth, _seed, _configure)
                : BoosterBackend.FitFastTreeTweedie(x, y, nFeatures, _options));
}
