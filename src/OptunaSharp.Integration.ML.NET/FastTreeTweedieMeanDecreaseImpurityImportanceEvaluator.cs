using System;
using System.Collections.Generic;
using Microsoft.ML.Trainers.FastTree;
using OptunaSharp.Collections;
using OptunaSharp.Importance.Forests.Internal;

namespace OptunaSharp.Importance.Booster;

/// <summary>
///     <b>Experimental / non-Optuna.</b> Mean Decrease Impurity (MDI) importance evaluator backed by
///     <b>ML.NET's FastTreeTweedie</b> (Tweedie-loss gradient boosting). It reuses the shared MDI
///     formula, but over a <i>gradient-boosted</i> Tweedie ensemble; the per-node impurities are
///     recomputed by routing the training set through the fitted trees (ML.NET exposes no per-node
///     statistics).
///
///     <para><b>Off-label — this is not what Optuna does.</b> Optuna's importance uses only
///     scikit-learn's <c>RandomForestRegressor</c> (bagging); there is no boosting, and certainly no
///     Tweedie loss, anywhere in it. Boosting trees fit the ensemble's <i>residual</i>, not the
///     objective, so MDI over them measures a different, non-standard quantity. There is <b>no
///     value-parity with Python Optuna</b> and only weak ranking agreement. Prefer Core's canonical,
///     value-exact <see cref="MeanDecreaseImpurityImportanceEvaluator" /> when parity matters.</para>
/// </summary>
public sealed class FastTreeTweedieMeanDecreaseImpurityImportanceEvaluator : BaseImportanceEvaluator
{
    private readonly FastTreeTweedieTrainer.Options? _options;
    private readonly Action<FastTreeTweedieTrainer.Options>? _configure;
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int? _seed;

    /// <param name="nTrees">The number of boosting iterations (trees).</param>
    /// <param name="maxDepth">The maximum depth of each tree.</param>
    /// <param name="seed">Controls the randomness of the ensemble. Specify a value for determinism.</param>
    public FastTreeTweedieMeanDecreaseImpurityImportanceEvaluator(int nTrees = 64, int maxDepth = 64, int? seed = null)
    {
        _nTrees = nTrees;
        _maxDepth = maxDepth;
        _seed = seed;
    }

    /// <summary>
    ///     Full-control overload: fit with a caller-supplied
    ///     <see cref="FastTreeTweedieTrainer.Options" />, exposing the <b>entire</b> ML.NET
    ///     FastTreeTweedie API (including the Tweedie <c>Index</c>) rather than only
    ///     <c>(n_trees, max_depth, seed)</c>.
    /// </summary>
    /// <param name="options">
    ///     The FastTreeTweedie trainer options. <c>LabelColumnName</c> and <c>FeatureColumnName</c> are
    ///     managed by the evaluator (bound to the transformed-trial schema) and are overwritten; every
    ///     other field is used as given. For a reproducible result set <c>Seed</c> and
    ///     <c>NumberOfThreads = 1</c>.
    /// </param>
    public FastTreeTweedieMeanDecreaseImpurityImportanceEvaluator(FastTreeTweedieTrainer.Options options)
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
    public FastTreeTweedieMeanDecreaseImpurityImportanceEvaluator(
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
        => ForestImportance.EvaluateMdi(study, @params, target,
            (x, y, nFeatures) => _options is null
                ? BoosterBackend.FitFastTreeTweedie(x, y, nFeatures, _nTrees, _maxDepth, _seed, _configure)
                : BoosterBackend.FitFastTreeTweedie(x, y, nFeatures, _options));
}
