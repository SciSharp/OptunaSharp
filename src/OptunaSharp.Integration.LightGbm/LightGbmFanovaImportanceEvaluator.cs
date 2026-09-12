using System;
using System.Collections.Generic;
using Microsoft.ML.Trainers.LightGbm;
using OptunaSharp.Collections;
using OptunaSharp.Importance;
using OptunaSharp.Importance.Forests.Internal;

namespace OptunaSharp.Integration.LightGbm;

/// <summary>
///     <b>Experimental / non-Optuna.</b> fANOVA importance evaluator backed by <b>ML.NET's LightGBM</b>
///     (a histogram-based gradient-boosting machine). It reuses the shared fANOVA marginal-variance
///     walk (<c>optuna/importance/_fanova</c>), but fits a <i>gradient-boosted</i> LightGBM ensemble
///     instead of the random forest fANOVA was designed for.
///
///     <para><b>Off-label — this is not what Optuna does.</b> Optuna's importance uses only
///     scikit-learn's <c>RandomForestRegressor</c> (bagging); there is no boosting anywhere in it.
///     LightGBM's trees fit the ensemble's <i>residual</i>, not the objective, so running fANOVA over
///     them measures a different, non-standard quantity. There is <b>no value-parity with Python
///     Optuna</b> and only weak ranking agreement. Prefer Core's canonical, value-exact
///     <see cref="FanovaImportanceEvaluator" /> when parity matters. (LightGBM's actual Optuna role is
///     the <c>optuna.integration.lightgbm</c> tuner — see <see cref="LightGBMTuner" />.)</para>
/// </summary>
public sealed class LightGbmFanovaImportanceEvaluator : BaseImportanceEvaluator
{
    private readonly LightGbmRegressionTrainer.Options? _options;
    private readonly Action<LightGbmRegressionTrainer.Options>? _configure;
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int? _seed;

    /// <param name="nTrees">The number of boosting iterations (trees).</param>
    /// <param name="maxDepth">The maximum number of leaves per tree is derived from this depth.</param>
    /// <param name="seed">Controls the randomness of the ensemble. Specify a value for determinism.</param>
    /// <summary>Creates the LightGBM fANOVA evaluator with the requested tree count, depth, and random seed.</summary>
    public LightGbmFanovaImportanceEvaluator(int nTrees = 64, int maxDepth = 64, int? seed = null)
    {
        _nTrees = nTrees;
        _maxDepth = maxDepth;
        _seed = seed;
    }

    /// <summary>
    ///     Full-control overload: fit with a caller-supplied
    ///     <see cref="LightGbmRegressionTrainer.Options" />, exposing the <b>entire</b> ML.NET LightGBM
    ///     API — <c>NumberOfIterations</c>, <c>NumberOfLeaves</c>, <c>LearningRate</c>,
    ///     <c>MaximumBinCountPerFeature</c>, the nested <c>Booster</c> (GBDT/DART/GOSS), <c>Seed</c>,
    ///     <c>NumberOfThreads</c>, … — rather than only <c>(n_trees, max_depth, seed)</c>.
    /// </summary>
    /// <param name="options">
    ///     The LightGBM trainer options. <c>LabelColumnName</c> and <c>FeatureColumnName</c> are managed
    ///     by the evaluator (bound to the transformed-trial schema) and are overwritten; every other
    ///     field is used as given. For a reproducible result set <c>Seed</c> and <c>NumberOfThreads = 1</c>.
    /// </param>
    /// <exception cref="System.ArgumentNullException"><paramref name="options" /> is null.</exception>
    public LightGbmFanovaImportanceEvaluator(LightGbmRegressionTrainer.Options options)
        => _options = options ?? throw new ArgumentNullException(nameof(options));

    /// <summary>
    ///     Convenience between the two ctors above: start from the reproducible default options for
    ///     <c>(nTrees, maxDepth, seed)</c> and <paramref name="configure" /> only the knobs you care
    ///     about (e.g. LearningRate, or the nested <c>Booster</c>) — without restating the defaults or
    ///     the two managed column names.
    /// </summary>
    /// <param name="configure">Applied to the default options just before fitting.</param>
    /// <param name="nTrees">The number of boosting iterations (the options baseline).</param>
    /// <param name="maxDepth">The maximum number of leaves per tree is derived from this depth.</param>
    /// <param name="seed">Baseline seed; may be overridden inside <paramref name="configure" />.</param>
    /// <exception cref="System.ArgumentNullException"><paramref name="configure" /> is null.</exception>
    public LightGbmFanovaImportanceEvaluator(
        Action<LightGbmRegressionTrainer.Options> configure, int nTrees = 64, int maxDepth = 64, int? seed = null)
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
                ? LightGbmBackend.Fit(x, y, nFeatures, _nTrees, _maxDepth, _seed, _configure)
                : LightGbmBackend.Fit(x, y, nFeatures, _options));
}
