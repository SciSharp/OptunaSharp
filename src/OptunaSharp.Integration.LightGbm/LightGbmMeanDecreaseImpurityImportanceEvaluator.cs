using System;
using System.Collections.Generic;
using OptunaSharp.Collections;
using OptunaSharp.Importance;
using OptunaSharp.Importance.Forests.Internal;

namespace OptunaSharp.Integration.LightGbm;

/// <summary>
///     <b>Experimental / non-Optuna.</b> Mean Decrease Impurity (MDI) importance evaluator backed by
///     <b>ML.NET's LightGBM</b> (a histogram-based gradient-boosting machine). It reuses the shared MDI
///     formula, but over a <i>gradient-boosted</i> LightGBM ensemble; the per-node impurities are
///     recomputed by routing the training set through the fitted trees (ML.NET exposes no per-node
///     statistics).
///
///     <para><b>Off-label — this is not what Optuna does.</b> Optuna's importance uses only
///     scikit-learn's <c>RandomForestRegressor</c> (bagging); there is no boosting anywhere in it.
///     LightGBM's trees fit the ensemble's <i>residual</i>, not the objective, so MDI over them
///     measures a different, non-standard quantity. There is <b>no value-parity with Python Optuna</b>
///     and only weak ranking agreement. Prefer Core's canonical, value-exact
///     <see cref="MeanDecreaseImpurityImportanceEvaluator" /> when parity matters. (LightGBM's actual
///     Optuna role is the <c>optuna.integration.lightgbm</c> tuner — planned for Phase 5.7.)</para>
/// </summary>
public sealed class LightGbmMeanDecreaseImpurityImportanceEvaluator : BaseImportanceEvaluator
{
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int? _seed;

    /// <param name="nTrees">The number of boosting iterations (trees).</param>
    /// <param name="maxDepth">The maximum number of leaves per tree is derived from this depth.</param>
    /// <param name="seed">Controls the randomness of the ensemble. Specify a value for determinism.</param>
    public LightGbmMeanDecreaseImpurityImportanceEvaluator(int nTrees = 64, int maxDepth = 64, int? seed = null)
    {
        _nTrees = nTrees;
        _maxDepth = maxDepth;
        _seed = seed;
    }

    /// <inheritdoc />
    public override OrderedDict<string, double> Evaluate(
        Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null)
        => ForestImportance.EvaluateMdi(study, @params, target,
            (x, y, nFeatures) => LightGbmBackend.Fit(x, y, nFeatures, _nTrees, _maxDepth, _seed));
}
