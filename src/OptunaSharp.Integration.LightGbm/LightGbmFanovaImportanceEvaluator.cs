using System;
using System.Collections.Generic;
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
///     the <c>optuna.integration.lightgbm</c> tuner — planned for Phase 5.7.)</para>
/// </summary>
public sealed class LightGbmFanovaImportanceEvaluator : BaseImportanceEvaluator
{
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int? _seed;

    /// <param name="nTrees">The number of boosting iterations (trees).</param>
    /// <param name="maxDepth">The maximum number of leaves per tree is derived from this depth.</param>
    /// <param name="seed">Controls the randomness of the ensemble. Specify a value for determinism.</param>
    public LightGbmFanovaImportanceEvaluator(int nTrees = 64, int maxDepth = 64, int? seed = null)
    {
        _nTrees = nTrees;
        _maxDepth = maxDepth;
        _seed = seed;
    }

    /// <inheritdoc />
    public override OrderedDict<string, double> Evaluate(
        Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null)
        => ForestImportance.EvaluateFanova(study, @params, target,
            (x, y, nFeatures) => LightGbmBackend.Fit(x, y, nFeatures, _nTrees, _maxDepth, _seed));
}
