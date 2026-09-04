using System;
using System.Collections.Generic;
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

    /// <inheritdoc />
    public override OrderedDict<string, double> Evaluate(
        Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null)
        => ForestImportance.EvaluateFanova(study, @params, target,
            (x, y, nFeatures) => BoosterBackend.FitFastTreeTweedie(x, y, nFeatures, _nTrees, _maxDepth, _seed));
}
