using System;
using System.Collections.Generic;
using OptunaSharp.Collections;
using OptunaSharp.Importance.Forests.Internal;
using OptunaSharp.Importance.Forests.Internal.Cart;

namespace OptunaSharp.Importance;

/// <summary>
///     Mean Decrease Impurity (MDI) parameter-importance evaluator. Port of
///     <c>optuna/importance/_mean_decrease_impurity.py: MeanDecreaseImpurityImportanceEvaluator</c>.
///
///     <para>Fits a random forest that predicts the objective values of
///     <see cref="TrialState.Complete" /> trials from their parameter configurations, then computes
///     feature importances via MDI (a port of
///     <c>sklearn.ensemble.RandomForestRegressor.feature_importances_</c>).</para>
///
///     <para>The random forest is a <b>pure-managed, bit-exact port of scikit-learn's
///     RandomForestRegressor</b>, and MDI reads each tree's stored per-node impurity and weighted
///     counts, so with a fixed seed the importances match Python Optuna to ~1 ULP. (An ML.NET
///     FastForest backend is also available as
///     <c>OptunaSharp.Importance.Forests.FastForestMeanDecreaseImpurityImportanceEvaluator</c> in the
///     <c>OptunaSharp.Integration.ML.NET</c> package.)</para>
/// </summary>
public sealed class MeanDecreaseImpurityImportanceEvaluator : BaseImportanceEvaluator
{
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int? _seed;

    /// <param name="nTrees">Number of trees in the random forest.</param>
    /// <param name="maxDepth">The maximum depth of each tree in the random forest.</param>
    /// <param name="seed">Seed for the random forest.</param>
    /// <summary>Configures the regression forest used to estimate importance from split impurity reductions.</summary>
    public MeanDecreaseImpurityImportanceEvaluator(int nTrees = 64, int maxDepth = 64, int? seed = null)
    {
        _nTrees = nTrees;
        _maxDepth = maxDepth;
        _seed = seed;
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">The requested parameters lack a consistent completed-trial search space, or a multi-objective study has no scalar target selector.</exception>
    public override OrderedDict<string, double> Evaluate(
        Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null)
        => ForestImportance.EvaluateMdi(study, @params, target,
            (x, y, nFeatures) => CartRandomForest.Fit(x, y, nFeatures, _nTrees, _maxDepth, _seed));
}
