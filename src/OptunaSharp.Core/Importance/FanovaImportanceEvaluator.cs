using System;
using System.Collections.Generic;
using OptunaSharp.Collections;
using OptunaSharp.Importance.Forests.Internal;
using OptunaSharp.Importance.Forests.Internal.Cart;

namespace OptunaSharp.Importance;

/// <summary>
///     fANOVA importance evaluator. Port of
///     <c>optuna/importance/_fanova/_evaluator.py: FanovaImportanceEvaluator</c>.
///
///     <para>Implements the fANOVA hyperparameter-importance algorithm of
///     <see href="http://proceedings.mlr.press/v32/hutter14.html">"An Efficient Approach for
///     Assessing Hyperparameter Importance" (Hutter et al., ICML 2014)</see>. It fits a random forest
///     that predicts the objective values of <see cref="TrialState.Complete" /> trials from their
///     parameter configurations, then decomposes the forest's prediction variance across the
///     parameters. The more accurate the forest, the more reliable the importances — an
///     exploration-oriented sampler such as <c>RandomSampler</c> is recommended.</para>
///
///     <para>The random forest is a <b>pure-managed, bit-exact port of scikit-learn's
///     RandomForestRegressor</b> (the same forest Optuna uses), so with a fixed seed the importances
///     match Python Optuna to ~1 ULP. (An ML.NET FastForest backend is also available as
///     <c>OptunaSharp.Importance.Forests.FastForestFanovaImportanceEvaluator</c> in the
///     <c>OptunaSharp.Integration.ML.NET</c> package.)</para>
/// </summary>
public sealed class FanovaImportanceEvaluator : BaseImportanceEvaluator
{
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int? _seed;

    /// <param name="nTrees">The number of trees in the forest.</param>
    /// <param name="maxDepth">The maximum depth of the trees in the forest.</param>
    /// <param name="seed">
    ///     Controls the randomness of the forest. For deterministic behavior, specify a value other
    ///     than <c>null</c>.
    /// </param>
    /// <summary>Configures the regression forest used to estimate importance by functional ANOVA.</summary>
    public FanovaImportanceEvaluator(int nTrees = 64, int maxDepth = 64, int? seed = null)
    {
        _nTrees = nTrees;
        _maxDepth = maxDepth;
        _seed = seed;
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">The requested parameters lack a consistent completed-trial search space, or a multi-objective study has no scalar target selector.</exception>
    /// <exception cref="System.InvalidOperationException">Every fitted tree has zero integrated prediction variance.</exception>
    public override OrderedDict<string, double> Evaluate(
        Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null)
        => ForestImportance.EvaluateFanova(study, @params, target,
            (x, y, nFeatures) => CartRandomForest.Fit(x, y, nFeatures, _nTrees, _maxDepth, _seed));
}
