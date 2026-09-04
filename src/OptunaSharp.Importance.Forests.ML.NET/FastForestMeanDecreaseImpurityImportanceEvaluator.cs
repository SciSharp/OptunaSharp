using System;
using System.Collections.Generic;
using OptunaSharp.Collections;
using OptunaSharp.Importance.Forests.Internal;
using OptunaSharp.Importance.Forests.MLNet;

namespace OptunaSharp.Importance;

/// <summary>
///     Mean Decrease Impurity (MDI) importance evaluator backed by <b>ML.NET's FastForest</b> — the
///     alternative forest backend to Core's pure-managed, scikit-learn-bit-exact
///     <see cref="MeanDecreaseImpurityImportanceEvaluator" />. Same MDI formula (shared with Core),
///     different random forest; the per-node impurities are recomputed by routing the training set
///     through the fitted trees (ML.NET exposes no per-node statistics).
///
///     <para><b>Divergence from Python Optuna.</b> FastForest is not tree-identical to scikit-learn,
///     so importance <i>values</i> differ from Optuna — a documented Tier X divergence. Prefer
///     <see cref="MeanDecreaseImpurityImportanceEvaluator" /> when Optuna parity matters.</para>
/// </summary>
public sealed class FastForestMeanDecreaseImpurityImportanceEvaluator : BaseImportanceEvaluator
{
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int? _seed;

    /// <param name="nTrees">Number of trees in the random forest.</param>
    /// <param name="maxDepth">The maximum depth of each tree in the random forest.</param>
    /// <param name="seed">Seed for the random forest.</param>
    public FastForestMeanDecreaseImpurityImportanceEvaluator(int nTrees = 64, int maxDepth = 64, int? seed = null)
    {
        _nTrees = nTrees;
        _maxDepth = maxDepth;
        _seed = seed;
    }

    /// <inheritdoc />
    public override OrderedDict<string, double> Evaluate(
        Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null)
        => ForestImportance.EvaluateMdi(study, @params, target,
            (x, y, nFeatures) => FastForestBackend.Fit(x, y, nFeatures, _nTrees, _maxDepth, _seed));
}
