using System;
using System.Collections.Generic;
using Microsoft.ML.Trainers.FastTree;
using OptunaSharp.Collections;
using OptunaSharp.Importance.Forests.Internal;
using OptunaSharp.Importance.Forests.MLNet;

namespace OptunaSharp.Importance.Forests;

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
    private readonly FastForestRegressionTrainer.Options? _options;
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

    /// <summary>
    ///     Full-control overload: fit the forest with a caller-supplied
    ///     <see cref="FastForestRegressionTrainer.Options" />, exposing the <b>entire</b> ML.NET
    ///     FastForest API rather than only <c>(n_trees, max_depth, seed)</c>.
    /// </summary>
    /// <param name="options">
    ///     The FastForest trainer options. <c>LabelColumnName</c> and <c>FeatureColumnName</c> are
    ///     managed by the evaluator (bound to the transformed-trial schema) and are overwritten; every
    ///     other field is used as given. For a reproducible result set <c>Seed</c> and
    ///     <c>NumberOfThreads = 1</c>.
    /// </param>
    public FastForestMeanDecreaseImpurityImportanceEvaluator(FastForestRegressionTrainer.Options options)
        => _options = options ?? throw new ArgumentNullException(nameof(options));

    /// <inheritdoc />
    public override OrderedDict<string, double> Evaluate(
        Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null)
        => ForestImportance.EvaluateMdi(study, @params, target,
            (x, y, nFeatures) => _options is null
                ? FastForestBackend.Fit(x, y, nFeatures, _nTrees, _maxDepth, _seed)
                : FastForestBackend.Fit(x, y, nFeatures, _options));
}
