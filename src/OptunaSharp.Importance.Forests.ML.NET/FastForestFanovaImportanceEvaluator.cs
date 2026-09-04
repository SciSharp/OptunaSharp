using System;
using System.Collections.Generic;
using OptunaSharp.Collections;
using OptunaSharp.Importance.Forests.Internal;
using OptunaSharp.Importance.Forests.MLNet;

namespace OptunaSharp.Importance;

/// <summary>
///     fANOVA importance evaluator backed by <b>ML.NET's FastForest</b> — the alternative forest
///     backend to Core's pure-managed, scikit-learn-bit-exact
///     <see cref="FanovaImportanceEvaluator" />. Same fANOVA algorithm
///     (<c>optuna/importance/_fanova</c>, shared with Core), different random forest.
///
///     <para><b>Divergence from Python Optuna.</b> FastForest is not tree-identical to scikit-learn,
///     so importance <i>values</i> differ from Optuna (a dominant parameter's ranking is robust,
///     exact values are not) — a documented Tier X divergence. Prefer
///     <see cref="FanovaImportanceEvaluator" /> when Optuna parity matters; use this backend when you
///     want ML.NET's forest (e.g. a familiar/consistent model across an ML.NET pipeline).</para>
/// </summary>
public sealed class FastForestFanovaImportanceEvaluator : BaseImportanceEvaluator
{
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int? _seed;

    /// <param name="nTrees">The number of trees in the forest.</param>
    /// <param name="maxDepth">The maximum depth of the trees in the forest.</param>
    /// <param name="seed">Controls the randomness of the forest. Specify a value for determinism.</param>
    public FastForestFanovaImportanceEvaluator(int nTrees = 64, int maxDepth = 64, int? seed = null)
    {
        _nTrees = nTrees;
        _maxDepth = maxDepth;
        _seed = seed;
    }

    /// <inheritdoc />
    public override OrderedDict<string, double> Evaluate(
        Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null)
        => ForestImportance.EvaluateFanova(study, @params, target,
            (x, y, nFeatures) => FastForestBackend.Fit(x, y, nFeatures, _nTrees, _maxDepth, _seed));
}
