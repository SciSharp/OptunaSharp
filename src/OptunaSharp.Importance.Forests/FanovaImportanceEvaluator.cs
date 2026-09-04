using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Importance.Forests.Internal;
using OptunaSharp.Internal;

namespace OptunaSharp.Importance;

/// <summary>
///     fANOVA importance evaluator. Port of
///     <c>optuna/importance/_fanova/_evaluator.py: FanovaImportanceEvaluator</c>.
///
///     <para>Implements the fANOVA hyperparameter-importance algorithm of
///     <see href="http://proceedings.mlr.press/v32/hutter14.html">"An Efficient Approach for
///     Assessing Hyperparameter Importance" (Hutter et al., ICML 2014)</see>. fANOVA fits a random
///     forest that predicts the objective values of <see cref="TrialState.Complete" /> trials from
///     their parameter configurations, then decomposes the forest's prediction variance across the
///     parameters. The more accurate the forest, the more reliable the importances — an
///     exploration-oriented sampler such as <c>RandomSampler</c> is recommended.</para>
///
///     <para><b>Divergence from Python Optuna.</b> The underlying random forest is ML.NET's
///     FastForest, which is not tree-identical to scikit-learn's <c>RandomForestRegressor</c>, so
///     importance values differ numerically from Optuna (the ranking of a clearly-dominant
///     parameter is robust; exact values are not). This is a documented Tier X divergence — see the
///     package README.</para>
/// </summary>
public sealed class FanovaImportanceEvaluator : BaseImportanceEvaluator
{
    private readonly Fanova _evaluator;

    /// <param name="nTrees">The number of trees in the forest.</param>
    /// <param name="maxDepth">The maximum depth of the trees in the forest.</param>
    /// <param name="seed">
    ///     Controls the randomness of the forest. For deterministic behavior, specify a value other
    ///     than <c>null</c>.
    /// </param>
    public FanovaImportanceEvaluator(int nTrees = 64, int maxDepth = 64, int? seed = null)
    {
        _evaluator = new Fanova(
            nTrees: nTrees, maxDepth: maxDepth, minSamplesSplit: 2, minSamplesLeaf: 1, seed: seed);
    }

    /// <inheritdoc />
    public override OrderedDict<string, double> Evaluate(
        Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null)
    {
        if (target is null && study.IsMultiObjective())
            throw new ValueError(
                "If the `study` is being used for multi-objective optimization, " +
                "please specify the `target`. For example, use " +
                "`target=lambda t: t.values[0]` for the first objective value.");

        var distributions = ImportanceBase.GetDistributions(study, @params);
        var paramList = @params?.ToList() ?? distributions.Keys.ToList();

        // fANOVA does not support parameter distributions with a single value. There is no reason to
        // compute importance in that case anyway — a constant parameter always has importance 0.
        var nonSingleDistributions = new OrderedDict<string, BaseDistribution>();
        var singleDistributions = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, dist) in distributions)
        {
            if (dist.Single())
                singleDistributions[name] = dist;
            else
                nonSingleDistributions[name] = dist;
        }

        if (nonSingleDistributions.Count == 0)
        {
            var zeros = new OrderedDict<string, double>(singleDistributions.Count);
            foreach (var k in singleDistributions.Keys)
                zeros[k] = 0.0;
            return zeros;
        }

        var trials = ImportanceBase.GetFilteredTrials(study, paramList, target);
        if (trials.Count <= 1)
        {
            var zeros = new OrderedDict<string, double>(paramList.Count);
            foreach (var k in paramList)
                zeros[k] = 0.0;
            return zeros;
        }

        var trans = new SearchSpaceTransform(
            nonSingleDistributions, transformLog: false, transformStep: false);

        double[][] transParams = ImportanceBase.GetTransParams(trials, trans);
        double[] targetValues = ImportanceBase.GetTargetValues(trials, target);

        _evaluator.Fit(
            x: transParams, y: targetValues,
            searchSpaces: trans.RawBounds,
            columnToEncodedColumns: trans.ColumnToEncodedColumns);

        var paramImportances = new double[nonSingleDistributions.Count];
        for (int i = 0; i < nonSingleDistributions.Count; i++)
            paramImportances[i] = _evaluator.GetImportance(i).Mean;

        var combined = new OrderedDict<string, double>();
        foreach (var (k, v) in ImportanceBase.ParamImportancesToDict(nonSingleDistributions.Keys.ToList(), paramImportances))
            combined[k] = v;
        foreach (var (k, v) in ImportanceBase.ParamImportancesToDict(singleDistributions.Keys.ToList(), 0.0))
            combined[k] = v;

        return ImportanceBase.SortDictByImportance(combined);
    }
}
