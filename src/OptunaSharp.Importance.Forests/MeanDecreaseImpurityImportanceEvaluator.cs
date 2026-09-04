using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Importance.Forests.Internal;
using OptunaSharp.Internal;

namespace OptunaSharp.Importance;

/// <summary>
///     Mean Decrease Impurity (MDI) parameter-importance evaluator. Port of
///     <c>optuna/importance/_mean_decrease_impurity.py: MeanDecreaseImpurityImportanceEvaluator</c>.
///
///     <para>Fits a random forest that predicts the objective values of
///     <see cref="TrialState.Complete" /> trials from their parameter configurations, then computes
///     feature importances via MDI (the analogue of
///     <c>sklearn.ensemble.RandomForestRegressor.feature_importances_</c>).</para>
///
///     <para><b>Divergence from Python Optuna.</b> The underlying forest is ML.NET's FastForest, not
///     scikit-learn's <c>RandomForestRegressor</c>, and the node impurities are recomputed by
///     routing the training set through the fitted trees (ML.NET does not expose per-node
///     bootstrap statistics), so importance values differ numerically from Optuna. The MDI formula
///     itself is a faithful port. Documented Tier X divergence — see the package README.</para>
/// </summary>
public sealed class MeanDecreaseImpurityImportanceEvaluator : BaseImportanceEvaluator
{
    private readonly RandomForestRegressor _forest;

    /// <param name="nTrees">Number of trees in the random forest.</param>
    /// <param name="maxDepth">The maximum depth of each tree in the random forest.</param>
    /// <param name="seed">Seed for the random forest.</param>
    public MeanDecreaseImpurityImportanceEvaluator(int nTrees = 64, int maxDepth = 64, int? seed = null)
    {
        // min_samples_split=2 / min_samples_leaf=1 are Optuna's (scikit-learn's) defaults.
        _forest = new RandomForestRegressor(
            nTrees: nTrees, maxDepth: maxDepth, minSamplesLeaf: 1, seed: seed);
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
        if (paramList.Count == 0)
            return new OrderedDict<string, double>();

        var trials = ImportanceBase.GetFilteredTrials(study, paramList, target);
        if (trials.Count <= 1)
        {
            var zeros = new OrderedDict<string, double>(paramList.Count);
            foreach (var k in paramList)
                zeros[k] = 0.0;
            return zeros;
        }

        var trans = new SearchSpaceTransform(
            distributions, transformLog: false, transformStep: false);

        double[][] transParams = ImportanceBase.GetTransParams(trials, trans);
        double[] targetValues = ImportanceBase.GetTargetValues(trials, target);

        int nEncodedColumns = trans.EncodedColumnToColumn.Length;
        _forest.Fit(transParams, targetValues, nEncodedColumns);
        double[] featureImportances = _forest.FeatureImportances();

        // Untransform feature importances to param importances by adding up the relevant encoded
        // (one-hot) columns — Python np.add.at(param_importances, trans.encoded_column_to_column, .).
        var paramImportances = new double[paramList.Count];
        long[] encodedColumnToColumn = trans.EncodedColumnToColumn;
        for (int encoded = 0; encoded < featureImportances.Length; encoded++)
            paramImportances[encodedColumnToColumn[encoded]] += featureImportances[encoded];

        return ImportanceBase.SortDictByImportance(
            ImportanceBase.ParamImportancesToDict(paramList, paramImportances));
    }
}
