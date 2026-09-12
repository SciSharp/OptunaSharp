using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Importance.Forests.Internal;

/// <summary>
///     Fits a random forest on the transformed trials and returns the trees as
///     <see cref="SklearnLikeTree" />s. Lets the evaluator body be shared between the native
///     scikit-learn CART backend (Core) and the ML.NET FastForest backend (the
///     <c>OptunaSharp.Importance.Forests</c> satellite).
/// </summary>
/// <param name="x">Feature values with one sample per row and one feature per column.</param>
/// <param name="y">Target values aligned with the feature rows.</param>
/// <param name="nFeatures">The number of feature columns in the training matrix.</param>
/// <returns>The fitted regression trees in ensemble order.</returns>
internal delegate SklearnLikeTree[] ForestFit(float[,] x, double[] y, int nFeatures);

/// <summary>
///     The backend-agnostic bodies of the fANOVA and MDI importance evaluators — the shared port of
///     <c>optuna/importance/_fanova/_evaluator.py</c> and <c>_mean_decrease_impurity.py</c> minus the
///     forest fit itself (supplied as a <see cref="ForestFit" />). scikit-learn casts X to float32,
///     which both backends reproduce.
/// </summary>
internal static class ForestImportance
{
    private const string MultiObjectiveMessage =
        "If the `study` is being used for multi-objective optimization, " +
        "please specify the `target`. For example, use " +
        "`target=lambda t: t.values[0]` for the first objective value.";

    /// <summary>Port of <c>FanovaImportanceEvaluator.evaluate</c>.</summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="params">Parameter names to include; null selects the parameters supported by this operation.</param>
    /// <param name="target">A selector for a scalar trial value; null uses the single objective value.</param>
    /// <param name="fit">The forest-fitting callback used to train the importance estimator.</param>
    /// <returns>Original-parameter importance values obtained by integrating the fitted forest.</returns>
    /// <exception cref="NumSharp.ValueError">A multi-objective study has no scalar target selector, or the requested parameter set has no consistent completed-trial search space.</exception>
    internal static OrderedDict<string, double> EvaluateFanova(
        Study study, IReadOnlyList<string>? @params, Func<FrozenTrial, double>? target, ForestFit fit)
    {
        if (target is null && study.IsMultiObjective())
            throw new ValueError(MultiObjectiveMessage);

        var distributions = ImportanceBase.GetDistributions(study, @params);
        var paramList = @params?.ToList() ?? distributions.Keys.ToList();

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

        var trans = new SearchSpaceTransform(nonSingleDistributions, transformLog: false, transformStep: false);
        double[][] transParams = ImportanceBase.GetTransParams(trials, trans);
        double[] targetValues = ImportanceBase.GetTargetValues(trials, target);

        int nEncoded = trans.RawBounds.GetLength(0);
        float[,] x = BuildFloat32(transParams, trials.Count, nEncoded);

        SklearnLikeTree[] forest = fit(x, targetValues, nEncoded);
        var evaluator = new Fanova(forest, trans.RawBounds, trans.ColumnToEncodedColumns);

        var paramImportances = new double[nonSingleDistributions.Count];
        for (int i = 0; i < nonSingleDistributions.Count; i++)
            paramImportances[i] = evaluator.GetImportance(i).Mean;

        var combined = new OrderedDict<string, double>();
        foreach (var (k, v) in ImportanceBase.ParamImportancesToDict(nonSingleDistributions.Keys.ToList(), paramImportances))
            combined[k] = v;
        foreach (var (k, v) in ImportanceBase.ParamImportancesToDict(singleDistributions.Keys.ToList(), 0.0))
            combined[k] = v;

        return ImportanceBase.SortDictByImportance(combined);
    }

    /// <summary>Port of <c>MeanDecreaseImpurityImportanceEvaluator.evaluate</c>.</summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="params">Parameter names to include; null selects the parameters supported by this operation.</param>
    /// <param name="target">A selector for a scalar trial value; null uses the single objective value.</param>
    /// <param name="fit">The forest-fitting callback used to train the importance estimator.</param>
    /// <returns>Original-parameter importance values obtained by aggregating split impurity decreases.</returns>
    /// <exception cref="NumSharp.ValueError">A multi-objective study has no scalar target selector, or the requested parameter set has no consistent completed-trial search space.</exception>
    internal static OrderedDict<string, double> EvaluateMdi(
        Study study, IReadOnlyList<string>? @params, Func<FrozenTrial, double>? target, ForestFit fit)
    {
        if (target is null && study.IsMultiObjective())
            throw new ValueError(MultiObjectiveMessage);

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

        var trans = new SearchSpaceTransform(distributions, transformLog: false, transformStep: false);
        double[][] transParams = ImportanceBase.GetTransParams(trials, trans);
        double[] targetValues = ImportanceBase.GetTargetValues(trials, target);

        int nEncoded = trans.EncodedColumnToColumn.Length;
        float[,] x = BuildFloat32(transParams, trials.Count, nEncoded);

        SklearnLikeTree[] forest = fit(x, targetValues, nEncoded);
        double[] featureImportances = Mdi.FeatureImportances(forest, nEncoded);

        // Untransform feature importances to param importances by adding up the encoded (one-hot)
        // columns — Python np.add.at(param_importances, trans.encoded_column_to_column, .).
        var paramImportances = new double[paramList.Count];
        long[] encodedColumnToColumn = trans.EncodedColumnToColumn;
        for (int encoded = 0; encoded < featureImportances.Length; encoded++)
            paramImportances[encodedColumnToColumn[encoded]] += featureImportances[encoded];

        return ImportanceBase.SortDictByImportance(
            ImportanceBase.ParamImportancesToDict(paramList, paramImportances));
    }

    /// <summary>scikit-learn casts X to float32 (DTYPE) before fitting the forest.</summary>
    /// <param name="transParams">The encoded numerical parameter rows in transform order.</param>
    /// <param name="nRows">The number of rows in the output feature matrix.</param>
    /// <param name="nCols">The number of columns in the output feature matrix.</param>
    /// <returns>The encoded feature matrix converted to scikit-learn's float32 input representation.</returns>
    private static float[,] BuildFloat32(double[][] transParams, int nRows, int nCols)
    {
        var x = new float[nRows, nCols];
        for (int i = 0; i < nRows; i++)
            for (int j = 0; j < nCols; j++)
                x[i, j] = (float)transParams[i][j];
        return x;
    }
}
