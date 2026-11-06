using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Importance.Forests.Internal;
using OptunaSharp.Importance.Forests.Internal.Cart;
using OptunaSharp.Importance.Shap.Internal;
using OptunaSharp.Internal;

namespace OptunaSharp.Importance.Shap;

/// <summary>
///     Shapley (SHAP) parameter-importance evaluator. Port of
///     <c>optuna_integration/shap/shap.py: ShapleyImportanceEvaluator</c> (Optuna exposes it as
///     <c>optuna.integration.ShapleyImportanceEvaluator</c>; the <c>optuna.integration.shap</c> shim is
///     deprecated in favor of <c>optuna_integration.shap</c>).
///
///     <para>Fits a random forest that predicts the objective values of
///     <see cref="TrialState.Complete" /> trials from their parameter configurations, then computes each
///     feature's importance as the <b>mean absolute SHAP value</b> over the trials.</para>
///
///     <para><b>Pure-managed, and byte-exact.</b> Optuna ships this evaluator in the separate
///     <c>optuna_integration</c> package because there it depends on the third-party <c>shap</c> and
///     <c>scikit-learn</c> Python packages. OptunaSharp's port needs <b>no foreign runtime</b> — the
///     surrogate is Core's bit-exact <c>RandomForestRegressor</c> port (the same forest
///     <see cref="MeanDecreaseImpurityImportanceEvaluator" /> uses) and the tree-path-dependent Tree
///     SHAP walk (<see cref="Internal.TreeShap" />) is a faithful, pure-arithmetic port of
///     <c>shap/cext/tree_shap.h</c> — so it lives in Core alongside the other importance evaluators
///     (namespace <c>OptunaSharp.Importance.Shap</c>). With a fixed <c>seed</c> the importances match
///     running Optuna's <c>ShapleyImportanceEvaluator</c> (SHAP 0.52.0) through pythonnet to within a
///     few ULP.</para>
/// </summary>
public sealed class ShapleyImportanceEvaluator : BaseImportanceEvaluator
{
    private readonly int _nTrees;
    private readonly int _maxDepth;
    private readonly int? _seed;

    /// <param name="nTrees">Number of trees in the random forest.</param>
    /// <param name="maxDepth">The maximum depth of each tree in the random forest.</param>
    /// <param name="seed">Seed for the random forest.</param>
    /// <summary>Configures the regression forest whose Tree SHAP contributions determine parameter importance.</summary>
    public ShapleyImportanceEvaluator(int nTrees = 64, int maxDepth = 64, int? seed = null)
    {
        // Optuna decorates the class with @experimental_class("3.0.0"), which raises an
        // ExperimentalWarning on construction. OptunaSharp targets the Optuna 5.x surface and treats
        // this evaluator as stable, so that warning is intentionally omitted (deliberate divergence).

        _nTrees = nTrees;
        _maxDepth = maxDepth;
        _seed = seed;
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.ValueError">The requested parameters lack a consistent completed-trial search space, or a multi-objective study has no scalar target selector.</exception>
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
        var trans = new SearchSpaceTransform(distributions, transformLog: false, transformStep: false);
        double[][] transParams = ImportanceBase.GetTransParams(trials, trans);
        double[] targetValues = ImportanceBase.GetTargetValues(trials, target);

        // scikit-learn casts X to float32 (DTYPE) before fitting; the bit-exact CART port reproduces it.
        int nEncoded = trans.EncodedColumnToColumn.Length;
        float[,] x = BuildFloat32(transParams, trials.Count, nEncoded);

        SklearnLikeTree[] forest = CartRandomForest.Fit(x, targetValues, nEncoded, _nTrees, _maxDepth, _seed);

        // explainer = TreeExplainer(forest); feature_shap_values = explainer.shap_values(trans_params).
        // shap traverses the trees on X cast to the model's input_dtype, which is float32 for scikit-learn forests
        // (TreeEnsemble sets input_dtype = np.float32; _validate_inputs does X.astype before _cext.dense_tree_shap).
        // Traversing the float64 values instead sent a value whose float32 rounding sits exactly on a split
        // threshold down the other branch: stepped floats such as 0.35000000000000003 hit it (QA-24).
        double[][] featureShapValues = TreeShap.ShapValues(forest, WidenFloat32(x, trials.Count, nEncoded), nEncoded);

        // np.add.at(param_shap_values.T, trans.encoded_column_to_column, feature_shap_values.T):
        // add up the (one-hot) encoded columns belonging to each parameter, per trial, in column order.
        long[] encodedColumnToColumn = trans.EncodedColumnToColumn;
        var paramShapValues = new double[trials.Count][];
        for (int t = 0; t < trials.Count; t++)
            paramShapValues[t] = new double[paramList.Count];
        for (int encoded = 0; encoded < nEncoded; encoded++)
        {
            int param = (int)encodedColumnToColumn[encoded];
            for (int t = 0; t < trials.Count; t++)
                paramShapValues[t][param] += featureShapValues[t][encoded];
        }

        // np.abs(param_shap_values).mean(axis=0): numpy reduces a C-contiguous 2-D array along axis 0
        // by accumulating whole rows in order (sequential, not pairwise), then divides by n_trials.
        var meanAbsShapValues = new double[paramList.Count];
        for (int param = 0; param < paramList.Count; param++)
        {
            double sum = 0.0;
            for (int t = 0; t < trials.Count; t++)
                sum += Math.Abs(paramShapValues[t][param]);
            meanAbsShapValues[param] = sum / trials.Count;
        }

        return ImportanceBase.SortDictByImportance(
            ImportanceBase.ParamImportancesToDict(paramList, meanAbsShapValues));
    }

    /// <summary>
    ///     shap's traversal input for a scikit-learn forest: the float32 feature matrix widened back to double, so each
    ///     value is exactly the float32 shap compares against the float64 thresholds.
    /// </summary>
    /// <param name="x">The float32 feature matrix the forest was fit on (scikit-learn's DTYPE cast).</param>
    /// <param name="nRows">Number of trial rows in <paramref name="x" />.</param>
    /// <param name="nCols">Number of encoded feature columns in <paramref name="x" />.</param>
    /// <returns>Row-major double rows holding the float32 values; NaN stays NaN, so shap's missing-value mask is unchanged.</returns>
    private static double[][] WidenFloat32(float[,] x, int nRows, int nCols)
    {
        var rows = new double[nRows][];
        for (int i = 0; i < nRows; i++)
        {
            rows[i] = new double[nCols];
            for (int j = 0; j < nCols; j++)
                rows[i][j] = x[i, j];
        }
        return rows;
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
