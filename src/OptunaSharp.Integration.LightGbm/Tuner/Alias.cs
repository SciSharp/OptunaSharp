using System.Collections;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Integration.LightGbm;

/// <summary>
///     Port of <c>optuna_integration.lightgbm._lightgbm_tuner.alias</c> — canonicalizes LightGBM
///     parameter and metric <b>aliases</b> so the tuner reads the value under one known key. LightGBM
///     accepts many spellings for the same option (e.g. <c>sub_feature</c> / <c>colsample_bytree</c> for
///     <c>feature_fraction</c>); the tuner tunes the canonical names, so the incoming <c>params</c> dict
///     is rewritten in place before tuning.
/// </summary>
internal static class Alias
{
    // Port of `_ALIAS_GROUP_LIST` (order-significant — the last matching alias wins per group).
    private static readonly (string ParamName, string[] AliasNames)[] AliasGroupList =
    {
        ("bagging_fraction", new[] { "sub_row", "subsample", "bagging" }),
        ("learning_rate", new[] { "shrinkage_rate", "eta" }),
        ("min_child_samples", new[] { "min_data_per_leaf", "min_data", "min_data_in_leaf", "min_samples_leaf" }),
        ("min_sum_hessian_in_leaf",
            new[] { "min_sum_hessian_per_leaf", "min_sum_hessian", "min_hessian", "min_child_weight" }),
        ("num_leaves", new[] { "num_leaf", "max_leaves", "max_leaf", "max_leaf_nodes" }),
        ("bagging_freq", new[] { "subsample_freq" }),
        ("feature_fraction", new[] { "sub_feature", "colsample_bytree" }),
        ("lambda_l1", new[] { "reg_alpha", "l1_regularization" }),
        ("lambda_l2", new[] { "reg_lambda", "lambda", "l2_regularization" }),
        ("min_gain_to_split", new[] { "min_split_gain" }),
    };

    // Port of `_ALIAS_METRIC_LIST` (metric_name -> alias_names, excluding the canonical name itself).
    private static readonly (string MetricName, string[] AliasNames)[] AliasMetricList =
    {
        ("ndcg", new[] { "lambdarank", "rank_xendcg", "xendcg", "xe_ndcg", "xe_ndcg_mart", "xendcg_mart" }),
        ("map", new[] { "mean_average_precision" }),
        ("l2", new[] { "regression", "regression_l2", "mean_squared_error", "mse" }),
        ("l1", new[] { "regression_l1", "mean_absolute_error", "mae" }),
        ("binary_logloss", new[] { "binary" }),
        ("multi_logloss", new[] { "multiclass", "softmax", "multiclassova", "multiclass_ova", "ova", "ovr" }),
        ("cross_entropy", new[] { "xentropy" }),
        ("cross_entropy_lambda", new[] { "xentlambda" }),
        ("kullback_leibler", new[] { "kldiv" }),
        ("mape", new[] { "mean_absolute_percentage_error" }),
        ("custom", new[] { "none", "null", "na" }),
        ("rmse", new[] { "l2_root", "root_mean_squared_error" }),
    };

    // Port of `_ALIAS_METRIC_MAP` (alias_name -> canonical metric_name).
    private static readonly Dictionary<string, string> AliasMetricMap =
        AliasMetricList
            .SelectMany(g => g.AliasNames.Select(a => (Alias: a, Canonical: g.MetricName)))
            .ToDictionary(x => x.Alias, x => x.Canonical);

    /// <summary>Port of <c>_handling_alias_parameters</c> — rewrite aliases to canonical names in place.</summary>
    /// <param name="lgbmParams">The mutable ordered LightGBM parameter dictionary, shared with the current tuning step.</param>
    internal static void HandleAliasParameters(OrderedDict<string, object?> lgbmParams)
    {
        foreach (var (paramName, aliasNames) in AliasGroupList)
        {
            var duplicatedAlias = new OrderedDict<string, object?>();
            foreach (var aliasName in aliasNames)
            {
                if (lgbmParams.TryGetValue(aliasName, out var value))
                {
                    duplicatedAlias[aliasName] = value;
                    lgbmParams[paramName] = value;
                    lgbmParams.Remove(aliasName);
                }
            }

            if (duplicatedAlias.Count > 1)
                OptunaWarnings.Warn(
                    $"{paramName} in param detected multiple identical aliases " +
                    $"{FormatDict(duplicatedAlias)}, but we use {paramName}={lgbmParams[paramName]}.",
                    WarningCategory.UserWarning);
        }
    }

    /// <summary>Port of <c>_handling_alias_metrics</c> — canonicalize the <c>metric</c> value in place.</summary>
    /// <param name="lgbmParams">The mutable ordered LightGBM parameter dictionary, shared with the current tuning step.</param>
    /// <exception cref="NumSharp.ValueError">The metric value is neither a string nor a supported iterable.</exception>
    internal static void HandleAliasMetrics(OrderedDict<string, object?> lgbmParams)
    {
        if (!lgbmParams.TryGetValue("metric", out var metric))
            return;

        if (metric is string s)
        {
            lgbmParams["metric"] = AliasMetricMap.GetValueOrDefault(s) ?? s;
            return;
        }

        // A non-string iterable of metrics (list/tuple). Anything else is an error, exactly like Python's
        // `not isinstance(metric, (str, Iterable))` guard.
        if (metric is IEnumerable enumerable and not string)
        {
            var canonical = new List<object?>();
            foreach (var m in enumerable)
            {
                string name = m as string ?? m?.ToString() ?? "";
                canonical.Add(AliasMetricMap.GetValueOrDefault(name) ?? name);
            }
            lgbmParams["metric"] = canonical;
            return;
        }

        throw new ValueError(
            "The `metric` parameter is expected to be a string or an iterable object, but got " +
            $"{metric?.GetType().Name ?? "None"}.");
    }

    /// <summary>Formats ordered LightGBM parameters for validation diagnostics.</summary>
    /// <param name="d">The ordered parameter mapping formatted for diagnostics.</param>
    /// <returns>The Python-style representation of the parameter mapping.</returns>
    private static string FormatDict(OrderedDict<string, object?> d)
        => "{" + string.Join(", ", d.Select(kv => $"'{kv.Key}': {kv.Value}")) + "}";
}
