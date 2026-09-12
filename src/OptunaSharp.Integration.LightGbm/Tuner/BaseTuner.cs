using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;

namespace OptunaSharp.Integration.LightGbm;

/// <summary>
///     Port of <c>_BaseTuner</c> (<c>optuna_integration.lightgbm._lightgbm_tuner.optimize</c>) — the
///     metric / direction logic shared by the tuner and its per-trial objective. It reads the LightGBM
///     <c>metric</c> parameter to decide the study direction (<see cref="HigherIsBetter" />), pick the
///     single metric Optuna optimizes (<see cref="GetMetricForObjective" />), and extract a booster's
///     validation score (<see cref="GetBoosterBestScore" />).
///
///     <para>Public only because it is the base of the public <see cref="LightGbmBaseTuner" /> (C#
///     forbids a public class deriving from an internal one); it is an implementation detail — the
///     supported surface is <see cref="LightGBMTuner" /> / <see cref="LightGBMTunerCV" />.</para>
/// </summary>
public abstract class BaseTuner
{
    // Metrics for which a larger value is better (Python `higher_is_better`).
    private static readonly HashSet<string> HigherIsBetterMetrics =
        new() { "auc", "auc_mu", "ndcg", "map", "average_precision" };

    /// <summary>The accumulated LightGBM parameters (native names), mutated as tuning progresses.</summary>
    protected internal OrderedDict<string, object?> LgbmParams;

    /// <summary>The validation set(s) — set on the train path; <c>null</c> for CV.</summary>
    protected IReadOnlyList<Dataset>? ValidSets;

    /// <summary>The validation name(s), when explicitly supplied by the caller; else <c>null</c>.</summary>
    protected IReadOnlyList<string>? ValidNames;

    /// <summary>Retains the shared LightGBM parameters used to resolve metrics and validation names.</summary>
    /// <param name="lgbmParams">The mutable ordered LightGBM parameter dictionary, shared with the current tuning step.</param>
    protected BaseTuner(OrderedDict<string, object?> lgbmParams)
        => LgbmParams = lgbmParams;

    /// <summary>Port of <c>_get_metric_for_objective</c> — the one metric Optuna optimizes.</summary>
    /// <returns>The final configured optimization metric, including a ranking cutoff where applicable.</returns>
    /// <exception cref="System.NotImplementedException">The metric configuration is empty or has an unsupported type.</exception>
    protected string GetMetricForObjective()
    {
        object? metric = LgbmParams.GetValueOrDefault("metric", "binary_logloss");

        string metricName;
        if (metric is string s)
            metricName = s;
        else if (metric is IEnumerable en and not string)
        {
            // Sequence / iterable -> the last element (Python `metric[-1]` / `list(metric)[-1]`).
            object? last = null;
            foreach (var item in en)
                last = item;
            metricName = last as string ?? last?.ToString()
                ?? throw new NotImplementedException("Empty `metric` iterable.");
        }
        else
            throw new NotImplementedException();

        return MetricWithEvalAt(metricName);
    }

    /// <summary>Port of <c>_metric_with_eval_at</c> — append <c>@k</c> for the ranking metrics.</summary>
    /// <param name="metric">The LightGBM metric name used for optimization or reporting.</param>
    /// <returns>The ranking metric name with the resolved evaluation cutoff appended.</returns>
    /// <exception cref="NumSharp.ValueError">The ranking cutoff is neither an integer nor an integer sequence.</exception>
    private string MetricWithEvalAt(string metric)
    {
        if (metric != "ndcg" && metric != "map")
            return metric;

        object? evalAt =
            LgbmParams.GetValueOrDefault("eval_at")
            ?? LgbmParams.GetValueOrDefault($"{metric}_at")
            ?? LgbmParams.GetValueOrDefault($"{metric}_eval_at")
            ?? new List<object?> { 1L, 2L, 3L, 4L, 5L };

        if (evalAt is IEnumerable seq and not string)
        {
            foreach (var item in seq)
                return $"{metric}@{Convert.ToInt64(item, CultureInfo.InvariantCulture)}";
            // fall through on an empty sequence -> error below
        }
        else if (evalAt is int or long)
            return $"{metric}@{evalAt}";

        throw new ValueError(
            "The value of eval_at is expected to be int or a list/tuple of int. " +
            $"'{evalAt}' is specified.");
    }

    /// <summary>Port of <c>higher_is_better</c> — is a larger metric value better?</summary>
    /// <returns>Whether the selected metric should be maximized.</returns>
    public bool HigherIsBetter()
    {
        object? metric = LgbmParams.GetValueOrDefault("metric", "binary_logloss");
        return metric is string s && HigherIsBetterMetrics.Contains(s);
    }

    /// <summary>Port of <c>compare_validation_metrics</c>.</summary>
    /// <param name="valScore">The newly observed validation score.</param>
    /// <param name="bestScore">The current best score, or the validation-set metric mapping when constructing a Booster.</param>
    /// <returns>Whether the new validation score improves on the incumbent in the metric's preferred direction.</returns>
    public bool CompareValidationMetrics(double valScore, double bestScore)
        => HigherIsBetter() ? valScore > bestScore : valScore < bestScore;

    /// <summary>Port of <c>_get_booster_best_score</c> — <c>booster.best_score[valid_name][metric]</c>.</summary>
    /// <param name="booster">The trained booster or collection of fold models to inspect or persist.</param>
    /// <returns>The selected validation-set metric from the booster.</returns>
    /// <exception cref="System.NotImplementedException">No validation set or metric can be selected from the tuner configuration.</exception>
    protected double GetBoosterBestScore(Booster booster)
    {
        string metric = GetMetricForObjective();

        string validName;
        if (ValidNames is not null)
            validName = ValidNames[^1];
        else if (ValidSets is { Count: 1 })
            validName = "valid_0";
        else if (ValidSets is { Count: > 0 })
            validName = $"valid_{ValidSets.Count - 1}";
        else
            throw new NotImplementedException();

        return booster.BestScore[validName][metric];
    }

    /// <summary>The name of every validation set, aligned with <see cref="ValidSets" />.</summary>
    /// <returns>Validation-set names in the same order as their datasets.</returns>
    protected IReadOnlyList<string> ResolveAllValidNames()
    {
        int n = ValidSets?.Count ?? 0;
        if (ValidNames is not null)
            return ValidNames;
        return Enumerable.Range(0, n).Select(i => $"valid_{i}").ToList();
    }
}
