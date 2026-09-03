using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Importance.PedAnova;

/// <summary>
///     Port of <c>optuna/importance/_ped_anova/evaluator.py: _QuantileFilter</c>.
/// </summary>
internal sealed class QuantileFilter
{
    private readonly double _quantile;
    private readonly bool _isLowerBetter;
    private readonly Func<FrozenTrial, double>? _target;

    internal QuantileFilter(double quantile, bool isLowerBetter, Func<FrozenTrial, double>? target)
    {
        System.Diagnostics.Debug.Assert(0 < quantile && quantile <= 1, "quantile must be in (0, 1].");
        _quantile = quantile;
        _isLowerBetter = isLowerBetter;
        _target = target;
    }

    internal List<FrozenTrial> Filter(IReadOnlyList<FrozenTrial> trials)
    {
        double sign = _isLowerBetter ? 1.0 : -1.0;
        var lossValues = new double[trials.Count];
        for (int i = 0; i < trials.Count; i++)
            lossValues[i] = sign * (_target is null ? trials[i].Value!.Value : _target(trials[i]));

        // np.partition(loss_values, cutoff_index)[cutoff_index] == sorted(loss_values)[cutoff_index].
        int cutoffIndex = (int)Math.Ceiling(_quantile * lossValues.Length) - 1;
        var sorted = (double[])lossValues.Clone();
        Array.Sort(sorted);
        double cutoffVal = sorted[cutoffIndex];

        var result = new List<FrozenTrial>();
        for (int i = 0; i < trials.Count; i++)
        {
            if (lossValues[i] <= cutoffVal)
                result.Add(trials[i]);
        }
        return result;
    }
}

/// <summary>
///     PED-ANOVA importance evaluator. Port of
///     <c>optuna/importance/_ped_anova/evaluator.py: PedAnovaImportanceEvaluator</c>.
///
///     <para>PED-ANOVA fits Parzen estimators of <see cref="TrialState.Complete" /> trials better
///     than a user-specified <c>targetQuantile</c>. The importance can be interpreted as how
///     important each hyperparameter is to get the performance better than <c>targetQuantile</c>.</para>
///
///     <para>References: "PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary
///     Subspaces" (IJCAI 2023) and "Conditional PED-ANOVA" (KDD 2026).</para>
/// </summary>
public sealed class PedAnovaImportanceEvaluator : BaseImportanceEvaluator
{
    private readonly double _targetQuantile;
    private readonly double _regionQuantile;
    private readonly bool _evaluateOnLocal;

    // Advanced Setups.
    private readonly int _nSteps = 50;             // Discretize a domain as np.linspace(low, high, n_steps).
    private readonly double _priorWeight = 1.0;    // Control the regularization effect by prior.
    private readonly int _minNTrialsInRegime = 2;  // How many trials must be included in each regime.

    /// <param name="targetQuantile">
    ///     Compute the importance of achieving the top-<paramref name="targetQuantile" /> quantile
    ///     objective value (<c>gamma'</c> in the paper).
    /// </param>
    /// <param name="regionQuantile">
    ///     Define the region where importance is computed (<c>gamma</c> in the paper).
    /// </param>
    /// <param name="evaluateOnLocal">
    ///     Whether the importance is measured in the local (<c>true</c>) or global space.
    /// </param>
    public PedAnovaImportanceEvaluator(
        double targetQuantile = 0.1, double regionQuantile = 1.0, bool evaluateOnLocal = true)
    {
        System.Diagnostics.Debug.Assert(
            0.0 < targetQuantile && targetQuantile < regionQuantile && regionQuantile <= 1.0,
            "condition 0.0 < `target_quantile` < `region_quantile` <= 1.0 must be satisfied");
        if (regionQuantile != 1.0 && !evaluateOnLocal)
            OptunaWarnings.Warn("If `evaluate_on_local` is False, `region_quantile` has no effect.");

        _targetQuantile = targetQuantile;
        _regionQuantile = regionQuantile;
        _evaluateOnLocal = evaluateOnLocal;
    }

    private List<FrozenTrial> GetTopQuantileTrials(
        Study study, IReadOnlyList<FrozenTrial> trials, double quantile, Func<FrozenTrial, double>? target)
    {
        if (quantile == 1.0)
            return trials.ToList();
        if (study.IsMultiObjective() && target is null)
        {
            int nBelow = (int)Math.Ceiling(quantile * trials.Count);
            var (topTrials, _) = TpeTrialSplit.SplitCompleteTrialsMultiObjective(trials, study, nBelow);
            return topTrials;
        }

        bool isLowerBetter = study.Directions[0] == StudyDirection.Minimize;
        if (target is not null)
        {
            OptunaWarnings.Warn(
                $"{nameof(PedAnovaImportanceEvaluator)} computes the importances of params to achieve " +
                "low `target` values. If this is not what you want, " +
                "please modify target, e.g., by multiplying the output by -1.");
            isLowerBetter = true;
        }

        return new QuantileFilter(quantile, isLowerBetter, target).Filter(trials);
    }

    private double ComputePearsonDivergence(
        string paramName, BaseDistribution dist,
        IReadOnlyList<FrozenTrial> targetTrials, IReadOnlyList<FrozenTrial> regionTrials)
    {
        // When pdf_all == pdf_top, i.e. all_trials == top_trials, this method gives 0.0.
        double priorWeight = _priorWeight;
        var (peTop, gridSize) = ScottParzenEstimatorFunctions.BuildParzenEstimatorOnGrid(
            paramName, dist, targetTrials, _nSteps, priorWeight);
        var grids = new double[gridSize];
        for (int g = 0; g < gridSize; g++)
            grids[g] = g;
        var pdfTopRaw = peTop.Pdf(grids);
        var pdfTop = new double[gridSize];
        for (int g = 0; g < gridSize; g++)
            pdfTop[g] = pdfTopRaw[g] + 1e-12;

        var pdfLocal = new double[gridSize];
        if (_evaluateOnLocal)  // The importance of param during the study.
        {
            var (peLocal, _) = ScottParzenEstimatorFunctions.BuildParzenEstimatorOnGrid(
                paramName, dist, regionTrials, _nSteps, priorWeight);
            var pdfLocalRaw = peLocal.Pdf(grids);
            for (int g = 0; g < gridSize; g++)
                pdfLocal[g] = pdfLocalRaw[g] + 1e-12;
        }
        else  // The importance of param in the search space.
        {
            for (int g = 0; g < gridSize; g++)
                pdfLocal[g] = 1.0 / gridSize;
        }

        // float(pdf_local @ ((pdf_top / pdf_local - 1) ** 2)), sequential dot (bit-exact vs @).
        double sum = 0.0;
        for (int g = 0; g < gridSize; g++)
        {
            double r = pdfTop[g] / pdfLocal[g] - 1.0;
            sum += pdfLocal[g] * (r * r);
        }
        return sum;
    }

    /// <inheritdoc />
    public override OrderedDict<string, double> Evaluate(
        Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null)
    {
        var resolvedParams = ResolveParams(study, @params);

        var trials = GetFilteredTrials(study, target);
        if (trials.Count <= 1)
        {
            OptunaWarnings.Warn(
                "The number of trials is too small to compute importances. " +
                "Parameter importances will be equal.");
            return ZeroImportances(resolvedParams);
        }

        var targetTrials = GetTopQuantileTrials(study, trials, _targetQuantile, target);
        var regionTrials = GetTopQuantileTrials(study, trials, _regionQuantile, target);
        if (targetTrials.Count == regionTrials.Count)
            OptunaWarnings.Warn(
                "Target and region quantiles select the same set of trials. " +
                "Parameter importances will be equal.");
        if (targetTrials.Count == 0)
            return ZeroImportances(resolvedParams);

        var targetTrialIds = new HashSet<int>(targetTrials.Select(t => t.TrialId));
        var regionTrialIds = new HashSet<int>(regionTrials.Select(t => t.TrialId));
        // HSSP is greedy, so target trials are guaranteed to be included in region trials.
        System.Diagnostics.Debug.Assert(targetTrialIds.IsSubsetOf(regionTrialIds));

        // Theorem 4.2 and Algorithm 1 in the original paper: https://arxiv.org/abs/2601.20800.
        double quantile = (double)targetTrials.Count / regionTrials.Count;  // gamma' / gamma
        var paramImportances = new OrderedDict<string, double>(resolvedParams.Count);
        foreach (var k in resolvedParams)
            paramImportances[k] = 0.0;

        foreach (var paramName in resolvedParams)
        {
            var regimeTrials = PartitionByRegime(paramName, regionTrials, _minNTrialsInRegime);
            foreach (var (dist, regionTrialsRegime) in regimeTrials)
            {
                var targetTrialsRegime = regionTrialsRegime
                    .Where(t => targetTrialIds.Contains(t.TrialId))
                    .ToList();
                double targetProbRegime = (double)targetTrialsRegime.Count / targetTrials.Count;  // alpha_i
                double regionProbRegime = (double)regionTrialsRegime.Count / regionTrials.Count;   // beta_i
                if (dist is not null && !dist.Single() && targetTrialsRegime.Count > 0)
                {
                    paramImportances[paramName] +=
                        Math.Pow(targetProbRegime, 2.0)
                        / regionProbRegime
                        * ComputePearsonDivergence(paramName, dist, targetTrialsRegime, regionTrialsRegime);
                }
            }
        }

        double quantileSq = Math.Pow(quantile, 2.0);
        var scaled = new OrderedDict<string, double>(paramImportances.Count);
        foreach (var (k, v) in paramImportances)
            scaled[k] = v * quantileSq;
        return ImportanceBase.SortDictByImportance(scaled);
    }

    private static OrderedDict<string, double> ZeroImportances(IReadOnlyList<string> @params)
    {
        var result = new OrderedDict<string, double>(@params.Count);
        foreach (var k in @params)
            result[k] = 0.0;
        return result;
    }

    /// <summary>Python <c>_partition_by_regime(param_name, trials, min_n_trials_in_regime)</c>.</summary>
    private static List<(BaseDistribution? Dist, List<FrozenTrial> Trials)> PartitionByRegime(
        string paramName, IReadOnlyList<FrozenTrial> trials, int minNTrialsInRegime)
    {
        // defaultdict(list) keyed by trial.distributions.get(param_name) (None for the inactive
        // regime), preserving first-appearance insertion order (Python dict semantics).
        var regimeTrials = new List<(BaseDistribution? Dist, List<FrozenTrial> Trials)>();
        foreach (var trial in trials)
        {
            BaseDistribution? key = trial.DistributionsDict.TryGetValue(paramName, out var d) ? d : null;
            int found = -1;
            for (int i = 0; i < regimeTrials.Count; i++)
            {
                if (regimeTrials[i].Dist is null ? key is null : regimeTrials[i].Dist!.Equals(key))
                {
                    found = i;
                    break;
                }
            }
            if (found < 0)
                regimeTrials.Add((key, new List<FrozenTrial> { trial }));
            else
                regimeTrials[found].Trials.Add(trial);
        }

        if (regimeTrials.Any(r => r.Trials.Count < minNTrialsInRegime))
            OptunaWarnings.Warn(
                $"Some regimes for parameter `{paramName}` have less than " +
                $"{minNTrialsInRegime} trials. The importance of the parameter may be inaccurate.");

        return regimeTrials.Where(r => r.Trials.Count >= minNTrialsInRegime).ToList();
    }

    /// <summary>Python <c>_get_filtered_trials(study, target)</c> (the evaluator-local variant).</summary>
    private static List<FrozenTrial> GetFilteredTrials(Study study, Func<FrozenTrial, double>? target)
    {
        var trials = study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete });
        return trials
            .Where(trial => target is not null
                ? double.IsFinite(target(trial))
                : trial.Values!.All(double.IsFinite))
            .ToList();
    }

    /// <summary>Python <c>_resolve_params(study, params)</c>.</summary>
    private static List<string> ResolveParams(Study study, IReadOnlyList<string>? @params)
    {
        var trials = study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete });
        // Python: list(set(k for t in trials for k in t.distributions)). Its order is not
        // deterministic; we use first-appearance order, which is stable and only affects the
        // ordering of importance ties (the result is otherwise sorted by importance).
        var seen = new HashSet<string>();
        var allParams = new List<string>();
        foreach (var trial in trials)
        {
            foreach (var name in trial.DistributionsDict.Keys)
            {
                if (seen.Add(name))
                    allParams.Add(name);
            }
        }

        if (@params is not null)
        {
            var missing = @params.Where(p => !seen.Contains(p)).ToList();
            if (missing.Count > 0)
                throw new ValueError(
                    "Study must contain at least one completed trial for each specified parameter. " +
                    $"Missing parameters: {PythonRepr.ListRepr(missing)}.");
            return @params.ToList();
        }

        return allParams;
    }
}
