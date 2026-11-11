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

    /// <summary>Configures the target direction and quantile cutoff used to retain better-performing trials.</summary>
    /// <param name="quantile">The fraction of best observations defining the selected region.</param>
    /// <param name="isLowerBetter">Whether smaller target values identify the preferred trials.</param>
    /// <param name="target">A selector for a scalar trial value; null uses the single objective value.</param>
    internal QuantileFilter(double quantile, bool isLowerBetter, Func<FrozenTrial, double>? target)
    {
        System.Diagnostics.Debug.Assert(0 < quantile && quantile <= 1, "quantile must be in (0, 1].");
        _quantile = quantile;
        _isLowerBetter = isLowerBetter;
        _target = target;
    }

    /// <summary>Returns the trials meeting the configured best-target quantile cutoff, including cutoff ties.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <returns>The trials meeting the configured best-target quantile cutoff, including cutoff ties.</returns>
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
    /// <summary>Configures the baseline region, reference-region convention, and smoothing used by PED-ANOVA.</summary>
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

    /// <summary>Returns the trials retained by the evaluator's best-region quantile filter.</summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="quantile">The fraction of best observations defining the selected region.</param>
    /// <param name="target">A scalar target selector; null allows the multi-objective TPE/Pareto selection rule.</param>
    /// <returns>The trials retained by the evaluator's best-region quantile filter.</returns>
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

    /// <summary>Returns the Pearson divergence between the better-performing and reference parameter densities.</summary>
    /// <param name="paramName">The parameter name identifying the distribution or values to process.</param>
    /// <param name="dist">The parameter distribution defining the values and support of the grid.</param>
    /// <param name="targetTrials">The trials defining the better-performing region whose density is compared.</param>
    /// <param name="regionTrials">The trials defining the reference region for density estimation.</param>
    /// <returns>The Pearson divergence between the better-performing and reference parameter densities.</returns>
    /// <remarks>
    ///     The final product and the Scott's-rule moments of <see cref="ScottParzenEstimatorFunctions" /> take NumPy's
    ///     BLAS order through <see cref="LinAlg.Dot(double[], double[])" />; they used to be left-to-right sums, which
    ///     agreed with Optuna end to end on the studies checked but not operation by operation. The densities themselves
    ///     run through the truncated normal's erf, so the divergence stays within the documented erf tolerance of Optuna's.
    /// </remarks>
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

        // float(pdf_local @ ((pdf_top / pdf_local - 1) ** 2)): the elementwise part is exact as NumPy's (an array ** 2
        // is a multiplication), and the 1-D @ is BLAS ddot, so it goes through NumSharp's np.dot to accumulate in
        // Optuna's order; a left-to-right sum rounds differently once the grid has more than a few points.
        var squaredRatio = new double[gridSize];
        for (int g = 0; g < gridSize; g++)
        {
            double r = pdfTop[g] / pdfLocal[g] - 1.0;
            squaredRatio[g] = r * r;
        }
        return LinAlg.Dot(pdfLocal, squaredRatio);
    }

    /// <inheritdoc />
    /// <param name="study">The study supplying completed trials and optimization directions.</param>
    /// <param name="params">The parameter names to assess, or null for every parameter appearing in any completed trial.</param>
    /// <param name="target">A scalar target selector, or null to use the study objectives, including multi-objective Pareto/TPE selection.</param>
    /// <returns>Descending parameter-importance values before optional normalization by GetParamImportances.</returns>
    /// <exception cref="NumSharp.ValueError">An explicitly requested parameter does not occur in any completed trial.</exception>
    /// <remarks>Unlike the base evaluator default, a null parameter list includes parameters that appear in any completed trial, including conditional parameters. A null target also supports multi-objective studies through Pareto/TPE trial selection.</remarks>
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

    /// <summary>Returns a parameter-name-ordered mapping assigning zero importance to each selected parameter.</summary>
    /// <param name="params">The selected parameter names to initialize with zero importance.</param>
    /// <returns>A parameter-name-ordered mapping assigning zero importance to each selected parameter.</returns>
    private static OrderedDict<string, double> ZeroImportances(IReadOnlyList<string> @params)
    {
        var result = new OrderedDict<string, double>(@params.Count);
        foreach (var k in @params)
            result[k] = 0.0;
        return result;
    }

    /// <summary>Python <c>_partition_by_regime(param_name, trials, min_n_trials_in_regime)</c>.</summary>
    /// <param name="paramName">The parameter name identifying the distribution or values to process.</param>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="minNTrialsInRegime">The minimum number of trials required before estimating a conditional regime.</param>
    /// <returns>Trial groups sharing the parameter's distribution, including a group for absent parameters.</returns>
    private static List<(BaseDistribution? Dist, List<FrozenTrial> Trials)> PartitionByRegime(
        string paramName, IReadOnlyList<FrozenTrial> trials, int minNTrialsInRegime)
    {
        // defaultdict(list) keyed by trial.distributions.get(param_name) (None for the inactive
        // regime), preserving first-appearance insertion order (Python dict semantics).
        var regimeTrials = new List<(BaseDistribution? Dist, List<FrozenTrial> Trials)>();
        foreach (var trial in trials)
        {
            BaseDistribution? key = trial.Distributions.TryGetValue(paramName, out var d) ? d : null;
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
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="target">A scalar target selector; null requires finite values for all objectives and allows multi-objective trial selection.</param>
    /// <returns>Completed trials containing the required parameters and a finite selected target value.</returns>
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
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="params">Parameter names to include; null selects the parameters supported by this operation.</param>
    /// <returns>The validated parameter names to evaluate, in deterministic order.</returns>
    /// <exception cref="NumSharp.ValueError">An explicitly requested parameter never appears in a completed trial.</exception>
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
            foreach (var name in trial.Distributions.Keys)
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
