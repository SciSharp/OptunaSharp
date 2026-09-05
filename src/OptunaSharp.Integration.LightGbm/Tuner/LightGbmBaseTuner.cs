using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Integration.LightGbm;

/// <summary>
///     Port of <c>_LightGBMBaseTuner</c> (<c>optuna_integration.lightgbm._lightgbm_tuner.optimize</c>) —
///     the shared engine of <see cref="LightGBMTuner" /> and <see cref="LightGBMTunerCV" />. It runs the
///     LightGBM Tuner's <b>stepwise</b> search: it tunes <c>feature_fraction</c>, then <c>num_leaves</c>,
///     then <c>bagging_fraction</c>/<c>bagging_freq</c>, then <c>feature_fraction</c> again (fine grid),
///     then <c>lambda_l1</c>/<c>lambda_l2</c>, then <c>min_child_samples</c> — each step a short Optuna
///     study over a fresh <c>GridSampler</c> or <c>TPESampler</c>, holding the previously-found best
///     parameters fixed. All steps share one Optuna <see cref="Study" />; each sampler is isolated to its
///     own trials by parameter/search-space (a shared study + per-step sampler reproduces Python's
///     <c>_StepwiseStudy</c> view exactly, since OptunaSharp's samplers read every trial from storage —
///     the analog of Python's <c>Study._get_trials</c> — and filter internally).
///
///     <para>The stepwise search is a faithful port and rides OptunaSharp's bit-exact samplers; only the
///     per-trial <b>training + metric evaluation</b> is an adapter, behind the <see cref="ILightGbmTrainer" />
///     seam (default: <see cref="MlNetLightGbmTrainer" />, ML.NET's native LightGBM).</para>
/// </summary>
public abstract class LightGbmBaseTuner : BaseTuner
{
    /// <summary>Port of <c>_DEFAULT_LIGHTGBM_PARAMETERS</c> (LightGBM's documented defaults).</summary>
    private static OrderedDict<string, object?> DefaultLightGbmParameters() => new()
    {
        ["lambda_l1"] = 0.0,
        ["lambda_l2"] = 0.0,
        ["num_leaves"] = 31L,
        ["feature_fraction"] = 1.0,
        ["bagging_fraction"] = 1.0,
        ["bagging_freq"] = 0L,
        ["min_child_samples"] = 20L,
    };

    private protected readonly Dataset TrainSet;
    private protected readonly int NumBoostRound;
    private protected readonly string? ModelDir;
    private protected readonly int? OptunaSeed;
    private protected readonly ILightGbmTrainer Trainer;
    private readonly int? _timeBudget;
    private readonly int? _sampleSize;
    private readonly bool _showProgressBar;
    private readonly IReadOnlyList<Action<Study, FrozenTrial>>? _optunaCallbacks;
    private readonly object? _customObjective;

    private Dataset? _trainSubset;
    private Stopwatch? _startTime;

    /// <summary>The Optuna study accumulating every step's trials.</summary>
    public Study Study { get; }

    /// <summary>The best booster seen this run + the trial number that produced it (in-memory).</summary>
    private protected (object Booster, int TrialNumber)? BestBoosterWithTrialNumber;

    private protected LightGbmBaseTuner(
        IReadOnlyDictionary<string, object?> @params, Dataset trainSet, int numBoostRound,
        int? timeBudget, int? sampleSize, Study? study,
        IEnumerable<Action<Study, FrozenTrial>>? optunaCallbacks, bool showProgressBar,
        string? modelDir, int? optunaSeed, ILightGbmTrainer? trainer)
        : base(DeepCopyParams(@params))
    {
        // Handling alias metrics (Python `_handling_alias_metrics(params)`).
        Alias.HandleAliasMetrics(LgbmParams);

        TrainSet = trainSet;
        NumBoostRound = numBoostRound;
        _timeBudget = timeBudget;
        _sampleSize = sampleSize;
        _showProgressBar = showProgressBar;
        _optunaCallbacks = optunaCallbacks?.ToList();
        ModelDir = modelDir;
        OptunaSeed = optunaSeed;
        Trainer = trainer ?? new MlNetLightGbmTrainer();
        _customObjective = GetCustomObjective(LgbmParams);

        // `model_dir` persists each trial's booster to disk so a resumed / distributed `get_best_booster`
        // can find it (Python pickles; the ML.NET port delegates to the trainer's ILightGbmBoosterStore).
        // The default MlNetLightGbmTrainer implements it; a custom trainer must too, else fail fast here.
        if (ModelDir is not null)
        {
            if (Trainer is not ILightGbmBoosterStore)
                throw new NotSupportedException(
                    $"`model_dir` requires the trainer to implement {nameof(ILightGbmBoosterStore)} so "
                    + $"boosters can be persisted for a resumed / distributed `GetBestBooster`. The default "
                    + $"{nameof(MlNetLightGbmTrainer)} does; the supplied {Trainer.GetType().Name} does not.");
            Directory.CreateDirectory(ModelDir);
        }

        // Should not alter data since `min_child_samples` is tuned (LightGBM `feature_pre_filter`).
        if (LgbmParams.TryGetValue("feature_pre_filter", out var fpf) && fpf is true)
            OptunaWarnings.Warn(
                "feature_pre_filter is given as True but will be set to False. This is required "
                + "for the tuner to tune min_child_samples.");
        LgbmParams["feature_pre_filter"] = false;

        if (study is null)
        {
            Study = Optuna.CreateStudy(
                direction: HigherIsBetter() ? StudyDirection.Maximize : StudyDirection.Minimize);
        }
        else
        {
            Study = study;
            string metricName = MetricName();
            if (HigherIsBetter())
            {
                if (Study.Direction != StudyDirection.Maximize)
                    throw new ValueError(
                        $"Study direction is inconsistent with the metric {metricName}. "
                        + "Please set 'maximize' as the direction.");
            }
            else if (Study.Direction != StudyDirection.Minimize)
            {
                throw new ValueError(
                    $"Study direction is inconsistent with the metric {metricName}. "
                    + "Please set 'minimize' as the direction.");
            }
        }
    }

    private string MetricName()
        => LgbmParams.GetValueOrDefault("metric", "binary_logloss")?.ToString() ?? "binary_logloss";

    /// <summary>Port of <c>best_score</c> — the value of the best booster (or ±inf before any trial).</summary>
    public double BestScore
    {
        get
        {
            try
            {
                return Study.BestValue;
            }
            catch (ValueError)
            {
                return HigherIsBetter() ? double.NegativeInfinity : double.PositiveInfinity;
            }
        }
    }

    /// <summary>Port of <c>best_params</c> — the LightGBM parameters of the best booster.</summary>
    public OrderedDict<string, object?> BestParams
    {
        get
        {
            OrderedDict<string, object?> @params;
            try
            {
                string json = (string)Study.BestTrial.SystemAttrs[OptunaObjective.LgbmParamsKey]!;
                @params = (OrderedDict<string, object?>)PyJson.Loads(json)!;
            }
            catch (ValueError)
            {
                // No trials have completed: defaults overlaid with any user-provided params.
                @params = DefaultLightGbmParameters();
                foreach (var kv in LgbmParams)
                    @params[kv.Key] = kv.Value;
            }

            if (_customObjective is not null)
                @params["objective"] = _customObjective;

            return @params;
        }
    }

    /// <summary>Port of <c>run</c> — perform the full stepwise tuning.</summary>
    public void Run()
    {
        // Handling aliases (Python `_handling_alias_parameters`).
        Alias.HandleAliasParameters(LgbmParams);

        SampleTrainSet();

        TuneFeatureFraction();
        TuneNumLeaves();
        TuneBagging();
        TuneFeatureFractionStage2();
        TuneRegularizationFactors();
        TuneMinDataInLeaf();
    }

    /// <summary>Port of <c>sample_train_set</c> — keep only the last <c>sample_size</c> rows for tuning.</summary>
    private void SampleTrainSet()
    {
        if (_sampleSize is null)
            return;

        int n = TrainSet.NumRows;
        if (n > _sampleSize.Value)
        {
            int offset = n - _sampleSize.Value;
            var data = new float[_sampleSize.Value][];
            var label = TrainSet.Label is null ? null : new double[_sampleSize.Value];
            var weight = TrainSet.Weight is null ? null : new double[_sampleSize.Value];
            for (int i = 0; i < _sampleSize.Value; i++)
            {
                data[i] = TrainSet.Data[offset + i];
                if (label is not null) label[i] = TrainSet.Label![offset + i];
                if (weight is not null) weight[i] = TrainSet.Weight![offset + i];
            }
            _trainSubset = new Dataset(data, label, weight, TrainSet.FeatureName);
        }
    }

    // ---- The six tuning steps (Python `tune_*`). ----

    /// <summary>Port of <c>tune_feature_fraction</c> — GridSampler over <c>linspace(0.4, 1.0, n)</c>.</summary>
    public void TuneFeatureFraction(int nTrials = 7)
    {
        const string paramName = "feature_fraction";
        var paramValues = MathUtil.Linspace(0.4, 1.0, nTrials);
        var sampler = new GridSampler(GridSpace(paramName, paramValues), OptunaSeed);
        TuneParams(new List<string> { paramName }, paramValues.Length, sampler, "feature_fraction");
    }

    /// <summary>Port of <c>tune_num_leaves</c> — TPESampler, 20 trials.</summary>
    public void TuneNumLeaves(int nTrials = 20)
        => TuneParams(new List<string> { "num_leaves" }, nTrials, new TPESampler(seed: OptunaSeed), "num_leaves");

    /// <summary>Port of <c>tune_bagging</c> — TPESampler over <c>bagging_fraction</c> + <c>bagging_freq</c>.</summary>
    public void TuneBagging(int nTrials = 10)
        => TuneParams(new List<string> { "bagging_fraction", "bagging_freq" }, nTrials,
            new TPESampler(seed: OptunaSeed), "bagging");

    /// <summary>Port of <c>tune_feature_fraction_stage2</c> — a fine GridSampler around the best value.</summary>
    public void TuneFeatureFractionStage2(int nTrials = 6)
    {
        const string paramName = "feature_fraction";
        double best = Convert.ToDouble(BestParams[paramName], CultureInfo.InvariantCulture);
        var paramValues = MathUtil.Linspace(best - 0.08, best + 0.08, nTrials)
            .Where(v => v >= 0.4 && v <= 1.0).ToArray();
        var sampler = new GridSampler(GridSpace(paramName, paramValues), OptunaSeed);
        TuneParams(new List<string> { paramName }, paramValues.Length, sampler, "feature_fraction_stage2");
    }

    /// <summary>Port of <c>tune_regularization_factors</c> — TPESampler over <c>lambda_l1</c> + <c>lambda_l2</c>.</summary>
    public void TuneRegularizationFactors(int nTrials = 20)
        => TuneParams(new List<string> { "lambda_l1", "lambda_l2" }, nTrials,
            new TPESampler(seed: OptunaSeed), "regularization_factors");

    /// <summary>Port of <c>tune_min_data_in_leaf</c> — GridSampler over <c>[5, 10, 25, 50, 100]</c>.</summary>
    public void TuneMinDataInLeaf()
    {
        const string paramName = "min_child_samples";
        long[] paramValues = { 5, 10, 25, 50, 100 };
        var space = new Dictionary<string, IReadOnlyList<CategoricalChoice>>
        {
            [paramName] = paramValues.Select(v => CategoricalChoice.From(v)).ToList(),
        };
        var sampler = new GridSampler(space, OptunaSeed);
        TuneParams(new List<string> { paramName }, paramValues.Length, sampler, "min_child_samples");
    }

    private static Dictionary<string, IReadOnlyList<CategoricalChoice>> GridSpace(string name, double[] values)
        => new() { [name] = values.Select(v => CategoricalChoice.From(v)).ToList() };

    /// <summary>Port of <c>_tune_params</c> — run one step's short study with the given sampler.</summary>
    private OptunaObjective TuneParams(
        List<string> targetParamNames, int nTrials, BaseSampler sampler, string stepName)
    {
        // Set current best parameters (Python `self.lgbm_params.update(self.best_params)`).
        foreach (var kv in BestParams)
            LgbmParams[kv.Key] = kv.Value;

        var trainSet = _trainSubset ?? TrainSet;
        var objective = CreateObjective(targetParamNames, trainSet, stepName);

        Study.Sampler = sampler;

        // Count this step's already-finished trials (the analog of `_StepwiseStudy.get_trials`).
        int completeTrials = Study
            .GetTrials(deepcopy: false, states: new[] { TrialState.Complete, TrialState.Pruned })
            .Count(t => t.SystemAttrs.TryGetValue(OptunaObjective.StepNameKey, out var s) && (s as string) == stepName);
        int trialsToRun = nTrials - completeTrials;

        _startTime ??= Stopwatch.StartNew();
        double? timeout = _timeBudget is not null
            ? _timeBudget.Value - _startTime.Elapsed.TotalSeconds
            : null;

        if (trialsToRun > 0)
            Study.Optimize(
                objective.Call,
                nTrials: trialsToRun,
                timeout: timeout,
                @catch: Array.Empty<Type>(),
                callbacks: _optunaCallbacks);

        if (objective.BestBoosterWithTrialNumber is not null)
            BestBoosterWithTrialNumber = objective.BestBoosterWithTrialNumber;

        return objective;
    }

    /// <summary>Create the step's objective (train vs CV) — Python's abstract <c>_create_objective</c>.</summary>
    private protected abstract OptunaObjective CreateObjective(
        List<string> targetParamNames, Dataset trainSet, string stepName);

    /// <summary>Port of <c>_get_custom_objective</c> — a non-string <c>objective</c> is a custom callable.</summary>
    private static object? GetCustomObjective(OrderedDict<string, object?> lgbmParams)
        => lgbmParams.TryGetValue("objective", out var o) && o is not null and not string ? o : null;

    private static OrderedDict<string, object?> DeepCopyParams(IReadOnlyDictionary<string, object?> @params)
    {
        var copy = new OrderedDict<string, object?>();
        foreach (var kv in @params)
            copy[kv.Key] = kv.Value is IEnumerable<object?> list ? list.ToList() : kv.Value;
        return copy;
    }
}
