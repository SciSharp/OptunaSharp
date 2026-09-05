using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using OptunaSharp.Collections;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Integration.LightGbm;

/// <summary>
///     Port of <c>_OptunaObjective</c> — the Optuna objective run for one tuning step. It suggests the
///     step's target parameters into the shared <c>lgbm_params</c> (<see cref="Preprocess" />), trains a
///     booster through the <see cref="ILightGbmTrainer" /> seam, returns the validation score Optuna
///     optimizes, and records the per-trial system attributes + the best booster seen.
/// </summary>
internal class OptunaObjective : BaseTuner
{
    // Trial system-attr keys (Python `_ELAPSED_SECS_KEY` etc.).
    internal const string ElapsedSecsKey = "lightgbm_tuner:elapsed_secs";
    internal const string AverageIterationTimeKey = "lightgbm_tuner:average_iteration_time";
    internal const string StepNameKey = "lightgbm_tuner:step_name";
    internal const string LgbmParamsKey = "lightgbm_tuner:lgbm_params";

    private const double Eps = 1e-12;                 // Python `_EPS`
    private const int DefaultTunerTreeDepth = 8;      // Python `_DEFAULT_TUNER_TREE_DEPTH`

    // Port of `_logger = optuna.logging.get_logger(__name__)`; used to announce a saved booster (model_dir).
    private protected static readonly Logger Log = OptunaLogging.GetLogger("optuna_integration.lightgbm");

    // The 7 tunable parameters (Python `_DEFAULT_LIGHTGBM_PARAMETERS` keys).
    internal static readonly string[] TunableParameters =
        { "lambda_l1", "lambda_l2", "num_leaves", "feature_fraction", "bagging_fraction", "bagging_freq", "min_child_samples" };

    protected readonly List<string> TargetParamNames;
    protected readonly Dataset TrainSet;
    protected readonly int NumBoostRound;
    protected readonly ILightGbmTrainer Trainer;
    protected readonly string StepName;
    protected readonly string? ModelDir;

    /// <summary>The best validation score this step has seen so far.</summary>
    internal double BestScoreValue;

    /// <summary>The best booster (or CV booster) + the trial number that produced it, or <c>null</c>.</summary>
    internal (object Booster, int TrialNumber)? BestBoosterWithTrialNumber;

    internal int TrialCount;

    internal OptunaObjective(
        List<string> targetParamNames, OrderedDict<string, object?> lgbmParams, Dataset trainSet,
        IReadOnlyList<Dataset>? validSets, IReadOnlyList<string>? validNames, int numBoostRound,
        ILightGbmTrainer trainer, double bestScore, string stepName, string? modelDir)
        : base(lgbmParams)
    {
        TargetParamNames = targetParamNames;
        TrainSet = trainSet;
        ValidSets = validSets;
        ValidNames = validNames;
        NumBoostRound = numBoostRound;
        Trainer = trainer;
        BestScoreValue = bestScore;
        StepName = stepName;
        ModelDir = modelDir;
        CheckTargetNamesSupported();
    }

    private void CheckTargetNamesSupported()
    {
        foreach (var name in TargetParamNames)
            if (Array.IndexOf(TunableParameters, name) < 0)
                throw new NotImplementedException($"Parameter `{name}` is not supported for tuning.");
    }

    /// <summary>Port of <c>_preprocess</c> — suggest the step's target params into <c>lgbm_params</c>.</summary>
    protected void Preprocess(Trial trial)
    {
        if (Has("lambda_l1"))
            LgbmParams["lambda_l1"] = trial.SuggestFloat("lambda_l1", 1e-8, 10.0, log: true);
        if (Has("lambda_l2"))
            LgbmParams["lambda_l2"] = trial.SuggestFloat("lambda_l2", 1e-8, 10.0, log: true);
        if (Has("num_leaves"))
        {
            long treeDepth = LgbmParams.TryGetValue("max_depth", out var md)
                ? Convert.ToInt64(md, CultureInfo.InvariantCulture)
                : DefaultTunerTreeDepth;
            long maxNumLeaves = treeDepth > 0 ? 1L << (int)treeDepth : 1L << DefaultTunerTreeDepth;
            LgbmParams["num_leaves"] = trial.SuggestInt("num_leaves", 2, maxNumLeaves);
        }
        if (Has("feature_fraction"))
            // GridSampler samples this; 1.0 is always in the grid. Clip the raw suggestion to <= 1.0.
            LgbmParams["feature_fraction"] = Math.Min(trial.SuggestFloat("feature_fraction", 0.4, 1.0 + Eps), 1.0);
        if (Has("bagging_fraction"))
            // TPESampler samples this; the raw suggestion may reach 1.0 + eps, so clip.
            LgbmParams["bagging_fraction"] = Math.Min(trial.SuggestFloat("bagging_fraction", 0.4, 1.0 + Eps), 1.0);
        if (Has("bagging_freq"))
            LgbmParams["bagging_freq"] = trial.SuggestInt("bagging_freq", 1, 7);
        if (Has("min_child_samples"))
            LgbmParams["min_child_samples"] = trial.SuggestInt("min_child_samples", 5, 100);

        bool Has(string name) => TargetParamNames.Contains(name);
    }

    /// <summary>Port of <c>__call__</c> — train, score, track best, and record system attrs.</summary>
    internal virtual double Call(Trial trial)
    {
        Preprocess(trial);

        var sw = Stopwatch.StartNew();
        var booster = Trainer.Train(
            new TrainRequest(LgbmParams, TrainSet, ValidSets!, ResolveAllValidNames(), NumBoostRound));

        double valScore = GetBoosterBestScore(booster);
        double elapsedSecs = sw.Elapsed.TotalSeconds;
        double averageIterationTime = elapsedSecs / Math.Max(1, booster.CurrentIteration);

        // Persist this trial's booster (Python `pickle.dump(booster, ...)`) so a resumed / distributed
        // `get_best_booster` can retrieve it. Enabled only when `model_dir` is set.
        if (ModelDir is not null && Trainer is ILightGbmBoosterStore store)
        {
            string path = BoosterPath(trial.Number);
            store.SaveBooster(booster, path);
            Log.Info($"The booster of trial#{trial.Number} was saved as {path}.");
        }

        if (CompareValidationMetrics(valScore, BestScoreValue))
        {
            BestScoreValue = valScore;
            BestBoosterWithTrialNumber = (booster, trial.Number);
        }

        Postprocess(trial, elapsedSecs, averageIterationTime);
        return valScore;
    }

    /// <summary>The extensionless base path for a trial's persisted booster (<c>{model_dir}/{trial}</c>).</summary>
    private protected string BoosterPath(int trialNumber)
        => Path.Combine(ModelDir!, trialNumber.ToString(CultureInfo.InvariantCulture));

    /// <summary>Port of <c>_postprocess</c> — record the elapsed / iteration-time / step / params attrs.</summary>
    protected void Postprocess(Trial trial, double elapsedSecs, double averageIterationTime)
    {
        trial.Storage.SetTrialSystemAttr(trial.TrialId, ElapsedSecsKey, elapsedSecs);
        trial.Storage.SetTrialSystemAttr(trial.TrialId, AverageIterationTimeKey, averageIterationTime);
        trial.Storage.SetTrialSystemAttr(trial.TrialId, StepNameKey, StepName);
        // json.dumps(deepcopy(lgbm_params)); byte-identical via PyJson (a custom callable objective — the
        // Python `_get_custom_objective` case — is not representable with the ML.NET trainer, so objective
        // stays a plain string here and needs no name substitution).
        var snapshot = new OrderedDict<string, object?>(LgbmParams);
        trial.Storage.SetTrialSystemAttr(trial.TrialId, LgbmParamsKey, PyJson.Dumps(snapshot));
        TrialCount++;
    }
}

/// <summary>Port of <c>_OptunaObjectiveCV</c> — the objective for the cross-validation tuner.</summary>
internal sealed class OptunaObjectiveCV : OptunaObjective
{
    private readonly int _nFold;
    private readonly bool _stratified;
    private readonly bool _shuffle;
    private readonly int _seed;
    private readonly bool _returnCvBooster;

    internal OptunaObjectiveCV(
        List<string> targetParamNames, OrderedDict<string, object?> lgbmParams, Dataset trainSet,
        int numBoostRound, ILightGbmTrainer trainer, double bestScore, string stepName, string? modelDir,
        int nFold, bool stratified, bool shuffle, int seed, bool returnCvBooster)
        : base(targetParamNames, lgbmParams, trainSet, validSets: null, validNames: null,
               numBoostRound, trainer, bestScore, stepName, modelDir)
    {
        _nFold = nFold;
        _stratified = stratified;
        _shuffle = shuffle;
        _seed = seed;
        _returnCvBooster = returnCvBooster;
    }

    /// <summary>Port of <c>_get_cv_scores</c> — the per-iteration mean-metric list.</summary>
    private IReadOnlyList<double> GetCvScores(CvResult cvResults)
    {
        string metric = GetMetricForObjective();
        string metricKey = $"{metric}-mean";
        // LightGBM v4.0.0+ prefixes the metric name with "valid ".
        return cvResults.Results.TryGetValue(metricKey, out var scores)
            ? scores
            : cvResults.Results["valid " + metricKey];
    }

    internal override double Call(Trial trial)
    {
        Preprocess(trial);

        var sw = Stopwatch.StartNew();
        var cvResults = Trainer.Cv(new CvRequest(
            LgbmParams, TrainSet, NumBoostRound, _nFold, _stratified, _shuffle, _seed, _returnCvBooster));

        var valScores = GetCvScores(cvResults);
        double valScore = valScores[^1];
        double elapsedSecs = sw.Elapsed.TotalSeconds;
        double averageIterationTime = elapsedSecs / Math.Max(1, valScores.Count);

        // Persist this trial's CV booster (Python `pickle.dump((cvbooster.boosters, best_iteration), ...)`),
        // matching Python's guard: only when `model_dir` is set AND `return_cvbooster` is on.
        if (ModelDir is not null && _returnCvBooster && cvResults.CvBooster is not null
            && Trainer is ILightGbmBoosterStore store)
        {
            string path = BoosterPath(trial.Number);
            store.SaveCvBooster(cvResults.CvBooster, path);
            Log.Info($"The booster of trial#{trial.Number} was saved as {path}.");
        }

        if (CompareValidationMetrics(valScore, BestScoreValue))
        {
            BestScoreValue = valScore;
            if (_returnCvBooster && cvResults.CvBooster is not null)
                BestBoosterWithTrialNumber = (cvResults.CvBooster, trial.Number);
        }

        Postprocess(trial, elapsedSecs, averageIterationTime);
        return valScore;
    }
}
