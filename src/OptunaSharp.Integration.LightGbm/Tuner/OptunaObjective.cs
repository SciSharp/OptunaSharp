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
    /// <summary>The trial system-attribute key holding elapsed training time in seconds.</summary>
    internal const string ElapsedSecsKey = "lightgbm_tuner:elapsed_secs";
    /// <summary>The trial system-attribute key holding seconds per boosting iteration.</summary>
    internal const string AverageIterationTimeKey = "lightgbm_tuner:average_iteration_time";
    /// <summary>The trial system-attribute key identifying the stepwise tuning stage.</summary>
    internal const string StepNameKey = "lightgbm_tuner:step_name";
    /// <summary>The trial system-attribute key holding the serialized LightGBM parameter snapshot.</summary>
    internal const string LgbmParamsKey = "lightgbm_tuner:lgbm_params";

    private const double Eps = 1e-12;                 // Python `_EPS`
    private const int DefaultTunerTreeDepth = 8;      // Python `_DEFAULT_TUNER_TREE_DEPTH`

    // Port of `_logger = optuna.logging.get_logger(__name__)`; used to announce a saved booster (model_dir).
    private protected static readonly Logger Log = OptunaLogging.GetLogger("optuna_integration.lightgbm");

    // The 7 tunable parameters (Python `_DEFAULT_LIGHTGBM_PARAMETERS` keys).
    /// <summary>The seven LightGBM parameter names supported by the stepwise objective.</summary>
    internal static readonly string[] TunableParameters =
        { "lambda_l1", "lambda_l2", "num_leaves", "feature_fraction", "bagging_fraction", "bagging_freq", "min_child_samples" };

    /// <summary>The subset of supported LightGBM parameters varied in this step.</summary>
    protected readonly List<string> TargetParamNames;
    /// <summary>The training dataset used by every trial in this step.</summary>
    protected readonly Dataset TrainSet;
    /// <summary>The requested boosting rounds per trial.</summary>
    protected readonly int NumBoostRound;
    /// <summary>The component responsible for fitting models and evaluating validation metrics.</summary>
    protected readonly ILightGbmTrainer Trainer;
    /// <summary>The stable step identifier recorded in trial metadata.</summary>
    protected readonly string StepName;
    /// <summary>The optional directory used to persist each trial's model.</summary>
    protected readonly string? ModelDir;

    /// <summary>The best validation score this step has seen so far.</summary>
    internal double BestScoreValue;

    /// <summary>The best booster (or CV booster) + the trial number that produced it, or <c>null</c>.</summary>
    internal (object Booster, int TrialNumber)? BestBoosterWithTrialNumber;

    /// <summary>The number of trials whose postprocessing has completed in this objective instance.</summary>
    internal int TrialCount;

    /// <summary>Creates the single-fit objective for one named tuning stage.</summary>
    /// <param name="targetParamNames">The LightGBM parameters varied during this tuning step.</param>
    /// <param name="lgbmParams">The mutable ordered LightGBM parameter dictionary, shared with the current tuning step.</param>
    /// <param name="trainSet">The dataset used for training or cross-validation.</param>
    /// <param name="validSets">The validation datasets whose metrics are recorded.</param>
    /// <param name="validNames">Names aligned with the validation datasets; omitted names are resolved by the tuner.</param>
    /// <param name="numBoostRound">The requested number of boosting rounds for each trial.</param>
    /// <param name="trainer">The training implementation, or null to use the ML.NET LightGBM adapter.</param>
    /// <param name="bestScore">The current best score, or the validation-set metric mapping when constructing a Booster.</param>
    /// <param name="stepName">The stable tuning-step name recorded in trial system attributes.</param>
    /// <param name="modelDir">An optional directory shared by workers for persisted trial boosters.</param>
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

    /// <summary>Rejects a tuning stage containing a parameter outside the supported stepwise search.</summary>
    /// <exception cref="System.NotImplementedException">A target parameter is outside the seven supported LightGBM tuning parameters.</exception>
    private void CheckTargetNamesSupported()
    {
        foreach (var name in TargetParamNames)
            if (Array.IndexOf(TunableParameters, name) < 0)
                throw new NotImplementedException($"Parameter `{name}` is not supported for tuning.");
    }

    /// <summary>Port of <c>_preprocess</c> — suggest the step's target params into <c>lgbm_params</c>.</summary>
    /// <param name="trial">The Optuna trial whose suggested parameters and training metadata are being updated.</param>
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
    /// <param name="trial">The Optuna trial whose suggested parameters and training metadata are being updated.</param>
    /// <returns>The validation score optimized by the current study direction.</returns>
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
    /// <param name="trialNumber">The zero-based study-local trial number used in the persisted booster path.</param>
    /// <returns>The extensionless model-store path for this trial number.</returns>
    private protected string BoosterPath(int trialNumber)
        => Path.Combine(ModelDir!, trialNumber.ToString(CultureInfo.InvariantCulture));

    /// <summary>Port of <c>_postprocess</c> — record the elapsed / iteration-time / step / params attrs.</summary>
    /// <param name="trial">The Optuna trial whose suggested parameters and training metadata are being updated.</param>
    /// <param name="elapsedSecs">The measured wall-clock duration of this trial's training and scoring in seconds.</param>
    /// <param name="averageIterationTime">The elapsed time divided by the number of completed boosting rounds.</param>
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

    /// <summary>Creates the cross-validation objective for one named tuning stage.</summary>
    /// <param name="targetParamNames">The LightGBM parameters varied during this tuning step.</param>
    /// <param name="lgbmParams">The mutable ordered LightGBM parameter dictionary, shared with the current tuning step.</param>
    /// <param name="trainSet">The dataset used for training or cross-validation.</param>
    /// <param name="numBoostRound">The requested number of boosting rounds for each trial.</param>
    /// <param name="trainer">The training implementation, or null to use the ML.NET LightGBM adapter.</param>
    /// <param name="bestScore">The current best score, or the validation-set metric mapping when constructing a Booster.</param>
    /// <param name="stepName">The stable tuning-step name recorded in trial system attributes.</param>
    /// <param name="modelDir">An optional directory shared by workers for persisted trial boosters.</param>
    /// <param name="nFold">The requested number of cross-validation folds.</param>
    /// <param name="stratified">The requested stratification flag passed through the trainer request; the default adapter currently splits rows or query groups.</param>
    /// <param name="shuffle">Whether to shuffle row or query-group order before assigning folds.</param>
    /// <param name="seed">The training or fold-shuffling seed, as specified by the operation.</param>
    /// <param name="returnCvBooster">Whether to retain each cross-validation fold's fitted model.</param>
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
    /// <param name="cvResults">The aggregated cross-validation metrics returned by the trainer.</param>
    /// <returns>The per-iteration mean scores of the selected optimization metric.</returns>
    private IReadOnlyList<double> GetCvScores(CvResult cvResults)
    {
        string metric = GetMetricForObjective();
        string metricKey = $"{metric}-mean";
        // LightGBM v4.0.0+ prefixes the metric name with "valid ".
        return cvResults.Results.TryGetValue(metricKey, out var scores)
            ? scores
            : cvResults.Results["valid " + metricKey];
    }

    /// <inheritdoc />
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
