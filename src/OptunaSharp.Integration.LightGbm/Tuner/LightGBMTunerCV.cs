using System;
using System.Collections.Generic;
using NumSharp;

namespace OptunaSharp.Integration.LightGbm;

/// <summary>
///     Hyperparameter tuner for LightGBM with cross-validation — a .NET port of
///     <c>optuna_integration.lightgbm.LightGBMTunerCV</c>. It runs the same stepwise search as
///     <see cref="LightGBMTuner" /> but scores each trial by <b>k-fold cross-validation</b> on
///     <c>trainSet</c> (the analog of <c>lightgbm.cv</c>) instead of a held-out validation set.
/// </summary>
public sealed class LightGBMTunerCV : LightGbmBaseTuner
{
    private readonly int _nFold;
    private readonly bool _stratified;
    private readonly bool _shuffle;
    private readonly int _seed;
    private readonly bool _returnCvBooster;

    /// <param name="params">LightGBM parameters (native names: <c>objective</c>, <c>metric</c>, …).</param>
    /// <param name="trainSet">The full training set to cross-validate.</param>
    /// <param name="numBoostRound">The number of boosting rounds per trial (<c>num_boost_round</c>).</param>
    /// <param name="nFold">The number of CV folds (<c>nfold</c>).</param>
    /// <param name="stratified">Whether to stratify folds (accepted for parity; ignored for regression).</param>
    /// <param name="shuffle">Whether to shuffle rows before folding.</param>
    /// <param name="seed">The fold-splitting seed (<c>seed</c>).</param>
    /// <param name="timeBudget">Optional wall-clock budget for tuning, in seconds.</param>
    /// <param name="sampleSize">Optional cap on training rows used for tuning (keeps the last N rows).</param>
    /// <param name="study">Optional pre-created study; its direction must match the metric's.</param>
    /// <param name="optunaCallbacks">Optional callbacks invoked at the end of each Optuna trial.</param>
    /// <param name="modelDir">Optional directory argument (booster disk-sharing is not yet supported).</param>
    /// <param name="showProgressBar">Accepted for parity; defaults to <c>false</c> (a .NET library divergence).</param>
    /// <param name="returnCvBooster">Whether <see cref="GetBestBooster" /> is enabled (retains fold models).</param>
    /// <param name="optunaSeed">Seed for the per-step <c>GridSampler</c>/<c>TPESampler</c> (reproducibility).</param>
    /// <param name="trainer">The training backend; defaults to <see cref="MlNetLightGbmTrainer" />.</param>
    public LightGBMTunerCV(
        IReadOnlyDictionary<string, object?> @params,
        Dataset trainSet,
        int numBoostRound = 1000,
        int nFold = 5,
        bool stratified = true,
        bool shuffle = true,
        int seed = 0,
        int? timeBudget = null,
        int? sampleSize = null,
        Study? study = null,
        IEnumerable<Action<Study, FrozenTrial>>? optunaCallbacks = null,
        string? modelDir = null,
        bool showProgressBar = false,
        bool returnCvBooster = false,
        int? optunaSeed = null,
        ILightGbmTrainer? trainer = null)
        : base(@params, trainSet, numBoostRound, timeBudget, sampleSize, study,
               optunaCallbacks, showProgressBar, modelDir, optunaSeed, trainer)
    {
        _nFold = nFold;
        _stratified = stratified;
        _shuffle = shuffle;
        _seed = seed;
        _returnCvBooster = returnCvBooster;
    }

    private protected override OptunaObjective CreateObjective(
        List<string> targetParamNames, Dataset trainSet, string stepName)
        => new OptunaObjectiveCV(
            targetParamNames, LgbmParams, trainSet, NumBoostRound, Trainer, BestScore, stepName,
            ModelDir, _nFold, _stratified, _shuffle, _seed, _returnCvBooster);

    /// <summary>Port of <c>get_best_booster</c> — the fold models of the best trial.</summary>
    public CvBooster GetBestBooster()
    {
        if (!_returnCvBooster)
            throw new ValueError(
                "LightGBMTunerCV requires `return_cvbooster=True` for method `get_best_booster()`.");

        if (BestBoosterWithTrialNumber is { } bb && bb.TrialNumber == Study.BestTrial.Number)
            return (CvBooster)bb.Booster;

        if (Study.Trials.Count == 0)
            throw new ValueError("The best booster is not available because no trials completed.");

        if (ModelDir is null)
            throw new ValueError(
                "The best booster cannot be found. It may be found in the other processes due to "
                + "resuming or distributed computing. Please set the `model_dir` argument of "
                + "`LightGBMTunerCV.__init__` and make sure that boosters are shared with all "
                + "processes.");

        throw new ValueError(
            $"The best booster cannot be found in {ModelDir}. Cross-process booster sharing via "
            + "`model_dir` is not yet supported by the ML.NET port; run tuning in a single process.");
    }
}
