using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
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
    /// <param name="stratified">The requested stratification policy passed to the trainer. The default ML.NET adapter does not stratify labels.</param>
    /// <param name="shuffle">Whether to shuffle rows before folding.</param>
    /// <param name="seed">The fold-splitting seed (<c>seed</c>).</param>
    /// <param name="timeBudget">Optional wall-clock budget for tuning, in seconds.</param>
    /// <param name="sampleSize">Optional cap on training rows used for tuning (keeps the last N rows).</param>
    /// <param name="study">Optional pre-created study; its direction must match the metric's.</param>
    /// <param name="optunaCallbacks">Optional callbacks invoked at the end of each Optuna trial.</param>
    /// <param name="modelDir">Optional directory in which each trial's CV booster is saved (with <c>returnCvBooster</c>; see <see cref="GetBestBooster" />); created if missing. Requires a trainer implementing <see cref="ILightGbmBoosterStore" /> (the default does).</param>
    /// <param name="showProgressBar">Accepted for parity; defaults to <c>false</c> (a .NET library divergence).</param>
    /// <param name="returnCvBooster">Whether <see cref="GetBestBooster" /> is enabled (retains fold models).</param>
    /// <param name="optunaSeed">Seed for the per-step <c>GridSampler</c>/<c>TPESampler</c> (reproducibility).</param>
    /// <param name="trainer">The training backend; defaults to <see cref="MlNetLightGbmTrainer" />.</param>
    /// <summary>Creates a stepwise LightGBM tuner that scores each trial through cross-validation.</summary>
    /// <exception cref="NumSharp.ValueError">The existing study direction is inconsistent with the selected validation metric.</exception>
    /// <exception cref="System.NotSupportedException">A model directory is requested but the custom trainer does not implement booster persistence.</exception>
    /// <exception cref="System.IO.IOException">The requested model directory cannot be created.</exception>
    /// <exception cref="System.UnauthorizedAccessException">Access to the requested model directory is denied.</exception>
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

    /// <inheritdoc />
    private protected override OptunaObjective CreateObjective(
        List<string> targetParamNames, Dataset trainSet, string stepName)
        => new OptunaObjectiveCV(
            targetParamNames, LgbmParams, trainSet, NumBoostRound, Trainer, BestScore, stepName,
            ModelDir, _nFold, _stratified, _shuffle, _seed, _returnCvBooster);

    /// <summary>
    ///     Port of <c>get_best_booster</c> — the fold models of the best trial. Returns the in-memory best
    ///     when it belongs to the current best trial; otherwise (a resumed / distributed run) it is reloaded
    ///     from <c>model_dir</c> via the trainer's <see cref="ILightGbmBoosterStore" />. Requires
    ///     <c>returnCvBooster</c> (as in Python).
    /// </summary>
    /// <returns>The best completed trial's booster, recovered from the model store when necessary.</returns>
    /// <exception cref="NumSharp.ValueError">No trial has completed, the best model cannot be found locally or in the shared model store, or CV models were not requested.</exception>
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

        // The best CV booster exists on disk but not in this instance (resuming / parallelization).
        var store = (ILightGbmBoosterStore)Trainer;   // guaranteed by the constructor when ModelDir is set
        string path = Path.Combine(ModelDir, Study.BestTrial.Number.ToString(CultureInfo.InvariantCulture));
        if (!store.BoosterExists(path))
            throw new ValueError(
                $"The best booster cannot be found in {ModelDir}. If you execute `LightGBMTunerCV` in "
                + "distributed environment, please use network file system (e.g., NFS) to share models "
                + "with multiple workers.");

        return store.LoadCvBooster(path);
    }
}
