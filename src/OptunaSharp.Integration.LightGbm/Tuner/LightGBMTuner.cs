using System;
using System.Collections.Generic;
using NumSharp;

namespace OptunaSharp.Integration.LightGbm;

/// <summary>
///     Hyperparameter tuner for LightGBM — a .NET port of
///     <c>optuna_integration.lightgbm.LightGBMTuner</c>. It optimizes, in a stepwise manner,
///     <c>lambda_l1</c>, <c>lambda_l2</c>, <c>num_leaves</c>, <c>feature_fraction</c>,
///     <c>bagging_fraction</c>, <c>bagging_freq</c>, and <c>min_child_samples</c> — training through
///     ML.NET's native LightGBM (or a caller-supplied <see cref="ILightGbmTrainer" />) and scoring on
///     <c>validSets</c>, exactly as the Python <c>LightGBMTuner</c> invokes <c>lightgbm.train</c>.
///
///     <para>The search itself (which parameters, in which order, with which Optuna sampler and seed) is a
///     faithful port and rides OptunaSharp's bit-exact <c>GridSampler</c> / <c>TPESampler</c>; the
///     training + metric evaluation is the adapter (Phase 6 is an adapter phase — validated by usage, not
///     bit-parity against Python LightGBM). See
///     <see href="https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258">
///     the LightGBM Tuner article</see> for the algorithm.</para>
/// </summary>
public sealed class LightGBMTuner : LightGbmBaseTuner
{
    /// <param name="params">LightGBM parameters (native names: <c>objective</c>, <c>metric</c>, …).</param>
    /// <param name="trainSet">The training set.</param>
    /// <param name="validSets">The validation set(s) to score. Required (as in Python).</param>
    /// <param name="numBoostRound">The number of boosting rounds per trial (<c>num_boost_round</c>).</param>
    /// <param name="validNames">Optional names for the validation sets (defaults to <c>valid_0</c>, …).</param>
    /// <param name="timeBudget">Optional wall-clock budget for tuning, in seconds.</param>
    /// <param name="sampleSize">Optional cap on training rows used for tuning (keeps the last N rows).</param>
    /// <param name="study">Optional pre-created study; its direction must match the metric's.</param>
    /// <param name="optunaCallbacks">Optional callbacks invoked at the end of each Optuna trial.</param>
    /// <param name="modelDir">Optional directory argument (booster disk-sharing is not yet supported).</param>
    /// <param name="showProgressBar">Accepted for parity; defaults to <c>false</c> (a .NET library divergence).</param>
    /// <param name="optunaSeed">Seed for the per-step <c>GridSampler</c>/<c>TPESampler</c> (reproducibility).</param>
    /// <param name="trainer">The training backend; defaults to <see cref="MlNetLightGbmTrainer" />.</param>
    public LightGBMTuner(
        IReadOnlyDictionary<string, object?> @params,
        Dataset trainSet,
        IReadOnlyList<Dataset>? validSets,
        int numBoostRound = 1000,
        IReadOnlyList<string>? validNames = null,
        int? timeBudget = null,
        int? sampleSize = null,
        Study? study = null,
        IEnumerable<Action<Study, FrozenTrial>>? optunaCallbacks = null,
        string? modelDir = null,
        bool showProgressBar = false,
        int? optunaSeed = null,
        ILightGbmTrainer? trainer = null)
        : base(@params, trainSet, numBoostRound, timeBudget, sampleSize, study,
               optunaCallbacks, showProgressBar, modelDir, optunaSeed, trainer)
    {
        if (validSets is null)
            throw new ValueError("`valid_sets` is required.");

        ValidSets = validSets;
        ValidNames = validNames;
    }

    private protected override OptunaObjective CreateObjective(
        List<string> targetParamNames, Dataset trainSet, string stepName)
        => new OptunaObjective(
            targetParamNames, LgbmParams, trainSet, ValidSets, ValidNames, NumBoostRound, Trainer,
            BestScore, stepName, ModelDir);

    /// <summary>
    ///     Port of <c>get_best_booster</c> — the booster of the best trial. Throws when the best booster
    ///     is not available in this process (the disk-sharing path of the Python version, which relied on
    ///     pickle, is not yet ported).
    /// </summary>
    public Booster GetBestBooster()
    {
        if (BestBoosterWithTrialNumber is { } bb && bb.TrialNumber == Study.BestTrial.Number)
            return (Booster)bb.Booster;

        if (Study.Trials.Count == 0)
            throw new ValueError("The best booster is not available because no trials completed.");

        if (ModelDir is null)
            throw new ValueError(
                "The best booster cannot be found. It may be found in the other processes due to "
                + "resuming or distributed computing. Please set the `model_dir` argument of "
                + "`LightGBMTuner.__init__` and make sure that boosters are shared with all "
                + "processes.");

        throw new ValueError(
            $"The best booster cannot be found in {ModelDir}. Cross-process booster sharing via "
            + "`model_dir` is not yet supported by the ML.NET port; run tuning in a single process.");
    }

    /// <summary>
    ///     Port of <c>optuna_integration.lightgbm.train</c> — the drop-in wrapper that builds a
    ///     <see cref="LightGBMTuner" />, runs the full stepwise tuning, and returns the best booster.
    /// </summary>
    public static Booster Train(
        IReadOnlyDictionary<string, object?> @params,
        Dataset trainSet,
        IReadOnlyList<Dataset>? validSets,
        int numBoostRound = 1000,
        IReadOnlyList<string>? validNames = null,
        int? timeBudget = null,
        int? sampleSize = null,
        Study? study = null,
        IEnumerable<Action<Study, FrozenTrial>>? optunaCallbacks = null,
        string? modelDir = null,
        bool showProgressBar = false,
        int? optunaSeed = null,
        ILightGbmTrainer? trainer = null)
    {
        var tuner = new LightGBMTuner(
            @params, trainSet, validSets, numBoostRound, validNames, timeBudget, sampleSize, study,
            optunaCallbacks, modelDir, showProgressBar, optunaSeed, trainer);
        tuner.Run();
        return tuner.GetBestBooster();
    }
}
