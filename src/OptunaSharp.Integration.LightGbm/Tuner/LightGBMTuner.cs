using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
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
    /// <param name="modelDir">Optional directory in which each trial's booster is saved (see <see cref="GetBestBooster" />); created if missing. Requires a trainer implementing <see cref="ILightGbmBoosterStore" /> (the default does).</param>
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
    ///     Port of <c>get_best_booster</c> — the booster of the best trial. Returns the in-memory best when
    ///     it belongs to the current best trial; otherwise (a resumed / distributed run) it is reloaded from
    ///     <c>model_dir</c> via the trainer's <see cref="ILightGbmBoosterStore" /> — the analog of Python's
    ///     pickle reload (the .NET port persists the ML.NET model + a JSON score sidecar, not a pickle).
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

        // The best booster exists on disk but not in this instance (resuming / parallelization).
        var store = (ILightGbmBoosterStore)Trainer;   // guaranteed by the constructor when ModelDir is set
        string path = Path.Combine(ModelDir, Study.BestTrial.Number.ToString(CultureInfo.InvariantCulture));
        if (!store.BoosterExists(path))
            throw new ValueError(
                $"The best booster cannot be found in {ModelDir}. If you execute `LightGBMTuner` in "
                + "distributed environment, please use network file system (e.g., NFS) to share models "
                + "with multiple workers.");

        return store.LoadBooster(path);
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
