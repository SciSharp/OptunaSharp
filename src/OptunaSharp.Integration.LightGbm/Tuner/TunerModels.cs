using System.Collections.Generic;

namespace OptunaSharp.Integration.LightGbm;

/// <summary>
///     The C# analog of a trained <c>lightgbm.Booster</c> — what a single tuner trial produces. The
///     tuner reads exactly two things from it, mirroring
///     <c>optuna_integration.lightgbm._lightgbm_tuner.optimize._OptunaObjective</c>:
///     <see cref="BestScore" /> (<c>booster.best_score[valid_name][metric]</c>) for the trial's
///     objective value, and <see cref="CurrentIteration" /> (<c>booster.current_iteration()</c>) for the
///     cosmetic <c>average_iteration_time</c> system attribute. <see cref="Model" /> carries the trained
///     model so <see cref="LightGBMTuner.GetBestBooster" /> can return it.
/// </summary>
public sealed class Booster
{
    /// <summary><c>best_score[valid_name][metric]</c> — the validation score per set, per metric.</summary>
    public IReadOnlyDictionary<string, IReadOnlyDictionary<string, double>> BestScore { get; }

    /// <summary><c>booster.current_iteration()</c> — the number of boosting rounds actually run.</summary>
    public int CurrentIteration { get; }

    /// <summary>
    ///     The trained model (an ML.NET <c>ITransformer</c> for the default trainer, or whatever a custom
    ///     <see cref="ILightGbmTrainer" /> returns). <c>null</c> for trainers that do not retain a model.
    /// </summary>
    public object? Model { get; }

    /// <summary>Retains one fitted model together with its validation metrics and completed-round count.</summary>
    /// <param name="bestScore">Validation metric scores grouped by validation-set name and metric name.</param>
    /// <param name="currentIteration">The number of boosting rounds actually completed.</param>
    /// <param name="model">The fitted model object, or null when no representable model is retained.</param>
    public Booster(
        IReadOnlyDictionary<string, IReadOnlyDictionary<string, double>> bestScore,
        int currentIteration, object? model = null)
    {
        BestScore = bestScore;
        CurrentIteration = currentIteration;
        Model = model;
    }
}

/// <summary>
///     The C# analog of a <c>lightgbm.CVBooster</c> — the per-fold models produced by a cross-validation
///     trial (returned by <see cref="LightGBMTunerCV.GetBestBooster" /> when <c>returnCvBooster</c> is on).
/// </summary>
public sealed class CvBooster
{
    /// <summary>The trained model of each fold.</summary>
    public IReadOnlyList<object?> Boosters { get; }

    /// <summary>The best boosting iteration (kept for parity; the default trainer runs a fixed count).</summary>
    public int BestIteration { get; }

    /// <summary>Retains the per-fold models and their selected boosting iteration.</summary>
    /// <param name="boosters">The retained model for each cross-validation fold, with null entries for folds lacking a representable model.</param>
    /// <param name="bestIteration">The boosting iteration associated with the retained fold models.</param>
    public CvBooster(IReadOnlyList<object?> boosters, int bestIteration)
    {
        Boosters = boosters;
        BestIteration = bestIteration;
    }
}

/// <summary>
///     The C# analog of the dict returned by <c>lightgbm.cv</c> — metric aggregates keyed
///     <c>"{metric}-mean"</c> / <c>"{metric}-stdv"</c>, each a per-iteration list (the tuner reads the
///     last element). Mirrors <c>_OptunaObjectiveCV._get_cv_scores</c>.
/// </summary>
public sealed class CvResult
{
    /// <summary><c>cv_results["{metric}-mean"]</c> etc. — one list per aggregated metric key.</summary>
    public IReadOnlyDictionary<string, IReadOnlyList<double>> Results { get; }

    /// <summary>The fold models, when the request asked for them (<c>return_cvbooster=True</c>).</summary>
    public CvBooster? CvBooster { get; }

    /// <summary>Retains aggregated cross-validation metric histories and optional fold models.</summary>
    /// <param name="results">Aggregated metric histories keyed by metric and aggregation suffix.</param>
    /// <param name="cvBooster">Optional retained cross-validation fold models.</param>
    public CvResult(
        IReadOnlyDictionary<string, IReadOnlyList<double>> results, CvBooster? cvBooster = null)
    {
        Results = results;
        CvBooster = cvBooster;
    }
}

/// <summary>The inputs of one <c>lightgbm.train</c> call — see <see cref="ILightGbmTrainer.Train" />.</summary>
public sealed class TrainRequest
{
    /// <summary>The accumulated LightGBM parameters for this trial (native LightGBM names).</summary>
    public IReadOnlyDictionary<string, object?> Params { get; }

    /// <summary>The training set.</summary>
    public Dataset TrainSet { get; }

    /// <summary>The validation set(s) to score.</summary>
    public IReadOnlyList<Dataset> ValidSets { get; }

    /// <summary>The name of each validation set (aligned with <see cref="ValidSets" />).</summary>
    public IReadOnlyList<string> ValidNames { get; }

    /// <summary>The number of boosting rounds (<c>num_boost_round</c>).</summary>
    public int NumBoostRound { get; }

    /// <summary>Packages the parameters and datasets for one single-fit LightGBM training call.</summary>
    /// <param name="params">LightGBM parameter names and values, using native LightGBM spellings.</param>
    /// <param name="trainSet">The dataset used for training or cross-validation.</param>
    /// <param name="validSets">The validation datasets whose metrics are recorded.</param>
    /// <param name="validNames">Names aligned with the validation datasets; omitted names are resolved by the tuner.</param>
    /// <param name="numBoostRound">The requested number of boosting rounds for each trial.</param>
    public TrainRequest(
        IReadOnlyDictionary<string, object?> @params, Dataset trainSet,
        IReadOnlyList<Dataset> validSets, IReadOnlyList<string> validNames, int numBoostRound)
    {
        Params = @params;
        TrainSet = trainSet;
        ValidSets = validSets;
        ValidNames = validNames;
        NumBoostRound = numBoostRound;
    }
}

/// <summary>The inputs of one <c>lightgbm.cv</c> call — see <see cref="ILightGbmTrainer.Cv" />.</summary>
public sealed class CvRequest
{
    /// <summary>The accumulated LightGBM parameters for this trial (native LightGBM names).</summary>
    public IReadOnlyDictionary<string, object?> Params { get; }

    /// <summary>The full training set to fold.</summary>
    public Dataset TrainSet { get; }

    /// <summary>The number of boosting rounds (<c>num_boost_round</c>).</summary>
    public int NumBoostRound { get; }

    /// <summary>The number of CV folds (<c>nfold</c>).</summary>
    public int NFold { get; }

    /// <summary>The requested stratification policy. The default ML.NET adapter currently uses row or whole-query-group folds without label stratification.</summary>
    public bool Stratified { get; }

    /// <summary>Whether to shuffle rows before folding.</summary>
    public bool Shuffle { get; }

    /// <summary>The fold-splitting seed (<c>seed</c>).</summary>
    public int Seed { get; }

    /// <summary>Whether to retain the per-fold models.</summary>
    public bool ReturnCvBooster { get; }

    /// <summary>Packages the parameters and fold settings for one LightGBM cross-validation call.</summary>
    /// <param name="params">LightGBM parameter names and values, using native LightGBM spellings.</param>
    /// <param name="trainSet">The dataset used for training or cross-validation.</param>
    /// <param name="numBoostRound">The requested number of boosting rounds for each trial.</param>
    /// <param name="nFold">The requested number of cross-validation folds.</param>
    /// <param name="stratified">The requested stratification flag passed through the trainer request; the default adapter currently splits rows or query groups.</param>
    /// <param name="shuffle">Whether to shuffle row or query-group order before assigning folds.</param>
    /// <param name="seed">The training or fold-shuffling seed, as specified by the operation.</param>
    /// <param name="returnCvBooster">Whether to retain each cross-validation fold's fitted model.</param>
    public CvRequest(
        IReadOnlyDictionary<string, object?> @params, Dataset trainSet, int numBoostRound,
        int nFold, bool stratified, bool shuffle, int seed, bool returnCvBooster)
    {
        Params = @params;
        TrainSet = trainSet;
        NumBoostRound = numBoostRound;
        NFold = nFold;
        Stratified = stratified;
        Shuffle = shuffle;
        Seed = seed;
        ReturnCvBooster = returnCvBooster;
    }
}

/// <summary>
///     The training seam of the LightGBM tuner — the one place the tuner touches a booster. The
///     <b>stepwise search</b> (which parameters, in which order, with which Optuna sampler) is a faithful
///     port of <c>optuna_integration.lightgbm</c> and rides OptunaSharp's own <c>GridSampler</c> /
///     <c>TPESampler</c>; only the <i>training + metric evaluation</i> is an adapter. The default
///     implementation, <see cref="MlNetLightGbmTrainer" />, bridges to ML.NET's native LightGBM; inject a
///     custom one to plug a different engine (or a deterministic scorer, for tests).
/// </summary>
public interface ILightGbmTrainer
{
    /// <summary>Train one booster and return its validation scores — the analog of <c>lightgbm.train</c>.</summary>
    /// <param name="request">The trial parameters, data, iteration count, and validation configuration to execute.</param>
    /// <returns>The booster with the best validation score after stepwise tuning.</returns>
    Booster Train(TrainRequest request);

    /// <summary>Cross-validate and return the metric aggregates — the analog of <c>lightgbm.cv</c>.</summary>
    /// <param name="request">The trial parameters, data, iteration count, and validation configuration to execute.</param>
    /// <returns>The aggregated cross-validation metric histories and requested fold models.</returns>
    CvResult Cv(CvRequest request);
}

/// <summary>
///     The optional booster-persistence capability behind the tuner's <c>model_dir</c> feature — the
///     analog of the Python tuner pickling each trial's booster to <c>{model_dir}/{trial}.pkl</c> so
///     <see cref="LightGBMTuner.GetBestBooster" /> / <see cref="LightGBMTunerCV.GetBestBooster" /> can
///     retrieve the best booster after resuming or across distributed workers. A trainer implements this
///     to teach the tuner how to write and read <i>its own</i> model type (Python's pickle is not portable
///     to a .NET / ML.NET model). The default <see cref="MlNetLightGbmTrainer" /> implements it (ML.NET's
///     <c>MLContext.Model.Save</c> / <c>Load</c> plus a small JSON sidecar for the scores); supplying a
///     <c>model_dir</c> with a custom <see cref="ILightGbmTrainer" /> that does not implement this is a
///     construction-time error.
///
///     <para>Each method receives an extensionless base <c>path</c> (e.g.
///     <c>{model_dir}/3</c>); the store owns the concrete on-disk layout under it.</para>
/// </summary>
public interface ILightGbmBoosterStore
{
    /// <summary>Persist a single-fit booster (the <see cref="LightGBMTuner" /> path) under <paramref name="path" />.</summary>
    /// <param name="booster">The trained booster or collection of fold models to inspect or persist.</param>
    /// <param name="path">The model store's extensionless base path, or the concrete JSON path for metadata helpers.</param>
    void SaveBooster(Booster booster, string path);

    /// <summary>Reload a booster previously written by <see cref="SaveBooster" />.</summary>
    /// <param name="path">The model store's extensionless base path, or the concrete JSON path for metadata helpers.</param>
    /// <returns>The restored single-fit booster with its stored metrics.</returns>
    Booster LoadBooster(string path);

    /// <summary>Persist a CV booster (the <see cref="LightGBMTunerCV" /> path) under <paramref name="path" />.</summary>
    /// <param name="booster">The trained booster or collection of fold models to inspect or persist.</param>
    /// <param name="path">The model store's extensionless base path, or the concrete JSON path for metadata helpers.</param>
    void SaveCvBooster(CvBooster booster, string path);

    /// <summary>Reload a CV booster previously written by <see cref="SaveCvBooster" />.</summary>
    /// <param name="path">The model store's extensionless base path, or the concrete JSON path for metadata helpers.</param>
    /// <returns>The restored fold models and their selected iteration.</returns>
    CvBooster LoadCvBooster(string path);

    /// <summary>Whether a booster (single-fit or CV) is stored under <paramref name="path" />.</summary>
    /// <param name="path">The model store's extensionless base path, or the concrete JSON path for metadata helpers.</param>
    /// <returns>Whether this model store has the metadata identifying a saved booster at the base path.</returns>
    bool BoosterExists(string path);
}
