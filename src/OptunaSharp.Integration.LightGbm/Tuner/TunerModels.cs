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

    /// <summary>Whether to stratify the folds (accepted for parity; ignored for regression).</summary>
    public bool Stratified { get; }

    /// <summary>Whether to shuffle rows before folding.</summary>
    public bool Shuffle { get; }

    /// <summary>The fold-splitting seed (<c>seed</c>).</summary>
    public int Seed { get; }

    /// <summary>Whether to retain the per-fold models.</summary>
    public bool ReturnCvBooster { get; }

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
    Booster Train(TrainRequest request);

    /// <summary>Cross-validate and return the metric aggregates — the analog of <c>lightgbm.cv</c>.</summary>
    CvResult Cv(CvRequest request);
}
