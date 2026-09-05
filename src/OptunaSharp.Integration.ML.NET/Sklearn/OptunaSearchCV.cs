using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Terminator;

namespace OptunaSharp.Integration;

/// <summary>
///     Hyperparameter search with cross-validation. A .NET port of
///     <c>optuna_integration.sklearn.OptunaSearchCV</c> — the Optuna-driven, scikit-learn-<c>GridSearchCV</c>-style
///     meta-estimator, here adapted over the <see cref="ICrossValidationEstimator" /> seam (default
///     backend: <see cref="MLNetEstimator" />, ML.NET).
///
///     <para>
///     Each trial samples one value per entry of <see cref="ParamDistributions" /> from the study's
///     sampler, cross-validates the estimator configured with those values, records the per-fold / mean /
///     std scores as trial user attributes, reports the cross-validation scores for the terminator, and
///     <b>maximizes</b> the mean validation score — matching the Python <c>_Objective</c>. Following the
///     scikit-learn convention, the direction of optimization is always <c>maximize</c>; for a minimization
///     problem, negate your metric.
///     </para>
///
///     <para>
///     <b>This is a Phase-6 adapter, not a behavior-parity port</b> (there is no Python oracle for the
///     ML.NET stack). The genuinely portable essence — the Optuna search, the attribute bookkeeping, the
///     terminator reporting, and the refit-timing surface — is reproduced faithfully. Concerns that are
///     scikit-learn-estimator specifics live on the estimator seam instead: the fold strategy
///     (<c>cv</c> — KFold vs. StratifiedKFold, the sampling-key column) and the per-fold error policy
///     (<c>error_score</c>) are configured on <see cref="MLNetEstimator" />. The <c>partial_fit</c>-based
///     pruning path (<c>enable_pruning</c> / <c>max_iter</c>) and <c>subsample</c> from the Python class
///     are <b>not</b> ported in this first cut (ML.NET has no universal online-learning surface); prune an
///     iterative estimator with <see cref="PruningCallback" /> inside its own training loop instead.
///     </para>
///
///     <para>
///     Prediction is done through the refit estimator: after <see cref="Fit" /> with <c>refit = true</c>,
///     downcast <see cref="Estimator" /> to <see cref="MLNetEstimator" /> and call its
///     <c>Predict</c> / <c>Transform</c> / <c>Score</c>.
///     </para>
/// </summary>
public class OptunaSearchCV
{
    private readonly ICrossValidationEstimator _estimator;
    private readonly IReadOnlyDictionary<string, BaseDistribution> _paramDistributions;
    private readonly int? _nTrials;
    private readonly double? _timeout;
    private readonly int _nJobs;
    private readonly int? _randomState;
    private readonly bool _refit;
    private readonly bool _returnTrainScore;
    private readonly Study? _study;
    private readonly int _verbose;
    private readonly IEnumerable<Action<Study, FrozenTrial>>? _callbacks;
    private readonly Type[]? _catch;

    private bool _isFitted;

    /// <param name="estimator">
    ///     The estimator + cross-validation + scorer bundle to search over (default:
    ///     <see cref="MLNetEstimator" />).
    /// </param>
    /// <param name="paramDistributions">
    ///     Optuna parameter name → <see cref="BaseDistribution" /> to sample that parameter from.
    /// </param>
    /// <param name="nTrials">
    ///     Number of trials. If <c>null</c>, there is no limitation on the number of trials (bounded then
    ///     only by <paramref name="timeout" />). Defaults to <c>10</c>.
    /// </param>
    /// <param name="timeout">
    ///     Time limit in seconds for the search. If <c>null</c>, the search runs without a time limit.
    /// </param>
    /// <param name="nJobs">
    ///     Number of parallel jobs. <c>null</c> means <c>1</c>; <c>-1</c> means the CPU count.
    /// </param>
    /// <param name="randomState">
    ///     Seed for the pseudo-random number generator used to derive the default sampler's seed. If
    ///     <c>null</c>, an unseeded generator is used.
    /// </param>
    /// <param name="refit">
    ///     If <c>true</c>, refit the estimator with the best found hyperparameters (making prediction
    ///     available through <see cref="Estimator" />) and set <see cref="RefitTimeSeconds" />.
    /// </param>
    /// <param name="returnTrainScore">If <c>true</c>, training scores are also computed and stored.</param>
    /// <param name="study">
    ///     The study for the optimization task. If <c>null</c>, a new <c>maximize</c> study with a
    ///     <see cref="TPESampler" /> is created. If provided, its direction must be <c>maximize</c>.
    /// </param>
    /// <param name="verbose">Verbosity level. Reserved (accepted for API recognizability).</param>
    /// <param name="callbacks">
    ///     Callbacks invoked at the end of each trial, receiving the <see cref="Study" /> and the resulting
    ///     <see cref="FrozenTrial" />.
    /// </param>
    /// <param name="catch">
    ///     Exception types for which the study keeps running when a trial raises them. Default (empty) means
    ///     the study stops for any exception except <see cref="TrialPruned" />.
    /// </param>
    public OptunaSearchCV(
        ICrossValidationEstimator estimator,
        IReadOnlyDictionary<string, BaseDistribution> paramDistributions,
        int? nTrials = 10,
        double? timeout = null,
        int? nJobs = null,
        int? randomState = null,
        bool refit = true,
        bool returnTrainScore = false,
        Study? study = null,
        int verbose = 0,
        IEnumerable<Action<Study, FrozenTrial>>? callbacks = null,
        Type[]? @catch = null)
    {
        _estimator = estimator ?? throw new ArgumentNullException(nameof(estimator));
        _paramDistributions = paramDistributions ?? throw new ArgumentNullException(nameof(paramDistributions));
        _nTrials = nTrials;
        _timeout = timeout;
        _nJobs = nJobs ?? 1;
        _randomState = randomState;
        _refit = refit;
        _returnTrainScore = returnTrainScore;
        _study = study;
        _verbose = verbose;
        _callbacks = callbacks;
        _catch = @catch;
    }

    /// <summary>The parameter-name → distribution map searched over.</summary>
    public IReadOnlyDictionary<string, BaseDistribution> ParamDistributions => _paramDistributions;

    /// <summary>The estimator being searched. After a refit, downcast to <see cref="MLNetEstimator" /> to predict.</summary>
    public ICrossValidationEstimator Estimator => _estimator;

    /// <summary>The actual study driving the search. Available after <see cref="Fit" />.</summary>
    public Study Study { get; private set; } = null!;

    /// <summary>Number of cross-validation splits. Available after <see cref="Fit" />.</summary>
    public int NSplits { get; private set; }

    /// <summary>Time for refitting the best estimator, in seconds. Set only when <c>refit = true</c>.</summary>
    public double RefitTimeSeconds { get; private set; }

    /// <summary>Parameters of the best trial in the study.</summary>
    public IReadOnlyDictionary<string, object?> BestParams
    {
        get { CheckIsFitted(); return Study.BestParams; }
    }

    /// <summary>Mean cross-validated score of the best estimator (scikit-learn's <c>best_score_</c>).</summary>
    public double BestScore
    {
        get { CheckIsFitted(); return Study.BestValue; }
    }

    /// <summary>Best trial in the study.</summary>
    public FrozenTrial BestTrial
    {
        get { CheckIsFitted(); return Study.BestTrial; }
    }

    /// <summary>Trial number corresponding to the best parameter setting (scikit-learn's <c>best_index_</c>).</summary>
    public int BestIndex => BestTrial.Number;

    /// <summary>All trials in the study.</summary>
    public IReadOnlyList<FrozenTrial> Trials
    {
        get { CheckIsFitted(); return Study.Trials; }
    }

    /// <summary>The actual number of trials run.</summary>
    public int NTrials => Trials.Count;

    /// <summary>User attributes of the study.</summary>
    public IReadOnlyDictionary<string, object?> UserAttrs
    {
        get { CheckIsFitted(); return Study.UserAttrs; }
    }

    /// <summary>
    ///     A dictionary mapping a metric name to the list of its cross-validation results across all trials
    ///     (scikit-learn's <c>cv_results_</c>). The keys are taken from the first trial's user attributes.
    /// </summary>
    public IReadOnlyDictionary<string, IReadOnlyList<object?>> CvResults
    {
        get
        {
            var perTrial = Trials.Select(t => t.UserAttrs).ToList();
            var result = new OrderedDict<string, IReadOnlyList<object?>>();
            if (perTrial.Count == 0)
                return result;

            foreach (var key in perTrial[0].Keys)
            {
                var column = new List<object?>(perTrial.Count);
                foreach (var attrs in perTrial)
                    column.Add(attrs.TryGetValue(key, out var v) ? v : null);
                result[key] = column;
            }

            return result;
        }
    }

    /// <summary>
    ///     Run fit with all sets of parameters: create (or use) the study, optimize the objective, and
    ///     optionally refit the best estimator.
    /// </summary>
    /// <returns>This instance.</returns>
    public OptunaSearchCV Fit()
    {
        CheckParams();

        // check_random_state(random_state).randint(0, np.iinfo("int32").max) — derive the sampler seed.
        var randomState = new LazyRandomState(_randomState);

        NSplits = _estimator.GetNumSplits();

        if (_study is null)
        {
            int seed = (int)randomState.Rng.randint(0, 2147483647);
            var sampler = new TPESampler(seed: seed);
            Study = Optuna.CreateStudy(direction: StudyDirection.Maximize, sampler: sampler);
        }
        else
        {
            Study = _study;
        }

        Study.Optimize(
            Objective,
            nTrials: _nTrials,
            timeout: _timeout,
            nJobs: _nJobs,
            @catch: _catch,
            callbacks: _callbacks);

        // _check_is_fitted checks study_/n_splits_ (always) and best_estimator_/refit_time_ (when refit).
        _isFitted = true;

        if (_refit)
            Refit();

        return this;
    }

    // Port of _Objective.__call__ (the non-pruning path). Returns the mean test score to maximize.
    private double Objective(Trial trial)
    {
        var parameters = new Dictionary<string, object?>();
        foreach (var (name, distribution) in _paramDistributions)
            parameters[name] = trial.Suggest(name, distribution);

        var folds = _estimator.CrossValidate(parameters, _returnTrainScore);

        int n = folds.Count;
        var testScore = new double[n];
        var fitTime = new double[n];
        var scoreTime = new double[n];
        double[]? trainScore = _returnTrainScore ? new double[n] : null;

        for (int i = 0; i < n; i++)
        {
            testScore[i] = folds[i].TestScore;
            fitTime[i] = folds[i].FitTimeSeconds;
            scoreTime[i] = folds[i].ScoreTimeSeconds;
            if (trainScore is not null)
                trainScore[i] = folds[i].TrainScore ?? double.NaN;
        }

        StoreScores(trial, testScore, fitTime, scoreTime, trainScore);

        try
        {
            TerminatorFunctions.ReportCrossValidationScores(trial, testScore);
        }
        catch (ValueError e)
        {
            OptunaWarnings.Warn(
                $"Failed to report cross validation scores for TerminatorCallback, with error: {e.Message}");
        }

        return NanMean(testScore);
    }

    // Port of _Objective._store_scores. Attribute order follows scikit-learn's cross_validate result dict
    // (fit_time, score_time, test_score[, train_score]) so cv_results_ column order is recognizable.
    private void StoreScores(
        Trial trial, double[] testScore, double[] fitTime, double[] scoreTime, double[]? trainScore)
    {
        trial.SetUserAttr("mean_fit_time", NanMean(fitTime));
        trial.SetUserAttr("std_fit_time", NanStd(fitTime));

        trial.SetUserAttr("mean_score_time", NanMean(scoreTime));
        trial.SetUserAttr("std_score_time", NanStd(scoreTime));

        for (int i = 0; i < testScore.Length; i++)
            trial.SetUserAttr($"split{i}_test_score", testScore[i]);
        trial.SetUserAttr("mean_test_score", NanMean(testScore));
        trial.SetUserAttr("std_test_score", NanStd(testScore));

        if (trainScore is not null)
        {
            for (int i = 0; i < trainScore.Length; i++)
                trial.SetUserAttr($"split{i}_train_score", trainScore[i]);
            trial.SetUserAttr("mean_train_score", NanMean(trainScore));
            trial.SetUserAttr("std_train_score", NanStd(trainScore));
        }
    }

    private void Refit()
    {
        var stopwatch = Stopwatch.StartNew();
        _estimator.Refit(Study.BestParams);
        RefitTimeSeconds = stopwatch.Elapsed.TotalSeconds;
    }

    // Port of _check_params (the parts that survive the adapter): only the study-direction guard remains.
    private void CheckParams()
    {
        if (_study is not null && _study.Direction != StudyDirection.Maximize)
            throw new ValueError("direction of study must be 'maximize'.");
    }

    private void CheckIsFitted()
    {
        if (!_isFitted)
            throw new InvalidOperationException(
                "This OptunaSearchCV instance is not fitted yet. Call Fit before using this estimator.");
    }

    // np.nanmean with numpy's default ddof=0 semantics; NaN entries are ignored, all-NaN -> NaN.
    private static double NanMean(double[] values)
    {
        double sum = 0.0;
        int count = 0;
        foreach (var v in values)
            if (!double.IsNaN(v))
            {
                sum += v;
                count++;
            }

        return count == 0 ? double.NaN : sum / count;
    }

    // np.nanstd (population std, ddof=0), NaN-ignoring.
    private static double NanStd(double[] values)
    {
        double mean = NanMean(values);
        if (double.IsNaN(mean))
            return double.NaN;

        double sumSquares = 0.0;
        int count = 0;
        foreach (var v in values)
            if (!double.IsNaN(v))
            {
                double delta = v - mean;
                sumSquares += delta * delta;
                count++;
            }

        return count == 0 ? double.NaN : Math.Sqrt(sumSquares / count);
    }
}
