using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.ML;
using NumSharp;

namespace OptunaSharp.Integration;

/// <summary>
///     The default <see cref="ICrossValidationEstimator" /> for <see cref="OptunaSearchCV" />, backed by
///     ML.NET. It retains an <see cref="IDataView" />, splits it into cross-validation folds, and for each
///     trial builds a pipeline from the sampled parameters, fits it on each fold's training split, and
///     scores it on the validation split.
///
///     <para>
///     This is the ML.NET analogue of scikit-learn's <c>(estimator, cv, scorer)</c> triple: you supply a
///     <b>pipeline factory</b> (build an ML.NET <see cref="IEstimator{TTransformer}" /> from the sampled
///     hyperparameters — the equivalent of <c>clone(estimator).set_params(**params)</c>) and a
///     <b>scorer</b> (evaluate a fitted model on a data view, <b>higher is better</b> — the equivalent of
///     scikit-learn's scorer). Folds are produced by <c>MLContext.Data.CrossValidationSplit</c>; pass a
///     <c>samplingKeyColumnName</c> to keep rows with the same key together in one fold.
///     </para>
///
///     <para>
///     After <see cref="OptunaSearchCV.Fit" /> with <c>refit = true</c>, the best model is fitted on the
///     full data and exposed via <see cref="BestModel" /> / <see cref="Predict" /> / <see cref="Transform" />
///     / <see cref="Score" />.
///     </para>
/// </summary>
public sealed class MLNetEstimator : ICrossValidationEstimator
{
    private readonly MLContext _ml;
    private readonly IDataView _data;
    private readonly Func<IReadOnlyDictionary<string, object?>, IEstimator<ITransformer>> _pipelineFactory;
    private readonly Func<MLContext, ITransformer, IDataView, double> _scorer;
    private readonly double _errorScore;
    private readonly List<(IDataView Train, IDataView Test)> _folds;

    /// <param name="mlContext">The ML.NET context used to split, fit, and evaluate.</param>
    /// <param name="data">The training data view searched over.</param>
    /// <param name="pipelineFactory">
    ///     Builds an ML.NET estimator (pipeline) from the hyperparameters sampled for a trial — the analogue
    ///     of <c>clone(estimator).set_params(**params)</c>.
    /// </param>
    /// <param name="scorer">
    ///     Evaluates a fitted model on a data view, returning a score where <b>higher is better</b> (for a
    ///     loss, negate it) — e.g. <c>(ml, model, view) =&gt; ml.Regression.Evaluate(model.Transform(view)).RSquared</c>.
    /// </param>
    /// <param name="folds">Number of cross-validation folds. Must be at least 2. Defaults to <c>5</c>.</param>
    /// <param name="samplingKeyColumnName">
    ///     Optional grouping column: rows with the same key stay in the same fold to avoid leakage
    ///     between related examples. <c>null</c> permits independent row assignment.
    /// </param>
    /// <param name="seed">Optional seed for the fold split, for reproducible folds.</param>
    /// <param name="errorScore">
    ///     The score assigned to a fold whose fit or scoring throws. Defaults to <c>NaN</c> (the fold is
    ///     ignored by the NaN-aware mean), matching scikit-learn's default <c>error_score=np.nan</c>.
    /// </param>
    /// <summary>Creates reusable cross-validation splits and retains the pipeline factory and scorer used by each trial.</summary>
    /// <exception cref="System.ArgumentNullException"><paramref name="mlContext" /> or <paramref name="data" /> or <paramref name="pipelineFactory" /> or <paramref name="scorer" /> is null.</exception>
    /// <exception cref="NumSharp.ValueError">The requested fold count is less than two.</exception>
    public MLNetEstimator(
        MLContext mlContext,
        IDataView data,
        Func<IReadOnlyDictionary<string, object?>, IEstimator<ITransformer>> pipelineFactory,
        Func<MLContext, ITransformer, IDataView, double> scorer,
        int folds = 5,
        string? samplingKeyColumnName = null,
        int? seed = null,
        double errorScore = double.NaN)
    {
        _ml = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _data = data ?? throw new ArgumentNullException(nameof(data));
        _pipelineFactory = pipelineFactory ?? throw new ArgumentNullException(nameof(pipelineFactory));
        _scorer = scorer ?? throw new ArgumentNullException(nameof(scorer));
        _errorScore = errorScore;

        if (folds < 2)
            throw new ValueError($"folds must be at least 2, got {folds}.");

        _folds = new List<(IDataView, IDataView)>(folds);
        foreach (var split in _ml.Data.CrossValidationSplit(
                     _data, numberOfFolds: folds, samplingKeyColumnName: samplingKeyColumnName, seed: seed))
            _folds.Add((split.TrainSet, split.TestSet));
    }

    /// <summary>The best model fitted on the full data after a refit, or <c>null</c> before one.</summary>
    public ITransformer? BestModel { get; private set; }

    /// <inheritdoc />
    public int GetNumSplits() => _folds.Count;

    /// <inheritdoc />
    /// <exception cref="System.Exception">The pipeline factory throws before fitting begins; fit and scorer failures inside a fold instead use the configured error score.</exception>
    public IReadOnlyList<CrossValidationFoldResult> CrossValidate(
        IReadOnlyDictionary<string, object?> parameters, bool returnTrainScore)
    {
        var results = new List<CrossValidationFoldResult>(_folds.Count);

        foreach (var (train, test) in _folds)
        {
            // Factory failures identify an invalid pipeline configuration before training starts;
            // only failures during fitting or scoring receive the configured per-fold error score.
            var pipeline = _pipelineFactory(parameters);
            var stopwatch = Stopwatch.StartNew();

            double testScore;
            double? trainScore = null;
            double fitTime;
            double scoreTime;

            try
            {
                var model = pipeline.Fit(train);
                fitTime = stopwatch.Elapsed.TotalSeconds;

                double afterFit = stopwatch.Elapsed.TotalSeconds;
                testScore = _scorer(_ml, model, test);
                scoreTime = stopwatch.Elapsed.TotalSeconds - afterFit;

                if (returnTrainScore)
                    trainScore = _scorer(_ml, model, train);
            }
            catch
            {
                // scikit-learn's error_score: assign the fold the configured score instead of failing.
                fitTime = stopwatch.Elapsed.TotalSeconds;
                testScore = _errorScore;
                scoreTime = 0.0;
                if (returnTrainScore)
                    trainScore = _errorScore;
            }

            results.Add(new CrossValidationFoldResult(testScore, trainScore, fitTime, scoreTime));
        }

        return results;
    }

    /// <inheritdoc />
    /// <exception cref="System.Exception">The pipeline factory or full-data fit throws; refit failures propagate unchanged.</exception>
    public void Refit(IReadOnlyDictionary<string, object?> parameters)
    {
        BestModel = _pipelineFactory(parameters).Fit(_data);
    }

    /// <summary>
    ///     Transform <paramref name="data" /> with the refit best model (its scored output view). Available
    ///     only after a refit.
    /// </summary>
    /// <param name="data">The feature rows or ML.NET data view used by the operation.</param>
    /// <returns>The best model's transformed data view, evaluated lazily by ML.NET.</returns>
    /// <exception cref="System.InvalidOperationException">The estimator has not been refitted with its best parameter configuration.</exception>
    public IDataView Predict(IDataView data)
    {
        if (BestModel is null)
            throw new InvalidOperationException(
                "The estimator has not been refit. Call OptunaSearchCV.Fit with refit = true first.");
        return BestModel.Transform(data);
    }

    /// <summary>Alias of <see cref="Predict" /> — transform <paramref name="data" /> with the refit best model.</summary>
    /// <param name="data">The feature rows or ML.NET data view used by the operation.</param>
    /// <returns>The best model's transformed data view, evaluated lazily by ML.NET.</returns>
    /// <exception cref="System.InvalidOperationException">The estimator has not been refitted with its best parameter configuration.</exception>
    public IDataView Transform(IDataView data) => Predict(data);

    /// <summary>Score <paramref name="data" /> with the refit best model, using the configured scorer (higher is better).</summary>
    /// <param name="data">The feature rows or ML.NET data view used by the operation.</param>
    /// <returns>The configured scorer's value, where a higher value is preferred.</returns>
    /// <exception cref="System.InvalidOperationException">The estimator has not been refitted with its best parameter configuration.</exception>
    /// <exception cref="System.Exception">The supplied scorer throws; its exception propagates unchanged.</exception>
    public double Score(IDataView data)
    {
        if (BestModel is null)
            throw new InvalidOperationException(
                "The estimator has not been refit. Call OptunaSearchCV.Fit with refit = true first.");
        return _scorer(_ml, BestModel, data);
    }
}
