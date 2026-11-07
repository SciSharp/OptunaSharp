using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.ML;
using Microsoft.ML.Data;
using OptunaSharp.Distributions;
using OptunaSharp.Integration;

namespace OptunaSharp.Tests.Integration.MLNet;

/// <summary>
///     <see cref="OptunaSearchCV" /> passes its <c>nJobs</c> to <c>Optimize</c>, so <c>nJobs</c> trials call one
///     <see cref="MLNetEstimator" /> at the same time. They share its <see cref="MLContext" />, its fold views and the
///     caller's pipeline factory (plan 10, item 2.6; THREADING.md §10).
/// </summary>
/// <remarks>
///     The estimator records any exception raised while a fold fits or scores as that fold's error score (NaN by
///     default), as scikit-learn's <c>error_score</c> does. A thread-safety failure therefore would not raise: it would
///     show as a NaN or a different fold score. Each fold score of the parallel run is compared with a sequential
///     evaluation of the same parameters, through a separate context, estimator and fold split.
/// </remarks>
[TestClass]
public class MLNetEstimatorNJobsTests
{
    private const int NumFeatures = 3;

    /// <summary>One regression example.</summary>
    private sealed class RegRow
    {
        /// <summary>The feature vector.</summary>
        [VectorType(NumFeatures)]
        public float[] Features = default!;

        /// <summary>The regression target.</summary>
        public float Label;
    }

    /// <summary>A learnable linear target with light noise, identical on every call.</summary>
    /// <param name="n">The number of rows.</param>
    /// <returns>The rows.</returns>
    private static List<RegRow> GenerateData(int n)
    {
        var random = new Random(0);
        var rows = new List<RegRow>(n);
        for (int i = 0; i < n; i++)
        {
            var f = new float[NumFeatures];
            for (int j = 0; j < NumFeatures; j++)
                f[j] = (float)random.NextDouble();
            float label = 2.0f * f[0] - 1.0f * f[1] + 0.5f * f[2] + (float)((random.NextDouble() - 0.5) * 0.1);
            rows.Add(new RegRow { Features = f, Label = label });
        }

        return rows;
    }

    /// <summary>The searched space: FastTree's leaf count and learning rate.</summary>
    private static readonly Dictionary<string, BaseDistribution> Space = new()
    {
        ["num_leaves"] = new IntDistribution(4, 32),
        ["learning_rate"] = new FloatDistribution(0.05, 0.5, log: true),
    };

    /// <summary>An estimator over its own context, data view and three folds, seeded so every instance splits alike.</summary>
    /// <param name="rows">The data.</param>
    /// <returns>The estimator.</returns>
    private static MLNetEstimator NewEstimator(List<RegRow> rows)
    {
        var ml = new MLContext(seed: 0);
        IDataView data = ml.Data.LoadFromEnumerable(rows);

        IEstimator<ITransformer> PipelineFactory(IReadOnlyDictionary<string, object?> parameters)
            => ml.Regression.Trainers.FastTree(
                labelColumnName: "Label",
                featureColumnName: "Features",
                numberOfLeaves: (int)(long)parameters["num_leaves"]!,
                numberOfTrees: 20,
                learningRate: (double)parameters["learning_rate"]!);

        static double Scorer(MLContext context, ITransformer model, IDataView view)
            => context.Regression.Evaluate(model.Transform(view), labelColumnName: "Label").RSquared;

        return new MLNetEstimator(ml, data, PipelineFactory, Scorer, folds: 3, seed: 0);
    }

    /// <summary>
    ///     At <c>nJobs: 4</c> every trial completes, no fold score is NaN, and every fold score equals the one a
    ///     sequential evaluation of the same parameters gives.
    /// </summary>
    [TestMethod]
    public void OptunaSearchCV_NJobs4_ScoresEveryFoldAsASequentialEvaluation()
    {
        var rows = GenerateData(150);
        var search = new OptunaSearchCV(NewEstimator(rows), Space, nTrials: 16, nJobs: 4, randomState: 0, refit: false);

        search.Fit();

        var trials = search.Trials;
        trials.Should().HaveCount(16).And.OnlyContain(t => t.State == TrialState.Complete);

        // The comparison below means something only if trials really ran at the same time: their summed durations
        // must clearly exceed the wall-clock span of the run (four workers would make it up to four times as long).
        double busySeconds = trials.Sum(t => (t.DatetimeComplete!.Value - t.DatetimeStart!.Value).TotalSeconds);
        double spanSeconds = (trials.Max(t => t.DatetimeComplete!.Value) - trials.Min(t => t.DatetimeStart!.Value)).TotalSeconds;
        busySeconds.Should().BeGreaterThan(1.5 * spanSeconds, "the trials must have overlapped");

        var reference = NewEstimator(rows);
        foreach (var trial in trials)
        {
            var parameters = trial.Params.ToDictionary(p => p.Key, p => p.Value);
            var expected = reference.CrossValidate(parameters, returnTrainScore: false);
            for (int fold = 0; fold < expected.Count; fold++)
            {
                double actual = (double)trial.UserAttrs[$"split{fold}_test_score"]!;
                double.IsNaN(actual).Should().BeFalse($"trial {trial.Number}, fold {fold} must not have failed");
                actual.Should().Be(expected[fold].TestScore, $"trial {trial.Number}, fold {fold}");
            }
        }
    }
}
