using System;
using System.Collections.Generic;
using AwesomeAssertions;
using Microsoft.ML;
using Microsoft.ML.Data;
using OptunaSharp.Distributions;
using OptunaSharp.Integration;

namespace OptunaSharp.Tests.Integration.MLNet;

/// <summary>
///     End-to-end test of the default <see cref="MLNetEstimator" /> backend for <see cref="OptunaSearchCV" />:
///     a small FastTree regression hyperparameter search on synthetic data. This is the adapter half of
///     Phase 6 — validated by running the real ML.NET stack rather than by bit-parity against Python.
/// </summary>
[TestClass]
public class MLNetEstimatorEndToEndTests
{
    private const int NumFeatures = 3;

    private sealed class RegRow
    {
        [VectorType(NumFeatures)]
        public float[] Features = default!;

        public float Label;
    }

    private static List<RegRow> GenerateData(int n)
    {
        var random = new Random(0);
        var rows = new List<RegRow>(n);
        for (int i = 0; i < n; i++)
        {
            var f = new float[NumFeatures];
            for (int j = 0; j < NumFeatures; j++)
                f[j] = (float)random.NextDouble();

            // A smooth linear target with light noise — learnable, so tuning moves R^2.
            float label = 2.0f * f[0] - 1.0f * f[1] + 0.5f * f[2] + (float)((random.NextDouble() - 0.5) * 0.1);
            rows.Add(new RegRow { Features = f, Label = label });
        }

        return rows;
    }

    [TestMethod]
    public void EndToEnd_FastTreeRegressionSearch_FitsAndPredicts()
    {
        var ml = new MLContext(seed: 0);
        IDataView data = ml.Data.LoadFromEnumerable(GenerateData(150));

        IEstimator<ITransformer> PipelineFactory(IReadOnlyDictionary<string, object?> parameters)
        {
            int numberOfLeaves = (int)(long)parameters["num_leaves"]!;
            double learningRate = (double)parameters["learning_rate"]!;
            return ml.Regression.Trainers.FastTree(
                labelColumnName: "Label",
                featureColumnName: "Features",
                numberOfLeaves: numberOfLeaves,
                numberOfTrees: 20,
                learningRate: learningRate);
        }

        double Scorer(MLContext context, ITransformer model, IDataView view)
            => context.Regression.Evaluate(model.Transform(view), labelColumnName: "Label").RSquared;

        var estimator = new MLNetEstimator(ml, data, PipelineFactory, Scorer, folds: 3, seed: 0);

        var space = new Dictionary<string, BaseDistribution>
        {
            ["num_leaves"] = new IntDistribution(4, 32),
            ["learning_rate"] = new FloatDistribution(0.05, 0.5, log: true),
        };

        var search = new OptunaSearchCV(estimator, space, nTrials: 5, randomState: 0);

        search.Fit();

        search.NTrials.Should().Be(5);
        search.NSplits.Should().Be(3);
        double.IsNaN(search.BestScore).Should().BeFalse();
        search.BestParams.Should().ContainKeys("num_leaves", "learning_rate");

        // The best estimator was refit and can predict / score.
        estimator.BestModel.Should().NotBeNull();
        estimator.Predict(data).Should().NotBeNull();
        double.IsNaN(estimator.Score(data)).Should().BeFalse();
    }
}
