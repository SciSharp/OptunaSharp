using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace OptunaSharp.Integration.LightGbm;

/// <summary>
///     The default <see cref="ILightGbmTrainer" /> — the <b>adapter</b> half of the LightGBM Tuner. It
///     maps the tuner's native-LightGBM parameter dict onto ML.NET's LightGBM trainer options, trains a
///     regression or binary-classification booster, and evaluates the requested metric on the validation
///     set(s). This is the only non-faithful part of the port: ML.NET's LightGBM is not Python's, so there
///     is no bit-parity here (Phase 6 is an adapter phase — the stepwise <i>search</i> is the faithful
///     part). The objective (regression vs binary) is chosen from the LightGBM <c>objective</c> parameter.
///
///     <para>Determinism: the LightGBM <c>seed</c> parameter is forwarded to <c>Options.Seed</c> and the
///     trainer runs single-threaded (unless overridden), so a fixed seed gives a reproducible fit.</para>
/// </summary>
public sealed class MlNetLightGbmTrainer : ILightGbmTrainer
{
    private const string LabelCol = "Label";
    private const string FeatureCol = "Features";

    /// <inheritdoc />
    public Booster Train(TrainRequest request)
    {
        var (model, isBinary) = FitModel(request.Params, request.TrainSet, request.NumBoostRound);

        var bestScore = new Dictionary<string, IReadOnlyDictionary<string, double>>();
        for (int i = 0; i < request.ValidSets.Count; i++)
        {
            string name = i < request.ValidNames.Count ? request.ValidNames[i] : $"valid_{i}";
            bestScore[name] = EvaluateMetrics(model, request.ValidSets[i], isBinary);
        }

        return new Booster(bestScore, request.NumBoostRound, model);
    }

    /// <inheritdoc />
    public CvResult Cv(CvRequest request)
    {
        var folds = MakeFolds(request.TrainSet.NumRows, request.NFold, request.Shuffle, request.Seed);

        // Accumulate each metric across folds, then average (the analog of `cv_results["{m}-mean"]`).
        var perMetricSums = new Dictionary<string, double>();
        var foldModels = new List<object?>();

        foreach (var (trainIdx, valIdx) in folds)
        {
            var foldTrain = Subset(request.TrainSet, trainIdx);
            var foldVal = Subset(request.TrainSet, valIdx);
            var (model, isBinary) = FitModel(request.Params, foldTrain, request.NumBoostRound);

            var metrics = EvaluateMetrics(model, foldVal, isBinary);
            foreach (var (metric, value) in metrics)
                perMetricSums[metric] = perMetricSums.GetValueOrDefault(metric) + value;

            if (request.ReturnCvBooster)
                foldModels.Add(model);
        }

        int nFold = folds.Count;
        var results = new Dictionary<string, IReadOnlyList<double>>();
        foreach (var (metric, sum) in perMetricSums)
            results[$"{metric}-mean"] = new List<double> { sum / nFold };

        var cvBooster = request.ReturnCvBooster ? new CvBooster(foldModels, request.NumBoostRound) : null;
        return new CvResult(results, cvBooster);
    }

    // ---- Fitting ----

    private static (ITransformer Model, bool IsBinary) FitModel(
        IReadOnlyDictionary<string, object?> @params, Dataset trainSet, int numBoostRound)
    {
        bool isBinary = IsBinaryObjective(@params);
        var ml = new MLContext();

        if (isBinary)
        {
            var options = BuildBinaryOptions(@params, numBoostRound);
            var data = LoadBinary(ml, trainSet);
            var model = ml.BinaryClassification.Trainers.LightGbm(options).Fit(data);
            return (model, true);
        }
        else
        {
            var options = BuildRegressionOptions(@params, numBoostRound);
            var data = LoadRegression(ml, trainSet);
            var model = ml.Regression.Trainers.LightGbm(options).Fit(data);
            return (model, false);
        }
    }

    private static bool IsBinaryObjective(IReadOnlyDictionary<string, object?> @params)
    {
        // `binary` (canonical) is the only binary objective the first cut maps; everything else -> regression.
        string objective = (@params.GetValueOrDefault("objective") as string ?? "").ToLowerInvariant();
        if (objective is "binary")
            return true;
        if (objective is "" or "regression" or "regression_l2" or "l2" or "mse" or "regression_l1"
            or "l1" or "mae" or "huber" or "fair" or "poisson" or "quantile" or "mape" or "gamma"
            or "tweedie" or "rmse")
            return false;
        if (objective.StartsWith("multiclass", StringComparison.Ordinal) || objective is "softmax")
            throw new NotImplementedException(
                $"The LightGBM objective '{objective}' is not supported by the ML.NET tuner backend yet "
                + "(the first cut supports regression and binary). Supply a custom ILightGbmTrainer.");
        // Unknown objective: default to regression (LightGBM's own default task).
        return false;
    }

    private static GradientBooster.Options BuildBooster(IReadOnlyDictionary<string, object?> @params)
    {
        var booster = new GradientBooster.Options();
        if (TryDouble(@params, "feature_fraction", out var ff)) booster.FeatureFraction = ff;
        if (TryDouble(@params, "bagging_fraction", out var bf)) booster.SubsampleFraction = bf;
        if (TryInt(@params, "bagging_freq", out var bfreq)) booster.SubsampleFrequency = bfreq;
        if (TryDouble(@params, "lambda_l1", out var l1)) booster.L1Regularization = l1;
        if (TryDouble(@params, "lambda_l2", out var l2)) booster.L2Regularization = l2;
        if (TryInt(@params, "max_depth", out var md)) booster.MaximumTreeDepth = md;
        if (TryDouble(@params, "min_gain_to_split", out var mg)) booster.MinimumSplitGain = mg;
        if (TryDouble(@params, "min_sum_hessian_in_leaf", out var mh)) booster.MinimumChildWeight = mh;
        return booster;
    }

    private static LightGbmRegressionTrainer.Options BuildRegressionOptions(
        IReadOnlyDictionary<string, object?> @params, int numBoostRound)
    {
        var options = new LightGbmRegressionTrainer.Options
        {
            LabelColumnName = LabelCol,
            FeatureColumnName = FeatureCol,
            NumberOfIterations = numBoostRound,
            Booster = BuildBooster(@params),
            Silent = true,
        };
        ApplyCommon(options, @params);
        return options;
    }

    private static LightGbmBinaryTrainer.Options BuildBinaryOptions(
        IReadOnlyDictionary<string, object?> @params, int numBoostRound)
    {
        var options = new LightGbmBinaryTrainer.Options
        {
            LabelColumnName = LabelCol,
            FeatureColumnName = FeatureCol,
            NumberOfIterations = numBoostRound,
            Booster = BuildBooster(@params),
            Silent = true,
        };
        // Set the shared numeric fields (duplicated because the two Options types share no settable base).
        if (TryDouble(@params, "learning_rate", out var lr)) options.LearningRate = lr;
        if (TryInt(@params, "num_leaves", out var nl)) options.NumberOfLeaves = nl;
        if (TryInt(@params, "min_child_samples", out var mcs)) options.MinimumExampleCountPerLeaf = mcs;
        if (TryInt(@params, "max_bin", out var mb)) options.MaximumBinCountPerFeature = mb;
        options.NumberOfThreads = TryInt(@params, "num_threads", out var nt) ? nt : 1;
        options.Seed = TryInt(@params, "seed", out var seed) ? seed : null;
        return options;
    }

    private static void ApplyCommon(LightGbmRegressionTrainer.Options options, IReadOnlyDictionary<string, object?> @params)
    {
        if (TryDouble(@params, "learning_rate", out var lr)) options.LearningRate = lr;
        if (TryInt(@params, "num_leaves", out var nl)) options.NumberOfLeaves = nl;
        if (TryInt(@params, "min_child_samples", out var mcs)) options.MinimumExampleCountPerLeaf = mcs;
        if (TryInt(@params, "max_bin", out var mb)) options.MaximumBinCountPerFeature = mb;
        options.NumberOfThreads = TryInt(@params, "num_threads", out var nt) ? nt : 1;
        options.Seed = TryInt(@params, "seed", out var seed) ? seed : null;
    }

    // ---- Metric evaluation ----

    private static IReadOnlyDictionary<string, double> EvaluateMetrics(
        ITransformer model, Dataset validSet, bool isBinary)
    {
        var ml = new MLContext();
        double[] y = validSet.GetLabelOrZeros();

        if (isBinary)
        {
            var scored = model.Transform(LoadBinary(ml, validSet));
            var probs = ml.Data.CreateEnumerable<BinOut>(scored, reuseRowObject: false)
                .Select(r => (double)r.Probability).ToArray();
            return BinaryMetrics(y, probs);
        }
        else
        {
            var scored = model.Transform(LoadRegression(ml, validSet));
            var preds = ml.Data.CreateEnumerable<RegOut>(scored, reuseRowObject: false)
                .Select(r => (double)r.Score).ToArray();
            return RegressionMetrics(y, preds);
        }
    }

    private static IReadOnlyDictionary<string, double> RegressionMetrics(double[] y, double[] pred)
    {
        int n = y.Length;
        double sqSum = 0, absSum = 0, apeSum = 0;
        int apeCount = 0;
        for (int i = 0; i < n; i++)
        {
            double d = pred[i] - y[i];
            sqSum += d * d;
            absSum += Math.Abs(d);
            if (y[i] != 0.0)
            {
                apeSum += Math.Abs(d / y[i]);
                apeCount++;
            }
        }
        double l2 = n > 0 ? sqSum / n : 0.0;
        return new Dictionary<string, double>
        {
            ["l2"] = l2,
            ["l1"] = n > 0 ? absSum / n : 0.0,
            ["rmse"] = Math.Sqrt(l2),
            ["mape"] = apeCount > 0 ? apeSum / apeCount : 0.0,
        };
    }

    private static IReadOnlyDictionary<string, double> BinaryMetrics(double[] y, double[] prob)
    {
        int n = y.Length;
        const double eps = 1e-15;
        double logloss = 0;
        int errors = 0;
        for (int i = 0; i < n; i++)
        {
            double p = Math.Min(1 - eps, Math.Max(eps, prob[i]));
            double yi = y[i] != 0.0 ? 1.0 : 0.0;
            logloss += -(yi * Math.Log(p) + (1 - yi) * Math.Log(1 - p));
            int predLabel = prob[i] >= 0.5 ? 1 : 0;
            if (predLabel != (int)yi) errors++;
        }
        return new Dictionary<string, double>
        {
            ["binary_logloss"] = n > 0 ? logloss / n : 0.0,
            ["binary_error"] = n > 0 ? (double)errors / n : 0.0,
            ["auc"] = Auc(y, prob),
        };
    }

    /// <summary>Rank-based AUC (Mann–Whitney U); ties get the average rank.</summary>
    private static double Auc(double[] y, double[] score)
    {
        int n = y.Length;
        var order = Enumerable.Range(0, n).OrderBy(i => score[i]).ToArray();
        var rank = new double[n];
        int idx = 0;
        while (idx < n)
        {
            int j = idx;
            while (j < n && score[order[j]] == score[order[idx]]) j++;
            double avgRank = (idx + j - 1) / 2.0 + 1.0; // ranks are 1-based
            for (int k = idx; k < j; k++) rank[order[k]] = avgRank;
            idx = j;
        }

        double sumRankPos = 0;
        int nPos = 0;
        for (int i = 0; i < n; i++)
        {
            if (y[i] != 0.0)
            {
                sumRankPos += rank[i];
                nPos++;
            }
        }
        int nNeg = n - nPos;
        if (nPos == 0 || nNeg == 0)
            return 0.0;
        return (sumRankPos - nPos * (nPos + 1) / 2.0) / ((double)nPos * nNeg);
    }

    // ---- Data loading + folding ----

    private static IDataView LoadRegression(MLContext ml, Dataset set)
    {
        int n = set.NumRows, d = set.NumFeatures;
        double[] label = set.GetLabelOrZeros();
        var rows = new RegInputRow[n];
        for (int i = 0; i < n; i++)
            rows[i] = new RegInputRow { Features = set.Data[i], Label = (float)label[i] };

        var schema = SchemaDefinition.Create(typeof(RegInputRow));
        schema[nameof(RegInputRow.Features)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, d);
        return ml.Data.LoadFromEnumerable(rows, schema);
    }

    private static IDataView LoadBinary(MLContext ml, Dataset set)
    {
        int n = set.NumRows, d = set.NumFeatures;
        double[] label = set.GetLabelOrZeros();
        var rows = new BinInputRow[n];
        for (int i = 0; i < n; i++)
            rows[i] = new BinInputRow { Features = set.Data[i], Label = label[i] != 0.0 };

        var schema = SchemaDefinition.Create(typeof(BinInputRow));
        schema[nameof(BinInputRow.Features)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, d);
        return ml.Data.LoadFromEnumerable(rows, schema);
    }

    private static Dataset Subset(Dataset set, IReadOnlyList<int> idx)
    {
        var data = new float[idx.Count][];
        var label = set.Label is null ? null : new double[idx.Count];
        var weight = set.Weight is null ? null : new double[idx.Count];
        for (int i = 0; i < idx.Count; i++)
        {
            data[i] = set.Data[idx[i]];
            if (label is not null) label[i] = set.Label![idx[i]];
            if (weight is not null) weight[i] = set.Weight![idx[i]];
        }
        return new Dataset(data, label, weight, set.FeatureName);
    }

    private static List<(int[] TrainIdx, int[] ValIdx)> MakeFolds(int n, int nFold, bool shuffle, int seed)
    {
        var order = Enumerable.Range(0, n).ToArray();
        if (shuffle)
        {
            var rng = new Random(seed);
            for (int i = n - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }
        }

        var folds = new List<(int[], int[])>();
        for (int f = 0; f < nFold; f++)
        {
            var val = new List<int>();
            var train = new List<int>();
            for (int i = 0; i < n; i++)
                (i % nFold == f ? val : train).Add(order[i]);
            folds.Add((train.ToArray(), val.ToArray()));
        }
        return folds;
    }

    // ---- Param access helpers ----

    private static bool TryDouble(IReadOnlyDictionary<string, object?> p, string key, out double value)
    {
        if (p.TryGetValue(key, out var v) && v is not null)
        {
            value = Convert.ToDouble(v, CultureInfo.InvariantCulture);
            return true;
        }
        value = 0;
        return false;
    }

    private static bool TryInt(IReadOnlyDictionary<string, object?> p, string key, out int value)
    {
        if (p.TryGetValue(key, out var v) && v is not null)
        {
            value = Convert.ToInt32(v, CultureInfo.InvariantCulture);
            return true;
        }
        value = 0;
        return false;
    }

    private sealed class RegInputRow
    {
        public float[] Features { get; set; } = Array.Empty<float>();
        public float Label { get; set; }
    }

    private sealed class BinInputRow
    {
        public float[] Features { get; set; } = Array.Empty<float>();
        public bool Label { get; set; }
    }

    private sealed class RegOut
    {
        public float Score { get; set; }
    }

    private sealed class BinOut
    {
        public float Probability { get; set; }
    }
}
