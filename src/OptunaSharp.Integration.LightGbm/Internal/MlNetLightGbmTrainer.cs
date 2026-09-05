using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using NumSharp;

namespace OptunaSharp.Integration.LightGbm;

/// <summary>
///     The default <see cref="ILightGbmTrainer" /> — the <b>adapter</b> half of the LightGBM Tuner. It
///     maps the tuner's native-LightGBM parameter dict onto ML.NET's LightGBM trainer options, trains a
///     <b>regression</b>, <b>binary-classification</b>, <b>multiclass-classification</b>, or <b>ranking</b>
///     booster (chosen from the LightGBM <c>objective</c> parameter), and evaluates the requested metric on
///     the validation set(s). This is the only non-faithful part of the port: ML.NET's LightGBM is not
///     Python's, so there is no bit-parity here (Phase 6 is an adapter phase — the stepwise <i>search</i> is
///     the faithful part). The <c>ndcg</c>/<c>map</c> ranking metrics and the multiclass log-loss are
///     computed here (ML.NET's evaluators do not expose exactly the <c>{metric}@k</c> keys the tuner reads),
///     so their values are standard textbook definitions rather than LightGBM's byte-for-byte.
///
///     <para>Determinism: the LightGBM <c>seed</c> parameter is forwarded to <c>Options.Seed</c> and the
///     trainer runs single-threaded (unless overridden), so a fixed seed gives a reproducible fit.</para>
///
///     <para>Also implements <see cref="ILightGbmBoosterStore" /> — the persistence behind the tuner's
///     <c>model_dir</c> feature: it saves each booster with <c>MLContext.Model.Save</c> and a small JSON
///     sidecar (the validation scores + boosting rounds), and reloads it for a resumed / distributed
///     <c>GetBestBooster</c>.</para>
///
///     <para><b>Degenerate-stump handling:</b> ML.NET's tree-parameter builder throws on <i>single-leaf</i>
///     trees, which LightGBM legitimately produces when a boosting round finds no useful split (strong
///     <c>lambda_l1</c>/<c>lambda_l2</c>, low signal, or a large <c>min_child_samples</c> — the tuner
///     sweeps all of these). Such a stump is a valid constant predictor, so this trainer catches that crash
///     and scores the fit as a constant predictor (see <c>TryFit</c>), keeping the tuner running exactly as
///     it would with real LightGBM; the resulting <see cref="Booster.Model" /> is <c>null</c> for that
///     trial. This is most common for ranking on small data.</para>
/// </summary>
public sealed class MlNetLightGbmTrainer : ILightGbmTrainer, ILightGbmBoosterStore
{
    private const string LabelCol = "Label";
    private const string FeatureCol = "Features";
    private const string GroupCol = "GroupId";

    // On-disk layout for `model_dir` persistence (Python pickles one `{n}.pkl`; the ML.NET model is a zip
    // and the scores/rounds ride alongside in a JSON sidecar — a documented format divergence).
    private const string ModelExt = ".model.zip";
    private const string MetaExt = ".meta.json";
    private const string CvMetaExt = ".cv.json";

    /// <summary>The four task types the default backend maps LightGBM <c>objective</c> values onto.</summary>
    private enum LgbmTask
    {
        Regression,
        Binary,
        Multiclass,
        Ranking,
    }

    private readonly struct FitResult
    {
        public FitResult(ITransformer? model, LgbmTask task, int numClass)
        {
            Model = model;
            Task = task;
            NumClass = numClass;
        }

        /// <summary>The trained model, or <c>null</c> for a degenerate stump ML.NET could not represent.</summary>
        public ITransformer? Model { get; }
        public LgbmTask Task { get; }
        public int NumClass { get; }   // multiclass only
    }

    /// <inheritdoc />
    public Booster Train(TrainRequest request)
    {
        var fit = FitModel(request.Params, request.TrainSet, request.NumBoostRound);

        var bestScore = new Dictionary<string, IReadOnlyDictionary<string, double>>();
        for (int i = 0; i < request.ValidSets.Count; i++)
        {
            string name = i < request.ValidNames.Count ? request.ValidNames[i] : $"valid_{i}";
            bestScore[name] = EvaluateMetrics(fit, request.ValidSets[i], request.Params);
        }

        return new Booster(bestScore, request.NumBoostRound, fit.Model);
    }

    /// <inheritdoc />
    public CvResult Cv(CvRequest request)
    {
        var foldSets = BuildFolds(request.TrainSet, request.Params, request.NFold, request.Shuffle, request.Seed);

        // Accumulate each metric across folds, then average (the analog of `cv_results["{m}-mean"]`).
        var perMetricSums = new Dictionary<string, double>();
        var foldModels = new List<object?>();

        foreach (var (foldTrain, foldVal) in foldSets)
        {
            var fit = FitModel(request.Params, foldTrain, request.NumBoostRound);

            var metrics = EvaluateMetrics(fit, foldVal, request.Params);
            foreach (var (metric, value) in metrics)
                perMetricSums[metric] = perMetricSums.GetValueOrDefault(metric) + value;

            if (request.ReturnCvBooster)
                foldModels.Add(fit.Model);
        }

        int nFold = foldSets.Count;
        var results = new Dictionary<string, IReadOnlyList<double>>();
        foreach (var (metric, sum) in perMetricSums)
            results[$"{metric}-mean"] = new List<double> { nFold > 0 ? sum / nFold : 0.0 };

        var cvBooster = request.ReturnCvBooster ? new CvBooster(foldModels, request.NumBoostRound) : null;
        return new CvResult(results, cvBooster);
    }

    // ---- Task resolution ----

    private static LgbmTask ResolveTask(IReadOnlyDictionary<string, object?> @params, Dataset trainSet, out int numClass)
    {
        numClass = 0;
        string objective = (@params.GetValueOrDefault("objective") as string ?? "").ToLowerInvariant();

        if (objective is "binary")
            return LgbmTask.Binary;

        // The LightGBM multiclass family: softmax (`multiclass`) and one-vs-all (`multiclassova`) + aliases.
        if (objective is "multiclass" or "softmax" or "multiclassova" or "multiclass_ova" or "ova" or "ovr")
        {
            numClass = ResolveNumClass(@params, trainSet);
            return LgbmTask.Multiclass;
        }

        // The LightGBM ranking family.
        if (objective is "lambdarank" or "rank_xendcg" or "rank_xendcg_mart" or "xendcg" or "xendcg_mart")
            return LgbmTask.Ranking;

        // Regression family + unknown objective -> plain L2 regression (LightGBM's default task). ML.NET's
        // regression LightGBM only trains L2, so the poisson/gamma/tweedie/quantile variants are approximated.
        return LgbmTask.Regression;
    }

    private static int ResolveNumClass(IReadOnlyDictionary<string, object?> @params, Dataset trainSet)
    {
        double[] y = trainSet.GetLabelOrZeros();
        int inferred = 1;
        foreach (var v in y)
        {
            int c = (int)Math.Round(v);
            if (c + 1 > inferred) inferred = c + 1;
        }

        // `num_class` (alias `num_classes`) is the LightGBM-required class count; honor it but never let it
        // fall below the labels actually present (an under-count silently drops classes as ML.NET key overflow).
        if (TryInt(@params, "num_class", out var nc) && nc > 0)
            return Math.Max(nc, inferred);
        if (TryInt(@params, "num_classes", out var nc2) && nc2 > 0)
            return Math.Max(nc2, inferred);
        return inferred;
    }

    // ---- Fitting ----

    private static FitResult FitModel(
        IReadOnlyDictionary<string, object?> @params, Dataset trainSet, int numBoostRound)
    {
        var task = ResolveTask(@params, trainSet, out int numClass);
        var ml = new MLContext();

        switch (task)
        {
            case LgbmTask.Binary:
                return new FitResult(TryFit(() => ml.BinaryClassification.Trainers
                    .LightGbm(BuildBinaryOptions(@params, numBoostRound))
                    .Fit(LoadBinary(ml, trainSet))), task, 0);
            case LgbmTask.Multiclass:
                return new FitResult(TryFit(() => ml.MulticlassClassification.Trainers
                    .LightGbm(BuildMulticlassOptions(@params, numBoostRound))
                    .Fit(LoadMulticlass(ml, trainSet, numClass))), task, numClass);
            case LgbmTask.Ranking:
                return new FitResult(TryFit(() => ml.Ranking.Trainers
                    .LightGbm(BuildRankingOptions(@params, numBoostRound))
                    .Fit(LoadRanking(ml, trainSet))), task, 0);
            default:
                return new FitResult(TryFit(() => ml.Regression.Trainers
                    .LightGbm(BuildRegressionOptions(@params, numBoostRound))
                    .Fit(LoadRegression(ml, trainSet))), task, 0);
        }
    }

    /// <summary>
    ///     Fit, tolerating ML.NET's degenerate-tree crash. LightGBM legitimately produces a single-leaf
    ///     stump when a round finds no useful split (strong <c>lambda_l1</c>/<c>lambda_l2</c>, low signal,
    ///     a large <c>min_child_samples</c>) — a valid constant predictor that ML.NET's tree-parameter
    ///     builder cannot represent (it throws <see cref="ArgumentNullException" /> on the null split
    ///     arrays). Returning <c>null</c> here makes the caller fall back to constant predictions for the
    ///     fit, so the tuner keeps going exactly as it would with real LightGBM's constant-predicting stump.
    /// </summary>
    private static ITransformer? TryFit(Func<ITransformer> fit)
    {
        try
        {
            return fit();
        }
        catch (ArgumentNullException)
        {
            return null;
        }
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

    // The LightGBM-core numeric knobs shared by every ML.NET trainer Options (they all derive from the same
    // generic OptionsBase, which cannot be named without its type arguments — so the values are parsed once
    // here and set on each concrete Options below).
    private readonly record struct CoreParams(double? Lr, int? Leaves, int? MinChild, int? MaxBin, int Threads, int? Seed);

    private static CoreParams ReadCore(IReadOnlyDictionary<string, object?> @params) => new(
        TryDouble(@params, "learning_rate", out var lr) ? lr : null,
        TryInt(@params, "num_leaves", out var nl) ? nl : null,
        TryInt(@params, "min_child_samples", out var mcs) ? mcs : null,
        TryInt(@params, "max_bin", out var mb) ? mb : null,
        TryInt(@params, "num_threads", out var nt) ? nt : 1,
        TryInt(@params, "seed", out var seed) ? seed : null);

    private static LightGbmRegressionTrainer.Options BuildRegressionOptions(
        IReadOnlyDictionary<string, object?> @params, int numBoostRound)
    {
        var o = new LightGbmRegressionTrainer.Options
        {
            LabelColumnName = LabelCol,
            FeatureColumnName = FeatureCol,
            NumberOfIterations = numBoostRound,
            Booster = BuildBooster(@params),
            Silent = true,
        };
        var c = ReadCore(@params);
        if (c.Lr is { } lr) o.LearningRate = lr;
        if (c.Leaves is { } nl) o.NumberOfLeaves = nl;
        if (c.MinChild is { } mcs) o.MinimumExampleCountPerLeaf = mcs;
        if (c.MaxBin is { } mb) o.MaximumBinCountPerFeature = mb;
        o.NumberOfThreads = c.Threads;
        o.Seed = c.Seed;
        return o;
    }

    private static LightGbmBinaryTrainer.Options BuildBinaryOptions(
        IReadOnlyDictionary<string, object?> @params, int numBoostRound)
    {
        var o = new LightGbmBinaryTrainer.Options
        {
            LabelColumnName = LabelCol,
            FeatureColumnName = FeatureCol,
            NumberOfIterations = numBoostRound,
            Booster = BuildBooster(@params),
            Silent = true,
        };
        var c = ReadCore(@params);
        if (c.Lr is { } lr) o.LearningRate = lr;
        if (c.Leaves is { } nl) o.NumberOfLeaves = nl;
        if (c.MinChild is { } mcs) o.MinimumExampleCountPerLeaf = mcs;
        if (c.MaxBin is { } mb) o.MaximumBinCountPerFeature = mb;
        o.NumberOfThreads = c.Threads;
        o.Seed = c.Seed;
        return o;
    }

    private static LightGbmMulticlassTrainer.Options BuildMulticlassOptions(
        IReadOnlyDictionary<string, object?> @params, int numBoostRound)
    {
        var o = new LightGbmMulticlassTrainer.Options
        {
            LabelColumnName = LabelCol,
            FeatureColumnName = FeatureCol,
            NumberOfIterations = numBoostRound,
            Booster = BuildBooster(@params),
            Silent = true,
        };
        // One-vs-all when the objective is the `multiclassova` family, else softmax.
        string objective = (@params.GetValueOrDefault("objective") as string ?? "").ToLowerInvariant();
        o.UseSoftmax = objective is not ("multiclassova" or "multiclass_ova" or "ova" or "ovr");
        var c = ReadCore(@params);
        if (c.Lr is { } lr) o.LearningRate = lr;
        if (c.Leaves is { } nl) o.NumberOfLeaves = nl;
        if (c.MinChild is { } mcs) o.MinimumExampleCountPerLeaf = mcs;
        if (c.MaxBin is { } mb) o.MaximumBinCountPerFeature = mb;
        o.NumberOfThreads = c.Threads;
        o.Seed = c.Seed;
        return o;
    }

    private static LightGbmRankingTrainer.Options BuildRankingOptions(
        IReadOnlyDictionary<string, object?> @params, int numBoostRound)
    {
        var o = new LightGbmRankingTrainer.Options
        {
            LabelColumnName = LabelCol,
            FeatureColumnName = FeatureCol,
            RowGroupColumnName = GroupCol,
            NumberOfIterations = numBoostRound,
            Booster = BuildBooster(@params),
            Silent = true,
        };
        var c = ReadCore(@params);
        if (c.Lr is { } lr) o.LearningRate = lr;
        if (c.Leaves is { } nl) o.NumberOfLeaves = nl;
        if (c.MinChild is { } mcs) o.MinimumExampleCountPerLeaf = mcs;
        if (c.MaxBin is { } mb) o.MaximumBinCountPerFeature = mb;
        o.NumberOfThreads = c.Threads;
        o.Seed = c.Seed;
        return o;
    }

    // ---- Metric evaluation ----

    private static IReadOnlyDictionary<string, double> EvaluateMetrics(
        FitResult fit, Dataset validSet, IReadOnlyDictionary<string, object?> @params)
    {
        double[] y = validSet.GetLabelOrZeros();

        // A degenerate stump ML.NET could not build (see TryFit): score it as a constant predictor.
        if (fit.Model is null)
            return DegenerateMetrics(fit, validSet, y, @params);

        var ml = new MLContext();

        switch (fit.Task)
        {
            case LgbmTask.Binary:
            {
                var scored = fit.Model.Transform(LoadBinary(ml, validSet));
                var probs = ml.Data.CreateEnumerable<BinOut>(scored, reuseRowObject: false)
                    .Select(r => (double)r.Probability).ToArray();
                return BinaryMetrics(y, probs);
            }
            case LgbmTask.Multiclass:
            {
                var scored = fit.Model.Transform(LoadMulticlass(ml, validSet, fit.NumClass));
                var probs = ml.Data.CreateEnumerable<MultiOut>(scored, reuseRowObject: false)
                    .Select(r => r.Score).ToArray();
                return MulticlassMetrics(y, probs);
            }
            case LgbmTask.Ranking:
            {
                var scored = fit.Model.Transform(LoadRanking(ml, validSet));
                var scores = ml.Data.CreateEnumerable<RankOut>(scored, reuseRowObject: false)
                    .Select(r => (double)r.Score).ToArray();
                return RankingMetrics(y, scores, validSet.Group!, @params);
            }
            default:
            {
                var scored = fit.Model.Transform(LoadRegression(ml, validSet));
                var preds = ml.Data.CreateEnumerable<RegOut>(scored, reuseRowObject: false)
                    .Select(r => (double)r.Score).ToArray();
                return RegressionMetrics(y, preds);
            }
        }
    }

    // The constant-predictor scores for a degenerate stump (Model == null): zero for regression / ranking,
    // 0.5 for binary, and a uniform 1/K for multiclass — a valid, poor score so the trial completes.
    private static IReadOnlyDictionary<string, double> DegenerateMetrics(
        FitResult fit, Dataset validSet, double[] y, IReadOnlyDictionary<string, object?> @params)
    {
        switch (fit.Task)
        {
            case LgbmTask.Ranking:
                return RankingMetrics(y, new double[y.Length], validSet.Group!, @params);
            case LgbmTask.Multiclass:
            {
                int k = Math.Max(1, fit.NumClass);
                var uniform = new float[y.Length][];
                for (int i = 0; i < y.Length; i++)
                {
                    uniform[i] = new float[k];
                    Array.Fill(uniform[i], 1f / k);
                }
                return MulticlassMetrics(y, uniform);
            }
            case LgbmTask.Binary:
            {
                var half = new double[y.Length];
                Array.Fill(half, 0.5);
                return BinaryMetrics(y, half);
            }
            default:
                return RegressionMetrics(y, new double[y.Length]);
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

    /// <summary>Multiclass log-loss + error from per-class probabilities (<c>Score[c] = P(class c)</c>, 0-based).</summary>
    internal static IReadOnlyDictionary<string, double> MulticlassMetrics(double[] y, float[][] probs)
    {
        int n = y.Length;
        const double eps = 1e-15;
        double logloss = 0;
        int errors = 0;
        for (int i = 0; i < n; i++)
        {
            int trueCls = (int)Math.Round(y[i]);
            float[] pr = probs[i];
            double pTrue = trueCls >= 0 && trueCls < pr.Length ? pr[trueCls] : 0.0;
            logloss += -Math.Log(Math.Min(1 - eps, Math.Max(eps, pTrue)));

            int arg = 0;
            for (int c = 1; c < pr.Length; c++)
                if (pr[c] > pr[arg]) arg = c;
            if (arg != trueCls) errors++;
        }
        return new Dictionary<string, double>
        {
            ["multi_logloss"] = n > 0 ? logloss / n : 0.0,
            ["multi_error"] = n > 0 ? (double)errors / n : 0.0,
        };
    }

    /// <summary>
    ///     ndcg@k / map@k over the validation set's query groups — standard textbook definitions (gain
    ///     <c>2^rel - 1</c>, discount <c>log2(rank + 1)</c>; MAP over binary relevance <c>rel &gt; 0</c>),
    ///     averaged over groups. Adapter metrics: not LightGBM's byte-for-byte, but higher-is-better and
    ///     monotone, which is all the stepwise search needs.
    /// </summary>
    internal static IReadOnlyDictionary<string, double> RankingMetrics(
        double[] rel, double[] score, int[] group, IReadOnlyDictionary<string, object?> @params)
    {
        var ndcgKs = ResolveEvalAt(@params, "ndcg");
        var mapKs = ResolveEvalAt(@params, "map");
        var ndcgSums = ndcgKs.ToDictionary(k => k, _ => 0.0);
        var mapSums = mapKs.ToDictionary(k => k, _ => 0.0);

        int numGroups = group.Length;
        int offset = 0;
        foreach (int gsize in group)
        {
            var idx = Enumerable.Range(offset, gsize).ToArray();
            var order = idx.OrderByDescending(i => score[i]).ToArray();   // predicted ranking
            var ideal = idx.OrderByDescending(i => rel[i]).ToArray();     // ideal ranking

            foreach (int k in ndcgKs) ndcgSums[k] += NdcgAtK(order, ideal, rel, k);
            foreach (int k in mapKs) mapSums[k] += MapAtK(order, rel, k);
            offset += gsize;
        }

        var result = new Dictionary<string, double>();
        foreach (int k in ndcgKs) result[$"ndcg@{k}"] = numGroups > 0 ? ndcgSums[k] / numGroups : 0.0;
        foreach (int k in mapKs) result[$"map@{k}"] = numGroups > 0 ? mapSums[k] / numGroups : 0.0;
        return result;
    }

    private static double NdcgAtK(int[] order, int[] ideal, double[] rel, int k)
    {
        double dcg = Dcg(order, rel, k);
        double idcg = Dcg(ideal, rel, k);
        return idcg > 0 ? dcg / idcg : 1.0;   // an all-zero-relevance query is trivially "perfectly" ranked
    }

    private static double Dcg(int[] order, double[] rel, int k)
    {
        int limit = Math.Min(k, order.Length);
        double sum = 0;
        for (int r = 0; r < limit; r++)
            sum += (Math.Pow(2.0, rel[order[r]]) - 1.0) / Math.Log(r + 2, 2);   // discount = log2((r+1)+1)
        return sum;
    }

    private static double MapAtK(int[] order, double[] rel, int k)
    {
        int limit = Math.Min(k, order.Length);
        int totalRelevant = order.Count(i => rel[i] > 0.0);
        if (totalRelevant == 0) return 1.0;

        int hits = 0;
        double sumPrec = 0;
        for (int r = 0; r < limit; r++)
            if (rel[order[r]] > 0.0)
            {
                hits++;
                sumPrec += (double)hits / (r + 1);   // precision at rank r+1
            }
        return sumPrec / Math.Min(totalRelevant, k);
    }

    /// <summary>Port of the <c>eval_at</c> resolution in <c>_metric_with_eval_at</c>, returning the full list.</summary>
    private static List<int> ResolveEvalAt(IReadOnlyDictionary<string, object?> @params, string metric)
    {
        object? evalAt = @params.GetValueOrDefault("eval_at")
            ?? @params.GetValueOrDefault($"{metric}_at")
            ?? @params.GetValueOrDefault($"{metric}_eval_at");

        if (evalAt is null || evalAt is string)
            return new List<int> { 1, 2, 3, 4, 5 };   // LightGBM's documented default

        if (evalAt is IEnumerable seq)
        {
            var list = new List<int>();
            foreach (var item in seq)
                list.Add(Convert.ToInt32(item, CultureInfo.InvariantCulture));
            return list.Count > 0 ? list : new List<int> { 1, 2, 3, 4, 5 };
        }

        return new List<int> { Convert.ToInt32(evalAt, CultureInfo.InvariantCulture) };
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

    // ---- Data loading ----

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

    private static IDataView LoadMulticlass(MLContext ml, Dataset set, int numClass)
    {
        int n = set.NumRows, d = set.NumFeatures;
        double[] label = set.GetLabelOrZeros();
        var rows = new MultiInputRow[n];
        for (int i = 0; i < n; i++)
            rows[i] = new MultiInputRow { Features = set.Data[i], Label = (uint)Math.Max(0, (int)Math.Round(label[i])) };

        var schema = SchemaDefinition.Create(typeof(MultiInputRow));
        schema[nameof(MultiInputRow.Features)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, d);
        // 0-based class id as a Key column (ML.NET's multiclass label contract; Score[c] is then P(class c)).
        schema[nameof(MultiInputRow.Label)].ColumnType = new KeyDataViewType(typeof(uint), (ulong)Math.Max(1, numClass));
        return ml.Data.LoadFromEnumerable(rows, schema);
    }

    private static IDataView LoadRanking(MLContext ml, Dataset set)
    {
        int n = set.NumRows, d = set.NumFeatures;
        double[] label = set.GetLabelOrZeros();
        uint[] groupIds = set.RowGroupIds();
        var rows = new RankInputRow[n];
        for (int i = 0; i < n; i++)
            rows[i] = new RankInputRow { Features = set.Data[i], Label = (float)label[i], GroupId = groupIds[i] };

        var schema = SchemaDefinition.Create(typeof(RankInputRow));
        schema[nameof(RankInputRow.Features)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, d);
        schema[nameof(RankInputRow.GroupId)].ColumnType = new KeyDataViewType(typeof(uint), (ulong)Math.Max(1, set.NumGroups));
        return ml.Data.LoadFromEnumerable(rows, schema);
    }

    // ---- Folding ----

    private static List<(Dataset Train, Dataset Val)> BuildFolds(
        Dataset set, IReadOnlyDictionary<string, object?> @params, int nFold, bool shuffle, int seed)
    {
        var task = ResolveTask(@params, set, out _);
        if (task == LgbmTask.Ranking)
            return MakeGroupFolds(set, nFold, shuffle, seed);

        return MakeRowFolds(set.NumRows, nFold, shuffle, seed)
            .Select(f => (Subset(set, f.TrainIdx), Subset(set, f.ValIdx)))
            .ToList();
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

    private static List<(int[] TrainIdx, int[] ValIdx)> MakeRowFolds(int n, int nFold, bool shuffle, int seed)
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

    // Ranking folds must never split a query group, so they partition whole GROUPS (not rows) into folds.
    private static List<(Dataset Train, Dataset Val)> MakeGroupFolds(Dataset set, int nFold, bool shuffle, int seed)
    {
        int g = set.NumGroups;
        if (g == 0)
            throw new ValueError("Ranking cross-validation requires the dataset's `group` sizes.");

        var order = Enumerable.Range(0, g).ToArray();
        if (shuffle)
        {
            var rng = new Random(seed);
            for (int i = g - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }
        }

        var folds = new List<(Dataset, Dataset)>();
        for (int f = 0; f < nFold; f++)
        {
            var val = new List<int>();
            var train = new List<int>();
            for (int i = 0; i < g; i++)
                (i % nFold == f ? val : train).Add(order[i]);
            val.Sort();
            train.Sort();
            folds.Add((set.SelectGroups(train), set.SelectGroups(val)));
        }
        return folds;
    }

    // ---- ILightGbmBoosterStore (model_dir persistence) ----

    /// <inheritdoc />
    public void SaveBooster(Booster booster, string path)
    {
        // A degenerate stump has no ML.NET model (Booster.Model == null); persist only its scores then.
        if (booster.Model is ITransformer model)
            new MLContext().Model.Save(model, inputSchema: null, path + ModelExt);
        WriteMeta(path + MetaExt, new BoosterMeta
        {
            CurrentIteration = booster.CurrentIteration,
            BestScore = booster.BestScore.ToDictionary(
                kv => kv.Key, kv => kv.Value.ToDictionary(m => m.Key, m => m.Value)),
        });
    }

    /// <inheritdoc />
    public Booster LoadBooster(string path)
    {
        ITransformer? model = File.Exists(path + ModelExt)
            ? new MLContext().Model.Load(path + ModelExt, out _)
            : null;   // a stump that was saved score-only
        var meta = ReadMeta<BoosterMeta>(path + MetaExt);
        var best = meta.BestScore.ToDictionary(
            kv => kv.Key, kv => (IReadOnlyDictionary<string, double>)kv.Value);
        return new Booster(best, meta.CurrentIteration, model);
    }

    /// <inheritdoc />
    public void SaveCvBooster(CvBooster booster, string path)
    {
        var ml = new MLContext();
        for (int i = 0; i < booster.Boosters.Count; i++)
            if (booster.Boosters[i] is ITransformer m)   // skip degenerate-stump folds (null model)
                ml.Model.Save(m, inputSchema: null, $"{path}.fold{i}{ModelExt}");
        WriteMeta(path + CvMetaExt, new CvMeta { BestIteration = booster.BestIteration, NumFolds = booster.Boosters.Count });
    }

    /// <inheritdoc />
    public CvBooster LoadCvBooster(string path)
    {
        var meta = ReadMeta<CvMeta>(path + CvMetaExt);
        var ml = new MLContext();
        var models = new List<object?>();
        for (int i = 0; i < meta.NumFolds; i++)
        {
            string foldPath = $"{path}.fold{i}{ModelExt}";
            models.Add(File.Exists(foldPath) ? ml.Model.Load(foldPath, out _) : null);
        }
        return new CvBooster(models, meta.BestIteration);
    }

    /// <inheritdoc />
    public bool BoosterExists(string path)
        => File.Exists(path + MetaExt) || File.Exists(path + CvMetaExt);

    private static void WriteMeta(string path, object meta)
        => File.WriteAllText(path, JsonSerializer.Serialize(meta));

    private static T ReadMeta<T>(string path)
        => JsonSerializer.Deserialize<T>(File.ReadAllText(path))
           ?? throw new ValueError($"The booster metadata at {path} could not be read.");

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

    private sealed class MultiInputRow
    {
        public float[] Features { get; set; } = Array.Empty<float>();
        public uint Label { get; set; }
    }

    private sealed class RankInputRow
    {
        public float[] Features { get; set; } = Array.Empty<float>();
        public float Label { get; set; }
        public uint GroupId { get; set; }
    }

    private sealed class RegOut
    {
        public float Score { get; set; }
    }

    private sealed class BinOut
    {
        public float Probability { get; set; }
    }

    private sealed class MultiOut
    {
        [VectorType]
        public float[] Score { get; set; } = Array.Empty<float>();
    }

    private sealed class RankOut
    {
        public float Score { get; set; }
    }

    private sealed class BoosterMeta
    {
        public int CurrentIteration { get; set; }
        public Dictionary<string, Dictionary<string, double>> BestScore { get; set; } = new();
    }

    private sealed class CvMeta
    {
        public int BestIteration { get; set; }
        public int NumFolds { get; set; }
    }
}
