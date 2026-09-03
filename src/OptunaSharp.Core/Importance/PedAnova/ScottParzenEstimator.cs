using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Tpe;

namespace OptunaSharp.Importance.PedAnova;

/// <summary>
///     1-D Parzen estimator using Scott's-rule bandwidth selection. Port of
///     <c>optuna/importance/_ped_anova/scott_parzen_estimator.py: ScottParzenEstimator</c>.
/// </summary>
internal sealed class ScottParzenEstimator
{
    private readonly MixtureOfProductDistribution _mixtureDistribution;

    internal ScottParzenEstimator(
        double[] observations, BaseDistribution searchSpace, double[] predeterminedWeights, double priorWeight)
    {
        if (priorWeight < 0)
            throw new ValueError($"prior_weight={PythonRepr.FloatRepr(priorWeight)} must be non-negative.");
        System.Diagnostics.Debug.Assert(observations.Length == predeterminedWeights.Length);

        double[] weights;
        if (observations.Length == 0)
        {
            weights = new[] { 1.0 };
        }
        else
        {
            weights = new double[predeterminedWeights.Length + 1];
            Array.Copy(predeterminedWeights, weights, predeterminedWeights.Length);
            weights[^1] = priorWeight;
        }
        double weightsTotal = MathUtil.PairwiseSum(weights);
        for (int i = 0; i < weights.Length; i++)
            weights[i] /= weightsTotal;

        BatchedDistributions dist = searchSpace is CategoricalDistribution cat
            ? CalculateCategoricalDistributions(observations, cat, priorWeight)
            : CalculateNumericalDistributions(observations, searchSpace, predeterminedWeights);

        _mixtureDistribution = new MixtureOfProductDistribution(weights, new[] { dist });
    }

    /// <summary>Exposed for the mirrored unit tests (Optuna asserts on <c>_mixture_distribution</c>).</summary>
    internal MixtureOfProductDistribution MixtureDistribution => _mixtureDistribution;

    private static BatchedDistributions CalculateCategoricalDistributions(
        double[] observations, CategoricalDistribution searchSpace, double priorWeight)
    {
        int nChoices = searchSpace.Choices.Count;
        if (observations.Length == 0)
        {
            var uniform = new double[1, nChoices];
            for (int c = 0; c < nChoices; c++)
                uniform[0, c] = 1.0 / nChoices;
            return new BatchedCategoricalDistributions(uniform);
        }

        int nKernels = observations.Length + 1;  // +1 for prior.
        var weights = new double[nKernels, nChoices];
        for (int k = 0; k < nKernels; k++)
        {
            for (int c = 0; c < nChoices; c++)
                weights[k, c] = priorWeight / nKernels;
        }
        for (int j = 0; j < observations.Length; j++)
            weights[j, (int)observations[j]] += 1;

        var row = new double[nChoices];
        for (int k = 0; k < nKernels; k++)
        {
            for (int c = 0; c < nChoices; c++)
                row[c] = weights[k, c];
            double rowSum = MathUtil.PairwiseSum(row);
            double denom = rowSum == 0 ? 1 : rowSum;
            for (int c = 0; c < nChoices; c++)
                weights[k, c] /= denom;
        }
        return new BatchedCategoricalDistributions(weights);
    }

    private static BatchedDistributions CalculateNumericalDistributions(
        double[] observations, BaseDistribution searchSpace, double[] weights)
    {
        // search_space is always the rounded IntDistribution(0, grid_size - 1): step == 1, log == false.
        var intDist = (IntDistribution)searchSpace;
        System.Diagnostics.Debug.Assert(intDist.Step == 1);
        double low = intDist.Low;
        double high = intDist.High;
        bool log = intDist.Log;

        var weightsCum = new double[weights.Length];
        double running = 0.0;
        for (int i = 0; i < weights.Length; i++)
        {
            running += weights[i];  // np.cumsum: sequential prefix sum.
            weightsCum[i] = running;
        }
        double weightsSum = weightsCum[^1];

        var obs = observations;
        if (log)
        {
            obs = observations.Select(x => Math.Log(x)).ToArray();
            low = Math.Log(low);
            high = Math.Log(high);
        }

        double meanEst = SeqDot(obs, weights) / weightsSum;
        var dev = new double[obs.Length];
        for (int i = 0; i < obs.Length; i++)
            dev[i] = (obs[i] - meanEst) * (obs[i] - meanEst);
        double sigmaEst = Math.Sqrt(SeqDot(dev, weights) / Math.Max(1, weightsSum - 1));

        int q1Idx = BisectLeft(weightsCum, Math.Floor(weightsSum / 4.0));
        int q3Idx = BisectRight(weightsCum, Math.Floor(weightsSum * 3.0 / 4.0));
        double iqr = obs[Math.Min(obs.Length - 1, q3Idx)] - obs[q1Idx];
        sigmaEst = 1.059 * Math.Min(iqr / 1.34, sigmaEst) * Math.Pow(weightsSum, -0.2);
        // To avoid numerical errors. 0.5/1.64 means 1.64sigma (=90%) will fit in the target grid.
        double sigmaMin = 0.5 / 1.64;

        var mu = new double[obs.Length + 1];
        Array.Copy(obs, mu, obs.Length);
        mu[^1] = (low + high) / 2.0;

        double sigmaValue = Math.Max(sigmaEst, sigmaMin);
        var sigma = new double[obs.Length + 1];
        for (int i = 0; i < obs.Length; i++)
            sigma[i] = sigmaValue;
        sigma[^1] = high - low + 1;

        return new BatchedDiscreteTruncNormDistributions(mu, sigma, low, high, step: 1);
    }

    /// <summary>Python <c>ScottParzenEstimator.pdf(samples)</c>.</summary>
    internal double[] Pdf(double[] samples)
    {
        var x = new double[samples.Length, 1];
        for (int s = 0; s < samples.Length; s++)
            x[s, 0] = samples[s];
        var logPdf = _mixtureDistribution.LogPdf(x);
        var result = new double[samples.Length];
        for (int s = 0; s < samples.Length; s++)
            result[s] = Math.Exp(logPdf[s]);
        return result;
    }

    /// <summary>Sequential (left-to-right) dot product — bit-exact vs NumPy's <c>@</c> for PED-ANOVA.</summary>
    private static double SeqDot(double[] a, double[] b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }

    /// <summary>NumPy <c>searchsorted(a, v, side="left")</c> for an ascending array (bisect left).</summary>
    private static int BisectLeft(double[] a, double v)
    {
        int lo = 0, hi = a.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (a[mid] < v)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

    /// <summary>NumPy <c>searchsorted(a, v, side="right")</c> for an ascending array (bisect right).</summary>
    private static int BisectRight(double[] a, double v)
    {
        int lo = 0, hi = a.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (v < a[mid])
                hi = mid;
            else
                lo = mid + 1;
        }
        return lo;
    }
}

/// <summary>
///     Port of the module-level grid-counting / builder helpers in
///     <c>optuna/importance/_ped_anova/scott_parzen_estimator.py</c>.
/// </summary>
internal static class ScottParzenEstimatorFunctions
{
    /// <summary>Python <c>_count_numerical_param_in_grid(param_name, dist, trials, n_steps)</c>.</summary>
    internal static long[] CountNumericalParamInGrid(
        string paramName, BaseDistribution dist, IReadOnlyList<FrozenTrial> trials, int nSteps)
    {
        double low, high;
        bool log;
        if (dist is IntDistribution intDist)
        {
            low = intDist.Low;
            high = intDist.High;
            log = intDist.Log;
            if (log)
            {
                int log2DomainSize = (int)Math.Ceiling(Math.Log(intDist.High - intDist.Low + 1) / Math.Log(2)) + 1;
                nSteps = Math.Min(log2DomainSize, nSteps);
            }
            else
            {
                // dist.step is not None for IntDistribution (always an integer step).
                nSteps = Math.Min((int)Math.Round((double)(intDist.High - intDist.Low) / intDist.Step,
                    MidpointRounding.ToEven) + 1, nSteps);
            }
        }
        else
        {
            var floatDist = (FloatDistribution)dist;
            low = floatDist.Low;
            high = floatDist.High;
            log = floatDist.Log;
            if (floatDist.Step is not null)
            {
                System.Diagnostics.Debug.Assert(!log);
                nSteps = Math.Min((int)Math.Round((floatDist.High - floatDist.Low) / floatDist.Step.Value,
                    MidpointRounding.ToEven) + 1, nSteps);
            }
        }

        if (log)
        {
            low = Math.Log(low);
            high = Math.Log(high);
        }

        var paramValues = new double[trials.Count];
        for (int t = 0; t < trials.Count; t++)
        {
            PyFloat.TryConvert(trials[t].ParamsDict[paramName], out double v);
            paramValues[t] = log ? Math.Log(v) : v;
        }

        double stepSize = (high - low) / (nSteps - 1);
        var counts = new long[nSteps];
        foreach (double pv in paramValues)
        {
            // For backward compatibility, midpoint ties go to the lower grid.
            int index = (int)Math.Ceiling((pv - low) / stepSize - 0.5);
            index = Math.Clamp(index, 0, nSteps - 1);
            counts[index] += 1;
        }
        return counts;
    }

    /// <summary>Python <c>_count_categorical_param_in_grid(param_name, dist, trials)</c>.</summary>
    internal static long[] CountCategoricalParamInGrid(
        string paramName, CategoricalDistribution dist, IReadOnlyList<FrozenTrial> trials)
    {
        var counts = new long[dist.Choices.Count];
        foreach (var trial in trials)
        {
            int index = (int)dist.ToInternalRepr(trial.ParamsDict[paramName]);
            counts[index] += 1;
        }
        return counts;
    }

    /// <summary>Python <c>build_parzen_estimator_on_grid(param_name, dist, trials, n_steps, prior_weight)</c>.</summary>
    internal static (ScottParzenEstimator Estimator, int GridSize) BuildParzenEstimatorOnGrid(
        string paramName, BaseDistribution dist, IReadOnlyList<FrozenTrial> trials, int nSteps, double priorWeight)
    {
        long[] counts;
        BaseDistribution roundedDist;
        if (dist is IntDistribution or FloatDistribution)
        {
            counts = CountNumericalParamInGrid(paramName, dist, trials, nSteps);
            roundedDist = new IntDistribution(low: 0, high: counts.Length - 1);
        }
        else if (dist is CategoricalDistribution catDist)
        {
            counts = CountCategoricalParamInGrid(paramName, catDist, trials);
            roundedDist = dist;
        }
        else
        {
            throw new ArgumentException($"Got an unknown dist with the type {dist.GetType()}.");
        }

        // np.flatnonzero(counts): the grid cells with a nonzero count, and their counts as weights.
        var observations = new List<double>();
        var weights = new List<double>();
        for (int i = 0; i < counts.Length; i++)
        {
            if (counts[i] != 0)
            {
                observations.Add(i);
                weights.Add(counts[i]);
            }
        }

        var pe = new ScottParzenEstimator(
            observations: observations.ToArray(),
            searchSpace: roundedDist,
            predeterminedWeights: weights.ToArray(),
            priorWeight: priorWeight);
        return (pe, counts.Length);
    }
}
