using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Gp;

/// <summary>
///     The GP-internal normalized search space — port of <c>optuna/_gp/search_space.py</c>'s
///     <c>SearchSpace</c>. It maps each Optuna distribution onto a normalized <c>[0, 1]</c> box the
///     Gaussian process works in: continuous parameters are affine- (or log-affine-) scaled,
///     discrete ones round to their grid, and categoricals carry their integer code. Every transform
///     is pure numpy in Optuna and reproduced bit-for-bit here (no <c>erf</c>, no LAPACK — <c>log</c>/
///     <c>exp</c>/<c>floor</c> match Python bit-for-bit).
///
///     <para><see cref="SampleNormalizedParams" /> draws the preliminary points from <b>scrambled</b>
///     Sobol' (<see cref="ScrambledSobol" />, a bit-exact port of <c>scipy.stats.qmc.Sobol(scramble=True)</c>)
///     — the acqf optimizer (Phase 3.2 part 3) depends on it.</para>
/// </summary>
internal sealed class GpSearchSpace
{
    internal enum ScaleType { Linear = 0, Log = 1, Categorical = 2 }

    private readonly string[] _names;
    private readonly BaseDistribution[] _distributions;
    private readonly long[] _scaleTypes;
    private readonly double[,] _bounds;  // (dim, 2)
    private readonly double[] _steps;

    internal int Dim { get; }
    internal bool[] IsCategorical { get; }
    internal int[] DiscreteIndices { get; }
    internal int[] ContinuousIndices { get; }

    internal IReadOnlyList<long> ScaleTypes => _scaleTypes;
    internal double[,] Bounds => _bounds;
    internal IReadOnlyList<double> Steps => _steps;

    internal GpSearchSpace(IReadOnlyDictionary<string, BaseDistribution> optunaSearchSpace)
    {
        _names = optunaSearchSpace.Keys.ToArray();
        _distributions = _names.Select(n => optunaSearchSpace[n]).ToArray();
        Dim = _names.Length;
        _scaleTypes = new long[Dim];
        _bounds = new double[Dim, 2];
        _steps = new double[Dim];

        for (int i = 0; i < Dim; i++)
        {
            BaseDistribution distribution = _distributions[i];
            switch (distribution)
            {
                case CategoricalDistribution cat:
                    _scaleTypes[i] = (long)ScaleType.Categorical;
                    _bounds[i, 0] = 0.0;
                    _bounds[i, 1] = cat.Choices.Count;
                    _steps[i] = 1.0;
                    break;
                case FloatDistribution f:
                    _scaleTypes[i] = f.Log ? (long)ScaleType.Log : (long)ScaleType.Linear;
                    _bounds[i, 0] = f.Low;
                    _bounds[i, 1] = f.High;
                    _steps[i] = f.Step ?? 0.0;
                    break;
                case IntDistribution n:
                    _scaleTypes[i] = n.Log ? (long)ScaleType.Log : (long)ScaleType.Linear;
                    _bounds[i, 0] = n.Low;
                    _bounds[i, 1] = n.High;
                    _steps[i] = n.Step;  // IntDistribution.step is always >= 1 (discrete)
                    break;
                default:
                    throw new ArgumentException($"Unsupported distribution type {distribution.GetType().Name}.");
            }
        }

        IsCategorical = _scaleTypes.Select(s => s == (long)ScaleType.Categorical).ToArray();
        DiscreteIndices = Enumerable.Range(0, Dim).Where(i => _steps[i] > 0).ToArray();
        ContinuousIndices = Enumerable.Range(0, Dim).Where(i => _steps[i] == 0.0).ToArray();
    }

    /// <summary>The normalized training matrix <c>(len(trials), dim)</c> — <c>get_normalized_params</c>.</summary>
    internal double[,] GetNormalizedParams(IReadOnlyList<IReadOnlyDictionary<string, object?>> trialParams)
    {
        int n = trialParams.Count;
        var values = new double[n, Dim];
        for (int i = 0; i < Dim; i++)
        {
            string name = _names[i];
            if (_distributions[i] is CategoricalDistribution cat)
            {
                for (int t = 0; t < n; t++)
                    values[t, i] = cat.ToInternalRepr(trialParams[t][name]);
            }
            else
            {
                var scaleType = (ScaleType)_scaleTypes[i];
                for (int t = 0; t < n; t++)
                    values[t, i] = NormalizeOneParam(
                        Convert.ToDouble(trialParams[t][name]), scaleType,
                        (_bounds[i, 0], _bounds[i, 1]), _steps[i]);
            }
        }
        return values;
    }

    /// <summary>Convert one normalized parameter vector back to external values — <c>get_unnormalized_param</c>.</summary>
    internal Dictionary<string, object?> GetUnnormalizedParam(double[] normalizedParam)
    {
        var ret = new Dictionary<string, object?>();
        for (int i = 0; i < Dim; i++)
        {
            BaseDistribution distribution = _distributions[i];
            if (distribution is CategoricalDistribution cat)
            {
                ret[_names[i]] = cat.ToExternalRepr(normalizedParam[i]);
                continue;
            }

            bool isInt = distribution is IntDistribution;
            var scaleType = (ScaleType)_scaleTypes[i];
            double low = _bounds[i, 0], high = _bounds[i, 1];
            double step = distribution switch
            {
                FloatDistribution f => f.Step ?? 0.0,
                IntDistribution n => n.Step,
                _ => 0.0
            };
            double value = MathUtilClip(
                UnnormalizeOneParam(normalizedParam[i], scaleType, (low, high), step), low, high);
            ret[_names[i]] = isInt ? (object)(long)Math.Round(value, MidpointRounding.ToEven) : value;
        }
        return ret;
    }

    /// <summary>The normalized grid of choices for each discrete dimension — <c>get_choices_of_discrete_params</c>.</summary>
    internal List<double[]> GetChoicesOfDiscreteParams()
    {
        var result = new List<double[]>();
        foreach (int i in DiscreteIndices)
        {
            if (IsCategorical[i])
            {
                result.Add(Arange(0.0, _bounds[i, 1], 1.0));
            }
            else
            {
                double[] grid = Arange(_bounds[i, 0], _bounds[i, 1] + 0.5 * _steps[i], _steps[i]);
                for (int k = 0; k < grid.Length; k++)
                    grid[k] = NormalizeOneParam(grid[k], (ScaleType)_scaleTypes[i], (_bounds[i, 0], _bounds[i, 1]), _steps[i]);
                result.Add(grid);
            }
        }
        return result;
    }

    /// <summary>
    ///     <c>_sample_normalized_params</c> — <paramref name="n" /> preliminary points for the acqf
    ///     optimizer. A Sobol' seed is drawn from <paramref name="rng" /> (the sampler's MT19937), the
    ///     scrambled Sobol' box-fill runs through <see cref="ScrambledSobol" />, then each categorical
    ///     dimension floors to a code and each discrete dimension rounds to its grid.
    /// </summary>
    internal double[,] SampleNormalizedParams(int n, NumPyRandom rng)
    {
        int seed = (int)rng.randint(0, 2147483647);  // rng.randint(np.iinfo(np.int32).max)
        double[,] values = ScrambledSobol.Random(Dim, n, seed);
        for (int i = 0; i < Dim; i++)
        {
            if (IsCategorical[i])
            {
                for (int t = 0; t < n; t++)
                    values[t, i] = Math.Floor(values[t, i] * _bounds[i, 1]);
            }
            else if (_steps[i] != 0.0)
            {
                var scaleType = (ScaleType)_scaleTypes[i];
                for (int t = 0; t < n; t++)
                    values[t, i] = RoundOneNormalizedParam(values[t, i], scaleType, (_bounds[i, 0], _bounds[i, 1]), _steps[i]);
            }
        }
        return values;
    }

    // ---- one-parameter transforms (pure numpy, bit-exact) -------------------------------------

    internal static double NormalizeOneParam(double value, ScaleType scaleType, (double Low, double High) bounds, double step)
    {
        if (scaleType == ScaleType.Categorical)
            return value;
        double low = bounds.Low - 0.5 * step;
        double high = bounds.High + 0.5 * step;
        if (scaleType == ScaleType.Log)
        {
            low = Math.Log(low);
            high = Math.Log(high);
            value = Math.Log(value);
        }
        if (high == low)
            return 0.5;
        return (value - low) / (high - low);
    }

    internal static double UnnormalizeOneParam(double value, ScaleType scaleType, (double Low, double High) bounds, double step)
    {
        if (scaleType == ScaleType.Categorical)
            return value;
        double low = bounds.Low - 0.5 * step;
        double high = bounds.High + 0.5 * step;
        if (scaleType == ScaleType.Log)
        {
            low = Math.Log(low);
            high = Math.Log(high);
        }
        value = value * (high - low) + low;
        if (scaleType == ScaleType.Log)
            value = Math.Exp(value);
        return value;
    }

    internal static double RoundOneNormalizedParam(double value, ScaleType scaleType, (double Low, double High) bounds, double step)
    {
        if (step == 0.0)
            return value;
        double unnormalized = UnnormalizeOneParam(value, scaleType, bounds, step);
        double rounded = MathUtilClip(
            Math.Floor((unnormalized - bounds.Low + 0.5 * step) / step) * step + bounds.Low,
            bounds.Low, bounds.High);
        return NormalizeOneParam(rounded, scaleType, bounds, step);
    }

    /// <summary>NumPy <c>np.arange(start, stop, step)</c> — length <c>ceil((stop - start) / step)</c>, value <c>start + i·step</c>.</summary>
    private static double[] Arange(double start, double stop, double step)
    {
        int count = (int)Math.Ceiling((stop - start) / step);
        if (count < 0)
            count = 0;
        var a = new double[count];
        for (int i = 0; i < count; i++)
            a[i] = start + i * step;
        return a;
    }

    private static double MathUtilClip(double value, double low, double high)
        => Math.Min(Math.Max(value, low), high);
}
