// Beyond-core copy of the GP-internal normalized search space (OptunaSharp.Gp.GpSearchSpace), duplicated
// into the model-based toolkit namespace so TuRBO / REMBO share the same [0,1]-box normalization,
// categorical coding, and discrete-grid handling WITHOUT depending on the Optuna-parity GP stack (the
// project rule: copy rather than modify a shared preexisting component). Verbatim duplicate — keep
// diff-identical to the parity SearchSpace.cs; changes meant for GPSampler belong in OptunaSharp.Gp.

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using NumSharp.Utilities;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.ModelBased.Gp;

/// <summary>
///     The GP-internal normalized search space — port of <c>optuna/_gp/search_space.py</c>'s
///     <c>SearchSpace</c>. It maps each Optuna distribution onto a normalized <c>[0, 1]</c> box the
///     Gaussian process works in: continuous parameters are affine- (or log-affine-) scaled,
///     discrete ones round to their grid, and categoricals carry their integer code. Every transform
///     is pure numpy in Optuna and reproduced bit-for-bit here (no <c>erf</c>, no LAPACK — <c>log</c>/
///     <c>exp</c>/<c>floor</c> match Python bit-for-bit; the discrete grids are NumSharp's <c>np.arange</c>
///     and the rounding's <c>//</c> is NumSharp's port of NumPy's <c>floor_divide</c>, since 2026-11-11).
///
///     <para><see cref="SampleNormalizedParams" /> draws the preliminary points from <b>scrambled</b>
///     Sobol' (<see cref="ScrambledSobol" />, a bit-exact port of <c>scipy.stats.qmc.Sobol(scramble=True)</c>)
///     — the acqf optimizer (Phase 3.2 part 3) depends on it.</para>
/// </summary>
internal sealed class GpSearchSpace
{
    /// <summary>The transformation applied to an external parameter before GP evaluation.</summary>
    internal enum ScaleType
    {
        /// <summary>Uses affine scaling in the original numeric coordinate.</summary>
        Linear = 0,

        /// <summary>Uses affine scaling after taking the natural logarithm.</summary>
        Log = 1,

        /// <summary>Represents categories by their distribution-choice indices.</summary>
        Categorical = 2
    }

    private readonly string[] _names;
    private readonly BaseDistribution[] _distributions;
    private readonly long[] _scaleTypes;
    private readonly double[,] _bounds;  // (dim, 2)
    private readonly double[] _steps;

    /// <summary>The number of parameters in the normalized coordinate layout.</summary>
    internal int Dim { get; }
    /// <summary>A flag per coordinate selecting categorical equality in the kernel.</summary>
    internal bool[] IsCategorical { get; }
    /// <summary>Coordinate indices whose support is a finite grid or categorical choice set.</summary>
    internal int[] DiscreteIndices { get; }
    /// <summary>Coordinate indices that permit unconstrained movement inside the normalized interval.</summary>
    internal int[] ContinuousIndices { get; }

    /// <summary>The linear, log, or categorical scale code of each coordinate.</summary>
    internal IReadOnlyList<long> ScaleTypes => _scaleTypes;
    /// <summary>The external lower/upper bounds in coordinate order; categorical bounds describe their index support.</summary>
    internal double[,] Bounds => _bounds;
    /// <summary>The external grid spacing of each coordinate; zero indicates continuity.</summary>
    internal IReadOnlyList<double> Steps => _steps;

    /// <summary>Builds a stable normalized-coordinate layout from the ordered parameter distributions.</summary>
    /// <param name="optunaSearchSpace">The parameter distributions; enumeration order fixes the GP coordinate order.</param>
    /// <exception cref="System.ArgumentException">A parameter uses an unsupported distribution type.</exception>
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
    /// <param name="trialParams">External parameter dictionaries containing every modeled parameter, one dictionary per input row.</param>
    /// <returns>The trial-by-coordinate matrix in the GP search space's normalized representation.</returns>
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
    /// <param name="normalizedParam">One normalized coordinate vector in search-space order.</param>
    /// <returns>External parameter values keyed by name after inverse scaling and grid rounding.</returns>
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
    /// <returns>The admissible normalized grid for each discrete coordinate, in discrete-index order.</returns>
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
    /// <param name="n">The number of elements or points to process.</param>
    /// <param name="rng">The random generator whose state is advanced when drawing samples.</param>
    /// <returns>A matrix of sampled normalized points rounded onto each coordinate's support.</returns>
    internal double[,] SampleNormalizedParams(int n, NumPyRandom rng)
    {
        int seed = (int)NumPyDraws.RandInt(rng, 0, 2147483647);  // rng.randint(np.iinfo(np.int32).max)
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

    /// <summary>Returns the external value mapped into its normalized GP coordinate.</summary>
    /// <param name="value">The external numerical value or categorical index to map into GP coordinates.</param>
    /// <param name="scaleType">Whether the parameter uses linear, logarithmic, or categorical coordinates.</param>
    /// <param name="bounds">The coordinate bounds used by this numerical operation.</param>
    /// <param name="step">The distribution's grid spacing; zero denotes a continuous coordinate.</param>
    /// <returns>The external value mapped into its normalized GP coordinate.</returns>
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

    /// <summary>Returns the external numerical value corresponding to the normalized coordinate.</summary>
    /// <param name="value">The normalized numerical coordinate or categorical index to map back to its external coordinate.</param>
    /// <param name="scaleType">Whether the parameter uses linear, logarithmic, or categorical coordinates.</param>
    /// <param name="bounds">The coordinate bounds used by this numerical operation.</param>
    /// <param name="step">The distribution's grid spacing; zero denotes a continuous coordinate.</param>
    /// <returns>The external numerical value corresponding to the normalized coordinate.</returns>
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

    /// <summary>Returns the nearest admissible discrete coordinate in normalized space.</summary>
    /// <param name="value">The normalized numerical coordinate to project onto the parameter's supported grid.</param>
    /// <param name="scaleType">Whether the parameter uses linear, logarithmic, or categorical coordinates.</param>
    /// <param name="bounds">The coordinate bounds used by this numerical operation.</param>
    /// <param name="step">The distribution's grid spacing; zero denotes a continuous coordinate.</param>
    /// <returns>The nearest admissible discrete coordinate in normalized space.</returns>
    /// <remarks>
    ///     Optuna's <c>(param_value - low + 0.5 * step) // step</c> is NumPy's <c>floor_divide</c>, which floors the
    ///     exact quotient (<c>npy_divmod</c>: <c>fmod</c>, then <c>(a - mod) / b</c> snapped to an integer), not the
    ///     rounded one: <c>0.9 // 0.05</c> is <c>17</c> where <c>Math.Floor(0.9 / 0.05)</c> is <c>18</c>, because the
    ///     double <c>0.05</c> is a little above 1/20 and the quotient rounds up to 18. NumSharp's
    ///     <see cref="NDDivision.FloorDivDouble(double, double)" /> is that routine, allocation-free. The former
    ///     <c>Math.Floor(a / b)</c> picked the next grid point in about 1 of 800 roundings of random and dyadic
    ///     (Sobol'-like) points (measured 2026-11-11 over 508,000 against NumPy 2.4.2).
    /// </remarks>
    internal static double RoundOneNormalizedParam(double value, ScaleType scaleType, (double Low, double High) bounds, double step)
    {
        if (step == 0.0)
            return value;
        double unnormalized = UnnormalizeOneParam(value, scaleType, bounds, step);
        double rounded = MathUtilClip(
            NDDivision.FloorDivDouble(unnormalized - bounds.Low + 0.5 * step, step) * step + bounds.Low,
            bounds.Low, bounds.High);
        return NormalizeOneParam(rounded, scaleType, bounds, step);
    }

    /// <summary>
    ///     NumPy <c>np.arange(start, stop, step)</c>, computed by NumSharp's <c>np.arange</c>: length
    ///     <c>ceil((stop - start) / step)</c>, elements <c>start</c>, <c>start + step</c>, then
    ///     <c>start + i * delta</c> with <c>delta = (start + step) - start</c>.
    /// </summary>
    /// <param name="start">The inclusive start of the interval or slice.</param>
    /// <param name="stop">The final value of the generated numerical interval.</param>
    /// <param name="step">The nonzero increment between successive grid values.</param>
    /// <returns>The generated arithmetic progression, excluding its stop value; empty when <paramref name="stop" /> is not past <paramref name="start" />.</returns>
    /// <remarks>
    ///     NumPy fills the ramp with the step as it was stored, <c>(start + step) - start</c>, not with <c>step</c>
    ///     (<c>PyArray_ArangeObj</c> and the dtype's <c>fill</c>), so a stepped float whose step does not survive that
    ///     round trip gets other grid points: <c>FloatDistribution(0.1, 1.0, step=0.3)</c> ends its grid at
    ///     <c>1.0</c> in Optuna, where the former <c>start + i * step</c> gave <c>0.9999999999999998</c> (QA-49; about
    ///     half of random ranges differ somewhere). The grid is built once per discrete dimension per
    ///     <c>GetChoicesOfDiscreteParams</c> call, so the NumSharp call's microseconds do not matter.
    /// </remarks>
    /// <exception cref="ZeroDivisionError"><paramref name="step" /> is zero: NumPy's <c>ZeroDivisionError</c>, which NumSharp's <c>np.arange</c> raises as Python's class since 0.81.0 (it was .NET's <see cref="DivideByZeroException" /> before); the callers only pass discrete steps, which are positive.</exception>
    private static double[] Arange(double start, double stop, double step)
    {
        using NDArray grid = np.arange(start, stop, step);
        return grid.ToArray<double>();
    }

    /// <summary>
    ///     NumPy <c>np.clip</c> against scalar bounds, through <see cref="MathUtil.Clip" />: Optuna's two clips here
    ///     (<c>_get_unnormalized_param</c> on a 0-d value, <c>_round_one_normalized_param</c> on an array against
    ///     <c>bounds[0]</c> / <c>bounds[1]</c>) both run NumPy's const loop, where a value equal to a bound keeps its own
    ///     zero sign.
    /// </summary>
    /// <param name="value">The value to clip; a NaN stays NaN.</param>
    /// <param name="low">The lower endpoint of the allowed range.</param>
    /// <param name="high">The upper endpoint of the allowed range.</param>
    /// <returns>The value when inside the range or on its edge, else the bound it crossed.</returns>
    /// <remarks>
    ///     The former <c>Math.Min(Math.Max(value, low), high)</c> turned <c>-0.0</c> on a <c>+0.0</c> bound into
    ///     <c>+0.0</c> and back (QA-30's class).
    /// </remarks>
    private static double MathUtilClip(double value, double low, double high)
        => MathUtil.Clip(value, low, high);
}
