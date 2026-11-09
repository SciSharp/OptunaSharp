using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp.Internal;

/// <summary>
///     Transform a search space and parameter configurations to continuous space.
///     Port of <c>optuna/_transform.py: _SearchSpaceTransform</c> — the numeric boundary of the
///     whole system.
///
///     The search space bounds and parameter configurations are represented as float64 arrays
///     and transformed into continuous space. Bounds and parameters associated with categorical
///     distributions are one-hot encoded. Parameter configurations in this space can additionally
///     be untransformed, or mapped back to the original space.
/// </summary>
/// <remarks>
///     <c>transformLog</c> and <c>transformStep</c> are useful for constructing bounds and
///     parameters without any actual transformations by setting those arguments to <c>false</c>.
///     This is needed for e.g. the hyperparameter importance assessments.
///     Parameter values are not scaled to the unit cube unless <c>transform01</c> is set.
/// </remarks>
internal sealed class SearchSpaceTransform
{
    private readonly double[,] _rawBounds;
    private readonly List<long[]> _columnToEncodedColumns;
    private readonly long[] _encodedColumnToColumn;
    private readonly OrderedDict<string, BaseDistribution> _searchSpace;
    private readonly bool _transformLog;
    private readonly bool _transform01;

    /// <summary>Builds the encoded parameter-column layout and transformed bounds for one ordered search space.</summary>
    /// <param name="searchSpace">The parameter distributions in the column order used by the transform or model.</param>
    /// <param name="transformLog">Whether logarithmic distributions are mapped through their natural logarithm.</param>
    /// <param name="transformStep">Whether discrete bounds are expanded by half a step before normalization.</param>
    /// <param name="transform01">Whether transformed numerical coordinates are rescaled to the unit interval.</param>
    public SearchSpaceTransform(
        OrderedDict<string, BaseDistribution> searchSpace,
        bool transformLog = true,
        bool transformStep = true,
        bool transform01 = false)
    {
        (_rawBounds, _columnToEncodedColumns, _encodedColumnToColumn) =
            TransformSearchSpace(searchSpace, transformLog, transformStep);
        _searchSpace = searchSpace;
        _transformLog = transformLog;
        _transform01 = transform01;
    }

    /// <summary>Constructed bounds from the given search space, shape <c>(n_bounds, 2)</c>, float64.</summary>
    public NDArray Bounds
    {
        get
        {
            int n = _rawBounds.GetLength(0);
            var bounds = new double[n, 2];
            if (_transform01)
            {
                for (int i = 0; i < n; i++)
                {
                    bounds[i, 0] = 0.0;
                    bounds[i, 1] = 1.0;
                }
            }
            else
            {
                Array.Copy(_rawBounds, bounds, _rawBounds.Length);
            }
            return np.array(bounds);
        }
    }

    /// <summary>Raw (untransformed-to-unit-cube) bounds, shape <c>(n_bounds, 2)</c>.</summary>
    internal double[,] RawBounds => _rawBounds;

    /// <summary>Mapping from original parameter column index to encoded column indices.</summary>
    public IReadOnlyList<long[]> ColumnToEncodedColumns => _columnToEncodedColumns;

    /// <summary>Mapping from encoded column index to original parameter column index.</summary>
    public long[] EncodedColumnToColumn => _encodedColumnToColumn;

    /// <summary>
    ///     Transform a parameter configuration from actual values to continuous space.
    /// </summary>
    /// <param name="parameters">A parameter configuration to transform.</param>
    /// <returns>A 1-dimensional float64 array holding the transformed parameters; the caller owns it and should dispose it.</returns>
    /// <remarks>
    ///     A caller that only needs the values should call <see cref="TransformValues" />: this overload wraps the same
    ///     values in a NumSharp array, which waits for the finalizer when it is not disposed (plan 14).
    /// </remarks>
    /// <exception cref="NumSharp.TypeError">A numerical parameter value is not a real number.</exception>
    public NDArray Transform(IReadOnlyDictionary<string, object?> parameters)
        => np.array(TransformValues(parameters));

    /// <summary>
    ///     Transform a parameter configuration from actual values to continuous space, as a managed array: the values
    ///     <see cref="Transform" /> returns, without the NumSharp array around them.
    /// </summary>
    /// <param name="parameters">A parameter configuration holding every parameter of the search space.</param>
    /// <returns>A new array of the transformed parameters, one entry per encoded column (one-hot columns for categoricals).</returns>
    /// <exception cref="NumSharp.TypeError">A numerical parameter value is not a real number.</exception>
    internal double[] TransformValues(IReadOnlyDictionary<string, object?> parameters)
    {
        int n = _rawBounds.GetLength(0);
        var transParams = new double[n];

        int boundIdx = 0;
        foreach (var (name, distribution) in _searchSpace)
        {
            System.Diagnostics.Debug.Assert(
                parameters.ContainsKey(name), "Parameter configuration must contain all distributions.");
            var param = parameters[name];

            if (distribution is CategoricalDistribution categorical)
            {
                int choiceIdx = (int)categorical.ToInternalRepr(param);
                transParams[boundIdx + choiceIdx] = 1;
                boundIdx += categorical.Choices.Count;
            }
            else
            {
                transParams[boundIdx] = TransformNumericalParam(
                    PyFloatValue(param), distribution, _transformLog);
                boundIdx += 1;
            }
        }

        if (_transform01)
        {
            for (int i = 0; i < n; i++)
            {
                bool single = _rawBounds[i, 0] == _rawBounds[i, 1];
                if (single)
                    transParams[i] = 0.5;
                else
                    transParams[i] = (transParams[i] - _rawBounds[i, 0]) / (_rawBounds[i, 1] - _rawBounds[i, 0]);
            }
        }

        return transParams;
    }

    /// <summary>
    ///     Untransform a parameter configuration from continuous space to actual values.
    /// </summary>
    /// <param name="transParams">A 1-dimensional array in the transformed space; any numeric dtype, read as float64.</param>
    /// <returns>An untransformed parameter configuration keyed by parameter name.</returns>
    /// <remarks>
    ///     Reads the values through NumPy's <c>astype(float64)</c> copy, which is disposed here, then untransforms them as
    ///     <see cref="UntransformValues" /> does. A caller holding a managed float64 array should call that
    ///     overload directly and skip the NumSharp copy.
    /// </remarks>
    public OrderedDict<string, object?> Untransform(NDArray transParams)
    {
        System.Diagnostics.Debug.Assert(transParams.shape.Length == 1 && transParams.shape[0] == _rawBounds.GetLength(0));

        // astype copies by default, so the copy is this method's to dispose; left alone it would survive a collection on
        // the finalization queue (plan 14).
        double[] values;
        using (NDArray asFloat64 = transParams.astype(np.float64))
            values = asFloat64.ToArray<double>();
        return UntransformValues(values);
    }

    /// <summary>
    ///     Untransform a parameter configuration from continuous space to actual values, reading float64 values directly:
    ///     the result <see cref="Untransform(NDArray)" /> gives for the same values, without a NumSharp array.
    /// </summary>
    /// <param name="transParams">One value per encoded column, in the transformed space; not modified.</param>
    /// <returns>An untransformed parameter configuration keyed by parameter name, in search-space order.</returns>
    /// <exception cref="System.InvalidOperationException">A distribution of the search space is not supported (cannot happen for a transform this class built).</exception>
    internal OrderedDict<string, object?> UntransformValues(ReadOnlySpan<double> transParams)
    {
        int n = _rawBounds.GetLength(0);
        System.Diagnostics.Debug.Assert(transParams.Length == n);

        ReadOnlySpan<double> values = transParams;
        if (_transform01)
        {
            var scaled = new double[n];
            for (int i = 0; i < n; i++)
                scaled[i] = _rawBounds[i, 0] + values[i] * (_rawBounds[i, 1] - _rawBounds[i, 0]);
            values = scaled;
        }

        // Sized for the search space up front: the result is built on every untransform (per trial and per independent
        // draw), and growing an OrderedDictionary from empty reallocates several times for a few dozen parameters.
        var parameters = new OrderedDict<string, object?>(_searchSpace.Count);

        int column = 0;
        foreach (var (name, distribution) in _searchSpace)
        {
            var encodedColumns = _columnToEncodedColumns[column];
            column++;

            object? param;
            if (distribution is CategoricalDistribution categorical)
            {
                // Select the highest rated one-hot encoding (np.argmax: first max wins).
                int argmax = 0;
                double best = values[(int)encodedColumns[0]];
                for (int i = 1; i < encodedColumns.Length; i++)
                {
                    double v = values[(int)encodedColumns[i]];
                    if (v > best)
                    {
                        best = v;
                        argmax = i;
                    }
                }
                param = categorical.ToExternalRepr(argmax);
            }
            else
            {
                param = UntransformNumericalParam(values[(int)encodedColumns[0]], distribution, _transformLog);
            }

            parameters[name] = param;
        }

        return parameters;
    }

    /// <summary>Python <c>_transform_search_space</c>.</summary>
    /// <param name="searchSpace">The parameter distributions in the column order used by the transform or model.</param>
    /// <param name="transformLog">Whether logarithmic distributions are mapped through their natural logarithm.</param>
    /// <param name="transformStep">Whether discrete bounds are expanded by half a step before normalization.</param>
    /// <returns>The transformed bounds, original-to-encoded column groups, and encoded-to-original column indices.</returns>
    /// <exception cref="System.InvalidOperationException">A distribution type other than the supported float, integer, or categorical distributions is encountered.</exception>
    private static (double[,], List<long[]>, long[]) TransformSearchSpace(
        OrderedDict<string, BaseDistribution> searchSpace, bool transformLog, bool transformStep)
    {
        System.Diagnostics.Debug.Assert(searchSpace.Count > 0, "Cannot transform if no distributions are given.");

        int nBounds = 0;
        foreach (var d in searchSpace.Values)
            nBounds += d is CategoricalDistribution c ? c.Choices.Count : 1;

        var bounds = new double[nBounds, 2];
        var columnToEncodedColumns = new List<long[]>();
        var encodedColumnToColumn = new long[nBounds];

        int boundIdx = 0;
        foreach (var distribution in searchSpace.Values)
        {
            switch (distribution)
            {
                case CategoricalDistribution categorical:
                {
                    int nChoices = categorical.Choices.Count;
                    var encodedColumns = new long[nChoices];
                    for (int i = 0; i < nChoices; i++)
                    {
                        bounds[boundIdx + i, 0] = 0;
                        bounds[boundIdx + i, 1] = 1;
                        encodedColumns[i] = boundIdx + i;
                        encodedColumnToColumn[boundIdx + i] = columnToEncodedColumns.Count;
                    }
                    columnToEncodedColumns.Add(encodedColumns);
                    boundIdx += nChoices;
                    break;
                }
                case FloatDistribution floatDist:
                {
                    double bdLow, bdHigh;
                    if (floatDist.Step is not null)
                    {
                        double halfStep = transformStep ? 0.5 * floatDist.Step.Value : 0.0;
                        bdLow = TransformNumericalParam(floatDist.Low, floatDist, transformLog) - halfStep;
                        bdHigh = TransformNumericalParam(floatDist.High, floatDist, transformLog) + halfStep;
                    }
                    else
                    {
                        bdLow = TransformNumericalParam(floatDist.Low, floatDist, transformLog);
                        bdHigh = TransformNumericalParam(floatDist.High, floatDist, transformLog);
                    }
                    bounds[boundIdx, 0] = bdLow;
                    bounds[boundIdx, 1] = bdHigh;
                    encodedColumnToColumn[boundIdx] = columnToEncodedColumns.Count;
                    columnToEncodedColumns.Add(new long[] { boundIdx });
                    boundIdx += 1;
                    break;
                }
                case IntDistribution intDist:
                {
                    double halfStep = transformStep ? 0.5 * intDist.Step : 0.0;
                    double bdLow, bdHigh;
                    if (intDist.Log)
                    {
                        bdLow = TransformNumericalParam(intDist.Low - halfStep, intDist, transformLog);
                        bdHigh = TransformNumericalParam(intDist.High + halfStep, intDist, transformLog);
                    }
                    else
                    {
                        bdLow = TransformNumericalParam(intDist.Low, intDist, transformLog) - halfStep;
                        bdHigh = TransformNumericalParam(intDist.High, intDist, transformLog) + halfStep;
                    }
                    bounds[boundIdx, 0] = bdLow;
                    bounds[boundIdx, 1] = bdHigh;
                    encodedColumnToColumn[boundIdx] = columnToEncodedColumns.Count;
                    columnToEncodedColumns.Add(new long[] { boundIdx });
                    boundIdx += 1;
                    break;
                }
                default:
                    throw new InvalidOperationException("Should not reach. Unexpected distribution.");
            }
        }

        System.Diagnostics.Debug.Assert(boundIdx == nBounds);

        return (bounds, columnToEncodedColumns, encodedColumnToColumn);
    }

    /// <summary>Python <c>_transform_numerical_param</c>.</summary>
    /// <param name="param">The parameter name to look up in trial data.</param>
    /// <param name="distribution">The distribution defining the external value's numeric support and scale.</param>
    /// <param name="transformLog">Whether logarithmic distributions are mapped through their natural logarithm.</param>
    /// <returns>The external numerical value in linear or log-transformed coordinates.</returns>
    /// <exception cref="System.InvalidOperationException">The distribution is categorical or is not a supported numerical distribution.</exception>
    internal static double TransformNumericalParam(double param, BaseDistribution distribution, bool transformLog)
        => distribution switch
        {
            CategoricalDistribution => throw new InvalidOperationException("Should not reach. Should be one-hot encoded."),
            FloatDistribution f => f.Log && transformLog ? Math.Log(param) : param,
            IntDistribution i => i.Log && transformLog ? Math.Log(param) : param,
            _ => throw new InvalidOperationException("Should not reach. Unexpected distribution."),
        };

    /// <summary>
    ///     Maps one transformed numerical coordinate back to the external value a sampler hands to the
    ///     trial (Python <c>_untransform_numerical_param</c>): a boxed <see cref="double" /> for float
    ///     distributions or a boxed <see cref="long" /> for int distributions, always inside the
    ///     distribution's bounds so the suggested value passes the distribution's own containment check.
    /// </summary>
    /// <remarks>
    ///     The clamping is deliberately asymmetric, exactly as in Optuna 5.0.0: a log float is clipped to
    ///     <c>[low, nextafter(high, high - 1)]</c> (both ends), a plain float only has its upper end pulled
    ///     below <c>high</c>, a stepped float snaps to the grid inside <c>[low, high]</c>, and ints round
    ///     half-up (linear) or half-to-even (log) before clamping to <c>[low, high]</c>. A NaN coordinate
    ///     stays NaN on the float branches, matching <c>np.clip</c>/<c>min</c> in Python.
    /// </remarks>
    /// <param name="transParam">The scalar coordinate in the transformed numerical space.</param>
    /// <param name="distribution">The distribution defining the external value's numeric support and scale.</param>
    /// <param name="transformLog">Whether logarithmic distributions are mapped through their natural logarithm.</param>
    /// <returns>The external numeric value after inverse scaling, clipping, and grid rounding.</returns>
    /// <exception cref="System.InvalidOperationException">The distribution is categorical or is not a supported numerical distribution.</exception>
    internal static object UntransformNumericalParam(double transParam, BaseDistribution distribution, bool transformLog)
    {
        switch (distribution)
        {
            case CategoricalDistribution:
                throw new InvalidOperationException("Should not reach. Should be one-hot encoded.");
            case FloatDistribution f:
            {
                double param;
                if (f.Log)
                {
                    param = transformLog ? Math.Exp(transParam) : transParam;
                    // Optuna 5.0.0 (#6799) clamps BOTH ends here: exp(log(low)) rounds to one ULP below
                    // low for roughly a third of possible bounds (exp(log(1e-7)) == 9.999999999999994e-08),
                    // so a sample on the lower edge of the box — QMC's first Sobol point, a CMA-ES bound
                    // repair, a GA clip — would otherwise leave the distribution. Clip propagates NaN the
                    // way np.clip does, so a NaN coordinate still reaches the caller unchanged.
                    if (!f.Single())
                        param = MathUtil.Clip(param, f.Low, MathUtil.NextAfter(f.High, f.High - 1));
                }
                else if (f.Step is not null)
                {
                    // np.round: banker's rounding — Math.Round's default mode.
                    param = MathUtil.Clip(
                        Math.Round((transParam - f.Low) / f.Step.Value) * f.Step.Value + f.Low, f.Low, f.High);
                }
                else
                {
                    param = f.Single()
                        ? transParam
                        : Math.Min(transParam, MathUtil.NextAfter(f.High, f.High - 1));
                }
                return param;
            }
            case IntDistribution i:
            {
                long param;
                if (i.Log)
                {
                    if (transformLog)
                    {
                        // Python round(): banker's rounding to int.
                        param = (long)Math.Round(Math.Exp(transParam), MidpointRounding.ToEven);
                        param = Math.Min(Math.Max(param, i.Low), i.High);
                    }
                    else
                    {
                        param = (long)transParam;
                    }
                }
                else
                {
                    // Python: math.floor((t - low) / step + 0.5) — round half up.
                    long stepIndex = (long)Math.Floor((transParam - i.Low) / i.Step + 0.5);
                    param = stepIndex * i.Step + i.Low;
                    param = Math.Min(Math.Max(param, i.Low), i.High);
                }
                return param;
            }
            default:
                throw new InvalidOperationException("Should not reach. Unexpected distribution.");
        }
    }

    /// <summary>Returns the Python-compatible floating-point conversion of the external scalar.</summary>
    /// <param name="param">The parameter name to look up in trial data.</param>
    /// <returns>The Python-compatible floating-point conversion of the external scalar.</returns>
    /// <exception cref="NumSharp.TypeError">The external parameter cannot be converted with Python-compatible floating-point semantics.</exception>
    private static double PyFloatValue(object? param)
    {
        if (!PyFloat.TryConvert(param, out double value))
            throw new NumSharp.TypeError($"must be real number, not {(param is null ? "NoneType" : param.GetType().Name)}");
        return value;
    }
}
