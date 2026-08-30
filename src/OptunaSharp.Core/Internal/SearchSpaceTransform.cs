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
    /// <returns>A 1-dimensional float64 array holding the transformed parameters.</returns>
    public NDArray Transform(IReadOnlyDictionary<string, object?> parameters)
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

        return np.array(transParams);
    }

    /// <summary>
    ///     Untransform a parameter configuration from continuous space to actual values.
    /// </summary>
    /// <param name="transParams">A 1-dimensional array in the transformed space.</param>
    /// <returns>An untransformed parameter configuration keyed by parameter name.</returns>
    public OrderedDict<string, object?> Untransform(NDArray transParams)
    {
        int n = _rawBounds.GetLength(0);
        System.Diagnostics.Debug.Assert(transParams.shape.Length == 1 && transParams.shape[0] == n);

        var values = transParams.astype(np.float64).ToArray<double>();

        if (_transform01)
        {
            var scaled = new double[n];
            for (int i = 0; i < n; i++)
                scaled[i] = _rawBounds[i, 0] + values[i] * (_rawBounds[i, 1] - _rawBounds[i, 0]);
            values = scaled;
        }

        var parameters = new OrderedDict<string, object?>();

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
                double best = values[encodedColumns[0]];
                for (int i = 1; i < encodedColumns.Length; i++)
                {
                    double v = values[encodedColumns[i]];
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
                param = UntransformNumericalParam(values[encodedColumns[0]], distribution, _transformLog);
            }

            parameters[name] = param;
        }

        return parameters;
    }

    /// <summary>Python <c>_transform_search_space</c>.</summary>
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
    internal static double TransformNumericalParam(double param, BaseDistribution distribution, bool transformLog)
        => distribution switch
        {
            CategoricalDistribution => throw new InvalidOperationException("Should not reach. Should be one-hot encoded."),
            FloatDistribution f => f.Log && transformLog ? Math.Log(param) : param,
            IntDistribution i => i.Log && transformLog ? Math.Log(param) : param,
            _ => throw new InvalidOperationException("Should not reach. Unexpected distribution."),
        };

    /// <summary>Python <c>_untransform_numerical_param</c>. Returns a boxed double (float) or long (int).</summary>
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
                    if (!f.Single())
                        param = Math.Min(param, MathUtil.NextAfter(f.High, f.High - 1));
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

    private static double PyFloatValue(object? param)
    {
        if (!PyFloat.TryConvert(param, out double value))
            throw new NumSharp.TypeError($"must be real number, not {(param is null ? "NoneType" : param.GetType().Name)}");
        return value;
    }
}
