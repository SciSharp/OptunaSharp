using System;
using System.Globalization;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Distributions;

/// <summary>
///     A distribution on floats. Port of <c>optuna/distributions.py: FloatDistribution</c>.
///
///     This object is instantiated by <see cref="Trial.SuggestFloat(string, double, double)" />, and
///     passed to samplers in general.
/// </summary>
/// <remarks>
///     When <c>step</c> is not <c>null</c>, if the range <c>[low, high]</c> is not divisible by
///     <c>step</c>, <c>high</c> will be replaced with the maximum of
///     <c>k * step + low &lt; high</c>, where <c>k</c> is an integer.
/// </remarks>
public class FloatDistribution : BaseDistribution
{
    /// <summary>A discretization step. <c>null</c> when the distribution is continuous.</summary>
    public double? Step { get; internal set; }

    /// <summary>Lower endpoint of the range of the distribution. <c>Low</c> is included in the range.</summary>
    public double Low { get; }

    /// <summary>Upper endpoint of the range of the distribution. <c>High</c> is included in the range.</summary>
    public double High { get; }

    /// <summary>If <c>true</c>, this distribution is in log-scaled domain.</summary>
    public bool Log { get; }

    public FloatDistribution(double low, double high, bool log = false, double? step = null)
    {
        if (log && step is not null)
            throw new ValueError("The parameter `step` is not supported when `log` is true.");

        if (low > high)
            throw new ValueError(
                $"`low <= high` must hold, but got (low={PythonRepr.FloatRepr(low)}, high={PythonRepr.FloatRepr(high)}).");

        if (log && low <= 0.0)
            throw new ValueError(
                $"`low > 0` must hold for `log=True`, but got (low={PythonRepr.FloatRepr(low)}, high={PythonRepr.FloatRepr(high)}).");

        if (step is not null && step <= 0)
            throw new ValueError($"`step > 0` must hold, but got step={PythonRepr.FloatRepr(step.Value)}.");

        Step = null;
        if (step is not null)
        {
            high = AdjustDiscreteUniformHigh(low, high, step.Value);
            Step = step.Value;
        }

        Low = low;
        High = high;
        Log = log;
    }

    public override bool Single()
    {
        if (Step is null)
            return Low == High;

        if (Low == High)
            return true;
        // Python computes (high - low) < step exactly, via Decimal on the shortest reprs.
        try
        {
            var high = decimal.Parse(PythonRepr.FloatRepr(High), NumberStyles.Float, CultureInfo.InvariantCulture);
            var low = decimal.Parse(PythonRepr.FloatRepr(Low), NumberStyles.Float, CultureInfo.InvariantCulture);
            var step = decimal.Parse(PythonRepr.FloatRepr(Step.Value), NumberStyles.Float, CultureInfo.InvariantCulture);
            return (high - low) < step;
        }
        catch (OverflowException)
        {
            // Magnitudes beyond decimal's range: fall back to double arithmetic.
            return (High - Low) < Step.Value;
        }
    }

    internal override bool Contains(double paramValueInInternalRepr)
    {
        double value = paramValueInInternalRepr;
        if (Step is null)
            return Low <= value && value <= High;

        double k = (value - Low) / Step.Value;
        // Math.Round defaults to banker's rounding — same as Python round().
        return Low <= value && value <= High && Math.Abs(k - Math.Round(k)) < 1.0e-8;
    }

    public override double ToInternalRepr(object? paramValueInExternalRepr)
    {
        if (!PyFloat.TryConvert(paramValueInExternalRepr, out double internalRepr))
            throw new ValueError(
                $"'{PythonRepr.Str(paramValueInExternalRepr)}' is not a valid type. float-castable value is expected.");

        if (double.IsNaN(internalRepr))
            throw new ValueError($"`{PythonRepr.Str(paramValueInExternalRepr)}` is invalid value.");
        if (Log && internalRepr <= 0.0)
            throw new ValueError(
                $"`{PythonRepr.Str(paramValueInExternalRepr)}` is invalid value for the case log=True.");
        return internalRepr;
    }

    internal override OrderedDict<string, object?> AsDict() => new()
    {
        // __init__ assignment order: step, low, high, log.
        ["step"] = Step,
        ["low"] = Low,
        ["high"] = High,
        ["log"] = Log,
    };

    public override bool Equals(BaseDistribution? other)
    {
        if (other is null || other.GetType() != GetType())
            return false;
        var o = (FloatDistribution)other;
        // Python compares __dict__ with ==; None == None and float equality (no NaN in bounds).
        return Nullable.Equals(Step, o.Step) && Low.Equals(o.Low) && High.Equals(o.High) && Log == o.Log;
    }

    public override int GetHashCode() => HashCode.Combine(GetType(), Step, Low, High, Log);

    /// <summary>Python <c>_adjust_discrete_uniform_high</c> (exact Decimal arithmetic on the shortest reprs).</summary>
    internal static double AdjustDiscreteUniformHigh(double low, double high, double step)
    {
        try
        {
            var dHigh = decimal.Parse(PythonRepr.FloatRepr(high), NumberStyles.Float, CultureInfo.InvariantCulture);
            var dLow = decimal.Parse(PythonRepr.FloatRepr(low), NumberStyles.Float, CultureInfo.InvariantCulture);
            var dStep = decimal.Parse(PythonRepr.FloatRepr(step), NumberStyles.Float, CultureInfo.InvariantCulture);

            var dR = dHigh - dLow;

            if (dR % dStep != 0m)
            {
                double oldHigh = high;
                // Python: float((d_r // d_step) * d_step + d_low); Decimal floor division.
                high = (double)(decimal.Floor(dR / dStep) * dStep + dLow);
                OptunaWarnings.Warn(
                    $"The distribution is specified by [{PythonRepr.FloatRepr(low)}, {PythonRepr.FloatRepr(oldHigh)}] " +
                    $"and step={PythonRepr.FloatRepr(step)}, but the range is " +
                    $"not divisible by `step`. It will be replaced with [{PythonRepr.FloatRepr(low)}, {PythonRepr.FloatRepr(high)}].");
            }

            return high;
        }
        catch (OverflowException)
        {
            return high;
        }
    }
}

/// <summary>
///     A uniform distribution in the linear domain.
///     Deprecated since Optuna 3.0.0: use <see cref="FloatDistribution" /> instead.
/// </summary>
[Obsolete(DeprecationMessages.UniformDistribution)]
public class UniformDistribution : FloatDistribution
{
    // Optuna's @deprecated_class wraps __init__ to warn on construction; mirror that with a
    // runtime FutureWarning carrying the identical message (see DeprecationMessages).
    public UniformDistribution(double low, double high) : base(low, high, log: false, step: null)
        => OptunaWarnings.Warn(DeprecationMessages.UniformDistribution, WarningCategory.FutureWarning);

    internal override OrderedDict<string, object?> AsDict() => new()
    {
        ["low"] = Low,
        ["high"] = High,
    };
}

/// <summary>
///     A uniform distribution in the log domain.
///     Deprecated since Optuna 3.0.0: use <see cref="FloatDistribution" /> instead.
/// </summary>
[Obsolete(DeprecationMessages.LogUniformDistribution)]
public class LogUniformDistribution : FloatDistribution
{
    public LogUniformDistribution(double low, double high) : base(low, high, log: true, step: null)
        => OptunaWarnings.Warn(DeprecationMessages.LogUniformDistribution, WarningCategory.FutureWarning);

    internal override OrderedDict<string, object?> AsDict() => new()
    {
        ["low"] = Low,
        ["high"] = High,
    };
}

/// <summary>
///     A discretized uniform distribution in the linear domain.
///     Deprecated since Optuna 3.0.0: use <see cref="FloatDistribution" /> instead.
/// </summary>
[Obsolete(DeprecationMessages.DiscreteUniformDistribution)]
public class DiscreteUniformDistribution : FloatDistribution
{
    public DiscreteUniformDistribution(double low, double high, double q) : base(low, high, step: q)
        => OptunaWarnings.Warn(DeprecationMessages.DiscreteUniformDistribution, WarningCategory.FutureWarning);

    /// <summary>
    ///     Discretization step: a proxy for the <see cref="FloatDistribution.Step" /> attribute.
    /// </summary>
    public double Q
    {
        get => Step!.Value;
        set => Step = value;
    }

    internal override OrderedDict<string, object?> AsDict() => new()
    {
        ["low"] = Low,
        ["high"] = High,
        ["q"] = Step,
    };
}

/// <summary>
///     Python <c>float()</c> conversion over the external-repr object model — accepts bools,
///     any numeric, and numeric strings (including <c>"nan"</c>/<c>"inf"</c>).
/// </summary>
internal static class PyFloat
{
    internal static bool TryConvert(object? value, out double result)
    {
        switch (value)
        {
            case bool b:
                result = b ? 1.0 : 0.0;
                return true;
            case double d:
                result = d;
                return true;
            case float f:
                result = f;
                return true;
            case sbyte or byte or short or ushort or int or uint or long:
                result = Convert.ToDouble(value, CultureInfo.InvariantCulture);
                return true;
            case ulong ul:
                result = ul;
                return true;
            case decimal m:
                result = (double)m;
                return true;
            case CategoricalChoice choice:
                return TryConvert(choice.Value, out result);
            case string s:
                return TryParse(s, out result);
            default:
                result = 0;
                return false;
        }
    }

    private static bool TryParse(string s, out double result)
    {
        s = s.Trim();
        // Python float() accepts "inf", "infinity", "nan" (any case, optional sign).
        string lower = s.ToLowerInvariant();
        switch (lower)
        {
            case "inf" or "+inf" or "infinity" or "+infinity":
                result = double.PositiveInfinity;
                return true;
            case "-inf" or "-infinity":
                result = double.NegativeInfinity;
                return true;
            case "nan" or "+nan" or "-nan":
                result = double.NaN;
                return true;
        }
        return double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out result);
    }
}
