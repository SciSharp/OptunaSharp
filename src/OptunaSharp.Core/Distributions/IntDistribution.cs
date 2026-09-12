using System;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Distributions;

/// <summary>
///     A distribution on integers. Port of <c>optuna/distributions.py: IntDistribution</c>.
///
///     This object is instantiated by <see cref="Trial.SuggestInt" />, and passed to samplers
///     in general.
/// </summary>
/// <remarks>
///     When <c>step</c> is not 1, if the range <c>[low, high]</c> is not divisible by
///     <c>step</c>, <c>high</c> will be replaced with the maximum of
///     <c>k * step + low &lt; high</c>, where <c>k</c> is an integer.
/// </remarks>
public class IntDistribution : BaseDistribution
{
    /// <summary>If <c>true</c>, this distribution is in log-scaled domain.</summary>
    public bool Log { get; }

    /// <summary>A discretization step. Must be a positive integer (1 when <c>log</c> is <c>true</c>).</summary>
    public long Step { get; }

    /// <summary>Lower endpoint of the range of the distribution. <c>Low</c> is included in the range.</summary>
    public long Low { get; }

    /// <summary>Upper endpoint of the range of the distribution. <c>High</c> is included in the range.</summary>
    public long High { get; }

    /// <summary>Creates a numeric distribution and normalizes a stepped upper endpoint.</summary>
    /// <param name="low">Inclusive lower endpoint of the parameter range.</param>
    /// <param name="high">Inclusive upper endpoint; a stepped range may lower it to the last reachable grid point.</param>
    /// <param name="log">Whether to sample in the logarithmic domain, which requires a strictly positive lower endpoint.</param>
    /// <param name="step">Positive integer grid spacing; logarithmic ranges require one.</param>
    /// <exception cref="NumSharp.ValueError">The lower endpoint exceeds the upper endpoint, logarithmic bounds are nonpositive, or the step is invalid for the selected scale.</exception>
    public IntDistribution(long low, long high, bool log = false, long step = 1)
    {
        if (log && step != 1)
            throw new ValueError(
                "Samplers and other components in Optuna only accept step is 1 " +
                "when `log` argument is True.");

        if (low > high)
            throw new ValueError($"`low <= high` must hold, but got (low={low}, high={high}).");

        if (log && low < 1)
            throw new ValueError($"`low >= 1` must hold for `log=True`, but got (low={low}, high={high}).");

        if (step <= 0)
            throw new ValueError($"`step > 0` must hold, but got step={step}.");

        Log = log;
        Step = step;
        Low = low;
        High = AdjustIntUniformHigh(Low, high, Step);
    }

    /// <summary>External representation is a boxed <see cref="long" /> (Python <c>int</c>).</summary>
    /// <param name="paramValueInInternalRepr">Numeric internal representation: a numeric value or a zero-based categorical index.</param>
    /// <returns>The internal value cast to a boxed 64-bit integer.</returns>
    public override object? ToExternalRepr(double paramValueInInternalRepr)
        => (long)paramValueInInternalRepr;

    /// <summary>Converts an external value into the numeric representation used by samplers.</summary>
    /// <param name="paramValueInExternalRepr">External parameter value to convert to the distribution's numeric representation.</param>
    /// <returns>The converted numeric value; range membership is checked separately by Contains.</returns>
    /// <exception cref="NumSharp.ValueError">The value cannot be converted to a number, is NaN, or is nonpositive for a logarithmic distribution.</exception>
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

    /// <summary>Tests whether this distribution admits only one candidate.</summary>
    /// <returns>True when sampling can be replaced by the sole allowed value.</returns>
    public override bool Single()
    {
        if (Log)
            return Low == High;

        if (Low == High)
            return true;
        return (High - Low) < Step;
    }

    /// <summary>Tests membership using the stored numeric representation.</summary>
    /// <param name="paramValueInInternalRepr">Numeric internal representation: a numeric value or a zero-based categorical index.</param>
    /// <returns>True when the value belongs to the range or categorical index domain.</returns>
    internal override bool Contains(double paramValueInInternalRepr)
    {
        double value = paramValueInInternalRepr;
        return Low <= value && value <= High && (value - Low) % Step == 0;
    }

    /// <summary>Builds the attribute payload used by Optuna JSON serialization.</summary>
    /// <returns>A new ordered attribute dictionary; values preserve their distribution representation.</returns>
    internal override OrderedDict<string, object?> AsDict() => new()
    {
        // __init__ assignment order: log, step, low, high.
        ["log"] = Log,
        ["step"] = Step,
        ["low"] = Low,
        ["high"] = High,
    };

    /// <summary>Compares distribution types and their defining attributes.</summary>
    /// <param name="other">Value to compare with this instance; null compares unequal.</param>
    /// <returns>True when the distributions have compatible types and equal attributes.</returns>
    public override bool Equals(BaseDistribution? other)
    {
        if (other is null || other.GetType() != GetType())
            return false;
        var o = (IntDistribution)other;
        return Log == o.Log && Step == o.Step && Low == o.Low && High == o.High;
    }

    /// <summary>Computes a hash from the distribution type and defining attributes.</summary>
    /// <returns>Hash code consistent with distribution equality.</returns>
    public override int GetHashCode() => HashCode.Combine(GetType(), Log, Step, Low, High);

    /// <summary>Python <c>_adjust_int_uniform_high</c>.</summary>
    /// <param name="low">Inclusive lower endpoint of the parameter range.</param>
    /// <param name="high">Inclusive upper endpoint; a stepped range may lower it to the last reachable grid point.</param>
    /// <param name="step">Positive grid spacing used to align the upper endpoint.</param>
    /// <returns>The greatest grid point not exceeding the requested upper endpoint.</returns>
    /// <exception cref="System.DivideByZeroException">The caller supplies a zero step.</exception>
    internal static long AdjustIntUniformHigh(long low, long high, long step)
    {
        long r = high - low;
        if (r % step != 0)
        {
            long oldHigh = high;
            high = r / step * step + low;
            OptunaWarnings.Warn(
                $"The distribution is specified by [{low}, {oldHigh}] and step={step}, but the range is " +
                $"not divisible by `step`. It will be replaced with [{low}, {high}].");
        }
        return high;
    }
}

/// <summary>
///     A uniform distribution on integers.
///     Deprecated since Optuna 3.0.0: use <see cref="IntDistribution" /> instead.
/// </summary>
[Obsolete(DeprecationMessages.IntUniformDistribution)]
public class IntUniformDistribution : IntDistribution
{
    // Optuna's @deprecated_class wraps __init__ to warn on construction; mirror that with a
    // runtime FutureWarning carrying the identical message (see DeprecationMessages).
    /// <summary>Creates a numeric distribution and normalizes a stepped upper endpoint.</summary>
    /// <param name="low">Inclusive lower endpoint of the parameter range.</param>
    /// <param name="high">Inclusive upper endpoint; a stepped range may lower it to the last reachable grid point.</param>
    /// <param name="step">Positive integer grid spacing; logarithmic ranges require one.</param>
    /// <exception cref="NumSharp.ValueError">The lower endpoint exceeds the upper endpoint, logarithmic bounds are nonpositive, or the step is invalid for the selected scale.</exception>
    public IntUniformDistribution(long low, long high, long step = 1) : base(low, high, log: false, step: step)
        => OptunaWarnings.Warn(DeprecationMessages.IntUniformDistribution, WarningCategory.FutureWarning);

    /// <summary>Builds the attribute payload used by Optuna JSON serialization.</summary>
    /// <returns>A new ordered attribute dictionary; values preserve their distribution representation.</returns>
    internal override OrderedDict<string, object?> AsDict() => new()
    {
        ["step"] = Step,
        ["low"] = Low,
        ["high"] = High,
    };
}

/// <summary>
///     A uniform distribution on integers in the log domain.
///     Deprecated since Optuna 3.0.0: use <see cref="IntDistribution" /> instead.
/// </summary>
[Obsolete(DeprecationMessages.IntLogUniformDistribution)]
public class IntLogUniformDistribution : IntDistribution
{
    /// <summary>Creates a numeric distribution and normalizes a stepped upper endpoint.</summary>
    /// <param name="low">Inclusive lower endpoint of the parameter range.</param>
    /// <param name="high">Inclusive upper endpoint; a stepped range may lower it to the last reachable grid point.</param>
    /// <param name="step">Positive integer grid spacing; logarithmic ranges require one.</param>
    /// <exception cref="NumSharp.ValueError">The lower endpoint exceeds the upper endpoint, logarithmic bounds are nonpositive, or the step is invalid for the selected scale.</exception>
    public IntLogUniformDistribution(long low, long high, long step = 1) : base(low, high, log: true, step: step)
        => OptunaWarnings.Warn(DeprecationMessages.IntLogUniformDistribution, WarningCategory.FutureWarning);

    /// <summary>Builds the attribute payload used by Optuna JSON serialization.</summary>
    /// <returns>A new ordered attribute dictionary; values preserve their distribution representation.</returns>
    internal override OrderedDict<string, object?> AsDict() => new()
    {
        ["step"] = Step,
        ["low"] = Low,
        ["high"] = High,
    };
}
