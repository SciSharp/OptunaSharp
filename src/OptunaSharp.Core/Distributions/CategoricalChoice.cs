using System;
using System.Globalization;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Distributions;

/// <summary>The kind of value a <see cref="CategoricalChoice" /> holds.</summary>
public enum CategoricalChoiceKind
{
    /// <summary>Python <c>None</c>.</summary>
    None = 0,
    /// <summary>A Python bool, stored as a CLR Boolean.</summary>
    Bool,
    /// <summary>A Python integer represented by a signed 64-bit value.</summary>
    Int,
    /// <summary>A Python float represented by a double.</summary>
    Float,
    /// <summary>A Python string represented by a CLR string.</summary>
    String,
}

/// <summary>
///     A categorical parameter value — Python's
///     <c>CategoricalChoiceType = Union[None, bool, int, float, str]</c> as a value type.
///
///     Equality implements <b>Python <c>==</c> semantics</b>, which is a plan §2 ground rule:
///     <c>True == 1</c>, <c>1.0 == 1</c>, <c>nan != nan</c>, and cross-type number/string
///     comparisons are <c>false</c>. This is the one place C# object equality would silently
///     diverge from Optuna, so it is pinned by dedicated tests.
/// </summary>
public readonly struct CategoricalChoice : IEquatable<CategoricalChoice>
{
    private readonly object? _value;

    /// <summary>Stores an already normalized categorical scalar.</summary>
    /// <param name="normalizedValue">Null, bool, long, double, or string; normalization is the caller's responsibility.</param>
    private CategoricalChoice(object? normalizedValue)
    {
        _value = normalizedValue;
    }

    /// <summary>Python <c>None</c>.</summary>
    public static CategoricalChoice None => default;

    /// <summary>
    ///     The underlying value: <c>null</c>, <see cref="bool" />, <see cref="long" />,
    ///     <see cref="double" /> or <see cref="string" />.
    /// </summary>
    public object? Value => _value;

    /// <summary>Gets the Python scalar category of the stored value.</summary>
    public CategoricalChoiceKind Kind => _value switch
    {
        null => CategoricalChoiceKind.None,
        bool => CategoricalChoiceKind.Bool,
        long => CategoricalChoiceKind.Int,
        double => CategoricalChoiceKind.Float,
        _ => CategoricalChoiceKind.String,
    };

    /// <summary>Wraps an arbitrary value, normalizing numerics (all integrals → <see cref="long" />, <c>float</c> → <see cref="double" />).</summary>
    /// <param name="value">Scalar to normalize as a categorical choice; null represents Python None.</param>
    /// <returns>A choice containing the normalized scalar, or the existing choice unchanged.</returns>
    /// <exception cref="NumSharp.TypeError">The value is unsupported, or an unsigned integer exceeds Int64.MaxValue.</exception>
    public static CategoricalChoice From(object? value) => value switch
    {
        null => None,
        CategoricalChoice choice => choice,
        bool b => new CategoricalChoice(b),
        string s => new CategoricalChoice(s),
        double d => new CategoricalChoice(d),
        float f => new CategoricalChoice((double)f),
        sbyte or byte or short or ushort or int or uint or long => new CategoricalChoice(Convert.ToInt64(value, CultureInfo.InvariantCulture)),
        ulong ul => ul <= long.MaxValue
            ? new CategoricalChoice((long)ul)
            : throw new TypeError($"Choices for a categorical distribution should be None, bool, int, float or str, but got {value} of type {value.GetType().Name}."),
        _ => throw new TypeError($"Choices for a categorical distribution should be None, bool, int, float or str, but got {value} of type {value.GetType().Name}."),
    };

    /// <summary>Wraps the scalar in the corresponding categorical representation.</summary>
    /// <param name="value">Scalar to normalize as a categorical choice; null represents Python None.</param>
    /// <returns>A normalized choice; a null string becomes None.</returns>
    public static implicit operator CategoricalChoice(bool value) => new(value);

    /// <summary>Wraps the scalar in the corresponding categorical representation.</summary>
    /// <param name="value">Scalar to normalize as a categorical choice; null represents Python None.</param>
    /// <returns>A normalized choice; a null string becomes None.</returns>
    public static implicit operator CategoricalChoice(long value) => new(value);

    /// <summary>Wraps the scalar in the corresponding categorical representation.</summary>
    /// <param name="value">Scalar to normalize as a categorical choice; null represents Python None.</param>
    /// <returns>A normalized choice; a null string becomes None.</returns>
    public static implicit operator CategoricalChoice(int value) => new((long)value);

    /// <summary>Wraps the scalar in the corresponding categorical representation.</summary>
    /// <param name="value">Scalar to normalize as a categorical choice; null represents Python None.</param>
    /// <returns>A normalized choice; a null string becomes None.</returns>
    public static implicit operator CategoricalChoice(double value) => new(value);

    /// <summary>Wraps the scalar in the corresponding categorical representation.</summary>
    /// <param name="value">Scalar to normalize as a categorical choice; null represents Python None.</param>
    /// <returns>A normalized choice; a null string becomes None.</returns>
    public static implicit operator CategoricalChoice(string? value) => value is null ? None : new(value);

    /// <summary>Gets whether the choice represents Python None.</summary>
    public bool IsNone => _value is null;

    /// <summary>True when the value is a real number whose float value is NaN (Python <c>isinstance(v, Real) and math.isnan(float(v))</c>).</summary>
    public bool IsNaN => _value is double d && double.IsNaN(d);

    /// <summary>The value as a Python <c>float(v)</c> would return it. Throws for None/str.</summary>
    /// <returns>Numeric value with Boolean choices represented as zero or one.</returns>
    /// <exception cref="NumSharp.TypeError">The choice is None or a string.</exception>
    internal double AsDouble() => _value switch
    {
        bool b => b ? 1.0 : 0.0,
        long l => l,
        double d => d,
        _ => throw new TypeError($"float() argument must be a real number, not '{(_value is null ? "NoneType" : "str")}'"),
    };

    /// <summary>Python <c>==</c>.</summary>
    /// <param name="other">Value to compare with this instance; null compares unequal.</param>
    /// <returns>True for equal normalized values; NaN remains unequal to itself.</returns>
    public bool Equals(CategoricalChoice other)
    {
        var a = _value;
        var b = other._value;
        if (a is null || b is null)
            return a is null && b is null;
        if (a is string sa)
            return b is string sb && string.Equals(sa, sb, StringComparison.Ordinal);
        if (b is string)
            return false;
        // Both numeric (bool counts as 0/1 — Python bool is an int subclass). NaN != NaN via IEEE.
        if (a is long la && b is long lb)
            return la == lb;
        return NumericValue(a) == NumericValue(b);
    }

    /// <summary>Extracts a normalized number for Python-style comparisons.</summary>
    /// <param name="value">Normalized Boolean, integer, or double scalar.</param>
    /// <returns>The numeric value, with bool mapped to zero or one.</returns>
    /// <exception cref="System.InvalidOperationException">The internal value is not bool, long, or double.</exception>
    private static double NumericValue(object value) => value switch
    {
        bool b => b ? 1.0 : 0.0,
        long l => l,
        double d => d,
        _ => throw new InvalidOperationException(),
    };

    /// <summary>Compares categorical values using Python scalar equality.</summary>
    /// <param name="obj">Object to compare with this instance.</param>
    /// <returns>True for equal normalized values; NaN remains unequal to itself.</returns>
    /// <exception cref="NumSharp.TypeError">An object operand has an unsupported categorical type or an unsigned value outside Int64.</exception>
    public override bool Equals(object? obj) => obj switch
    {
        CategoricalChoice choice => Equals(choice),
        null => IsNone,
        _ => Equals(From(obj)),
    };

    /// <summary>Hashes equal Python numeric values alike across Boolean, integer, and float kinds.</summary>
    /// <returns>Hash code consistent with supported categorical equality.</returns>
    public override int GetHashCode()
    {
        // Values equal under Python == must hash alike: hash all numerics by their double image
        // (Python: hash(True) == hash(1) == hash(1.0)).
        return _value switch
        {
            null => 0,
            string s => s.GetHashCode(),
            _ => NumericValue(_value).GetHashCode(),
        };
    }

    /// <summary>Applies the <c>==</c> comparison using the type's value equality.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator ==(CategoricalChoice left, CategoricalChoice right) => left.Equals(right);

    /// <summary>Applies the <c>!=</c> comparison using the type's value equality.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator !=(CategoricalChoice left, CategoricalChoice right) => !left.Equals(right);

    /// <summary>
    ///     Python <c>_categorical_choice_equal</c>: <c>==</c> except that two NaNs compare equal.
    /// </summary>
    /// <param name="value1">First categorical choice.</param>
    /// <param name="value2">Second categorical choice.</param>
    /// <returns>True for Python-equal choices or a pair of NaNs.</returns>
    internal static bool ChoicesEqual(CategoricalChoice value1, CategoricalChoice value2)
        => value1.Equals(value2) || (value1.IsNaN && value2.IsNaN);

    /// <summary>Python <c>str(value)</c> rendering (<c>None</c> / <c>True</c> / <c>1.0</c> / raw string).</summary>
    /// <returns>Python str-style text, leaving strings unquoted.</returns>
    public override string ToString() => PythonRepr.Str(_value);

    /// <summary>Python <c>repr(value)</c> rendering (strings quoted).</summary>
    /// <returns>Python repr-style text, quoting and escaping strings.</returns>
    public string Repr() => PythonRepr.Repr(_value);
}
