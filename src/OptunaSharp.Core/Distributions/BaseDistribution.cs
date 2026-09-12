using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Distributions;

/// <summary>
///     Base class for distributions. Port of <c>optuna/distributions.py: BaseDistribution</c>.
///
///     Note that distribution classes are not supposed to be called by library users.
///     They are used by <see cref="Trial" /> and samplers internally.
/// </summary>
public abstract class BaseDistribution : IEquatable<BaseDistribution>
{
    /// <summary>
    ///     Convert internal representation of a parameter value into external representation.
    /// </summary>
    /// <param name="paramValueInInternalRepr">Optuna's internal representation of a parameter value.</param>
    /// <returns>Optuna's external representation of a parameter value.</returns>
    public virtual object? ToExternalRepr(double paramValueInInternalRepr)
        => paramValueInInternalRepr;

    /// <summary>
    ///     Convert external representation of a parameter value into internal representation.
    /// </summary>
    /// <param name="paramValueInExternalRepr">Optuna's external representation of a parameter value.</param>
    /// <returns>Optuna's internal representation of a parameter value.</returns>
    /// <exception cref="NumSharp.ValueError">The value cannot be converted, is NaN for a numeric distribution, is nonpositive in a logarithmic distribution, or is not a categorical candidate.</exception>
    public abstract double ToInternalRepr(object? paramValueInExternalRepr);

    /// <summary>
    ///     Test whether the range of this distribution contains just a single value.
    /// </summary>
    /// <returns>True when sampling can be replaced by the sole allowed value.</returns>
    public abstract bool Single();

    /// <summary>
    ///     Test if a parameter value is contained in the range of this distribution.
    ///     (Python: <c>_contains</c>.)
    /// </summary>
    /// <param name="paramValueInInternalRepr">Numeric internal representation: a numeric value or a zero-based categorical index.</param>
    /// <returns>True when the value belongs to the range or categorical index domain.</returns>
    internal abstract bool Contains(double paramValueInInternalRepr);

    /// <summary>
    ///     The attribute view serialized to JSON and shown in <c>repr</c> (Python: <c>_asdict</c>).
    ///     Key insertion order matches the Python class's <c>__dict__</c> exactly — it is
    ///     observable in <c>distribution_to_json</c> output.
    /// </summary>
    /// <returns>A new ordered attribute dictionary; values preserve their distribution representation.</returns>
    internal abstract OrderedDict<string, object?> AsDict();

    /// <summary>Python <c>__eq__</c>: same concrete type and equal attributes.</summary>
    /// <param name="other">Value to compare with this instance; null compares unequal.</param>
    /// <returns>True when the distributions have compatible types and equal attributes.</returns>
    public abstract bool Equals(BaseDistribution? other);

    /// <summary>Compares distribution types and their defining attributes.</summary>
    /// <param name="obj">Object to compare with this instance.</param>
    /// <returns>True when the distributions have compatible types and equal attributes.</returns>
    public override bool Equals(object? obj) => obj is BaseDistribution other && Equals(other);

    /// <summary>Computes a hash from the distribution type and defining attributes.</summary>
    /// <returns>Hash code consistent with distribution equality.</returns>
    public abstract override int GetHashCode();

    /// <summary>Applies the <c>==</c> comparison using the type's value equality.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator ==(BaseDistribution? left, BaseDistribution? right)
        => left is null ? right is null : left.Equals(right);

    /// <summary>Applies the <c>!=</c> comparison using the type's value equality.</summary>
    /// <param name="left">Left operand of the comparison.</param>
    /// <param name="right">Right operand of the comparison.</param>
    /// <returns>True when the comparison holds.</returns>
    public static bool operator !=(BaseDistribution? left, BaseDistribution? right) => !(left == right);

    /// <summary>
    ///     Python <c>__repr__</c>: <c>ClassName(key1=value1, key2=value2)</c> with keys sorted.
    /// </summary>
    /// <returns>Python-style distribution representation with attribute names sorted ordinally.</returns>
    public override string ToString()
    {
        var sb = new StringBuilder(GetType().Name).Append('(');
        bool first = true;
        foreach (var kv in AsDict().OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            if (!first)
                sb.Append(", ");
            sb.Append(kv.Key).Append('=').Append(ReprAttr(kv.Value));
            first = false;
        }
        return sb.Append(')').ToString();
    }

    /// <summary>Formats a distribution attribute with Python representation rules.</summary>
    /// <param name="value">Attribute value to format.</param>
    /// <returns>Python representation, using tuple syntax for categorical choices.</returns>
    private static string ReprAttr(object? value)
        => value is IReadOnlyList<CategoricalChoice> choices
            ? PythonRepr.TupleRepr(choices.Select(c => c.Value))
            : PythonRepr.Repr(value);
}
