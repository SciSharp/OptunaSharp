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
    public abstract double ToInternalRepr(object? paramValueInExternalRepr);

    /// <summary>
    ///     Test whether the range of this distribution contains just a single value.
    /// </summary>
    public abstract bool Single();

    /// <summary>
    ///     Test if a parameter value is contained in the range of this distribution.
    ///     (Python: <c>_contains</c>.)
    /// </summary>
    internal abstract bool Contains(double paramValueInInternalRepr);

    /// <summary>
    ///     The attribute view serialized to JSON and shown in <c>repr</c> (Python: <c>_asdict</c>).
    ///     Key insertion order matches the Python class's <c>__dict__</c> exactly — it is
    ///     observable in <c>distribution_to_json</c> output.
    /// </summary>
    internal abstract OrderedDict<string, object?> AsDict();

    /// <summary>Python <c>__eq__</c>: same concrete type and equal attributes.</summary>
    public abstract bool Equals(BaseDistribution? other);

    public override bool Equals(object? obj) => obj is BaseDistribution other && Equals(other);

    public abstract override int GetHashCode();

    public static bool operator ==(BaseDistribution? left, BaseDistribution? right)
        => left is null ? right is null : left.Equals(right);

    public static bool operator !=(BaseDistribution? left, BaseDistribution? right) => !(left == right);

    /// <summary>
    ///     Python <c>__repr__</c>: <c>ClassName(key1=value1, key2=value2)</c> with keys sorted.
    /// </summary>
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

    private static string ReprAttr(object? value)
        => value is IReadOnlyList<CategoricalChoice> choices
            ? PythonRepr.TupleRepr(choices.Select(c => c.Value))
            : PythonRepr.Repr(value);
}
