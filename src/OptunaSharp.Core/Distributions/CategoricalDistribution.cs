using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Distributions;

/// <summary>
///     A categorical distribution. Port of <c>optuna/distributions.py: CategoricalDistribution</c>.
///
///     This object is instantiated by <see cref="Trial.SuggestCategorical(string, System.Collections.Generic.IEnumerable{CategoricalChoice})" />,
///     and passed to samplers in general.
/// </summary>
public class CategoricalDistribution : BaseDistribution
{
    /// <summary>Parameter value candidates (Python: an immutable tuple).</summary>
    public IReadOnlyList<CategoricalChoice> Choices { get; }

    /// <summary>Copies a nonempty candidate sequence into a categorical distribution.</summary>
    /// <param name="choices">Nonempty sequence of categorical candidates; order is retained and determines stored indices.</param>
    /// <exception cref="NumSharp.ValueError">No choices are supplied.</exception>
    /// <exception cref="System.ArgumentNullException">The choices sequence is null.</exception>
    public CategoricalDistribution(IEnumerable<CategoricalChoice> choices)
    {
        var materialized = choices.ToArray();
        if (materialized.Length == 0)
            throw new ValueError("The `choices` must contain one or more elements.");
        // The Python constructor warns when a choice is not None/bool/int/float/str; the
        // CategoricalChoice value type makes that state unrepresentable in C#.
        Choices = materialized;
    }

    /// <summary>Copies a nonempty candidate sequence into a categorical distribution.</summary>
    /// <param name="choices">Nonempty sequence of categorical candidates; order is retained and determines stored indices.</param>
    /// <exception cref="NumSharp.ValueError">No choices are supplied.</exception>
    /// <exception cref="System.ArgumentNullException">The choices sequence is null.</exception>
    public CategoricalDistribution(params CategoricalChoice[] choices)
        : this((IEnumerable<CategoricalChoice>)choices) { }

    /// <summary>External representation is the choice at the internal-repr index.</summary>
    /// <param name="paramValueInInternalRepr">Numeric internal representation: a numeric value or a zero-based categorical index.</param>
    /// <returns>The underlying value of the categorical choice at the converted index.</returns>
    /// <exception cref="System.ArgumentOutOfRangeException">The converted index lies outside the choices array.</exception>
    public override object? ToExternalRepr(double paramValueInInternalRepr)
        => Choices[(int)paramValueInInternalRepr].Value;

    /// <summary>The choice at the internal-repr index, typed.</summary>
    /// <param name="paramValueInInternalRepr">Numeric internal representation: a numeric value or a zero-based categorical index.</param>
    /// <returns>The categorical choice at the truncated internal index.</returns>
    /// <exception cref="System.ArgumentOutOfRangeException">The converted index lies outside the choices array.</exception>
    public CategoricalChoice ChoiceAt(double paramValueInInternalRepr)
        => Choices[(int)paramValueInInternalRepr];

    // Lazy first-index-by-choice map for ToInternalRepr. CategoricalChoice's == IS its Equals and its
    // GetHashCode is consistent with it (equal-under-== values hash alike), so a default-comparer
    // dictionary keyed on the FIRST occurrence of each choice returns exactly what the reference linear
    // scan returns — a hit is the same index, a miss falls through to the same fallback loops. Built on
    // first use so distributions that never convert externally pay nothing; volatile publish so a racing
    // reader either sees the finished map or builds its own identical one (idempotent).
    private volatile Dictionary<CategoricalChoice, int>? _indexByChoice;

    /// <summary>Converts an external value into the numeric representation used by samplers.</summary>
    /// <param name="paramValueInExternalRepr">External parameter value to convert to the distribution's numeric representation.</param>
    /// <returns>Index of the first matching choice, treating two NaNs as matching in the fallback comparison.</returns>
    /// <exception cref="NumSharp.ValueError">The external value does not match any candidate under categorical choice equality.</exception>
    /// <exception cref="NumSharp.TypeError">The external value is not a supported categorical scalar.</exception>
    public override double ToInternalRepr(object? paramValueInExternalRepr)
    {
        var value = CategoricalChoice.From(paramValueInExternalRepr);
        // NOTE(nabenabe): With this implementation, we cannot distinguish some values
        // such as True and 1, or 1.0 and 1. For example, if choices=[True, 1] and external_repr
        // is 1, this method wrongly returns 0 instead of 1. However, we decided to accept this
        // bug for such exceptional choices for less complexity and faster processing.
        //
        // O(1) fast path replacing the reference O(n) scan bit-for-bit (see _indexByChoice): a GA genome
        // suggests every gene of every trial through here, so with hundreds of choices this scan was a
        // dominant per-trial cost.
        Dictionary<CategoricalChoice, int>? map = _indexByChoice;
        if (map is null)
        {
            map = new Dictionary<CategoricalChoice, int>(Choices.Count);
            for (int i = 0; i < Choices.Count; i++)
                map.TryAdd(Choices[i], i);   // first occurrence wins, like the scan
            _indexByChoice = map;
        }
        if (map.TryGetValue(value, out int found))
            return found;
        for (int index = 0; index < Choices.Count; index++)
        {
            if (Choices[index] == value)
                return index;
        }
        // Python's fallback loop additionally treats NaN == NaN (and handles float precision).
        for (int index = 0; index < Choices.Count; index++)
        {
            if (CategoricalChoice.ChoicesEqual(value, Choices[index]))
                return index;
        }

        throw new ValueError(
            $"'{PythonRepr.Str(paramValueInExternalRepr is CategoricalChoice c ? c.Value : paramValueInExternalRepr)}' " +
            $"not in {PythonRepr.TupleRepr(Choices.Select(ch => ch.Value))}.");
    }

    /// <summary>Tests whether this distribution admits only one candidate.</summary>
    /// <returns>True when sampling can be replaced by the sole allowed value.</returns>
    public override bool Single() => Choices.Count == 1;

    /// <summary>Tests membership using the stored numeric representation.</summary>
    /// <param name="paramValueInInternalRepr">Numeric internal representation: a numeric value or a zero-based categorical index.</param>
    /// <returns>True when the value belongs to the range or categorical index domain.</returns>
    internal override bool Contains(double paramValueInInternalRepr)
    {
        int index = (int)paramValueInInternalRepr;
        return 0 <= index && index < Choices.Count;
    }

    /// <summary>Builds the attribute payload used by Optuna JSON serialization.</summary>
    /// <returns>A new ordered attribute dictionary; values preserve their distribution representation.</returns>
    internal override OrderedDict<string, object?> AsDict() => new()
    {
        ["choices"] = Choices,
    };

    /// <summary>Compares distribution types and their defining attributes.</summary>
    /// <param name="other">Value to compare with this instance; null compares unequal.</param>
    /// <returns>True when the distributions have compatible types and equal attributes.</returns>
    public override bool Equals(BaseDistribution? other)
    {
        // Same instance ⇒ trivially equal — pure shortcut, no semantic change. Callers that reuse one
        // distribution instance per parameter (a GA genome suggesting the same domains every trial) turn
        // the per-suggest compatibility check from an O(choices) pairwise compare into O(1).
        if (ReferenceEquals(this, other))
            return true;
        // Python: isinstance check (a subclass compares equal to the base by attribute), then
        // choices compared pairwise with _categorical_choice_equal.
        if (other is not CategoricalDistribution o || !GetType().IsInstanceOfType(other) && !other.GetType().IsInstanceOfType(this))
            return false;
        if (Choices.Count != o.Choices.Count)
            return false;
        for (int i = 0; i < Choices.Count; i++)
        {
            if (!CategoricalChoice.ChoicesEqual(Choices[i], o.Choices[i]))
                return false;
        }
        return true;
    }

    /// <summary>Computes a hash from the distribution type and defining attributes.</summary>
    /// <returns>Hash code consistent with distribution equality.</returns>
    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(typeof(CategoricalDistribution));
        foreach (var choice in Choices)
            hash.Add(choice);
        return hash.ToHashCode();
    }
}
