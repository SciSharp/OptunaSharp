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

    public CategoricalDistribution(IEnumerable<CategoricalChoice> choices)
    {
        var materialized = choices.ToArray();
        if (materialized.Length == 0)
            throw new ValueError("The `choices` must contain one or more elements.");
        // The Python constructor warns when a choice is not None/bool/int/float/str; the
        // CategoricalChoice value type makes that state unrepresentable in C#.
        Choices = materialized;
    }

    public CategoricalDistribution(params CategoricalChoice[] choices)
        : this((IEnumerable<CategoricalChoice>)choices) { }

    /// <summary>External representation is the choice at the internal-repr index.</summary>
    public override object? ToExternalRepr(double paramValueInInternalRepr)
        => Choices[(int)paramValueInInternalRepr].Value;

    /// <summary>The choice at the internal-repr index, typed.</summary>
    public CategoricalChoice ChoiceAt(double paramValueInInternalRepr)
        => Choices[(int)paramValueInInternalRepr];

    public override double ToInternalRepr(object? paramValueInExternalRepr)
    {
        var value = CategoricalChoice.From(paramValueInExternalRepr);
        // NOTE(nabenabe): With this implementation, we cannot distinguish some values
        // such as True and 1, or 1.0 and 1. For example, if choices=[True, 1] and external_repr
        // is 1, this method wrongly returns 0 instead of 1. However, we decided to accept this
        // bug for such exceptional choices for less complexity and faster processing.
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

    public override bool Single() => Choices.Count == 1;

    internal override bool Contains(double paramValueInInternalRepr)
    {
        int index = (int)paramValueInInternalRepr;
        return 0 <= index && index < Choices.Count;
    }

    internal override OrderedDict<string, object?> AsDict() => new()
    {
        ["choices"] = Choices,
    };

    public override bool Equals(BaseDistribution? other)
    {
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

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(typeof(CategoricalDistribution));
        foreach (var choice in Choices)
            hash.Add(choice);
        return hash.ToHashCode();
    }
}
