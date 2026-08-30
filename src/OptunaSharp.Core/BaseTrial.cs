using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp;

/// <summary>
///     Base class for trials. Port of <c>optuna/trial/_base.py: BaseTrial</c>.
///
///     Note that this class is not supposed to be directly accessed by library users.
/// </summary>
public abstract class BaseTrial
{
    public abstract double SuggestFloat(string name, double low, double high, double? step = null, bool log = false);

    [Obsolete("Deprecated in v3.0.0. This feature will be removed in v6.0.0. Use SuggestFloat instead.")]
    public virtual double SuggestUniform(string name, double low, double high)
        => SuggestFloat(name, low, high);

    [Obsolete("Deprecated in v3.0.0. This feature will be removed in v6.0.0. Use SuggestFloat(..., log: true) instead.")]
    public virtual double SuggestLogUniform(string name, double low, double high)
        => SuggestFloat(name, low, high, log: true);

    [Obsolete("Deprecated in v3.0.0. This feature will be removed in v6.0.0. Use SuggestFloat(..., step: ...) instead.")]
    public virtual double SuggestDiscreteUniform(string name, double low, double high, double q)
        => SuggestFloat(name, low, high, step: q);

    public abstract long SuggestInt(string name, long low, long high, long step = 1, bool log = false);

    /// <summary>Suggest a value for the categorical parameter. The value is sampled from <paramref name="choices" />.</summary>
    public abstract CategoricalChoice SuggestCategorical(string name, IEnumerable<CategoricalChoice> choices);

    // Typed convenience overloads mirroring Python's @overload set: the return type follows the
    // element type of `choices`.

    public string SuggestCategorical(string name, IEnumerable<string> choices)
        => (string)SuggestCategorical(name, choices.Select(c => (CategoricalChoice)c)).Value!;

    public bool SuggestCategorical(string name, IEnumerable<bool> choices)
        => (bool)SuggestCategorical(name, choices.Select(c => (CategoricalChoice)c)).Value!;

    public long SuggestCategorical(string name, IEnumerable<int> choices)
        => (long)SuggestCategorical(name, choices.Select(c => (CategoricalChoice)c)).Value!;

    public long SuggestCategorical(string name, IEnumerable<long> choices)
        => (long)SuggestCategorical(name, choices.Select(c => (CategoricalChoice)c)).Value!;

    public double SuggestCategorical(string name, IEnumerable<double> choices)
        => (double)SuggestCategorical(name, choices.Select(c => (CategoricalChoice)c)).Value!;

    public abstract void Report(double value, int step);

    public abstract bool ShouldPrune();

    public abstract void SetUserAttr(string key, object? value);

    [Obsolete("Deprecated in v3.1.0. This feature will be removed in v5.0.0.")]
    public abstract void SetSystemAttr(string key, object? value);

    public abstract IReadOnlyDictionary<string, object?> Params { get; }

    public abstract IReadOnlyDictionary<string, BaseDistribution> Distributions { get; }

    public abstract IReadOnlyDictionary<string, object?> UserAttrs { get; }

    public abstract IReadOnlyDictionary<string, object?> SystemAttrs { get; }

    public abstract DateTime? DatetimeStart { get; }

    public abstract int Number { get; }

    public abstract OrderedDict<string, double> Constraints { get; }

    public abstract void SetConstraint(string key, double value);
}
