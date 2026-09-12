using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp;

/// <summary>
///     Base class for trials. Port of <c>optuna/trial/_base.py: BaseTrial</c>.
///
///     Note that this class is not supposed to be directly accessed by library users.
/// </summary>
public abstract class BaseTrial
{
    public abstract double SuggestFloat(string name, double low, double high, double? step = null, bool log = false);

    // Optuna's @deprecated_func warns (FutureWarning) before delegating to suggest_float; mirror
    // that here with the identical message (see DeprecationMessages). These are `virtual` with a
    // default body, so the runtime warning covers Trial / FrozenTrial / FixedTrial alike.
    [Obsolete(DeprecationMessages.SuggestUniform)]
    public virtual double SuggestUniform(string name, double low, double high)
    {
        OptunaWarnings.Warn(DeprecationMessages.SuggestUniform, WarningCategory.FutureWarning);
        return SuggestFloat(name, low, high);
    }

    [Obsolete(DeprecationMessages.SuggestLogUniform)]
    public virtual double SuggestLogUniform(string name, double low, double high)
    {
        OptunaWarnings.Warn(DeprecationMessages.SuggestLogUniform, WarningCategory.FutureWarning);
        return SuggestFloat(name, low, high, log: true);
    }

    [Obsolete(DeprecationMessages.SuggestDiscreteUniform)]
    public virtual double SuggestDiscreteUniform(string name, double low, double high, double q)
    {
        OptunaWarnings.Warn(DeprecationMessages.SuggestDiscreteUniform, WarningCategory.FutureWarning);
        return SuggestFloat(name, low, high, step: q);
    }

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

    /// <summary>
    ///     Expose this trial's observed-components vector — the per-trial residual terms
    ///     <c>r = (r_1, …, r_m)</c> of an objective that decomposes as <c>f(x) = mean_i r_i(x)</c> — on
    ///     the <see cref="ComponentLosses.Key" /> user attribute. Samplers that exploit residual
    ///     structure (notably <see cref="Samplers.JanusSampler" /> in its recommended component mode)
    ///     read it back through <see cref="FrozenTrial.TryGetComponentLosses" />. The vector must be
    ///     non-empty and all-finite (a non-finite entry raises <c>ValueError</c>); see
    ///     <see cref="ComponentLosses" /> for the full contract.
    /// </summary>
    /// <param name="components">The residual vector; use a length consistent across trials.</param>
    public void SetComponentLosses(ReadOnlySpan<double> components)
        => SetUserAttr(ComponentLosses.Key, ComponentLosses.Validate(components));

    // Abstract, so the runtime FutureWarning lives in each concrete override (Trial / FrozenTrial /
    // FixedTrial), each of which Optuna decorates with @deprecated_func individually.
    [Obsolete(DeprecationMessages.SetSystemAttr)]
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
