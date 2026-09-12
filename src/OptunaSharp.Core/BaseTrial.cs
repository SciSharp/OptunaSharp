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
    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="low">Inclusive lower endpoint of the parameter range.</param>
    /// <param name="high">Inclusive upper endpoint; a stepped range may lower it to the last reachable grid point.</param>
    /// <param name="step">Positive float grid spacing, or null for a continuous range; cannot be combined with log.</param>
    /// <param name="log">Whether to sample in the logarithmic domain, which requires a strictly positive lower endpoint.</param>
    /// <summary>Suggests a floating-point parameter value for the objective.</summary>
    /// <returns>The suggested or previously recorded floating-point value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    /// <exception cref="System.FormatException">A stepped bound or step is nonfinite and cannot be parsed as a decimal.</exception>
    /// <exception cref="System.DivideByZeroException">The positive step rounds to zero in decimal arithmetic while aligning the upper endpoint.</exception>
    public abstract double SuggestFloat(string name, double low, double high, double? step = null, bool log = false);

    // Optuna's @deprecated_func warns (FutureWarning) before delegating to suggest_float; mirror
    // that here with the identical message (see DeprecationMessages). These are `virtual` with a
    // default body, so the runtime warning covers Trial / FrozenTrial / FixedTrial alike.
    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="low">Inclusive lower endpoint of the parameter range.</param>
    /// <param name="high">Inclusive upper endpoint; a stepped range may lower it to the last reachable grid point.</param>
    /// <summary>Uses the deprecated float-suggestion alias and emits a FutureWarning.</summary>
    /// <returns>The value returned by SuggestFloat.</returns>
    /// <exception cref="NumSharp.ValueError">The bounds or discretization are invalid, or the delegated suggestion is incompatible with the stored parameter.</exception>
    [Obsolete(DeprecationMessages.SuggestUniform)]
    public virtual double SuggestUniform(string name, double low, double high)
    {
        OptunaWarnings.Warn(DeprecationMessages.SuggestUniform, WarningCategory.FutureWarning);
        return SuggestFloat(name, low, high);
    }

    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="low">Inclusive lower endpoint of the parameter range.</param>
    /// <param name="high">Inclusive upper endpoint; a stepped range may lower it to the last reachable grid point.</param>
    /// <summary>Uses the deprecated float-suggestion alias and emits a FutureWarning.</summary>
    /// <returns>The value returned by SuggestFloat.</returns>
    /// <exception cref="NumSharp.ValueError">The bounds or discretization are invalid, or the delegated suggestion is incompatible with the stored parameter.</exception>
    [Obsolete(DeprecationMessages.SuggestLogUniform)]
    public virtual double SuggestLogUniform(string name, double low, double high)
    {
        OptunaWarnings.Warn(DeprecationMessages.SuggestLogUniform, WarningCategory.FutureWarning);
        return SuggestFloat(name, low, high, log: true);
    }

    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="low">Inclusive lower endpoint of the parameter range.</param>
    /// <param name="high">Inclusive upper endpoint; a stepped range may lower it to the last reachable grid point.</param>
    /// <summary>Uses the deprecated float-suggestion alias and emits a FutureWarning.</summary>
    /// <param name="q">Positive float discretization step.</param>
    /// <returns>The value returned by SuggestFloat.</returns>
    /// <exception cref="NumSharp.ValueError">The bounds or discretization are invalid, or the delegated suggestion is incompatible with the stored parameter.</exception>
    [Obsolete(DeprecationMessages.SuggestDiscreteUniform)]
    public virtual double SuggestDiscreteUniform(string name, double low, double high, double q)
    {
        OptunaWarnings.Warn(DeprecationMessages.SuggestDiscreteUniform, WarningCategory.FutureWarning);
        return SuggestFloat(name, low, high, step: q);
    }

    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="low">Inclusive lower endpoint of the parameter range.</param>
    /// <param name="high">Inclusive upper endpoint; a stepped range may lower it to the last reachable grid point.</param>
    /// <param name="step">Positive integer grid spacing; must be one for logarithmic sampling.</param>
    /// <param name="log">Whether to sample in the logarithmic domain, which requires a strictly positive lower endpoint.</param>
    /// <summary>Suggests a integer parameter value for the objective.</summary>
    /// <returns>The suggested or previously recorded integer value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    public abstract long SuggestInt(string name, long low, long high, long step = 1, bool log = false);

    /// <summary>Suggest a value for the categorical parameter. The value is sampled from <paramref name="choices" />.</summary>
    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="choices">Nonempty sequence of categorical candidates; order is retained and determines stored indices.</param>
    /// <returns>The suggested or previously recorded categorical value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    /// <exception cref="System.ArgumentNullException">The choices sequence is null.</exception>
    public abstract CategoricalChoice SuggestCategorical(string name, IEnumerable<CategoricalChoice> choices);

    // Typed convenience overloads mirroring Python's @overload set: the return type follows the
    // element type of `choices`.

    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="choices">Nonempty sequence of categorical candidates; order is retained and determines stored indices.</param>
    /// <summary>Suggests a categorical parameter value for the objective.</summary>
    /// <returns>The suggested or previously recorded categorical value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    /// <exception cref="System.ArgumentNullException">The choices sequence is null.</exception>
    /// <exception cref="System.InvalidCastException">The delegated suggestion has an underlying CLR type incompatible with this typed overload, for example when replaying a Python-equal value of a different scalar kind.</exception>
    public string SuggestCategorical(string name, IEnumerable<string> choices)
        => (string)SuggestCategorical(name, choices.Select(c => (CategoricalChoice)c)).Value!;

    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="choices">Nonempty sequence of categorical candidates; order is retained and determines stored indices.</param>
    /// <summary>Suggests a categorical parameter value for the objective.</summary>
    /// <returns>The suggested or previously recorded categorical value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    /// <exception cref="System.ArgumentNullException">The choices sequence is null.</exception>
    /// <exception cref="System.InvalidCastException">The delegated suggestion has an underlying CLR type incompatible with this typed overload, for example when replaying a Python-equal value of a different scalar kind.</exception>
    public bool SuggestCategorical(string name, IEnumerable<bool> choices)
        => (bool)SuggestCategorical(name, choices.Select(c => (CategoricalChoice)c)).Value!;

    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="choices">Nonempty sequence of categorical candidates; order is retained and determines stored indices.</param>
    /// <summary>Suggests a categorical parameter value for the objective.</summary>
    /// <returns>The suggested or previously recorded categorical value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    /// <exception cref="System.ArgumentNullException">The choices sequence is null.</exception>
    /// <exception cref="System.InvalidCastException">The delegated suggestion has an underlying CLR type incompatible with this typed overload, for example when replaying a Python-equal value of a different scalar kind.</exception>
    public long SuggestCategorical(string name, IEnumerable<int> choices)
        => (long)SuggestCategorical(name, choices.Select(c => (CategoricalChoice)c)).Value!;

    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="choices">Nonempty sequence of categorical candidates; order is retained and determines stored indices.</param>
    /// <summary>Suggests a categorical parameter value for the objective.</summary>
    /// <returns>The suggested or previously recorded categorical value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    /// <exception cref="System.ArgumentNullException">The choices sequence is null.</exception>
    /// <exception cref="System.InvalidCastException">The delegated suggestion has an underlying CLR type incompatible with this typed overload, for example when replaying a Python-equal value of a different scalar kind.</exception>
    public long SuggestCategorical(string name, IEnumerable<long> choices)
        => (long)SuggestCategorical(name, choices.Select(c => (CategoricalChoice)c)).Value!;

    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="choices">Nonempty sequence of categorical candidates; order is retained and determines stored indices.</param>
    /// <summary>Suggests a categorical parameter value for the objective.</summary>
    /// <returns>The suggested or previously recorded categorical value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    /// <exception cref="System.ArgumentNullException">The choices sequence is null.</exception>
    /// <exception cref="System.InvalidCastException">The delegated suggestion has an underlying CLR type incompatible with this typed overload, for example when replaying a Python-equal value of a different scalar kind.</exception>
    public double SuggestCategorical(string name, IEnumerable<double> choices)
        => (double)SuggestCategorical(name, choices.Select(c => (CategoricalChoice)c)).Value!;

    /// <param name="value">Intermediate objective value; ignored by replay trials.</param>
    /// <param name="step">Nonnegative reporting step; duplicate reports retain the first observation.</param>
    /// <summary>Records the first intermediate objective value for a nonnegative step.</summary>
    public abstract void Report(double value, int step);

    /// <summary>Asks the study pruner whether the current trial should stop early.</summary>
    /// <returns>True if the configured pruner recommends stopping the trial.</returns>
    public abstract bool ShouldPrune();

    /// <param name="key">Attribute name; an existing attribute is overwritten.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    /// <summary>Stores a user attribute by name.</summary>
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
    /// <exception cref="NumSharp.ValueError">The vector is empty or contains NaN or infinity.</exception>
    /// <exception cref="OptunaSharp.UpdateFinishedTrialError">An active storage-backed trial has already finished.</exception>
    public void SetComponentLosses(ReadOnlySpan<double> components)
        => SetUserAttr(ComponentLosses.Key, ComponentLosses.Validate(components));

    // Abstract, so the runtime FutureWarning lives in each concrete override (Trial / FrozenTrial /
    // FixedTrial), each of which Optuna decorates with @deprecated_func individually.
    /// <param name="key">Attribute name; an existing attribute is overwritten.</param>
    /// <param name="value">Attribute value; persistent backends require JSON-serializable data.</param>
    /// <summary>Stores a internal attribute by name.</summary>
    [Obsolete(DeprecationMessages.SetSystemAttr)]
    public abstract void SetSystemAttr(string key, object? value);

    /// <summary>Gets parameters already suggested through this trial, using their external values.</summary>
    public abstract IReadOnlyDictionary<string, object?> Params { get; }

    /// <summary>Gets the distribution recorded for each suggested parameter.</summary>
    public abstract IReadOnlyDictionary<string, BaseDistribution> Distributions { get; }

    /// <summary>Gets the user metadata attached to this trial.</summary>
    public abstract IReadOnlyDictionary<string, object?> UserAttrs { get; }

    /// <summary>Gets the internal metadata used by samplers and storage.</summary>
    public abstract IReadOnlyDictionary<string, object?> SystemAttrs { get; }

    /// <summary>Gets the trial creation or start time, or null for a waiting trial.</summary>
    public abstract DateTime? DatetimeStart { get; }

    /// <summary>Gets the zero-based trial number within the study or replay context.</summary>
    public abstract int Number { get; }

    /// <summary>Gets named constraint values; feasibility requires every value to be nonpositive.</summary>
    public abstract OrderedDict<string, double> Constraints { get; }

    /// <param name="key">Constraint name; later writes of the same name are ignored with a warning.</param>
    /// <param name="value">Constraint value; zero or negative is feasible and NaN is forbidden.</param>
    /// <summary>Records the first value of a named constraint; nonpositive values are feasible.</summary>
    /// <exception cref="NumSharp.ValueError">The constraint value is NaN.</exception>
    public abstract void SetConstraint(string key, double value);
}
