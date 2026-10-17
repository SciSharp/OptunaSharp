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

    /// <summary>
    ///     Suggest a value for the categorical parameter from a <b>prebuilt, reusable</b> distribution — the
    ///     allocation-free sibling of <see cref="SuggestCategorical(string, IEnumerable{CategoricalChoice})" />
    ///     for callers that suggest the same fixed domain on every trial (a GA genome does this for every gene
    ///     of every trial, where rebuilding a hundreds-of-choices distribution per call dominates the loop).
    /// </summary>
    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="distribution">
    ///     The reusable distribution. Treat it as immutable and shared: live trials store this very instance
    ///     into their record and the study's parameter registry (which is what makes the per-suggest
    ///     compatibility check O(1) by reference), so mutating it afterwards would corrupt recorded trials.
    /// </param>
    /// <returns>The suggested or previously recorded categorical value.</returns>
    /// <exception cref="NumSharp.ValueError">A required stored parameter is missing, or a repeated suggestion changes its distribution kind or categorical choices.</exception>
    /// <exception cref="System.ArgumentNullException">The distribution is null.</exception>
    /// <remarks>
    ///     The base implementation forwards to the sequence overload (a copy, but correct) so every
    ///     <see cref="BaseTrial" /> subclass — replay trials included — accepts it; <see cref="Trial" />
    ///     overrides it to reuse the instance without any copy.
    /// </remarks>
    public virtual CategoricalChoice SuggestCategorical(string name, CategoricalDistribution distribution)
    {
        if (distribution is null)
            throw new ArgumentNullException(nameof(distribution));
        return SuggestCategorical(name, distribution.Choices);
    }

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

    // Enum convenience overloads (beyond Optuna). A TEnum is optimized as its member-NAME strings: the
    // choices handed to the primitive are those names, so the emitted CategoricalDistribution — and
    // everything downstream (storage, samplers, JSON) — is byte-identical to hand-writing the names,
    // and the enum type is never persisted (it is re-supplied by the caller at read time). Non-virtual,
    // exactly like the scalar typed overloads above; the `struct, Enum` constraint keeps them from ever
    // binding for the string/bool/int/long/double overload sets.

    /// <summary>
    ///     Suggests a categorical parameter over <b>every declared member</b> of <typeparamref name="TEnum" />.
    ///     Choice order follows <see cref="System.Enum.GetNames{TEnum}" /> — sorted by underlying value,
    ///     <b>not</b> source-declaration order — which fixes the stored categorical indices; pass an
    ///     explicit sequence to the sibling overload when a specific order or a subset is required.
    /// </summary>
    /// <typeparam name="TEnum">The enum whose declared members are the choices.</typeparam>
    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <returns>The suggested (or previously recorded) enum member.</returns>
    /// <exception cref="NumSharp.ValueError">The enum has no members, a required stored parameter is missing, a repeated suggestion changes the choices, or a stored string is not a declared member of <typeparamref name="TEnum" />.</exception>
    /// <exception cref="NumSharp.TypeError">A stored categorical value is not a member-name string.</exception>
    public TEnum SuggestCategorical<TEnum>(string name) where TEnum : struct, Enum
        => EnumCategorical.Resolve<TEnum>(
            SuggestCategorical(name, EnumCategorical.Choices<TEnum>()).Value);

    /// <summary>
    ///     Suggests a categorical parameter over a chosen subset of <typeparamref name="TEnum" /> members,
    ///     preserving the given order (which determines the stored categorical indices).
    /// </summary>
    /// <typeparam name="TEnum">The enum whose members form the choices.</typeparam>
    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="choices">Nonempty sequence of members; order is retained and determines stored indices.</param>
    /// <returns>The suggested (or previously recorded) enum member.</returns>
    /// <exception cref="System.ArgumentNullException">The choices sequence is null.</exception>
    /// <exception cref="System.ArgumentException">A supplied member is not a single declared value (a <c>[Flags]</c> combination or an undefined value has no categorical name).</exception>
    /// <exception cref="NumSharp.ValueError">The choices are empty, a required stored parameter is missing, a repeated suggestion changes the choices, or a stored string is not among the given members.</exception>
    /// <exception cref="NumSharp.TypeError">A stored categorical value is not a member-name string.</exception>
    public TEnum SuggestCategorical<TEnum>(string name, IEnumerable<TEnum> choices) where TEnum : struct, Enum
    {
        if (choices is null)
            throw new ArgumentNullException(nameof(choices));
        // Enum.GetName<TEnum>(TEnum) resolves the member NAME with no boxing; null means the value is not
        // a single declared member (undefined/[Flags] combo) — reject it here rather than storing a
        // number-like or comma-joined string that could not round-trip back to a member.
        var names = choices.Select(c => (CategoricalChoice)(Enum.GetName(c)
            ?? throw new ArgumentException(
                $"'{c}' is not a single declared member of enum '{typeof(TEnum).FullName}'.", nameof(choices))));
        return EnumCategorical.Resolve<TEnum>(SuggestCategorical(name, names).Value);
    }

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

    // Optuna 5.0.0 (#6834) removed the abstract set_system_attr and system_attrs from BaseTrial (both had
    // been deprecated since 3.1.0 for removal in 5.0.0). System attributes are optuna-internal state:
    // samplers and storages read and write them through the storage API, and the read-only views that
    // remain (FrozenTrial.SystemAttrs, FixedTrial.SystemAttrs) are declared on those classes directly.

    /// <summary>Gets parameters already suggested through this trial, using their external values.</summary>
    public abstract IReadOnlyDictionary<string, object?> Params { get; }

    /// <summary>Gets the distribution recorded for each suggested parameter.</summary>
    public abstract IReadOnlyDictionary<string, BaseDistribution> Distributions { get; }

    /// <summary>Gets the user metadata attached to this trial.</summary>
    public abstract IReadOnlyDictionary<string, object?> UserAttrs { get; }

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
