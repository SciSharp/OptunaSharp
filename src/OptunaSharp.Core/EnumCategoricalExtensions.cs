using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Distributions;

namespace OptunaSharp;

/// <summary>
///     Typed read-back for enum-valued categorical parameters — the read side of the
///     <c>TEnum</c> lens (see <see cref="EnumCategorical" />). Because a categorical is persisted as a
///     bare member-name string with no type memory, the caller must name <c>TEnum</c>
///     at the moment of reading; resolution is therefore deferred to here and can fail if the stored
///     string is not a member of the requested enum.
/// </summary>
public static class EnumCategoricalExtensions
{
    /// <summary>
    ///     Reads a categorical parameter as <typeparamref name="TEnum" /> from any parameter view. A
    ///     single lens covers every read surface that exposes params this way —
    ///     <see cref="BaseTrial.Params" />, <see cref="FrozenTrial.Params" />, and
    ///     <see cref="Study.BestParams" /> alike.
    /// </summary>
    /// <typeparam name="TEnum">The enum the caller asserts this parameter was suggested from.</typeparam>
    /// <param name="parameters">A parameter dictionary mapping names to their external values.</param>
    /// <param name="name">The parameter name to read.</param>
    /// <returns>The stored member-name string resolved to its enum member.</returns>
    /// <exception cref="System.ArgumentNullException">The parameter dictionary is null.</exception>
    /// <exception cref="NumSharp.KeyError">No parameter with that name is present.</exception>
    /// <exception cref="NumSharp.ValueError">The stored string is not a declared member of <typeparamref name="TEnum" />.</exception>
    /// <exception cref="NumSharp.TypeError">The stored value is not a member-name string (e.g. the parameter is not categorical, or belongs to a different enum).</exception>
    public static TEnum GetCategorical<TEnum>(this IReadOnlyDictionary<string, object?> parameters, string name)
        where TEnum : struct, Enum
    {
        if (parameters is null)
            throw new ArgumentNullException(nameof(parameters));
        // Missing keys surface as KeyError to mirror Python's trial.params[name] KeyError, rather than
        // the CLR's KeyNotFoundException — keeps the failure vocabulary consistent with the rest of Core.
        if (!parameters.TryGetValue(name, out var value))
            throw new KeyError($"'{name}'");
        return EnumCategorical.Resolve<TEnum>(value);
    }

    /// <summary>Reads a categorical parameter of an active or replay trial as <typeparamref name="TEnum" />.</summary>
    /// <typeparam name="TEnum">The enum the caller asserts this parameter was suggested from.</typeparam>
    /// <param name="trial">The trial whose already-suggested parameters are read.</param>
    /// <param name="name">The parameter name to read.</param>
    /// <returns>The stored member-name string resolved to its enum member.</returns>
    /// <exception cref="System.ArgumentNullException">The trial is null.</exception>
    /// <exception cref="NumSharp.KeyError">No parameter with that name has been suggested.</exception>
    /// <exception cref="NumSharp.ValueError">The stored string is not a declared member of <typeparamref name="TEnum" />.</exception>
    /// <exception cref="NumSharp.TypeError">The stored value is not a member-name string.</exception>
    public static TEnum GetCategorical<TEnum>(this BaseTrial trial, string name)
        where TEnum : struct, Enum
        => (trial ?? throw new ArgumentNullException(nameof(trial))).Params.GetCategorical<TEnum>(name);
}
