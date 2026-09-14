using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Distributions;

/// <summary>
///     Per-closed-generic reflection cache backing the <c>TEnum</c> categorical convenience layer.
///
///     A <see cref="CategoricalDistribution" /> can only carry Python scalars
///     (<see cref="CategoricalChoiceKind" />), so an enum participates in optimization only as the
///     <b>member-name string</b> — never its numeric value, and never a new wire type. This holder
///     computes the name/value pairing for a given <typeparamref name="TEnum" /> exactly once: the CLR
///     initializes one static instance per closed generic type, so the suggest/read paths pay no
///     runtime <see cref="Type" /> lookup and take no lock. It is the lock-free counterpart to the
///     boxed <see cref="ConcurrentDictionary{TKey,TValue}" /> path in <see cref="EnumCategorical" />,
///     which is used only where the concrete enum type is not a compile-time parameter.
/// </summary>
/// <typeparam name="TEnum">The enum whose declared members define the categorical choices.</typeparam>
internal static class EnumCategoricalCache<TEnum> where TEnum : struct, Enum
{
    /// <summary>
    ///     Declared member names in <see cref="Enum.GetNames{TEnum}" /> order — i.e. sorted by
    ///     underlying value, <b>not</b> source-declaration order. This is the order the choices are
    ///     stored in, so it fixes categorical indices; it is deterministic and stable for a given enum
    ///     (good for reproducibility), but a member list written out of value order is reordered here.
    ///     Pass an explicit choice sequence when a specific order or subset is required.
    /// </summary>
    internal static readonly string[] Names = Enum.GetNames<TEnum>();

    /// <summary>Declared member values, parallel to <see cref="Names" /> (same index ⇒ same member).</summary>
    internal static readonly TEnum[] Values = Enum.GetValues<TEnum>();

    /// <summary>
    ///     Member-name → value map for read-back, built from <see cref="Names" />/<see cref="Values" />.
    ///     Ordinal, case-sensitive, and restricted to <em>declared</em> names — so it rejects numeric
    ///     strings and <c>[Flags]</c> combinations that <see cref="Enum.Parse(Type,string)" /> would
    ///     otherwise accept. For aliased members (two names sharing one value) the first name in
    ///     <see cref="Names" /> wins.
    /// </summary>
    internal static readonly IReadOnlyDictionary<string, TEnum> ByName = BuildByName();

    /// <summary>Pairs each declared name with its value, keeping the first name for aliased values.</summary>
    /// <returns>An ordinal, case-sensitive name→value map over the declared members.</returns>
    private static Dictionary<string, TEnum> BuildByName()
    {
        var map = new Dictionary<string, TEnum>(Names.Length, StringComparer.Ordinal);
        for (int i = 0; i < Names.Length; i++)
            // TryAdd (not indexer) so the first alias name wins, matching the value→name direction.
            map.TryAdd(Names[i], Values[i]);
        return map;
    }

    /// <summary>Resolves a stored member-name string back to its enum value.</summary>
    /// <param name="name">A declared member name, exactly as <see cref="Enum.GetName{TEnum}" /> renders it.</param>
    /// <returns>The enum member with that name.</returns>
    /// <exception cref="NumSharp.ValueError">The string is not a declared member of <typeparamref name="TEnum" /> — e.g. it was written by a different enum, a plain-string suggest, or a member renamed since the trial was stored.</exception>
    internal static TEnum Parse(string name)
        => ByName.TryGetValue(name, out var value)
            ? value
            // The stored categorical is a bare string with no type memory, so a mismatch can only be
            // detected here, at the moment the caller names TEnum — surface the valid members to help.
            : throw new ValueError(
                $"'{name}' is not a declared member of enum '{typeof(TEnum).FullName}'. " +
                $"Members: {string.Join(", ", Names)}.");
}

/// <summary>
///     Bridges CLR enums to the string-only categorical machinery. An enum is convenience sugar over
///     a <see cref="CategoricalDistribution" /> whose choices are the member-name strings: the type is
///     <b>never persisted</b> — it is a stateless lens the caller re-supplies at each boundary
///     (<c>SuggestCategorical&lt;TEnum&gt;</c> on the way in, <c>GetCategorical&lt;TEnum&gt;</c> on the
///     way out). Consequences to know: (1) the store cannot enforce that the same enum is used to write
///     and read a parameter — that consistency lives in your source; (2) renaming a member changes the
///     emitted string, so trials stored before the rename no longer resolve. This keeps the wire,
///     storage, and samplers byte-identical to a hand-written string categorical.
/// </summary>
public static class EnumCategorical
{
    // Boxed value→name maps keyed by runtime Type — the one place the concrete enum is NOT a
    // compile-time parameter (NormalizeParam receives object?). Cold-path (enqueue / fixed-trial /
    // add-trial ingress), so a shared concurrent cache is ample; per-TEnum generic callers use the
    // lock-free EnumCategoricalCache<TEnum> instead.
    private static readonly ConcurrentDictionary<Type, IReadOnlyDictionary<object, string>> _nameByBoxedValue = new();

    /// <summary>
    ///     The declared members of <typeparamref name="TEnum" /> as categorical choices (member-name
    ///     strings). Use it to build a <see cref="OptunaSharp.Samplers.GridSampler" /> search space or
    ///     any other <c>IReadOnlyList&lt;CategoricalChoice&gt;</c> over an enum, since there is no
    ///     implicit enum→<see cref="CategoricalChoice" /> conversion.
    /// </summary>
    /// <typeparam name="TEnum">The enum whose declared members become the choices.</typeparam>
    /// <returns>A fresh array of the member names as choices, in <see cref="Enum.GetNames{TEnum}" /> (by-value) order.</returns>
    public static CategoricalChoice[] Choices<TEnum>() where TEnum : struct, Enum
    {
        var names = EnumCategoricalCache<TEnum>.Names;
        var choices = new CategoricalChoice[names.Length];
        for (int i = 0; i < names.Length; i++)
            choices[i] = names[i];   // implicit string → CategoricalChoice
        return choices;
    }

    /// <summary>
    ///     Resolves a stored categorical parameter value to <typeparamref name="TEnum" /> — the read
    ///     side of the lens. This is where a deferred, caller-supplied type is applied, so it is also
    ///     where a mismatch is detected and can fail.
    /// </summary>
    /// <typeparam name="TEnum">The enum the caller asserts this parameter was suggested from.</typeparam>
    /// <param name="value">The external parameter value: normally the member-name string, or a boxed <typeparamref name="TEnum" /> passed through unchanged.</param>
    /// <returns>The enum member the stored value names.</returns>
    /// <exception cref="NumSharp.ValueError">The string is not a declared member of <typeparamref name="TEnum" />.</exception>
    /// <exception cref="NumSharp.TypeError">The value is null, an enum of a different type, or a non-string scalar (e.g. the parameter is not a categorical, or belongs to a different enum).</exception>
    public static TEnum Resolve<TEnum>(object? value) where TEnum : struct, Enum
        => value switch
        {
            // Defensive: a caller-built FixedTrial could hand back a boxed TEnum verbatim.
            TEnum e => e,
            string s => EnumCategoricalCache<TEnum>.Parse(s),
            Enum other => throw new TypeError(
                $"Parameter value is a member of enum '{other.GetType().FullName}', not '{typeof(TEnum).FullName}'."),
            null => throw new TypeError(
                $"Parameter value is null and cannot be resolved as enum '{typeof(TEnum).FullName}'."),
            _ => throw new TypeError(
                $"Parameter value '{PythonRepr.Str(value)}' of type {value.GetType().Name} is not a " +
                $"member-name string of enum '{typeof(TEnum).FullName}'."),
        };

    /// <summary>
    ///     Converts a boxed enum to its member-name string, leaving every other value untouched. This
    ///     is the write-boundary normalizer: it runs at ingress points that take heterogeneous
    ///     <c>object?</c> parameter values (enqueue / fixed-trial), so a boxed enum becomes the same
    ///     string a <c>SuggestCategorical&lt;TEnum&gt;</c> would store <b>before</b> the value reaches
    ///     JSON serialization or <see cref="CategoricalChoice.From(object)" /> (which deliberately still
    ///     rejects enums, matching Optuna's typed-error contract).
    /// </summary>
    /// <param name="value">A parameter value that may or may not be a boxed enum.</param>
    /// <returns>The member-name string for a boxed enum; otherwise <paramref name="value" /> unchanged.</returns>
    /// <exception cref="System.ArgumentException">The enum value is not a single declared member (a <c>[Flags]</c> combination or an undefined numeric value), which has no categorical name.</exception>
    internal static object? NormalizeParam(object? value)
        => value is Enum e ? NameOf(e) : value;

    /// <summary>
    ///     Normalizes every boxed-enum value in a parameter dictionary to its member-name string,
    ///     returning the same instance when there is nothing to convert.
    /// </summary>
    /// <param name="params">The parameter dictionary supplied at a write boundary.</param>
    /// <returns>The original dictionary when it holds no enum values (allocation-free, identity preserved); otherwise a new dictionary with enum values replaced by their member-name strings and all other entries copied verbatim.</returns>
    /// <exception cref="System.ArgumentException">A value is an enum that is not a single declared member.</exception>
    internal static OrderedDict<string, object?> NormalizeParams(OrderedDict<string, object?> @params)
    {
        // Scan first so the common (enum-free) enqueue path allocates nothing and keeps the caller's
        // dictionary identity; only a dict that actually carries an enum is rebuilt.
        bool anyEnum = false;
        foreach (var value in @params.Values)
        {
            if (value is Enum)
            {
                anyEnum = true;
                break;
            }
        }

        if (!anyEnum)
            return @params;

        var normalized = new OrderedDict<string, object?>();
        foreach (var (key, value) in @params)
            normalized[key] = NormalizeParam(value);
        return normalized;
    }

    /// <summary>Maps a boxed enum value to its declared member name via the boxed-value cache.</summary>
    /// <param name="value">A boxed enum value; its runtime type selects the cached name map.</param>
    /// <returns>The declared member name for the value.</returns>
    /// <exception cref="System.ArgumentException">The value is not a single declared member (a <c>[Flags]</c> combination or an undefined numeric value).</exception>
    private static string NameOf(Enum value)
    {
        var map = _nameByBoxedValue.GetOrAdd(value.GetType(), BuildNameByBoxedValue);
        if (map.TryGetValue(value, out var name))
            return name;
        // Only single, declared members are categorical choices; combinations/undefined values have no
        // single name and would break round-tripping, so reject them at the boundary with a clear error.
        throw new ArgumentException(
            $"'{value}' is not a single declared member of enum '{value.GetType().FullName}'. " +
            "Only declared members are supported as categorical choices ([Flags] combinations and " +
            "undefined numeric values are not).",
            nameof(value));
    }

    /// <summary>Builds a boxed-value → name map for one enum type (declared members only).</summary>
    /// <param name="enumType">A runtime enum type to reflect over.</param>
    /// <returns>A map from each declared member's boxed value to its name; the first name wins for aliased values.</returns>
    /// <remarks>
    ///     Boxed enum values are valid dictionary keys: <see cref="Enum" /> overrides equality and hash
    ///     by underlying value, so a lookup with a boxed value from a different source still matches.
    /// </remarks>
    private static IReadOnlyDictionary<object, string> BuildNameByBoxedValue(Type enumType)
    {
        string[] names = Enum.GetNames(enumType);
        Array values = Enum.GetValues(enumType);
        var map = new Dictionary<object, string>(names.Length);
        for (int i = 0; i < names.Length; i++)
        {
            object boxed = values.GetValue(i)!;
            // First alias name wins, consistent with EnumCategoricalCache<TEnum>.ByName.
            if (!map.ContainsKey(boxed))
                map[boxed] = names[i];
        }
        return map;
    }
}
