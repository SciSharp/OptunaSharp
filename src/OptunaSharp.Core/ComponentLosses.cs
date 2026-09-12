using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using NumSharp;

namespace OptunaSharp;

/// <summary>
///     The <b>observed-components channel</b>: the convention by which an objective exposes a per-trial
///     residual/component vector <c>r = (r_1, …, r_m)</c> for an objective that decomposes as
///     <c>f(x) = mean_i r_i(x)</c>. It is carried on a trial as the <see cref="Key" /> user attribute so
///     that samplers which can exploit residual structure — <see cref="Samplers.JanusSampler" /> (the
///     paper's primary, recommended mode, where its Jacobian signal is sharpest) and the multi-objective
///     JANUS metric — read the <i>same</i> vector through one documented reader instead of re-deriving an
///     ad-hoc string-key reader each.
///
///     <para><b>Contract.</b> Components are per-trial terms of a mean decomposition, so all trials that
///     carry them should use a <i>consistent length</i> <c>m</c> across the study; the terms are treated
///     as <i>nonnegative</i> residual magnitudes (JANUS uses <c>√max(r, 0)</c>, so a negative entry is
///     permitted but read by the algorithm at its clamp of zero). A vector containing a non-finite value
///     (NaN / ±∞) is <b>rejected</b> — written vectors are validated up front, and a stored vector that
///     is empty, non-numeric, or non-finite reads back as "absent" so the consumer degrades gracefully
///     (JANUS falls back to its scalar residual-coordinate mode when not <i>all</i> trials in its window
///     carry valid components).</para>
///
///     <para><b>Wire key.</b> The underlying user-attr key is exactly <c>"component_losses"</c>
///     (<c>"rae"</c> is a recognized read alias), so an objective written against OptunaSharp runs
///     unchanged against the Python reference <c>janus_optuna</c>, and vice versa — the channel is pure
///     convention plus this reader/writer, storage-agnostic (it round-trips through the same user-attr
///     path RDB / journal storages already serialize).</para>
/// </summary>
/// <example>
///     <code>
///     study.Optimize(t =>
///     {
///         var r = new double[dim];
///         double f = 0;
///         for (int i = 0; i &lt; dim; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); r[i] = (x - 0.5) * (x - 0.5); f += r[i]; }
///         t.SetComponentLosses(r);   // expose the residual vector; JANUS fits J directly from it
///         return f / dim;
///     }, nTrials: 300);
///     </code>
/// </example>
public static class ComponentLosses
{
    /// <summary>
    ///     The user-attribute key that carries the component vector. Kept verbatim for objective
    ///     portability with the Python reference (see the type remarks).
    /// </summary>
    public const string Key = "component_losses";

    /// <summary>
    ///     A recognized read-only alias for <see cref="Key" /> (the reference accepts
    ///     <c>user_attrs["component_losses"] or user_attrs["rae"]</c>). Reading prefers <see cref="Key" />
    ///     and only falls back to this alias when the canonical value is absent or invalid.
    /// </summary>
    public const string LegacyKey = "rae";

    /// <summary>
    ///     Read the observed-components vector from a trial's user attributes. Tolerant of the value
    ///     being a <c>double[]</c> (the in-process form) or any numeric sequence (the JSON wire form that
    ///     storages deserialize to). Returns <c>false</c> — leaving <paramref name="components" /> empty —
    ///     when the attribute is absent, empty, non-numeric, contains <c>null</c>, or contains a
    ///     non-finite value; on success <paramref name="components" /> is a fresh, caller-owned copy.
    /// </summary>
    /// <param name="userAttrs">The trial's user-attribute dictionary (e.g. <see cref="FrozenTrial.UserAttrs" />).</param>
    /// <param name="components">The parsed residual vector, or an empty array when none is present.</param>
    /// <returns><c>true</c> if a valid, non-empty, all-finite vector was read.</returns>
    public static bool TryRead(IReadOnlyDictionary<string, object?> userAttrs, out double[] components)
    {
        components = Array.Empty<double>();
        if (userAttrs is null || userAttrs.Count == 0)
            return false;

        // Python `or` semantics: prefer the canonical key, fall through to the alias when the canonical
        // value yields no valid non-empty vector (mirrors the reference's
        // `t.user_attrs.get("component_losses") or t.user_attrs.get("rae")`).
        if (userAttrs.TryGetValue(Key, out var primary) && ToComponents(primary) is { } a)
        {
            components = a;
            return true;
        }
        if (userAttrs.TryGetValue(LegacyKey, out var legacy) && ToComponents(legacy) is { } b)
        {
            components = b;
            return true;
        }
        return false;
    }

    /// <summary>
    ///     Validate a to-be-written component vector against the channel contract and return an owned
    ///     copy suitable for <see cref="BaseTrial.SetUserAttr" />. Rejects an empty vector and any
    ///     non-finite entry (so a stored vector always reads back — a non-finite one never silently
    ///     becomes "absent"). Negative entries are accepted (read at their zero clamp downstream).
    /// </summary>
    /// <param name="components">Nonempty finite residual vector to persist.</param>
    /// <returns>A caller-owned copy of the validated component vector.</returns>
    /// <exception cref="NumSharp.ValueError">The vector is empty or contains NaN or infinity.</exception>
    internal static double[] Validate(ReadOnlySpan<double> components)
    {
        if (components.Length == 0)
            throw new ValueError("Attempted to set `component_losses`, but the vector is empty.");

        var copy = new double[components.Length];
        for (int i = 0; i < components.Length; i++)
        {
            double v = components[i];
            if (double.IsNaN(v) || double.IsInfinity(v))
                throw new ValueError(
                    "Attempted to set `component_losses`, but a non-finite value " +
                    $"({v.ToString("R", CultureInfo.InvariantCulture)}) is not allowed.");
            copy[i] = v;
        }
        return copy;
    }

    /// <summary>
    ///     Coerce a raw user-attr value to a validated component vector, or <c>null</c> when it is not a
    ///     usable residual vector (absent element, non-numeric, non-finite, or empty).
    /// </summary>
    /// <param name="raw">Raw user-attribute value, possibly a numeric sequence.</param>
    /// <returns>A fresh finite nonempty vector, or null for unsupported input.</returns>
    private static double[]? ToComponents(object? raw)
    {
        switch (raw)
        {
            case null:
                return null;
            case double[] da:
                return AllFinite(da) && da.Length > 0 ? (double[])da.Clone() : null;
            case string:
                // A string is IEnumerable<char> but never a component vector — reject explicitly.
                return null;
            case IEnumerable seq:
                var list = new List<double>();
                foreach (object? o in seq)
                {
                    if (o is null)
                        return null;
                    double v;
                    try
                    {
                        v = Convert.ToDouble(o, CultureInfo.InvariantCulture);
                    }
                    catch (Exception)
                    {
                        return null;
                    }
                    if (double.IsNaN(v) || double.IsInfinity(v))
                        return null;
                    list.Add(v);
                }
                return list.Count > 0 ? list.ToArray() : null;
            default:
                return null;
        }
    }

    /// <summary>Checks whether every component is neither NaN nor infinity.</summary>
    /// <param name="a">Array of component values to inspect.</param>
    /// <returns>True when every entry is finite; true for an empty array.</returns>
    private static bool AllFinite(double[] a)
    {
        foreach (double v in a)
            if (double.IsNaN(v) || double.IsInfinity(v))
                return false;
        return true;
    }
}
