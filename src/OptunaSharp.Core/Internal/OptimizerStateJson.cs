using System.Text.Json;
using System.Text.Json.Serialization;

namespace OptunaSharp.Internal;

/// <summary>
///     The single System.Text.Json configuration for every sampler's persisted optimizer state — the
///     tagged JSON that CMA-ES (all three variants), LM-CMA, VkD-CMA, DE, PSO and the OpenAI-ES / SNES /
///     xNES samplers store in trial system attributes where Optuna stores its pickled optimizer.
/// </summary>
/// <remarks>
///     <para><b>Why named floating-point literals.</b> Optuna pickles its optimizers, and pickle stores
///     <c>inf</c>, <c>-inf</c> and <c>nan</c> like any other float. System.Text.Json's default options
///     refuse them: writing a state that holds one throws <see cref="System.ArgumentException" />
///     (".NET number values such as positive and negative infinity cannot be written as valid JSON").
///     A single ±inf objective value — routine for a diverged training run — reaches CMA-ES's
///     <c>funhist</c> and the fitness arrays of the population samplers, so before this type existed every
///     later ask of such a study failed. <see cref="JsonNumberHandling.AllowNamedFloatingPointLiterals" />
///     writes those values as the JSON strings <c>"Infinity"</c>, <c>"-Infinity"</c> and <c>"NaN"</c>
///     and reads them back to the same <see cref="double" />; finite numbers are written exactly as the
///     default options write them.</para>
///     <para><b>Compatibility.</b> Reading still accepts plain JSON numbers, and property names and shapes
///     are untouched, so a state stored by an earlier OptunaSharp version (which could only ever hold
///     finite values) restores unchanged. The output never contains a newline, which the samplers rely on
///     to split their <c>&lt;tag&gt;\n&lt;json&gt;</c> payloads.</para>
///     <para>Footgun: route every optimizer-state write and read through this type. A call site that falls
///     back to <c>JsonSerializer</c>'s default options reintroduces the ±inf crash for that sampler only,
///     and nothing else fails until a study produces an infinite value.</para>
/// </remarks>
internal static class OptimizerStateJson
{
    // One shared, immutable instance: JsonSerializerOptions caches type metadata per instance, so a fresh
    // instance per call would rebuild the converters on every trial.
    private static readonly JsonSerializerOptions Options = new()
    {
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    /// <summary>Serializes an optimizer-state DTO, encoding non-finite doubles as named literals.</summary>
    /// <typeparam name="T">The optimizer's state DTO type (for example <c>CmaState</c>).</typeparam>
    /// <param name="state">The state snapshot to persist; non-finite values are allowed anywhere in it.</param>
    /// <returns>Single-line JSON text for the snapshot.</returns>
    /// <exception cref="System.NotSupportedException"><typeparamref name="T" /> contains a member type System.Text.Json cannot serialize.</exception>
    internal static string Serialize<T>(T state) => JsonSerializer.Serialize(state, Options);

    /// <summary>Deserializes an optimizer-state DTO written by <see cref="Serialize{T}" /> or by an earlier OptunaSharp version.</summary>
    /// <typeparam name="T">The optimizer's state DTO type the payload was written from.</typeparam>
    /// <param name="json">The JSON text after the sampler's <c>&lt;tag&gt;\n</c> prefix.</param>
    /// <returns>The reconstructed, non-null state snapshot.</returns>
    /// <exception cref="JsonException">The text is not valid JSON for <typeparamref name="T" />, or it is the JSON literal <c>null</c>.</exception>
    /// <exception cref="System.NotSupportedException"><typeparamref name="T" /> contains a member type System.Text.Json cannot deserialize.</exception>
    internal static T Deserialize<T>(string json)
        // A bare `null` payload can only come from a corrupted or foreign attribute; failing here with a
        // JsonException beats the NullReferenceException the restore code would otherwise hit later.
        => JsonSerializer.Deserialize<T>(json, Options)
           ?? throw new JsonException($"The persisted {typeof(T).Name} is the JSON literal null.");
}
