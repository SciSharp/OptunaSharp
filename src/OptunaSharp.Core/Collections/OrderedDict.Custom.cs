using System.Collections.Generic;

namespace OptunaSharp.Collections;

// Members shared across both target frameworks (net8.0 vendored impl + net9.0+ in-box subclass).
// These sit on OrderedDict regardless of which base it has, so OptunaSharp code sees one surface.
/// <summary>Insertion-order-preserving dictionary helpers shared by all target frameworks.</summary>
/// <typeparam name="TKey">The non-null key type compared by the dictionary comparer.</typeparam>
/// <typeparam name="TValue">The type of values stored with each key.</typeparam>
public sealed partial class OrderedDict<TKey, TValue>
{
    /// <summary>Python <c>dict.get(key, default)</c>.</summary>
    /// <param name="key">The non-null lookup key, compared with the dictionary's configured comparer.</param>
    /// <param name="defaultValue">The value returned when the key has no entry.</param>
    /// <returns>The stored value when the key exists, otherwise the supplied fallback.</returns>
    public TValue? GetValueOrDefault(TKey key, TValue? defaultValue = default)
        => TryGetValue(key, out var value) ? value : defaultValue;

    /// <summary>Shallow copy preserving insertion order (Python <c>copy.copy(dict)</c>).</summary>
    /// <returns>A new dictionary preserving insertion order and sharing the original key and value objects.</returns>
    public OrderedDict<TKey, TValue> ShallowClone() => new(this);
}
