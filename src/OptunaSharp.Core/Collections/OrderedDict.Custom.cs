using System.Collections.Generic;

namespace OptunaSharp.Collections;

// Members shared across both target frameworks (net8.0 vendored impl + net9.0+ in-box subclass).
// These sit on OrderedDict regardless of which base it has, so OptunaSharp code sees one surface.
public sealed partial class OrderedDict<TKey, TValue>
{
    /// <summary>Python <c>dict.get(key, default)</c>.</summary>
    public TValue? GetValueOrDefault(TKey key, TValue? defaultValue = default)
        => TryGetValue(key, out var value) ? value : defaultValue;

    /// <summary>Shallow copy preserving insertion order (Python <c>copy.copy(dict)</c>).</summary>
    public OrderedDict<TKey, TValue> ShallowClone() => new(this);
}
