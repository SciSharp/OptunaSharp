#if NET9_0_OR_GREATER
using System.Collections.Generic;

namespace OptunaSharp.Collections;

/// <summary>
///     An insertion-order-preserving dictionary with Python <c>dict</c> semantics.
///
///     Optuna relies on Python dicts preserving insertion order in many load-bearing places —
///     search-space construction, parameter/distribution maps, transform bounds layout, JSON
///     attribute serialization — so OptunaSharp uses this collection wherever Optuna uses
///     <c>dict</c>. <see cref="System.Collections.Generic.Dictionary{TKey,TValue}" /> does not
///     guarantee enumeration order, which would silently break parity.
///
///     On .NET 9+ this is a thin subclass of the in-box
///     <see cref="System.Collections.Generic.OrderedDictionary{TKey,TValue}" /> (added in .NET 9),
///     inheriting its entries-array + hash-bucket implementation. On net8.0 — where the in-box type
///     does not exist — <c>OrderedDict</c> is instead the vendored dotnet/runtime source of that same
///     type (see <c>OrderedDict.Net8.cs</c>), so the behavior is identical across target frameworks.
///     The shared members (<see cref="GetValueOrDefault" />, <see cref="ShallowClone" />) live
///     alongside on both frameworks (see <c>OrderedDict.Custom.cs</c>).
/// </summary>
/// <typeparam name="TKey">The non-null key type compared by the dictionary comparer.</typeparam>
/// <typeparam name="TValue">The type of values stored with each key.</typeparam>
public sealed partial class OrderedDict<TKey, TValue> : System.Collections.Generic.OrderedDictionary<TKey, TValue>
    where TKey : notnull
{
    /// <summary>Creates an insertion-ordered dictionary with the requested initial storage capacity when supplied.</summary>
    public OrderedDict()
    {
    }

    /// <summary>Creates an insertion-ordered dictionary with the requested initial storage capacity when supplied.</summary>
    /// <param name="capacity">The initial number of entries for which storage should be reserved.</param>
    public OrderedDict(int capacity) : base(capacity)
    {
    }

    /// <summary>Shallow-copies another dictionary preserving its enumeration order (Python <c>copy.copy(dict)</c>).</summary>
    /// <remarks>
    ///     Populates via the indexer so duplicate keys follow Python <c>dict(pairs)</c> semantics
    ///     (last value wins, first-occurrence position); the in-box <c>IEnumerable</c> constructor
    ///     throws on duplicate keys instead.
    /// </remarks>
    /// <param name="source">The source entries copied in enumeration order; repeated keys replace values without changing the first key position.</param>
    public OrderedDict(IEnumerable<KeyValuePair<TKey, TValue>> source)
    {
        foreach (var kv in source)
            this[kv.Key] = kv.Value;
    }
}
#endif
