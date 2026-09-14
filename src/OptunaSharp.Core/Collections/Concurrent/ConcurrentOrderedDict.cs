#nullable enable

using System;
using System.Collections;
using System.Collections.Generic;
using OptunaSharp.Collections.Concurrent;

namespace OptunaSharp.Collections;

/// <summary>
///     A thread-safe collection with <b>two access paths over one set of entries</b>: a dictionary path keyed
///     by <typeparamref name="TKey" /> (uniqueness, dedup and O(1) lookup) and a list path addressed by
///     insertion-order <c>int</c> index. Enumeration, <see cref="ToArray" /> and <see cref="Values" /> yield the
///     <typeparamref name="TValue" />s in index order, so the type reads like a <see cref="List{T}" /> of values
///     that also happens to be keyed. It is the concurrent counterpart of <see cref="OrderedDict{TKey,TValue}" />.
/// </summary>
/// <remarks>
///     <para>
///         <b>Why this exists.</b> OptunaSharp needs a concurrent, insertion-ordered, index-addressable map:
///         a <see cref="System.Collections.Concurrent.ConcurrentDictionary{TKey,TValue}" /> gives thread-safe
///         key access but no order and no positional access, while <see cref="OrderedDict{TKey,TValue}" /> gives
///         order but is not thread-safe. This type composes the two: the key path is a vendored clone of dotnet's
///         <see cref="Concurrent.ConcurrentDictionary{TKey,TValue}" /> (striped locks, lock-free reads); the list
///         path is an immutable, <b>contiguous</b> value/key snapshot published through a single volatile field.
///     </para>
///     <para>
///         <b>How the two paths are wired.</b> The key path maps each key to a stable <see cref="Entry" /> handle
///         that holds the value — so a key lookup is a single lock-free hop and never depends on the entry's
///         current position. The list path is a pair of contiguous arrays (values and keys, in order); enumeration,
///         <see cref="ToArray" />, <see cref="this[int]" /> and <see cref="CopyTo" /> scan them directly, which is
///         what keeps them close to <see cref="List{T}" />. A write updates both representations, so both paths
///         observe the same value.
///     </para>
///     <para>
///         <b>The <c>int</c>-key footgun.</b> Both <see cref="this[TKey]" /> and <see cref="this[int]" /> exist.
///         When <typeparamref name="TKey" /> is <c>int</c> they collide: the C# compiler binds <c>d[5]</c> to the
///         <i>index</i> indexer (the concrete <c>int</c> overload wins over the generic one), so the key path for
///         <c>int</c> keys must go through methods (<see cref="TryGetValue" />, <see cref="GetByKey" />,
///         <see cref="SetByKey" />). For every other key type the two brackets disambiguate by argument type.
///     </para>
///     <para>
///         <b>Performance.</b> Reads are lock-free: lookup by key, membership, <see cref="IndexOf" />, positional
///         get, <see cref="Count" />, enumeration and <see cref="ToArray" /> take no lock. Key lookups match or
///         beat <see cref="System.Collections.Concurrent.ConcurrentDictionary{TKey,TValue}" />; enumeration,
///         <see cref="ToArray" /> and index get scan a contiguous array and stay close to <see cref="List{T}" />.
///         Insertion is amortized O(1). <b>Writes are serialized</b> by a single lock (unlike the striped write
///         locks of the plain concurrent dictionary) because a global insertion order and the contiguous cache
///         cannot be maintained per-stripe; that is the deliberate cost of the ordering. <b>Order-preserving
///         removal is O(n)</b> (like <see cref="List{T}.RemoveAt" />): it shifts the trailing values down to keep
///         indices contiguous. <b>Replacing the value of an existing key is O(1) when <typeparamref name="TValue" />
///         is written atomically</b> (reference types and small primitives) and O(n) otherwise (a non-atomically
///         writable value type forces a fresh value array so lock-free readers never see a torn value).
///     </para>
///     <para>
///         <b>Memory.</b> Each value is held twice — once in its <see cref="Entry" /> (for the lock-free key path)
///         and once in the contiguous cache (for the List-like list path) — plus one small <see cref="Entry" />
///         object per key. This denormalization is what buys both a consistent O(1) key path and contiguous
///         enumeration at once.
///     </para>
///     <para>
///         <b>Consistency.</b> A lock-free reader always sees valid, untorn values and never corrupts state. In the
///         absence of concurrent <i>writers</i> the key and list paths are exactly consistent. While a writer runs,
///         a reader may briefly observe a just-added element through one path a hair before the other, an
///         <see cref="IndexOf" /> off by concurrent removals, or (for atomically-written values) a value update in
///         one path before the other — snapshot semantics, in the spirit of the plain concurrent dictionary. An
///         enumerator captures the value array as it was when enumeration began.
///     </para>
/// </remarks>
/// <typeparam name="TKey">The non-null key type; uniqueness and lookups use the configured comparer.</typeparam>
/// <typeparam name="TValue">The value type stored per key and yielded, in index order, by enumeration.</typeparam>
public sealed class ConcurrentOrderedDict<TKey, TValue> : IReadOnlyList<TValue>
    where TKey : notnull
{
    /// <summary>The initial ordered-array capacity, and the size the arrays first grow to from empty.</summary>
    /// <remarks>Deliberately small (unlike the key hashtable's prime capacity) because many ordered maps stay tiny.</remarks>
    private const int DefaultCapacity = 4;

    /// <summary>
    ///     The stable, authoritative handle for the key path: it holds the value and the entry's current
    ///     insertion-order position, and its identity never changes while the key is present, so a key lookup
    ///     returns a consistent value without depending on where the entry currently sits.
    /// </summary>
    /// <remarks>
    ///     A reference type on purpose: swapping the whole <see cref="Entry" /> (an atomic reference write) is how a
    ///     value is replaced tear-free on the key path when <typeparamref name="TValue" /> cannot be written
    ///     atomically. <see cref="_value" /> is the source of truth for the key path; the list path keeps its own
    ///     contiguous copy (see <see cref="Store" />) that a write keeps in sync.
    /// </remarks>
    private sealed class Entry
    {
        /// <summary>The key this entry was added under; immutable for the entry's lifetime.</summary>
        internal readonly TKey _key;

        /// <summary>The value on the key path. Mutated in place for atomically-writable value types; otherwise the whole entry is swapped.</summary>
        internal TValue _value;

        /// <summary>The entry's current position in insertion order; rewritten under the write lock when removals shift it.</summary>
        internal int _index;

        /// <summary>Creates an entry binding a key to a value at a position.</summary>
        /// <param name="key">The entry's immutable key.</param>
        /// <param name="value">The initial value.</param>
        /// <param name="index">The entry's initial insertion-order position.</param>
        internal Entry(TKey key, TValue value, int index)
        {
            _key = key;
            _value = value;
            _index = index;
        }
    }

    /// <summary>
    ///     An immutable snapshot of the list path: contiguous keys and values in insertion order plus the live
    ///     count. Published atomically through the single volatile <see cref="_store" /> field so a lock-free
    ///     reader gets a consistent (arrays, count) triple from one read, and can scan the values like a
    ///     <see cref="List{T}" />.
    /// </summary>
    /// <remarks>
    ///     Only <c>[0, <see cref="_count" />)</c> of each array is valid; the tail may hold spare capacity or stale
    ///     data and must never be read. The fields are <c>readonly</c> and set before the instance is published, so
    ///     the publication is safe without further barriers.
    /// </remarks>
    private sealed class Store
    {
        /// <summary>Keys in insertion order, occupying <c>[0, <see cref="_count" />)</c>. Parallel to <see cref="_values" />.</summary>
        internal readonly TKey[] _keys;

        /// <summary>Values in insertion order, occupying <c>[0, <see cref="_count" />)</c>. The contiguous store that makes the list path fast.</summary>
        internal readonly TValue[] _values;

        /// <summary>The number of live entries — the collection's <see cref="Count" /> when this snapshot was published.</summary>
        internal readonly int _count;

        /// <summary>Wraps parallel key/value arrays and their live count into a publishable snapshot.</summary>
        /// <param name="keys">The ordered keys (only the first <paramref name="count" /> slots are meaningful).</param>
        /// <param name="values">The ordered values (only the first <paramref name="count" /> slots are meaningful).</param>
        /// <param name="count">The number of live entries.</param>
        internal Store(TKey[] keys, TValue[] values, int count)
        {
            _keys = keys;
            _values = values;
            _count = count;
        }

        /// <summary>The shared empty snapshot used before the first insertion and after <see cref="Clear" />.</summary>
        internal static readonly Store Empty = new(Array.Empty<TKey>(), Array.Empty<TValue>(), 0);
    }

    /// <summary>
    ///     The key path: maps each key to its stable <see cref="Entry" />. A vendored clone of dotnet's thread-safe
    ///     dictionary, giving lock-free O(1) key reads and enforcing key uniqueness/dedup.
    /// </summary>
    private readonly ConcurrentDictionary<TKey, Entry> _byKey;

    /// <summary>The list path: the current contiguous ordered snapshot. Volatile so structural writes publish atomically to lock-free readers.</summary>
    private volatile Store _store;

    /// <summary>
    ///     Serializes every mutation so the global insertion order, the contiguous cache and the key map stay
    ///     consistent. Reads never acquire it.
    /// </summary>
    private readonly object _writeLock = new();

    /// <summary>Creates an empty collection using the default comparer for <typeparamref name="TKey" />.</summary>
    public ConcurrentOrderedDict() : this((IEqualityComparer<TKey>?)null)
    {
    }

    /// <summary>Creates an empty collection using the supplied key comparer.</summary>
    /// <param name="comparer">The comparer used for key uniqueness and lookups, or <see langword="null" /> for the default.</param>
    public ConcurrentOrderedDict(IEqualityComparer<TKey>? comparer)
    {
        _byKey = new ConcurrentDictionary<TKey, Entry>(comparer);
        _store = Store.Empty;
    }

    /// <summary>Creates an empty collection with room reserved for <paramref name="capacity" /> entries.</summary>
    /// <param name="capacity">The number of entries to pre-size both paths for; must be non-negative.</param>
    /// <param name="comparer">The comparer used for key uniqueness and lookups, or <see langword="null" /> for the default.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="capacity" /> is negative.</exception>
    public ConcurrentOrderedDict(int capacity, IEqualityComparer<TKey>? comparer = null)
    {
        if (capacity < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity));
        }

        // Match the key hashtable's initial capacity to the request, and pre-size the ordered arrays so the first
        // several inserts don't reallocate.
        _byKey = new ConcurrentDictionary<TKey, Entry>(Environment.ProcessorCount, capacity, comparer);
        _store = capacity == 0 ? Store.Empty : new Store(new TKey[capacity], new TValue[capacity], 0);
    }

    /// <summary>
    ///     Creates a collection populated from <paramref name="source" /> in enumeration order, with Python
    ///     <c>dict(pairs)</c> duplicate semantics (last value wins, first-occurrence position) — matching
    ///     <see cref="OrderedDict{TKey,TValue}" />, not the throw-on-duplicate behavior of the framework dictionary.
    /// </summary>
    /// <param name="source">The key/value pairs to copy; repeated keys replace the value without moving the key.</param>
    /// <param name="comparer">The comparer used for key uniqueness and lookups, or <see langword="null" /> for the default.</param>
    /// <exception cref="ArgumentNullException"><paramref name="source" /> is <see langword="null" />.</exception>
    public ConcurrentOrderedDict(IEnumerable<KeyValuePair<TKey, TValue>> source, IEqualityComparer<TKey>? comparer = null)
        : this(comparer)
    {
        if (source is null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        foreach (var kv in source)
        {
            SetByKey(kv.Key, kv.Value);
        }
    }

    /// <summary>Gets the number of entries — a moment-in-time snapshot, consistent with lock-free enumeration.</summary>
    public int Count => _store._count;

    /// <summary>Gets whether the collection currently has no entries.</summary>
    public bool IsEmpty => _store._count == 0;

    /// <summary>Gets the comparer used to determine key equality and hashing.</summary>
    public IEqualityComparer<TKey> Comparer => _byKey.Comparer;

    // ---------------------------------------------------------------------------------------------------------
    // Key path (dictionary semantics). For int keys, reach these through the named methods, not the indexer.
    // ---------------------------------------------------------------------------------------------------------

    /// <summary>
    ///     Gets or sets the value for <paramref name="key" />. Get throws if the key is absent; set updates the
    ///     value in place when the key exists, otherwise appends a new entry at the end of insertion order.
    /// </summary>
    /// <param name="key">The key to read or write.</param>
    /// <value>The value associated with <paramref name="key" />.</value>
    /// <remarks>When <typeparamref name="TKey" /> is <c>int</c> this indexer is shadowed by <see cref="this[int]" />; use <see cref="GetByKey" />/<see cref="SetByKey" /> instead.</remarks>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="KeyNotFoundException">The get accessor is used and <paramref name="key" /> is absent.</exception>
    public TValue this[TKey key]
    {
        get => GetByKey(key);
        set => SetByKey(key, value);
    }

    /// <summary>Reads the value for <paramref name="key" />, throwing if absent — the method form of the key getter (use this for <c>int</c> keys). Lock-free and O(1).</summary>
    /// <param name="key">The key to look up.</param>
    /// <returns>The value associated with <paramref name="key" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="KeyNotFoundException"><paramref name="key" /> is absent.</exception>
    public TValue GetByKey(TKey key)
    {
        if (!TryGetValue(key, out var value))
        {
            throw new KeyNotFoundException(Concurrent.SR.Format(Concurrent.SR.Arg_KeyNotFoundWithKey, key.ToString()));
        }

        return value;
    }

    /// <summary>
    ///     Upserts <paramref name="key" />: updates its value if present (keeping its position), otherwise appends
    ///     a new entry — the method form of the key setter (use this for <c>int</c> keys). Serialized with other writes.
    /// </summary>
    /// <param name="key">The key to write.</param>
    /// <param name="value">The value to store.</param>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public void SetByKey(TKey key, TValue value)
    {
        NullCheck(key);

        lock (_writeLock)
        {
            if (_byKey.TryGetValue(key, out var e))
            {
                ReplaceExistingUnderLock(e, value);
            }
            else
            {
                AppendUnderLock(key, value);
            }
        }
    }

    /// <summary>Determines whether <paramref name="key" /> is present. Lock-free and O(1).</summary>
    /// <param name="key">The key to test.</param>
    /// <returns><see langword="true" /> if the key exists; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public bool ContainsKey(TKey key) => _byKey.ContainsKey(key);

    /// <summary>Attempts to read the value for <paramref name="key" /> without throwing. Lock-free and O(1).</summary>
    /// <param name="key">The key to look up.</param>
    /// <param name="value">On success, the associated value; otherwise the default of <typeparamref name="TValue" />.</param>
    /// <returns><see langword="true" /> if the key was found; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public bool TryGetValue(TKey key, out TValue value)
    {
        if (_byKey.TryGetValue(key, out var e))
        {
            value = e._value;
            return true;
        }

        value = default!;
        return false;
    }

    /// <summary>Adds <paramref name="key" />/<paramref name="value" /> only if the key is absent, appending it at the end of order.</summary>
    /// <param name="key">The key to add.</param>
    /// <param name="value">The value to add.</param>
    /// <returns><see langword="true" /> if the entry was added; <see langword="false" /> if the key already existed.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public bool TryAdd(TKey key, TValue value)
    {
        NullCheck(key);

        if (_byKey.ContainsKey(key))
        {
            return false; // lock-free fast negative
        }

        lock (_writeLock)
        {
            if (_byKey.ContainsKey(key))
            {
                return false; // another thread won the race while we waited for the lock
            }

            AppendUnderLock(key, value);
            return true;
        }
    }

    /// <summary>Adds <paramref name="key" />/<paramref name="value" />, throwing if the key already exists.</summary>
    /// <param name="key">The key to add.</param>
    /// <param name="value">The value to add.</param>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    /// <exception cref="ArgumentException">An entry with the same key already exists.</exception>
    public void Add(TKey key, TValue value)
    {
        if (!TryAdd(key, value))
        {
            throw new ArgumentException(Concurrent.SR.ConcurrentDictionary_KeyAlreadyExisted, nameof(key));
        }
    }

    /// <summary>Returns the existing value for <paramref name="key" />, or appends and returns <paramref name="value" /> if absent.</summary>
    /// <param name="key">The key to read or add.</param>
    /// <param name="value">The value to append if the key is absent.</param>
    /// <returns>The existing value, or the newly added <paramref name="value" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public TValue GetOrAdd(TKey key, TValue value)
    {
        NullCheck(key);

        if (_byKey.TryGetValue(key, out var e))
        {
            return e._value;
        }

        lock (_writeLock)
        {
            if (_byKey.TryGetValue(key, out e))
            {
                return e._value;
            }

            AppendUnderLock(key, value);
            return value;
        }
    }

    /// <summary>Returns the existing value for <paramref name="key" />, or appends the result of <paramref name="valueFactory" /> if absent.</summary>
    /// <param name="key">The key to read or add.</param>
    /// <param name="valueFactory">Invoked with the key to produce the value when the key is absent; runs under the write lock, so keep it fast and non-reentrant.</param>
    /// <returns>The existing value, or the newly produced and added value.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> or <paramref name="valueFactory" /> is <see langword="null" />.</exception>
    public TValue GetOrAdd(TKey key, Func<TKey, TValue> valueFactory)
    {
        NullCheck(key);
        if (valueFactory is null)
        {
            throw new ArgumentNullException(nameof(valueFactory));
        }

        if (_byKey.TryGetValue(key, out var e))
        {
            return e._value;
        }

        lock (_writeLock)
        {
            if (_byKey.TryGetValue(key, out e))
            {
                return e._value;
            }

            TValue value = valueFactory(key);
            AppendUnderLock(key, value);
            return value;
        }
    }

    /// <summary>Appends <paramref name="addValue" /> if the key is absent, otherwise replaces the value via <paramref name="updateValueFactory" />.</summary>
    /// <param name="key">The key to add or update.</param>
    /// <param name="addValue">The value to append if the key is absent.</param>
    /// <param name="updateValueFactory">Invoked with the key and current value to produce the replacement; runs under the write lock.</param>
    /// <returns>The value now stored for the key (the added or the updated value).</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> or <paramref name="updateValueFactory" /> is <see langword="null" />.</exception>
    public TValue AddOrUpdate(TKey key, TValue addValue, Func<TKey, TValue, TValue> updateValueFactory)
    {
        NullCheck(key);
        if (updateValueFactory is null)
        {
            throw new ArgumentNullException(nameof(updateValueFactory));
        }

        lock (_writeLock)
        {
            if (_byKey.TryGetValue(key, out var e))
            {
                TValue updated = updateValueFactory(key, e._value);
                ReplaceExistingUnderLock(e, updated);
                return updated;
            }

            AppendUnderLock(key, addValue);
            return addValue;
        }
    }

    /// <summary>Replaces the value for <paramref name="key" /> only if its current value equals <paramref name="comparisonValue" />.</summary>
    /// <param name="key">The key whose value is conditionally replaced.</param>
    /// <param name="newValue">The replacement value.</param>
    /// <param name="comparisonValue">The value the current value must equal (by the default value comparer) for the update to occur.</param>
    /// <returns><see langword="true" /> if the value was replaced; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public bool TryUpdate(TKey key, TValue newValue, TValue comparisonValue)
    {
        NullCheck(key);

        lock (_writeLock)
        {
            if (_byKey.TryGetValue(key, out var e) &&
                EqualityComparer<TValue>.Default.Equals(e._value, comparisonValue))
            {
                ReplaceExistingUnderLock(e, newValue);
                return true;
            }

            return false;
        }
    }

    /// <summary>
    ///     Removes <paramref name="key" /> and its value, shifting later entries down to keep indices contiguous.
    ///     <b>O(n)</b> in the number of entries after the removed one.
    /// </summary>
    /// <param name="key">The key to remove.</param>
    /// <param name="value">On success, the removed value; otherwise the default of <typeparamref name="TValue" />.</param>
    /// <returns><see langword="true" /> if an entry was removed; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public bool TryRemove(TKey key, out TValue value)
    {
        NullCheck(key);

        if (!_byKey.ContainsKey(key))
        {
            value = default!;
            return false; // lock-free fast negative
        }

        lock (_writeLock)
        {
            if (!_byKey.TryGetValue(key, out var e))
            {
                value = default!;
                return false;
            }

            value = e._value;
            RemoveCoreUnderLock(e);
            return true;
        }
    }

    /// <summary>Removes <paramref name="key" /> if present. Convenience wrapper over <see cref="TryRemove" />; O(n) like it.</summary>
    /// <param name="key">The key to remove.</param>
    /// <returns><see langword="true" /> if an entry was removed; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public bool Remove(TKey key) => TryRemove(key, out _);

    // ---------------------------------------------------------------------------------------------------------
    // List path (index semantics). Enumeration and ToArray are here too, yielding values in index order.
    // ---------------------------------------------------------------------------------------------------------

    /// <summary>
    ///     Gets or sets the value at insertion-order position <paramref name="index" />. The setter takes a
    ///     <typeparamref name="TValue" /> and replaces the value at that position, leaving its key and position
    ///     unchanged (it never inserts). This indexer wins over <see cref="this[TKey]" /> when <typeparamref name="TKey" /> is <c>int</c>.
    /// </summary>
    /// <param name="index">The zero-based position in insertion order.</param>
    /// <value>The value at <paramref name="index" />.</value>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative or not less than <see cref="Count" />.</exception>
    public TValue this[int index]
    {
        get
        {
            Store s = _store;
            if ((uint)index >= (uint)s._count)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            return s._values[index];
        }
        set => SetAt(index, value);
    }

    /// <summary>Replaces the value at position <paramref name="index" /> without changing its key or position. Serialized with other writes.</summary>
    /// <param name="index">The zero-based position in insertion order.</param>
    /// <param name="value">The replacement value.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative or not less than <see cref="Count" />.</exception>
    public void SetAt(int index, TValue value)
    {
        lock (_writeLock)
        {
            Store s = _store;
            if ((uint)index >= (uint)s._count)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            TKey key = s._keys[index];
            _byKey.TryGetValue(key, out var e); // e is non-null under the lock; the key path holds this position

            if (ConcurrentDictionaryTypeProps<TValue>.IsWriteAtomic)
            {
                // In-place atomic writes; lock-free readers see the old or new value, never a torn one.
                s._values[index] = value;
                if (e is not null)
                {
                    e._value = value;
                }
            }
            else
            {
                // Non-atomic value: swap the key-path entry and publish a fresh value array so no reader tears.
                if (e is not null)
                {
                    _byKey[key] = new Entry(key, value, index);
                }

                var values = (TValue[])s._values.Clone();
                values[index] = value;
                _store = new Store(s._keys, values, s._count);
            }
        }
    }

    /// <summary>Attempts to read the value at position <paramref name="index" /> without throwing. Lock-free and O(1).</summary>
    /// <param name="index">The zero-based position in insertion order.</param>
    /// <param name="value">On success, the value at the position; otherwise the default of <typeparamref name="TValue" />.</param>
    /// <returns><see langword="true" /> if <paramref name="index" /> is in range; otherwise <see langword="false" />.</returns>
    public bool TryGetAt(int index, out TValue value)
    {
        Store s = _store;
        if ((uint)index >= (uint)s._count)
        {
            value = default!;
            return false;
        }

        value = s._values[index];
        return true;
    }

    /// <summary>Returns the key at position <paramref name="index" /> in insertion order. Lock-free and O(1).</summary>
    /// <param name="index">The zero-based position in insertion order.</param>
    /// <returns>The key stored at that position.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative or not less than <see cref="Count" />.</exception>
    public TKey GetKeyAt(int index)
    {
        Store s = _store;
        if ((uint)index >= (uint)s._count)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        return s._keys[index];
    }

    /// <summary>
    ///     Returns the insertion-order position of <paramref name="key" />, or -1 if absent — the "TKey to index"
    ///     mapping. Lock-free and O(1).
    /// </summary>
    /// <param name="key">The key to locate.</param>
    /// <returns>The zero-based index of the key, or -1 if it is not present.</returns>
    /// <remarks>Under a concurrent structural writer the returned index is a snapshot and may be stale by the number of concurrent removals.</remarks>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public int IndexOf(TKey key) => _byKey.TryGetValue(key, out var e) ? e._index : -1;

    /// <summary>Attempts to map <paramref name="key" /> to its insertion-order position without allocating a sentinel.</summary>
    /// <param name="key">The key to locate.</param>
    /// <param name="index">On success, the key's index; otherwise -1.</param>
    /// <returns><see langword="true" /> if the key was found; otherwise <see langword="false" />.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    public bool TryGetIndex(TKey key, out int index)
    {
        if (_byKey.TryGetValue(key, out var e))
        {
            index = e._index;
            return true;
        }

        index = -1;
        return false;
    }

    /// <summary>Removes the entry at position <paramref name="index" />, shifting later entries down. <b>O(n).</b></summary>
    /// <param name="index">The zero-based position in insertion order to remove.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative or not less than <see cref="Count" />.</exception>
    public void RemoveAt(int index)
    {
        lock (_writeLock)
        {
            Store s = _store;
            if ((uint)index >= (uint)s._count)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            if (_byKey.TryGetValue(s._keys[index], out var e))
            {
                RemoveCoreUnderLock(e);
            }
        }
    }

    /// <summary>Removes all entries and resets both paths to empty.</summary>
    public void Clear()
    {
        lock (_writeLock)
        {
            _byKey.Clear();
            _store = Store.Empty;
        }
    }

    /// <summary>
    ///     Copies the values, in index order, into a new array — the same content and order as enumeration, close
    ///     to <see cref="List{T}.ToArray" /> in cost (a contiguous block copy).
    /// </summary>
    /// <returns>A new array of the values in insertion order (a point-in-time snapshot).</returns>
    public TValue[] ToArray()
    {
        Store s = _store;
        int n = s._count;
        if (n == 0)
        {
            return Array.Empty<TValue>();
        }

        var array = new TValue[n];
        Array.Copy(s._values, array, n);
        return array;
    }

    /// <summary>Copies the values, in index order, into <paramref name="array" /> starting at <paramref name="arrayIndex" /> (a contiguous block copy).</summary>
    /// <param name="array">The destination array.</param>
    /// <param name="arrayIndex">The zero-based position in <paramref name="array" /> at which to start writing.</param>
    /// <exception cref="ArgumentNullException"><paramref name="array" /> is <see langword="null" />.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="arrayIndex" /> is negative.</exception>
    /// <exception cref="ArgumentException"><paramref name="array" /> has too little room from <paramref name="arrayIndex" />.</exception>
    public void CopyTo(TValue[] array, int arrayIndex)
    {
        if (array is null)
        {
            throw new ArgumentNullException(nameof(array));
        }

        if (arrayIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(arrayIndex));
        }

        Store s = _store;
        int n = s._count;
        if (array.Length - arrayIndex < n)
        {
            throw new ArgumentException(Concurrent.SR.ConcurrentDictionary_ArrayNotLargeEnough, nameof(array));
        }

        Array.Copy(s._values, 0, array, arrayIndex, n);
    }

    /// <summary>
    ///     Gets a point-in-time array of the keys in insertion order (a contiguous block copy). A fresh array each
    ///     call (snapshot semantics), parallel to <see cref="Values" />.
    /// </summary>
    public TKey[] Keys
    {
        get
        {
            Store s = _store;
            int n = s._count;
            if (n == 0)
            {
                return Array.Empty<TKey>();
            }

            var keys = new TKey[n];
            Array.Copy(s._keys, keys, n);
            return keys;
        }
    }

    /// <summary>Gets a point-in-time array of the values in insertion order (equivalent to <see cref="ToArray" />).</summary>
    public TValue[] Values => ToArray();

    /// <summary>
    ///     Enumerates the entries as key/value pairs in insertion order — the dictionary-shaped view, for when the
    ///     value-only default enumeration is not enough. A snapshot is taken when enumeration begins.
    /// </summary>
    public IEnumerable<KeyValuePair<TKey, TValue>> Pairs
    {
        get
        {
            Store s = _store;
            TKey[] keys = s._keys;
            TValue[] values = s._values;
            int n = s._count;
            for (int i = 0; i < n; i++)
            {
                yield return new KeyValuePair<TKey, TValue>(keys[i], values[i]);
            }
        }
    }

    /// <summary>Returns an allocation-free enumerator over the values in insertion order (snapshotted at this call).</summary>
    /// <returns>A struct enumerator; each iteration yields the next value in index order.</returns>
    public Enumerator GetEnumerator() => new(this);

    /// <inheritdoc />
    IEnumerator<TValue> IEnumerable<TValue>.GetEnumerator() => GetEnumerator();

    /// <inheritdoc />
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>
    ///     A forward, value-order enumerator bound to the contiguous value array captured when it was created; it
    ///     scans that array like a <see cref="List{T}" /> enumerator, observes a stable count, and never throws for
    ///     concurrent modification.
    /// </summary>
    public struct Enumerator : IEnumerator<TValue>
    {
        private readonly TValue[] _values;
        private readonly int _count;
        private int _i;
        private TValue _current;

        /// <summary>Binds the enumerator to a snapshot of the owner's current ordered store.</summary>
        /// <param name="owner">The collection to snapshot and enumerate.</param>
        internal Enumerator(ConcurrentOrderedDict<TKey, TValue> owner)
        {
            Store store = owner._store;
            _values = store._values;
            _count = store._count;
            _i = -1;
            _current = default!;
        }

        /// <summary>Gets the value at the enumerator's current position.</summary>
        public TValue Current => _current;

        /// <inheritdoc />
        object? IEnumerator.Current => _current;

        /// <summary>Advances to the next value in index order.</summary>
        /// <returns><see langword="true" /> if a value is now current; <see langword="false" /> once the snapshot is exhausted.</returns>
        public bool MoveNext()
        {
            int next = _i + 1;
            if (next < _count)
            {
                _current = _values[next];
                _i = next;
                return true;
            }

            _current = default!;
            return false;
        }

        /// <summary>Resets the enumerator to before the first value of its snapshot.</summary>
        public void Reset()
        {
            _i = -1;
            _current = default!;
        }

        /// <summary>No-op; the enumerator holds no unmanaged resources.</summary>
        public void Dispose()
        {
        }
    }

    // ---------------------------------------------------------------------------------------------------------
    // Private write helpers. All callers hold _writeLock.
    // ---------------------------------------------------------------------------------------------------------

    /// <summary>
    ///     Appends a brand-new entry (caller has verified the key is absent) at the end of insertion order, growing
    ///     the ordered arrays by doubling when full, then publishing the new snapshot. Amortized O(1).
    /// </summary>
    /// <param name="key">The absent key to add.</param>
    /// <param name="value">The value to store.</param>
    /// <remarks>The caller must hold <see cref="_writeLock" />.</remarks>
    private void AppendUnderLock(TKey key, TValue value)
    {
        Store s = _store;
        int idx = s._count;
        TKey[] keys = s._keys;
        TValue[] values = s._values;

        if (idx == values.Length)
        {
            // Grow by doubling. Old readers keep the old arrays; only new readers (after publication) see the new ones.
            int newCap = values.Length == 0 ? DefaultCapacity : values.Length * 2;
            var grownKeys = new TKey[newCap];
            var grownValues = new TValue[newCap];
            Array.Copy(keys, grownKeys, idx);
            Array.Copy(values, grownValues, idx);
            keys = grownKeys;
            values = grownValues;
        }

        keys[idx] = key;
        values[idx] = value; // written before publication; old snapshots never read slot idx (their count is idx)

        // Make the key findable (its value is already valid via the entry), then publish the ordered snapshot.
        _byKey[key] = new Entry(key, value, idx);
        _store = new Store(keys, values, idx + 1);
    }

    /// <summary>
    ///     Replaces the value of an entry that is already present, on both paths. Writes in place for
    ///     atomically-writable values (O(1)); otherwise swaps the key-path entry and publishes a fresh value array
    ///     so lock-free readers never see a torn value (O(n)).
    /// </summary>
    /// <param name="e">The existing entry to update.</param>
    /// <param name="value">The replacement value.</param>
    /// <remarks>The caller must hold <see cref="_writeLock" />.</remarks>
    private void ReplaceExistingUnderLock(Entry e, TValue value)
    {
        Store s = _store;
        int idx = e._index;

        if (ConcurrentDictionaryTypeProps<TValue>.IsWriteAtomic)
        {
            e._value = value;
            if ((uint)idx < (uint)s._count)
            {
                s._values[idx] = value;
            }

            return;
        }

        _byKey[e._key] = new Entry(e._key, value, idx);
        if ((uint)idx < (uint)s._count)
        {
            var values = (TValue[])s._values.Clone();
            values[idx] = value;
            _store = new Store(s._keys, values, s._count);
        }
    }

    /// <summary>
    ///     Removes <paramref name="e" /> from both paths, shifting the entries after it down by one (and fixing the
    ///     recorded positions of the shifted entries) into fresh arrays, then publishing the new snapshot. O(n) in
    ///     the tail length.
    /// </summary>
    /// <param name="e">The entry to remove (located via key or index by the caller).</param>
    /// <remarks>The caller must hold <see cref="_writeLock" />.</remarks>
    private void RemoveCoreUnderLock(Entry e)
    {
        Store s = _store;
        int n = s._count;
        int idx = e._index;

        // Fresh arrays so concurrent readers of the old snapshot are undisturbed by the shift.
        var keys = new TKey[s._keys.Length];
        var values = new TValue[s._values.Length];
        Array.Copy(s._keys, keys, idx);                       // keep [0, idx)
        Array.Copy(s._keys, idx + 1, keys, idx, n - idx - 1); // shift (idx, n) down
        Array.Copy(s._values, values, idx);
        Array.Copy(s._values, idx + 1, values, idx, n - idx - 1);

        // Repair the recorded positions of the shifted entries so IndexOf stays correct.
        for (int i = idx; i < n - 1; i++)
        {
            if (_byKey.TryGetValue(keys[i], out var shifted))
            {
                shifted._index = i;
            }
        }

        _byKey.TryRemove(e._key, out _);
        _store = new Store(keys, values, n - 1);
    }

    /// <summary>Throws <see cref="ArgumentNullException" /> if <paramref name="key" /> is null (guards the reference-key case).</summary>
    /// <param name="key">The key to validate.</param>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is <see langword="null" />.</exception>
    private static void NullCheck(TKey key)
    {
        if (key is null)
        {
            throw new ArgumentNullException(nameof(key));
        }
    }
}
