using System;
using System.Collections;
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
/// </summary>
public sealed class OrderedDict<TKey, TValue> : IDictionary<TKey, TValue>, IReadOnlyDictionary<TKey, TValue>, IDictionary
    where TKey : notnull
{
    private readonly List<TKey> _order;
    private readonly Dictionary<TKey, TValue> _map;

    public OrderedDict()
    {
        _order = new List<TKey>();
        _map = new Dictionary<TKey, TValue>();
    }

    public OrderedDict(int capacity)
    {
        _order = new List<TKey>(capacity);
        _map = new Dictionary<TKey, TValue>(capacity);
    }

    /// <summary>Shallow-copies another dictionary preserving its enumeration order (Python <c>copy.copy(dict)</c>).</summary>
    public OrderedDict(IEnumerable<KeyValuePair<TKey, TValue>> source) : this()
    {
        foreach (var kv in source)
            this[kv.Key] = kv.Value;
    }

    public TValue this[TKey key]
    {
        get => _map[key];
        set
        {
            if (!_map.ContainsKey(key))
                _order.Add(key);
            _map[key] = value;
        }
    }

    /// <summary>Keys in insertion order.</summary>
    public ICollection<TKey> Keys => _order.AsReadOnly();

    /// <summary>Values in insertion order.</summary>
    public ICollection<TValue> Values
    {
        get
        {
            var values = new List<TValue>(_order.Count);
            foreach (var key in _order)
                values.Add(_map[key]);
            return values.AsReadOnly();
        }
    }

    IEnumerable<TKey> IReadOnlyDictionary<TKey, TValue>.Keys => _order;

    IEnumerable<TValue> IReadOnlyDictionary<TKey, TValue>.Values
    {
        get
        {
            foreach (var key in _order)
                yield return _map[key];
        }
    }

    public int Count => _order.Count;

    public bool IsReadOnly => false;

    public void Add(TKey key, TValue value)
    {
        _map.Add(key, value);  // Throws on duplicate before touching the order list.
        _order.Add(key);
    }

    public void Add(KeyValuePair<TKey, TValue> item) => Add(item.Key, item.Value);

    public void Clear()
    {
        _order.Clear();
        _map.Clear();
    }

    public bool Contains(KeyValuePair<TKey, TValue> item)
        => _map.TryGetValue(item.Key, out var value) && EqualityComparer<TValue>.Default.Equals(value, item.Value);

    public bool ContainsKey(TKey key) => _map.ContainsKey(key);

    public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
    {
        foreach (var kv in this)
            array[arrayIndex++] = kv;
    }

    public bool Remove(TKey key)
    {
        if (!_map.Remove(key))
            return false;
        _order.Remove(key);
        return true;
    }

    public bool Remove(KeyValuePair<TKey, TValue> item)
        => Contains(item) && Remove(item.Key);

    public bool TryGetValue(TKey key, out TValue value) => _map.TryGetValue(key, out value!);

    /// <summary>Python <c>dict.get(key, default)</c>.</summary>
    public TValue? GetValueOrDefault(TKey key, TValue? defaultValue = default)
        => _map.TryGetValue(key, out var value) ? value : defaultValue;

    public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
    {
        foreach (var key in _order)
            yield return new KeyValuePair<TKey, TValue>(key, _map[key]);
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>Shallow copy preserving insertion order (Python <c>copy.copy(dict)</c>).</summary>
    public OrderedDict<TKey, TValue> ShallowClone() => new(this);

    // ------------------------------------------------------------------ non-generic IDictionary
    // (needed so serialization/repr code that pattern-matches on IDictionary sees dict semantics)

    bool IDictionary.IsFixedSize => false;

    bool IDictionary.IsReadOnly => false;

    ICollection IDictionary.Keys => (ICollection)Keys;

    ICollection IDictionary.Values => (ICollection)Values;

    bool ICollection.IsSynchronized => false;

    object ICollection.SyncRoot => this;

    object? IDictionary.this[object key]
    {
        get => key is TKey typed && _map.TryGetValue(typed, out var value) ? value : null;
        set => this[(TKey)key] = (TValue)value!;
    }

    void IDictionary.Add(object key, object? value) => Add((TKey)key, (TValue)value!);

    bool IDictionary.Contains(object key) => key is TKey typed && _map.ContainsKey(typed);

    void IDictionary.Remove(object key)
    {
        if (key is TKey typed)
            Remove(typed);
    }

    void ICollection.CopyTo(Array array, int index)
    {
        foreach (var kv in this)
            array.SetValue(new DictionaryEntry(kv.Key, kv.Value), index++);
    }

    IDictionaryEnumerator IDictionary.GetEnumerator() => new OrderedDictEnumerator(GetEnumerator());

    private sealed class OrderedDictEnumerator : IDictionaryEnumerator
    {
        private readonly IEnumerator<KeyValuePair<TKey, TValue>> _inner;

        internal OrderedDictEnumerator(IEnumerator<KeyValuePair<TKey, TValue>> inner)
        {
            _inner = inner;
        }

        public DictionaryEntry Entry => new(_inner.Current.Key, _inner.Current.Value);

        public object Key => _inner.Current.Key;

        public object? Value => _inner.Current.Value;

        public object Current => Entry;

        public bool MoveNext() => _inner.MoveNext();

        public void Reset() => _inner.Reset();
    }
}
