using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp.Internal;

/// <summary>
///     A finished trial's dictionary held in a compact form (Rustuna plan 13, item S12): readable in place through
///     <see cref="IReadOnlyDictionary{TKey,TValue}" />, and turned back into the <see cref="OrderedDict{TKey,TValue}" />
///     it was built from when a caller asks a <see cref="FrozenTrial" /> for its mutable dictionary.
/// </summary>
/// <typeparam name="TKey">The dictionary's key type.</typeparam>
/// <typeparam name="TValue">The dictionary's value type.</typeparam>
/// <remarks>
///     Only <see cref="Storages.InMemoryStorage" /> built with <c>compactFinishedTrials: true</c> produces these. Every
///     implementation is immutable after construction, so any number of threads may read one, and a trial may share one
///     with its <see cref="FrozenTrial.ShallowCopy" /> or with every other trial of the same parameter schema.
/// </remarks>
internal interface ICompactDictionary<TKey, TValue> : IReadOnlyDictionary<TKey, TValue>
    where TKey : notnull
{
    /// <summary>A new dictionary holding the same entries in the same order, owned by the caller.</summary>
    /// <returns>A fresh <see cref="OrderedDict{TKey,TValue}" /> sized to the entry count.</returns>
    /// <remarks>
    ///     Values are rebuilt the way a read rebuilds them: a parameter's external value is a new box with the stored
    ///     bits, a compacted string a new string with the stored characters. Equal to the original entries, not the same
    ///     objects; nested containers in attributes are the same objects.
    /// </remarks>
    OrderedDict<TKey, TValue> Materialize();
}

/// <summary>
///     The read-only dictionary surface every compact form shares: lookups, ordered enumeration, and the non-generic
///     <see cref="IDictionary" /> that <see cref="PythonRepr" /> and <see cref="PyJson" /> recognize as a Python dict.
/// </summary>
/// <typeparam name="TKey">The dictionary's key type.</typeparam>
/// <typeparam name="TValue">The dictionary's value type.</typeparam>
/// <remarks>
///     <para>
///         Mutation through <see cref="IDictionary" /> throws <see cref="NotSupportedException" />, which is the point:
///         a caller that wants to change a compact trial must go through the trial's <c>…Dict</c> accessor, which
///         replaces the compact form with a real dictionary first, so the change is kept. A cast to
///         <see cref="OrderedDict{TKey,TValue}" /> fails instead of writing into a copy that would be thrown away.
///     </para>
///     <para>
///         Entries enumerate in the order the source dictionary held them, which is what Optuna's insertion-ordered
///         dicts promise.
///     </para>
/// </remarks>
internal abstract class CompactDictionaryView<TKey, TValue> : ICompactDictionary<TKey, TValue>, IDictionary
    where TKey : notnull
{
    /// <summary>Number of entries.</summary>
    public abstract int Count { get; }

    /// <summary>The key at an enumeration position.</summary>
    /// <param name="index">Position in <c>[0, Count)</c>.</param>
    /// <returns>The key stored at that position.</returns>
    protected abstract TKey KeyAt(int index);

    /// <summary>The value at an enumeration position, rebuilt from the compact form if it is stored compactly.</summary>
    /// <param name="index">Position in <c>[0, Count)</c>.</param>
    /// <returns>The value a reader of the original dictionary would have seen, possibly as a new object.</returns>
    protected abstract TValue ValueAt(int index);

    /// <summary>The position of a key, or -1 when the dictionary does not hold it.</summary>
    /// <param name="key">Key to find, compared ordinally like the source dictionary's default comparer.</param>
    /// <returns>The position in <c>[0, Count)</c>, or -1.</returns>
    protected abstract int IndexOf(TKey key);

    /// <summary>The value stored under <paramref name="key" />.</summary>
    /// <param name="key">Key to look up.</param>
    /// <returns>The value a reader of the original dictionary would get, rebuilt if it is stored compactly.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is null.</exception>
    /// <exception cref="KeyNotFoundException">The dictionary does not hold <paramref name="key" />; the message is the BCL dictionaries' own.</exception>
    public TValue this[TKey key]
    {
        get
        {
            ArgumentNullException.ThrowIfNull(key);
            int index = IndexOf(key);
            if (index < 0)
                throw new KeyNotFoundException($"The given key '{key}' was not present in the dictionary.");
            return ValueAt(index);
        }
    }

    /// <summary>The keys in enumeration order.</summary>
    public virtual IEnumerable<TKey> Keys
    {
        get
        {
            for (int i = 0; i < Count; i++)
                yield return KeyAt(i);
        }
    }

    /// <summary>The values in enumeration order, each rebuilt as <see cref="ValueAt" /> rebuilds it.</summary>
    public IEnumerable<TValue> Values
    {
        get
        {
            for (int i = 0; i < Count; i++)
                yield return ValueAt(i);
        }
    }

    /// <summary>Whether the dictionary holds <paramref name="key" />.</summary>
    /// <param name="key">Key to look up.</param>
    /// <returns>True when the key is present.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is null.</exception>
    public bool ContainsKey(TKey key)
    {
        ArgumentNullException.ThrowIfNull(key);
        return IndexOf(key) >= 0;
    }

    /// <summary>Looks up <paramref name="key" /> without throwing when it is absent.</summary>
    /// <param name="key">Key to look up.</param>
    /// <param name="value">The value when present, otherwise the type's default.</param>
    /// <returns>True when the key is present.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is null.</exception>
    public bool TryGetValue(TKey key, out TValue value)
    {
        ArgumentNullException.ThrowIfNull(key);
        int index = IndexOf(key);
        if (index < 0)
        {
            value = default!;
            return false;
        }
        value = ValueAt(index);
        return true;
    }

    /// <summary>Enumerates the entries in order, rebuilding each value as it is reached.</summary>
    /// <returns>An enumerator over key/value pairs.</returns>
    public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
    {
        for (int i = 0; i < Count; i++)
            yield return new KeyValuePair<TKey, TValue>(KeyAt(i), ValueAt(i));
    }

    /// <summary>Enumerates the entries in order.</summary>
    /// <returns>An enumerator over key/value pairs.</returns>
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <inheritdoc />
    public OrderedDict<TKey, TValue> Materialize()
    {
        var dictionary = new OrderedDict<TKey, TValue>(Count);
        for (int i = 0; i < Count; i++)
            dictionary.Add(KeyAt(i), ValueAt(i));
        return dictionary;
    }

    // ------------------------------------------------------------------ IDictionary (read-only)

    /// <summary>Always true: no entry can be added or removed.</summary>
    bool IDictionary.IsFixedSize => true;

    /// <summary>Always true: see the class remarks for how to change a compact trial.</summary>
    bool IDictionary.IsReadOnly => true;

    /// <summary>A snapshot array of the keys in enumeration order.</summary>
    ICollection IDictionary.Keys
    {
        get
        {
            var keys = new TKey[Count];
            for (int i = 0; i < keys.Length; i++)
                keys[i] = KeyAt(i);
            return keys;
        }
    }

    /// <summary>A snapshot array of the values in enumeration order.</summary>
    ICollection IDictionary.Values
    {
        get
        {
            var values = new TValue[Count];
            for (int i = 0; i < values.Length; i++)
                values[i] = ValueAt(i);
            return values;
        }
    }

    /// <summary>The value under a key, or null when absent or of another key type (the non-generic contract).</summary>
    /// <param name="key">Key to look up.</param>
    /// <returns>The value, rebuilt if stored compactly, or null.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is null.</exception>
    /// <exception cref="NotSupportedException">On set: the dictionary is read-only.</exception>
    object? IDictionary.this[object key]
    {
        get
        {
            ArgumentNullException.ThrowIfNull(key);
            if (key is TKey typed)
            {
                int index = IndexOf(typed);
                if (index >= 0)
                    return ValueAt(index);
            }
            return null;
        }
        set => throw ReadOnly();
    }

    /// <summary>Not supported: the dictionary is read-only.</summary>
    /// <param name="key">Ignored.</param>
    /// <param name="value">Ignored.</param>
    /// <exception cref="NotSupportedException">Always.</exception>
    void IDictionary.Add(object key, object? value) => throw ReadOnly();

    /// <summary>Not supported: the dictionary is read-only.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    void IDictionary.Clear() => throw ReadOnly();

    /// <summary>Whether a key of the dictionary's key type is present.</summary>
    /// <param name="key">Key to look up.</param>
    /// <returns>True when present.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="key" /> is null.</exception>
    bool IDictionary.Contains(object key)
    {
        ArgumentNullException.ThrowIfNull(key);
        return key is TKey typed && IndexOf(typed) >= 0;
    }

    /// <summary>A non-generic enumerator yielding <see cref="DictionaryEntry" /> values in order.</summary>
    /// <returns>The enumerator.</returns>
    IDictionaryEnumerator IDictionary.GetEnumerator() => new EntryEnumerator(this);

    /// <summary>Not supported: the dictionary is read-only.</summary>
    /// <param name="key">Ignored.</param>
    /// <exception cref="NotSupportedException">Always.</exception>
    void IDictionary.Remove(object key) => throw ReadOnly();

    /// <summary>Copies the entries, as key/value pairs or <see cref="DictionaryEntry" /> values, into an array.</summary>
    /// <param name="array">A one-dimensional array of <see cref="KeyValuePair{TKey,TValue}" />, <see cref="DictionaryEntry" /> or <see cref="object" />.</param>
    /// <param name="index">First position written.</param>
    /// <exception cref="ArgumentNullException"><paramref name="array" /> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is negative.</exception>
    /// <exception cref="ArgumentException">The array is too short or of an element type that cannot hold an entry.</exception>
    void ICollection.CopyTo(Array array, int index)
    {
        ArgumentNullException.ThrowIfNull(array);
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        if (array.Length - index < Count)
            throw new ArgumentException("Destination array is not long enough to copy all the items in the collection.");
        switch (array)
        {
            case KeyValuePair<TKey, TValue>[] pairs:
                for (int i = 0; i < Count; i++)
                    pairs[index + i] = new KeyValuePair<TKey, TValue>(KeyAt(i), ValueAt(i));
                return;
            case DictionaryEntry[] entries:
                for (int i = 0; i < Count; i++)
                    entries[index + i] = new DictionaryEntry(KeyAt(i), ValueAt(i));
                return;
            case object?[] objects:
                for (int i = 0; i < Count; i++)
                    objects[index + i] = new KeyValuePair<TKey, TValue>(KeyAt(i), ValueAt(i));
                return;
            default:
                throw new ArgumentException("Target array type is not compatible with the type of items in the collection.");
        }
    }

    /// <summary>The entry count.</summary>
    int ICollection.Count => Count;

    /// <summary>False: reads need no synchronization because the view never changes.</summary>
    bool ICollection.IsSynchronized => false;

    /// <summary>This view itself.</summary>
    object ICollection.SyncRoot => this;

    /// <summary>The exception every mutation through <see cref="IDictionary" /> throws.</summary>
    /// <returns>A <see cref="NotSupportedException" /> naming the way to change a compact trial.</returns>
    private static NotSupportedException ReadOnly()
        => new("This dictionary is a finished trial's compact form and is read-only; change the trial through its " +
               "ParamsDict / DistributionsDict / UserAttrsDict / SystemAttrsDict accessor instead.");

    /// <summary>The non-generic enumerator over a view, in entry order.</summary>
    private sealed class EntryEnumerator : IDictionaryEnumerator
    {
        private readonly CompactDictionaryView<TKey, TValue> _view;
        private int _index = -1;
        private DictionaryEntry _current;

        /// <summary>Starts before the first entry of <paramref name="view" />.</summary>
        /// <param name="view">The view to enumerate.</param>
        internal EntryEnumerator(CompactDictionaryView<TKey, TValue> view) => _view = view;

        /// <summary>The current entry.</summary>
        public DictionaryEntry Entry => _current;

        /// <summary>The current key.</summary>
        public object Key => _current.Key;

        /// <summary>The current value.</summary>
        public object? Value => _current.Value;

        /// <summary>The current entry, boxed.</summary>
        public object Current => _current;

        /// <summary>Advances to the next entry.</summary>
        /// <returns>False past the last entry.</returns>
        public bool MoveNext()
        {
            if (_index + 1 >= _view.Count)
            {
                _index = _view.Count;
                return false;
            }
            _index++;
            _current = new DictionaryEntry(_view.KeyAt(_index), _view.ValueAt(_index));
            return true;
        }

        /// <summary>Returns to the position before the first entry.</summary>
        public void Reset()
        {
            _index = -1;
            _current = default;
        }
    }
}

/// <summary>
///     The ordered parameter names and distribution instances that many finished trials of a study share (plan 13, item
///     S12): a compact trial stores only its internal values against one of these.
/// </summary>
/// <remarks>
///     Immutable, so trials and threads share it freely. Two trials share a schema only when their parameter names come
///     in the same order and their distributions are the same instances; <see cref="Storages.InMemoryStorage" /> already
///     shares bitwise-identical distributions within a study (plan 13, item S7), so a static search space yields one
///     schema for the whole study.
/// </remarks>
internal sealed class ParamSchema
{
    private readonly Dictionary<string, int> _index;

    /// <summary>Builds a schema from parallel arrays, which it keeps (the caller hands over ownership).</summary>
    /// <param name="names">Parameter names in the trials' order; never changed after this call.</param>
    /// <param name="distributions">The distribution instance of each name, in the same order.</param>
    internal ParamSchema(string[] names, BaseDistribution[] distributions)
    {
        Names = names;
        Distributions = distributions;
        _index = new Dictionary<string, int>(names.Length);
        for (int i = 0; i < names.Length; i++)
            _index[names[i]] = i;
        ReadOnlyNames = new ReadOnlyCollection<string>(names);
        View = new DistributionsView(this);
    }

    /// <summary>The parameter names in order. Not to be changed: every trial of the schema reads it.</summary>
    internal string[] Names { get; }

    /// <summary>The distribution of each name, in order. Not to be changed, as <see cref="Names" />.</summary>
    internal BaseDistribution[] Distributions { get; }

    /// <summary>The names wrapped read-only, handed out as a view's <c>Keys</c> so no caller can change the array.</summary>
    internal ReadOnlyCollection<string> ReadOnlyNames { get; }

    /// <summary>The one read-only distributions dictionary every trial of this schema returns from <c>Distributions</c>.</summary>
    internal DistributionsView View { get; }

    /// <summary>The position of a parameter name, or -1.</summary>
    /// <param name="name">Parameter name, compared ordinally like the trials' dictionaries.</param>
    /// <returns>Its position, or -1 when the schema does not hold it.</returns>
    internal int IndexOf(string name) => _index.TryGetValue(name, out int index) ? index : -1;

    /// <summary>Marks, in <see cref="PositionsFor" />, a parameter the schema does not hold.</summary>
    internal const int Absent = -1;

    /// <summary>
    ///     Marks, in <see cref="PositionsFor" />, a parameter the schema holds under a distribution that may convert
    ///     values differently from the reader's, so only rebuilding the external value is exact.
    /// </summary>
    internal const int NotInterchangeable = -2;

    /// <summary>
    ///     For each parameter a reader converts with its own distribution, where a compact trial of this schema stores a
    ///     value that is exactly what that conversion would return (plan 13, item S12).
    /// </summary>
    /// <param name="names">The parameters the reader wants, in its order.</param>
    /// <param name="distributions">The distribution the reader converts each one with, in the same order.</param>
    /// <returns>
    ///     Per parameter: its position in the schema when the reader's distribution is interchangeable with the schema's
    ///     (<see cref="DistributionIdentity.AreInterchangeable" />), so <see cref="CompactParams.InternalAt" /> equals
    ///     <c>distribution.ToInternalRepr(Params[name])</c>; <see cref="Absent" /> when the schema lacks it;
    ///     <see cref="NotInterchangeable" /> otherwise.
    /// </returns>
    /// <remarks>
    ///     Why the stored value is exact: compaction keeps <c>internal = d.ToInternalRepr(stored)</c> only when
    ///     <c>d.ToExternalRepr(internal)</c> is indistinguishable from <c>stored</c>, and an interchangeable distribution
    ///     converts every value as <c>d</c> does, so it maps the rebuilt external value back to that same internal.
    /// </remarks>
    internal int[] PositionsFor(string[] names, BaseDistribution[] distributions)
    {
        var positions = new int[names.Length];
        for (int i = 0; i < names.Length; i++)
        {
            int position = IndexOf(names[i]);
            positions[i] = position < 0
                ? Absent
                : DistributionIdentity.AreInterchangeable(distributions[i], Distributions[position])
                    ? position
                    : NotInterchangeable;
        }
        return positions;
    }

    /// <summary>
    ///     Whether a trial's dictionaries have this schema: the same names in the same order in both, and distributions
    ///     interchangeable with the schema's (<see cref="DistributionIdentity.AreInterchangeable" />).
    /// </summary>
    /// <param name="params">The trial's parameter dictionary.</param>
    /// <param name="distributions">The trial's distribution dictionary.</param>
    /// <returns>True when the trial can store its values against this schema.</returns>
    internal bool Matches(OrderedDict<string, object?> @params, OrderedDict<string, BaseDistribution> distributions)
    {
        if (@params.Count != Names.Length || distributions.Count != Names.Length)
            return false;
        for (int i = 0; i < Names.Length; i++)
        {
            KeyValuePair<string, BaseDistribution> entry = distributions.GetAt(i);
            if (!string.Equals(entry.Key, Names[i], StringComparison.Ordinal)
                || !string.Equals(@params.GetAt(i).Key, Names[i], StringComparison.Ordinal)
                || !DistributionIdentity.AreInterchangeable(entry.Value, Distributions[i]))
                return false;
        }
        return true;
    }

    /// <summary>
    ///     The hash a <see cref="ParamSchemaCache" /> files a schema under: names by value and distributions by
    ///     <see cref="DistributionIdentity.HashOf" />, in order.
    /// </summary>
    /// <param name="distributions">A trial's distribution dictionary.</param>
    /// <returns>A hash consistent with <see cref="Matches" />: trials that match one schema hash alike.</returns>
    internal static int HashOf(OrderedDict<string, BaseDistribution> distributions)
    {
        var hash = new HashCode();
        hash.Add(distributions.Count);
        for (int i = 0; i < distributions.Count; i++)
        {
            KeyValuePair<string, BaseDistribution> entry = distributions.GetAt(i);
            hash.Add(entry.Key, StringComparer.Ordinal);
            hash.Add(DistributionIdentity.HashOf(entry.Value));
        }
        return hash.ToHashCode();
    }
}

/// <summary>
///     When two distribution objects can stand for each other in a compact trial (plan 13, item S12): no read can tell
///     them apart except by asking whether they are the same object.
/// </summary>
/// <remarks>
///     <para>
///         Float and integer distributions follow <see cref="Distribution.IsBitwiseIdentical" />, the rule the storage
///         already shares instances by (plan 13, item S7). A plain <see cref="CategoricalDistribution" /> qualifies when
///         its choices are identical one by one (<see cref="IsSameValue" />): every <c>SuggestCategorical</c> call builds
///         a new distribution, so without this a categorical parameter would give every trial a schema of its own. Python-equal choices that
///         print differently (<c>1</c> and <c>1.0</c>) are not identical.
///     </para>
///     <para>
///         Anything else (a subclass of the categorical distribution, a custom distribution) must be the same instance.
///     </para>
/// </remarks>
internal static class DistributionIdentity
{
    /// <summary>Whether <paramref name="a" /> can be returned where a trial stored <paramref name="b" />, and the other way round.</summary>
    /// <param name="a">One distribution.</param>
    /// <param name="b">The other.</param>
    /// <returns>True for the same instance or indistinguishable ones (see the class remarks).</returns>
    internal static bool AreInterchangeable(BaseDistribution a, BaseDistribution b)
    {
        if (ReferenceEquals(a, b) || Distribution.IsBitwiseIdentical(a, b))
            return true;
        if (a.GetType() != typeof(CategoricalDistribution) || b.GetType() != typeof(CategoricalDistribution))
            return false;
        IReadOnlyList<CategoricalChoice> choicesA = ((CategoricalDistribution)a).Choices;
        IReadOnlyList<CategoricalChoice> choicesB = ((CategoricalDistribution)b).Choices;
        if (choicesA.Count != choicesB.Count)
            return false;
        for (int i = 0; i < choicesA.Count; i++)
            if (!IsSameValue(choicesA[i].Value, choicesB[i].Value))
                return false;
        return true;
    }

    /// <summary>A hash consistent with <see cref="AreInterchangeable" />: interchangeable distributions hash alike.</summary>
    /// <param name="distribution">The distribution to hash.</param>
    /// <returns>Its field-wise hash for the kinds compared by content, its identity hash otherwise.</returns>
    internal static int HashOf(BaseDistribution distribution)
    {
        switch (distribution)
        {
            case FloatDistribution f:
                return HashCode.Combine(f.GetType(), BitConverter.DoubleToInt64Bits(f.Low),
                    BitConverter.DoubleToInt64Bits(f.High), f.Step is null ? 0L : BitConverter.DoubleToInt64Bits(f.Step.Value),
                    f.Log);
            case IntDistribution i:
                return HashCode.Combine(i.GetType(), i.Low, i.High, i.Step, i.Log);
            case CategoricalDistribution c when c.GetType() == typeof(CategoricalDistribution):
                var hash = new HashCode();
                hash.Add(c.Choices.Count);
                foreach (CategoricalChoice choice in c.Choices)
                    hash.Add(ValueHash(choice.Value));
                return hash.ToHashCode();
            default:
                return RuntimeHelpers.GetHashCode(distribution);
        }
    }

    /// <summary>
    ///     Whether two values are indistinguishable except by object identity: the same object, or values of the same
    ///     type with the same bits (floating point), the same number (integers), the same truth value, or the same
    ///     characters (strings).
    /// </summary>
    /// <param name="a">One value.</param>
    /// <param name="b">The other.</param>
    /// <returns>True when a reader comparing anything but references sees one value.</returns>
    /// <remarks>
    ///     Bits, not <c>==</c>, for floating point: -0.0 and 0.0 are equal but print differently. Any other kind of
    ///     object (a decimal, whose <c>1.0m</c> and <c>1.00m</c> are equal but print differently, a list, a user type)
    ///     must be the same instance.
    /// </remarks>
    internal static bool IsSameValue(object? a, object? b)
    {
        if (ReferenceEquals(a, b))
            return true;
        if (a is null || b is null || a.GetType() != b.GetType())
            return false;
        return a switch
        {
            double x => BitConverter.DoubleToInt64Bits(x) == BitConverter.DoubleToInt64Bits((double)b),
            float x => BitConverter.SingleToInt32Bits(x) == BitConverter.SingleToInt32Bits((float)b),
            string x => string.Equals(x, (string)b, StringComparison.Ordinal),
            long or int or short or sbyte or byte or ulong or uint or ushort or bool or char => a.Equals(b),
            _ => false,
        };
    }

    /// <summary>A hash consistent with <see cref="IsSameValue" />.</summary>
    /// <param name="value">The value to hash.</param>
    /// <returns>A content hash for the kinds compared by content, an identity hash otherwise.</returns>
    private static int ValueHash(object? value) => value switch
    {
        null => 0,
        double x => HashCode.Combine(typeof(double), BitConverter.DoubleToInt64Bits(x)),
        float x => HashCode.Combine(typeof(float), BitConverter.SingleToInt32Bits(x)),
        string x => HashCode.Combine(typeof(string), StringComparer.Ordinal.GetHashCode(x)),
        long or int or short or sbyte or byte or ulong or uint or ushort or bool or char
            => HashCode.Combine(value.GetType(), value.GetHashCode()),
        _ => RuntimeHelpers.GetHashCode(value),
    };
}

/// <summary>
///     A study's parameter schemas (plan 13, item S12), found or created as its trials finish. Not thread-safe:
///     <see cref="Storages.InMemoryStorage" /> uses it under its lock.
/// </summary>
/// <remarks>
///     A static search space hits <see cref="FindOrAdd" />'s last-schema check every time, which costs one pass over the
///     names. A dynamic one keeps one schema per distinct shape up to <see cref="MaxSchemas" />; trials of further
///     shapes keep their dictionaries, which only forgoes the memory saving.
/// </remarks>
internal sealed class ParamSchemaCache
{
    /// <summary>The most schemas one study keeps; enough for conditional spaces, bounded for pathological ones.</summary>
    internal const int MaxSchemas = 256;

    private readonly Dictionary<int, List<ParamSchema>> _byHash = new();
    private ParamSchema? _last;

    /// <summary>How many schemas the study has; a diagnostic for tests.</summary>
    internal int Count { get; private set; }

    /// <summary>
    ///     The schema a finished trial's dictionaries have, created on first sight, or null when the trial cannot use
    ///     one.
    /// </summary>
    /// <param name="params">The trial's parameter dictionary.</param>
    /// <param name="distributions">The trial's distribution dictionary.</param>
    /// <returns>
    ///     The matching schema, or null when the two dictionaries do not hold the same names in the same order (a trial
    ///     added from a template may list them differently) or the study already has <see cref="MaxSchemas" />.
    /// </returns>
    internal ParamSchema? FindOrAdd(OrderedDict<string, object?> @params, OrderedDict<string, BaseDistribution> distributions)
    {
        if (_last is not null && _last.Matches(@params, distributions))
            return _last;
        if (@params.Count != distributions.Count)
            return null;

        int hash = ParamSchema.HashOf(distributions);
        if (_byHash.TryGetValue(hash, out List<ParamSchema>? candidates))
        {
            foreach (ParamSchema candidate in candidates)
            {
                if (candidate.Matches(@params, distributions))
                    return _last = candidate;
            }
        }

        // A new shape: the parameter names must come in the distributions' order, or one schema cannot rebuild both.
        int n = distributions.Count;
        var names = new string[n];
        var instances = new BaseDistribution[n];
        for (int i = 0; i < n; i++)
        {
            KeyValuePair<string, BaseDistribution> entry = distributions.GetAt(i);
            if (!string.Equals(@params.GetAt(i).Key, entry.Key, StringComparison.Ordinal))
                return null;
            names[i] = entry.Key;
            instances[i] = entry.Value;
        }
        if (Count >= MaxSchemas)
            return null;

        var schema = new ParamSchema(names, instances);
        if (candidates is null)
            _byHash[hash] = candidates = new List<ParamSchema>(1);
        candidates.Add(schema);
        Count++;
        return _last = schema;
    }
}

/// <summary>
///     A compact trial's parameters (plan 13, item S12): one internal-representation value per name of a shared
///     <see cref="ParamSchema" />, read back through each distribution's <c>ToExternalRepr</c>.
/// </summary>
/// <remarks>
///     <see cref="TryCreate" /> accepts a trial only when every stored value comes back indistinguishable from its
///     internal form (<see cref="DistributionIdentity.IsSameValue" />: same type and bits for numbers, same characters for
///     strings, the same object otherwise), so a read returns what the trial stored; anything else keeps its dictionary.
///     A read of a numeric value allocates a new box each time.
/// </remarks>
internal sealed class CompactParams : CompactDictionaryView<string, object?>
{
    private readonly double[] _internals;

    /// <summary>Wraps values already verified by <see cref="TryCreate" />.</summary>
    /// <param name="schema">The shared names and distributions.</param>
    /// <param name="internals">One internal value per schema entry; owned by this object from now on.</param>
    private CompactParams(ParamSchema schema, double[] internals)
    {
        Schema = schema;
        _internals = internals;
    }

    /// <summary>The shared names and distributions the values are stored against.</summary>
    internal ParamSchema Schema { get; }

    /// <inheritdoc />
    public override int Count => _internals.Length;

    /// <summary>The schema's names, read-only.</summary>
    public override IEnumerable<string> Keys => Schema.ReadOnlyNames;

    /// <inheritdoc />
    protected override string KeyAt(int index) => Schema.Names[index];

    /// <inheritdoc />
    protected override object? ValueAt(int index) => Schema.Distributions[index].ToExternalRepr(_internals[index]);

    /// <inheritdoc />
    protected override int IndexOf(string key) => Schema.IndexOf(key);

    /// <summary>
    ///     The stored internal value at a schema position, without rebuilding the external value: what
    ///     <c>d.ToInternalRepr(Params[name])</c> returns for the schema's distribution <c>d</c> or any distribution
    ///     interchangeable with it (see <see cref="ParamSchema.PositionsFor" />).
    /// </summary>
    /// <param name="position">A position <see cref="ParamSchema.PositionsFor" /> returned for this trial's schema.</param>
    /// <returns>The internal value stored at that position.</returns>
    /// <exception cref="IndexOutOfRangeException"><paramref name="position" /> is not a position of the schema.</exception>
    internal double InternalAt(int position) => _internals[position];

    /// <summary>
    ///     The stored internal value of a parameter without rebuilding its external value: what
    ///     <c>Distributions[name].ToInternalRepr(Params[name])</c> returns, since <see cref="TryCreate" /> stored exactly
    ///     that.
    /// </summary>
    /// <param name="name">Parameter name.</param>
    /// <param name="value">The internal value, or 0 when absent.</param>
    /// <returns>True when the trial has the parameter.</returns>
    internal bool TryGetInternal(string name, out double value)
    {
        int index = Schema.IndexOf(name);
        if (index < 0)
        {
            value = 0;
            return false;
        }
        value = _internals[index];
        return true;
    }

    /// <summary>
    ///     The compact form of a finished trial's parameters, or null when any value would not read back identical.
    /// </summary>
    /// <param name="schema">The schema <see cref="ParamSchemaCache.FindOrAdd" /> matched to the trial.</param>
    /// <param name="params">The trial's parameter dictionary, in the schema's order.</param>
    /// <returns>The compact parameters, or null to keep the dictionary.</returns>
    /// <remarks>
    ///     A value a trial got from <c>SetTrialParam</c> is the distribution's own <c>ToExternalRepr</c> of its internal
    ///     value and always qualifies. A value from a template trial may not: an <c>int</c> under a float distribution
    ///     (Optuna keeps it an <c>int</c>), a 64-bit integer beyond 2^53, or a categorical choice equal to an earlier one
    ///     (<c>1</c> and <c>1.0</c>), whose first index is the one <c>ToInternalRepr</c> finds.
    /// </remarks>
    internal static CompactParams? TryCreate(ParamSchema schema, OrderedDict<string, object?> @params)
    {
        var internals = new double[schema.Names.Length];
        for (int i = 0; i < internals.Length; i++)
        {
            object? stored = @params.GetAt(i).Value;
            BaseDistribution distribution = schema.Distributions[i];
            double internalRepr;
            try
            {
                internalRepr = distribution.ToInternalRepr(stored);
            }
            catch (Exception)
            {
                // A value its own distribution cannot convert (only a hand-built template can hold one) keeps the
                // dictionary as it is; compaction must never fail the finish that triggered it.
                return null;
            }
            // The rebuilt value must be indistinguishable from the stored one; the schema's distribution may be an
            // interchangeable copy of the trial's own, whose choice objects then differ only by identity.
            if (!DistributionIdentity.IsSameValue(stored, distribution.ToExternalRepr(internalRepr)))
                return null;
            internals[i] = internalRepr;
        }
        return new CompactParams(schema, internals);
    }
}

/// <summary>
///     The read-only distributions dictionary of every compact trial of one <see cref="ParamSchema" /> (plan 13, item
///     S12): one instance per schema, not per trial.
/// </summary>
internal sealed class DistributionsView : CompactDictionaryView<string, BaseDistribution>
{
    private readonly ParamSchema _schema;

    /// <summary>Wraps a schema; created once by the schema itself.</summary>
    /// <param name="schema">The schema whose names and distributions this view exposes.</param>
    internal DistributionsView(ParamSchema schema) => _schema = schema;

    /// <inheritdoc />
    public override int Count => _schema.Names.Length;

    /// <summary>The schema's names, read-only.</summary>
    public override IEnumerable<string> Keys => _schema.ReadOnlyNames;

    /// <inheritdoc />
    protected override string KeyAt(int index) => _schema.Names[index];

    /// <inheritdoc />
    protected override BaseDistribution ValueAt(int index) => _schema.Distributions[index];

    /// <inheritdoc />
    protected override int IndexOf(string key) => _schema.IndexOf(key);
}

/// <summary>
///     A compact trial's user or system attributes (plan 13, item S12): two arrays instead of a dictionary, with long
///     ASCII strings held as bytes (<see cref="CompactString" />), and one shared instance for every empty one.
/// </summary>
/// <remarks>
///     Values other than long strings are kept as they are, so a nested list or dictionary is the same object the trial
///     stored. Lookups scan the keys while there are few of them and use an index built on the first lookup otherwise.
/// </remarks>
internal sealed class CompactAttrs : CompactDictionaryView<string, object?>
{
    /// <summary>Up to this many keys a lookup scans; more build an index once.</summary>
    internal const int LinearScanLimit = 16;

    /// <summary>The one compact form of every empty attribute dictionary.</summary>
    internal static readonly CompactAttrs Empty = new(Array.Empty<string>(), Array.Empty<object?>());

    private readonly string[] _keys;
    private readonly object?[] _values;
    private Dictionary<string, int>? _index;

    /// <summary>Wraps the arrays <see cref="Create" /> built.</summary>
    /// <param name="keys">Keys in order; owned by this object.</param>
    /// <param name="values">Values in order, long strings as <see cref="CompactString" />; owned by this object.</param>
    private CompactAttrs(string[] keys, object?[] values)
    {
        _keys = keys;
        _values = values;
    }

    /// <inheritdoc />
    public override int Count => _keys.Length;

    /// <inheritdoc />
    protected override string KeyAt(int index) => _keys[index];

    /// <inheritdoc />
    protected override object? ValueAt(int index)
        => _values[index] is CompactString compact ? compact.Materialize() : _values[index];

    /// <summary>The value as stored, a <see cref="CompactString" /> for a compacted string; a diagnostic for tests.</summary>
    /// <param name="key">An attribute name.</param>
    /// <returns>The stored object, or null when the key is absent.</returns>
    internal object? StoredValue(string key)
    {
        int index = IndexOf(key);
        return index < 0 ? null : _values[index];
    }

    /// <inheritdoc />
    protected override int IndexOf(string key)
    {
        if (_keys.Length <= LinearScanLimit)
        {
            for (int i = 0; i < _keys.Length; i++)
                if (string.Equals(_keys[i], key, StringComparison.Ordinal))
                    return i;
            return -1;
        }

        // Built once per object; two readers may both build one, and either result is correct, so the race only
        // costs a duplicate build. Volatile so a reader never sees a dictionary before its entries.
        Dictionary<string, int>? index = Volatile.Read(ref _index);
        if (index is null)
        {
            index = new Dictionary<string, int>(_keys.Length);
            for (int i = 0; i < _keys.Length; i++)
                index[_keys[i]] = i;
            Volatile.Write(ref _index, index);
        }
        return index.TryGetValue(key, out int position) ? position : -1;
    }

    /// <summary>The compact form of an attribute dictionary.</summary>
    /// <param name="attrs">The dictionary; read, never changed.</param>
    /// <returns><see cref="Empty" /> for an empty dictionary, otherwise a new compact copy of its entries.</returns>
    internal static CompactAttrs Create(OrderedDict<string, object?> attrs)
    {
        int n = attrs.Count;
        if (n == 0)
            return Empty;
        var keys = new string[n];
        var values = new object?[n];
        for (int i = 0; i < n; i++)
        {
            KeyValuePair<string, object?> entry = attrs.GetAt(i);
            keys[i] = entry.Key;
            values[i] = entry.Value is string text ? CompactString.TryCreate(text) ?? (object)text : entry.Value;
        }
        return new CompactAttrs(keys, values);
    }
}

/// <summary>
///     A long ASCII string held as bytes (plan 13, item S12): half the memory of the UTF-16 string, or a quarter for a
///     lowercase hexadecimal string such as the CMA-ES optimizer state Optuna stores in system attributes.
/// </summary>
/// <remarks>
///     <para>
///         <see cref="Materialize" /> returns a new string with exactly the original characters on every call; a form is
///         chosen only when that round trip is exact. Hexadecimal digits are stored two to a byte, and an odd count keeps
///         its last digit aside: CMA-ES splits its state into 2,045-character chunks, an odd length.
///     </para>
/// </remarks>
internal sealed class CompactString
{
    /// <summary>Strings shorter than this stay strings: the saving would not cover the extra object.</summary>
    internal const int MinLength = 64;

    private readonly byte[] _bytes;
    private readonly bool _lowerHex;
    private readonly char _trailingDigit;

    /// <summary>Wraps the stored bytes.</summary>
    /// <param name="bytes">ASCII characters, or the decoded digit pairs of a lowercase hexadecimal string.</param>
    /// <param name="lowerHex">True when <paramref name="bytes" /> must be re-encoded as lowercase hexadecimal.</param>
    /// <param name="trailingDigit">The last digit of an odd-length hexadecimal string, or <c>'\0'</c> for none.</param>
    private CompactString(byte[] bytes, bool lowerHex, char trailingDigit)
    {
        _bytes = bytes;
        _lowerHex = lowerHex;
        _trailingDigit = trailingDigit;
    }

    /// <summary>Whether this string is stored as hexadecimal digit pairs; a diagnostic for tests.</summary>
    internal bool IsHex => _lowerHex;

    /// <summary>How many bytes the stored form holds; a diagnostic for tests.</summary>
    internal int StoredBytes => _bytes.Length;

    /// <summary>The compact form of a string, or null when it is short or not ASCII.</summary>
    /// <param name="text">The string to store.</param>
    /// <returns>A compact string whose <see cref="Materialize" /> returns <paramref name="text" />'s characters, or null.</returns>
    internal static CompactString? TryCreate(string text)
    {
        if (text.Length < MinLength)
            return null;
        if (IsLowerHex(text))
        {
            int pairs = text.Length / 2;
            return new CompactString(
                Convert.FromHexString(text.AsSpan(0, pairs * 2)),
                lowerHex: true,
                trailingDigit: (text.Length & 1) != 0 ? text[^1] : '\0');
        }
        if (!Ascii.IsValid(text))
            return null;
        return new CompactString(Encoding.ASCII.GetBytes(text), lowerHex: false, trailingDigit: '\0');
    }

    /// <summary>A new string with the stored characters.</summary>
    /// <returns>The original string's characters, as a new instance.</returns>
    internal string Materialize()
    {
        if (!_lowerHex)
            return Encoding.ASCII.GetString(_bytes);
        int length = _bytes.Length * 2 + (_trailingDigit != '\0' ? 1 : 0);
        return string.Create(length, this, static (chars, compact) =>
        {
            const string digits = "0123456789abcdef";
            byte[] bytes = compact._bytes;
            for (int i = 0; i < bytes.Length; i++)
            {
                chars[2 * i] = digits[bytes[i] >> 4];
                chars[2 * i + 1] = digits[bytes[i] & 0xF];
            }
            if (compact._trailingDigit != '\0')
                chars[^1] = compact._trailingDigit;
        });
    }

    /// <summary>Whether every character is a lowercase hexadecimal digit, so bytes and one spare digit rebuild it exactly.</summary>
    /// <param name="text">The string to test.</param>
    /// <returns>True for <c>[0-9a-f]</c> only; uppercase digits would come back lowercase, so they do not qualify.</returns>
    private static bool IsLowerHex(string text)
    {
        foreach (char c in text)
        {
            if (c is not ((>= '0' and <= '9') or (>= 'a' and <= 'f')))
                return false;
        }
        return true;
    }
}
