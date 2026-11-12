using System;
using System.Collections.Generic;

namespace OptunaSharp.Internal;

/// <summary>
///     CPython's <c>set</c> of non-negative ints, modeled closely enough to reproduce its iteration order
///     (<c>list(s)</c>): the open-addressing hash table of CPython 3.12's <c>Objects/setobject.c</c>, where
///     <c>hash(k) == k</c>.
/// </summary>
/// <remarks>
///     <para>
///         Optuna lists some candidates in set order and then indexes the list with a random draw, so any other order
///         (ascending, or a <see cref="HashSet{T}" />'s insertion order) picks another candidate for the same draw.
///         GridSampler's <c>rng.choice(list(set(range(n)) - set(visited) - set(running)))</c> is the use this serves
///         (QA-68): <c>list(set(range(16)) - {0, 1, 2, 3, 5, 6, 7, 9, 10, 12, 13, 14, 15})</c> is <c>[8, 11, 4]</c>,
///         because the difference builds a fresh 8-slot table where 8 and 11 wrap onto slots 0 and 3.
///     </para>
///     <para>
///         Modeled: the 8-slot minimum table; the probe sequence (the home slot and up to 9 linear probes, then
///         <c>i = (i * 5 + 1 + perturb) &amp; mask</c> with <c>perturb &gt;&gt;= 5</c>); growth on insert once
///         <c>fill * 5 &gt;= mask * 3</c>, to the first power of two above <c>used * 4</c> (<c>used * 2</c> past 50,000
///         elements); the dummy a discard leaves, which an insert reuses (the last one its probe passed) and a
///         difference update purges once dummies exceed a quarter of the table; <c>set_merge</c>'s copy; and
///         <c>set_difference</c>'s two strategies (copy, then discard, when this set is more than four times the other;
///         otherwise a fresh set filled in this set's slot order). Only what GridSampler needs is exposed, and the
///         committed corpus <c>oracle/cpython_order.json</c>, recorded with CPython 3.12, checks it.
///     </para>
///     <para>
///         Not thread-safe; each instance is built, read and dropped within one call. A table holds <c>int</c> slots,
///         4 bytes per slot and at most about 8 slots per element, so a grid of a million points briefly costs
///         ~16 MB, as <c>set(range(n))</c> costs CPython.
///     </para>
/// </remarks>
internal sealed class PyIntSet
{
    /// <summary><c>PySet_MINSIZE</c>: the smallest table, the one every new set starts with.</summary>
    private const int MinSize = 8;

    /// <summary><c>LINEAR_PROBES</c>: slots probed past the home slot before the perturbed jump.</summary>
    private const int LinearProbes = 9;

    /// <summary><c>PERTURB_SHIFT</c>: how far the perturbation shifts right on every jump.</summary>
    private const int PerturbShift = 5;

    /// <summary>A never-used slot (CPython's NULL key): it ends every probe sequence.</summary>
    private const int Empty = -1;

    /// <summary>A discarded slot (CPython's dummy key): probes pass it, inserts may reuse it, iteration skips it.</summary>
    private const int Dummy = -2;

    /// <summary>The slots: a key (&gt;= 0), <see cref="Empty" /> or <see cref="Dummy" />; the length is a power of two.</summary>
    private int[] _table = NewTable(MinSize);

    /// <summary>CPython's <c>fill</c>: active plus dummy slots, which drives growth.</summary>
    private int _fill;

    /// <summary>CPython's <c>used</c>: active slots, the set's length.</summary>
    private int _used;

    /// <summary>The number of elements, <c>len(s)</c>.</summary>
    internal int Count => _used;

    /// <summary><c>set(range(n))</c>: the ints 0 to <paramref name="n" /> - 1, inserted one by one as CPython does.</summary>
    /// <param name="n">The number of elements; 0 gives an empty set.</param>
    /// <returns>The new set.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="n" /> is negative.</exception>
    internal static PyIntSet Range(int n)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(n);
        var set = new PyIntSet();
        for (int k = 0; k < n; k++)
            set.AddEntry(k);
        return set;
    }

    /// <summary>
    ///     <c>self - other</c> where <paramref name="other" /> is a Python set: a new set whose slot order is CPython's.
    /// </summary>
    /// <param name="other">
    ///     The elements to leave out, as a set (its <see cref="IReadOnlyCollection{T}.Count" /> is <c>len(other)</c>,
    ///     which picks the strategy). Elements this set does not hold, negative ones included, are ignored.
    /// </param>
    /// <returns>The difference; this set is unchanged.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="other" /> is null.</exception>
    internal PyIntSet Difference(IReadOnlySet<int> other)
    {
        ArgumentNullException.ThrowIfNull(other);
        // set_difference: when this set is much larger than the other, copying it and discarding the other's elements is
        // cheaper, and leaves the copy's layout (dummies included, purged only past a quarter of the table).
        if ((_used >> 2) > other.Count)
        {
            PyIntSet copy = Copy();
            copy.DifferenceUpdate(other);
            return copy;
        }

        // Otherwise a fresh set receives this set's remaining elements in slot order (set_next), each through the
        // ordinary insert, so the result's layout comes from inserting them in that order into a growing table.
        var result = new PyIntSet();
        foreach (int key in _table)
        {
            if (key >= 0 && !other.Contains(key))
                result.AddEntry(key);
        }
        return result;
    }

    /// <summary><c>list(s)</c>: the elements in slot order, skipping empty and dummy slots.</summary>
    /// <returns>A new list.</returns>
    internal List<int> ToList()
    {
        var list = new List<int>(_used);
        foreach (int key in _table)
        {
            if (key >= 0)
                list.Add(key);
        }
        return list;
    }

    /// <summary>A table of <paramref name="size" /> empty slots.</summary>
    /// <param name="size">The slot count, a power of two.</param>
    /// <returns>The new table.</returns>
    private static int[] NewTable(int size)
    {
        var table = new int[size];
        Array.Fill(table, Empty);
        return table;
    }

    /// <summary>
    ///     <c>set_add_entry</c>: inserts <paramref name="key" /> unless present, then grows the table when an empty slot
    ///     was consumed and <c>fill * 5 &gt;= mask * 3</c>.
    /// </summary>
    /// <param name="key">The element, &gt;= 0.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="key" /> is negative (CPython hashes those differently).</exception>
    private void AddEntry(int key)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(key);
        ulong mask = (ulong)(_table.Length - 1);
        ulong i = (ulong)key & mask;
        ulong perturb = (ulong)key;
        int freeSlot = -1;
        int emptySlot;
        while (true)
        {
            // The home slot plus LinearProbes more when they fit below the mask (CPython's do { } while (probes--)).
            int probes = i + LinearProbes <= mask ? LinearProbes : 0;
            int j = (int)i;
            while (true)
            {
                int slot = _table[j];
                if (slot == Empty)
                {
                    emptySlot = j;
                    goto FoundUnusedOrDummy;
                }
                if (slot == key)
                    return;  // found_active: already a member
                if (slot == Dummy)
                    freeSlot = j;  // CPython keeps the LAST dummy the probe passed (measured: 3.12 reuses it)
                j++;
                if (probes-- == 0)
                    break;
            }
            perturb >>= PerturbShift;
            i = (i * 5 + 1 + perturb) & mask;
        }

    FoundUnusedOrDummy:
        if (freeSlot >= 0)
        {
            // A reused dummy does not raise fill, so it never triggers growth.
            _used++;
            _table[freeSlot] = key;
            return;
        }
        _fill++;
        _used++;
        _table[emptySlot] = key;
        if ((ulong)_fill * 5 < mask * 3)
            return;
        Resize(_used > 50000 ? _used * 2 : _used * 4);
    }

    /// <summary>
    ///     <c>set_table_resize</c>: moves the active elements, in old-slot order, into a table of the first power of two
    ///     above <paramref name="minUsed" />, dropping dummies.
    /// </summary>
    /// <param name="minUsed">The size the new table must exceed.</param>
    private void Resize(int minUsed)
    {
        long newSize = MinSize;
        while (newSize <= minUsed)
            newSize <<= 1;
        // CPython keeps the 8-slot small table as it is when it would rebuild it with nothing to purge.
        if (newSize == MinSize && _table.Length == MinSize && _fill == _used)
            return;
        int[] oldTable = _table;
        int[] newTable = NewTable((int)newSize);
        int newMask = (int)newSize - 1;
        foreach (int key in oldTable)
        {
            if (key >= 0)
                InsertClean(newTable, newMask, key);
        }
        _table = newTable;
        _fill = _used;
    }

    /// <summary><c>set_insert_clean</c>: places a key known to be absent into a table without dummies.</summary>
    /// <param name="table">The table, with no dummies and at least one empty slot.</param>
    /// <param name="mask">The table's length minus one.</param>
    /// <param name="key">The element, &gt;= 0.</param>
    private static void InsertClean(int[] table, int mask, int key)
    {
        ulong umask = (ulong)mask;
        ulong i = (ulong)key & umask;
        ulong perturb = (ulong)key;
        while (true)
        {
            if (table[i] == Empty)
            {
                table[i] = key;
                return;
            }
            if (i + LinearProbes <= umask)
            {
                for (ulong j = 1; j <= LinearProbes; j++)
                {
                    if (table[i + j] == Empty)
                    {
                        table[i + j] = key;
                        return;
                    }
                }
            }
            perturb >>= PerturbShift;
            i = (i * 5 + 1 + perturb) & umask;
        }
    }

    /// <summary><c>set_lookkey</c>: the slot holding <paramref name="key" />, or -1 once the probe reaches an empty slot.</summary>
    /// <param name="key">The element, &gt;= 0.</param>
    /// <returns>The slot index, or -1 when absent.</returns>
    private int LookKey(int key)
    {
        ulong mask = (ulong)(_table.Length - 1);
        ulong i = (ulong)key & mask;
        ulong perturb = (ulong)key;
        while (true)
        {
            int probes = i + LinearProbes <= mask ? LinearProbes : 0;
            int j = (int)i;
            while (true)
            {
                int slot = _table[j];
                if (slot == Empty)
                    return -1;
                if (slot == key)
                    return j;
                j++;
                if (probes-- == 0)
                    break;
            }
            perturb >>= PerturbShift;
            i = (i * 5 + 1 + perturb) & mask;
        }
    }

    /// <summary>
    ///     <c>set_copy</c>: a new set merged from this one (<c>set_merge</c> into an empty table): one resize up front when
    ///     needed, then a slot-for-slot copy when the sizes match and this set has no dummies, else clean inserts in this
    ///     set's slot order.
    /// </summary>
    /// <returns>The copy.</returns>
    private PyIntSet Copy()
    {
        var result = new PyIntSet();
        if (_used == 0)
            return result;
        ulong mask = (ulong)(result._table.Length - 1);
        if ((ulong)(result._fill + _used) * 5 >= mask * 3)
            result.Resize((result._used + _used) * 2);
        if (result._table.Length == _table.Length && _fill == _used)
        {
            Array.Copy(_table, result._table, _table.Length);
            result._fill = _fill;
            result._used = _used;
            return result;
        }
        int newMask = result._table.Length - 1;
        result._fill = _used;
        result._used = _used;
        foreach (int key in _table)
        {
            if (key >= 0)
                InsertClean(result._table, newMask, key);
        }
        return result;
    }

    /// <summary>
    ///     <c>set_difference_update_internal</c> with a set argument: discards each of <paramref name="other" />'s elements
    ///     (leaving dummies), then rebuilds the table when dummies exceed a quarter of it.
    /// </summary>
    /// <param name="other">The elements to discard; the order they are discarded in does not change the result.</param>
    private void DifferenceUpdate(IReadOnlySet<int> other)
    {
        // CPython iterates the intersection instead when the other set is more than 8 times larger; the discarded keys
        // are the same either way, and a discard's effect does not depend on the order.
        foreach (int key in other)
        {
            if (key < 0)
                continue;  // never a member; and -2 would otherwise match the dummy sentinel
            int slot = LookKey(key);
            if (slot < 0)
                continue;
            _table[slot] = Dummy;
            _used--;
        }
        if ((ulong)(_fill - _used) <= (ulong)(_table.Length - 1) / 4)
            return;
        Resize(_used > 50000 ? _used * 2 : _used * 4);
    }
}
