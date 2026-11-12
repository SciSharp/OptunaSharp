using System;
using System.Collections.Generic;

namespace OptunaSharp.Internal;

/// <summary>
///     CPython 3.12's <c>list.sort</c> (<c>Objects/listobject.c</c>): timsort with the powersort merge policy, ported
///     step for step, so that a comparison that is not a total order leaves the elements where CPython leaves them.
/// </summary>
/// <remarks>
///     <para>
///         With a strict weak order every stable sort gives the same result and <see cref="List{T}.Sort()" /> with a
///         tie-breaker would do. Optuna sorts some tuples whose first element can be NaN, which compares neither smaller
///         nor larger than anything, and then the result is whatever timsort's exact sequence of comparisons produces:
///         NSGA-III's <c>list.sort(reverse=True)</c> over <c>(distance, index)</c> pairs is the use this serves (QA-56),
///         where an all-NaN borderline front keeps its input order and <c>pop()</c> takes the last pair first.
///     </para>
///     <para>
///         Ported: <c>reverse</c> as CPython does it (reverse, stable sort, reverse again); <c>count_run</c> (an
///         ascending run, or a strictly descending one reversed in place); runs shorter than <c>minrun</c> extended by
///         binary insertion; the powersort merge policy (<c>powerloop</c>, <c>found_new_run</c>); <c>merge_at</c> with its
///         galloping trims; <c>merge_lo</c> / <c>merge_hi</c> with the adaptive <c>min_gallop</c>; and
///         <c>merge_force_collapse</c>. Every comparison is the caller's <c>lessThan</c>, CPython's <c>&lt;</c>, called
///         with the same operands in the same order. The committed corpus <c>oracle/cpython_order.json</c>, recorded
///         with CPython 3.12, checks NaN-bearing lists up to 1,000 elements. CPython 3.13 rewrote <c>count_run</c>, so
///         a 3.13 interpreter can order such lists differently; the port follows 3.12, the interop suite's interpreter.
///     </para>
///     <para>
///         <c>lessThan</c> must not throw or mutate the list (CPython's own guarantees under an exception are not
///         modeled); it may be inconsistent. Not thread-safe for a shared list.
///     </para>
/// </remarks>
internal static class PyListSort
{
    /// <summary><c>MIN_GALLOP</c>: the run of consecutive wins that switches a merge into galloping mode.</summary>
    private const int MinGallopStart = 7;

    /// <summary><c>MAX_MERGE_PENDING</c>: room for the pending-run stack of any list that fits in memory.</summary>
    private const int MaxMergePending = 85;

    /// <summary>Sorts <paramref name="list" /> in place exactly as CPython's <c>list.sort(reverse=...)</c> does.</summary>
    /// <param name="list">The list to sort.</param>
    /// <param name="lessThan">CPython's <c>a &lt; b</c> for two elements; it may be inconsistent (NaN).</param>
    /// <param name="reverse">Python's <c>reverse=True</c>: reversed, sorted stably, reversed again.</param>
    /// <typeparam name="T">The element type.</typeparam>
    /// <exception cref="ArgumentNullException"><paramref name="list" /> or <paramref name="lessThan" /> is null.</exception>
    internal static void Sort<T>(List<T> list, Func<T, T, bool> lessThan, bool reverse = false)
    {
        ArgumentNullException.ThrowIfNull(list);
        ArgumentNullException.ThrowIfNull(lessThan);
        int count = list.Count;
        // CPython returns before reversing for fewer than two elements; reversing them would change nothing anyway.
        if (count < 2)
            return;

        T[] keys = list.ToArray();
        if (reverse)
            Array.Reverse(keys);

        var ms = new MergeState<T>(keys, lessThan);
        int lo = 0;
        int nRemaining = count;
        int minRun = ComputeMinRun(nRemaining);
        do
        {
            // Identify the next run, and extend a short one to min(minrun, nremaining) by binary insertion.
            int n = CountRun(ms, lo, lo + nRemaining, out bool descending);
            if (descending)
                Array.Reverse(keys, lo, n);
            if (n < minRun)
            {
                int force = nRemaining <= minRun ? nRemaining : minRun;
                BinarySort(ms, lo, lo + force, lo + n);
                n = force;
            }

            // Merge pending runs per the powersort policy, then push this one.
            FoundNewRun(ms, n);
            ms.Base[ms.Count] = lo;
            ms.Length[ms.Count] = n;
            ms.Count++;
            lo += n;
            nRemaining -= n;
        } while (nRemaining > 0);

        MergeForceCollapse(ms);

        if (reverse)
            Array.Reverse(keys);
        for (int i = 0; i < count; i++)
            list[i] = keys[i];
    }

    /// <summary><c>merge_compute_minrun</c>: the minimum run length, in [32, 64] for lists of 64 elements or more.</summary>
    /// <param name="n">The list's length.</param>
    /// <returns><paramref name="n" /> itself below 64; else its top 6 bits, plus 1 if any lower bit is set.</returns>
    private static int ComputeMinRun(int n)
    {
        int r = 0;  // becomes 1 if any 1 bits are shifted off
        while (n >= 64)
        {
            r |= n & 1;
            n >>= 1;
        }
        return n + r;
    }

    /// <summary>
    ///     <c>count_run</c> (3.12): the length of the run starting at <paramref name="lo" />, either ascending
    ///     (<c>a[0] &lt;= a[1] &lt;= ...</c>, i.e. no next element is less) or strictly descending.
    /// </summary>
    /// <param name="ms">The sort state.</param>
    /// <param name="lo">The run's first index.</param>
    /// <param name="hi">One past the last index the run may reach; greater than <paramref name="lo" />.</param>
    /// <param name="descending">Set when the run is strictly descending, which the caller reverses.</param>
    /// <returns>The run's length, at least 1.</returns>
    /// <typeparam name="T">The element type.</typeparam>
    private static int CountRun<T>(MergeState<T> ms, int lo, int hi, out bool descending)
    {
        T[] a = ms.Keys;
        descending = false;
        ++lo;
        if (lo == hi)
            return 1;

        int n = 2;
        if (ms.LessThan(a[lo], a[lo - 1]))
        {
            descending = true;
            for (lo = lo + 1; lo < hi; ++lo, ++n)
            {
                if (!ms.LessThan(a[lo], a[lo - 1]))
                    break;
            }
        }
        else
        {
            for (lo = lo + 1; lo < hi; ++lo, ++n)
            {
                if (ms.LessThan(a[lo], a[lo - 1]))
                    break;
            }
        }
        return n;
    }

    /// <summary>
    ///     <c>binarysort</c> (3.12): extends the sorted prefix <c>[lo, start)</c> to <c>[lo, hi)</c> by binary insertion,
    ///     each new element placed after any it does not compare less than (which keeps the sort stable).
    /// </summary>
    /// <param name="ms">The sort state.</param>
    /// <param name="lo">The slice's first index.</param>
    /// <param name="hi">One past the slice's last index.</param>
    /// <param name="start">The first index not yet in the sorted prefix.</param>
    /// <typeparam name="T">The element type.</typeparam>
    private static void BinarySort<T>(MergeState<T> ms, int lo, int hi, int start)
    {
        T[] a = ms.Keys;
        if (lo == start)
            ++start;
        for (; start < hi; ++start)
        {
            // pivot >= all in [lo, l); pivot < all in [r, start).
            int l = lo;
            int r = start;
            T pivot = a[r];
            do
            {
                int p = l + ((r - l) >> 1);
                if (ms.LessThan(pivot, a[p]))
                    r = p;
                else
                    l = p + 1;
            } while (l < r);
            for (int p = start; p > l; --p)
                a[p] = a[p - 1];
            a[l] = pivot;
        }
    }

    /// <summary>
    ///     <c>powerloop</c>: the "power" of the run <c>[s1, s1 + n1)</c> followed by one of length <paramref name="n2" />
    ///     in a list of <paramref name="n" />: its depth in the conceptual binary merge tree powersort follows.
    /// </summary>
    /// <param name="s1">The first run's start index.</param>
    /// <param name="n1">The first run's length.</param>
    /// <param name="n2">The next run's length.</param>
    /// <param name="n">The list's length.</param>
    /// <returns>The power, at least 1.</returns>
    private static int PowerLoop(long s1, long n1, long n2, long n)
    {
        int result = 0;
        // 2a and 2b, twice the two runs' midpoints, so the halves stay integral; the bits of 2a/n and 2b/n are those
        // of a/n and b/n shifted by one, which does not change where they first differ.
        long a = 2 * s1 + n1;
        long b = a + n1 + n2;
        while (true)
        {
            ++result;
            if (a >= n)
            {
                // both quotient bits are 1
                a -= n;
                b -= n;
            }
            else if (b >= n)
            {
                break;  // a/n bit is 0, b/n bit is 1
            }
            // else both quotient bits are 0
            a <<= 1;
            b <<= 1;
        }
        return result;
    }

    /// <summary>
    ///     <c>found_new_run</c>: before a new run of length <paramref name="n2" /> is pushed, merges the pending runs whose
    ///     power exceeds the topmost run's, and records that power.
    /// </summary>
    /// <param name="ms">The sort state.</param>
    /// <param name="n2">The new run's length.</param>
    /// <typeparam name="T">The element type.</typeparam>
    private static void FoundNewRun<T>(MergeState<T> ms, int n2)
    {
        if (ms.Count == 0)
            return;
        int s1 = ms.Base[ms.Count - 1];
        int n1 = ms.Length[ms.Count - 1];
        int power = PowerLoop(s1, n1, n2, ms.Keys.Length);
        while (ms.Count > 1 && ms.Power[ms.Count - 2] > power)
            MergeAt(ms, ms.Count - 2);
        ms.Power[ms.Count - 1] = power;
    }

    /// <summary>
    ///     <c>merge_force_collapse</c>: merges every pending run, preferring the smaller neighbour of the second-last run.
    /// </summary>
    /// <param name="ms">The sort state.</param>
    /// <typeparam name="T">The element type.</typeparam>
    private static void MergeForceCollapse<T>(MergeState<T> ms)
    {
        while (ms.Count > 1)
        {
            int n = ms.Count - 2;
            if (n > 0 && ms.Length[n - 1] < ms.Length[n + 1])
                --n;
            MergeAt(ms, n);
        }
    }

    /// <summary>
    ///     <c>merge_at</c>: merges pending runs <paramref name="i" /> and <paramref name="i" /> + 1, after galloping off the
    ///     prefix of the first and the suffix of the second that are already in place.
    /// </summary>
    /// <param name="ms">The sort state.</param>
    /// <param name="i">The first run's stack index: the second-last or third-last run.</param>
    /// <typeparam name="T">The element type.</typeparam>
    private static void MergeAt<T>(MergeState<T> ms, int i)
    {
        T[] a = ms.Keys;
        int baseA = ms.Base[i];
        int na = ms.Length[i];
        int baseB = ms.Base[i + 1];
        int nb = ms.Length[i + 1];

        // Record the combined length; if i is the third-last run, slide the last run over (power included).
        ms.Length[i] = na + nb;
        if (i == ms.Count - 3)
        {
            ms.Base[i + 1] = ms.Base[i + 2];
            ms.Length[i + 1] = ms.Length[i + 2];
            ms.Power[i + 1] = ms.Power[i + 2];
        }
        ms.Count--;

        // Where does b start in a? Elements in a before that are already in place.
        int k = GallopRight(ms, a[baseB], a, baseA, na, 0);
        baseA += k;
        na -= k;
        if (na == 0)
            return;

        // Where does a end in b? Elements in b after that are already in place.
        nb = GallopLeft(ms, a[baseA + na - 1], a, baseB, nb, nb - 1);
        if (nb <= 0)
            return;

        if (na <= nb)
            MergeLo(ms, baseA, na, baseB, nb);
        else
            MergeHi(ms, baseA, na, baseB, nb);
    }

    /// <summary>
    ///     <c>gallop_left</c>: the index in <c>arr[at, at + n)</c> where <paramref name="key" /> belongs, left of any equal
    ///     element: <c>arr[at + k - 1] &lt; key &lt;= arr[at + k]</c>, found by galloping from <paramref name="hint" /> and
    ///     then binary search.
    /// </summary>
    /// <param name="ms">The sort state.</param>
    /// <param name="key">The element to place.</param>
    /// <param name="arr">The array holding the run (the list or the merge's temporary copy).</param>
    /// <param name="at">The run's first index in <paramref name="arr" />.</param>
    /// <param name="n">The run's length, at least 1.</param>
    /// <param name="hint">Where to start galloping, in [0, n).</param>
    /// <returns>The offset in [0, n].</returns>
    /// <typeparam name="T">The element type.</typeparam>
    private static int GallopLeft<T>(MergeState<T> ms, T key, T[] arr, int at, int n, int hint)
    {
        int a = at + hint;
        int lastOfs = 0;
        int ofs = 1;
        if (ms.LessThan(arr[a], key))
        {
            // arr[hint] < key: gallop right until arr[hint + lastOfs] < key <= arr[hint + ofs].
            int maxOfs = n - hint;
            while (ofs < maxOfs)
            {
                if (ms.LessThan(arr[a + ofs], key))
                {
                    lastOfs = ofs;
                    ofs = (ofs << 1) + 1;
                }
                else
                {
                    break;
                }
            }
            if (ofs > maxOfs)
                ofs = maxOfs;
            lastOfs += hint;
            ofs += hint;
        }
        else
        {
            // key <= arr[hint]: gallop left until arr[hint - ofs] < key <= arr[hint - lastOfs].
            int maxOfs = hint + 1;
            while (ofs < maxOfs)
            {
                if (ms.LessThan(arr[a - ofs], key))
                    break;
                lastOfs = ofs;
                ofs = (ofs << 1) + 1;
            }
            if (ofs > maxOfs)
                ofs = maxOfs;
            int k = lastOfs;
            lastOfs = hint - ofs;
            ofs = hint - k;
        }

        // arr[lastOfs] < key <= arr[ofs]: binary search with invariant arr[lastOfs - 1] < key <= arr[ofs].
        ++lastOfs;
        while (lastOfs < ofs)
        {
            int m = lastOfs + ((ofs - lastOfs) >> 1);
            if (ms.LessThan(arr[at + m], key))
                lastOfs = m + 1;
            else
                ofs = m;
        }
        return ofs;
    }

    /// <summary>
    ///     <c>gallop_right</c>: like <see cref="GallopLeft{T}" />, but right of any equal element:
    ///     <c>arr[at + k - 1] &lt;= key &lt; arr[at + k]</c>.
    /// </summary>
    /// <param name="ms">The sort state.</param>
    /// <param name="key">The element to place.</param>
    /// <param name="arr">The array holding the run (the list or the merge's temporary copy).</param>
    /// <param name="at">The run's first index in <paramref name="arr" />.</param>
    /// <param name="n">The run's length, at least 1.</param>
    /// <param name="hint">Where to start galloping, in [0, n).</param>
    /// <returns>The offset in [0, n].</returns>
    /// <typeparam name="T">The element type.</typeparam>
    private static int GallopRight<T>(MergeState<T> ms, T key, T[] arr, int at, int n, int hint)
    {
        int a = at + hint;
        int lastOfs = 0;
        int ofs = 1;
        if (ms.LessThan(key, arr[a]))
        {
            // key < arr[hint]: gallop left until arr[hint - ofs] <= key < arr[hint - lastOfs].
            int maxOfs = hint + 1;
            while (ofs < maxOfs)
            {
                if (ms.LessThan(key, arr[a - ofs]))
                {
                    lastOfs = ofs;
                    ofs = (ofs << 1) + 1;
                }
                else
                {
                    break;
                }
            }
            if (ofs > maxOfs)
                ofs = maxOfs;
            int k = lastOfs;
            lastOfs = hint - ofs;
            ofs = hint - k;
        }
        else
        {
            // arr[hint] <= key: gallop right until arr[hint + lastOfs] <= key < arr[hint + ofs].
            int maxOfs = n - hint;
            while (ofs < maxOfs)
            {
                if (ms.LessThan(key, arr[a + ofs]))
                    break;
                lastOfs = ofs;
                ofs = (ofs << 1) + 1;
            }
            if (ofs > maxOfs)
                ofs = maxOfs;
            lastOfs += hint;
            ofs += hint;
        }

        // arr[lastOfs] <= key < arr[ofs]: binary search with invariant arr[lastOfs - 1] <= key < arr[ofs].
        ++lastOfs;
        while (lastOfs < ofs)
        {
            int m = lastOfs + ((ofs - lastOfs) >> 1);
            if (ms.LessThan(key, arr[at + m]))
                ofs = m;
            else
                lastOfs = m + 1;
        }
        return ofs;
    }

    /// <summary>
    ///     <c>merge_lo</c>: merges adjacent runs <c>[pa, pa + na)</c> and <c>[pb, pb + nb)</c> left to right, the first run
    ///     (the shorter or equal one) copied aside, galloping once one side wins <c>min_gallop</c> times in a row.
    /// </summary>
    /// <param name="ms">The sort state.</param>
    /// <param name="pa">The first run's start; its first element belongs after b's first.</param>
    /// <param name="na">The first run's length, at least 1.</param>
    /// <param name="pb">The second run's start, <c>pa + na</c>.</param>
    /// <param name="nb">The second run's length, at least 1; its last element belongs after a's last.</param>
    /// <typeparam name="T">The element type.</typeparam>
    private static void MergeLo<T>(MergeState<T> ms, int pa, int na, int pb, int nb)
    {
        T[] a = ms.Keys;
        T[] tmp = ms.Temp(na);
        Array.Copy(a, pa, tmp, 0, na);
        int dest = pa;
        int ta = 0;

        a[dest++] = a[pb++];
        --nb;
        if (nb == 0)
            goto Succeed;
        if (na == 1)
            goto CopyB;

        int minGallop = ms.MinGallop;
        while (true)
        {
            int acount = 0;  // # of times A won in a row
            int bcount = 0;  // # of times B won in a row

            // The straightforward merge, until (if ever) one run appears to win consistently.
            while (true)
            {
                if (ms.LessThan(a[pb], tmp[ta]))
                {
                    a[dest++] = a[pb++];
                    ++bcount;
                    acount = 0;
                    --nb;
                    if (nb == 0)
                        goto Succeed;
                    if (bcount >= minGallop)
                        break;
                }
                else
                {
                    a[dest++] = tmp[ta++];
                    ++acount;
                    bcount = 0;
                    --na;
                    if (na == 1)
                        goto CopyB;
                    if (acount >= minGallop)
                        break;
                }
            }

            // Galloping, until neither run appears to be winning consistently anymore.
            ++minGallop;
            do
            {
                minGallop -= minGallop > 1 ? 1 : 0;
                ms.MinGallop = minGallop;
                int k = GallopRight(ms, a[pb], tmp, ta, na, 0);
                acount = k;
                if (k != 0)
                {
                    Array.Copy(tmp, ta, a, dest, k);
                    dest += k;
                    ta += k;
                    na -= k;
                    if (na == 1)
                        goto CopyB;
                    // Impossible for a consistent comparison, but NaN makes it possible.
                    if (na == 0)
                        goto Succeed;
                }
                a[dest++] = a[pb++];
                --nb;
                if (nb == 0)
                    goto Succeed;

                k = GallopLeft(ms, tmp[ta], a, pb, nb, 0);
                bcount = k;
                if (k != 0)
                {
                    Array.Copy(a, pb, a, dest, k);  // memmove: dest < pb, Array.Copy handles the overlap
                    dest += k;
                    pb += k;
                    nb -= k;
                    if (nb == 0)
                        goto Succeed;
                }
                a[dest++] = tmp[ta++];
                --na;
                if (na == 1)
                    goto CopyB;
            } while (acount >= MinGallopStart || bcount >= MinGallopStart);
            ++minGallop;  // penalize it for leaving galloping mode
            ms.MinGallop = minGallop;
        }

    Succeed:
        if (na != 0)
            Array.Copy(tmp, ta, a, dest, na);
        return;

    CopyB:
        // na == 1 and nb > 0: the last element of a belongs at the end of the merge.
        Array.Copy(a, pb, a, dest, nb);
        a[dest + nb] = tmp[ta];
    }

    /// <summary>
    ///     <c>merge_hi</c>: merges adjacent runs <c>[pa, pa + na)</c> and <c>[pb, pb + nb)</c> right to left, the second
    ///     run (the shorter one) copied aside, galloping once one side wins <c>min_gallop</c> times in a row.
    /// </summary>
    /// <param name="ms">The sort state.</param>
    /// <param name="pa">The first run's start; its first element belongs after b's first.</param>
    /// <param name="na">The first run's length, at least 1.</param>
    /// <param name="pb">The second run's start, <c>pa + na</c>.</param>
    /// <param name="nb">The second run's length, at least 1; its last element belongs after a's last.</param>
    /// <typeparam name="T">The element type.</typeparam>
    private static void MergeHi<T>(MergeState<T> ms, int pa, int na, int pb, int nb)
    {
        T[] a = ms.Keys;
        T[] tmp = ms.Temp(nb);
        Array.Copy(a, pb, tmp, 0, nb);
        int dest = pb + nb - 1;
        int ia = pa + na - 1;  // a's last remaining element, in the list
        int tb = nb - 1;       // b's last remaining element, in tmp

        a[dest--] = a[ia--];
        --na;
        if (na == 0)
            goto Succeed;
        if (nb == 1)
            goto CopyA;

        int minGallop = ms.MinGallop;
        while (true)
        {
            int acount = 0;  // # of times A won in a row
            int bcount = 0;  // # of times B won in a row

            // The straightforward merge, until (if ever) one run appears to win consistently.
            while (true)
            {
                if (ms.LessThan(tmp[tb], a[ia]))
                {
                    a[dest--] = a[ia--];
                    ++acount;
                    bcount = 0;
                    --na;
                    if (na == 0)
                        goto Succeed;
                    if (acount >= minGallop)
                        break;
                }
                else
                {
                    a[dest--] = tmp[tb--];
                    ++bcount;
                    acount = 0;
                    --nb;
                    if (nb == 1)
                        goto CopyA;
                    if (bcount >= minGallop)
                        break;
                }
            }

            // Galloping, until neither run appears to be winning consistently anymore.
            ++minGallop;
            do
            {
                minGallop -= minGallop > 1 ? 1 : 0;
                ms.MinGallop = minGallop;
                int k = GallopRight(ms, tmp[tb], a, pa, na, na - 1);
                k = na - k;
                acount = k;
                if (k != 0)
                {
                    dest -= k;
                    ia -= k;
                    Array.Copy(a, ia + 1, a, dest + 1, k);  // memmove: dest > ia, Array.Copy handles the overlap
                    na -= k;
                    if (na == 0)
                        goto Succeed;
                }
                a[dest--] = tmp[tb--];
                --nb;
                if (nb == 1)
                    goto CopyA;

                k = GallopLeft(ms, a[ia], tmp, 0, nb, nb - 1);
                k = nb - k;
                bcount = k;
                if (k != 0)
                {
                    dest -= k;
                    tb -= k;
                    Array.Copy(tmp, tb + 1, a, dest + 1, k);
                    nb -= k;
                    if (nb == 1)
                        goto CopyA;
                    // Impossible for a consistent comparison, but NaN makes it possible.
                    if (nb == 0)
                        goto Succeed;
                }
                a[dest--] = a[ia--];
                --na;
                if (na == 0)
                    goto Succeed;
            } while (acount >= MinGallopStart || bcount >= MinGallopStart);
            ++minGallop;  // penalize it for leaving galloping mode
            ms.MinGallop = minGallop;
        }

    Succeed:
        if (nb != 0)
            Array.Copy(tmp, 0, a, dest - (nb - 1), nb);
        return;

    CopyA:
        // nb == 1 and na > 0: the first element of b belongs at the front of the merge.
        Array.Copy(a, ia - na + 1, a, dest - na + 1, na);
        dest -= na;
        a[dest] = tmp[tb];
    }

    /// <summary>
    ///     CPython's <c>MergeState</c>: the list being sorted, the comparison, the adaptive gallop threshold, the merge
    ///     scratch space and the stack of pending runs with their powers.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    private sealed class MergeState<T>
    {
        /// <summary>The list's elements, sorted in place.</summary>
        internal readonly T[] Keys;

        /// <summary>CPython's <c>&lt;</c> for two elements.</summary>
        internal readonly Func<T, T, bool> LessThan;

        /// <summary><c>min_gallop</c>: adapts across merges, starting at 7.</summary>
        internal int MinGallop = MinGallopStart;

        /// <summary>The pending runs' start indices.</summary>
        internal readonly int[] Base = new int[MaxMergePending];

        /// <summary>The pending runs' lengths.</summary>
        internal readonly int[] Length = new int[MaxMergePending];

        /// <summary>The pending runs' powers (the topmost run's is set when the next run is found).</summary>
        internal readonly int[] Power = new int[MaxMergePending];

        /// <summary>The number of pending runs.</summary>
        internal int Count;

        /// <summary>The merge scratch space, grown on demand.</summary>
        private T[] _temp = Array.Empty<T>();

        /// <summary>Creates the state for one sort.</summary>
        /// <param name="keys">The elements to sort in place.</param>
        /// <param name="lessThan">CPython's <c>&lt;</c>.</param>
        internal MergeState(T[] keys, Func<T, T, bool> lessThan)
        {
            Keys = keys;
            LessThan = lessThan;
        }

        /// <summary><c>MERGE_GETMEM</c>: scratch space for at least <paramref name="need" /> elements.</summary>
        /// <param name="need">The elements a merge copies aside.</param>
        /// <returns>The scratch array, possibly longer than needed.</returns>
        internal T[] Temp(int need)
        {
            if (_temp.Length < need)
                _temp = new T[need];
            return _temp;
        }
    }
}
