using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace OptunaSharp.Internal;

/// <summary>
///     A trial history searchable by trial number without a dictionary rebuilt on every ask (plan 15): a binary search
///     over a list in strictly ascending number order, which is how every storage returns a study's trials, and a
///     dictionary over the list only when it is not in that order.
/// </summary>
/// <remarks>
///     <para>
///         <b>Why.</b> The steady-state samplers (SMS-EMOA, MOEA/D) resolved their population by number or id through
///         dictionaries built over every completed trial on every ask: O(n) inserts and about half of their allocations
///         per trial. A lookup here costs O(log n) after one allocation-free order check over the list.
///     </para>
///     <para>
///         <b>Same answers.</b> In a list whose numbers strictly ascend, each number occurs at most once, so the binary
///         search finds exactly the record a dictionary keyed by number would hold. A list in any other order gets that
///         dictionary itself, built as the replaced code built it (a later record replaces an earlier one with the same
///         number), so even an unexpected storage order changes nothing.
///     </para>
///     <para>
///         <b>Compiled optimized from the first call.</b> <see cref="Over" /> and <see cref="TryGetValue" /> carry
///         <see cref="MethodImplOptions.AggressiveOptimization" />: as unoptimized tier-0 code the order check made the
///         first few thousand asks of a process slower than the dictionaries it replaced (measured on MOEA/D,
///         2,000-trial runs, until tiering caught up). They are small loops over a span, which profile-guided
///         optimization has nothing to add to.
///     </para>
///     <para>Holds the list by reference: it must not change while the lookup is in use (the samplers' lists are per-ask snapshots).</para>
/// </remarks>
internal readonly struct TrialsByNumber
{
    private readonly List<FrozenTrial> _trials;
    private readonly Dictionary<int, FrozenTrial>? _outOfOrder;

    /// <summary>Wraps a list and, when it is out of order, the dictionary that answers for it instead.</summary>
    /// <param name="trials">The history.</param>
    /// <param name="outOfOrder">The dictionary over <paramref name="trials" /> when they are not in ascending number order; otherwise null.</param>
    private TrialsByNumber(List<FrozenTrial> trials, Dictionary<int, FrozenTrial>? outOfOrder)
    {
        _trials = trials;
        _outOfOrder = outOfOrder;
    }

    /// <summary>
    ///     Whether the history is in strictly ascending trial-number order: then a stable sort by number would leave it as
    ///     it is, so a caller can skip one (MOEA/D's offspring scan).
    /// </summary>
    internal bool IsAscending => _outOfOrder is null;

    /// <summary>A lookup over <paramref name="trials" />: checks the order once, and builds a dictionary only if it must.</summary>
    /// <param name="trials">The history, normally in ascending trial-number order; not copied, so keep it unchanged while the lookup is used.</param>
    /// <returns>The lookup.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="trials" /> is null.</exception>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static TrialsByNumber Over(List<FrozenTrial> trials)
    {
        ArgumentNullException.ThrowIfNull(trials);
        ReadOnlySpan<FrozenTrial> span = CollectionsMarshal.AsSpan(trials);
        for (int i = 1; i < span.Length; i++)
        {
            if (span[i - 1].Number >= span[i].Number)
            {
                // Out of order (or a repeated number): the binary search could miss, so answer exactly as the replaced
                // code did, from a dictionary filled in list order.
                var byNumber = new Dictionary<int, FrozenTrial>(trials.Count);
                foreach (FrozenTrial t in trials)
                    byNumber[t.Number] = t;
                return new TrialsByNumber(trials, byNumber);
            }
        }
        return new TrialsByNumber(trials, null);
    }

    /// <summary>Whether the history holds a trial with this number, and that trial.</summary>
    /// <param name="number">The trial number to find.</param>
    /// <param name="trial">The trial with that number when found; otherwise null.</param>
    /// <returns>True when the history holds the number.</returns>
    /// <exception cref="NullReferenceException">The lookup is a <c>default</c> value rather than one made by <see cref="Over" />.</exception>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal bool TryGetValue(int number, out FrozenTrial trial)
    {
        if (_outOfOrder is not null)
            return _outOfOrder.TryGetValue(number, out trial!);

        ReadOnlySpan<FrozenTrial> span = CollectionsMarshal.AsSpan(_trials);
        int lo = 0;
        int hi = span.Length - 1;
        while (lo <= hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            int midNumber = span[mid].Number;
            if (midNumber == number)
            {
                trial = span[mid];
                return true;
            }
            if (midNumber < number)
                lo = mid + 1;
            else
                hi = mid - 1;
        }
        trial = null!;
        return false;
    }
}
