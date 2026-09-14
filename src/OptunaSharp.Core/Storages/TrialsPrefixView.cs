using System;
using System.Collections;
using System.Collections.Generic;

namespace OptunaSharp.Storages;

/// <summary>
///     A zero-copy, fixed-length window over the first <see cref="Count" /> trials of a study's live trial
///     list — what <see cref="BaseStorage.GetTrialsPrefixView" /> returns. Constructing one is O(1) where
///     <c>GetAllTrials(deepcopy: false)</c> is O(N) (it copies the list), which is the difference between a
///     linear and a quadratic optimization loop for samplers that consult history on every ask.
/// </summary>
/// <remarks>
///     <para>
///         <b>Liveness.</b> The window's <em>membership</em> is frozen (indices <c>0..Count-1</c> at creation
///         time — appends after creation are invisible), but each read returns the trial's <em>current</em>
///         snapshot instance: storages that copy-on-write a trial on update (as
///         <see cref="InMemoryStorage" /> does) are seen through this view at their latest state, unlike a
///         copied list whose elements go stale. Samplers want exactly that — the freshest attrs of a frozen
///         membership.
///     </para>
///     <para>
///         <b>Footgun — concurrency.</b> Reads are lock-free against a backing <see cref="List{T}" /> whose
///         element writes are atomic reference stores, so reading while <em>elements</em> are replaced is
///         safe; reading while the list is <em>growing</em> (a resize moves the backing array) is safe only
///         because a resize replaces the array atomically and indices below <see cref="Count" /> exist in
///         both copies. Do not use this view to observe trials being appended concurrently — take it after
///         the mutation, as the samplers do (one ask/tell at a time per study).
///     </para>
/// </remarks>
public sealed class TrialsPrefixView : IReadOnlyList<FrozenTrial>
{
    private readonly List<FrozenTrial> _trials;

    /// <summary>Wraps the first <paramref name="count" /> elements of <paramref name="trials" />.</summary>
    /// <param name="trials">The live storage list; borrowed, never mutated through this view.</param>
    /// <param name="count">The frozen membership size; must not exceed the list's length at any read.</param>
    /// <exception cref="ArgumentNullException"><paramref name="trials" /> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count" /> is negative.</exception>
    public TrialsPrefixView(List<FrozenTrial> trials, int count)
    {
        _trials = trials ?? throw new ArgumentNullException(nameof(trials));
        if (count < 0)
            throw new ArgumentOutOfRangeException(nameof(count), count, "A prefix length cannot be negative.");
        Count = count;
    }

    /// <summary>The frozen membership size — appends to the backing list after construction are not visible.</summary>
    public int Count { get; }

    /// <summary>The current snapshot of trial number <paramref name="index" /> (trial numbers equal list positions).</summary>
    /// <param name="index">Zero-based trial position, below <see cref="Count" />.</param>
    /// <returns>The trial's latest stored instance (copy-on-write storages update it in place of older instances).</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index" /> is outside the frozen window.</exception>
    public FrozenTrial this[int index]
    {
        get
        {
            if ((uint)index >= (uint)Count)
                throw new ArgumentOutOfRangeException(nameof(index), index, $"The view holds {Count} trials.");
            return _trials[index];
        }
    }

    /// <summary>Enumerates the frozen window front to back, reading current instances.</summary>
    /// <returns>An enumerator over the first <see cref="Count" /> trials.</returns>
    public IEnumerator<FrozenTrial> GetEnumerator()
    {
        for (int i = 0; i < Count; i++)
            yield return _trials[i];
    }

    /// <inheritdoc />
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
