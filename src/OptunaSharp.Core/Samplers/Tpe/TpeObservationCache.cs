using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using OptunaSharp.Distributions;
using OptunaSharp.Storages;

namespace OptunaSharp.Samplers.Tpe;

/// <summary>
///     The internal value of each parameter TPE has read from a finished trial, by trial number (plan 14; Rustuna's
///     "reuse completed observations across trials", ddd9c56 / e7a22d7). Every ask converts the whole history once per
///     parameter (twice in the independent mode, for the below and the above group); a finished trial never changes, so
///     its conversion is done once and later asks read a dense column instead of chasing each trial's parameter
///     dictionary and boxed value.
/// </summary>
/// <remarks>
///     <para><b>Same values.</b> A column is keyed by the parameter name and the search-space distribution the value is
///     converted with (value equality: equal distributions convert every external value alike), and holds exactly what
///     <see cref="BaseDistribution.ToInternalRepr" /> returned for that finished trial, or that the trial lacks the name.
///     Unfinished trials (constant liar's running trials, whose relative parameters still arrive) are never cached.</para>
///     <para><b>Lifetime.</b> The cache answers for one study at a time: a call for another study or storage (a sampler
///     reused across studies) starts over. Dynamic search spaces add a column per distinct distribution; past
///     <see cref="MaxColumns" /> the cache starts over rather than grow without bound. The storage is held weakly, as
///     <see cref="Internal.PerStudyState{TState}" /> holds it, so a long-lived sampler does not keep the last storage it
///     served alive.</para>
///     <para><b>Recreated studies.</b> The storage object and the study id do not identify a history on their own: SQLite
///     reuses the highest deleted study and trial ids, and a gRPC proxy or a custom storage passes them through as one
///     storage object, so a study deleted and recreated under the same id shows the cache different trials under the
///     numbers it holds. Each cached row therefore keeps its trial's identity (storage id and start time, the
///     <see cref="Internal.TrialAnchor" /> rule), and every <see cref="ColumnsFor" /> checks the finished trials it is
///     given against it before any row is read: one mismatch and the cache starts over.</para>
///     <para><b>Concurrency.</b> Not thread-safe; TPE uses it only inside sampling, which runs under the sampler's lock
///     (each <see cref="SamplerConcurrency.PerWorker" /> instance has its own cache).</para>
/// </remarks>
internal sealed class TpeObservationCache
{
    /// <summary>How many (name, distribution) columns are kept before the cache starts over.</summary>
    internal const int MaxColumns = 256;

    /// <summary>The state of a column entry whose trial has not been read yet.</summary>
    internal const byte Unknown = 0;

    /// <summary>The state of a column entry holding the finished trial's internal value.</summary>
    internal const byte Present = 1;

    /// <summary>The state of a column entry whose finished trial lacks the parameter.</summary>
    internal const byte Absent = 2;

    /// <summary>The trial id recorded for a trial number no column holds a row for (storage ids are never negative).</summary>
    private const int NoRow = -1;

    /// <summary>The start time recorded for a cached trial that had none, which then matches any start time.</summary>
    private const long NoStart = long.MinValue;

    private readonly Dictionary<(string Name, BaseDistribution Distribution), Column> _columns = new();
    private WeakReference<BaseStorage>? _storage;
    private int _studyId;
    // The identity of the trial each cached row was read from, by trial number: its storage id (NoRow where no row is
    // cached) and its start time in ticks (NoStart where it had none). Shared by every column, since a row is one trial.
    private int[] _rowTrialIds = Array.Empty<int>();
    private long[] _rowStartTicks = Array.Empty<long>();

    /// <summary>
    ///     How many times the cache started over because a trial number it holds now carries another trial (a study
    ///     recreated under a reused id), for the tests that check a live study never triggers it.
    /// </summary>
    internal int IdentityRestarts { get; private set; }

    /// <summary>One parameter's cached values by trial number, with each entry's state.</summary>
    internal sealed class Column
    {
        private double[] _values = Array.Empty<double>();
        private byte[] _states = Array.Empty<byte>();

        /// <summary>The state of trial <paramref name="number" />'s entry: <see cref="Unknown" />, <see cref="Present" /> or <see cref="Absent" />.</summary>
        /// <param name="number">The trial's number in its study.</param>
        /// <returns>The entry's state; <see cref="Unknown" /> beyond the entries written so far.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal byte StateOf(int number) => (uint)number < (uint)_states.Length ? _states[number] : Unknown;

        /// <summary>The internal value cached for trial <paramref name="number" />; meaningful only when its state is <see cref="Present" />.</summary>
        /// <param name="number">The trial's number in its study.</param>
        /// <returns>The cached internal value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal double ValueOf(int number) => _values[number];

        /// <summary>Records the internal value a finished trial's parameter converts to.</summary>
        /// <param name="number">The trial's number in its study.</param>
        /// <param name="value">The value <see cref="BaseDistribution.ToInternalRepr" /> returned.</param>
        internal void SetPresent(int number, double value)
        {
            EnsureCapacity(number);
            _values[number] = value;
            _states[number] = Present;
        }

        /// <summary>Records that a finished trial lacks the parameter.</summary>
        /// <param name="number">The trial's number in its study.</param>
        internal void SetAbsent(int number)
        {
            EnsureCapacity(number);
            _states[number] = Absent;
        }

        /// <summary>Grows both arrays (doubling) so <paramref name="number" /> is a valid index; new entries are <see cref="Unknown" />.</summary>
        /// <param name="number">The index that must become valid.</param>
        private void EnsureCapacity(int number)
        {
            if (number < _states.Length)
                return;
            int size = Math.Max(Math.Max(16, number + 1), 2 * _states.Length);
            Array.Resize(ref _values, size);
            Array.Resize(ref _states, size);
        }
    }

    /// <summary>How many (name, distribution) columns the cache holds now.</summary>
    internal int ColumnCount => _columns.Count;

    /// <summary>
    ///     Forgets everything, as a new sampler would know nothing. Sampling never needs it (a changed study resets the cache
    ///     on its own); tests use it to compare a warm cache with a cold one.
    /// </summary>
    internal void Reset()
    {
        ForgetRows();
        _storage = null;
        _studyId = 0;
    }

    /// <summary>
    ///     The columns for one search space, in its order, after starting over when the study changed, when a trial number
    ///     the cache holds now carries another trial, or when the cache grew past <see cref="MaxColumns" />.
    /// </summary>
    /// <param name="study">The study being sampled; its id and storage identify the history the cache answers for.</param>
    /// <param name="trials">
    ///     The trials the caller is about to read rows for. Each one whose number has a cached row must be the trial that
    ///     row was read from (same storage id and, when both are known, start time), or the cache starts over before
    ///     anything is read.
    /// </param>
    /// <param name="names">The search space's parameter names, in order.</param>
    /// <param name="distributions">The search space's distributions, in the same order.</param>
    /// <returns>One column per parameter, aligned with <paramref name="names" />.</returns>
    internal Column[] ColumnsFor(
        Study study, IReadOnlyList<FrozenTrial> trials, string[] names, BaseDistribution[] distributions)
    {
        BaseStorage storage = study.Storage;
        if (_storage is null || !_storage.TryGetTarget(out BaseStorage? held) || !ReferenceEquals(held, storage)
            || _studyId != study.StudyId)
        {
            ForgetRows();
            _storage = new WeakReference<BaseStorage>(storage);
            _studyId = study.StudyId;
        }
        else if (!RowsMatch(trials))
        {
            // The same storage object and study id now hold another history: the study was deleted and recreated under a
            // reused id. Nothing cached for the old one may be read, so every row goes, before the caller reads any.
            ForgetRows();
            IdentityRestarts++;
        }
        // Only the columns go: the row identities still describe this study's trials, and a row without column entries is
        // never read.
        if (_columns.Count + names.Length > MaxColumns)
            _columns.Clear();

        var result = new Column[names.Length];
        for (int i = 0; i < names.Length; i++)
        {
            var key = (names[i], distributions[i]);
            if (!_columns.TryGetValue(key, out Column? column))
            {
                column = new Column();
                _columns[key] = column;
            }
            result[i] = column;
        }
        return result;
    }

    /// <summary>
    ///     Records which trial a finished trial's row was read from; called whenever a column records its value or its
    ///     absence, so that <see cref="ColumnsFor" /> can tell the trial from another one under the same number later.
    /// </summary>
    /// <param name="trial">The finished trial whose row a column now holds.</param>
    internal void RememberTrial(FrozenTrial trial)
    {
        int number = trial.Number;
        if (number >= _rowTrialIds.Length)
        {
            // Doubling, like the columns; the new entries hold no row until a trial is recorded there.
            int oldSize = _rowTrialIds.Length;
            int size = Math.Max(Math.Max(16, number + 1), 2 * oldSize);
            Array.Resize(ref _rowTrialIds, size);
            Array.Resize(ref _rowStartTicks, size);
            Array.Fill(_rowTrialIds, NoRow, oldSize, size - oldSize);
        }
        _rowTrialIds[number] = trial.TrialId;
        _rowStartTicks[number] = trial.DatetimeStart?.Ticks ?? NoStart;
    }

    /// <summary>
    ///     Whether every trial in <paramref name="trials" /> whose number has a cached row is the trial that row was read
    ///     from: the same storage id and, when both start times are known, the same start time (the
    ///     <see cref="Internal.TrialAnchor" /> rule; SQLite reuses deleted trial ids too, but a recreated trial starts
    ///     later).
    /// </summary>
    /// <param name="trials">The trials about to be read; numbers without a cached row are not checked.</param>
    /// <returns><c>true</c> when no cached row belongs to another trial.</returns>
    private bool RowsMatch(IReadOnlyList<FrozenTrial> trials)
    {
        int[] ids = _rowTrialIds;
        for (int i = 0; i < trials.Count; i++)
        {
            FrozenTrial trial = trials[i];
            int number = trial.Number;
            if ((uint)number >= (uint)ids.Length || ids[number] == NoRow)
                continue;
            if (ids[number] != trial.TrialId)
                return false;
            long cachedStart = _rowStartTicks[number];
            if (cachedStart != NoStart && trial.DatetimeStart is { } start && start.Ticks != cachedStart)
                return false;
        }
        return true;
    }

    /// <summary>Drops every column and every row identity; the study and storage the cache answers for stay.</summary>
    private void ForgetRows()
    {
        _columns.Clear();
        _rowTrialIds = Array.Empty<int>();
        _rowStartTicks = Array.Empty<long>();
    }
}
