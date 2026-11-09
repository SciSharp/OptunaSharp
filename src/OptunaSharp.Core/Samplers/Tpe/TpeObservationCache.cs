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
///     <see cref="MaxColumns" /> the cache starts over rather than grow without bound.</para>
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

    private readonly Dictionary<(string Name, BaseDistribution Distribution), Column> _columns = new();
    private BaseStorage? _storage;
    private int _studyId;

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
        _columns.Clear();
        _storage = null;
        _studyId = 0;
    }

    /// <summary>
    ///     The columns for one search space, in its order, after starting over when the study changed or the cache grew past
    ///     <see cref="MaxColumns" />.
    /// </summary>
    /// <param name="study">The study being sampled; its id and storage identify the history the cache answers for.</param>
    /// <param name="names">The search space's parameter names, in order.</param>
    /// <param name="distributions">The search space's distributions, in the same order.</param>
    /// <returns>One column per parameter, aligned with <paramref name="names" />.</returns>
    internal Column[] ColumnsFor(Study study, string[] names, BaseDistribution[] distributions)
    {
        if (!ReferenceEquals(_storage, study.Storage) || _studyId != study.StudyId)
        {
            _columns.Clear();
            _storage = study.Storage;
            _studyId = study.StudyId;
        }
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
}
