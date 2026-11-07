using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Storages.Journal;

/// <summary>
///     The journal snapshot codecs of <see cref="JournalStorageReplayResult" />: the binary v2 format it writes (plan 10,
///     item 2.7) and the dispatch that restores v2 and the older v1 JSON format alike.
/// </summary>
/// <remarks>
///     <para>
///         <b>Why a binary format (the owner's choice, 2026-11-07).</b> Snapshots of 20,000 trials took 446 ms to save,
///         under the storage's lock, and 736 ms to restore at a cold start (Release builds), so a snapshot bought a
///         standalone cold start little. Most of that went to the JSON text and the dictionary tree PyJson builds from
///         it. v2 writes the replay state straight into bytes and reads it straight back into trials: doubles as their
///         eight bytes, integers as variable-length numbers, each distribution's JSON once in a table the trials refer
///         to by index, and only attribute values through a small tagged encoding of PyJson's object model.
///     </para>
///     <para>
///         <b>Compatibility.</b> A v2 snapshot restores exactly what restoring the v1 snapshot of the same state gives:
///         the same studies and trials in the same order, every value equal (NaN payloads canonical, as JSON's
///         <c>NaN</c> makes them), datetimes through the same UTC microseconds, distributions through the same JSON
///         text. The parity tests (<c>JournalSnapshotV2Tests</c>) hold it to that against <see cref="SerializeV1" /> and
///         <see cref="TryDeserializeV1" />. v1 snapshots, which older OptunaSharp versions write, still restore. An
///         older version cannot read v2: it logs <c>Failed to restore `JournalStorageReplayResult`.</c> and replays the
///         whole journal, which the owner accepted for the speed. Python never reads OptunaSharp's snapshot key.
///     </para>
///     <para>
///         <b>Layout</b> (little-endian; "uint" and "int" are LEB128 variable-length numbers, "int" zigzag-encoded):
///     </para>
///     <code>
///     header        "OSJS" u16 version=2 u16 flags=0
///     body          int log_number_read, int next_study_id,
///                   uint n, n x string distribution JSON (the table),
///                   uint n, n x study: int id, string name, uint n + n x int direction,
///                                      dict user_attrs, dict system_attrs, uint n + n x int trial_id
///                   uint n, n x trial: int id, int number, int state,
///                                      u8 values kind (0 none, 1 one: f64, 2 several: uint n + n x f64),
///                                      u8 + i64 UTC ticks for datetime_start, then datetime_complete (0 = null),
///                                      uint n + n x (string name, uint table index): distributions,
///                                      uint n + n x (string name, f64 internal value): params,
///                                      dict user_attrs, dict system_attrs,
///                                      uint n + n x (int step, f64 value): intermediate values, int study_id
///     string        uint (chars &lt;&lt; 1 | utf16), then the chars as Latin-1 bytes or UTF-16 code units
///     dict          uint n, n x (string key, value)
///     value         u8 tag: 0 null, 1 false, 2 true, 3 int, 4 f64, 5 string, 6 list (uint n + values), 7 dict
///     </code>
/// </remarks>
internal sealed partial class JournalStorageReplayResult
{
    /// <summary>The four bytes every v2 snapshot starts with: <c>OSJS</c>, which neither JSON nor a pickle starts with.</summary>
    internal static ReadOnlySpan<byte> SnapshotV2Magic => "OSJS"u8;

    /// <summary>The format version v2 snapshots carry after <see cref="SnapshotV2Magic" />.</summary>
    internal const ushort SnapshotV2Version = 2;

    /// <summary>The header's size: the magic, the version and the flags.</summary>
    private const int SnapshotV2HeaderSize = 8;

    // The last snapshot's size, so the next save starts with a buffer that fits instead of growing to it.
    private int _snapshotSizeHint;

    /// <summary>
    ///     Serializes the replay state for <see cref="IBaseJournalSnapshot" /> backends, in the binary v2 format.
    ///     [Misaligned] Not Python's pickle; snapshots are an OptunaSharp-only cache under OptunaSharp's own key.
    /// </summary>
    /// <returns>The snapshot's bytes: v2, or v1 JSON when the v2 writer fails.</returns>
    /// <exception cref="NumSharp.TypeError">An attribute value has no JSON representation (from the v1 fallback).</exception>
    /// <exception cref="NumSharp.RuntimeError">An attribute value nests too deeply for the v1 fallback too.</exception>
    /// <remarks>
    ///     Callers hold the storage's lock, as Python saves under its own, so the state cannot change while it is
    ///     written. A failure of the v2 writer, which no value parsed from a journal record can cause, falls back to
    ///     <see cref="SerializeV1" />, so the outcome is at worst what it was before v2: v1's bytes or v1's exception.
    /// </remarks>
    internal byte[] Serialize()
    {
        try
        {
            return SerializeV2();
        }
        catch (Exception)
        {
            return SerializeV1();
        }
    }

    /// <summary>
    ///     Restores a snapshot of either format: v2 (<see cref="SnapshotV2Magic" />) or the older v1 JSON.
    /// </summary>
    /// <param name="snapshot">The snapshot's bytes, as a backend stored them.</param>
    /// <returns>The restored replay state, or <c>null</c> for any format or content failure (the caller then replays the journal).</returns>
    internal static JournalStorageReplayResult? TryDeserialize(byte[] snapshot)
        => IsSnapshotV2(snapshot) ? TryDeserializeV2(snapshot) : TryDeserializeV1(snapshot);

    /// <summary>Whether bytes start like a v2 snapshot: the magic, then version 2.</summary>
    /// <param name="snapshot">The snapshot's bytes.</param>
    /// <returns><c>true</c> for a v2 header; v1 JSON starts with <c>{</c> and Python's pickle with 0x80.</returns>
    internal static bool IsSnapshotV2(ReadOnlySpan<byte> snapshot)
        => snapshot.Length >= SnapshotV2HeaderSize
           && snapshot.StartsWith(SnapshotV2Magic)
           && BinaryPrimitives.ReadUInt16LittleEndian(snapshot.Slice(4)) == SnapshotV2Version;

    /// <summary>Writes the replay state in the v2 layout.</summary>
    /// <returns>The snapshot's bytes.</returns>
    /// <exception cref="Exception">Any failure; <see cref="Serialize" /> then writes v1, whose outcome it keeps.</exception>
    private byte[] SerializeV2()
    {
        var writer = new SnapshotWriter(Math.Max(_snapshotSizeHint, 4096));
        try
        {
            writer.WriteRaw(SnapshotV2Magic);
            writer.WriteUInt16(SnapshotV2Version);
            writer.WriteUInt16(0);
            writer.WriteInt(LogNumberRead);
            writer.WriteInt(_nextStudyId);

            // The distribution table: each distinct JSON text once, trials refer to it by index. Objects are looked
            // up by reference first, which hits for nearly every trial since replay shares them (JournalDistributionCache),
            // so Distribution.ToJson runs once per distinct object rather than once per parameter. The lookup lives for
            // this save only, so a distribution with a public setter changed between saves is written as it is now.
            var tableByObject = new Dictionary<BaseDistribution, int>(ReferenceEqualityComparer.Instance);
            var tableByJson = new Dictionary<string, int>(StringComparer.Ordinal);
            var table = new List<string>();
            foreach (var (_, trial) in _trials)
            {
                foreach (var (_, distribution) in trial.DistributionsDict)
                {
                    if (tableByObject.ContainsKey(distribution))
                        continue;
                    string json = Distribution.ToJson(distribution);
                    if (!tableByJson.TryGetValue(json, out int index))
                    {
                        index = table.Count;
                        table.Add(json);
                        tableByJson[json] = index;
                    }
                    tableByObject[distribution] = index;
                }
            }
            writer.WriteUInt((ulong)table.Count);
            foreach (string json in table)
                writer.WriteString(json);

            writer.WriteUInt((ulong)_studies.Count);
            foreach (var (studyId, study) in _studies)
            {
                writer.WriteInt(studyId);
                writer.WriteString(study.StudyName);
                writer.WriteUInt((ulong)study.Directions.Count);
                foreach (StudyDirection direction in study.Directions)
                    writer.WriteInt((int)direction);
                writer.WriteDictionary(study.UserAttrs);
                writer.WriteDictionary(study.SystemAttrs);
                List<int> trialIds = _studyIdToTrialIds[studyId];
                writer.WriteUInt((ulong)trialIds.Count);
                foreach (int trialId in trialIds)
                    writer.WriteInt(trialId);
            }

            writer.WriteUInt((ulong)_trials.Count);
            foreach (var (trialId, trial) in _trials)
            {
                writer.WriteInt(trialId);
                writer.WriteInt(trial.Number);
                writer.WriteInt((int)trial.State);

                // v1's two fields: "value" for exactly one objective, "values" for several, neither otherwise.
                List<double>? values = trial.Values;
                if (values is { Count: 1 })
                {
                    writer.WriteByte(1);
                    writer.WriteDouble(values[0]);
                }
                else if (values is { Count: > 1 })
                {
                    writer.WriteByte(2);
                    writer.WriteUInt((ulong)values.Count);
                    foreach (double value in values)
                        writer.WriteDouble(value);
                }
                else
                {
                    writer.WriteByte(0);
                }

                writer.WriteDateTime(trial.DatetimeStart);
                writer.WriteDateTime(trial.DatetimeComplete);

                OrderedDict<string, BaseDistribution> distributions = trial.DistributionsDict;
                writer.WriteUInt((ulong)distributions.Count);
                foreach (var (name, distribution) in distributions)
                {
                    writer.WriteString(name);
                    writer.WriteUInt((ulong)tableByObject[distribution]);
                }

                // The internal representation, as v1 writes it: restoring converts it back through the distribution.
                OrderedDict<string, object?> @params = trial.ParamsDict;
                writer.WriteUInt((ulong)@params.Count);
                foreach (var (name, param) in @params)
                {
                    writer.WriteString(name);
                    writer.WriteDouble(distributions[name].ToInternalRepr(param));
                }

                writer.WriteDictionary(trial.UserAttrsDict);
                writer.WriteDictionary(trial.SystemAttrsDict);

                OrderedDict<long, double> intermediate = trial.IntermediateValues;
                writer.WriteUInt((ulong)intermediate.Count);
                foreach (var (step, value) in intermediate)
                {
                    writer.WriteInt(step);
                    writer.WriteDouble(value);
                }

                writer.WriteInt(_trialIdToStudyId[trialId]);
            }

            byte[] snapshot = writer.ToArray();
            _snapshotSizeHint = snapshot.Length;
            return snapshot;
        }
        finally
        {
            writer.Release();
        }
    }

    /// <summary>Restores a v2 snapshot; <c>null</c> on any failure, as <see cref="TryDeserializeV1" /> does.</summary>
    /// <param name="snapshot">The snapshot's bytes, starting with a v2 header.</param>
    /// <returns>The restored replay state, or <c>null</c> when the bytes are not a complete, well-formed v2 snapshot.</returns>
    /// <remarks>
    ///     Builds every study and trial with the arguments <see cref="TryDeserializeV1" /> passes for the same state, so
    ///     the two formats restore equal results. Counts are checked against the bytes left before anything is
    ///     allocated, so corrupt bytes fail instead of asking for huge arrays, and trailing bytes fail too.
    /// </remarks>
    private static JournalStorageReplayResult? TryDeserializeV2(ReadOnlySpan<byte> snapshot)
    {
        try
        {
            var reader = new SnapshotReader(snapshot);
            reader.Skip(4);   // the magic, already checked
            if (reader.ReadUInt16() != SnapshotV2Version || reader.ReadUInt16() != 0)
                return null;

            var result = new JournalStorageReplayResult(string.Empty)
            {
                LogNumberRead = reader.ReadInt32(),
                _nextStudyId = reader.ReadInt32(),
                _snapshotSizeHint = snapshot.Length,
            };

            int tableCount = reader.ReadCount(minBytesEach: 1);
            var tableJson = new string[tableCount];
            for (int i = 0; i < tableCount; i++)
                tableJson[i] = reader.ReadString();
            // Each table entry is parsed through the result's cache: a shareable distribution is then handed to every
            // trial that refers to it, as replaying the records would; any other is parsed per use, as v1 did.
            var shared = new BaseDistribution?[tableCount];

            // Six fields of at least one byte each: id, name, direction count, two attribute counts, trial id count.
            int studyCount = reader.ReadCount(minBytesEach: 6);
            for (int i = 0; i < studyCount; i++)
            {
                int studyId = reader.ReadInt32();
                string studyName = reader.ReadString();
                int directionCount = reader.ReadCount(minBytesEach: 1);
                var directions = new List<StudyDirection>(directionCount);
                for (int d = 0; d < directionCount; d++)
                    directions.Add((StudyDirection)reader.ReadInt32());
                OrderedDict<string, object?> userAttrs = reader.ReadDictionary(depth: 0);
                OrderedDict<string, object?> systemAttrs = reader.ReadDictionary(depth: 0);
                int trialIdCount = reader.ReadCount(minBytesEach: 1);
                var trialIds = new List<int>(trialIdCount);
                for (int t = 0; t < trialIdCount; t++)
                    trialIds.Add(reader.ReadInt32());

                result._studies[studyId] = new FrozenStudy(
                    studyName: studyName,
                    direction: null,
                    userAttrs: userAttrs,
                    systemAttrs: systemAttrs,
                    studyId: studyId,
                    directions: directions);
                result._studyIdToTrialIds[studyId] = trialIds;
            }

            // Twelve fields of at least one byte each: id, number, state, values kind, two datetime flags, three counts,
            // two attribute dictionaries' counts and the study id.
            int trialCount = reader.ReadCount(minBytesEach: 12);
            for (int i = 0; i < trialCount; i++)
            {
                int trialId = reader.ReadInt32();
                int number = reader.ReadInt32();
                var state = (TrialState)reader.ReadInt32();

                double? value = null;
                List<double>? values = null;
                switch (reader.ReadByte())
                {
                    case 0:
                        break;
                    case 1:
                        value = reader.ReadDouble();
                        break;
                    case 2:
                        int valueCount = reader.ReadCount(minBytesEach: 8);
                        values = new List<double>(valueCount);
                        for (int v = 0; v < valueCount; v++)
                            values.Add(reader.ReadDouble());
                        break;
                    default:
                        return null;
                }

                DateTime? datetimeStart = reader.ReadDateTime();
                DateTime? datetimeComplete = reader.ReadDateTime();

                int distributionCount = reader.ReadCount(minBytesEach: 2);
                var distributions = new OrderedDict<string, BaseDistribution>(distributionCount);
                for (int d = 0; d < distributionCount; d++)
                {
                    string name = reader.ReadString();
                    int index = checked((int)reader.ReadUInt());
                    if ((uint)index >= (uint)tableCount)
                        return null;
                    distributions[name] = result.ResolveTableDistribution(tableJson, shared, index);
                }

                int paramCount = reader.ReadCount(minBytesEach: 9);
                var @params = new OrderedDict<string, object?>(paramCount);
                for (int p = 0; p < paramCount; p++)
                {
                    string name = reader.ReadString();
                    double internalValue = reader.ReadDouble();
                    // v1's conversion from the internal representation; a name without a distribution fails as in v1.
                    @params[name] = distributions[name].ToExternalRepr(internalValue);
                }

                OrderedDict<string, object?> userAttrs = reader.ReadDictionary(depth: 0);
                OrderedDict<string, object?> systemAttrs = reader.ReadDictionary(depth: 0);

                int intermediateCount = reader.ReadCount(minBytesEach: 9);
                var intermediateValues = new OrderedDict<long, double>(intermediateCount);
                for (int s = 0; s < intermediateCount; s++)
                {
                    long step = reader.ReadInt64();
                    intermediateValues[step] = reader.ReadDouble();
                }

                int studyId = reader.ReadInt32();

                result._trials[trialId] = new FrozenTrial(
                    trialId: trialId,
                    number: number,
                    state: state,
                    @params: @params,
                    distributions: distributions,
                    userAttrs: userAttrs,
                    systemAttrs: systemAttrs,
                    value: value,
                    values: values,
                    intermediateValues: intermediateValues,
                    datetimeStart: datetimeStart,
                    datetimeComplete: datetimeComplete);
                result._trialIdToStudyId[trialId] = studyId;
            }

            return reader.AtEnd ? result : null;
        }
        catch (Exception)
        {
            // Malformed, truncated or corrupt bytes, or a value v1 would refuse as well: the caller replays the journal.
            return null;
        }
    }

    /// <summary>
    ///     The distribution of a v2 table entry for one trial: the shared object for a shareable distribution, else a
    ///     new parse per use, as <see cref="TryDeserializeV1" /> builds one per trial.
    /// </summary>
    /// <param name="tableJson">The table's JSON texts.</param>
    /// <param name="shared">Per entry, its object once it is known to be shareable.</param>
    /// <param name="index">The entry.</param>
    /// <returns>The distribution for the trial.</returns>
    /// <exception cref="Exception">The entry is not a valid distribution (the errors of <see cref="Distribution.FromJson" />).</exception>
    private BaseDistribution ResolveTableDistribution(string[] tableJson, BaseDistribution?[] shared, int index)
    {
        if (shared[index] is { } known)
            return known;
        // Through the cache, so the records replayed after this restore get the same objects as the restored trials;
        // a deprecated distribution is parsed again for every trial, which keeps its FutureWarning per use as in v1.
        BaseDistribution distribution = _distributionCache.FromJson(tableJson[index]);
        if (JournalDistributionCache.IsShareable(distribution))
            shared[index] = distribution;
        return distribution;
    }

    /// <summary>
    ///     Writes a v2 snapshot into a pooled buffer: little-endian fixed-size numbers, LEB128 variable-length integers,
    ///     Latin-1 or UTF-16 strings, and PyJson's object model as tagged values.
    /// </summary>
    private sealed class SnapshotWriter
    {
        private byte[] _buffer;
        private int _length;

        /// <summary>Rents a buffer of at least the given size.</summary>
        /// <param name="initialCapacity">The expected snapshot size; the buffer grows past it when needed.</param>
        internal SnapshotWriter(int initialCapacity)
        {
            _buffer = ArrayPool<byte>.Shared.Rent(initialCapacity);
        }

        /// <summary>Returns the buffer to the pool; the writer is unusable afterwards.</summary>
        internal void Release()
        {
            byte[] buffer = _buffer;
            _buffer = Array.Empty<byte>();
            if (buffer.Length > 0)
                ArrayPool<byte>.Shared.Return(buffer);
        }

        /// <summary>The bytes written so far, as a new array sized exactly.</summary>
        /// <returns>The snapshot.</returns>
        internal byte[] ToArray() => _buffer.AsSpan(0, _length).ToArray();

        /// <summary>Grows the buffer to hold <paramref name="count" /> more bytes.</summary>
        /// <param name="count">The bytes about to be written.</param>
        /// <returns>The writable span at the current position.</returns>
        /// <exception cref="OverflowException">The snapshot would exceed 2 GB.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Span<byte> Reserve(int count)
        {
            if (_buffer.Length - _length < count)
                Grow(count);
            return _buffer.AsSpan(_length);
        }

        /// <summary>Moves to a pooled buffer at least twice as large that also fits <paramref name="count" /> more bytes.</summary>
        /// <param name="count">The bytes about to be written.</param>
        /// <exception cref="OverflowException">The snapshot would exceed 2 GB.</exception>
        private void Grow(int count)
        {
            int needed = checked(_length + count);
            byte[] larger = ArrayPool<byte>.Shared.Rent(Math.Max(needed, (int)Math.Min(int.MaxValue, 2L * _buffer.Length)));
            _buffer.AsSpan(0, _length).CopyTo(larger);
            ArrayPool<byte>.Shared.Return(_buffer);
            _buffer = larger;
        }

        /// <summary>Appends bytes as they are.</summary>
        /// <param name="bytes">The bytes.</param>
        internal void WriteRaw(ReadOnlySpan<byte> bytes)
        {
            bytes.CopyTo(Reserve(bytes.Length));
            _length += bytes.Length;
        }

        /// <summary>Appends one byte.</summary>
        /// <param name="value">The byte.</param>
        internal void WriteByte(byte value)
        {
            Reserve(1)[0] = value;
            _length++;
        }

        /// <summary>Appends a little-endian 16-bit unsigned integer.</summary>
        /// <param name="value">The integer.</param>
        internal void WriteUInt16(ushort value)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(Reserve(2), value);
            _length += 2;
        }

        /// <summary>Appends an unsigned LEB128 integer: seven bits a byte, the high bit set on all but the last.</summary>
        /// <param name="value">The integer.</param>
        internal void WriteUInt(ulong value)
        {
            Span<byte> span = Reserve(10);
            int n = 0;
            while (value >= 0x80)
            {
                span[n++] = (byte)(value | 0x80);
                value >>= 7;
            }
            span[n++] = (byte)value;
            _length += n;
        }

        /// <summary>Appends a signed integer, zigzag-encoded (0, -1, 1, -2, ... become 0, 1, 2, 3, ...) then LEB128.</summary>
        /// <param name="value">The integer.</param>
        internal void WriteInt(long value) => WriteUInt((ulong)((value << 1) ^ (value >> 63)));

        /// <summary>Appends a double's eight bytes, little-endian, every bit kept.</summary>
        /// <param name="value">The double.</param>
        internal void WriteDouble(double value)
        {
            BinaryPrimitives.WriteDoubleLittleEndian(Reserve(8), value);
            _length += 8;
        }

        /// <summary>
        ///     Appends a nullable datetime as v1 stores it: the UTC instant (<see cref="DateTime.ToUniversalTime" />)
        ///     truncated to microseconds, as v1's <c>ffffff</c> format truncates it.
        /// </summary>
        /// <param name="value">The datetime, naive local time as the replay state keeps it, or <c>null</c>.</param>
        internal void WriteDateTime(DateTime? value)
        {
            if (value is null)
            {
                WriteByte(0);
                return;
            }
            long ticks = value.Value.ToUniversalTime().Ticks;
            ticks -= ticks % 10;
            WriteByte(1);
            BinaryPrimitives.WriteInt64LittleEndian(Reserve(8), ticks);
            _length += 8;
        }

        /// <summary>
        ///     Appends a string: its length shifted left with a flag, then Latin-1 bytes when every character fits one
        ///     byte, else its UTF-16 code units, so lone surrogates survive as v1's <c>\u</c> escapes keep them.
        /// </summary>
        /// <param name="value">The string.</param>
        /// <exception cref="OverflowException">The string would exceed the snapshot's 2 GB.</exception>
        internal void WriteString(string value)
        {
            ReadOnlySpan<char> chars = value.AsSpan();
            if (!chars.ContainsAnyExceptInRange((char)0, (char)0xFF))
            {
                WriteUInt((ulong)chars.Length << 1);
                Span<byte> span = Reserve(chars.Length);
                Encoding.Latin1.GetBytes(chars, span);
                _length += chars.Length;
                return;
            }

            WriteUInt(((ulong)chars.Length << 1) | 1);
            int byteCount = checked(chars.Length * 2);
            Span<byte> utf16 = Reserve(byteCount);
            if (BitConverter.IsLittleEndian)
            {
                MemoryMarshal.AsBytes(chars).CopyTo(utf16);
            }
            else
            {
                for (int i = 0; i < chars.Length; i++)
                    BinaryPrimitives.WriteUInt16LittleEndian(utf16.Slice(i * 2), chars[i]);
            }
            _length += byteCount;
        }

        /// <summary>Appends an attribute dictionary: its count, then each key and value in order.</summary>
        /// <param name="dictionary">The dictionary.</param>
        /// <exception cref="Exception">A value cannot be written; <see cref="Serialize" /> then falls back to v1.</exception>
        internal void WriteDictionary(OrderedDict<string, object?> dictionary)
        {
            WriteUInt((ulong)dictionary.Count);
            foreach (var (key, value) in dictionary)
            {
                WriteString(key);
                WriteValue(value, depth: 0);
            }
        }

        /// <summary>
        ///     Appends a value of PyJson's object model as a tag and its payload. Any other value is first passed through
        ///     JSON, as v1 would write and read it back, so both formats restore the same value.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="depth">The number of lists and dictionaries enclosing it within the attribute.</param>
        /// <exception cref="InvalidOperationException">The value nests deeper than <see cref="PyJson.MaxDepth" /> or this thread's stack allows.</exception>
        /// <exception cref="NumSharp.TypeError">A value outside the object model has no JSON representation.</exception>
        private void WriteValue(object? value, int depth)
        {
            switch (value)
            {
                case null:
                    WriteByte(0);
                    return;
                case bool flag:
                    WriteByte(flag ? (byte)2 : (byte)1);
                    return;
                case long integer:
                    WriteByte(3);
                    WriteInt(integer);
                    return;
                case double number:
                    WriteByte(4);
                    WriteDouble(number);
                    return;
                case string text:
                    WriteByte(5);
                    WriteString(text);
                    return;
                case List<object?> list:
                    EnterContainer(depth);
                    WriteByte(6);
                    WriteUInt((ulong)list.Count);
                    foreach (object? item in list)
                        WriteValue(item, depth + 1);
                    return;
                case OrderedDict<string, object?> dictionary:
                    EnterContainer(depth);
                    WriteByte(7);
                    WriteUInt((ulong)dictionary.Count);
                    foreach (var (key, item) in dictionary)
                    {
                        WriteString(key);
                        WriteValue(item, depth + 1);
                    }
                    return;
                default:
                    // Not a value a journal record can carry (an int, an enum, a decimal, another collection): v1 wrote it
                    // as JSON and read back PyJson's value, so write that value.
                    WriteValue(PyJson.Loads(PyJson.DumpsCompact(value)), depth);
                    return;
            }
        }

        /// <summary>Admits one more container level, within PyJson's depth limit and this thread's stack.</summary>
        /// <param name="depth">The containers already enclosing the one about to be written.</param>
        /// <exception cref="InvalidOperationException">The limit or the stack is reached; <see cref="Serialize" /> then falls back to v1.</exception>
        private static void EnterContainer(int depth)
        {
            if (depth >= PyJson.MaxDepth || !RuntimeHelpers.TryEnsureSufficientExecutionStack())
                throw new InvalidOperationException("The attribute nests too deeply for the v2 snapshot writer.");
        }
    }

    /// <summary>
    ///     Reads a v2 snapshot. Every read checks the bytes left, so truncated or corrupt input throws, and
    ///     <see cref="TryDeserializeV2" /> turns any throw into "no snapshot".
    /// </summary>
    private ref struct SnapshotReader
    {
        private readonly ReadOnlySpan<byte> _bytes;
        private int _position;

        // Strings that repeat across trials (parameter names, attribute keys) decoded once per restore.
        private ShortStringCache _strings;

        /// <summary>Starts reading at the first byte.</summary>
        /// <param name="bytes">The snapshot.</param>
        internal SnapshotReader(ReadOnlySpan<byte> bytes)
        {
            _bytes = bytes;
            _position = 0;
            _strings = new ShortStringCache();
        }

        /// <summary>Whether every byte has been read.</summary>
        internal readonly bool AtEnd => _position == _bytes.Length;

        /// <summary>Takes the next <paramref name="count" /> bytes.</summary>
        /// <param name="count">The bytes to take.</param>
        /// <returns>The bytes.</returns>
        /// <exception cref="InvalidOperationException">Fewer bytes are left.</exception>
        private ReadOnlySpan<byte> Take(int count)
        {
            if (count < 0 || _bytes.Length - _position < count)
                throw new InvalidOperationException("The v2 snapshot is truncated.");
            ReadOnlySpan<byte> taken = _bytes.Slice(_position, count);
            _position += count;
            return taken;
        }

        /// <summary>Skips bytes already checked.</summary>
        /// <param name="count">The bytes to skip.</param>
        /// <exception cref="InvalidOperationException">Fewer bytes are left.</exception>
        internal void Skip(int count) => Take(count);

        /// <summary>Reads one byte.</summary>
        /// <returns>The byte.</returns>
        /// <exception cref="InvalidOperationException">No byte is left.</exception>
        internal byte ReadByte() => Take(1)[0];

        /// <summary>Reads a little-endian 16-bit unsigned integer.</summary>
        /// <returns>The integer.</returns>
        /// <exception cref="InvalidOperationException">Fewer than two bytes are left.</exception>
        internal ushort ReadUInt16() => BinaryPrimitives.ReadUInt16LittleEndian(Take(2));

        /// <summary>Reads an unsigned LEB128 integer of at most ten bytes.</summary>
        /// <returns>The integer.</returns>
        /// <exception cref="InvalidOperationException">The bytes end inside it, or it runs past ten bytes.</exception>
        internal ulong ReadUInt()
        {
            ulong value = 0;
            for (int shift = 0; shift < 70; shift += 7)
            {
                byte b = ReadByte();
                value |= (ulong)(b & 0x7F) << shift;
                if (b < 0x80)
                    return value;
            }
            throw new InvalidOperationException("A v2 snapshot integer runs past ten bytes.");
        }

        /// <summary>Reads a zigzag-encoded signed integer.</summary>
        /// <returns>The integer.</returns>
        /// <exception cref="InvalidOperationException">The bytes end inside it.</exception>
        internal long ReadInt64()
        {
            ulong raw = ReadUInt();
            return (long)(raw >> 1) ^ -(long)(raw & 1);
        }

        /// <summary>Reads a zigzag-encoded signed integer that must fit 32 bits.</summary>
        /// <returns>The integer.</returns>
        /// <exception cref="InvalidOperationException">The bytes end inside it.</exception>
        /// <exception cref="OverflowException">It does not fit 32 bits.</exception>
        internal int ReadInt32() => checked((int)ReadInt64());

        /// <summary>
        ///     Reads an element count and checks that the bytes left can hold that many elements of at least
        ///     <paramref name="minBytesEach" /> bytes, before the caller allocates for them.
        /// </summary>
        /// <param name="minBytesEach">The fewest bytes one element takes.</param>
        /// <returns>The count.</returns>
        /// <exception cref="InvalidOperationException">The count cannot fit the bytes left.</exception>
        internal int ReadCount(int minBytesEach)
        {
            ulong count = ReadUInt();
            if (count > (ulong)((_bytes.Length - _position) / minBytesEach))
                throw new InvalidOperationException("A v2 snapshot count exceeds the bytes left.");
            return (int)count;
        }

        /// <summary>Reads a double; any NaN comes back as <see cref="double.NaN" />, as v1's <c>NaN</c> token does.</summary>
        /// <returns>The double.</returns>
        /// <exception cref="InvalidOperationException">Fewer than eight bytes are left.</exception>
        internal double ReadDouble()
        {
            double value = BinaryPrimitives.ReadDoubleLittleEndian(Take(8));
            return double.IsNaN(value) ? double.NaN : value;
        }

        /// <summary>
        ///     Reads a nullable datetime as v1 restores it: the UTC instant converted to naive local time
        ///     (<see cref="JournalDatetime.FromIsoToLocalNaive" /> on the same instant gives the same value).
        /// </summary>
        /// <returns>The datetime, or <c>null</c>.</returns>
        /// <exception cref="InvalidOperationException">The bytes end inside it, or its flag is neither 0 nor 1.</exception>
        /// <exception cref="ArgumentOutOfRangeException">The ticks are outside <see cref="DateTime" />'s range.</exception>
        internal DateTime? ReadDateTime()
        {
            switch (ReadByte())
            {
                case 0:
                    return null;
                case 1:
                    long ticks = BinaryPrimitives.ReadInt64LittleEndian(Take(8));
                    return new DateTimeOffset(ticks, TimeSpan.Zero).ToLocalTime().DateTime;
                default:
                    throw new InvalidOperationException("A v2 snapshot datetime has an unknown flag.");
            }
        }

        /// <summary>Reads a string: Latin-1 bytes or UTF-16 code units, as the length's flag says.</summary>
        /// <returns>The string.</returns>
        /// <exception cref="InvalidOperationException">The bytes end inside it.</exception>
        internal string ReadString()
        {
            ulong header = ReadUInt();
            if (header >> 1 > int.MaxValue)
                throw new InvalidOperationException("A v2 snapshot string is too long.");
            int length = (int)(header >> 1);
            if ((header & 1) == 0)
                return _strings.GetOrAdd(Take(length));

            ReadOnlySpan<byte> utf16 = Take(checked(length * 2));
            if (BitConverter.IsLittleEndian)
                return new string(MemoryMarshal.Cast<byte, char>(utf16));
            return string.Create(length, utf16.ToArray(), static (chars, bytes) =>
            {
                for (int i = 0; i < chars.Length; i++)
                    chars[i] = (char)BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(i * 2));
            });
        }

        /// <summary>Reads an attribute dictionary written by <see cref="SnapshotWriter.WriteDictionary" />.</summary>
        /// <param name="depth">The containers enclosing it within its attribute.</param>
        /// <returns>The dictionary, keys in their written order.</returns>
        /// <exception cref="InvalidOperationException">The bytes are malformed or nest too deeply.</exception>
        internal OrderedDict<string, object?> ReadDictionary(int depth)
        {
            int count = ReadCount(minBytesEach: 2);
            var dictionary = new OrderedDict<string, object?>(count);
            for (int i = 0; i < count; i++)
            {
                string key = ReadString();
                // The indexer, as PyJson builds dictionaries: a repeated key keeps its first place and its last value.
                dictionary[key] = ReadValue(depth);
            }
            return dictionary;
        }

        /// <summary>Reads a tagged value of PyJson's object model.</summary>
        /// <param name="depth">The containers enclosing it within its attribute.</param>
        /// <returns>The value.</returns>
        /// <exception cref="InvalidOperationException">The bytes are malformed or nest too deeply.</exception>
        private object? ReadValue(int depth)
        {
            switch (ReadByte())
            {
                case 0:
                    return null;
                case 1:
                    return BoxedFalse;
                case 2:
                    return BoxedTrue;
                case 3:
                    return ReadInt64();
                case 4:
                    return ReadDouble();
                case 5:
                    return ReadString();
                case 6:
                {
                    EnterContainer(depth);
                    int count = ReadCount(minBytesEach: 1);
                    var list = new List<object?>(count);
                    for (int i = 0; i < count; i++)
                        list.Add(ReadValue(depth + 1));
                    return list;
                }
                case 7:
                    EnterContainer(depth);
                    return ReadDictionary(depth + 1);
                default:
                    throw new InvalidOperationException("A v2 snapshot value has an unknown tag.");
            }
        }

        /// <summary>Admits one more container level, within PyJson's depth limit and this thread's stack.</summary>
        /// <param name="depth">The containers already enclosing the one about to be read.</param>
        /// <exception cref="InvalidOperationException">The limit or the stack is reached.</exception>
        private static void EnterContainer(int depth)
        {
            if (depth >= PyJson.MaxDepth || !RuntimeHelpers.TryEnsureSufficientExecutionStack())
                throw new InvalidOperationException("A v2 snapshot attribute nests too deeply.");
        }
    }

    /// <summary>The one box for <c>true</c> a restore hands out; booleans are immutable, so sharing it is invisible.</summary>
    private static readonly object BoxedTrue = true;

    /// <summary>The one box for <c>false</c> a restore hands out.</summary>
    private static readonly object BoxedFalse = false;

    /// <summary>
    ///     Reuses the string of a short Latin-1 text seen before in the same restore: a direct-mapped table from the
    ///     bytes to the string. Parameter names and attribute keys repeat in every trial.
    /// </summary>
    private struct ShortStringCache
    {
        /// <summary>The longest text cached; longer ones are decoded each time.</summary>
        private const int MaxBytes = 32;

        /// <summary>The table's size, a power of two.</summary>
        private const int Slots = 256;

        private byte[]?[]? _bytes;
        private string?[]? _strings;

        /// <summary>The string for Latin-1 bytes: from the table when the same bytes were seen last in their slot.</summary>
        /// <param name="latin1">The bytes.</param>
        /// <returns>The text: equal to <c>Encoding.Latin1.GetString(latin1)</c>.</returns>
        internal string GetOrAdd(ReadOnlySpan<byte> latin1)
        {
            if (latin1.Length > MaxBytes)
                return Encoding.Latin1.GetString(latin1);

            _bytes ??= new byte[]?[Slots];
            _strings ??= new string?[Slots];
            // FNV-1a; a collision only costs a decode, since the bytes are compared before reuse.
            uint hash = 2166136261;
            foreach (byte b in latin1)
                hash = (hash ^ b) * 16777619;
            int slot = (int)(hash & (Slots - 1));
            if (_bytes[slot] is { } known && latin1.SequenceEqual(known))
                return _strings[slot]!;
            string text = Encoding.Latin1.GetString(latin1);
            _bytes[slot] = latin1.ToArray();
            _strings[slot] = text;
            return text;
        }
    }
}
