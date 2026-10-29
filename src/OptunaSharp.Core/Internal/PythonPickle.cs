using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using NumSharp;

namespace OptunaSharp.Internal;

/// <summary>
///     A numpy <c>float64</c> scalar, as opposed to a Python <c>float</c> (a <see cref="double" /> in the value
///     model). Both hold the same IEEE double, but pickle writes them differently, and a Python object restored from
///     the pickle behaves differently on edge cases (a numpy scalar divides by zero to <c>inf</c>, a Python float
///     raises), so a state that should be byte-identical to Python's must keep the distinction per attribute.
/// </summary>
/// <remarks>
///     A class, not a struct: pickle memoizes objects by identity, and the reader returns the same instance for every
///     reference to one memoized scalar, which the writer then reproduces as a memo reference.
/// </remarks>
internal sealed class NpFloat64
{
    /// <summary>Wraps a double as a numpy float64 scalar.</summary>
    /// <param name="value">The scalar's value; any double, including NaN and the infinities.</param>
    /// <param name="dtype">
    ///     The dtype object the scalar's pickle references. numpy pickles a scalar as
    ///     <c>scalar(dtype, bytes)</c>; every scalar of one pickle normally shares one dtype object.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="dtype" /> is <c>null</c>.</exception>
    internal NpFloat64(double value, NpDtype dtype)
    {
        Value = value;
        Dtype = dtype ?? throw new ArgumentNullException(nameof(dtype));
    }

    /// <summary>The scalar's value.</summary>
    internal double Value { get; }

    /// <summary>The dtype object this scalar's pickle references (shared by identity with other values).</summary>
    internal NpDtype Dtype { get; }
}

/// <summary>
///     One numpy dtype object of a pickle, by identity: every value referencing the same instance shares one pickled
///     dtype, written once and then referenced through the memo, exactly as Python's pickler shares one dtype object.
/// </summary>
/// <remarks>
///     numpy gives a dtype object to every array, and which arrays share one depends on how each array was made
///     (an array untouched since unpickling keeps the dtype unpickling created, one rebuilt by arithmetic gets the
///     builtin <c>float64</c> dtype). A pickle is byte-identical to Python's only if the same values share an
///     instance here; the cmaes state builders encode that model per attribute.
/// </remarks>
internal sealed class NpDtype
{
    /// <summary>Creates a dtype identity with the given numpy type code.</summary>
    /// <param name="code"><c>f8</c> (float64) or <c>i8</c> (int64), the codes numpy pickles.</param>
    /// <exception cref="ArgumentException"><paramref name="code" /> is not one of the two supported codes.</exception>
    internal NpDtype(string code)
    {
        if (code is not ("f8" or "i8"))
            throw new ArgumentException($"Unsupported numpy dtype code '{code}'; only 'f8' and 'i8' are supported.", nameof(code));
        Code = code;
    }

    /// <summary>The numpy type code: <c>f8</c> or <c>i8</c>.</summary>
    internal string Code { get; }
}

/// <summary>A C-contiguous numpy array of float64 (<see cref="Doubles" />) or int64 (<see cref="Longs" />) values.</summary>
internal sealed class NpArray
{
    /// <summary>Creates a float64 array.</summary>
    /// <param name="dtype">The array's dtype object; its code must be <c>f8</c>.</param>
    /// <param name="shape">The array's shape, outermost dimension first.</param>
    /// <param name="data">The elements in C (row-major) order; its length must be the product of <paramref name="shape" />.</param>
    /// <exception cref="ArgumentException">The dtype is not float64, or the data length does not match the shape.</exception>
    /// <exception cref="OverflowException">The shape's element count does not fit 64 bits.</exception>
    internal NpArray(NpDtype dtype, int[] shape, double[] data)
    {
        if (dtype.Code != "f8")
            throw new ArgumentException("A double[] array needs an f8 dtype.", nameof(dtype));
        Dtype = dtype;
        Shape = shape;
        Doubles = data;
        CheckLength(data.Length);
    }

    /// <summary>Creates an int64 array.</summary>
    /// <param name="dtype">The array's dtype object; its code must be <c>i8</c>.</param>
    /// <param name="shape">The array's shape, outermost dimension first.</param>
    /// <param name="data">The elements in C (row-major) order; its length must be the product of <paramref name="shape" />.</param>
    /// <exception cref="ArgumentException">The dtype is not int64, or the data length does not match the shape.</exception>
    /// <exception cref="OverflowException">The shape's element count does not fit 64 bits.</exception>
    internal NpArray(NpDtype dtype, int[] shape, long[] data)
    {
        if (dtype.Code != "i8")
            throw new ArgumentException("A long[] array needs an i8 dtype.", nameof(dtype));
        Dtype = dtype;
        Shape = shape;
        Longs = data;
        CheckLength(data.Length);
    }

    /// <summary>The array's dtype object.</summary>
    internal NpDtype Dtype { get; }

    /// <summary>The array's shape, outermost dimension first.</summary>
    internal int[] Shape { get; }

    /// <summary>The float64 elements in C order, or <c>null</c> for an int64 array.</summary>
    internal double[]? Doubles { get; }

    /// <summary>The int64 elements in C order, or <c>null</c> for a float64 array.</summary>
    internal long[]? Longs { get; }

    /// <summary>Rejects data whose length disagrees with the shape, which numpy could never have pickled.</summary>
    /// <param name="length">The element count of the data.</param>
    /// <exception cref="ArgumentException">The length is not the product of the shape.</exception>
    /// <exception cref="OverflowException">The product does not fit 64 bits.</exception>
    private void CheckLength(int length)
    {
        // checked: a product beyond 64 bits cannot describe any data, and must not wrap into one that does.
        long count = 1;
        foreach (int d in Shape)
            count = checked(count * d);
        if (count != length)
            throw new ArgumentException($"An array of shape ({string.Join(", ", Shape)}) holds {count} elements, not {length}.");
    }
}

/// <summary>
///     A Python object pickled through <c>copyreg.__newobj__</c>: its class (module and qualified name) and the
///     state dictionary <c>BUILD</c> hands to <c>__setstate__</c>, in its insertion order.
/// </summary>
internal sealed class PyInstance
{
    /// <summary>Creates an instance description.</summary>
    /// <param name="module">The class's module, for example <c>cmaes._cma</c>.</param>
    /// <param name="qualName">The class's qualified name, for example <c>CMA</c>.</param>
    /// <param name="state">The state dictionary's items in insertion order; Python restores attributes in this order.</param>
    internal PyInstance(string module, string qualName, List<KeyValuePair<string, object?>> state)
    {
        Module = module;
        QualName = qualName;
        State = state;
    }

    /// <summary>The class's module.</summary>
    internal string Module { get; }

    /// <summary>The class's qualified name.</summary>
    internal string QualName { get; }

    /// <summary>The state dictionary's items, in the order Python iterates them.</summary>
    internal List<KeyValuePair<string, object?>> State { get; }

    /// <summary>Looks up a state attribute.</summary>
    /// <param name="key">The attribute name.</param>
    /// <param name="value">The attribute's value when present.</param>
    /// <returns>True when the state holds the attribute.</returns>
    internal bool TryGet(string key, out object? value)
    {
        foreach (var kv in State)
        {
            if (kv.Key == key)
            {
                value = kv.Value;
                return true;
            }
        }
        value = null;
        return false;
    }

    /// <summary>Returns a state attribute that must be present.</summary>
    /// <param name="key">The attribute name.</param>
    /// <returns>The attribute's value (possibly <c>null</c>, Python's <c>None</c>).</returns>
    /// <exception cref="ValueError">The state has no such attribute.</exception>
    internal object? Get(string key)
        => TryGet(key, out var value)
            ? value
            : throw new ValueError($"The pickled {Module}.{QualName} has no attribute '{key}'.");
}

/// <summary>
///     Python's <c>pickle</c> for the closed set of values a CMA-ES optimizer state holds: numpy float64 and int64
///     arrays and float64 scalars, Python ints, floats, bools, strings and <c>None</c>, and objects of the cmaes
///     classes. <see cref="Dumps" /> writes protocol 4 exactly as CPython 3.12's <c>pickle.dumps</c> does, so a
///     state OptunaSharp stores is byte-identical to the one Optuna stores for the same optimizer state;
///     <see cref="Loads" /> reads what Python writes (protocols 2 to 5).
/// </summary>
/// <remarks>
///     <para>
///         <b>Why.</b> Optuna's <c>CmaEsSampler</c> keeps its optimizer as <c>pickle.dumps(optimizer).hex()</c> in
///         trial system attributes. A C# worker that stores anything else there makes every Python worker on the same
///         study fail in <c>bytes.fromhex</c>, and one that cannot read pickles fails every trial after a Python worker
///         (review item P3-2). Speaking the same pickle lets both languages share one CMA-ES study.
///     </para>
///     <para>
///         <b>Safety.</b> Unlike Python's unpickler, the reader never imports or calls anything: it accepts only the
///         numpy and cmaes globals listed in <see cref="Reader" /> and builds plain C# values from them, so a hostile
///         pickle can at worst fail to load.
///     </para>
///     <para>
///         <b>Byte identity</b> follows CPython's pickler rules that matter here: every non-atomic object is memoized
///         by identity (<c>MEMOIZE</c>, later <c>BINGET</c>); strings are memoized by content, as attribute names,
///         module names and the one-character byte-order string are interned in Python, except a dtype's type code,
///         which numpy builds anew each time; the empty and one-byte bytes objects are CPython singletons (numpy's
///         <c>b'b'</c> placeholder, an empty array's data), written once and then referenced; and the output is framed
///         (<c>FRAME</c>) in chunks of at least 64 KiB, committed when a save starts, with a bytes or string payload of
///         64 KiB or more written outside any frame.
///     </para>
/// </remarks>
internal static class PythonPickle
{
    // Protocol-2..5 opcodes (pickletools' names).
    private const byte Mark = 0x28, Stop = 0x2e, Pop = 0x30, PopMark = 0x31, Dup = 0x32, BinInt = 0x4a, BinInt1 = 0x4b,
        BinInt2 = 0x4d, None = 0x4e, Reduce = 0x52, BinUnicode = 0x58, Append = 0x61, Build = 0x62, Global = 0x63,
        Dict = 0x64, EmptyDict = 0x7d, Appends = 0x65, BinGet = 0x68, LongBinGet = 0x6a, List = 0x6c, EmptyList = 0x5d,
        BinPut = 0x71, LongBinPut = 0x72, SetItem = 0x73, Tuple = 0x74, EmptyTuple = 0x29, SetItems = 0x75,
        BinFloat = 0x47, BinBytes = 0x42, ShortBinBytes = 0x43, Proto = 0x80, NewObj = 0x81, Tuple1 = 0x85,
        Tuple2 = 0x86, Tuple3 = 0x87, NewTrue = 0x88, NewFalse = 0x89, Long1 = 0x8a, Long4 = 0x8b,
        ShortBinUnicode = 0x8c, BinUnicode8 = 0x8d, BinBytes8 = 0x8e, StackGlobal = 0x93, Memoize = 0x94, Frame = 0x95,
        ByteArray8 = 0x96;

    // numpy 2 pickles its scalar and array helpers from numpy._core; numpy 1.x wrote numpy.core, which numpy 2 reads.
    private const string NumpyMultiarray = "numpy._core.multiarray";

    /// <summary>
    ///     Pickles a value with protocol 4, byte for byte as CPython 3.12's <c>pickle.dumps(value)</c> does for the
    ///     equivalent Python object graph.
    /// </summary>
    /// <param name="value">
    ///     The root value: <c>null</c>, <see cref="bool" />, <see cref="int" />, <see cref="long" />,
    ///     <see cref="double" />, <see cref="string" />, <see cref="NpFloat64" />, <see cref="NpArray" />,
    ///     <see cref="NpDtype" /> or <see cref="PyInstance" />, nested through instance states.
    /// </param>
    /// <returns>The pickle's bytes; <c>bytes.fromhex</c> of their lowercase hex is what Python's unpickler reads.</returns>
    /// <exception cref="ArgumentException">The value graph holds a type outside the supported set.</exception>
    internal static byte[] Dumps(object? value) => new Writer().Dump(value);

    /// <summary>
    ///     Unpickles bytes written by Python (protocols 2 to 5) or by <see cref="Dumps" /> into the value model.
    /// </summary>
    /// <param name="data">The pickle's bytes.</param>
    /// <returns>The root value: a <see cref="PyInstance" /> for a cmaes optimizer.</returns>
    /// <exception cref="ValueError">
    ///     The data is truncated or malformed, uses an opcode or global outside the supported set (only numpy arrays,
    ///     numpy scalars and the cmaes optimizer classes are accepted), or holds a numpy value this model cannot
    ///     represent (an unsupported dtype, byte order or memory layout).
    /// </exception>
    internal static object? Loads(byte[] data) => new Reader(data).Load();

    /// <summary>The protocol-4 pickler: CPython's <c>save</c> rules for the supported values.</summary>
    private sealed class Writer
    {
        // CPython's _FRAME_SIZE_TARGET and _FRAME_SIZE_MIN.
        private const int FrameSizeTarget = 64 * 1024;
        private const int FrameSizeMin = 4;

        private readonly MemoryStream _out = new();
        private readonly MemoryStream _frame = new();

        // Memo keys: values by reference identity; strings by content (interned in Python); globals by
        // "module\nname"; bytes of at most one byte by content (CPython's singletons; slot 256 is b'').
        private readonly Dictionary<object, int> _memoByRef = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<string, int> _memoStrings = new(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _memoGlobals = new(StringComparer.Ordinal);
        private readonly int[] _memoSmallBytes = CreateSmallBytesMemo();
        private int _memoCount;

        /// <summary>Creates the memo slots of the 257 bytes singletons, all unwritten (-1).</summary>
        /// <returns>The slot array: index = the byte's value, 256 = the empty bytes object.</returns>
        private static int[] CreateSmallBytesMemo()
        {
            var slots = new int[257];
            Array.Fill(slots, -1);
            return slots;
        }

        /// <summary>Pickles the root value: <c>PROTO 4</c>, the framed body, <c>STOP</c> inside the last frame.</summary>
        /// <param name="value">The root value.</param>
        /// <returns>The complete pickle.</returns>
        /// <exception cref="ArgumentException">The value graph holds an unsupported type.</exception>
        internal byte[] Dump(object? value)
        {
            // PROTO precedes framing, as in CPython's dump(): the first frame starts after it.
            _out.WriteByte(Proto);
            _out.WriteByte(4);
            Save(value);
            Write(Stop);
            CommitFrame(force: true);
            return _out.ToArray();
        }

        /// <summary>
        ///     Writes the current frame out once it reaches the 64 KiB target (or always, when forced), as CPython's
        ///     <c>_Pickler_OpcodeBoundary</c>/<c>_Pickler_CommitFrame</c> do. A frame under 4 bytes gets no header.
        /// </summary>
        /// <param name="force">Commit whatever the frame holds, regardless of its size.</param>
        private void CommitFrame(bool force)
        {
            long length = _frame.Length;
            if (length == 0 || (length < FrameSizeTarget && !force))
                return;
            if (length >= FrameSizeMin)
            {
                Span<byte> header = stackalloc byte[9];
                header[0] = Frame;
                BinaryPrimitives.WriteUInt64LittleEndian(header[1..], (ulong)length);
                _out.Write(header);
            }
            _frame.WriteTo(_out);
            _frame.SetLength(0);
        }

        /// <summary>Appends one opcode byte to the current frame.</summary>
        /// <param name="b">The byte.</param>
        private void Write(byte b) => _frame.WriteByte(b);

        /// <summary>Appends bytes to the current frame.</summary>
        /// <param name="bytes">The bytes.</param>
        private void Write(ReadOnlySpan<byte> bytes) => _frame.Write(bytes);

        /// <summary>
        ///     Writes a header and its payload. A payload of 64 KiB or more bypasses framing: the current frame is
        ///     committed and both go straight to the output, as CPython's <c>_Pickler_write_bytes</c> does.
        /// </summary>
        /// <param name="header">The opcode and its length field.</param>
        /// <param name="payload">The bytes or UTF-8 text.</param>
        private void WritePayload(ReadOnlySpan<byte> header, ReadOnlySpan<byte> payload)
        {
            if (payload.Length >= FrameSizeTarget)
            {
                CommitFrame(force: true);
                _out.Write(header);
                _out.Write(payload);
                return;
            }
            Write(header);
            Write(payload);
        }

        /// <summary>The start of every CPython <c>save()</c>: commit the frame if it has reached the target.</summary>
        private void Boundary() => CommitFrame(force: false);

        /// <summary>Records the object just written under the next memo index and emits <c>MEMOIZE</c>.</summary>
        /// <returns>The memo index assigned.</returns>
        private int MemoPut()
        {
            Write(Memoize);
            return _memoCount++;
        }

        /// <summary>Emits a memo reference: <c>BINGET</c> below 256, <c>LONG_BINGET</c> above.</summary>
        /// <param name="index">The memo index.</param>
        private void Get(int index)
        {
            if (index < 256)
            {
                Write(BinGet);
                Write((byte)index);
                return;
            }
            Span<byte> buf = stackalloc byte[5];
            buf[0] = LongBinGet;
            BinaryPrimitives.WriteInt32LittleEndian(buf[1..], index);
            Write(buf);
        }

        /// <summary>Saves one value of the model, dispatching on its type.</summary>
        /// <param name="value">The value.</param>
        /// <exception cref="ArgumentException">The value's type is not supported.</exception>
        private void Save(object? value)
        {
            switch (value)
            {
                case null:
                    Boundary();
                    Write(None);
                    return;
                case bool b:
                    SaveBool(b);
                    return;
                case int i:
                    SaveLong(i);
                    return;
                case long l:
                    SaveLong(l);
                    return;
                case double d:
                    SaveFloat(d);
                    return;
                case string s:
                    SaveString(s);
                    return;
                case NpFloat64 scalar:
                    SaveScalar(scalar);
                    return;
                case NpArray array:
                    SaveArray(array);
                    return;
                case NpDtype dtype:
                    SaveDtype(dtype);
                    return;
                case PyInstance instance:
                    SaveInstance(instance);
                    return;
                default:
                    throw new ArgumentException($"Cannot pickle a value of type {value.GetType().Name}.", nameof(value));
            }
        }

        /// <summary>Saves a Python bool (<c>NEWTRUE</c>/<c>NEWFALSE</c>; atoms are not memoized).</summary>
        /// <param name="b">The value.</param>
        private void SaveBool(bool b)
        {
            Boundary();
            Write(b ? NewTrue : NewFalse);
        }

        /// <summary>
        ///     Saves a Python int the way CPython's <c>save_long</c> does: <c>BININT1</c>, <c>BININT2</c> or
        ///     <c>BININT</c> when it fits a signed 32-bit integer (chosen from its four little-endian bytes), else
        ///     <c>LONG1</c> with the minimal two's-complement encoding.
        /// </summary>
        /// <param name="v">The value.</param>
        private void SaveLong(long v)
        {
            Boundary();
            if (v >= int.MinValue && v <= int.MaxValue)
            {
                uint u = unchecked((uint)(int)v);
                if ((u & 0xFFFF0000) != 0)
                {
                    Span<byte> buf = stackalloc byte[5];
                    buf[0] = BinInt;
                    BinaryPrimitives.WriteUInt32LittleEndian(buf[1..], u);
                    Write(buf);
                }
                else if ((u & 0xFF00) != 0)
                {
                    Span<byte> buf = stackalloc byte[3];
                    buf[0] = BinInt2;
                    BinaryPrimitives.WriteUInt16LittleEndian(buf[1..], (ushort)u);
                    Write(buf);
                }
                else
                {
                    Write(BinInt1);
                    Write((byte)u);
                }
                return;
            }
            // encode_long: the shortest little-endian two's complement that round-trips the value.
            byte[] bytes = new BigInteger(v).ToByteArray();
            Write(Long1);
            Write((byte)bytes.Length);
            Write(bytes);
        }

        /// <summary>Saves a Python float as <c>BINFLOAT</c> (big-endian IEEE double; floats are not memoized).</summary>
        /// <param name="d">The value; NaN and the infinities keep their exact bits.</param>
        private void SaveFloat(double d)
        {
            Boundary();
            Span<byte> buf = stackalloc byte[9];
            buf[0] = BinFloat;
            BinaryPrimitives.WriteDoubleBigEndian(buf[1..], d);
            Write(buf);
        }

        /// <summary>Saves an interned string: a memo reference if the same text was saved before, else the text.</summary>
        /// <param name="s">The string.</param>
        private void SaveString(string s)
        {
            Boundary();
            if (_memoStrings.TryGetValue(s, out int index))
            {
                Get(index);
                return;
            }
            WriteUnicode(s);
            _memoStrings[s] = MemoPut();
        }

        /// <summary>
        ///     Saves a string Python builds anew each time (a dtype's type code), so it is memoized but never
        ///     referenced again.
        /// </summary>
        /// <param name="s">The string.</param>
        private void SaveFreshString(string s)
        {
            Boundary();
            WriteUnicode(s);
            MemoPut();
        }

        /// <summary>Writes UTF-8 text: <c>SHORT_BINUNICODE</c> up to 255 bytes, <c>BINUNICODE</c> (or <c>BINUNICODE8</c>) above.</summary>
        /// <param name="s">The string.</param>
        private void WriteUnicode(string s)
        {
            byte[] utf8 = Encoding.UTF8.GetBytes(s);
            if (utf8.Length <= 0xFF)
            {
                WritePayload(stackalloc byte[] { ShortBinUnicode, (byte)utf8.Length }, utf8);
                return;
            }
            Span<byte> header = stackalloc byte[5];
            header[0] = BinUnicode;
            BinaryPrimitives.WriteUInt32LittleEndian(header[1..], (uint)utf8.Length);
            WritePayload(header, utf8);
        }

        /// <summary>
        ///     Saves a bytes object: <c>SHORT_BINBYTES</c> up to 255 bytes, <c>BINBYTES</c> above, memoized.
        /// </summary>
        /// <param name="data">The bytes.</param>
        /// <remarks>
        ///     Bytes are new objects (an array's or scalar's data), except the empty bytes and the one-byte bytes,
        ///     which CPython keeps as singletons: numpy's <c>b'b'</c> placeholder in every array's reduce, and the
        ///     data of every empty array (CMAwM's discrete-index arrays on a continuous space). Those are written
        ///     once per pickle and then referenced from the memo, as CPython's memo-by-identity does.
        /// </remarks>
        private void SaveBytes(ReadOnlySpan<byte> data)
        {
            Boundary();
            int singleton = data.Length switch { 0 => 256, 1 => data[0], _ => -1 };
            if (singleton >= 0 && _memoSmallBytes[singleton] >= 0)
            {
                Get(_memoSmallBytes[singleton]);
                return;
            }
            if (data.Length <= 0xFF)
            {
                WritePayload(stackalloc byte[] { ShortBinBytes, (byte)data.Length }, data);
            }
            else
            {
                Span<byte> header = stackalloc byte[5];
                header[0] = BinBytes;
                BinaryPrimitives.WriteUInt32LittleEndian(header[1..], (uint)data.Length);
                WritePayload(header, data);
            }
            int index = MemoPut();
            if (singleton >= 0)
                _memoSmallBytes[singleton] = index;
        }

        /// <summary>
        ///     Saves a module-level class or function the way protocol 4's <c>save_global</c> does: module and name as
        ///     strings, then <c>STACK_GLOBAL</c>, memoized; later references reuse the memo.
        /// </summary>
        /// <param name="module">The module name.</param>
        /// <param name="name">The qualified name.</param>
        private void SaveGlobal(string module, string name)
        {
            Boundary();
            string key = module + "\n" + name;
            if (_memoGlobals.TryGetValue(key, out int index))
            {
                Get(index);
                return;
            }
            SaveString(module);
            SaveString(name);
            Write(StackGlobal);
            _memoGlobals[key] = MemoPut();
        }

        /// <summary>
        ///     Saves a fresh tuple of the given elements, as CPython's <c>save_tuple</c> does: <c>EMPTY_TUPLE</c> (not
        ///     memoized) when empty, <c>TUPLE1</c>-<c>TUPLE3</c> up to three elements, <c>MARK … TUPLE</c> above; the
        ///     tuple is memoized after its elements.
        /// </summary>
        /// <param name="elements">One action per element, each saving that element.</param>
        private void SaveTuple(params Action[] elements)
        {
            Boundary();
            if (elements.Length == 0)
            {
                Write(EmptyTuple);
                return;
            }
            if (elements.Length <= 3)
            {
                foreach (var save in elements)
                    save();
                Write(elements.Length switch { 1 => Tuple1, 2 => Tuple2, _ => Tuple3 });
            }
            else
            {
                Write(Mark);
                foreach (var save in elements)
                    save();
                Write(Tuple);
            }
            MemoPut();
        }

        /// <summary>Saves an array shape as a tuple of Python ints.</summary>
        /// <param name="shape">The shape.</param>
        private void SaveShape(int[] shape)
        {
            var elements = new Action[shape.Length];
            for (int i = 0; i < shape.Length; i++)
            {
                int dim = shape[i];
                elements[i] = () => SaveLong(dim);
            }
            SaveTuple(elements);
        }

        /// <summary>
        ///     Saves a numpy dtype as numpy's <c>__reduce__</c> describes it:
        ///     <c>dtype(code, False, True)</c>, then <c>BUILD</c> with <c>(3, '&lt;', None, None, None, -1, -1, 0)</c>.
        ///     Memoized by identity, so arrays sharing the object reference it.
        /// </summary>
        /// <param name="dtype">The dtype object.</param>
        private void SaveDtype(NpDtype dtype)
        {
            Boundary();
            if (_memoByRef.TryGetValue(dtype, out int index))
            {
                Get(index);
                return;
            }
            SaveGlobal("numpy", "dtype");
            SaveTuple(() => SaveFreshString(dtype.Code), () => SaveBool(false), () => SaveBool(true));
            Write(Reduce);
            _memoByRef[dtype] = MemoPut();
            SaveTuple(
                () => SaveLong(3), () => SaveString("<"), () => Save(null), () => Save(null), () => Save(null),
                () => SaveLong(-1), () => SaveLong(-1), () => SaveLong(0));
            Write(Build);
        }

        /// <summary>
        ///     Saves a numpy float64 scalar as numpy's <c>__reduce__</c> does: <c>scalar(dtype, raw little-endian bytes)</c>.
        /// </summary>
        /// <param name="scalar">The scalar.</param>
        private void SaveScalar(NpFloat64 scalar)
        {
            Boundary();
            if (_memoByRef.TryGetValue(scalar, out int index))
            {
                Get(index);
                return;
            }
            SaveGlobal(NumpyMultiarray, "scalar");
            var raw = new byte[8];
            BinaryPrimitives.WriteDoubleLittleEndian(raw, scalar.Value);
            SaveTuple(() => SaveDtype(scalar.Dtype), () => SaveBytes(raw));
            Write(Reduce);
            _memoByRef[scalar] = MemoPut();
        }

        /// <summary>
        ///     Saves a numpy array as numpy's protocol-4 <c>__reduce__</c> does:
        ///     <c>_reconstruct(ndarray, (0,), b'b')</c>, then <c>BUILD</c> with
        ///     <c>(1, shape, dtype, False, raw C-order bytes)</c>.
        /// </summary>
        /// <param name="array">The array.</param>
        private void SaveArray(NpArray array)
        {
            Boundary();
            if (_memoByRef.TryGetValue(array, out int index))
            {
                Get(index);
                return;
            }
            SaveGlobal(NumpyMultiarray, "_reconstruct");
            SaveTuple(() => SaveGlobal("numpy", "ndarray"), () => SaveTuple(() => SaveLong(0)), () => SaveBytes("b"u8));
            Write(Reduce);
            _memoByRef[array] = MemoPut();
            SaveTuple(
                () => SaveLong(1), () => SaveShape(array.Shape), () => SaveDtype(array.Dtype), () => SaveBool(false),
                () => SaveBytes(RawBytes(array)));
            Write(Build);
        }

        /// <summary>
        ///     Saves an object through <c>copyreg.__newobj__</c>: its class, an empty argument tuple and
        ///     <c>NEWOBJ</c>, memoized, then its state dictionary and <c>BUILD</c>.
        /// </summary>
        /// <param name="instance">The object.</param>
        private void SaveInstance(PyInstance instance)
        {
            Boundary();
            if (_memoByRef.TryGetValue(instance, out int index))
            {
                Get(index);
                return;
            }
            SaveGlobal(instance.Module, instance.QualName);
            SaveTuple();
            Write(NewObj);
            _memoByRef[instance] = MemoPut();
            SaveDict(instance.State);
            Write(Build);
        }

        /// <summary>
        ///     Saves a dict as CPython's <c>save_dict</c>/<c>batch_dict_exact</c> do: <c>EMPTY_DICT</c>, memoized, then
        ///     one <c>SETITEM</c> for a single item or <c>MARK … SETITEMS</c> batches of 1000 items.
        /// </summary>
        /// <param name="items">The items in insertion order.</param>
        private void SaveDict(List<KeyValuePair<string, object?>> items)
        {
            Boundary();
            Write(EmptyDict);
            MemoPut();
            if (items.Count == 0)
                return;
            if (items.Count == 1)
            {
                SaveString(items[0].Key);
                Save(items[0].Value);
                Write(SetItem);
                return;
            }
            // CPython's do-while: a batch that fills all 1000 slots is followed by another MARK … SETITEMS, which
            // is empty when the dict size is a multiple of 1000; mirrored so even that case stays byte-identical.
            int start = 0, batch;
            do
            {
                Write(Mark);
                batch = Math.Min(1000, items.Count - start);
                for (int i = start; i < start + batch; i++)
                {
                    SaveString(items[i].Key);
                    Save(items[i].Value);
                }
                start += batch;
                Write(SetItems);
            } while (batch == 1000);
        }

        /// <summary>The array's elements as raw little-endian bytes, the payload numpy's <c>tobytes()</c> gives.</summary>
        /// <param name="array">The array.</param>
        /// <returns>The raw bytes.</returns>
        private static byte[] RawBytes(NpArray array)
        {
            if (array.Doubles is { } doubles)
            {
                var raw = new byte[doubles.Length * 8];
                for (int i = 0; i < doubles.Length; i++)
                    BinaryPrimitives.WriteDoubleLittleEndian(raw.AsSpan(i * 8), doubles[i]);
                return raw;
            }
            long[] longs = array.Longs!;
            var rawLongs = new byte[longs.Length * 8];
            for (int i = 0; i < longs.Length; i++)
                BinaryPrimitives.WriteInt64LittleEndian(rawLongs.AsSpan(i * 8), longs[i]);
            return rawLongs;
        }
    }

    /// <summary>
    ///     A protocol-2-to-5 unpickler for the supported values. It runs Python's pickle machine (stack, mark stack,
    ///     memo) but resolves globals against a fixed allow-list instead of importing them.
    /// </summary>
    /// <remarks>
    ///     Accepted globals: numpy's <c>scalar</c>, <c>_reconstruct</c> and <c>_frombuffer</c> (from
    ///     <c>numpy._core</c> or numpy 1.x's <c>numpy.core</c>), <c>numpy.ndarray</c> and <c>numpy.dtype</c>, and the
    ///     cmaes classes <c>CMA</c>, <c>SepCMA</c> and <c>CMAwM</c>.
    /// </remarks>
    private sealed class Reader
    {
        private readonly byte[] _data;
        private int _pos;
        private readonly List<object?> _stack = new();
        private readonly Stack<List<object?>> _marks = new();
        private readonly Dictionary<int, object?> _memo = new();

        /// <summary>Creates a reader over pickle bytes.</summary>
        /// <param name="data">The pickle's bytes.</param>
        internal Reader(byte[] data) => _data = data;

        /// <summary>A global the pickle referenced: a class or function, kept by name.</summary>
        /// <param name="Module">Its module.</param>
        /// <param name="Name">Its qualified name.</param>
        private sealed record GlobalRef(string Module, string Name);

        /// <summary>A Python dict under construction: insertion-ordered, last write of a key wins in place.</summary>
        private sealed class PyDict
        {
            /// <summary>The items in insertion order.</summary>
            internal readonly List<KeyValuePair<object?, object?>> Items = new();

            /// <summary>Sets an item, keeping the original position of an existing key, as a Python dict does.</summary>
            /// <param name="key">The key.</param>
            /// <param name="value">The value.</param>
            internal void Set(object? key, object? value)
            {
                for (int i = 0; i < Items.Count; i++)
                {
                    if (Equals(Items[i].Key, key))
                    {
                        Items[i] = new(key, value);
                        return;
                    }
                }
                Items.Add(new(key, value));
            }
        }

        /// <summary>An array <c>_reconstruct</c> created; <c>BUILD</c> fills it into an <see cref="NpArray" />.</summary>
        private sealed class ArrayShell
        {
            /// <summary>The array once <c>BUILD</c> has filled it.</summary>
            internal NpArray? Built;
        }

        /// <summary>Runs the pickle machine to <c>STOP</c>.</summary>
        /// <returns>The root value.</returns>
        /// <exception cref="ValueError">The pickle is malformed or uses unsupported opcodes or globals.</exception>
        internal object? Load()
        {
            while (true)
            {
                byte op = ReadByte();
                switch (op)
                {
                    case Proto:
                        if (ReadByte() > 5)
                            throw Error("pickle protocol above 5");
                        break;
                    case Frame:
                        // Frames only bound reads; the opcodes inside continue the same stream.
                        ReadBytes(8);
                        break;
                    case Stop:
                        return Finish(Pop1());
                    case Mark:
                        _marks.Push(new List<object?>(_stack));
                        _stack.Clear();
                        break;
                    case Pop:
                        Pop1();
                        break;
                    case PopMark:
                        PopMarkItems();
                        break;
                    case Dup:
                        _stack.Add(Peek());
                        break;
                    case None:
                        _stack.Add(null);
                        break;
                    case NewTrue:
                        _stack.Add(true);
                        break;
                    case NewFalse:
                        _stack.Add(false);
                        break;
                    case BinInt:
                        _stack.Add((long)BinaryPrimitives.ReadInt32LittleEndian(ReadBytes(4)));
                        break;
                    case BinInt1:
                        _stack.Add((long)ReadByte());
                        break;
                    case BinInt2:
                        _stack.Add((long)BinaryPrimitives.ReadUInt16LittleEndian(ReadBytes(2)));
                        break;
                    case Long1:
                        _stack.Add(DecodeLong(ReadBytes(ReadByte())));
                        break;
                    case Long4:
                        _stack.Add(DecodeLong(ReadBytes(checked((int)BinaryPrimitives.ReadUInt32LittleEndian(ReadBytes(4))))));
                        break;
                    case BinFloat:
                        _stack.Add(BinaryPrimitives.ReadDoubleBigEndian(ReadBytes(8)));
                        break;
                    case ShortBinUnicode:
                        _stack.Add(Encoding.UTF8.GetString(ReadBytes(ReadByte())));
                        break;
                    case BinUnicode:
                        _stack.Add(Encoding.UTF8.GetString(ReadBytes(checked((int)BinaryPrimitives.ReadUInt32LittleEndian(ReadBytes(4))))));
                        break;
                    case BinUnicode8:
                        _stack.Add(Encoding.UTF8.GetString(ReadBytes(checked((int)BinaryPrimitives.ReadUInt64LittleEndian(ReadBytes(8))))));
                        break;
                    case ShortBinBytes:
                        _stack.Add(ReadBytes(ReadByte()).ToArray());
                        break;
                    case BinBytes:
                        _stack.Add(ReadBytes(checked((int)BinaryPrimitives.ReadUInt32LittleEndian(ReadBytes(4)))).ToArray());
                        break;
                    case BinBytes8:
                    case ByteArray8:
                        // Protocol 5 writes numpy's in-band buffers as BYTEARRAY8; both are plain bytes here.
                        _stack.Add(ReadBytes(checked((int)BinaryPrimitives.ReadUInt64LittleEndian(ReadBytes(8)))).ToArray());
                        break;
                    case EmptyTuple:
                        _stack.Add(Array.Empty<object?>());
                        break;
                    case Tuple1:
                        _stack.Add(new[] { Pop1() });
                        break;
                    case Tuple2:
                    {
                        object? b = Pop1(), a = Pop1();
                        _stack.Add(new[] { a, b });
                        break;
                    }
                    case Tuple3:
                    {
                        object? c = Pop1(), b = Pop1(), a = Pop1();
                        _stack.Add(new[] { a, b, c });
                        break;
                    }
                    case Tuple:
                        _stack.Add(PopMarkItems().ToArray());
                        break;
                    case EmptyList:
                        _stack.Add(new List<object?>());
                        break;
                    case List:
                        _stack.Add(PopMarkItems());
                        break;
                    case Append:
                    {
                        object? item = Pop1();
                        AsList(Peek()).Add(item);
                        break;
                    }
                    case Appends:
                    {
                        var items = PopMarkItems();
                        AsList(Peek()).AddRange(items);
                        break;
                    }
                    case EmptyDict:
                        _stack.Add(new PyDict());
                        break;
                    case Dict:
                    {
                        var items = PopMarkItems();
                        var dict = new PyDict();
                        SetPairs(dict, items);
                        _stack.Add(dict);
                        break;
                    }
                    case SetItem:
                    {
                        object? value = Pop1(), key = Pop1();
                        AsDict(Peek()).Set(key, value);
                        break;
                    }
                    case SetItems:
                    {
                        var items = PopMarkItems();
                        SetPairs(AsDict(Peek()), items);
                        break;
                    }
                    case Memoize:
                        _memo[_memo.Count] = Peek();
                        break;
                    case BinPut:
                        _memo[ReadByte()] = Peek();
                        break;
                    case LongBinPut:
                        _memo[checked((int)BinaryPrimitives.ReadUInt32LittleEndian(ReadBytes(4)))] = Peek();
                        break;
                    case BinGet:
                        _stack.Add(MemoAt(ReadByte()));
                        break;
                    case LongBinGet:
                        _stack.Add(MemoAt(checked((int)BinaryPrimitives.ReadUInt32LittleEndian(ReadBytes(4)))));
                        break;
                    case Global:
                    {
                        string module = ReadLine(), name = ReadLine();
                        _stack.Add(ResolveGlobal(module, name));
                        break;
                    }
                    case StackGlobal:
                    {
                        object? name = Pop1(), module = Pop1();
                        if (module is not string m || name is not string n)
                            throw Error("STACK_GLOBAL needs two strings");
                        _stack.Add(ResolveGlobal(m, n));
                        break;
                    }
                    case Reduce:
                    {
                        object? args = Pop1(), callable = Pop1();
                        _stack.Add(Call(callable, AsTuple(args)));
                        break;
                    }
                    case NewObj:
                    {
                        object? args = Pop1(), cls = Pop1();
                        _stack.Add(NewObject(cls, AsTuple(args)));
                        break;
                    }
                    case Build:
                    {
                        object? state = Pop1();
                        ApplyBuild(Peek(), state);
                        break;
                    }
                    default:
                        throw Error($"unsupported opcode 0x{op:x2}");
                }
            }
        }

        /// <summary>Allows only the numpy and cmaes globals a CMA-ES state uses; the reader never imports anything.</summary>
        /// <param name="module">The global's module.</param>
        /// <param name="name">The global's name.</param>
        /// <returns>A reference by name.</returns>
        /// <exception cref="ValueError">The global is not on the allow-list.</exception>
        private GlobalRef ResolveGlobal(string module, string name)
        {
            bool allowed = (module, name) switch
            {
                ("numpy._core.multiarray" or "numpy.core.multiarray", "scalar" or "_reconstruct") => true,
                ("numpy._core.numeric" or "numpy.core.numeric", "_frombuffer") => true,
                ("numpy", "ndarray" or "dtype") => true,
                ("cmaes._cma", "CMA") or ("cmaes._sepcma", "SepCMA") or ("cmaes._cmawm", "CMAwM") => true,
                _ => false,
            };
            if (!allowed)
                throw Error($"global {module}.{name} is not allowed in a CMA-ES optimizer state");
            return new GlobalRef(module, name);
        }

        /// <summary>Performs <c>REDUCE</c> for the allowed numpy callables.</summary>
        /// <param name="callable">The callable from the stack.</param>
        /// <param name="args">Its arguments.</param>
        /// <returns>The value the call stands for.</returns>
        /// <exception cref="ValueError">The callable is not one this reader can apply, or its arguments are wrong.</exception>
        private object Call(object? callable, object?[] args)
        {
            if (callable is not GlobalRef g)
                throw Error("REDUCE on a non-global callable");
            switch (g.Name)
            {
                case "scalar" when args.Length == 2 && args[0] is NpDtype dtype && args[1] is byte[] raw:
                    // A float64 scalar keeps its dtype object's identity; an int64 scalar has no numpy form in this
                    // model, so it becomes a Python int (cmaes states hold none).
                    if (raw.Length != 8)
                        throw Error("a numpy scalar's data must be 8 bytes");
                    return dtype.Code == "f8"
                        ? new NpFloat64(BinaryPrimitives.ReadDoubleLittleEndian(raw), dtype)
                        : BinaryPrimitives.ReadInt64LittleEndian(raw);
                case "_reconstruct" when args.Length == 3:
                    return new ArrayShell();
                case "dtype" when args.Length >= 1 && args[0] is string code:
                    return code switch
                    {
                        "f8" or "float64" => new NpDtype("f8"),
                        "i8" or "int64" => new NpDtype("i8"),
                        _ => throw Error($"unsupported numpy dtype '{code}'"),
                    };
                case "_frombuffer" when args.Length == 4 && args[0] is byte[] buffer && args[1] is NpDtype dtype && args[2] is object?[] shape:
                    if (args[3] is string order && order != "C")
                        throw Error("only C-ordered numpy arrays are supported");
                    return MakeArray(dtype, ToShape(shape), buffer);
                default:
                    throw Error($"cannot apply {g.Module}.{g.Name} to {args.Length} arguments");
            }
        }

        /// <summary>Performs <c>NEWOBJ</c> for the allowed cmaes classes.</summary>
        /// <param name="cls">The class from the stack.</param>
        /// <param name="args">Its <c>__new__</c> arguments, which must be empty.</param>
        /// <returns>An instance whose state <c>BUILD</c> fills.</returns>
        /// <exception cref="ValueError">The class is not a cmaes optimizer class, or arguments were passed.</exception>
        private object NewObject(object? cls, object?[] args)
        {
            if (cls is not GlobalRef { Module: var module, Name: var name } || !module.StartsWith("cmaes.", StringComparison.Ordinal))
                throw Error("NEWOBJ of a class that is not a cmaes optimizer");
            if (args.Length != 0)
                throw Error("NEWOBJ with arguments");
            return new PyInstance(module, name, new List<KeyValuePair<string, object?>>());
        }

        /// <summary>
        ///     Performs <c>BUILD</c>: fills an array from numpy's state tuple, checks a dtype's state, or sets an
        ///     instance's attributes from its state dict.
        /// </summary>
        /// <param name="target">The object on top of the stack.</param>
        /// <param name="state">The state just popped.</param>
        /// <exception cref="ValueError">The state does not fit the object.</exception>
        private void ApplyBuild(object? target, object? state)
        {
            switch (target)
            {
                case ArrayShell shell:
                {
                    // numpy's ndarray.__setstate__((version, shape, dtype, is_fortran, raw)).
                    var t = AsTuple(state);
                    if (t.Length != 5 || t[2] is not NpDtype dtype || t[3] is not bool fortran || t[4] is not byte[] raw || t[1] is not object?[] shape)
                        throw Error("an array state must be (1, shape, dtype, is_fortran, bytes)");
                    if (fortran)
                        throw Error("only C-ordered numpy arrays are supported");
                    shell.Built = MakeArray(dtype, ToShape(shape), raw);
                    return;
                }
                case NpDtype:
                {
                    // (version, byteorder, subarray, names, fields, elsize, alignment, flags): only plain
                    // little-endian (or byte-order-free) scalars are representable.
                    var t = AsTuple(state);
                    if (t.Length < 2 || t[1] is not ("<" or "|" or "="))
                        throw Error("only little-endian numpy dtypes are supported");
                    if (t.Length > 4 && (t[2] is not null || t[3] is not null || t[4] is not null))
                        throw Error("structured numpy dtypes are not supported");
                    return;
                }
                case PyInstance instance:
                {
                    if (state is not PyDict dict)
                        throw Error("a cmaes object's state must be a dict");
                    foreach (var kv in dict.Items)
                    {
                        if (kv.Key is not string key)
                            throw Error("a cmaes object's attribute names must be strings");
                        instance.State.Add(new(key, Finish(kv.Value)));
                    }
                    return;
                }
                default:
                    throw Error("BUILD on an unsupported object");
            }
        }

        /// <summary>Replaces array shells by the arrays <c>BUILD</c> made; other values pass through.</summary>
        /// <param name="value">A value from the machine.</param>
        /// <returns>The value as the model represents it.</returns>
        /// <exception cref="ValueError">An array was never built.</exception>
        private object? Finish(object? value)
            => value is ArrayShell shell ? shell.Built ?? throw Error("an array was never built") : value;

        /// <summary>Builds an array from raw little-endian bytes.</summary>
        /// <param name="dtype">The array's dtype object.</param>
        /// <param name="shape">The array's shape.</param>
        /// <param name="raw">The raw data.</param>
        /// <returns>The array.</returns>
        /// <exception cref="ValueError">The data length does not match the shape.</exception>
        private NpArray MakeArray(NpDtype dtype, int[] shape, byte[] raw)
        {
            // The element count must equal the data's 8-byte words. The product is bounded by that count as it is
            // formed, so a hostile shape (say three dimensions of 2^31 - 1) fails here instead of overflowing into
            // a size that matches by accident or into a huge allocation. A zero dimension makes any other legal.
            long count = 1, words = raw.Length / 8;
            if (Array.IndexOf(shape, 0) >= 0)
                count = 0;
            else
                foreach (int d in shape)
                    if ((count *= d) > words)
                        throw Error("an array's data does not match its shape");
            if (raw.Length != count * 8)
                throw Error("an array's data does not match its shape");
            if (dtype.Code == "f8")
            {
                var values = new double[count];
                for (int i = 0; i < values.Length; i++)
                    values[i] = BinaryPrimitives.ReadDoubleLittleEndian(raw.AsSpan(i * 8));
                return new NpArray(dtype, shape, values);
            }
            var longs = new long[count];
            for (int i = 0; i < longs.Length; i++)
                longs[i] = BinaryPrimitives.ReadInt64LittleEndian(raw.AsSpan(i * 8));
            return new NpArray(dtype, shape, longs);
        }

        /// <summary>Converts a tuple of Python ints to a shape.</summary>
        /// <param name="shape">The tuple.</param>
        /// <returns>The shape.</returns>
        /// <exception cref="ValueError">An element is not a non-negative int.</exception>
        private int[] ToShape(object?[] shape)
        {
            var dims = new int[shape.Length];
            for (int i = 0; i < shape.Length; i++)
                dims[i] = shape[i] is long d && d >= 0 && d <= int.MaxValue ? (int)d : throw Error("an array shape must hold non-negative ints");
            return dims;
        }

        /// <summary>Decodes pickle's little-endian two's-complement integer.</summary>
        /// <param name="bytes">The encoded bytes; empty means zero.</param>
        /// <returns>The value.</returns>
        /// <exception cref="ValueError">The value does not fit a 64-bit integer.</exception>
        private long DecodeLong(ReadOnlySpan<byte> bytes)
        {
            var big = new BigInteger(bytes, isUnsigned: false, isBigEndian: false);
            if (big < long.MinValue || big > long.MaxValue)
                throw Error("an integer outside the 64-bit range");
            return (long)big;
        }

        /// <summary>Adds the alternating keys and values of a mark group to a dict.</summary>
        /// <param name="dict">The dict.</param>
        /// <param name="items">Key, value, key, value, …</param>
        /// <exception cref="ValueError">The group has an odd number of items.</exception>
        private void SetPairs(PyDict dict, List<object?> items)
        {
            if (items.Count % 2 != 0)
                throw Error("SETITEMS with an odd number of items");
            for (int i = 0; i < items.Count; i += 2)
                dict.Set(items[i], items[i + 1]);
        }

        /// <summary>Pops everything above the last mark, restoring the stack below it.</summary>
        /// <returns>The items above the mark, oldest first.</returns>
        /// <exception cref="ValueError">There is no mark.</exception>
        private List<object?> PopMarkItems()
        {
            if (_marks.Count == 0)
                throw Error("no MARK on the stack");
            var items = new List<object?>(_stack);
            _stack.Clear();
            _stack.AddRange(_marks.Pop());
            return items;
        }

        /// <summary>Pops the top of the stack.</summary>
        /// <returns>The value.</returns>
        /// <exception cref="ValueError">The stack is empty.</exception>
        private object? Pop1()
        {
            if (_stack.Count == 0)
                throw Error("stack underflow");
            object? top = _stack[^1];
            _stack.RemoveAt(_stack.Count - 1);
            return top;
        }

        /// <summary>Reads the top of the stack without popping it.</summary>
        /// <returns>The value.</returns>
        /// <exception cref="ValueError">The stack is empty.</exception>
        private object? Peek() => _stack.Count > 0 ? _stack[^1] : throw Error("stack underflow");

        /// <summary>Returns a memoized value.</summary>
        /// <param name="index">The memo index.</param>
        /// <returns>The value stored under it.</returns>
        /// <exception cref="ValueError">The index was never stored.</exception>
        private object? MemoAt(int index) => _memo.TryGetValue(index, out var value) ? value : throw Error($"memo index {index} was never stored");

        /// <summary>Casts a stack value to a tuple.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The tuple's items.</returns>
        /// <exception cref="ValueError">The value is not a tuple.</exception>
        private object?[] AsTuple(object? value) => value as object?[] ?? throw Error("a tuple was expected");

        /// <summary>Casts a stack value to a list.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The list.</returns>
        /// <exception cref="ValueError">The value is not a list.</exception>
        private List<object?> AsList(object? value) => value as List<object?> ?? throw Error("a list was expected");

        /// <summary>Casts a stack value to a dict.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The dict.</returns>
        /// <exception cref="ValueError">The value is not a dict.</exception>
        private PyDict AsDict(object? value) => value as PyDict ?? throw Error("a dict was expected");

        /// <summary>Reads one byte.</summary>
        /// <returns>The byte.</returns>
        /// <exception cref="ValueError">The data ended.</exception>
        private byte ReadByte() => _pos < _data.Length ? _data[_pos++] : throw Error("the pickle data ended before STOP");

        /// <summary>Reads a run of bytes.</summary>
        /// <param name="count">How many bytes.</param>
        /// <returns>A view of the bytes.</returns>
        /// <exception cref="ValueError">The data ended.</exception>
        private ReadOnlySpan<byte> ReadBytes(int count)
        {
            if (count < 0 || _pos + count > _data.Length)
                throw Error("the pickle data ended before STOP");
            var span = new ReadOnlySpan<byte>(_data, _pos, count);
            _pos += count;
            return span;
        }

        /// <summary>Reads a newline-terminated ASCII line (the text form of <c>GLOBAL</c>).</summary>
        /// <returns>The line without its newline.</returns>
        /// <exception cref="ValueError">The data ended before a newline.</exception>
        private string ReadLine()
        {
            int end = Array.IndexOf(_data, (byte)'\n', _pos);
            if (end < 0)
                throw Error("the pickle data ended before STOP");
            string line = Encoding.ASCII.GetString(_data, _pos, end - _pos);
            _pos = end + 1;
            return line;
        }

        /// <summary>Builds the reader's error, with the offset where it happened.</summary>
        /// <param name="what">What went wrong.</param>
        /// <returns>The exception to throw.</returns>
        private ValueError Error(string what) => new($"Cannot read the pickled CMA-ES optimizer state: {what} (at byte {_pos}).");
    }
}
