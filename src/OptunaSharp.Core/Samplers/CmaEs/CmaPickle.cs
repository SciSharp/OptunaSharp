using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.CmaEs;

/// <summary>
///     The CMA-ES optimizer state as Optuna's <c>CmaEsSampler</c> stores it: <c>pickle.dumps(optimizer).hex()</c> of a
///     cmaes <c>CMA</c>, <c>SepCMA</c> or <c>CMAwM</c>. OptunaSharp writes and reads the same pickles, so C# and Python
///     workers can share one CMA-ES study in either order (review item P3-2).
/// </summary>
/// <remarks>
///     <para>
///         <b>Layout.</b> The states follow cmaes 0.13.1. <c>CMA</c> and <c>SepCMA</c> pickle the same attributes in the
///         same order in every cmaes Optuna supports (0.12.0 on); <c>CMAwM</c> changed in 0.13.1, so a with-margin study
///         needs cmaes 0.13.1 on its Python workers.
///     </para>
///     <para>
///         <b>Byte identity.</b> A state OptunaSharp writes is byte-identical to the one cmaes writes for the same
///         optimizer, including Python's scalar types and numpy's dtype sharing (see <see cref="CmaPickleLayout" />).
///         The one thing no other process can reproduce is the content numpy leaves in <c>_funhist_values</c>, which
///         cmaes allocates with <c>np.empty</c> and fills two entries per generation: OptunaSharp starts it at zeros.
///     </para>
/// </remarks>
internal static class CmaPickle
{
    /// <summary>
    ///     Pickles an optimizer the way Optuna stores it: <c>pickle.dumps(optimizer).hex()</c>, lowercase.
    /// </summary>
    /// <param name="optimizer">A <see cref="Cma" />, <see cref="SepCma" /> or <see cref="CmaWm" />.</param>
    /// <param name="restored">
    ///     Whether the optimizer was restored from a stored state before its last tell (true for every state the sampler
    ///     stores except the first of a study). Python's pickle of a restored optimizer orders <c>CMA</c>'s covariance
    ///     last and keeps the dtype objects unpickling created, so this decides the bytes, not the values.
    /// </param>
    /// <returns>The hex string Optuna's <c>_restore_optimizer</c> reads with <c>bytes.fromhex</c>.</returns>
    /// <exception cref="NotSupportedException">The optimizer is another implementation of <see cref="ICmaOptimizer" />.</exception>
    internal static string ToHex(ICmaOptimizer optimizer, bool restored)
        => Convert.ToHexString(Dumps(optimizer, restored)).ToLowerInvariant();

    /// <summary>Pickles an optimizer as cmaes 0.13.1 would; see <see cref="ToHex" />.</summary>
    /// <param name="optimizer">A <see cref="Cma" />, <see cref="SepCma" /> or <see cref="CmaWm" />.</param>
    /// <param name="restored">Whether the optimizer was restored from a stored state before its last tell.</param>
    /// <returns>The pickle's bytes.</returns>
    /// <exception cref="NotSupportedException">The optimizer is another implementation of <see cref="ICmaOptimizer" />.</exception>
    internal static byte[] Dumps(ICmaOptimizer optimizer, bool restored)
    {
        var layout = new CmaPickleLayout(restored);
        PyInstance state = optimizer switch
        {
            Cma c => c.ToPyInstance(layout),
            SepCma s => s.ToPyInstance(layout),
            CmaWm w => w.ToPyInstance(layout),
            _ => throw new NotSupportedException($"Cannot pickle a CMA-ES optimizer of type {optimizer.GetType().Name}."),
        };
        return PythonPickle.Dumps(state);
    }

    /// <summary>
    ///     Restores an optimizer from the hex text Optuna (or OptunaSharp) stored: <c>pickle.loads(bytes.fromhex(text))</c>.
    /// </summary>
    /// <param name="hex">The concatenated <c>…:optimizer:{i}</c> chunks.</param>
    /// <returns>The optimizer the pickle holds, as its own class (a one-dimensional "separable" study stores a CMA).</returns>
    /// <exception cref="ValueError">
    ///     The text is not hexadecimal (Python's <c>bytes.fromhex</c> message, with the position), or the pickle is not
    ///     a cmaes optimizer state this version reads (see <see cref="Loads" />).
    /// </exception>
    internal static ICmaOptimizer FromHex(string hex) => Loads(FromHexString(hex));

    /// <summary>Unpickles a cmaes optimizer state and restores the matching OptunaSharp optimizer.</summary>
    /// <param name="pickle">The pickle's bytes.</param>
    /// <returns>A <see cref="Cma" />, <see cref="SepCma" /> or <see cref="CmaWm" />, chosen by the pickled class.</returns>
    /// <exception cref="ValueError">
    ///     The pickle is malformed, references anything but numpy and the cmaes optimizer classes, is not an optimizer,
    ///     or holds a state that is not cmaes 0.13.1's (a missing, mistyped or misshapen attribute).
    /// </exception>
    internal static ICmaOptimizer Loads(byte[] pickle)
        => PythonPickle.Loads(pickle) switch
        {
            PyInstance { Module: "cmaes._cma", QualName: "CMA" } s => Cma.FromPyInstance(s),
            PyInstance { Module: "cmaes._sepcma", QualName: "SepCMA" } s => SepCma.FromPyInstance(s),
            PyInstance { Module: "cmaes._cmawm", QualName: "CMAwM" } s => CmaWm.FromPyInstance(s),
            var other => throw new ValueError(
                $"The stored CMA-ES optimizer state is not a cmaes optimizer but {Describe(other)}."),
        };

    /// <summary>
    ///     Python's <c>bytes.fromhex</c>: two hex digits per byte, either case, ASCII whitespace allowed between bytes.
    /// </summary>
    /// <param name="hex">The text.</param>
    /// <returns>The bytes.</returns>
    /// <exception cref="ValueError">
    ///     A character is not a hex digit, or the last byte has one digit: Python 3.12's
    ///     <c>non-hexadecimal number found in fromhex() arg at position N</c>, N being the first non-ASCII character
    ///     when there is one (Python checks that first), else the offending character. This is what a Python worker
    ///     reports for the JSON state OptunaSharp stored before 2026-10-29.
    /// </exception>
    internal static byte[] FromHexString(string hex)
    {
        for (int i = 0; i < hex.Length; i++)
            if (hex[i] > 127)
                throw NonHex(i);
        var bytes = new byte[hex.Length / 2];
        int count = 0, pos = 0;
        while (pos < hex.Length)
        {
            if (IsPySpace(hex[pos]))
            {
                while (pos < hex.Length && IsPySpace(hex[pos]))
                    pos++;
                if (pos >= hex.Length)
                    break;
            }
            int top = HexDigit(hex[pos]);
            if (top < 0)
                throw NonHex(pos);
            pos++;
            int bottom = pos < hex.Length ? HexDigit(hex[pos]) : -1;
            if (bottom < 0)
                throw NonHex(pos);
            pos++;
            bytes[count++] = (byte)((top << 4) | bottom);
        }
        Array.Resize(ref bytes, count);
        return bytes;
    }

    /// <summary>The value of a hex digit, or -1.</summary>
    /// <param name="c">An ASCII character.</param>
    /// <returns>0 to 15, or -1 when <paramref name="c" /> is not a hex digit.</returns>
    private static int HexDigit(char c) => c switch
    {
        >= '0' and <= '9' => c - '0',
        >= 'a' and <= 'f' => c - 'a' + 10,
        >= 'A' and <= 'F' => c - 'A' + 10,
        _ => -1,
    };

    /// <summary>CPython's <c>Py_ISSPACE</c>: the six ASCII whitespace characters <c>bytes.fromhex</c> skips.</summary>
    /// <param name="c">A character.</param>
    /// <returns>True for space, tab, line feed, vertical tab, form feed and carriage return.</returns>
    private static bool IsPySpace(char c) => c is ' ' or '\t' or '\n' or '\v' or '\f' or '\r';

    /// <summary>Builds Python's <c>bytes.fromhex</c> error.</summary>
    /// <param name="position">The offending character's index.</param>
    /// <returns>The exception to throw.</returns>
    private static ValueError NonHex(int position)
        => new($"non-hexadecimal number found in fromhex() arg at position {position}");

    /// <summary>Names an unpickled value for an error message.</summary>
    /// <param name="value">The value.</param>
    /// <returns>The pickled class, or the C# type of a plain value.</returns>
    private static string Describe(object? value) => value switch
    {
        null => "None",
        PyInstance p => $"{p.Module}.{p.QualName}",
        _ => value.GetType().Name,
    };

    // ---- state readers: one attribute of an unpickled instance, type- and shape-checked --------

    /// <summary>Reads a Python int attribute that must fit an <see cref="int" />.</summary>
    /// <param name="s">The unpickled instance.</param>
    /// <param name="key">The attribute name.</param>
    /// <returns>The value.</returns>
    /// <exception cref="ValueError">The attribute is missing, not an int, or out of range.</exception>
    internal static int Int(PyInstance s, string key)
        => s.Get(key) is long v && v >= int.MinValue && v <= int.MaxValue
            ? (int)v
            : throw Mistyped(s, key, "an int");

    /// <summary>
    ///     Reads a real-valued attribute: a <c>numpy.float64</c>, a <c>float</c> or an <c>int</c> (cmaes stores
    ///     <c>_cm = 1</c>).
    /// </summary>
    /// <param name="s">The unpickled instance.</param>
    /// <param name="key">The attribute name.</param>
    /// <returns>The value as a double.</returns>
    /// <exception cref="ValueError">The attribute is missing or not a number.</exception>
    internal static double Float(PyInstance s, string key) => s.Get(key) switch
    {
        NpFloat64 v => v.Value,
        double v => v,
        long v => v,
        _ => throw Mistyped(s, key, "a number"),
    };

    /// <summary>Whether a real-valued attribute is a <c>numpy.float64</c> rather than a Python number.</summary>
    /// <param name="s">The unpickled instance.</param>
    /// <param name="key">The attribute name.</param>
    /// <returns>True for a numpy scalar.</returns>
    /// <exception cref="ValueError">The attribute is missing.</exception>
    internal static bool IsNumpy(PyInstance s, string key) => s.Get(key) is NpFloat64;

    /// <summary>Reads a Python bool attribute.</summary>
    /// <param name="s">The unpickled instance.</param>
    /// <param name="key">The attribute name.</param>
    /// <returns>The value.</returns>
    /// <exception cref="ValueError">The attribute is missing or not a bool.</exception>
    internal static bool Bool(PyInstance s, string key)
        => s.Get(key) is bool v ? v : throw Mistyped(s, key, "a bool");

    /// <summary>Reads a one-dimensional float64 array attribute of a known length.</summary>
    /// <param name="s">The unpickled instance.</param>
    /// <param name="key">The attribute name.</param>
    /// <param name="length">The length the optimizer's dimensions require.</param>
    /// <returns>A copy of the elements.</returns>
    /// <exception cref="ValueError">The attribute is missing, not a float64 array, or has another shape.</exception>
    internal static double[] Vector(PyInstance s, string key, int length)
        => s.Get(key) is NpArray { Doubles: { } d, Shape: { Length: 1 } shape } && shape[0] == length
            ? (double[])d.Clone()
            : throw Mistyped(s, key, $"a float64 array of shape ({length},)");

    /// <summary>Reads a one-dimensional float64 array attribute that may be <c>None</c>.</summary>
    /// <param name="s">The unpickled instance.</param>
    /// <param name="key">The attribute name.</param>
    /// <param name="length">The length the optimizer's dimensions require.</param>
    /// <returns>A copy of the elements, or null for <c>None</c>.</returns>
    /// <exception cref="ValueError">The attribute is missing, or neither <c>None</c> nor such an array.</exception>
    internal static double[]? OptionalVector(PyInstance s, string key, int length)
        => s.Get(key) is null ? null : Vector(s, key, length);

    /// <summary>Reads a two-dimensional float64 array attribute of a known shape.</summary>
    /// <param name="s">The unpickled instance.</param>
    /// <param name="key">The attribute name.</param>
    /// <param name="rows">The number of rows the optimizer's dimensions require.</param>
    /// <param name="cols">The number of columns.</param>
    /// <returns>A copy of the elements (C order is row-major, as a C# rectangular array).</returns>
    /// <exception cref="ValueError">The attribute is missing, not a float64 array, or has another shape.</exception>
    internal static double[,] Matrix(PyInstance s, string key, int rows, int cols)
    {
        if (s.Get(key) is not NpArray { Doubles: { } d, Shape: { Length: 2 } shape } || shape[0] != rows || shape[1] != cols)
            throw Mistyped(s, key, $"a float64 array of shape ({rows}, {cols})");
        var m = new double[rows, cols];
        Buffer.BlockCopy(d, 0, m, 0, d.Length * sizeof(double));
        return m;
    }

    /// <summary>Reads a two-dimensional float64 array attribute that may be <c>None</c>.</summary>
    /// <param name="s">The unpickled instance.</param>
    /// <param name="key">The attribute name.</param>
    /// <param name="rows">The number of rows the optimizer's dimensions require.</param>
    /// <param name="cols">The number of columns.</param>
    /// <returns>A copy of the elements, or null for <c>None</c>.</returns>
    /// <exception cref="ValueError">The attribute is missing, or neither <c>None</c> nor such an array.</exception>
    internal static double[,]? OptionalMatrix(PyInstance s, string key, int rows, int cols)
        => s.Get(key) is null ? null : Matrix(s, key, rows, cols);

    /// <summary>Reads a one-dimensional int64 array attribute of a known length.</summary>
    /// <param name="s">The unpickled instance.</param>
    /// <param name="key">The attribute name.</param>
    /// <param name="length">The length the optimizer's dimensions require.</param>
    /// <returns>A copy of the elements.</returns>
    /// <exception cref="ValueError">The attribute is missing, not an int64 array, or has another shape.</exception>
    internal static long[] Int64Vector(PyInstance s, string key, int length)
        => s.Get(key) is NpArray { Longs: { } l, Shape: { Length: 1 } shape } && shape[0] == length
            ? (long[])l.Clone()
            : throw Mistyped(s, key, $"an int64 array of shape ({length},)");

    /// <summary>Reads an int64 array of coordinate indices, each in <c>[0, dim)</c>.</summary>
    /// <param name="s">The unpickled instance.</param>
    /// <param name="key">The attribute name.</param>
    /// <param name="length">The number of indices.</param>
    /// <param name="dim">The optimizer's number of coordinates.</param>
    /// <returns>The indices.</returns>
    /// <exception cref="ValueError">The attribute is missing, misshapen, or holds an index outside the coordinates.</exception>
    internal static int[] IndexVector(PyInstance s, string key, int length, int dim)
    {
        long[] raw = Int64Vector(s, key, length);
        var idx = new int[raw.Length];
        for (int i = 0; i < raw.Length; i++)
            idx[i] = raw[i] >= 0 && raw[i] < dim ? (int)raw[i] : throw Mistyped(s, key, $"indices in [0, {dim})");
        return idx;
    }

    /// <summary>Builds the reader's error for an attribute that is not what cmaes stores there.</summary>
    /// <param name="s">The unpickled instance.</param>
    /// <param name="key">The attribute name.</param>
    /// <param name="expected">What cmaes 0.13.1 stores there.</param>
    /// <returns>The exception to throw.</returns>
    private static ValueError Mistyped(PyInstance s, string key, string expected)
        => new($"The pickled {s.Module}.{s.QualName} attribute '{key}' is not {expected}.");

    // ---- value helpers shared by the optimizers' state mappings --------------------------------

    /// <summary>
    ///     A number Python holds as an <c>int</c> when it is integral (cmaes' <c>_cm = 1</c>), else as a <c>float</c>.
    /// </summary>
    /// <param name="v">The value.</param>
    /// <returns>A <see cref="long" /> for an integral value in range, else the double.</returns>
    internal static object PyNumber(double v)
        => v == Math.Floor(v) && Math.Abs(v) < 9.2e18 ? (object)(long)v : v;

    /// <summary>
    ///     cmaes' <c>_compress_symmetric</c>: the upper triangle, row by row (row <c>i</c> from column <c>i</c>).
    /// </summary>
    /// <param name="m">A square matrix.</param>
    /// <returns>The <c>n(n+1)/2</c> upper-triangle entries.</returns>
    internal static double[] CompressSymmetric(double[,] m)
    {
        int n = m.GetLength(0);
        var sym1d = new double[n * (n + 1) / 2];
        int k = 0;
        for (int i = 0; i < n; i++)
            for (int j = i; j < n; j++)
                sym1d[k++] = m[i, j];
        return sym1d;
    }

    /// <summary>
    ///     cmaes' <c>_decompress_symmetric</c>: both triangles from the stored upper triangle, so the matrix is exactly
    ///     symmetric.
    /// </summary>
    /// <param name="sym1d">The upper triangle, as <see cref="CompressSymmetric" /> lays it out.</param>
    /// <param name="n">The matrix order; <paramref name="sym1d" /> has <c>n(n+1)/2</c> entries.</param>
    /// <returns>The symmetric matrix.</returns>
    internal static double[,] DecompressSymmetric(double[] sym1d, int n)
    {
        var m = new double[n, n];
        int k = 0;
        for (int i = 0; i < n; i++)
            for (int j = i; j < n; j++)
            {
                m[i, j] = sym1d[k];
                m[j, i] = sym1d[k];
                k++;
            }
        return m;
    }
}

/// <summary>
///     The numpy dtype objects of one CMA-ES pickle, and the rule that assigns them, so a pickled state shares dtype
///     objects exactly as cmaes' would.
/// </summary>
/// <remarks>
///     <para>
///         numpy pickles each distinct dtype object once and references it after, so which arrays share one is part of
///         the bytes. In an optimizer built and told in one process every float64 value uses numpy's builtin
///         <c>float64</c> dtype (<see cref="Canonical" />), and so does every numpy scalar, always. Unpickling a state
///         creates one new dtype object per pickled one; a tell then rebuilds some arrays on the builtin dtype (a numpy
///         result takes its first operand's dtype object, and those expressions start from a scalar) while the arrays it
///         updates in place, or builds from an untouched array, keep the unpickled one (<see cref="Carried" />).
///     </para>
///     <para>
///         Each optimizer's mapping names, per array, whether its tell rebuilds it; the set is the same every tell, so a
///         state never holds more than these two float64 dtype objects. Measured on 20 seeded runs × 16 stored states per
///         optimizer variant against cmaes 0.13.1 and numpy 2.4.2. All int64 arrays (CMAwM's indices and sizes) share
///         one dtype object in either case.
///     </para>
/// </remarks>
internal sealed class CmaPickleLayout
{
    /// <summary>Creates the dtype objects for one pickle.</summary>
    /// <param name="restored">Whether the optimizer was restored from a stored state before its last tell.</param>
    internal CmaPickleLayout(bool restored) => Restored = restored;

    /// <summary>Whether the optimizer was restored from a stored state before its last tell.</summary>
    internal bool Restored { get; }

    /// <summary>numpy's builtin float64 dtype: every scalar's, and every array's in an optimizer never restored.</summary>
    internal NpDtype Canonical { get; } = new("f8");

    /// <summary>The float64 dtype object unpickling created, kept by the arrays a tell does not rebuild.</summary>
    internal NpDtype Carried { get; } = new("f8");

    /// <summary>The int64 dtype object all integer arrays share.</summary>
    internal NpDtype Int64 { get; } = new("i8");

    /// <summary>The dtype object a float64 array of the optimizer's state uses.</summary>
    /// <param name="rebuiltByTell">Whether every tell replaces the array by a new one on the builtin dtype.</param>
    /// <returns><see cref="Carried" /> for a kept array of a restored optimizer, else <see cref="Canonical" />.</returns>
    internal NpDtype F8(bool rebuiltByTell) => Restored && !rebuiltByTell ? Carried : Canonical;

    /// <summary>A <c>numpy.float64</c> scalar.</summary>
    /// <param name="v">The value.</param>
    /// <returns>A new scalar on the builtin dtype.</returns>
    internal NpFloat64 Scalar(double v) => new(v, Canonical);

    /// <summary>A real value as Python holds it: a <c>numpy.float64</c> or a <c>float</c>.</summary>
    /// <param name="v">The value.</param>
    /// <param name="isNumpy">Whether Python holds a numpy scalar.</param>
    /// <returns>An <see cref="NpFloat64" /> or the boxed double.</returns>
    internal object Float(double v, bool isNumpy) => isNumpy ? Scalar(v) : v;

    /// <summary>A one-dimensional float64 array.</summary>
    /// <param name="v">The elements; copied.</param>
    /// <param name="rebuiltByTell">Whether every tell replaces the array (see <see cref="F8" />).</param>
    /// <returns>The array.</returns>
    internal NpArray Vector(double[] v, bool rebuiltByTell) => new(F8(rebuiltByTell), new[] { v.Length }, (double[])v.Clone());

    /// <summary>A two-dimensional float64 array.</summary>
    /// <param name="m">The elements, row-major as numpy's C order; copied.</param>
    /// <param name="rebuiltByTell">Whether every tell replaces the array (see <see cref="F8" />).</param>
    /// <returns>The array.</returns>
    internal NpArray Matrix(double[,] m, bool rebuiltByTell)
    {
        var data = new double[m.Length];
        Buffer.BlockCopy(m, 0, data, 0, data.Length * sizeof(double));
        return new NpArray(F8(rebuiltByTell), new[] { m.GetLength(0), m.GetLength(1) }, data);
    }

    /// <summary>A one-dimensional int64 array of coordinate indices.</summary>
    /// <param name="v">The elements.</param>
    /// <returns>The array, on the shared int64 dtype.</returns>
    internal NpArray Int64Vector(int[] v)
    {
        var data = new long[v.Length];
        for (int i = 0; i < v.Length; i++)
            data[i] = v[i];
        return new NpArray(Int64, new[] { v.Length }, data);
    }

    /// <summary>A one-dimensional int64 array.</summary>
    /// <param name="v">The elements; copied.</param>
    /// <returns>The array, on the shared int64 dtype.</returns>
    internal NpArray Int64Vector(long[] v) => new(Int64, new[] { v.Length }, (long[])v.Clone());
}
