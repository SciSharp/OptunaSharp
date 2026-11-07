using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using OptunaSharp.Collections;

namespace OptunaSharp.Internal;

/// <summary>
///     <see cref="PyJson.Loads" /> for UTF-8 bytes: the same grammar, the same object model and the same exceptions,
///     several times faster. The journal reads every record through here, and so does the journal snapshot's reader.
/// </summary>
/// <remarks>
///     <para>
///         <b>Why (plan 10, item 2.7).</b> A cold start replays the whole journal, and parsing dominated it: 100,001
///         records of a 20,000-trial study took 185 ms through <see cref="PyJson.Loads" /> after a UTF-8 decode, the
///         10.9 MB snapshot 392 ms. Reading the bytes directly, building every list and dictionary at its final size
///         from a reusable value stack, and reusing repeated keys and small integers took 115 ms and 122 ms (Release
///         builds, measured on 2026-11-07).
///     </para>
///     <para>
///         <b>The contract: every result is <see cref="PyJson.Loads" />'s.</b> For any input,
///         <c>Loads(bytes)</c> returns what <c>PyJson.Loads(Encoding.UTF8.GetString(bytes))</c> returns, with the
///         same runtime types (an integer literal is a <see cref="long" />, a float a <see cref="double" />), the same
///         key order and the same values, or throws what it throws. Two things make that hold:
///     </para>
///     <list type="bullet">
///         <item>The fast reader takes PyJson's decisions at every position: the same whitespace, literals, number
///         tokens (parsed by the same <see cref="long.TryParse(ReadOnlySpan{char}, NumberStyles, IFormatProvider, out long)" />
///         and <see cref="double.Parse(ReadOnlySpan{char}, NumberStyles, IFormatProvider)" /> calls on the same
///         characters), escapes, duplicate keys (the first place, the last value) and depth limit. Outside JSON strings
///         only ASCII bytes are ever accepted, so byte and character positions agree there; inside strings, text
///         between an opening quote, a backslash and a closing quote decodes as one UTF-8 run, exactly as the whole
///         document would (an ASCII byte always ends a multi-byte sequence, so the decoder's state matches at those
///         boundaries, and its replacement of invalid bytes with U+FFFD does too).</item>
///         <item>Anything the fast reader cannot finish, malformed input above all, is handed to
///         <see cref="PyJson.Loads" /> on the decoded text, which then produces the result or the exception itself.
///         Errors therefore keep PyJson's types and messages (<see cref="NumSharp.ValueError" /> with its
///         character position, <see cref="FormatException" />, <see cref="NumSharp.RuntimeError" /> for depth), and
///         callers that catch <see cref="NumSharp.ValueError" /> to detect a torn last record keep working.</item>
///     </list>
///     <para>
///         The parity tests (<c>PyJsonUtf8Tests</c>) hold the fast reader to this on Python-written records of every
///         kind and on random valid and corrupted documents.
///     </para>
///     <para>
///         <b>Sharing.</b> Results may share instances PyJson would allocate separately: object keys seen before (the
///         same string object), the integers -128 to 1023 and the booleans (the same boxes). All of them are immutable,
///         so only <see cref="object.ReferenceEquals" /> can tell. Each thread reuses one reader and its buffers; a
///         parse never runs user code, so it is never re-entered on its thread.
///     </para>
/// </remarks>
internal static class PyJsonUtf8
{
    /// <summary>Parses a UTF-8 JSON document exactly as <see cref="PyJson.Loads" /> parses its decoded text.</summary>
    /// <param name="utf8">The document's UTF-8 bytes; a byte order mark is not skipped, as PyJson does not skip it.</param>
    /// <returns>The parsed scalar, list, or insertion-ordered dictionary; JSON null becomes null.</returns>
    /// <exception cref="NumSharp.ValueError">The document is malformed or has trailing data after its first value.</exception>
    /// <exception cref="FormatException">A number token is malformed in a way PyJson reports through .NET's parser (for example <c>1.2.3</c>).</exception>
    /// <exception cref="NumSharp.RuntimeError">Arrays and objects nest deeper than <see cref="PyJson.MaxDepth" />.</exception>
    internal static object? Loads(ReadOnlySpan<byte> utf8)
    {
        if (PyJsonUtf8Reader.TryParse(utf8, out object? value))
            return value;
        // The fast reader stopped: malformed input, a construct it leaves to PyJson, or a thread short of stack.
        // PyJson decides from here, so the result or the exception is exactly its own.
        return PyJson.Loads(Encoding.UTF8.GetString(utf8));
    }

    /// <summary>
    ///     Parses JSON text a client library has already decoded (the ServiceStack.Redis records): through the UTF-8
    ///     reader when the text converts to UTF-8 and back unchanged, else through <see cref="PyJson.Loads" />.
    /// </summary>
    /// <param name="text">The document's text.</param>
    /// <returns>The parsed scalar, list, or insertion-ordered dictionary; JSON null becomes null.</returns>
    /// <exception cref="NumSharp.ValueError">The document is malformed or has trailing data after its first value.</exception>
    /// <exception cref="FormatException">A number token is malformed in a way PyJson reports through .NET's parser.</exception>
    /// <exception cref="NumSharp.RuntimeError">Arrays and objects nest deeper than <see cref="PyJson.MaxDepth" />.</exception>
    /// <remarks>
    ///     A surrogate code unit, paired or lone, sends the text to <see cref="PyJson.Loads" /> directly: a lone one
    ///     would turn into U+FFFD on the way to UTF-8 and the parse would differ. Text a client decoded from UTF-8 never
    ///     holds a lone surrogate, and Python's and OptunaSharp's journal records are ASCII.
    /// </remarks>
    internal static object? Loads(string text)
    {
        if (text.AsSpan().IndexOfAnyInRange('\uD800', '\uDFFF') >= 0)
            return PyJson.Loads(text);
        return Loads(Encoding.UTF8.GetBytes(text));
    }
}

/// <summary>
///     The fast reader behind <see cref="PyJsonUtf8" />: PyJson's grammar on UTF-8 bytes, building containers at their
///     final size from a reusable value stack. One instance per thread.
/// </summary>
/// <remarks>
///     Every method mirrors the PyJson method of the same name. Where PyJson raises an error, the reader throws its
///     private <see cref="Miss" /> instead, and <see cref="TryParse" /> reports failure, so that PyJson reparses the
///     text and raises its own error. Exceptions from .NET's number parsers are reported the same way.
/// </remarks>
internal sealed class PyJsonUtf8Reader
{
    // Thread-local, as the buffers are reused: a parse never calls out of this class, so it is never re-entered.
    [ThreadStatic] private static PyJsonUtf8Reader? _current;

    // A value stack shared by every container being built: a container's members sit above the stack height its
    // parse started at until its closing bracket, when they move into a list or dictionary created at their count.
    private object?[] _values = new object?[64];
    private string?[] _keys = new string?[64];
    private int _top;

    // Unescaped text of a string with escapes, before its one allocation.
    private char[] _chars = new char[256];

    private readonly KeyCache _keyCache = new();

    // Buffers a huge document grew past these sizes are dropped after its parse, so a thread does not keep them.
    private const int MaxRetainedStack = 1 << 16;
    private const int MaxRetainedChars = 1 << 16;

    /// <summary>The boxes <see cref="ParseNumber" /> reuses for the integers -128 to 1023.</summary>
    private static readonly object[] SmallIntegers = CreateSmallIntegers();

    /// <summary>The one box for <c>true</c>.</summary>
    private static readonly object True = true;

    /// <summary>The one box for <c>false</c>.</summary>
    private static readonly object False = false;

    /// <summary>The lowest integer <see cref="SmallIntegers" /> holds.</summary>
    private const long SmallIntegerMin = -128;

    /// <summary>One past the highest integer <see cref="SmallIntegers" /> holds.</summary>
    private const long SmallIntegerEnd = 1024;

    /// <summary>Boxes the integers <see cref="SmallIntegerMin" /> to <see cref="SmallIntegerEnd" /> - 1 once.</summary>
    /// <returns>The boxes, indexed by the value minus <see cref="SmallIntegerMin" />.</returns>
    private static object[] CreateSmallIntegers()
    {
        var boxes = new object[SmallIntegerEnd - SmallIntegerMin];
        for (int i = 0; i < boxes.Length; i++)
            boxes[i] = SmallIntegerMin + i;
        return boxes;
    }

    /// <summary>Thrown inside the reader where PyJson would raise an error; never leaves <see cref="TryParse" />.</summary>
    /// <remarks>
    ///     A new instance per throw: the runtime writes stack-trace state into an exception object each time it is
    ///     thrown, so one shared instance thrown on two threads at once would race. Only malformed input throws.
    /// </remarks>
    private sealed class Miss : Exception
    {
    }

    /// <summary>
    ///     Parses a whole document when the fast reader can finish it; otherwise reports failure so the caller can let
    ///     <see cref="PyJson.Loads" /> decide.
    /// </summary>
    /// <param name="utf8">The document's UTF-8 bytes.</param>
    /// <param name="value">The parsed value when this returns <c>true</c>.</param>
    /// <returns><c>true</c> when the document parsed; <c>false</c> when PyJson must parse it (malformed input included).</returns>
    internal static bool TryParse(ReadOnlySpan<byte> utf8, out object? value)
    {
        var reader = _current ??= new PyJsonUtf8Reader();
        try
        {
            int pos = 0;
            value = reader.ParseValue(utf8, ref pos, depth: 0);
            SkipWhitespace(utf8, ref pos);
            if (pos != utf8.Length)
            {
                // PyJson's "Extra data" error.
                value = null;
                return false;
            }
            return true;
        }
        catch (Exception)
        {
            // Miss, or .NET's number parser refusing a token PyJson hands it too: PyJson reparses and raises.
            value = null;
            return false;
        }
        finally
        {
            reader.Reset();
        }
    }

    /// <summary>Empties the value stack, so no parsed value stays reachable, and drops buffers a huge document grew.</summary>
    private void Reset()
    {
        // A failed parse leaves members on the stack; clearing them keeps a thread from holding a document's values.
        Array.Clear(_values, 0, _top);
        Array.Clear(_keys, 0, _top);
        _top = 0;
        if (_values.Length > MaxRetainedStack)
        {
            _values = new object?[64];
            _keys = new string?[64];
        }
        if (_chars.Length > MaxRetainedChars)
            _chars = new char[256];
    }

    /// <summary>Advances the cursor past PyJson's whitespace: space, tab, line feed, carriage return.</summary>
    /// <param name="s">The document.</param>
    /// <param name="pos">The cursor.</param>
    private static void SkipWhitespace(ReadOnlySpan<byte> s, ref int pos)
    {
        while (pos < s.Length)
        {
            byte c = s[pos];
            if (c != (byte)' ' && c != (byte)'\t' && c != (byte)'\n' && c != (byte)'\r')
                return;
            pos++;
        }
    }

    /// <summary>Parses the value at the cursor, as PyJson's <c>ParseValue</c>.</summary>
    /// <param name="s">The document.</param>
    /// <param name="pos">The cursor; advanced past the value.</param>
    /// <param name="depth">The number of containers enclosing the value.</param>
    /// <returns>The value.</returns>
    /// <exception cref="Miss">PyJson would raise an error here.</exception>
    /// <exception cref="FormatException">.NET's number parser refuses the token, as it would in PyJson.</exception>
    private object? ParseValue(ReadOnlySpan<byte> s, ref int pos, int depth)
    {
        SkipWhitespace(s, ref pos);
        if (pos >= s.Length)
            throw new Miss();

        switch (s[pos])
        {
            case (byte)'{':
                return ParseObject(s, ref pos, depth + 1);
            case (byte)'[':
                return ParseArray(s, ref pos, depth + 1);
            case (byte)'"':
                return ParseString(s, ref pos, isKey: false);
            case (byte)'t':
                Expect(s, ref pos, "true"u8);
                return True;
            case (byte)'f':
                Expect(s, ref pos, "false"u8);
                return False;
            case (byte)'n':
                Expect(s, ref pos, "null"u8);
                return null;
            case (byte)'N':
                Expect(s, ref pos, "NaN"u8);
                return double.NaN;
            case (byte)'I':
                Expect(s, ref pos, "Infinity"u8);
                return double.PositiveInfinity;
            default:
                if (s[pos] == (byte)'-' && pos + 1 < s.Length && s[pos + 1] == (byte)'I')
                {
                    pos++;
                    Expect(s, ref pos, "Infinity"u8);
                    return double.NegativeInfinity;
                }
                return ParseNumber(s, ref pos);
        }
    }

    /// <summary>Consumes an ASCII literal at the cursor, as PyJson's <c>Expect</c>.</summary>
    /// <param name="s">The document.</param>
    /// <param name="pos">The cursor; advanced past the literal.</param>
    /// <param name="literal">The literal's bytes.</param>
    /// <exception cref="Miss">The literal is not at the cursor.</exception>
    private static void Expect(ReadOnlySpan<byte> s, ref int pos, ReadOnlySpan<byte> literal)
    {
        if (!s.Slice(pos).StartsWith(literal))
            throw new Miss();
        pos += literal.Length;
    }

    /// <summary>Makes room for one more member on the value stack.</summary>
    /// <param name="key">The member's key, or <c>null</c> for an array element.</param>
    /// <param name="value">The member's value.</param>
    private void Push(string? key, object? value)
    {
        if (_top == _values.Length)
        {
            Array.Resize(ref _values, _top * 2);
            Array.Resize(ref _keys, _top * 2);
        }
        _keys[_top] = key;
        _values[_top++] = value;
    }

    /// <summary>Parses the object at the cursor, as PyJson's <c>ParseObject</c>.</summary>
    /// <param name="s">The document.</param>
    /// <param name="pos">The cursor, on the opening brace; advanced past the closing one.</param>
    /// <param name="depth">The object's own nesting depth; the outermost container is 1.</param>
    /// <returns>The object's members in document order; a repeated key keeps its first place and its last value.</returns>
    /// <exception cref="Miss">The object is malformed, or nests too deeply for PyJson or for this thread's stack.</exception>
    /// <exception cref="FormatException">.NET's number parser refuses a token inside.</exception>
    private OrderedDict<string, object?> ParseObject(ReadOnlySpan<byte> s, ref int pos, int depth)
    {
        // PyJson raises RuntimeError past MaxDepth, and on a thread short of stack; it then raises its own.
        if (depth > PyJson.MaxDepth || !RuntimeHelpers.TryEnsureSufficientExecutionStack())
            throw new Miss();
        pos++;  // '{'
        SkipWhitespace(s, ref pos);
        if (pos < s.Length && s[pos] == (byte)'}')
        {
            pos++;
            return new OrderedDict<string, object?>();
        }

        int start = _top;
        while (true)
        {
            SkipWhitespace(s, ref pos);
            if (pos >= s.Length || s[pos] != (byte)'"')
                throw new Miss();
            string key = ParseString(s, ref pos, isKey: true);
            SkipWhitespace(s, ref pos);
            if (pos >= s.Length || s[pos] != (byte)':')
                throw new Miss();
            pos++;
            object? value = ParseValue(s, ref pos, depth);
            Push(key, value);
            SkipWhitespace(s, ref pos);
            if (pos >= s.Length)
                throw new Miss();
            if (s[pos] == (byte)',')
            {
                pos++;
                continue;
            }
            if (s[pos] == (byte)'}')
            {
                pos++;
                break;
            }
            throw new Miss();
        }

        var result = new OrderedDict<string, object?>(_top - start);
        for (int i = start; i < _top; i++)
        {
            // The indexer, in document order, as PyJson assigns while it parses: a repeated key keeps the place of its
            // first occurrence and the value of its last.
            result[_keys[i]!] = _values[i];
            _keys[i] = null;
            _values[i] = null;
        }
        _top = start;
        return result;
    }

    /// <summary>Parses the array at the cursor, as PyJson's <c>ParseArray</c>.</summary>
    /// <param name="s">The document.</param>
    /// <param name="pos">The cursor, on the opening bracket; advanced past the closing one.</param>
    /// <param name="depth">The array's own nesting depth; the outermost container is 1.</param>
    /// <returns>The array's elements in document order.</returns>
    /// <exception cref="Miss">The array is malformed, or nests too deeply for PyJson or for this thread's stack.</exception>
    /// <exception cref="FormatException">.NET's number parser refuses a token inside.</exception>
    private List<object?> ParseArray(ReadOnlySpan<byte> s, ref int pos, int depth)
    {
        if (depth > PyJson.MaxDepth || !RuntimeHelpers.TryEnsureSufficientExecutionStack())
            throw new Miss();
        pos++;  // '['
        SkipWhitespace(s, ref pos);
        if (pos < s.Length && s[pos] == (byte)']')
        {
            pos++;
            return new List<object?>();
        }

        int start = _top;
        while (true)
        {
            Push(null, ParseValue(s, ref pos, depth));
            SkipWhitespace(s, ref pos);
            if (pos >= s.Length)
                throw new Miss();
            if (s[pos] == (byte)',')
            {
                pos++;
                continue;
            }
            if (s[pos] == (byte)']')
            {
                pos++;
                break;
            }
            throw new Miss();
        }

        var result = new List<object?>(_top - start);
        for (int i = start; i < _top; i++)
        {
            result.Add(_values[i]);
            _values[i] = null;
        }
        _top = start;
        return result;
    }

    /// <summary>Parses the string at the cursor, as PyJson's <c>ParseString</c>.</summary>
    /// <param name="s">The document.</param>
    /// <param name="pos">The cursor, on the opening quote; advanced past the closing one.</param>
    /// <param name="isKey">Whether the string is an object key, whose text is looked up in the key cache.</param>
    /// <returns>The decoded text.</returns>
    /// <exception cref="Miss">The string is unterminated, or an escape is one PyJson refuses.</exception>
    /// <exception cref="FormatException">A <c>\u</c> escape's four characters are not a number <see cref="ushort.Parse(ReadOnlySpan{char}, NumberStyles, IFormatProvider)" /> accepts, as in PyJson.</exception>
    private string ParseString(ReadOnlySpan<byte> s, ref int pos, bool isKey)
    {
        pos++;  // opening '"'
        int start = pos;
        int run = s.Slice(pos).IndexOfAny((byte)'"', (byte)'\\');
        if (run < 0)
            throw new Miss();
        if (s[pos + run] == (byte)'"')
        {
            // No escape: the bytes between the quotes are the text, decoded as one UTF-8 run.
            pos += run + 1;
            ReadOnlySpan<byte> raw = s.Slice(start, run);
            return isKey ? _keyCache.GetOrAdd(raw) : Encoding.UTF8.GetString(raw);
        }

        // Escapes: unescape into the reusable buffer, then allocate the string once. The text never has more
        // characters than the raw bytes it came from, which bounds each step's room.
        int n = 0;
        Span<char> hex = stackalloc char[4];
        while (true)
        {
            run = s.Slice(pos).IndexOfAny((byte)'"', (byte)'\\');
            if (run < 0)
                throw new Miss();
            EnsureChars(n + run + 1);
            if (run > 0)
                n += Encoding.UTF8.GetChars(s.Slice(pos, run), _chars.AsSpan(n));
            pos += run;
            if (s[pos++] == (byte)'"')
                return new string(_chars, 0, n);
            if (pos >= s.Length)
                throw new Miss();

            char c;
            switch (s[pos++])
            {
                case (byte)'"': c = '"'; break;
                case (byte)'\\': c = '\\'; break;
                case (byte)'/': c = '/'; break;
                case (byte)'b': c = '\b'; break;
                case (byte)'f': c = '\f'; break;
                case (byte)'n': c = '\n'; break;
                case (byte)'r': c = '\r'; break;
                case (byte)'t': c = '\t'; break;
                case (byte)'u':
                    if (pos + 4 > s.Length)
                        throw new Miss();
                    for (int i = 0; i < 4; i++)
                    {
                        // PyJson takes the next four characters; only when they are ASCII are they these four bytes.
                        byte h = s[pos + i];
                        if (h >= 0x80)
                            throw new Miss();
                        hex[i] = (char)h;
                    }
                    // PyJson's call on the same four characters: hexadecimal, surrounding whitespace allowed.
                    c = (char)ushort.Parse(hex, NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                    pos += 4;
                    break;
                default:
                    throw new Miss();
            }
            EnsureChars(n + 1);
            _chars[n++] = c;
        }
    }

    /// <summary>Grows the unescaping buffer to hold at least <paramref name="needed" /> characters.</summary>
    /// <param name="needed">The characters the buffer must hold.</param>
    private void EnsureChars(int needed)
    {
        if (needed > _chars.Length)
            Array.Resize(ref _chars, Math.Max(needed, _chars.Length * 2));
    }

    /// <summary>Parses the number at the cursor, as PyJson's <c>ParseNumber</c>.</summary>
    /// <param name="s">The document.</param>
    /// <param name="pos">The cursor; advanced past the number's token.</param>
    /// <returns>A <see cref="long" /> for an integer literal that fits, else a <see cref="double" />.</returns>
    /// <exception cref="Miss">No token starts at the cursor (PyJson's "Expecting value").</exception>
    /// <exception cref="FormatException">.NET's parser refuses the token, as it does in PyJson (for example <c>1.2.3</c>).</exception>
    private static object ParseNumber(ReadOnlySpan<byte> s, ref int pos)
    {
        int start = pos;
        if (pos < s.Length && s[pos] == (byte)'-')
            pos++;
        bool isDouble = false;
        while (pos < s.Length)
        {
            byte c = s[pos];
            if ((uint)(c - '0') <= 9)
            {
                pos++;
            }
            else if (c == (byte)'.' || c == (byte)'e' || c == (byte)'E' || c == (byte)'+' || c == (byte)'-')
            {
                isDouble = isDouble || c == (byte)'.' || c == (byte)'e' || c == (byte)'E';
                pos++;
            }
            else
            {
                break;
            }
        }

        int length = pos - start;
        if (length == 0 || (length == 1 && s[start] == (byte)'-'))
            throw new Miss();

        // The token's bytes are ASCII digits and signs, so widening them gives PyJson's substring; parse it with the
        // very calls PyJson makes, for the same results and the same refusals.
        Span<char> token = length <= 64 ? stackalloc char[length] : new char[length];
        for (int i = 0; i < length; i++)
            token[i] = (char)s[start + i];
        if (!isDouble && long.TryParse(token, NumberStyles.Integer, CultureInfo.InvariantCulture, out long integer))
        {
            return integer >= SmallIntegerMin && integer < SmallIntegerEnd
                ? SmallIntegers[integer - SmallIntegerMin]
                : integer;
        }
        return double.Parse(token, NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    /// <summary>
    ///     Reuses the string of an object key seen before: a direct-mapped table from a key's raw bytes to its string,
    ///     for keys of up to <see cref="MaxKeyBytes" /> bytes. Journal records repeat a handful of keys
    ///     (<c>op_code</c>, <c>worker_id</c>, <c>trial_id</c>, ...), so most lookups hit.
    /// </summary>
    private sealed class KeyCache
    {
        /// <summary>The longest key cached; longer keys are decoded each time.</summary>
        private const int MaxKeyBytes = 32;

        /// <summary>The table's size, a power of two.</summary>
        private const int Slots = 1024;

        private readonly byte[]?[] _bytes = new byte[]?[Slots];
        private readonly string?[] _strings = new string?[Slots];

        /// <summary>Returns the key's string, from the table when the same bytes were seen last in its slot.</summary>
        /// <param name="raw">The key's raw bytes, between the quotes and without escapes.</param>
        /// <returns>The key's text: equal to <c>Encoding.UTF8.GetString(raw)</c>.</returns>
        internal string GetOrAdd(ReadOnlySpan<byte> raw)
        {
            if (raw.Length > MaxKeyBytes)
                return Encoding.UTF8.GetString(raw);

            // FNV-1a over the bytes; a collision only costs a decode, since the bytes are compared before reuse.
            uint hash = 2166136261;
            foreach (byte b in raw)
                hash = (hash ^ b) * 16777619;
            int slot = (int)(hash & (Slots - 1));

            byte[]? known = _bytes[slot];
            if (known is not null && raw.SequenceEqual(known))
                return _strings[slot]!;

            string text = Encoding.UTF8.GetString(raw);
            _bytes[slot] = raw.ToArray();
            _strings[slot] = text;
            return text;
        }
    }
}
