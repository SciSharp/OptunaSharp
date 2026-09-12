using System;
using System.Collections;
using System.Globalization;
using System.Collections.Generic;
using System.Text;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp.Internal;

/// <summary>
///     A self-contained JSON reader/writer matching CPython's <c>json</c> module semantics.
///
///     System.Text.Json cannot express what Optuna's storage formats need — <c>NaN</c> /
///     <c>Infinity</c> / <c>-Infinity</c> literals (Python's <c>allow_nan=True</c> default),
///     Python <c>repr</c> float formatting (<c>1.0</c>, not <c>1</c> — CPython's json uses
///     <c>float.__repr__</c>), <c>ensure_ascii=True</c> escaping, and insertion-ordered objects.
///     The journal-file format is byte-compatible with Python's because both sides serialize
///     through these rules.
///
///     The object model is: <c>null</c>, <see cref="bool" />, <see cref="long" />,
///     <see cref="double" />, <see cref="string" />, <see cref="List{T}" /> of <c>object?</c>,
///     and <see cref="OrderedDict{TKey,TValue}" /> of <c>string → object?</c> — the exact image
///     of Python's <c>json.loads</c> output.
/// </summary>
internal static class PyJson
{
    // ------------------------------------------------------------------ writing

    /// <summary>
    ///     Python <c>json.dumps(value)</c> with the default separators <c>(", ", ": ")</c>.
    /// </summary>
    /// <param name="value">The value to convert, format, or compare using the operation's stated convention.</param>
    /// <returns>The Python-compatible JSON representation of the value.</returns>
    /// <exception cref="NumSharp.TypeError">The value or one of its nested values has no supported JSON representation.</exception>
    internal static string Dumps(object? value) => Dumps(value, compact: false);

    /// <summary>
    ///     Python <c>json.dumps(value, separators=(",", ":"))</c> — the journal-file format.
    /// </summary>
    /// <param name="value">The value to convert, format, or compare using the operation's stated convention.</param>
    /// <returns>The JSON representation with compact separators.</returns>
    /// <exception cref="NumSharp.TypeError">The value or one of its nested values has no supported JSON representation.</exception>
    internal static string DumpsCompact(object? value) => Dumps(value, compact: true);

    /// <summary>Returns the Python-compatible JSON representation of the value.</summary>
    /// <param name="value">The value to convert, format, or compare using the operation's stated convention.</param>
    /// <param name="compact">Whether separators omit the spaces used by Python's default JSON formatting.</param>
    /// <returns>The Python-compatible JSON representation of the value.</returns>
    /// <exception cref="NumSharp.TypeError">The value or one of its nested values has no supported JSON representation.</exception>
    private static string Dumps(object? value, bool compact)
    {
        var sb = new StringBuilder();
        WriteValue(sb, value, compact);
        return sb.ToString();
    }

    /// <summary>
    ///     Python <c>json.dumps(value, indent=n)</c> — pretty-printed with an <paramref name="indent" />-space
    ///     step, the separators <c>(",", ": ")</c>, and <c>{}</c>/<c>[]</c> for empty containers. Leaves
    ///     reuse the same writers as <see cref="Dumps(object)" />, so floats/strings/NaN stay Python-identical.
    /// </summary>
    /// <param name="value">The value to convert, format, or compare using the operation's stated convention.</param>
    /// <param name="indent">The number of spaces per JSON nesting level.</param>
    /// <returns>The JSON representation with the requested indentation width.</returns>
    /// <exception cref="NumSharp.TypeError">The value or one of its nested values has no supported JSON representation.</exception>
    internal static string DumpsIndented(object? value, int indent)
    {
        var sb = new StringBuilder();
        WriteValueIndented(sb, value, indent, 0);
        return sb.ToString();
    }

    /// <summary>Dispatches a JSON-compatible value to the indented scalar, list, or mapping writer.</summary>
    /// <param name="sb">The text buffer to append serialized output to.</param>
    /// <param name="value">The value to convert, format, or compare using the operation's stated convention.</param>
    /// <param name="indent">The number of spaces per JSON nesting level.</param>
    /// <param name="level">The current JSON nesting depth.</param>
    /// <exception cref="NumSharp.TypeError">The value or one of its nested values has no supported JSON representation.</exception>
    private static void WriteValueIndented(StringBuilder sb, object? value, int indent, int level)
    {
        switch (value)
        {
            case IDictionary dict:
                WriteObjectIndented(sb, dict, indent, level);
                break;
            case string:
                WriteValue(sb, value, compact: false);
                break;
            case IEnumerable seq:
                WriteArrayIndented(sb, seq, indent, level);
                break;
            default:
                WriteValue(sb, value, compact: false);  // leaf (null/bool/number/enum/CategoricalChoice).
                break;
        }
    }

    /// <summary>Appends an insertion-ordered JSON object with indentation at the current nesting depth.</summary>
    /// <param name="sb">The text buffer to append serialized output to.</param>
    /// <param name="dict">The dictionary whose insertion-ordered keys and values are processed.</param>
    /// <param name="indent">The number of spaces per JSON nesting level.</param>
    /// <param name="level">The current JSON nesting depth.</param>
    /// <exception cref="NumSharp.TypeError">The value or one of its nested values has no supported JSON representation.</exception>
    private static void WriteObjectIndented(StringBuilder sb, IDictionary dict, int indent, int level)
    {
        if (dict.Count == 0)
        {
            sb.Append("{}");
            return;
        }
        sb.Append("{\n");
        string childPad = new string(' ', indent * (level + 1));
        bool first = true;
        foreach (DictionaryEntry entry in dict)
        {
            if (!first)
                sb.Append(",\n");
            sb.Append(childPad);
            WriteString(sb, KeyToString(entry.Key));
            sb.Append(": ");
            WriteValueIndented(sb, entry.Value, indent, level + 1);
            first = false;
        }
        sb.Append('\n').Append(' ', indent * level).Append('}');
    }

    /// <summary>Appends an indented JSON array in source enumeration order.</summary>
    /// <param name="sb">The text buffer to append serialized output to.</param>
    /// <param name="seq">The sequence whose elements are serialized in enumeration order.</param>
    /// <param name="indent">The number of spaces per JSON nesting level.</param>
    /// <param name="level">The current JSON nesting depth.</param>
    /// <exception cref="NumSharp.TypeError">The value or one of its nested values has no supported JSON representation.</exception>
    private static void WriteArrayIndented(StringBuilder sb, IEnumerable seq, int indent, int level)
    {
        var e = seq.GetEnumerator();
        if (!e.MoveNext())
        {
            sb.Append("[]");
            return;
        }
        sb.Append("[\n");
        string childPad = new string(' ', indent * (level + 1));
        bool first = true;
        do
        {
            if (!first)
                sb.Append(",\n");
            sb.Append(childPad);
            WriteValueIndented(sb, e.Current, indent, level + 1);
            first = false;
        } while (e.MoveNext());
        sb.Append('\n').Append(' ', indent * level).Append(']');
    }

    /// <summary>Dispatches a JSON-compatible value to the scalar, list, or mapping writer.</summary>
    /// <param name="sb">The text buffer to append serialized output to.</param>
    /// <param name="value">The value to convert, format, or compare using the operation's stated convention.</param>
    /// <param name="compact">Whether separators omit the spaces used by Python's default JSON formatting.</param>
    /// <exception cref="NumSharp.TypeError">A supplied value has no supported JSON representation.</exception>
    private static void WriteValue(StringBuilder sb, object? value, bool compact)
    {
        switch (value)
        {
            case null:
                sb.Append("null");
                break;
            case bool b:
                sb.Append(b ? "true" : "false");
                break;
            case string s:
                WriteString(sb, s);
                break;
            case double d:
                WriteDouble(sb, d);
                break;
            case float f:
                WriteDouble(sb, f);
                break;
            case sbyte or byte or short or ushort or int or uint or long:
                sb.Append(Convert.ToInt64(value, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture));
                break;
            case ulong ul:
                sb.Append(ul.ToString(CultureInfo.InvariantCulture));
                break;
            case decimal m:
                WriteDouble(sb, (double)m);
                break;
            case Enum e:
                // Python IntEnum members (TrialState, StudyDirection, JournalOperation)
                // serialize as their integer value.
                sb.Append(Convert.ToInt64(e, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture));
                break;
            case CategoricalChoice choice:
                WriteValue(sb, choice.Value, compact);
                break;
            case IDictionary dict:
                WriteObject(sb, dict, compact);
                break;
            case IEnumerable seq:
                WriteArray(sb, seq, compact);
                break;
            default:
                throw new TypeError($"Object of type {value.GetType().Name} is not JSON serializable");
        }
    }

    /// <summary>Appends a floating-point token using Python-compatible finite and nonfinite spellings.</summary>
    /// <param name="sb">The text buffer to append serialized output to.</param>
    /// <param name="d">The floating-point value whose JSON token is appended.</param>
    private static void WriteDouble(StringBuilder sb, double d)
    {
        // Python: allow_nan=True writes the (non-standard) literals NaN / Infinity / -Infinity.
        if (double.IsNaN(d))
            sb.Append("NaN");
        else if (double.IsPositiveInfinity(d))
            sb.Append("Infinity");
        else if (double.IsNegativeInfinity(d))
            sb.Append("-Infinity");
        else
            sb.Append(PythonRepr.FloatRepr(d));
    }

    /// <summary>Appends a quoted JSON string with the required control-character and Unicode escapes.</summary>
    /// <param name="sb">The text buffer to append serialized output to.</param>
    /// <param name="s">The unescaped text to quote and encode as a JSON string.</param>
    private static void WriteString(StringBuilder sb, string s)
    {
        // Python default: ensure_ascii=True — non-ASCII escapes to \uXXXX.
        sb.Append('"');
        foreach (char c in s)
        {
            switch (c)
            {
                case '"':
                    sb.Append("\\\"");
                    break;
                case '\\':
                    sb.Append("\\\\");
                    break;
                case '\n':
                    sb.Append("\\n");
                    break;
                case '\r':
                    sb.Append("\\r");
                    break;
                case '\t':
                    sb.Append("\\t");
                    break;
                case '\b':
                    sb.Append("\\b");
                    break;
                case '\f':
                    sb.Append("\\f");
                    break;
                default:
                    if (c < 0x20 || c > 0x7e)
                        sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    else
                        sb.Append(c);
                    break;
            }
        }
        sb.Append('"');
    }

    /// <summary>Appends an insertion-ordered JSON mapping with the requested separator spacing.</summary>
    /// <param name="sb">The text buffer to append serialized output to.</param>
    /// <param name="dict">The dictionary whose insertion-ordered keys and values are processed.</param>
    /// <param name="compact">Whether separators omit the spaces used by Python's default JSON formatting.</param>
    /// <exception cref="NumSharp.TypeError">The value or one of its nested values has no supported JSON representation.</exception>
    private static void WriteObject(StringBuilder sb, IDictionary dict, bool compact)
    {
        sb.Append('{');
        bool first = true;
        foreach (DictionaryEntry entry in dict)
        {
            if (!first)
                sb.Append(compact ? "," : ", ");
            WriteString(sb, KeyToString(entry.Key));
            sb.Append(compact ? ":" : ": ");
            WriteValue(sb, entry.Value, compact);
            first = false;
        }
        sb.Append('}');
    }

    /// <summary>Appends a JSON array with the requested separator spacing.</summary>
    /// <param name="sb">The text buffer to append serialized output to.</param>
    /// <param name="seq">The sequence whose elements are serialized in enumeration order.</param>
    /// <param name="compact">Whether separators omit the spaces used by Python's default JSON formatting.</param>
    /// <exception cref="NumSharp.TypeError">The value or one of its nested values has no supported JSON representation.</exception>
    private static void WriteArray(StringBuilder sb, IEnumerable seq, bool compact)
    {
        sb.Append('[');
        bool first = true;
        foreach (var item in seq)
        {
            if (!first)
                sb.Append(compact ? "," : ", ");
            WriteValue(sb, item, compact);
            first = false;
        }
        sb.Append(']');
    }

    /// <summary>Python json key coercion: str as-is, int/float/bool/None keys stringified Python-style.</summary>
    /// <param name="key">The key whose associated entry is requested.</param>
    /// <returns>The JSON object-key text using Python-compatible scalar formatting.</returns>
    private static string KeyToString(object? key) => key switch
    {
        string s => s,
        bool b => b ? "true" : "false",
        null => "null",
        double d => double.IsNaN(d) ? "NaN"
            : double.IsPositiveInfinity(d) ? "Infinity"
            : double.IsNegativeInfinity(d) ? "-Infinity"
            : PythonRepr.FloatRepr(d),
        Enum e => Convert.ToInt64(e, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture),
        _ => Convert.ToInt64(key, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture),
    };

    // ------------------------------------------------------------------ reading

    /// <summary>Python <c>json.loads(text)</c>.</summary>
    /// <param name="text">The Python-compatible JSON document to parse.</param>
    /// <returns>The parsed scalar, list, or insertion-ordered dictionary; JSON null becomes null.</returns>
    /// <exception cref="NumSharp.ValueError">The document is malformed or contains trailing data after its first value.</exception>
    internal static object? Loads(string text)
    {
        int pos = 0;
        var value = ParseValue(text, ref pos);
        SkipWhitespace(text, ref pos);
        if (pos != text.Length)
            throw new ValueError($"Extra data: line 1 column {pos + 1} (char {pos})");
        return value;
    }

    /// <summary>Advances the parse cursor past JSON whitespace characters.</summary>
    /// <param name="s">The source text being parsed or formatted.</param>
    /// <param name="pos">The current character position; parsing advances it past consumed input.</param>
    private static void SkipWhitespace(string s, ref int pos)
    {
        while (pos < s.Length && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
            pos++;
    }

    /// <summary>Returns the parsed value, with the cursor advanced past its token.</summary>
    /// <param name="s">The JSON source text containing the value at the current cursor.</param>
    /// <param name="pos">The current character position; parsing advances it past consumed input.</param>
    /// <returns>The parsed value, with the cursor advanced past its token.</returns>
    /// <exception cref="NumSharp.ValueError">The cursor does not begin a valid JSON value or the nested value is malformed.</exception>
    private static object? ParseValue(string s, ref int pos)
    {
        SkipWhitespace(s, ref pos);
        if (pos >= s.Length)
            throw new ValueError("Expecting value: line 1 column 1 (char 0)");

        char c = s[pos];
        switch (c)
        {
            case '{':
                return ParseObject(s, ref pos);
            case '[':
                return ParseArray(s, ref pos);
            case '"':
                return ParseString(s, ref pos);
            case 't':
                Expect(s, ref pos, "true");
                return true;
            case 'f':
                Expect(s, ref pos, "false");
                return false;
            case 'n':
                Expect(s, ref pos, "null");
                return null;
            case 'N':
                Expect(s, ref pos, "NaN");
                return double.NaN;
            case 'I':
                Expect(s, ref pos, "Infinity");
                return double.PositiveInfinity;
            default:
                if (c == '-' && pos + 1 < s.Length && s[pos + 1] == 'I')
                {
                    pos++;
                    Expect(s, ref pos, "Infinity");
                    return double.NegativeInfinity;
                }
                return ParseNumber(s, ref pos);
        }
    }

    /// <summary>Consumes the expected literal or rejects the malformed JSON token.</summary>
    /// <param name="s">The source text being parsed or formatted.</param>
    /// <param name="pos">The current character position; parsing advances it past consumed input.</param>
    /// <param name="literal">The exact JSON token expected at the current parse position.</param>
    /// <exception cref="NumSharp.ValueError">The expected literal does not occur at the current cursor.</exception>
    private static void Expect(string s, ref int pos, string literal)
    {
        if (pos + literal.Length > s.Length || string.CompareOrdinal(s, pos, literal, 0, literal.Length) != 0)
            throw new ValueError($"Expecting value: line 1 column {pos + 1} (char {pos})");
        pos += literal.Length;
    }

    /// <summary>Returns the parsed insertion-ordered mapping, with the cursor advanced past the closing brace.</summary>
    /// <param name="s">The JSON source text containing the object at the current cursor.</param>
    /// <param name="pos">The current character position; parsing advances it past consumed input.</param>
    /// <returns>The parsed insertion-ordered mapping, with the cursor advanced past the closing brace.</returns>
    /// <exception cref="NumSharp.ValueError">An object key, delimiter, or nested value is malformed.</exception>
    private static OrderedDict<string, object?> ParseObject(string s, ref int pos)
    {
        var result = new OrderedDict<string, object?>();
        pos++;  // '{'
        SkipWhitespace(s, ref pos);
        if (pos < s.Length && s[pos] == '}')
        {
            pos++;
            return result;
        }
        while (true)
        {
            SkipWhitespace(s, ref pos);
            if (pos >= s.Length || s[pos] != '"')
                throw new ValueError($"Expecting property name enclosed in double quotes: line 1 column {pos + 1} (char {pos})");
            string key = ParseString(s, ref pos);
            SkipWhitespace(s, ref pos);
            if (pos >= s.Length || s[pos] != ':')
                throw new ValueError($"Expecting ':' delimiter: line 1 column {pos + 1} (char {pos})");
            pos++;
            result[key] = ParseValue(s, ref pos);  // Python keeps the LAST duplicate key.
            SkipWhitespace(s, ref pos);
            if (pos >= s.Length)
                throw new ValueError($"Expecting ',' delimiter: line 1 column {pos + 1} (char {pos})");
            if (s[pos] == ',')
            {
                pos++;
                continue;
            }
            if (s[pos] == '}')
            {
                pos++;
                return result;
            }
            throw new ValueError($"Expecting ',' delimiter: line 1 column {pos + 1} (char {pos})");
        }
    }

    /// <summary>Returns the parsed list, with the cursor advanced past the closing bracket.</summary>
    /// <param name="s">The JSON source text containing the array at the current cursor.</param>
    /// <param name="pos">The current character position; parsing advances it past consumed input.</param>
    /// <returns>The parsed list, with the cursor advanced past the closing bracket.</returns>
    /// <exception cref="NumSharp.ValueError">An array delimiter or nested value is malformed.</exception>
    private static List<object?> ParseArray(string s, ref int pos)
    {
        var result = new List<object?>();
        pos++;  // '['
        SkipWhitespace(s, ref pos);
        if (pos < s.Length && s[pos] == ']')
        {
            pos++;
            return result;
        }
        while (true)
        {
            result.Add(ParseValue(s, ref pos));
            SkipWhitespace(s, ref pos);
            if (pos >= s.Length)
                throw new ValueError($"Expecting ',' delimiter: line 1 column {pos + 1} (char {pos})");
            if (s[pos] == ',')
            {
                pos++;
                continue;
            }
            if (s[pos] == ']')
            {
                pos++;
                return result;
            }
            throw new ValueError($"Expecting ',' delimiter: line 1 column {pos + 1} (char {pos})");
        }
    }

    /// <summary>Returns the decoded string, with the cursor advanced past the closing quote.</summary>
    /// <param name="s">The JSON source text containing the quoted string at the current cursor.</param>
    /// <param name="pos">The current character position; parsing advances it past consumed input.</param>
    /// <returns>The decoded string, with the cursor advanced past the closing quote.</returns>
    /// <exception cref="NumSharp.ValueError">The string is unterminated or contains an unsupported or incomplete escape.</exception>
    private static string ParseString(string s, ref int pos)
    {
        pos++;  // opening '"'
        var sb = new StringBuilder();
        while (true)
        {
            if (pos >= s.Length)
                throw new ValueError($"Unterminated string starting at: line 1 column {pos} (char {pos - 1})");
            char c = s[pos++];
            if (c == '"')
                return sb.ToString();
            if (c == '\\')
            {
                if (pos >= s.Length)
                    throw new ValueError($"Unterminated string starting at: line 1 column {pos} (char {pos - 1})");
                char esc = s[pos++];
                switch (esc)
                {
                    case '"': sb.Append('"'); break;
                    case '\\': sb.Append('\\'); break;
                    case '/': sb.Append('/'); break;
                    case 'b': sb.Append('\b'); break;
                    case 'f': sb.Append('\f'); break;
                    case 'n': sb.Append('\n'); break;
                    case 'r': sb.Append('\r'); break;
                    case 't': sb.Append('\t'); break;
                    case 'u':
                        if (pos + 4 > s.Length)
                            throw new ValueError($"Invalid \\uXXXX escape: line 1 column {pos} (char {pos - 1})");
                        sb.Append((char)ushort.Parse(s.Substring(pos, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture));
                        pos += 4;
                        break;
                    default:
                        throw new ValueError($"Invalid \\escape: line 1 column {pos} (char {pos - 1})");
                }
            }
            else
            {
                sb.Append(c);
            }
        }
    }

    /// <summary>Returns the parsed integer or floating-point value, with the cursor advanced past its token.</summary>
    /// <param name="s">The JSON source text containing the number at the current cursor.</param>
    /// <param name="pos">The current character position; parsing advances it past consumed input.</param>
    /// <returns>The parsed integer or floating-point value, with the cursor advanced past its token.</returns>
    /// <exception cref="NumSharp.ValueError">No supported numeric token begins at the current cursor.</exception>
    private static object ParseNumber(string s, ref int pos)
    {
        int start = pos;
        if (pos < s.Length && s[pos] == '-')
            pos++;
        bool isDouble = false;
        while (pos < s.Length)
        {
            char c = s[pos];
            if (c >= '0' && c <= '9')
            {
                pos++;
            }
            else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')
            {
                isDouble = isDouble || c == '.' || c == 'e' || c == 'E';
                pos++;
            }
            else
            {
                break;
            }
        }
        string token = s.Substring(start, pos - start);
        if (token.Length == 0 || token == "-")
            throw new ValueError($"Expecting value: line 1 column {start + 1} (char {start})");

        if (!isDouble)
        {
            // Python json parses integer literals as int (arbitrary precision); long covers
            // every value Optuna produces. Overflow falls back to double.
            if (long.TryParse(token, NumberStyles.Integer, CultureInfo.InvariantCulture, out long l))
                return l;
        }
        return double.Parse(token, NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    // ------------------------------------------------------------------ deep copy

    /// <summary>
    ///     Deep-copies a JSON-model value (Python <c>copy.deepcopy</c> over json-serializable data).
    /// </summary>
    /// <param name="value">The value to convert, format, or compare using the operation's stated convention.</param>
    /// <returns>A recursively copied JSON-compatible value; immutable scalar values are reused.</returns>
    internal static object? DeepCopy(object? value) => value switch
    {
        List<object?> list => DeepCopyList(list),
        OrderedDict<string, object?> dict => DeepCopyDict(dict),
        _ => value,  // null / bool / long / double / string / immutable values copy by reference.
    };

    /// <summary>Returns a new list containing recursively copied element values.</summary>
    /// <param name="list">The list whose nested values are copied recursively.</param>
    /// <returns>A new list containing recursively copied element values.</returns>
    private static List<object?> DeepCopyList(List<object?> list)
    {
        var copy = new List<object?>(list.Count);
        foreach (var item in list)
            copy.Add(DeepCopy(item));
        return copy;
    }

    /// <summary>Returns a new insertion-ordered dictionary containing recursively copied values.</summary>
    /// <param name="dict">The dictionary whose insertion-ordered keys and values are processed.</param>
    /// <returns>A new insertion-ordered dictionary containing recursively copied values.</returns>
    private static OrderedDict<string, object?> DeepCopyDict(OrderedDict<string, object?> dict)
    {
        var copy = new OrderedDict<string, object?>(dict.Count);
        foreach (var kv in dict)
            copy[kv.Key] = DeepCopy(kv.Value);
        return copy;
    }

    /// <summary>
    ///     Python-<c>==</c> equality over the JSON object model (used by attr comparisons in tests
    ///     and by <c>FrozenTrial.__eq__</c>).
    /// </summary>
    /// <param name="a">The first numerical operand; its dimensions must agree with the operation described in the summary.</param>
    /// <param name="b">The second numerical operand; its dimensions must agree with the first operand.</param>
    /// <returns>True when the JSON-compatible values are structurally equal under Python-compatible numeric comparison.</returns>
    internal static bool ValueEquals(object? a, object? b)
    {
        if (a is null || b is null)
            return a is null && b is null;
        if (a is bool ba)
            return b is bool bb ? ba == bb
                : b is long lb2 ? (ba ? 1 : 0) == lb2
                : b is double db2 && (ba ? 1.0 : 0.0) == db2;
        if (b is bool)
            return ValueEquals(b, a);
        if (a is long la)
            return b is long lb ? la == lb : b is double db && la == db;
        if (a is double da)
            return b is double db3 ? da == db3 : b is long lb3 && da == lb3;
        if (a is string sa)
            return b is string sb && sa == sb;
        if (a is List<object?> listA)
        {
            if (b is not List<object?> listB || listA.Count != listB.Count)
                return false;
            for (int i = 0; i < listA.Count; i++)
                if (!ValueEquals(listA[i], listB[i]))
                    return false;
            return true;
        }
        if (a is OrderedDict<string, object?> dictA)
        {
            if (b is not OrderedDict<string, object?> dictB || dictA.Count != dictB.Count)
                return false;
            foreach (var kv in dictA)
            {
                if (!dictB.TryGetValue(kv.Key, out var other) || !ValueEquals(kv.Value, other))
                    return false;
            }
            return true;
        }
        return a.Equals(b);
    }
}
