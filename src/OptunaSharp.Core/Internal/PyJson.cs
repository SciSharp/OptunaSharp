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
    internal static string Dumps(object? value) => Dumps(value, compact: false);

    /// <summary>
    ///     Python <c>json.dumps(value, separators=(",", ":"))</c> — the journal-file format.
    /// </summary>
    internal static string DumpsCompact(object? value) => Dumps(value, compact: true);

    private static string Dumps(object? value, bool compact)
    {
        var sb = new StringBuilder();
        WriteValue(sb, value, compact);
        return sb.ToString();
    }

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
    internal static object? Loads(string text)
    {
        int pos = 0;
        var value = ParseValue(text, ref pos);
        SkipWhitespace(text, ref pos);
        if (pos != text.Length)
            throw new ValueError($"Extra data: line 1 column {pos + 1} (char {pos})");
        return value;
    }

    private static void SkipWhitespace(string s, ref int pos)
    {
        while (pos < s.Length && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
            pos++;
    }

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

    private static void Expect(string s, ref int pos, string literal)
    {
        if (pos + literal.Length > s.Length || string.CompareOrdinal(s, pos, literal, 0, literal.Length) != 0)
            throw new ValueError($"Expecting value: line 1 column {pos + 1} (char {pos})");
        pos += literal.Length;
    }

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
    internal static object? DeepCopy(object? value) => value switch
    {
        List<object?> list => DeepCopyList(list),
        OrderedDict<string, object?> dict => DeepCopyDict(dict),
        _ => value,  // null / bool / long / double / string / immutable values copy by reference.
    };

    private static List<object?> DeepCopyList(List<object?> list)
    {
        var copy = new List<object?>(list.Count);
        foreach (var item in list)
            copy.Add(DeepCopy(item));
        return copy;
    }

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
