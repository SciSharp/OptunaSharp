using System;
using System.Collections;
using System.Globalization;
using System.Text;
using OptunaSharp.Distributions;

namespace OptunaSharp.Internal;

/// <summary>
///     Renders values the way CPython's <c>repr()</c> / <c>str()</c> do, so that OptunaSharp can
///     reproduce Optuna's error and log messages verbatim (a plan §2 ground rule). The float
///     formatting also backs the Python-compatible JSON writer — CPython's <c>json</c> uses
///     <c>float.__repr__</c> for numbers.
/// </summary>
internal static class PythonRepr
{
    /// <summary>
    ///     CPython <c>repr(float)</c>: the shortest decimal string that round-trips, formatted
    ///     fixed when the decimal point position is in (-4, 16], scientific otherwise
    ///     (see CPython <c>pystrtod.c: format_float_short</c>, mode 'r').
    /// </summary>
    /// <param name="value">The value to convert, format, or compare using the operation's stated convention.</param>
    /// <returns>The Python-style floating-point literal, including nonfinite-value spellings.</returns>
    internal static string FloatRepr(double value)
    {
        if (double.IsNaN(value))
            return "nan";
        if (double.IsPositiveInfinity(value))
            return "inf";
        if (double.IsNegativeInfinity(value))
            return "-inf";

        // .NET "R" yields the shortest round-trip decimal, e.g. "1", "-0", "1.5", "1E-05", "1.234E+16".
        string s = value.ToString("R", CultureInfo.InvariantCulture);

        bool negative = s[0] == '-';
        if (negative)
            s = s.Substring(1);

        // Split into digit string and decimal exponent (decpt = position of the decimal point
        // relative to the start of the digit string).
        string digits;
        int decpt;
        int e = s.IndexOfAny(new[] { 'E', 'e' });
        string mantissa = e >= 0 ? s.Substring(0, e) : s;
        int exp = e >= 0 ? int.Parse(s.Substring(e + 1), CultureInfo.InvariantCulture) : 0;
        int dot = mantissa.IndexOf('.');
        if (dot >= 0)
        {
            digits = mantissa.Remove(dot, 1);
            decpt = dot + exp;
        }
        else
        {
            digits = mantissa;
            decpt = mantissa.Length + exp;
        }

        // Strip leading zeros (keep at least one digit), adjusting decpt.
        int lead = 0;
        while (lead < digits.Length - 1 && digits[lead] == '0')
        {
            lead++;
            decpt--;
        }
        digits = digits.Substring(lead);
        // Strip trailing zeros (they carry no round-trip information).
        digits = digits.TrimEnd('0');
        if (digits.Length == 0)
        {
            digits = "0";
            decpt = 1;
        }

        var sb = new StringBuilder();
        if (negative)
            sb.Append('-');

        if (decpt <= -4 || decpt > 16)
        {
            // Scientific: d[.ddd]e±XX with a sign and at least two exponent digits.
            sb.Append(digits[0]);
            if (digits.Length > 1)
            {
                sb.Append('.');
                sb.Append(digits, 1, digits.Length - 1);
            }
            int sciExp = decpt - 1;
            sb.Append('e');
            sb.Append(sciExp < 0 ? '-' : '+');
            int absExp = Math.Abs(sciExp);
            if (absExp < 10)
                sb.Append('0');
            sb.Append(absExp.ToString(CultureInfo.InvariantCulture));
        }
        else if (decpt <= 0)
        {
            // 0.00ddd
            sb.Append("0.");
            sb.Append('0', -decpt);
            sb.Append(digits);
        }
        else if (decpt >= digits.Length)
        {
            // ddd00.0 (ADD_DOT_0: always show a fractional part in fixed notation).
            sb.Append(digits);
            sb.Append('0', decpt - digits.Length);
            sb.Append(".0");
        }
        else
        {
            sb.Append(digits, 0, decpt);
            sb.Append('.');
            sb.Append(digits, decpt, digits.Length - decpt);
        }

        return sb.ToString();
    }

    /// <summary>
    ///     CPython <c>repr(str)</c>: single-quoted unless the text contains <c>'</c> but no <c>"</c>, with
    ///     <c>\\</c>, <c>\n</c>, <c>\r</c>, <c>\t</c> and the quote escaped, and every character Python
    ///     considers non-printable written as <c>\xNN</c>, <c>\uNNNN</c> or <c>\UNNNNNNNN</c>.
    /// </summary>
    /// <param name="value">The value to convert, format, or compare using the operation's stated convention.</param>
    /// <returns>A quoted Python-style string literal with the required escapes.</returns>
    /// <remarks>
    ///     Non-ASCII characters follow <c>unicode_repr</c> in <c>Objects/unicodeobject.c</c>: printable ones
    ///     are kept as they are, the others (format and control characters, separators other than the ASCII
    ///     space, private-use, unassigned and lone surrogate code points) are escaped. A surrogate pair is one
    ///     code point, as it is in a Python <c>str</c>. The verdict comes from .NET's Unicode tables, so a
    ///     character assigned after the Unicode version of the running .NET can differ from CPython's.
    /// </remarks>
    internal static string StringRepr(string value)
    {
        char quote = '\'';
        if (value.IndexOf('\'') >= 0 && value.IndexOf('"') < 0)
            quote = '"';

        var sb = new StringBuilder(value.Length + 2);
        sb.Append(quote);
        for (int i = 0; i < value.Length; i++)
        {
            char c = value[i];
            switch (c)
            {
                case '\\':
                    sb.Append("\\\\");
                    continue;
                case '\n':
                    sb.Append("\\n");
                    continue;
                case '\r':
                    sb.Append("\\r");
                    continue;
                case '\t':
                    sb.Append("\\t");
                    continue;
            }
            if (c == quote)
            {
                sb.Append('\\').Append(c);
                continue;
            }
            if (c < 0x20 || c == 0x7f)
            {
                sb.Append("\\x").Append(((int)c).ToString("x2", CultureInfo.InvariantCulture));
                continue;
            }
            if (c < 0x7f)
            {
                sb.Append(c);
                continue;
            }

            // Python strings hold code points, so a surrogate pair is one character there; a lone surrogate
            // stays a code point of its own (category Cs) and is escaped below.
            int codePoint = c;
            int width = 1;
            if (char.IsHighSurrogate(c) && i + 1 < value.Length && char.IsLowSurrogate(value[i + 1]))
            {
                codePoint = char.ConvertToUtf32(c, value[i + 1]);
                width = 2;
            }
            if (IsPrintable(codePoint))
                sb.Append(value, i, width);
            else if (codePoint <= 0xff)
                sb.Append("\\x").Append(codePoint.ToString("x2", CultureInfo.InvariantCulture));
            else if (codePoint <= 0xffff)
                sb.Append("\\u").Append(codePoint.ToString("x4", CultureInfo.InvariantCulture));
            else
                sb.Append("\\U").Append(codePoint.ToString("x8", CultureInfo.InvariantCulture));
            i += width - 1;
        }
        sb.Append(quote);
        return sb.ToString();
    }

    /// <summary>
    ///     CPython's <c>Py_UNICODE_ISPRINTABLE</c> for a non-ASCII code point: everything except the
    ///     "Other" (Cc, Cf, Cs, Co, Cn) and "Separator" (Zs, Zl, Zp) categories.
    /// </summary>
    /// <param name="codePoint">The Unicode code point to classify; ASCII is handled by the caller.</param>
    /// <returns><see langword="true" /> when <c>repr()</c> writes the character as it is.</returns>
    private static bool IsPrintable(int codePoint)
        => CharUnicodeInfo.GetUnicodeCategory(codePoint) switch
        {
            UnicodeCategory.Control or UnicodeCategory.Format or UnicodeCategory.Surrogate
                or UnicodeCategory.PrivateUse or UnicodeCategory.OtherNotAssigned
                or UnicodeCategory.SpaceSeparator or UnicodeCategory.LineSeparator
                or UnicodeCategory.ParagraphSeparator => false,
            _ => true,
        };

    /// <summary>CPython <c>repr(value)</c> for the JSON-ish object model OptunaSharp carries around.</summary>
    /// <param name="value">The value to convert, format, or compare using the operation's stated convention.</param>
    /// <returns>The Python-style diagnostic representation of the supplied value.</returns>
    internal static string Repr(object? value)
    {
        switch (value)
        {
            case null:
                return "None";
            case bool b:
                return b ? "True" : "False";
            case string s:
                return StringRepr(s);
            case double d:
                return FloatRepr(d);
            case float f:
                return FloatRepr(f);
            case sbyte or byte or short or ushort or int or uint or long:
                return Convert.ToInt64(value, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture);
            case ulong ul:
                return ul.ToString(CultureInfo.InvariantCulture);
            case CategoricalChoice choice:
                return Repr(choice.Value);
            case IDictionary dict:
                return DictRepr(dict);
            case IEnumerable seq:
                return ListRepr(seq);
            default:
                return value.ToString() ?? "None";
        }
    }

    /// <summary>CPython <c>str(value)</c>: like <see cref="Repr" /> but raw for strings.</summary>
    /// <param name="value">The value to convert, format, or compare using the operation's stated convention.</param>
    /// <returns>The Python-style user-facing text for the supplied value.</returns>
    internal static string Str(object? value)
        => value is string s ? s : value is CategoricalChoice c && c.Value is string cs ? cs : Repr(value);

    /// <summary>Python list repr: <c>[a, b]</c>.</summary>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <returns>A bracketed comma-separated Python list representation.</returns>
    internal static string ListRepr(IEnumerable values)
    {
        var sb = new StringBuilder("[");
        bool first = true;
        foreach (var item in values)
        {
            if (!first)
                sb.Append(", ");
            sb.Append(Repr(item));
            first = false;
        }
        return sb.Append(']').ToString();
    }

    /// <summary>Python tuple repr: <c>(a, b)</c>, with the single-element trailing comma <c>(a,)</c>.</summary>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <returns>A parenthesized Python tuple representation, including the singleton comma when needed.</returns>
    internal static string TupleRepr(IEnumerable values)
    {
        var sb = new StringBuilder("(");
        int count = 0;
        foreach (var item in values)
        {
            if (count > 0)
                sb.Append(", ");
            sb.Append(Repr(item));
            count++;
        }
        if (count == 1)
            sb.Append(',');
        return sb.Append(')').ToString();
    }

    /// <summary>Python dict repr: <c>{'a': 1, 'b': 2}</c> in insertion order.</summary>
    /// <param name="dict">The dictionary whose insertion-ordered keys and values are processed.</param>
    /// <returns>A Python dictionary representation preserving enumeration order.</returns>
    internal static string DictRepr(IDictionary dict)
    {
        var sb = new StringBuilder("{");
        bool first = true;
        foreach (DictionaryEntry entry in dict)
        {
            if (!first)
                sb.Append(", ");
            sb.Append(Repr(entry.Key)).Append(": ").Append(Repr(entry.Value));
            first = false;
        }
        return sb.Append('}').ToString();
    }

    /// <summary>Python set repr over string elements: <c>{'a', 'b'}</c> (or <c>set()</c> when empty).</summary>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <returns>A Python set representation, with the empty-set spelling when needed.</returns>
    internal static string SetRepr(IEnumerable values)
    {
        var sb = new StringBuilder("{");
        bool first = true;
        foreach (var item in values)
        {
            if (!first)
                sb.Append(", ");
            sb.Append(Repr(item));
            first = false;
        }
        if (first)
            return "set()";
        return sb.Append('}').ToString();
    }
}
