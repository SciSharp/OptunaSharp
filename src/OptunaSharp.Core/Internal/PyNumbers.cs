using System;
using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;
using NumSharp;

namespace OptunaSharp.Internal;

/// <summary>
///     CPython's <c>int(str)</c> and <c>float(str)</c>, with their exact acceptance rules and error messages.
///     Storage URLs reach Python's drivers as strings (a SQLite <c>?timeout=</c>, a Redis <c>?db=</c>, a port),
///     and the drivers convert them with these two builtins, so the port must accept and refuse the same text.
/// </summary>
/// <remarks>
///     <para>What CPython accepts and .NET's parsers do not: Unicode decimal digits (<c>"٥"</c> is 5), Unicode
///     whitespace around the number, and single underscores between digits (<c>"1_000"</c>). What .NET accepts
///     and CPython does not: thousands separators, hexadecimal, leading/trailing characters .NET's
///     <see cref="NumberStyles" /> tolerate. Both functions therefore validate the text first and use .NET only
///     to convert a string already known to be well formed.</para>
///     <para>The rewriting before the grammar check follows <c>_PyUnicode_TransformDecimalAndSpaceToASCII</c>:
///     every non-ASCII Unicode decimal digit becomes its ASCII digit, every non-ASCII whitespace character a
///     space, and the first non-ASCII character that is neither ends the text with an invalid marker. ASCII is
///     kept as it is, which is why <c>"\x1c5"</c> (a C0 separator that <c>str.isspace</c> accepts) is still
///     refused: CPython's number parsers only skip the six C whitespace characters.</para>
/// </remarks>
internal static class PyNumbers
{
    // _Py_dg_strtod's grammar after the underscores are gone: an optional sign, then a decimal mantissa with at
    // least one digit and an optional exponent, or one of the two special words (any case).
    private static readonly Regex FloatGrammar = new(
        @"\A[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|(?i:inf|infinity|nan))\z",
        RegexOptions.CultureInvariant);

    /// <summary>
    ///     Python's <c>float(Decimal)</c>: the double nearest to the decimal's exact value, ties to even. Every port of
    ///     Python <c>Decimal</c> arithmetic converts its result through here instead of <c>(double)decimal</c>.
    /// </summary>
    /// <remarks>
    ///     .NET's explicit <c>(double)decimal</c> conversion is not correctly rounded and can land one ULP off. It did
    ///     for 8.6% of realistic stepped float ranges (the adjusted <c>high</c>, so C# and Python workers held
    ///     different distributions for one parameter, QA-2) and for 7% of BruteForce float candidates (QA-11).
    ///     <see cref="decimal.ToString(IFormatProvider)" /> prints every stored digit with no exponent, and
    ///     <see cref="double.Parse(string, NumberStyles, IFormatProvider)" /> is IEEE correctly rounded since .NET Core
    ///     3.0, so the round trip through the exact digits is the correctly rounded conversion. A negative zero stays
    ///     negative, as Python's conversion keeps the sign; callers that model Python's sign of a zero sum normalize
    ///     before converting (QA-1).
    /// </remarks>
    /// <param name="value">The decimal to convert; every decimal is finite, so the result is always finite.</param>
    /// <returns>The correctly rounded double, with the decimal's sign.</returns>
    internal static double DecimalToDouble(decimal value)
        => double.Parse(value.ToString(CultureInfo.InvariantCulture), NumberStyles.Float, CultureInfo.InvariantCulture);

    /// <summary>
    ///     Python <c>int(text)</c> in base 10: optional whitespace and sign, decimal digits (any Unicode decimal
    ///     digit counts) with single underscores allowed only between two digits.
    /// </summary>
    /// <param name="text">The text to convert.</param>
    /// <returns>The value, unbounded like a Python <c>int</c>; callers check the range they need.</returns>
    /// <exception cref="ValueError">
    ///     The text is not a base-10 integer literal; the message is CPython's
    ///     <c>invalid literal for int() with base 10: '…'</c>, with the repr cut to 200 characters as CPython's
    ///     <c>%.200R</c> does.
    /// </exception>
    internal static BigInteger ParseInt(string text)
    {
        if (!TryParseInt(text, out BigInteger value))
            throw new ValueError(IntLiteralMessage(text));
        return value;
    }

    /// <summary>The non-throwing form of <see cref="ParseInt" />.</summary>
    /// <param name="text">The text to convert.</param>
    /// <param name="value">The converted value when the text is a valid literal, otherwise zero.</param>
    /// <returns><see langword="true" /> when Python's <c>int(text)</c> succeeds.</returns>
    internal static bool TryParseInt(string text, out BigInteger value)
    {
        value = BigInteger.Zero;
        string ascii = TransformDecimalAndSpaceToAscii(text);
        int start = 0;
        int end = ascii.Length;
        while (start < end && IsCSpace(ascii[start]))
            start++;
        while (end > start && IsCSpace(ascii[end - 1]))
            end--;
        bool negative = false;
        if (start < end && (ascii[start] == '+' || ascii[start] == '-'))
        {
            negative = ascii[start] == '-';
            start++;
        }

        // PyLong_FromString: digits, each underscore between two digits, at least one digit.
        var digits = new StringBuilder(end - start);
        char previous = '\0';
        for (int i = start; i < end; i++)
        {
            char c = ascii[i];
            if (c == '_')
            {
                if (previous is < '0' or > '9')
                    return false;
            }
            else if (c is >= '0' and <= '9')
            {
                digits.Append(c);
            }
            else
            {
                return false;
            }
            previous = c;
        }
        if (previous == '_' || digits.Length == 0)
            return false;

        value = BigInteger.Parse(digits.ToString(), NumberStyles.None, CultureInfo.InvariantCulture);
        if (negative)
            value = -value;
        return true;
    }

    /// <summary>
    ///     Python <c>float(text)</c>: optional whitespace, a sign, a decimal mantissa with an optional exponent or
    ///     <c>inf</c>/<c>infinity</c>/<c>nan</c> in any case; single underscores only between digits; an
    ///     out-of-range magnitude becomes an infinity or a zero, as in CPython.
    /// </summary>
    /// <param name="text">The text to convert.</param>
    /// <returns>The IEEE 754 double CPython produces (correctly rounded, as both runtimes round).</returns>
    /// <exception cref="ValueError">
    ///     The text is not a float literal; the message is CPython's
    ///     <c>could not convert string to float: '…'</c> with the full repr of the original text.
    /// </exception>
    internal static double ParseFloat(string text)
    {
        if (!TryParseFloat(text, out double value))
            throw new ValueError("could not convert string to float: " + PythonRepr.StringRepr(text));
        return value;
    }

    /// <summary>The non-throwing form of <see cref="ParseFloat" />.</summary>
    /// <param name="text">The text to convert.</param>
    /// <param name="value">The converted value when the text is a valid literal, otherwise zero.</param>
    /// <returns><see langword="true" /> when Python's <c>float(text)</c> succeeds.</returns>
    internal static bool TryParseFloat(string text, out double value)
    {
        value = 0;
        string ascii = TransformDecimalAndSpaceToAscii(text);

        // _Py_string_to_number_with_underscores runs before whitespace is stripped, on the whole text: an
        // underscore must follow a digit and precede one, so "1_000" passes and "1_.5" or "_1" do not.
        if (ascii.IndexOf('_') >= 0)
        {
            var kept = new StringBuilder(ascii.Length);
            char previous = '\0';
            foreach (char c in ascii)
            {
                if (c == '_')
                {
                    if (previous is < '0' or > '9')
                        return false;
                }
                else
                {
                    if (previous == '_' && c is < '0' or > '9')
                        return false;
                    kept.Append(c);
                }
                previous = c;
            }
            if (previous == '_')
                return false;
            ascii = kept.ToString();
        }

        int start = 0;
        int end = ascii.Length;
        while (start < end && IsCSpace(ascii[start]))
            start++;
        while (end > start && IsCSpace(ascii[end - 1]))
            end--;
        string core = ascii.Substring(start, end - start);
        if (!FloatGrammar.IsMatch(core))
            return false;

        string lower = core.ToLowerInvariant();
        string unsigned = lower.TrimStart('+', '-');
        bool negative = lower.StartsWith('-');
        if (unsigned is "inf" or "infinity")
        {
            value = negative ? double.NegativeInfinity : double.PositiveInfinity;
            return true;
        }
        if (unsigned == "nan")
        {
            // CPython keeps the sign bit of "-nan"; .NET's NaN constant has it set, so build both explicitly.
            value = negative ? -Math.Abs(double.NaN) : Math.Abs(double.NaN);
            return true;
        }

        // The grammar is validated, so .NET's correctly rounded parser now sees exactly the IEEE conversion
        // _Py_dg_strtod performs; an overflowing exponent yields an infinity on both sides.
        value = double.Parse(core, NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint | NumberStyles.AllowExponent,
            CultureInfo.InvariantCulture);
        return true;
    }

    /// <summary>
    ///     CPython's message for a failed <c>int(text)</c>: <c>invalid literal for int() with base 10: </c>
    ///     followed by the text's repr, cut to its first 200 characters as the <c>%.200R</c> format does.
    /// </summary>
    /// <param name="text">The text whose conversion failed.</param>
    /// <returns>The verbatim message.</returns>
    internal static string IntLiteralMessage(string text)
    {
        string repr = PythonRepr.StringRepr(text);
        if (repr.Length > 200)
            repr = repr.Substring(0, 200);
        return "invalid literal for int() with base 10: " + repr;
    }

    /// <summary>
    ///     Converts a double to a C <c>int</c> the way CPython's C code does on x86-64 (<c>(int)value</c>, which
    ///     compiles to <c>cvttsd2si</c>): truncation toward zero, and <see cref="int.MinValue" /> — the
    ///     "integer indefinite" value — for NaN, an infinity or anything outside the 32-bit range.
    /// </summary>
    /// <param name="value">The double to truncate.</param>
    /// <returns>The value an x86-64 CPython computes for <c>(int)value</c>.</returns>
    /// <remarks>
    ///     .NET's own cast saturates on .NET 9 and later and returns the indefinite value on .NET 8, so the
    ///     conversion is spelled out to give one answer on every target framework. CPython on ARM64 saturates
    ///     instead (C leaves the out-of-range case undefined); the x86-64 answer is the one used here because it
    ///     is what CPython computes on the platforms Optuna's continuous integration runs on.
    /// </remarks>
    internal static int CTruncateToInt32(double value)
    {
        if (double.IsNaN(value) || value <= int.MinValue - 1.0 || value >= int.MaxValue + 1.0)
            return int.MinValue;
        return (int)Math.Truncate(value);
    }

    /// <summary>
    ///     <c>_PyUnicode_TransformDecimalAndSpaceToASCII</c>: ASCII characters below 0x7F stay, a non-ASCII
    ///     whitespace character becomes a space, a non-ASCII decimal digit its ASCII digit, and the first other
    ///     character turns into <c>'?'</c> and ends the result.
    /// </summary>
    /// <param name="text">The text to rewrite.</param>
    /// <returns>The ASCII text CPython hands to its number parsers.</returns>
    private static string TransformDecimalAndSpaceToAscii(string text)
    {
        var sb = new StringBuilder(text.Length);
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c < 0x7f)
            {
                sb.Append(c);
                continue;
            }
            // Decimal digits outside the BMP (e.g. U+1D7D3, a mathematical bold 5) are surrogate pairs here.
            int width = char.IsHighSurrogate(c) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]) ? 2 : 1;
            if (width == 1 && IsUnicodeSpace(c))
            {
                sb.Append(' ');
                continue;
            }
            int digit = CharUnicodeInfo.GetDecimalDigitValue(text, i);
            if (digit < 0)
            {
                sb.Append('?');
                break;
            }
            sb.Append((char)('0' + digit));
            i += width - 1;
        }
        return sb.ToString();
    }

    /// <summary>The six characters C's <c>isspace</c> accepts in the "C" locale (<c>Py_ISSPACE</c>).</summary>
    /// <param name="c">The ASCII character to test.</param>
    /// <returns><see langword="true" /> for space, tab, newline, vertical tab, form feed and carriage return.</returns>
    private static bool IsCSpace(char c) => c is ' ' or '\t' or '\n' or '\v' or '\f' or '\r';

    /// <summary>
    ///     <c>Py_UNICODE_ISSPACE</c> for a non-ASCII character: the characters with Unicode bidirectional type
    ///     WS, B or S, or general category Zs.
    /// </summary>
    /// <param name="c">The non-ASCII character to test.</param>
    /// <returns><see langword="true" /> when CPython treats the character as whitespace.</returns>
    private static bool IsUnicodeSpace(char c)
        => c is '\u0085' or '\u00A0' or '\u1680' or '\u2028' or '\u2029' or '\u202F' or '\u205F' or '\u3000'
           || c is >= '\u2000' and <= '\u200A';
}
