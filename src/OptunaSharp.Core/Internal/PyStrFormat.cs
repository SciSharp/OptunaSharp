using System;
using System.Globalization;
using System.Numerics;
using System.Text;
using NumSharp;

namespace OptunaSharp.Internal;

/// <summary>
///     CPython 3.12's <c>str.format</c> for a template with one keyword argument whose value is an <c>int</c> — the
///     call <c>RDBStorage._fill_storage_url_template</c> makes (<c>url.format(SCHEMA_VERSION=12)</c>) on every storage
///     URL before SQLAlchemy sees it.
/// </summary>
/// <remarks>
///     <para><b>Why a port and not <see cref="string.Replace(string, string)" />.</b> <c>str.format</c> gives every
///     brace a meaning: <c>{{</c> and <c>}}</c> are escapes for one brace, any other <c>{name}</c> is a field that
///     must exist, and a lone <c>}</c> is an error. A password such as <c>p{{w</c> therefore reaches the database as
///     <c>p{w</c> in Python, and <c>p}w</c> or <c>p{x}w</c> makes Python refuse the URL. Replacing only the literal
///     <c>{SCHEMA_VERSION}</c> would send a different password, or accept a URL Python rejects.</para>
///     <para><b>Scope.</b> The parser follows <c>Objects/stringlib/unicode_format.h</c> (markup iteration, field
///     names, conversions, nested format specs, the recursion limit) and the renderer follows
///     <c>Python/formatter_unicode.c</c> for the two value types a field can produce here: the <c>int</c> itself and
///     the <c>str</c> a <c>!r</c>/<c>!s</c>/<c>!a</c> conversion gives. Every error is CPython's type and message.
///     The only attributes resolved on the integer are its data attributes (<c>real</c>, <c>imag</c>,
///     <c>numerator</c>, <c>denominator</c>); naming one of its methods (<c>bit_length</c>, …) raises
///     <see cref="NotSupportedException" />, because CPython would render a bound method's repr, which contains a
///     memory address. Locale-dependent <c>n</c> formatting uses the "C" locale, which is what a Python process has
///     unless it calls <c>locale.setlocale</c>.</para>
///     <para><b>Characters are code points.</b> CPython indexes a <c>str</c> by code point, so a fill, a presentation
///     type or a conversion outside the Basic Multilingual Plane (an emoji) is one character there and two UTF-16
///     units here; the parser reads such a surrogate pair as one character and widths count code points, so
///     <c>{SCHEMA_VERSION:😀^5}</c> gives <c>😀12😀😀</c> as in Python.</para>
/// </remarks>
internal static class PyStrFormat
{
    /// <summary>PEP 3101's nesting limit: a field's format spec may contain fields, theirs may not.</summary>
    private const int RecursionDepth = 2;

    /// <summary>
    ///     The longest string a 64-bit .NET runtime allocates, in UTF-16 units (<c>0x3FFFFFDF</c>); padding past it is
    ///     refused with <see cref="OutOfMemoryException" /> before any allocation is attempted.
    /// </summary>
    private const long MaxStringLength = 0x3FFFFFDF;

    /// <summary>The methods and class-level attributes of Python's <c>int</c> type, for the AttributeError check.</summary>
    private static readonly string[] IntMethodAttributes =
    {
        "as_integer_ratio", "bit_count", "bit_length", "conjugate", "from_bytes", "is_integer", "to_bytes",
    };

    /// <summary>
    ///     Python <c>template.format(**{name: value})</c>: literal text with <c>{{</c>/<c>}}</c> unescaped and every
    ///     replacement field rendered.
    /// </summary>
    /// <param name="template">The format string.</param>
    /// <param name="name">The keyword argument's name, for example <c>SCHEMA_VERSION</c>.</param>
    /// <param name="value">The keyword argument's integer value.</param>
    /// <returns>The formatted string.</returns>
    /// <exception cref="ValueError">
    ///     A malformed template (a single brace, an unterminated field, a bad conversion or format spec, too deep a
    ///     nesting), with CPython's message.
    /// </exception>
    /// <exception cref="KeyError">A field names a keyword other than <paramref name="name" />; the message is the name's repr.</exception>
    /// <exception cref="IndexError">A field is positional (<c>{}</c>, <c>{0}</c>); there are no positional arguments.</exception>
    /// <exception cref="AttributeError">A field reads an attribute Python's <c>int</c> does not have.</exception>
    /// <exception cref="TypeError">A field indexes the integer (<c>{SCHEMA_VERSION[0]}</c>).</exception>
    /// <exception cref="NotSupportedException">A field reads one of <c>int</c>'s methods (see the remarks).</exception>
    /// <exception cref="OverflowException">A <c>c</c> format of a value outside <c>range(0x110000)</c>, or a float format of an integer too large for a double.</exception>
    /// <exception cref="OutOfMemoryException">
    ///     A width pads a field past the longest string .NET allocates (about 2<sup>30</sup> characters). CPython
    ///     builds such a string when the memory is there, so this is the one place the port stops short of it.
    /// </exception>
    internal static string FormatKeyword(string template, string name, BigInteger value)
        => BuildString(template, 0, template.Length, name, value, RecursionDepth);

    /// <summary><c>build_string</c>: checks the nesting depth, then renders literal text and fields in order.</summary>
    /// <param name="s">The text holding the format string.</param>
    /// <param name="start">The format string's first index.</param>
    /// <param name="end">The index just past the format string.</param>
    /// <param name="name">The keyword argument's name.</param>
    /// <param name="value">The keyword argument's value.</param>
    /// <param name="depth">The remaining nesting depth.</param>
    /// <returns>The rendered text.</returns>
    /// <exception cref="ValueError">The nesting limit is exceeded, or the format string is malformed.</exception>
    /// <exception cref="KeyError">A field names another keyword.</exception>
    /// <exception cref="IndexError">A field is positional.</exception>
    /// <exception cref="AttributeError">A field reads an attribute Python's <c>int</c> does not have.</exception>
    /// <exception cref="TypeError">A field indexes the integer.</exception>
    /// <exception cref="NotSupportedException">A field reads one of <c>int</c>'s methods.</exception>
    /// <exception cref="OverflowException">A <c>c</c> format out of range, or a float format of an integer too large for a double.</exception>
    /// <exception cref="OutOfMemoryException">A width beyond the longest .NET string.</exception>
    private static string BuildString(string s, int start, int end, string name, BigInteger value, int depth)
    {
        if (depth <= 0)
            throw new ValueError("Max string recursion exceeded");
        var output = new StringBuilder(end - start);
        int pos = start;
        while (pos < end)
        {
            // MarkupIterator_next: literal text up to a brace; a doubled brace is one literal brace.
            int literalStart = pos;
            char c = '\0';
            bool markupFollows = false;
            while (pos < end)
            {
                c = s[pos++];
                if (c is '{' or '}')
                {
                    markupFollows = true;
                    break;
                }
            }
            bool atEnd = pos >= end;
            int literalLength = pos - literalStart;
            if (markupFollows && c == '}' && (atEnd || s[pos] != '}'))
                throw new ValueError("Single '}' encountered in format string");
            if (markupFollows && atEnd && c == '{')
                throw new ValueError("Single '{' encountered in format string");
            if (markupFollows && !atEnd)
            {
                if (s[pos] == c)
                {
                    // An escaped brace: keep one, skip the other, no field follows.
                    pos++;
                    markupFollows = false;
                }
                else
                {
                    literalLength--;
                }
            }
            output.Append(s, literalStart, literalLength);
            if (!markupFollows)
                continue;

            ParseField(s, ref pos, end, out int nameStart, out int nameEnd, out int conversion,
                out int specStart, out int specEnd, out bool specNeedsExpanding);
            object field = GetFieldObject(s, nameStart, nameEnd, name, value);
            // CPython's "no conversion" sentinel is NUL, so a literal "!\0" converts nothing there either.
            if (conversion != 0)
                field = DoConversion(field, conversion);
            string spec = specNeedsExpanding
                ? BuildString(s, specStart, specEnd, name, value, depth - 1)
                : s.Substring(specStart, specEnd - specStart);
            output.Append(field is string text ? FormatString(text, spec) : FormatInteger((BigInteger)field, spec));
        }
        return output.ToString();
    }

    /// <summary>
    ///     <c>parse_field</c>: reads the field name up to <c>!</c>, <c>:</c> or <c>}</c> (skipping over <c>[...]</c>),
    ///     then an optional one-character conversion and an optional format spec up to the matching <c>}</c>.
    /// </summary>
    /// <param name="s">The text holding the format string.</param>
    /// <param name="pos">On entry the index after the opening brace; on return the index after the closing one.</param>
    /// <param name="end">The index just past the format string.</param>
    /// <param name="nameStart">The field name's first index.</param>
    /// <param name="nameEnd">The index just past the field name.</param>
    /// <param name="conversion">
    ///     The conversion character's code point, or 0 for none; one Python character, so an emoji after <c>!</c> is a
    ///     (bad) conversion, as in CPython, not a conversion followed by stray text.
    /// </param>
    /// <param name="specStart">The format spec's first index.</param>
    /// <param name="specEnd">The index just past the format spec (equal to <paramref name="specStart" /> when absent).</param>
    /// <param name="specNeedsExpanding">Whether the format spec contains fields of its own.</param>
    /// <exception cref="ValueError">The field is malformed; CPython's messages.</exception>
    private static void ParseField(string s, ref int pos, int end, out int nameStart, out int nameEnd,
        out int conversion, out int specStart, out int specEnd, out bool specNeedsExpanding)
    {
        conversion = 0;
        specNeedsExpanding = false;
        nameStart = pos;
        char c = '\0';
        while (pos < end)
        {
            c = s[pos++];
            if (c == '{')
                throw new ValueError("unexpected '{' in field name");
            if (c == '[')
            {
                // An index may contain any character, even '}' or ':', up to its ']'.
                while (pos < end && s[pos] != ']')
                    pos++;
                continue;
            }
            if (c is '}' or ':' or '!')
                break;
        }
        nameEnd = pos - 1;
        specStart = specEnd = pos;
        if (c is '!' or ':')
        {
            if (c == '!')
            {
                if (pos >= end)
                    throw new ValueError("end of string while looking for conversion specifier");
                conversion = CodePointAt(s, pos, end, out int units);
                pos += units;
                if (pos < end)
                {
                    char next = s[pos++];
                    if (next == '}')
                    {
                        specStart = specEnd = pos;
                        return;
                    }
                    if (next != ':')
                        throw new ValueError("expected ':' after conversion specifier");
                }
            }
            specStart = pos;
            int count = 1;
            while (pos < end)
            {
                c = s[pos++];
                if (c == '{')
                {
                    specNeedsExpanding = true;
                    count++;
                }
                else if (c == '}' && --count == 0)
                {
                    specEnd = pos - 1;
                    return;
                }
            }
            throw new ValueError("unmatched '{' in format spec");
        }
        if (c != '}')
            throw new ValueError("expected '}' before end of string");
    }

    /// <summary>
    ///     <c>get_field_object</c>: looks the first part of the field name up (a keyword, or a positional index there
    ///     is no argument for), then applies each <c>.attribute</c> and <c>[index]</c> in turn.
    /// </summary>
    /// <param name="s">The text holding the field name.</param>
    /// <param name="start">The field name's first index.</param>
    /// <param name="end">The index just past the field name.</param>
    /// <param name="name">The keyword argument's name.</param>
    /// <param name="value">The keyword argument's value.</param>
    /// <returns>The field's value: always the integer or one of its integer attributes.</returns>
    /// <exception cref="KeyError">The first part names another keyword.</exception>
    /// <exception cref="IndexError">The first part is empty or numeric.</exception>
    /// <exception cref="ValueError">The rest of the field name is malformed.</exception>
    /// <exception cref="AttributeError">An attribute the integer lacks.</exception>
    /// <exception cref="TypeError">An index on the integer.</exception>
    /// <exception cref="NotSupportedException">A method attribute of the integer.</exception>
    private static object GetFieldObject(string s, int start, int end, string name, BigInteger value)
    {
        int i = start;
        while (i < end && s[i] is not ('.' or '['))
            i++;
        string first = s.Substring(start, i - start);
        if (first.Length == 0)
            throw new IndexError("Replacement index 0 out of range for positional args tuple");
        if (TryDecimal(first, out BigInteger index))
            throw new IndexError($"Replacement index {index.ToString(CultureInfo.InvariantCulture)} out of range for positional args tuple");
        if (first != name)
            throw new KeyError(PythonRepr.StringRepr(first));

        BigInteger current = value;
        while (i < end)
        {
            char c = s[i++];
            if (c == '.')
            {
                int attrStart = i;
                while (i < end && s[i] is not ('.' or '['))
                    i++;
                string attr = s.Substring(attrStart, i - attrStart);
                if (attr.Length == 0)
                    throw new ValueError("Empty attribute in format string");
                current = attr switch
                {
                    "real" or "numerator" => current,
                    "imag" => BigInteger.Zero,
                    "denominator" => BigInteger.One,
                    _ when Array.IndexOf(IntMethodAttributes, attr) >= 0 || attr.StartsWith("__", StringComparison.Ordinal)
                        => throw new NotSupportedException(
                            $"The storage URL template reads the attribute '{attr}' of the integer {name}; CPython would " +
                            "insert the repr of a method or type object there, which OptunaSharp does not reproduce."),
                    _ => throw new AttributeError($"'int' object has no attribute {PythonRepr.StringRepr(attr)}"),
                };
            }
            else if (c == '[')
            {
                int keyStart = i;
                while (i < end && s[i] != ']')
                    i++;
                if (i >= end)
                    throw new ValueError("Missing ']' in format string");
                // CPython reads a numeric key with get_integer before the empty-key check, so an absurdly long
                // index raises its "Too many decimal digits" error first.
                string key = s.Substring(keyStart, i - keyStart);
                if (key.Length > 0)
                    TryDecimal(key, out _);
                if (i == keyStart)
                    throw new ValueError("Empty attribute in format string");
                throw new TypeError("'int' object is not subscriptable");
            }
            else
            {
                throw new ValueError("Only '.' or '[' may follow ']' in format field specifier");
            }
        }
        return current;
    }

    /// <summary>
    ///     <c>get_integer</c> on a field name part: every character a Unicode decimal digit.
    /// </summary>
    /// <param name="text">The candidate text, non-empty.</param>
    /// <param name="value">The parsed index.</param>
    /// <returns><see langword="true" /> when the text is all decimal digits.</returns>
    /// <exception cref="ValueError">The index does not fit CPython's <c>Py_ssize_t</c>.</exception>
    private static bool TryDecimal(string text, out BigInteger value)
    {
        value = BigInteger.Zero;
        for (int i = 0; i < text.Length; i++)
        {
            int digit = CharUnicodeInfo.GetDecimalDigitValue(text, i);
            if (digit < 0)
                return false;
            if (char.IsHighSurrogate(text[i]))
                i++;
            value = value * 10 + digit;
            if (value > long.MaxValue)
                throw new ValueError("Too many decimal digits in format string");
        }
        return true;
    }

    /// <summary><c>do_conversion</c>: <c>!r</c>, <c>!s</c> and <c>!a</c> all give the integer's decimal text.</summary>
    /// <param name="field">The field's integer value.</param>
    /// <param name="conversion">The conversion character's code point, not 0.</param>
    /// <returns>The converted value, a string.</returns>
    /// <exception cref="ValueError">
    ///     An unknown conversion character; CPython's message, which quotes printable ASCII as it is and anything else as
    ///     <c>\x</c> with the code point in hexadecimal (<c>\x1f600</c> for an emoji).
    /// </exception>
    private static object DoConversion(object field, int conversion)
    {
        if (conversion is 'r' or 's' or 'a')
            return ((BigInteger)field).ToString(CultureInfo.InvariantCulture);
        if (conversion is > 32 and < 127)
            throw new ValueError($"Unknown conversion specifier {(char)conversion}");
        throw new ValueError($"Unknown conversion specifier \\x{conversion.ToString("x", CultureInfo.InvariantCulture)}");
    }

    // ------------------------------------------------------------------ format specs (formatter_unicode.c)

    /// <summary>A parsed <c>InternalFormatSpec</c>.</summary>
    private sealed class Spec
    {
        /// <summary>
        ///     The fill character (default space) as UTF-16 text: one Python character, so two units when it lies outside
        ///     the Basic Multilingual Plane (an emoji fill is legal in CPython).
        /// </summary>
        public string Fill = " ";

        /// <summary>The alignment: <c>&lt;</c>, <c>&gt;</c>, <c>^</c> or <c>=</c>.</summary>
        public char Align;

        /// <summary>The <c>#</c> flag.</summary>
        public bool Alternate;

        /// <summary>The <c>z</c> flag.</summary>
        public bool NoNegativeZero;

        /// <summary>The sign option: <c>+</c>, <c>-</c>, space, or <c>'\0'</c> when absent.</summary>
        public char Sign;

        /// <summary>The minimum width, or -1.</summary>
        public long Width = -1;

        /// <summary>The grouping separator: <c>,</c>, <c>_</c>, or <c>'\0'</c>; <see cref="GroupSize" /> is 3 or 4.</summary>
        public char Separator;

        /// <summary>Digits per group: 3, or 4 for <c>_</c> with binary, octal or hexadecimal types.</summary>
        public int GroupSize = 3;

        /// <summary>The precision, or -1.</summary>
        public long Precision = -1;

        /// <summary>
        ///     The presentation type as a code point (<c>Py_UCS4</c> in CPython): the caller's default unless the spec
        ///     ends in a type character, which may be any code point, NUL included. A code point rather than a
        ///     <see cref="char" /> because an astral type character must reach the error message whole.
        /// </summary>
        public int Type;
    }

    /// <summary>
    ///     <c>parse_internal_render_format_spec</c>: <c>[[fill]align][sign][z][#][0][width][,|_][.precision][type]</c>
    ///     with CPython's validation of the separator against the type.
    /// </summary>
    /// <param name="spec">The format spec.</param>
    /// <param name="typeName">The Python type name for messages (<c>int</c> or <c>str</c>).</param>
    /// <param name="defaultType">The presentation type when none is given.</param>
    /// <param name="defaultAlign">The alignment when none is given.</param>
    /// <returns>The parsed spec.</returns>
    /// <exception cref="ValueError">The spec is malformed; CPython's messages.</exception>
    private static Spec ParseSpec(string spec, string typeName, char defaultType, char defaultAlign)
    {
        var f = new Spec { Align = defaultAlign, Type = defaultType };
        int pos = 0;
        int end = spec.Length;
        bool alignSpecified = false;
        bool fillSpecified = false;
        // CPython indexes the spec by code point, so "the second character is an alignment token" means the one after
        // the first code point: an emoji fill occupies two UTF-16 units here and one character there.
        int fillUnits = end > 0 ? CodePointUnits(spec, 0, end) : 0;
        if (end - pos > fillUnits && IsAlign(spec[pos + fillUnits]))
        {
            f.Align = spec[pos + fillUnits];
            f.Fill = spec.Substring(pos, fillUnits);
            fillSpecified = alignSpecified = true;
            pos += fillUnits + 1;
        }
        else if (end - pos >= 1 && IsAlign(spec[pos]))
        {
            f.Align = spec[pos];
            alignSpecified = true;
            pos++;
        }
        if (end - pos >= 1 && spec[pos] is ' ' or '+' or '-')
            f.Sign = spec[pos++];
        if (end - pos >= 1 && spec[pos] == 'z')
        {
            f.NoNegativeZero = true;
            pos++;
        }
        if (end - pos >= 1 && spec[pos] == '#')
        {
            f.Alternate = true;
            pos++;
        }
        if (!fillSpecified && end - pos >= 1 && spec[pos] == '0')
        {
            // Backwards-compatible zero padding: '0' alone means fill 0 with '=' alignment for numbers.
            f.Fill = "0";
            if (!alignSpecified && defaultAlign == '>')
                f.Align = '=';
            pos++;
        }
        if (ReadInteger(spec, ref pos, out long width) > 0)
            f.Width = width;
        if (end - pos > 0 && spec[pos] == ',')
        {
            f.Separator = ',';
            pos++;
        }
        if (end - pos > 0 && spec[pos] == '_')
        {
            if (f.Separator != '\0')
                throw new ValueError("Cannot specify both ',' and '_'.");
            f.Separator = '_';
            pos++;
        }
        if (end - pos > 0 && spec[pos] == ',' && f.Separator == '_')
            throw new ValueError("Cannot specify both ',' and '_'.");
        if (end - pos > 0 && spec[pos] == '.')
        {
            pos++;
            if (ReadInteger(spec, ref pos, out long precision) == 0)
                throw new ValueError("Format specifier missing precision");
            f.Precision = precision;
        }
        // "More than one character remains" counts code points too: a lone astral type character is a type, which
        // then fails as an unknown format code, not as an invalid specifier.
        if (end > pos)
        {
            int type = CodePointAt(spec, pos, end, out int typeUnits);
            if (end - pos > typeUnits)
                throw new ValueError($"Invalid format specifier '{spec}' for object of type '{typeName}'");
            f.Type = type;
        }

        if (f.Separator != '\0')
        {
            switch (f.Type)
            {
                case 'd' or 'e' or 'f' or 'g' or 'E' or 'G' or '%' or 'F' or '\0':
                    break;
                case 'b' or 'o' or 'x' or 'X' when f.Separator == '_':
                    f.GroupSize = 4;
                    break;
                default:
                    throw new ValueError($"Cannot specify '{f.Separator}' with {TypeForMessage(f.Type)}.");
            }
        }
        return f;
    }

    /// <summary>Whether a character is an alignment token.</summary>
    /// <param name="c">The character.</param>
    /// <returns><see langword="true" /> for <c>&lt;</c>, <c>&gt;</c>, <c>=</c> and <c>^</c>.</returns>
    private static bool IsAlign(char c) => c is '<' or '>' or '=' or '^';

    /// <summary>
    ///     The presentation type as CPython quotes it (<c>unknown_presentation_type</c>): <c>'x'</c> for printable
    ///     ASCII, <c>'\x..'</c> with the code point in hexadecimal for anything else, an emoji included
    ///     (<c>'\x1f600'</c>).
    /// </summary>
    /// <param name="type">The presentation type's code point.</param>
    /// <returns>The quoted text.</returns>
    private static string TypeForMessage(int type)
        => type is > 32 and < 128
            ? $"'{(char)type}'"
            : $"'\\x{type.ToString("x", CultureInfo.InvariantCulture)}'";

    /// <summary>
    ///     The number of UTF-16 units of the Python character at an index: 2 for a surrogate pair, else 1. A lone
    ///     surrogate is a character of its own, as it is in a Python <c>str</c>.
    /// </summary>
    /// <param name="s">The text.</param>
    /// <param name="index">The index of the character's first unit; below <paramref name="end" />.</param>
    /// <param name="end">The index just past the text being parsed; a pair never reaches past it.</param>
    /// <returns>1 or 2.</returns>
    private static int CodePointUnits(string s, int index, int end)
        => index + 1 < end && char.IsSurrogatePair(s[index], s[index + 1]) ? 2 : 1;

    /// <summary>The code point of the Python character at an index (see <see cref="CodePointUnits" />).</summary>
    /// <param name="s">The text.</param>
    /// <param name="index">The index of the character's first unit; below <paramref name="end" />.</param>
    /// <param name="end">The index just past the text being parsed.</param>
    /// <param name="units">The number of UTF-16 units the character occupies, 1 or 2.</param>
    /// <returns>The code point; a lone surrogate's own value.</returns>
    private static int CodePointAt(string s, int index, int end, out int units)
    {
        units = CodePointUnits(s, index, end);
        return units == 2 ? char.ConvertToUtf32(s[index], s[index + 1]) : s[index];
    }

    /// <summary>The length of a text in Python characters, which is what format widths and precisions count.</summary>
    /// <param name="text">The text.</param>
    /// <returns>The number of code points: a surrogate pair counts once.</returns>
    private static int CodePointLength(string text)
    {
        int length = 0;
        for (int i = 0; i < text.Length; i += CodePointUnits(text, i, text.Length))
            length++;
        return length;
    }

    /// <summary>
    ///     Appends the fill character <paramref name="count" /> times, refusing a result .NET cannot hold.
    /// </summary>
    /// <param name="sb">The output.</param>
    /// <param name="fill">The fill character as UTF-16 text (one or two units).</param>
    /// <param name="count">How many times to append it; zero or more.</param>
    /// <exception cref="OutOfMemoryException">
    ///     The padded text would exceed the longest string .NET allocates. CPython builds such a string when memory
    ///     allows (a width of 3,000,000,000 gives a 3 GB URL); a .NET string stops near 2<sup>30</sup> characters.
    /// </exception>
    private static void AppendFill(StringBuilder sb, string fill, long count)
    {
        if (count <= 0)
            return;
        // Checked up front: a cast to int would go negative and StringBuilder would throw ArgumentOutOfRangeException,
        // a misleading report of the "too large to build" condition CPython reports as MemoryError when it runs out.
        if (count > (MaxStringLength - sb.Length) / fill.Length)
            throw TooLong();
        sb.Insert(sb.Length, fill, (int)count);
    }

    /// <summary><c>get_integer</c> inside a format spec: consumes Unicode decimal digits.</summary>
    /// <param name="spec">The format spec.</param>
    /// <param name="pos">The index to read from, advanced past the digits.</param>
    /// <param name="value">The parsed number, or zero when no digit was read.</param>
    /// <returns>The number of digits consumed.</returns>
    /// <exception cref="ValueError">The number does not fit CPython's <c>Py_ssize_t</c>.</exception>
    private static int ReadInteger(string spec, ref int pos, out long value)
    {
        value = 0;
        int consumed = 0;
        while (pos < spec.Length)
        {
            int digit = CharUnicodeInfo.GetDecimalDigitValue(spec, pos);
            if (digit < 0)
                break;
            if (value > (long.MaxValue - digit) / 10)
                throw new ValueError("Too many decimal digits in format string");
            value = value * 10 + digit;
            pos += char.IsHighSurrogate(spec[pos]) ? 2 : 1;
            consumed++;
        }
        return consumed;
    }

    /// <summary><c>format(str, spec)</c>: <c>_PyUnicode_FormatAdvancedWriter</c> and <c>format_string_internal</c>.</summary>
    /// <param name="value">The string.</param>
    /// <param name="specText">The format spec.</param>
    /// <returns>The formatted string.</returns>
    /// <exception cref="ValueError">The spec does not apply to strings; CPython's messages.</exception>
    /// <exception cref="OutOfMemoryException">The width is beyond what a .NET string holds (see <see cref="AppendFill" />).</exception>
    private static string FormatString(string value, string specText)
    {
        if (specText.Length == 0)
            return value;
        Spec f = ParseSpec(specText, "str", 's', '<');
        if (f.Type != 's')
            throw new ValueError($"Unknown format code {TypeForMessage(f.Type)} for object of type 'str'");
        if (f.Sign != '\0')
            throw new ValueError(f.Sign == ' ' ? "Space not allowed in string format specifier" : "Sign not allowed in string format specifier");
        if (f.NoNegativeZero)
            throw new ValueError("Negative zero coercion (z) not allowed in string format specifier");
        if (f.Alternate)
            throw new ValueError("Alternate form (#) not allowed in string format specifier");
        if (f.Align == '=')
            throw new ValueError("'=' alignment not allowed in string format specifier");
        // The precision keeps that many Python characters, so a surrogate pair is kept or cut as a whole.
        if (f.Precision >= 0 && CodePointLength(value) > f.Precision)
        {
            int units = 0;
            for (long kept = 0; kept < f.Precision; kept++)
                units += CodePointUnits(value, units, value.Length);
            value = value.Substring(0, units);
        }
        return Pad(value, f.Width, f.Align, f.Fill);
    }

    /// <summary><c>calc_padding</c> + <c>fill_padding</c> for a whole value; widths count code points.</summary>
    /// <param name="text">The content.</param>
    /// <param name="width">The minimum width, or -1.</param>
    /// <param name="align">The alignment.</param>
    /// <param name="fill">The fill character as UTF-16 text (one or two units).</param>
    /// <returns>The padded content.</returns>
    /// <exception cref="OutOfMemoryException">The width is beyond what a .NET string holds (see <see cref="AppendFill" />).</exception>
    private static string Pad(string text, long width, char align, string fill)
    {
        int length = CodePointLength(text);
        long total = Math.Max(length, width);
        long left = align switch
        {
            '>' => total - length,
            '^' => (total - length) / 2,
            _ => 0,
        };
        long right = total - length - left;
        var sb = new StringBuilder(text.Length);
        AppendFill(sb, fill, left);
        sb.Append(text);
        AppendFill(sb, fill, right);
        return sb.ToString();
    }

    /// <summary><c>format(int, spec)</c>: <c>_PyLong_FormatAdvancedWriter</c>, integer and float presentation types.</summary>
    /// <param name="value">The integer.</param>
    /// <param name="specText">The format spec.</param>
    /// <returns>The formatted string.</returns>
    /// <exception cref="ValueError">The spec does not apply to integers; CPython's messages.</exception>
    /// <exception cref="OverflowException">A <c>c</c> format out of range, or a float format of an integer too large for a double.</exception>
    /// <exception cref="OutOfMemoryException">The width is beyond what a .NET string holds (see <see cref="AppendFill" />).</exception>
    private static string FormatInteger(BigInteger value, string specText)
    {
        if (specText.Length == 0)
            return value.ToString(CultureInfo.InvariantCulture);
        Spec f = ParseSpec(specText, "int", 'd', '>');
        switch (f.Type)
        {
            case 'b' or 'c' or 'd' or 'o' or 'x' or 'X' or 'n':
                return FormatLong(value, f);
            case 'e' or 'E' or 'f' or 'F' or 'g' or 'G' or '%':
                return FormatFloat(value, f);
            default:
                throw new ValueError($"Unknown format code {TypeForMessage(f.Type)} for object of type 'int'");
        }
    }

    /// <summary><c>format_long_internal</c>: integer presentation types with sign, prefix, padding and grouping.</summary>
    /// <param name="value">The integer.</param>
    /// <param name="f">The parsed spec.</param>
    /// <returns>The formatted string.</returns>
    /// <exception cref="ValueError">Precision, <c>z</c>, or a sign/<c>#</c> with <c>c</c>; CPython's messages.</exception>
    /// <exception cref="OverflowException">A <c>c</c> format of a value outside <c>range(0x110000)</c>.</exception>
    /// <exception cref="OutOfMemoryException">The width is beyond what a .NET string holds (see <see cref="AppendFill" />).</exception>
    private static string FormatLong(BigInteger value, Spec f)
    {
        if (f.Precision != -1)
            throw new ValueError("Precision not allowed in integer format specifier");
        if (f.NoNegativeZero)
            throw new ValueError("Negative zero coercion (z) not allowed in integer format specifier");
        if (f.Type == 'c')
        {
            if (f.Sign != '\0')
                throw new ValueError("Sign not allowed with integer format specifier 'c'");
            if (f.Alternate)
                throw new ValueError("Alternate form (#) not allowed with integer format specifier 'c'");
            if (value < 0 || value > 0x10ffff)
                throw new OverflowException("%c arg not in range(0x110000)");
            // 'c' content counts as "remainder": it is copied, never grouped or zero-extended.
            return LayoutNumber(sign: '\0', prefix: "", digits: "", decimalPoint: "", remainder: char.ConvertFromUtf32((int)value), f);
        }

        int radix = f.Type switch { 'b' => 2, 'o' => 8, 'x' or 'X' => 16, _ => 10 };
        string digits = ToRadix(BigInteger.Abs(value), radix);
        string prefix = !f.Alternate ? "" : f.Type switch { 'b' => "0b", 'o' => "0o", 'x' => "0x", 'X' => "0X", _ => "" };
        if (f.Type == 'X')
            digits = digits.ToUpperInvariant();
        // 'n' groups by the current locale; a Python process runs in the "C" locale, which has no grouping.
        if (f.Type == 'n')
            f.Separator = '\0';
        return LayoutNumber(value.Sign < 0 ? '-' : '\0', prefix, digits, decimalPoint: "", remainder: "", f);
    }

    /// <summary>
    ///     <c>format_float_internal</c> for an <c>int</c> given a float presentation type: the integer is converted to
    ///     a double, then rendered with <c>PyOS_double_to_string</c>'s rules.
    /// </summary>
    /// <param name="value">The integer.</param>
    /// <param name="f">The parsed spec.</param>
    /// <returns>The formatted string.</returns>
    /// <exception cref="ValueError">The precision is above <see cref="int.MaxValue" />.</exception>
    /// <exception cref="OverflowException">The integer does not fit a double (<c>int too large to convert to float</c>).</exception>
    /// <exception cref="OutOfMemoryException">The width is beyond what a .NET string holds (see <see cref="AppendFill" />).</exception>
    private static string FormatFloat(BigInteger value, Spec f)
    {
        if (f.Precision > int.MaxValue)
            throw new ValueError("precision too big");
        double d = (double)value;
        if (double.IsInfinity(d))
            throw new OverflowException("int too large to convert to float");

        int type = f.Type;
        bool percent = type == '%';
        if (percent)
        {
            type = 'f';
            d *= 100;
        }
        int precision = f.Precision < 0 ? 6 : (int)f.Precision;
        // The double holds an integer, so its exact decimal expansion is an integer: BigInteger(double) is exact.
        BigInteger exact = new BigInteger(Math.Abs(d));
        string body = type switch
        {
            'f' or 'F' => FixedDigits(exact, precision, f.Alternate),
            'e' or 'E' => ExponentDigits(exact, precision, f.Alternate, type == 'E' ? 'E' : 'e'),
            _ => GeneralDigits(exact, precision == 0 ? 1 : precision, f.Alternate, type == 'G' ? 'E' : 'e'),
        };
        if (percent)
            body += "%";
        char sign = d < 0 || (d == 0 && double.IsNegative(d) && !f.NoNegativeZero) ? '-' : '\0';

        // parse_number: integer digits, then an optional '.', then the remainder (fraction, exponent, '%').
        int intEnd = 0;
        while (intEnd < body.Length && body[intEnd] is >= '0' and <= '9')
            intEnd++;
        bool hasDecimal = intEnd < body.Length && body[intEnd] == '.';
        string remainder = body.Substring(hasDecimal ? intEnd + 1 : intEnd);
        return LayoutNumber(sign, "", body.Substring(0, intEnd), hasDecimal ? "." : "", remainder, f);
    }

    /// <summary><c>'%.*f'</c> of an exact non-negative integer.</summary>
    /// <param name="exact">The integer.</param>
    /// <param name="precision">Digits after the point.</param>
    /// <param name="alternate">Keep the point when the precision is zero.</param>
    /// <returns>The digits, with a point when needed.</returns>
    private static string FixedDigits(BigInteger exact, int precision, bool alternate)
    {
        string integer = exact.ToString(CultureInfo.InvariantCulture);
        if (precision == 0)
            return alternate ? integer + "." : integer;
        return integer + "." + new string('0', precision);
    }

    /// <summary><c>'%.*e'</c> of an exact non-negative integer, rounding half to even as <c>_Py_dg_dtoa</c> does.</summary>
    /// <param name="exact">The integer.</param>
    /// <param name="precision">Digits after the point.</param>
    /// <param name="alternate">Keep the point when the precision is zero.</param>
    /// <param name="e">The exponent character.</param>
    /// <returns>The mantissa and exponent.</returns>
    private static string ExponentDigits(BigInteger exact, int precision, bool alternate, char e)
    {
        (string digits, int exponent) = RoundSignificant(exact, precision + 1);
        string mantissa = digits.Length > 1 || alternate ? digits.Substring(0, 1) + "." + digits.Substring(1) : digits;
        return mantissa + e + (exponent < 0 ? "-" : "+") + Math.Abs(exponent).ToString("00", CultureInfo.InvariantCulture);
    }

    /// <summary>
    ///     <c>'%.*g'</c> of an exact non-negative integer: fixed notation when the decimal exponent is in
    ///     <c>[-4, precision)</c>, scientific otherwise; trailing zeros removed unless <c>#</c>.
    /// </summary>
    /// <param name="exact">The integer.</param>
    /// <param name="precision">Significant digits, at least one.</param>
    /// <param name="alternate">Keep trailing zeros and the point.</param>
    /// <param name="e">The exponent character.</param>
    /// <returns>The formatted digits.</returns>
    private static string GeneralDigits(BigInteger exact, int precision, bool alternate, char e)
    {
        (string digits, int exponent) = RoundSignificant(exact, precision);
        if (exponent >= -4 && exponent < precision)
        {
            // Fixed: exponent+1 integer digits, the rest after the point.
            string integer = digits.Substring(0, exponent + 1);
            string fraction = digits.Substring(exponent + 1);
            if (!alternate)
                fraction = fraction.TrimEnd('0');
            return fraction.Length > 0 || alternate ? integer + "." + fraction : integer;
        }
        string tail = digits.Substring(1);
        if (!alternate)
            tail = tail.TrimEnd('0');
        string mantissa = tail.Length > 0 || alternate ? digits.Substring(0, 1) + "." + tail : digits.Substring(0, 1);
        return mantissa + e + (exponent < 0 ? "-" : "+") + Math.Abs(exponent).ToString("00", CultureInfo.InvariantCulture);
    }

    /// <summary>
    ///     Rounds an exact non-negative integer to <paramref name="count" /> significant digits, half to even, and
    ///     returns them with the decimal exponent of the first digit.
    /// </summary>
    /// <param name="exact">The integer.</param>
    /// <param name="count">The number of significant digits, at least one.</param>
    /// <returns>Exactly <paramref name="count" /> digits and the exponent.</returns>
    private static (string Digits, int Exponent) RoundSignificant(BigInteger exact, int count)
    {
        if (exact.IsZero)
            return (new string('0', count), 0);
        string all = exact.ToString(CultureInfo.InvariantCulture);
        int exponent = all.Length - 1;
        if (all.Length <= count)
            return (all + new string('0', count - all.Length), exponent);
        BigInteger head = BigInteger.Parse(all.Substring(0, count), CultureInfo.InvariantCulture);
        string rest = all.Substring(count);
        int cmp = rest[0] > '5' ? 1 : rest[0] < '5' ? -1 : rest.Substring(1).TrimEnd('0').Length > 0 ? 1 : 0;
        if (cmp > 0 || (cmp == 0 && !head.IsEven))
            head += 1;
        string digits = head.ToString(CultureInfo.InvariantCulture);
        if (digits.Length > count)
        {
            // 99.5 → 1.0e+02: the carry adds a digit and moves the exponent.
            digits = digits.Substring(0, count);
            exponent++;
        }
        return (digits, exponent);
    }

    /// <summary>The digits of a non-negative integer in base 2, 8, 10 or 16 (lower case).</summary>
    /// <param name="value">The integer.</param>
    /// <param name="radix">The base.</param>
    /// <returns>The digits, at least one.</returns>
    private static string ToRadix(BigInteger value, int radix)
    {
        if (radix == 10)
            return value.ToString(CultureInfo.InvariantCulture);
        if (value.IsZero)
            return "0";
        var sb = new StringBuilder();
        while (!value.IsZero)
        {
            int digit = (int)(value % radix);
            sb.Insert(0, "0123456789abcdef"[digit]);
            value /= radix;
        }
        return sb.ToString();
    }

    /// <summary>
    ///     <c>calc_number_widths</c> + <c>fill_number</c>: lays out
    ///     <c>[lpadding][sign][prefix][spadding][grouped digits][decimal][remainder][rpadding]</c>, where zero padding
    ///     with <c>=</c> alignment is absorbed into the grouped digits (so <c>'010,'</c> gives <c>00,000,012</c>).
    /// </summary>
    /// <param name="sign">The value's sign, <c>'-'</c> or <c>'\0'</c>.</param>
    /// <param name="prefix">The base prefix, or empty.</param>
    /// <param name="digits">The integer digits to group, or empty (the <c>c</c> type).</param>
    /// <param name="decimalPoint">The decimal point, or empty.</param>
    /// <param name="remainder">Fraction, exponent and suffix characters, copied as they are.</param>
    /// <param name="f">The parsed spec.</param>
    /// <returns>The formatted number.</returns>
    /// <exception cref="OutOfMemoryException">The width is beyond what a .NET string holds (see <see cref="AppendFill" />).</exception>
    private static string LayoutNumber(char sign, string prefix, string digits, string decimalPoint, string remainder, Spec f)
    {
        char signChar = f.Sign switch
        {
            '+' => sign == '-' ? '-' : '+',
            ' ' => sign == '-' ? '-' : ' ',
            _ => sign == '-' ? '-' : '\0',
        };
        int nSign = signChar == '\0' ? 0 : 1;
        // Widths count code points: a 'c' of an astral value is one character of remainder, two UTF-16 units.
        long nonDigit = nSign + prefix.Length + decimalPoint.Length + CodePointLength(remainder);
        long minWidth = f.Fill == "0" && f.Align == '=' ? f.Width - nonDigit : 0;
        string grouped = digits.Length == 0 ? "" : Group(digits, minWidth, f.Separator, f.GroupSize);
        long padding = f.Width - (nonDigit + grouped.Length);
        long left = 0, inner = 0, right = 0;
        if (padding > 0)
        {
            switch (f.Align)
            {
                case '<':
                    right = padding;
                    break;
                case '^':
                    left = padding / 2;
                    right = padding - left;
                    break;
                case '=':
                    inner = padding;
                    break;
                default:
                    left = padding;
                    break;
            }
        }
        var sb = new StringBuilder();
        AppendFill(sb, f.Fill, left);
        if (nSign == 1)
            sb.Append(signChar);
        sb.Append(prefix);
        AppendFill(sb, f.Fill, inner);
        sb.Append(grouped).Append(decimalPoint).Append(remainder);
        AppendFill(sb, f.Fill, right);
        return sb.ToString();
    }

    /// <summary>
    ///     <c>_PyUnicode_InsertThousandsGrouping</c>: groups the digits from the right, zero-extending (with
    ///     separators) until the result is at least <paramref name="minWidth" /> wide.
    /// </summary>
    /// <param name="digits">The digits, at least one.</param>
    /// <param name="minWidth">The width the grouped digits must reach; zero or negative for none.</param>
    /// <param name="separator">The separator, or <c>'\0'</c> for no grouping.</param>
    /// <param name="groupSize">Digits per group.</param>
    /// <returns>The grouped digits.</returns>
    /// <exception cref="OutOfMemoryException">The zero-padded width is beyond what a .NET string holds (see <see cref="AppendFill" />).</exception>
    private static string Group(string digits, long minWidth, char separator, int groupSize)
    {
        minWidth = Math.Max(0, minWidth);
        // A zero-padded width CPython may build but .NET cannot: refuse it before the zero groups are made.
        if (minWidth > MaxStringLength)
            throw TooLong();
        // The groups are produced right to left, so each is appended and the list reversed once at the end; inserting
        // at the front made a zero-padded, grouped width of a few million characters quadratic (minutes, not
        // milliseconds as in CPython).
        var parts = new System.Collections.Generic.List<string>();
        long remaining = digits.Length;
        int sepLength = separator == '\0' ? 0 : 1;
        bool broken = false;
        if (separator != '\0')
        {
            while (true)
            {
                long l = Math.Min(groupSize, Math.Max(Math.Max(remaining, minWidth), 1));
                long zeros = Math.Max(0, l - remaining);
                long chars = Math.Max(0, Math.Min(remaining, l));
                parts.Add(new string('0', (int)zeros) + digits.Substring((int)(remaining - chars), (int)chars));
                remaining -= chars;
                minWidth -= l;
                if (remaining <= 0 && minWidth <= 0)
                {
                    broken = true;
                    break;
                }
                minWidth -= sepLength;
            }
        }
        if (!broken)
        {
            long l = Math.Max(Math.Max(remaining, minWidth), 1);
            long zeros = Math.Max(0, l - remaining);
            long chars = Math.Max(0, Math.Min(remaining, l));
            parts.Add(new string('0', (int)zeros) + digits.Substring(0, (int)chars));
        }
        parts.Reverse();
        return string.Join(separator == '\0' ? "" : separator.ToString(), parts);
    }

    /// <summary>The error for a field padded past the longest string .NET allocates (see <see cref="AppendFill" />).</summary>
    /// <returns>The exception to throw.</returns>
    private static OutOfMemoryException TooLong()
        => new("The storage URL template pads a field past the longest string .NET can hold " +
               $"({MaxStringLength.ToString(CultureInfo.InvariantCulture)} UTF-16 units); CPython builds such a string " +
               "when memory allows.");
}
