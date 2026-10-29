using System.Globalization;
using System.Text;
using NumSharp;

namespace OptunaSharp.Internal;

/// <summary>
///     Python's <c>UnicodeDecodeError</c>: a <see cref="ValueError" />, as in Python, so code that catches
///     <see cref="ValueError" /> catches it too.
/// </summary>
internal sealed class UnicodeDecodeError : ValueError
{
    /// <summary>Creates the error with CPython's message.</summary>
    /// <param name="message">The message, for example <c>'utf-8' codec can't decode byte 0xff in position 0: invalid start byte</c>.</param>
    internal UnicodeDecodeError(string message) : base(message)
    {
    }
}

/// <summary>
///     Python's <c>UnicodeEncodeError</c>: a <see cref="ValueError" />, as in Python, so code that catches
///     <see cref="ValueError" /> catches it too.
/// </summary>
internal sealed class UnicodeEncodeError : ValueError
{
    /// <summary>Creates the error with CPython's message.</summary>
    /// <param name="message">The message, for example <c>'latin-1' codec can't encode character '\u20ac' in position 0: ordinal not in range(256)</c>.</param>
    internal UnicodeEncodeError(string message) : base(message)
    {
    }
}

/// <summary>
///     CPython's Latin-1 encoder with <c>errors="strict"</c> (<c>str.encode("latin1")</c>), reduced to its verdict:
///     text whose every code point is below U+0100 encodes, anything else raises CPython's error.
/// </summary>
internal static class PyLatin1
{
    /// <summary>Raises what <c>text.encode("latin1")</c> raises, or returns when the text encodes.</summary>
    /// <param name="text">The text, indexed by code point as Python indexes it.</param>
    /// <exception cref="UnicodeEncodeError">
    ///     A code point above U+00FF. CPython reports the first run of consecutive unencodable characters: one is
    ///     named (<c>'€'</c>, <c>'\U0001f600'</c>), a run is given as a position range.
    /// </exception>
    internal static void CheckEncodable(string text)
    {
        int position = 0;
        for (int i = 0; i < text.Length;)
        {
            int cp = CodePoint(text, i, out int units);
            if (cp > 0xFF)
            {
                int start = position;
                int first = cp;
                int end = position + 1;
                for (i += units; i < text.Length; end++)
                {
                    int next = CodePoint(text, i, out int step);
                    if (next <= 0xFF)
                        break;
                    i += step;
                }
                string what = end - start == 1
                    ? $"character '{(first > 0xFFFF ? "\\U" + first.ToString("x8", CultureInfo.InvariantCulture) : "\\u" + first.ToString("x4", CultureInfo.InvariantCulture))}' in position {start}"
                    : $"characters in position {start}-{end - 1}";
                throw new UnicodeEncodeError($"'latin-1' codec can't encode {what}: ordinal not in range(256)");
            }
            i += units;
            position++;
        }
    }

    /// <summary>The code point at an index: a surrogate pair is one, a lone surrogate is its own value.</summary>
    /// <param name="text">The text.</param>
    /// <param name="index">The index of the character's first UTF-16 unit.</param>
    /// <param name="units">The number of units it occupies.</param>
    /// <returns>The code point.</returns>
    private static int CodePoint(string text, int index, out int units)
    {
        if (index + 1 < text.Length && char.IsSurrogatePair(text[index], text[index + 1]))
        {
            units = 2;
            return char.ConvertToUtf32(text[index], text[index + 1]);
        }
        units = 1;
        return text[index];
    }
}

/// <summary>
///     CPython's strict UTF-8 decoder (<c>bytes.decode()</c>, <c>PyUnicode_FromString</c>): the same bytes are accepted
///     and refused, and a refusal carries CPython's exact message, so a driver that decodes what libpq returns fails
///     here as it fails in Python.
/// </summary>
/// <remarks>
///     .NET's <see cref="UTF8Encoding" /> with <c>throwOnInvalidBytes</c> refuses the same byte sequences but reports
///     them differently; the classification below follows <c>Objects/stringlib/codecs.h</c> (<c>utf8_decode</c>) and
///     <c>unicode_decode_utf8</c>: an invalid lead byte is <c>invalid start byte</c>, a bad byte after a valid lead is
///     <c>invalid continuation byte</c> (with the stricter second-byte ranges after <c>E0</c>, <c>ED</c>, <c>F0</c> and
///     <c>F4</c> that exclude overlong forms, surrogates and code points above U+10FFFF), and a sequence cut off by the
///     end of the data is <c>unexpected end of data</c>.
/// </remarks>
internal static class PyUtf8
{
    /// <summary>Decodes UTF-8 bytes the way CPython does with <c>errors="strict"</c>.</summary>
    /// <param name="bytes">The bytes.</param>
    /// <returns>The text.</returns>
    /// <exception cref="UnicodeDecodeError">The bytes are not well-formed UTF-8; CPython's message.</exception>
    internal static string DecodeStrict(byte[] bytes)
    {
        int i = 0;
        while (i < bytes.Length)
        {
            int lead = bytes[i];
            if (lead < 0x80)
            {
                i++;
                continue;
            }
            int continuations = lead switch
            {
                < 0xC2 => -1,
                < 0xE0 => 1,
                < 0xF0 => 2,
                < 0xF5 => 3,
                _ => -1,
            };
            if (continuations < 0)
                throw Error(bytes, i, i + 1, "invalid start byte");
            // The second byte's range is narrower after four leads: E0 (no overlong forms), ED (no surrogates),
            // F0 (no overlong forms) and F4 (nothing above U+10FFFF).
            int low = lead == 0xE0 ? 0xA0 : lead == 0xF0 ? 0x90 : 0x80;
            int high = lead == 0xED ? 0x9F : lead == 0xF4 ? 0x8F : 0xBF;
            for (int k = 1; k <= continuations; k++)
            {
                // The bytes seen so far are a valid prefix, so running out of data is "unexpected end"; CPython
                // reports the whole tail, from the lead byte to the end.
                if (i + k >= bytes.Length)
                    throw Error(bytes, i, bytes.Length, "unexpected end of data");
                int b = bytes[i + k];
                bool valid = k == 1 ? b >= low && b <= high : b is >= 0x80 and <= 0xBF;
                if (!valid)
                    throw Error(bytes, i, i + k, "invalid continuation byte");
            }
            i += continuations + 1;
        }
        return Encoding.UTF8.GetString(bytes);
    }

    /// <summary>
    ///     CPython's message for a decoding error over <c>[start, end)</c>: one byte is named with its value, a range by
    ///     its first and last positions.
    /// </summary>
    /// <param name="bytes">The data.</param>
    /// <param name="start">The first offending byte.</param>
    /// <param name="end">One past the last offending byte.</param>
    /// <param name="reason">The reason CPython gives.</param>
    /// <returns>The exception to throw.</returns>
    private static UnicodeDecodeError Error(byte[] bytes, int start, int end, string reason)
        => new(end - start == 1
            ? $"'utf-8' codec can't decode byte 0x{bytes[start]:x2} in position {start}: {reason}"
            : $"'utf-8' codec can't decode bytes in position {start}-{end - 1}: {reason}");
}
