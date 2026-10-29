using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using NumSharp;
using OptunaSharp.Storages.SqlAlchemy;

namespace OptunaSharp.Internal;

/// <summary>
///     The parts of CPython 3.12's <c>urllib.parse</c> that storage URLs go through: SQLAlchemy decodes a URL's
///     user name, password and query with <c>unquote</c> and <c>parse_qsl</c>, and redis-py splits a Redis URL with
///     <c>urlparse</c> and reads its query with <c>parse_qs</c>. Each member follows the Python function line by
///     line, so a URL decodes to the same strings in both languages.
/// </summary>
/// <remarks>
///     <para>Why not <see cref="Uri" /> or <see cref="Uri.UnescapeDataString(string)" />: <see cref="Uri" />
///     normalizes (lower-cases the scheme, resolves dot segments, rejects or re-encodes characters) and
///     <c>UnescapeDataString</c> leaves an invalid UTF-8 escape such as <c>%FF</c> as it is, where Python writes
///     U+FFFD. Neither decodes <c>+</c> as a space, which <c>parse_qsl</c> does. Each of those differences changes
///     a password, a database name or a query value.</para>
/// </remarks>
internal static class PyUrllib
{
    /// <summary>Characters <c>quote</c> never escapes (<c>_ALWAYS_SAFE</c>).</summary>
    private const string AlwaysSafe = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-~";

    /// <summary>Characters allowed in a URL scheme (<c>scheme_chars</c>).</summary>
    private const string SchemeChars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.";

    /// <summary>The decoder <c>bytes.decode('utf-8', 'replace')</c> corresponds to: invalid bytes become U+FFFD.</summary>
    private static readonly UTF8Encoding Utf8Replace = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: false);

    /// <summary>
    ///     <c>urllib.parse.unquote(text)</c> (UTF-8, <c>errors='replace'</c>): every <c>%XX</c> escape becomes its
    ///     byte, each run of ASCII characters is decoded from UTF-8 with invalid sequences replaced by U+FFFD, and
    ///     non-ASCII characters pass through.
    /// </summary>
    /// <param name="text">The percent-encoded text.</param>
    /// <returns>The decoded text.</returns>
    /// <remarks>
    ///     Decoding run by run is Python's behavior, not an optimization: an escape sequence split by a non-ASCII
    ///     character (<c>%C3é%A9</c>) decodes as two invalid halves, exactly as in CPython.
    /// </remarks>
    internal static string Unquote(string text)
    {
        if (text.IndexOf('%') < 0)
            return text;
        var sb = new StringBuilder(text.Length);
        int i = 0;
        while (i < text.Length)
        {
            int runStart = i;
            while (i < text.Length && text[i] < 0x80)
                i++;
            if (i > runStart)
                sb.Append(Utf8Replace.GetString(UnquoteToBytes(text, runStart, i)));
            int otherStart = i;
            while (i < text.Length && text[i] >= 0x80)
                i++;
            sb.Append(text, otherStart, i - otherStart);
        }
        return sb.ToString();
    }

    /// <summary><c>urllib.parse.unquote_plus(text)</c>: every <c>+</c> becomes a space, then <see cref="Unquote" />.</summary>
    /// <param name="text">The form-encoded text.</param>
    /// <returns>The decoded text.</returns>
    internal static string UnquotePlus(string text) => Unquote(text.Replace('+', ' '));

    /// <summary>
    ///     <c>urllib.parse.parse_qsl(query)</c> with Python's defaults: pairs split on <c>&amp;</c>, empty pairs
    ///     skipped, a pair without <c>=</c> or with an empty value dropped (<c>keep_blank_values=False</c>), name
    ///     and value decoded with <see cref="UnquotePlus" />.
    /// </summary>
    /// <param name="query">The query string, without the leading <c>?</c>.</param>
    /// <returns>The name/value pairs in their order of appearance; names may repeat.</returns>
    internal static List<KeyValuePair<string, string>> ParseQsl(string query)
    {
        var pairs = new List<KeyValuePair<string, string>>();
        if (query.Length == 0)
            return pairs;
        foreach (string nameValue in query.Split('&'))
        {
            if (nameValue.Length == 0)
                continue;
            int eq = nameValue.IndexOf('=');
            string value = eq < 0 ? "" : nameValue.Substring(eq + 1);
            // keep_blank_values=False: "a" and "a=" carry no value and are left out.
            if (value.Length == 0)
                continue;
            pairs.Add(new KeyValuePair<string, string>(UnquotePlus(nameValue.Substring(0, eq)), UnquotePlus(value)));
        }
        return pairs;
    }

    /// <summary>
    ///     <c>urllib.parse.parse_qs(query)</c>: <see cref="ParseQsl" /> grouped by name, names in the order of their
    ///     first appearance, each with every value in order.
    /// </summary>
    /// <param name="query">The query string, without the leading <c>?</c>.</param>
    /// <returns>The names with their values; no name has an empty list.</returns>
    internal static List<KeyValuePair<string, List<string>>> ParseQs(string query)
    {
        var grouped = new List<KeyValuePair<string, List<string>>>();
        var index = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, string> pair in ParseQsl(query))
        {
            if (index.TryGetValue(pair.Key, out int at))
            {
                grouped[at].Value.Add(pair.Value);
                continue;
            }
            index[pair.Key] = grouped.Count;
            grouped.Add(new KeyValuePair<string, List<string>>(pair.Key, new List<string> { pair.Value }));
        }
        return grouped;
    }

    /// <summary>
    ///     <c>urllib.parse.quote(text, safe)</c>: the UTF-8 bytes of the text, with every byte outside
    ///     <c>_ALWAYS_SAFE</c> and <paramref name="safe" /> written as an upper-case <c>%XX</c> escape.
    /// </summary>
    /// <param name="text">The text to escape.</param>
    /// <param name="safe">Additional ASCII characters to keep as they are.</param>
    /// <returns>The escaped text.</returns>
    internal static string Quote(string text, string safe)
    {
        var sb = new StringBuilder(text.Length);
        foreach (byte b in Encoding.UTF8.GetBytes(text))
        {
            char c = (char)b;
            if (b < 0x80 && (AlwaysSafe.IndexOf(c) >= 0 || safe.IndexOf(c) >= 0))
                sb.Append(c);
            else
                sb.Append('%').Append(b.ToString("X2", CultureInfo.InvariantCulture));
        }
        return sb.ToString();
    }

    /// <summary>
    ///     <c>urllib.parse.quote_plus(text)</c> with <c>safe=''</c>: like <see cref="Quote" />, with spaces written as
    ///     <c>+</c> (and a literal <c>+</c> escaped as <c>%2B</c>).
    /// </summary>
    /// <param name="text">The text to escape.</param>
    /// <returns>The escaped text.</returns>
    internal static string QuotePlus(string text)
        => text.IndexOf(' ') < 0 ? Quote(text, "") : Quote(text, " ").Replace(' ', '+');

    /// <summary>
    ///     <c>urllib.parse.urlsplit(url)</c> (Python 3.12): leading C0 controls and spaces stripped, tabs and line
    ///     breaks removed anywhere, the scheme lower-cased, the network location split at the first <c>/</c>,
    ///     <c>?</c> or <c>#</c>, then fragment and query split off, with Python's bracket and NFKC checks.
    /// </summary>
    /// <param name="url">The URL to split.</param>
    /// <returns>The five components, with Python's <c>username</c>/<c>password</c>/<c>hostname</c>/<c>port</c> views.</returns>
    /// <exception cref="ValueError">
    ///     The network location has unbalanced brackets or an invalid bracketed host, or contains characters that
    ///     NFKC normalization turns into URL delimiters. Messages are Python's.
    /// </exception>
    /// <remarks>
    ///     <c>urlparse</c> differs from <c>urlsplit</c> only by splitting <c>;params</c> off the path, and only for
    ///     the schemes in <c>uses_params</c> (http, ftp, sip, …). Callers that port <c>urlparse</c> for other
    ///     schemes (redis-py's <c>redis</c>, <c>rediss</c> and <c>unix</c>) can use this directly.
    /// </remarks>
    internal static PySplitResult UrlSplit(string url)
    {
        // Only the start is stripped: CPython keeps a trailing space because some applications rely on it.
        url = url.TrimStart(WhatwgC0ControlOrSpace);
        url = url.Replace("\t", "", StringComparison.Ordinal).Replace("\r", "", StringComparison.Ordinal)
            .Replace("\n", "", StringComparison.Ordinal);

        string scheme = "";
        string netloc = "";
        string query = "";
        string fragment = "";
        int colon = url.IndexOf(':');
        if (colon > 0 && url[0] < 0x80 && char.IsAsciiLetter(url[0]))
        {
            bool allSchemeChars = true;
            for (int k = 0; k < colon; k++)
            {
                if (SchemeChars.IndexOf(url[k]) < 0)
                {
                    allSchemeChars = false;
                    break;
                }
            }
            if (allSchemeChars)
            {
                scheme = url.Substring(0, colon).ToLowerInvariant();
                url = url.Substring(colon + 1);
            }
        }
        if (url.StartsWith("//", StringComparison.Ordinal))
        {
            int delim = url.Length;
            foreach (char d in "/?#")
            {
                int w = url.IndexOf(d, 2);
                if (w >= 0)
                    delim = Math.Min(delim, w);
            }
            netloc = url.Substring(2, delim - 2);
            url = url.Substring(delim);
            bool open = netloc.Contains('['), close = netloc.Contains(']');
            if (open != close)
                throw new ValueError("Invalid IPv6 URL");
            if (open)
                CheckBracketedNetloc(netloc);
        }
        int hash = url.IndexOf('#');
        if (hash >= 0)
        {
            fragment = url.Substring(hash + 1);
            url = url.Substring(0, hash);
        }
        int question = url.IndexOf('?');
        if (question >= 0)
        {
            query = url.Substring(question + 1);
            url = url.Substring(0, question);
        }
        CheckNetloc(netloc);
        return new PySplitResult(scheme, netloc, url, query, fragment);
    }

    /// <summary>The characters <c>urlsplit</c> strips from the start of a URL (<c>_WHATWG_C0_CONTROL_OR_SPACE</c>).</summary>
    private static readonly char[] WhatwgC0ControlOrSpace = BuildC0ControlOrSpace();

    /// <summary>Builds U+0000 through U+0020.</summary>
    /// <returns>The 33 characters.</returns>
    private static char[] BuildC0ControlOrSpace()
    {
        var chars = new char[0x21];
        for (int i = 0; i <= 0x20; i++)
            chars[i] = (char)i;
        return chars;
    }

    /// <summary>
    ///     <c>_unquote_impl</c> over one ASCII run: <c>%</c> followed by two hexadecimal digits becomes that byte,
    ///     any other <c>%</c> stays literal.
    /// </summary>
    /// <param name="text">The text holding the run.</param>
    /// <param name="start">The run's first index.</param>
    /// <param name="end">The index just past the run.</param>
    /// <returns>The run's bytes after unescaping.</returns>
    private static byte[] UnquoteToBytes(string text, int start, int end)
    {
        var bytes = new List<byte>(end - start);
        for (int i = start; i < end; i++)
        {
            char c = text[i];
            // Both digits must lie inside the run: a '%' near the run's end stays literal, as Python's
            // `_hextobyte[item[:2]]` lookup fails on a shorter item.
            if (c == '%' && i + 2 < end && IsHex(text[i + 1]) && IsHex(text[i + 2]))
            {
                bytes.Add((byte)((HexValue(text[i + 1]) << 4) | HexValue(text[i + 2])));
                i += 2;
                continue;
            }
            bytes.Add((byte)c);
        }
        return bytes.ToArray();
    }

    /// <summary>Whether a character is one of <c>_hexdig</c> (ASCII hexadecimal, either case).</summary>
    /// <param name="c">The character to test.</param>
    /// <returns><see langword="true" /> for 0-9, a-f and A-F.</returns>
    private static bool IsHex(char c) => c is >= '0' and <= '9' or >= 'a' and <= 'f' or >= 'A' and <= 'F';

    /// <summary>The value of an ASCII hexadecimal digit.</summary>
    /// <param name="c">A character for which <see cref="IsHex" /> holds.</param>
    /// <returns>0 through 15.</returns>
    private static int HexValue(char c) => c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10;

    /// <summary>
    ///     <c>_check_bracketed_netloc</c>: nothing may precede the <c>[</c> after the user information, nothing but
    ///     a <c>:port</c> may follow the <c>]</c>, and the bracketed host must be valid.
    /// </summary>
    /// <param name="netloc">A network location containing both brackets.</param>
    /// <exception cref="ValueError">The bracketed part is malformed; Python's messages.</exception>
    private static void CheckBracketedNetloc(string netloc)
    {
        string hostAndPort = netloc.Substring(netloc.LastIndexOf('@') + 1);
        int open = hostAndPort.IndexOf('[');
        string hostname;
        if (open >= 0)
        {
            if (open > 0)
                throw new ValueError("Invalid IPv6 URL");
            string bracketed = hostAndPort.Substring(open + 1);
            int close = bracketed.IndexOf(']');
            hostname = close < 0 ? bracketed : bracketed.Substring(0, close);
            string port = close < 0 ? "" : bracketed.Substring(close + 1);
            if (port.Length > 0 && !port.StartsWith(':'))
                throw new ValueError("Invalid IPv6 URL");
        }
        else
        {
            int portColon = hostAndPort.IndexOf(':');
            hostname = portColon < 0 ? hostAndPort : hostAndPort.Substring(0, portColon);
        }
        CheckBracketedHost(hostname);
    }

    /// <summary>
    ///     <c>_check_bracketed_host</c>: an IPvFuture literal (<c>v1.x</c>) must match its grammar; anything else
    ///     must be an IPv6 address per <c>ipaddress.ip_address</c>, and an IPv4 address is refused.
    /// </summary>
    /// <param name="hostname">The text between the brackets.</param>
    /// <exception cref="ValueError">The host is not a valid bracketed host; Python's messages.</exception>
    private static void CheckBracketedHost(string hostname)
    {
        if (hostname.StartsWith('v'))
        {
            if (!Regex.IsMatch(hostname, @"\Av[a-fA-F0-9]+\.[^\n]+\z", RegexOptions.CultureInvariant))
                throw new ValueError("IPvFuture address is invalid");
            return;
        }
        if (PyIpAddress.IsIPv4(hostname))
            throw new ValueError("An IPv4 address cannot be in brackets");
        if (!PyIpAddress.IsIPv6(hostname))
            throw new ValueError($"{PythonRepr.StringRepr(hostname)} does not appear to be an IPv4 or IPv6 address");
    }

    /// <summary>
    ///     <c>_checknetloc</c>: a non-ASCII network location must not gain a <c>/</c>, <c>?</c>, <c>#</c>, <c>@</c>
    ///     or <c>:</c> under NFKC normalization (characters like U+2100 expand to <c>a/c</c>).
    /// </summary>
    /// <param name="netloc">The network location.</param>
    /// <exception cref="ValueError">Normalization introduces a delimiter; Python's message.</exception>
    private static void CheckNetloc(string netloc)
    {
        bool ascii = true;
        foreach (char c in netloc)
        {
            if (c >= 0x80)
            {
                ascii = false;
                break;
            }
        }
        if (netloc.Length == 0 || ascii)
            return;
        string n = netloc.Replace("@", "", StringComparison.Ordinal).Replace(":", "", StringComparison.Ordinal)
            .Replace("#", "", StringComparison.Ordinal).Replace("?", "", StringComparison.Ordinal);
        string normalized = n.Normalize(NormalizationForm.FormKC);
        if (n == normalized)
            return;
        foreach (char c in "/?#@:")
        {
            if (normalized.Contains(c))
                throw new ValueError($"netloc '{netloc}' contains invalid characters under NFKC normalization");
        }
    }
}

/// <summary>
///     <c>urllib.parse.SplitResult</c>: the five components of a split URL, plus the derived properties Python
///     computes from the network location on access.
/// </summary>
/// <param name="Scheme">The lower-cased scheme, or empty.</param>
/// <param name="Netloc">The network location (user information, host and port), or empty.</param>
/// <param name="Path">The path, possibly empty.</param>
/// <param name="Query">The query without its <c>?</c>, or empty.</param>
/// <param name="Fragment">The fragment without its <c>#</c>, or empty.</param>
internal sealed record PySplitResult(string Scheme, string Netloc, string Path, string Query, string Fragment)
{
    /// <summary>
    ///     <c>.username</c>: the part of the user information before its first <c>:</c>, still percent-encoded;
    ///     <see langword="null" /> when the network location has no <c>@</c>. The split is at the last <c>@</c>, so
    ///     <c>u:p@s@h</c> has user <c>u</c> and password <c>p@s</c>.
    /// </summary>
    internal string? Username
    {
        get
        {
            int at = Netloc.LastIndexOf('@');
            if (at < 0)
                return null;
            string userInfo = Netloc.Substring(0, at);
            int colon = userInfo.IndexOf(':');
            return colon < 0 ? userInfo : userInfo.Substring(0, colon);
        }
    }

    /// <summary>
    ///     <c>.password</c>: the part of the user information after its first <c>:</c>, still percent-encoded;
    ///     <see langword="null" /> without an <c>@</c> or without a <c>:</c> before it.
    /// </summary>
    internal string? Password
    {
        get
        {
            int at = Netloc.LastIndexOf('@');
            if (at < 0)
                return null;
            string userInfo = Netloc.Substring(0, at);
            int colon = userInfo.IndexOf(':');
            return colon < 0 ? null : userInfo.Substring(colon + 1);
        }
    }

    /// <summary>
    ///     <c>.hostname</c>: the host after the user information, without brackets, lower-cased except for an IPv6
    ///     zone after <c>%</c>; <see langword="null" /> when empty.
    /// </summary>
    /// <remarks>
    ///     Lower-cased with Python's <c>str.lower</c> (<see cref="PyStrings.Lower" />), not .NET's invariant
    ///     lower case: the two differ on U+0130 (two characters in Python) and on a word-final <c>Σ</c> (<c>ς</c> in
    ///     Python), so an internationalized host name such as <c>ΑΣ</c> resolves to the name Python looks up.
    /// </remarks>
    internal string? Hostname
    {
        get
        {
            string host = HostInfo().Host;
            if (host.Length == 0)
                return null;
            int percent = host.IndexOf('%');
            return percent < 0
                ? PyStrings.Lower(host)
                : PyStrings.Lower(host.Substring(0, percent)) + host.Substring(percent);
        }
    }

    /// <summary>
    ///     <c>.port</c>: the port after the host as an integer, <see langword="null" /> when absent or empty.
    /// </summary>
    /// <exception cref="ValueError">
    ///     The port is not ASCII digits (<c>Port could not be cast to integer value as '…'</c>) or is above 65535
    ///     (<c>Port out of range 0-65535</c>); Python raises on access, so this property does too.
    /// </exception>
    internal int? Port
    {
        get
        {
            string? port = HostInfo().Port;
            if (port is null)
                return null;
            foreach (char c in port)
            {
                if (c is < '0' or > '9')
                    throw new ValueError($"Port could not be cast to integer value as {PythonRepr.StringRepr(port)}");
            }
            // str.isdigit() accepts any length (leading zeros included), so the value is range-checked unbounded.
            System.Numerics.BigInteger value = System.Numerics.BigInteger.Parse(port, NumberStyles.None, CultureInfo.InvariantCulture);
            if (value > 65535)
                throw new ValueError("Port out of range 0-65535");
            return (int)value;
        }
    }

    /// <summary><c>_hostinfo</c>: host and port text after the last <c>@</c>, brackets removed from the host.</summary>
    /// <returns>The host (possibly empty) and the port (<see langword="null" /> when absent or empty).</returns>
    private (string Host, string? Port) HostInfo()
    {
        string hostInfo = Netloc.Substring(Netloc.LastIndexOf('@') + 1);
        string host;
        string port;
        int open = hostInfo.IndexOf('[');
        if (open >= 0)
        {
            string bracketed = hostInfo.Substring(open + 1);
            int close = bracketed.IndexOf(']');
            host = close < 0 ? bracketed : bracketed.Substring(0, close);
            string after = close < 0 ? "" : bracketed.Substring(close + 1);
            int colon = after.IndexOf(':');
            port = colon < 0 ? "" : after.Substring(colon + 1);
        }
        else
        {
            int colon = hostInfo.IndexOf(':');
            host = colon < 0 ? hostInfo : hostInfo.Substring(0, colon);
            port = colon < 0 ? "" : hostInfo.Substring(colon + 1);
        }
        return (host, port.Length == 0 ? null : port);
    }
}

/// <summary>
///     The address checks of CPython's <c>ipaddress</c> module that <c>urlsplit</c> relies on: strict dotted-quad
///     IPv4 (no leading zeros, as in <c>inet_pton</c>) and IPv6 with an optional <c>%scope</c>.
/// </summary>
internal static class PyIpAddress
{
    /// <summary><c>IPv4Address(text)</c> succeeds: four decimal octets of one to three ASCII digits, each at most 255, no leading zero.</summary>
    /// <param name="text">The candidate address.</param>
    /// <returns><see langword="true" /> when Python accepts the address.</returns>
    internal static bool IsIPv4(string text)
    {
        if (text.Length == 0 || text.Contains('/'))
            return false;
        string[] octets = text.Split('.');
        if (octets.Length != 4)
            return false;
        foreach (string octet in octets)
        {
            if (octet.Length is 0 or > 3)
                return false;
            foreach (char c in octet)
            {
                if (c is < '0' or > '9')
                    return false;
            }
            if (octet != "0" && octet[0] == '0')
                return false;
            if (int.Parse(octet, NumberStyles.None, CultureInfo.InvariantCulture) > 255)
                return false;
        }
        return true;
    }

    /// <summary>
    ///     <c>IPv6Address(text)</c> succeeds: eight hextets of one to four hexadecimal digits, one <c>::</c> standing
    ///     for at least one zero hextet, an optional embedded IPv4 tail, and an optional non-empty <c>%scope</c>.
    /// </summary>
    /// <param name="text">The candidate address.</param>
    /// <returns><see langword="true" /> when Python accepts the address.</returns>
    internal static bool IsIPv6(string text)
    {
        if (text.Contains('/'))
            return false;
        int percent = text.IndexOf('%');
        if (percent >= 0)
        {
            string scope = text.Substring(percent + 1);
            if (scope.Length == 0 || scope.Contains('%'))
                return false;
            text = text.Substring(0, percent);
        }
        if (text.Length == 0)
            return false;
        var parts = new List<string>(text.Split(':'));
        if (parts.Count < 3)
            return false;
        if (parts[^1].Contains('.'))
        {
            if (!IsIPv4(parts[^1]))
                return false;
            parts.RemoveAt(parts.Count - 1);
            parts.Add("0");
            parts.Add("0");
        }
        if (parts.Count > 9)
            return false;
        int skip = -1;
        for (int i = 1; i < parts.Count - 1; i++)
        {
            if (parts[i].Length == 0)
            {
                if (skip >= 0)
                    return false;
                skip = i;
            }
        }
        int hi, lo;
        if (skip >= 0)
        {
            hi = skip;
            lo = parts.Count - skip - 1;
            if (parts[0].Length == 0 && --hi != 0)
                return false;
            if (parts[^1].Length == 0 && --lo != 0)
                return false;
            if (8 - (hi + lo) < 1)
                return false;
        }
        else
        {
            if (parts.Count != 8 || parts[0].Length == 0 || parts[^1].Length == 0)
                return false;
            hi = parts.Count;
            lo = 0;
        }
        for (int i = 0; i < hi; i++)
        {
            if (!IsHextet(parts[i]))
                return false;
        }
        for (int i = parts.Count - lo; i < parts.Count; i++)
        {
            if (!IsHextet(parts[i]))
                return false;
        }
        return true;
    }

    /// <summary><c>_parse_hextet</c>: one to four ASCII hexadecimal digits.</summary>
    /// <param name="hextet">The candidate group.</param>
    /// <returns><see langword="true" /> for a valid group.</returns>
    private static bool IsHextet(string hextet)
    {
        if (hextet.Length is 0 or > 4)
            return false;
        foreach (char c in hextet)
        {
            if (!(c is >= '0' and <= '9' or >= 'a' and <= 'f' or >= 'A' and <= 'F'))
                return false;
        }
        return true;
    }
}
