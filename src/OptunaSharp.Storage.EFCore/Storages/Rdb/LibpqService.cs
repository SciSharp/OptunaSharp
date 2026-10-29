using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using OptunaSharp.Collections;

namespace OptunaSharp.Storages.EFCore;

/// <summary>
///     libpq 18's connection service file (<c>pg_service.conf</c>): <c>parseServiceInfo</c> and
///     <c>parseServiceFile</c>, so <c>?service=name</c> (or <c>PGSERVICE</c>) fills in the parameters a URL leaves out
///     exactly as it does for psycopg2.
/// </summary>
/// <remarks>
///     <para><b>Where libpq looks.</b> <c>PGSERVICEFILE</c> when it is set (a missing file is then an error), else
///     <c>~/.pg_service.conf</c> — <c>%APPDATA%/postgresql/.pg_service.conf</c> on Windows — when it exists; if the
///     group is not there, <c>$PGSYSCONFDIR/pg_service.conf</c>. libpq's last resort, the <c>SYSCONFDIR</c> it was
///     compiled with, is a path baked into each libpq build (psycopg2's wheels each carry their own); OptunaSharp
///     cannot know it, so without <c>PGSYSCONFDIR</c> that file is not read. Set <c>PGSYSCONFDIR</c> to use one.</para>
///     <para><b>The format.</b> <c>[name]</c> starts a group; inside it every non-blank line that is not a
///     <c>#</c> comment is <c>keyword=value</c>, split at the first <c>=</c> with nothing trimmed around it (so
///     <c>port = 5432</c> is a syntax error, as in libpq). A keyword the connection string already sets keeps the
///     string's value. A line of 1023 bytes or more is refused, as libpq's fixed buffer refuses it.</para>
/// </remarks>
internal static class LibpqService
{
    /// <summary>libpq's line buffer: <c>fgets(buf, 1024, f)</c>, and a line that fills it is an error.</summary>
    private const int LineBuffer = 1024;

    /// <summary>
    ///     The parameters the service group supplies, for the keywords <paramref name="given" /> leaves unset; empty
    ///     when no service is named.
    /// </summary>
    /// <param name="given">The parameters the connection string sets (its <c>service</c>, if any, is the group).</param>
    /// <param name="keywords">libpq's connection keywords; a service file may set only these.</param>
    /// <returns>The group's parameters, in file order.</returns>
    /// <exception cref="FormatException">
    ///     libpq's errors: <c>service file "…" not found</c> (an explicit <c>PGSERVICEFILE</c>), <c>line N too long in
    ///     service file "…"</c>, <c>syntax error in service file "…", line N</c>, <c>nested service specifications not
    ///     supported in service file "…", line N</c>, and <c>definition of service "…" not found</c>.
    /// </exception>
    internal static OrderedDict<string, string> Options(OrderedDict<string, string> given, ICollection<string> keywords)
    {
        var found = new OrderedDict<string, string>();
        // PGSERVICE stands in for a missing service parameter here, before the other PG* variables apply.
        string? service = given.TryGetValue("service", out string? named) ? named : Environment.GetEnvironmentVariable("PGSERVICE");
        if (service is null)
            return found;

        bool groupFound = false;
        string? file = Environment.GetEnvironmentVariable("PGSERVICEFILE");
        if (file is null)
        {
            // pqGetHomeDirectory: $HOME (else the account's home) on POSIX, %APPDATA%/postgresql on Windows; libpq
            // joins with '/', which shows in its messages.
            string? home = HomeDirectory();
            file = home is null ? null : home + "/.pg_service.conf";
            if (file is not null && !File.Exists(file))
                file = null;
        }
        if (file is not null)
        {
            // An explicit PGSERVICEFILE that cannot be opened is an error; the default file was checked to exist.
            ParseFile(file, service, given, keywords, found, ref groupFound);
            if (groupFound)
                return found;
        }

        string? sysconfdir = Environment.GetEnvironmentVariable("PGSYSCONFDIR");
        if (sysconfdir is not null)
        {
            string system = sysconfdir + "/pg_service.conf";
            if (File.Exists(system))
                ParseFile(system, service, given, keywords, found, ref groupFound);
        }
        if (!groupFound)
            throw new FormatException($"definition of service \"{service}\" not found");
        return found;
    }

    /// <summary>libpq's <c>pqGetHomeDirectory</c>.</summary>
    /// <returns>The directory, or <see langword="null" /> when there is none.</returns>
    private static string? HomeDirectory()
    {
        if (OperatingSystem.IsWindows())
        {
            string appData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            return appData.Length == 0 ? null : appData + "/postgresql";
        }
        string? home = Environment.GetEnvironmentVariable("HOME");
        if (!string.IsNullOrEmpty(home))
            return home;
        string profile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return profile.Length == 0 ? null : profile;
    }

    /// <summary>
    ///     libpq's <c>parseServiceFile</c>: reads the file line by line and takes the <c>keyword=value</c> lines of
    ///     the group named <paramref name="service" />, stopping at the next group.
    /// </summary>
    /// <param name="path">The service file, as libpq names it in messages.</param>
    /// <param name="service">The group name.</param>
    /// <param name="given">The connection string's parameters, which a service line never overrides.</param>
    /// <param name="keywords">libpq's connection keywords.</param>
    /// <param name="found">Where the group's parameters are added.</param>
    /// <param name="groupFound">Set when the group appears in the file.</param>
    /// <exception cref="FormatException">The file cannot be opened, a line is too long, or a group line is malformed; libpq's messages.</exception>
    private static void ParseFile(string path, string service, OrderedDict<string, string> given, ICollection<string> keywords,
        OrderedDict<string, string> found, ref bool groupFound)
    {
        groupFound = false;
        byte[] content;
        try
        {
            content = File.ReadAllBytes(path);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            throw new FormatException($"service file \"{path}\" not found");
        }

        int lineNumber = 0;
        int position = 0;
        while (position < content.Length)
        {
            // fgets: a line runs to its '\n' inclusive. Windows opens the file in text mode, where "\r\n" reads as "\n".
            int newline = Array.IndexOf(content, (byte)'\n', position);
            int end = newline < 0 ? content.Length : newline + 1;
            int length = end - position;
            if (OperatingSystem.IsWindows() && newline > position && content[newline - 1] == '\r')
                length--;
            lineNumber++;
            if (length >= LineBuffer - 1)
                throw new FormatException($"line {lineNumber} too long in service file \"{path}\"");
            string line = Trim(content, position, end);
            position = end;

            if (line.Length == 0 || line[0] == '#')
                continue;
            if (line[0] == '[')
            {
                // The next group ends the one being read; a group line matches on its name and the ']' after it,
                // whatever follows.
                if (groupFound)
                    return;
                groupFound = line.Length > service.Length + 1 && string.CompareOrdinal(line, 1, service, 0, service.Length) == 0
                             && line[service.Length + 1] == ']';
                continue;
            }
            if (!groupFound)
                continue;

            int equals = line.IndexOf('=');
            if (equals < 0)
                throw new FormatException($"syntax error in service file \"{path}\", line {lineNumber}");
            string key = line.Substring(0, equals);
            string value = line.Substring(equals + 1);
            if (key == "service")
                throw new FormatException($"nested service specifications not supported in service file \"{path}\", line {lineNumber}");
            if (!keywords.Contains(key))
                throw new FormatException($"syntax error in service file \"{path}\", line {lineNumber}");
            // "Set the parameter --- but don't override any previous explicit setting."
            if (!given.ContainsKey(key) && !found.ContainsKey(key))
                found[key] = value;
        }
    }

    /// <summary>A service-file line without the surrounding C <c>isspace</c> characters (the newline included).</summary>
    /// <param name="content">The file's bytes.</param>
    /// <param name="start">The line's first byte.</param>
    /// <param name="end">One past its last byte.</param>
    /// <returns>The trimmed line as text; libpq hands the bytes on as they are, so they are read as UTF-8.</returns>
    private static string Trim(byte[] content, int start, int end)
    {
        static bool IsSpace(byte b) => b is (byte)' ' or (byte)'\t' or (byte)'\n' or (byte)'\v' or (byte)'\f' or (byte)'\r';
        while (end > start && IsSpace(content[end - 1]))
            end--;
        while (start < end && IsSpace(content[start]))
            start++;
        return Encoding.UTF8.GetString(content, start, end - start);
    }
}
