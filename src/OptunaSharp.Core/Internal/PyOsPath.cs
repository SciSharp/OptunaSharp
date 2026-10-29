using System;
using System.Collections.Generic;
using System.IO;
using OptunaSharp.Storages.SqlAlchemy;

namespace OptunaSharp.Internal;

/// <summary>
///     CPython 3.12's <c>os.path.abspath</c> and <c>os.path.normpath</c>, for the running operating system
///     (<c>ntpath</c> on Windows, <c>posixpath</c> elsewhere). SQLAlchemy's SQLite dialect passes every file
///     database through <c>os.path.abspath</c> before <c>sqlite3.connect</c>, so the file a SQLite URL opens is
///     the one these functions compute.
/// </summary>
/// <remarks>
///     <para>Why not <see cref="Path.GetFullPath(string)" /> alone: Python normalizes first (collapsing
///     <c>..</c> even inside <c>\\?\</c> paths, dropping a trailing separator) and falls back to joining with the
///     working directory where the OS call fails (a name of only spaces). <see cref="Path.GetFullPath(string)" />
///     is the Windows step after normalization — it calls the same <c>GetFullPathNameW</c> CPython's
///     <c>_getfullpathname</c> does, which strips trailing dots and spaces.</para>
/// </remarks>
internal static class PyOsPath
{
    /// <summary><c>os.path.abspath(path)</c>.</summary>
    /// <param name="path">A relative or absolute path.</param>
    /// <returns>The absolute, normalized path Python computes.</returns>
    internal static string AbsPath(string path)
    {
        if (!OperatingSystem.IsWindows())
            return PosixNormPath(path.StartsWith('/') ? path : PosixJoin(Directory.GetCurrentDirectory(), path));
        try
        {
            return Path.GetFullPath(NtNormPath(path));
        }
        catch (Exception e) when (e is ArgumentException or IOException or NotSupportedException)
        {
            // _abspath_fallback: join with the working directory unless absolute, then normalize.
            return NtNormPath(NtIsAbs(path) ? path : NtJoin(Directory.GetCurrentDirectory(), path));
        }
    }

    /// <summary><c>ntpath.normpath</c>: <c>/</c> becomes <c>\</c>, empty and <c>.</c> components vanish, <c>..</c> eats its parent (never the root).</summary>
    /// <param name="path">The path.</param>
    /// <returns>The normalized path; <c>.</c> when nothing is left.</returns>
    internal static string NtNormPath(string path)
    {
        path = path.Replace('/', '\\');
        (string drive, string root, string rest) = NtSplitRoot(path);
        string prefix = drive + root;
        var comps = new List<string>(rest.Split('\\'));
        int i = 0;
        while (i < comps.Count)
        {
            if (comps[i].Length == 0 || comps[i] == ".")
            {
                comps.RemoveAt(i);
            }
            else if (comps[i] == "..")
            {
                if (i > 0 && comps[i - 1] != "..")
                {
                    comps.RemoveRange(i - 1, 2);
                    i--;
                }
                else if (i == 0 && root.Length > 0)
                {
                    comps.RemoveAt(i);
                }
                else
                {
                    i++;
                }
            }
            else
            {
                i++;
            }
        }
        if (prefix.Length == 0 && comps.Count == 0)
            comps.Add(".");
        return prefix + string.Join("\\", comps);
    }

    /// <summary><c>ntpath.splitroot</c>: drive (letter, UNC share or device), root separator, and the rest.</summary>
    /// <param name="p">The path, with forward slashes already turned into backslashes.</param>
    /// <returns>The three parts; concatenated they give the input.</returns>
    private static (string Drive, string Root, string Tail) NtSplitRoot(string p)
    {
        if (p.StartsWith('\\'))
        {
            if (p.Length > 1 && p[1] == '\\')
            {
                // \\server\share and \\?\UNC\server\share are UNC drives; \\.\device and \\?\device are devices.
                int start = p.Length >= 8 && p.Substring(0, 8).ToUpperInvariant() == "\\\\?\\UNC\\" ? 8 : 2;
                int index = p.IndexOf('\\', start);
                if (index == -1)
                    return (p, "", "");
                int index2 = p.IndexOf('\\', index + 1);
                if (index2 == -1)
                    return (p, "", "");
                return (p.Substring(0, index2), p.Substring(index2, 1), p.Substring(index2 + 1));
            }
            return ("", p.Substring(0, 1), p.Substring(1));
        }
        if (p.Length > 1 && p[1] == ':')
        {
            if (p.Length > 2 && p[2] == '\\')
                return (p.Substring(0, 2), p.Substring(2, 1), p.Substring(3));
            return (p.Substring(0, 2), "", p.Substring(2));
        }
        return ("", "", p);
    }

    /// <summary><c>ntpath.isabs</c>: a drive-rooted path or a UNC/device path.</summary>
    /// <param name="path">The path.</param>
    /// <returns><see langword="true" /> when the path is absolute on Windows.</returns>
    private static bool NtIsAbs(string path)
    {
        string s = path.Replace('/', '\\');
        if (s.StartsWith("\\\\", StringComparison.Ordinal))
            return true;
        (string drive, string root, _) = NtSplitRoot(s);
        return drive.Length > 0 && root.Length > 0;
    }

    /// <summary><c>ntpath.join(a, b)</c> for a relative or root-relative <paramref name="b" />.</summary>
    /// <param name="a">The base directory.</param>
    /// <param name="b">The path to append.</param>
    /// <returns>The joined path.</returns>
    private static string NtJoin(string a, string b)
    {
        (string bDrive, string bRoot, string bRest) = NtSplitRoot(b.Replace('/', '\\'));
        (string aDrive, _, _) = NtSplitRoot(a.Replace('/', '\\'));
        if (bRoot.Length > 0)
            return (bDrive.Length > 0 ? bDrive : aDrive) + bRoot + bRest;
        if (bDrive.Length > 0 && bDrive != aDrive)
        {
            // Python compares drives with str.lower(), which OrdinalIgnoreCase does not reproduce (it equates ς and
            // σ, and not the Kelvin sign and k); different drives make b replace a entirely.
            if (PyStrings.Lower(bDrive) != PyStrings.Lower(aDrive))
                return b;
            // The same drive spelled in another case: Python keeps b's spelling of it.
            a = bDrive + a.Substring(aDrive.Length);
            aDrive = bDrive;
        }
        // No separator after an existing one, nor after a bare drive letter ("C:" + "x" is the drive-relative "C:x");
        // a bare UNC drive does get one.
        bool bareDrive = aDrive.Length == a.Length && aDrive.EndsWith(':');
        string sep = a.EndsWith('\\') || a.EndsWith('/') || bareDrive ? "" : "\\";
        return a + sep + bRest;
    }

    /// <summary>
    ///     <c>os.path.expanduser(path)</c> (CPython 3.12) for the running operating system: a leading <c>~</c> or
    ///     <c>~user</c> becomes that user's home directory; any other path is returned unchanged.
    /// </summary>
    /// <param name="path">The path.</param>
    /// <returns>The expanded path, or <paramref name="path" /> when Python cannot expand it.</returns>
    /// <remarks>
    ///     Windows (<c>ntpath</c>): the home is <c>USERPROFILE</c>, else <c>HOMEDRIVE</c> joined with
    ///     <c>HOMEPATH</c> (neither set: unchanged); <c>~other</c> is guessed as a sibling of the current user's home,
    ///     and only when that home's last component is the current <c>USERNAME</c>. POSIX (<c>posixpath</c>): the home
    ///     is <c>HOME</c>, else the account's directory; <c>~other</c> is looked up in <c>/etc/passwd</c> (CPython asks
    ///     <c>getpwnam</c>, which may also consult directory services; those accounts are not found here).
    /// </remarks>
    internal static string ExpandUser(string path)
    {
        if (!path.StartsWith('~'))
            return path;
        if (OperatingSystem.IsWindows())
        {
            int i = 1;
            while (i < path.Length && path[i] is not ('\\' or '/'))
                i++;
            string userhome;
            if (Environment.GetEnvironmentVariable("USERPROFILE") is { } profile)
                userhome = profile;
            else if (Environment.GetEnvironmentVariable("HOMEPATH") is not { } homePath)
                return path;
            else
                userhome = NtJoin(Environment.GetEnvironmentVariable("HOMEDRIVE") ?? "", homePath);
            if (i != 1)
            {
                string target = path.Substring(1, i - 1);
                string? current = Environment.GetEnvironmentVariable("USERNAME");
                if (target != current)
                {
                    // Profiles usually sit side by side, named after their users; a home that does not follow that
                    // layout makes the guess unreliable, and Python then leaves the path alone.
                    (string head, string tail) = NtSplit(userhome);
                    if (current != tail)
                        return path;
                    userhome = NtJoin(head, target);
                }
            }
            return userhome + path.Substring(i);
        }

        int slash = path.IndexOf('/', 1);
        int end = slash < 0 ? path.Length : slash;
        string? home;
        if (end == 1)
        {
            home = Environment.GetEnvironmentVariable("HOME");
            if (home is null)
            {
                string account = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                if (account.Length == 0)
                    return path;
                home = account;
            }
        }
        else
        {
            home = PasswdHome(path.Substring(1, end - 1));
            if (home is null)
                return path;
        }
        string expanded = home.TrimEnd('/') + path.Substring(end);
        return expanded.Length > 0 ? expanded : "/";
    }

    /// <summary><c>ntpath.split</c>: the directory and the last component, the directory without trailing separators.</summary>
    /// <param name="path">The path.</param>
    /// <returns>The head and the tail.</returns>
    private static (string Head, string Tail) NtSplit(string path)
    {
        (string drive, string root, string rest) = NtSplitRoot(path.Replace('/', '\\'));
        // Slices of the original text, as Python takes them; only the separator test uses the normalized form.
        string original = path.Substring(drive.Length + root.Length);
        int i = rest.Length;
        while (i > 0 && rest[i - 1] != '\\')
            i--;
        return (path.Substring(0, drive.Length + root.Length) + original.Substring(0, i).TrimEnd('\\', '/'), original.Substring(i));
    }

    /// <summary>The home directory <c>/etc/passwd</c> lists for a user, as <c>pwd.getpwnam(name).pw_dir</c> reads it.</summary>
    /// <param name="name">The user name.</param>
    /// <returns>The directory, or <see langword="null" /> when the user is not listed or the file cannot be read.</returns>
    private static string? PasswdHome(string name)
    {
        try
        {
            foreach (string line in File.ReadLines("/etc/passwd"))
            {
                // name:password:uid:gid:gecos:home:shell
                string[] fields = line.Split(':');
                if (fields.Length >= 6 && fields[0] == name)
                    return fields[5];
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            // No readable password file: Python's lookup fails too, and expanduser leaves the path alone.
        }
        return null;
    }

    /// <summary><c>posixpath.join(a, b)</c> for a relative <paramref name="b" />.</summary>
    /// <param name="a">The base directory.</param>
    /// <param name="b">The relative path.</param>
    /// <returns>The joined path.</returns>
    private static string PosixJoin(string a, string b) => a.EndsWith('/') ? a + b : a + "/" + b;

    /// <summary>
    ///     <c>posixpath.normpath</c>: empty and <c>.</c> components vanish, <c>..</c> eats its parent; exactly two
    ///     leading slashes are kept, one or three-plus collapse to one.
    /// </summary>
    /// <param name="path">The path.</param>
    /// <returns>The normalized path; <c>.</c> when nothing is left.</returns>
    internal static string PosixNormPath(string path)
    {
        if (path.Length == 0)
            return ".";
        string initialSlashes;
        string rest;
        if (!path.StartsWith('/'))
        {
            initialSlashes = "";
            rest = path;
        }
        else if (path.Length < 2 || path[1] != '/' || (path.Length > 2 && path[2] == '/'))
        {
            initialSlashes = "/";
            rest = path.Substring(1);
        }
        else
        {
            initialSlashes = "//";
            rest = path.Substring(2);
        }
        var comps = new List<string>();
        foreach (string comp in rest.Split('/'))
        {
            if (comp.Length == 0 || comp == ".")
                continue;
            if (comp != ".." || (initialSlashes.Length == 0 && comps.Count == 0) || (comps.Count > 0 && comps[^1] == ".."))
                comps.Add(comp);
            else if (comps.Count > 0)
                comps.RemoveAt(comps.Count - 1);
        }
        string result = initialSlashes + string.Join("/", comps);
        return result.Length == 0 ? "." : result;
    }
}
