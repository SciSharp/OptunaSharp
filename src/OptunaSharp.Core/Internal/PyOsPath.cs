using System;
using System.Collections.Generic;
using System.IO;

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
        if (bDrive.Length > 0 && !string.Equals(bDrive, aDrive, StringComparison.OrdinalIgnoreCase))
            return b;
        string sep = a.EndsWith('\\') || a.EndsWith('/') ? "" : "\\";
        return a + sep + bRest;
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
