using System;
using System.IO;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Where the JANUS parity tests import the reference implementation from: a clone of
///     <c>https://github.com/hongyuanyu/JANUS</c> (CI pins <c>53e372f</c> through <c>JANUS_REF_PATH</c>).
/// </summary>
internal static class JanusReference
{
    /// <summary>
    ///     The reference clone's folder: <c>JANUS_REF_PATH</c> when set, otherwise a <c>JANUS</c> folder next to the
    ///     repository root, i.e. a sibling clone (<c>…/JANUS</c> beside <c>…/OptunaSharp</c>).
    /// </summary>
    /// <remarks>
    ///     The default used to be one machine's absolute path. The repository root is the first ancestor of the test
    ///     binary that holds <c>SciSharp.OptunaSharp.sln</c>. A build outside the tree finds none; the path then falls
    ///     back to a <c>JANUS</c> folder beside the test output folder, which normally does not exist, so the tests
    ///     report Inconclusive naming the path they tried.
    /// </remarks>
    /// <returns>The folder to put on Python's <c>sys.path</c> before importing <c>janus_optuna</c>.</returns>
    internal static string Folder()
    {
        string? fromEnvironment = Environment.GetEnvironmentVariable("JANUS_REF_PATH");
        if (!string.IsNullOrEmpty(fromEnvironment))
            return fromEnvironment;

        string? dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "SciSharp.OptunaSharp.sln")))
            dir = Path.GetDirectoryName(dir);
        return Path.GetFullPath(Path.Combine(dir ?? AppContext.BaseDirectory, "..", "JANUS"));
    }
}
