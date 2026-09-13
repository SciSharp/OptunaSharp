using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using AwesomeAssertions;

namespace OptunaSharp.Tests.PackageParity;

/// <summary>
///     The 20th shipped package, <c>OptunaSharp.Cli</c>, is a <c>dotnet tool</c> (<c>PackAsTool</c>): its
///     <c>.nupkg</c> has no <c>lib/</c> assembly to reference, so it cannot be a PackageReference and is not
///     covered by <see cref="PackagedAssembliesSmokeTests" />. This gate exercises it the only way a
///     consumer can — it installs the tool FROM THE LOCAL FEED into a throwaway tool-path and runs it —
///     proving the shipped tool package restores its whole dependency graph (Core + NumSharp/OpenBLAS +
///     the CLI's own deps) and actually executes.
/// </summary>
/// <remarks>
///     The install runs with the working directory set to the feed's parent (this project's directory) so it
///     picks up the project's <c>NuGet.config</c> — the same source mapping the PackageReferences use
///     (OptunaSharp* from the feed, everything else from nuget.org). Third-party deps are fetched at install
///     time, so this case assumes network access to nuget.org (present in CI and local compose runs); when
///     the feed has not been populated at all it self-skips (Inconclusive) rather than fail spuriously.
/// </remarks>
[TestClass]
public class PackagedCliToolSmokeTests
{
    /// <summary>
    ///     Installs <c>OptunaSharp.Cli</c> from the feed and runs <c>optunasharp --help</c>, asserting the
    ///     tool process starts, loads its packaged dependencies, and responds successfully.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresTool")]
    public void PackagedCliTool_InstallsFromFeed_AndRuns()
    {
        string feedDir = ResolveFeedDir();
        string version = Environment.GetEnvironmentVariable("OPTUNASHARP_VERSION") is { Length: > 0 } v ? v : "0.1.0";
        string cliNupkg = Path.Combine(feedDir, $"OptunaSharp.Cli.{version}.nupkg");

        // If the feed was never populated (e.g., the assembly was run outside the packaged-parity flow),
        // there is nothing to consume — skip rather than mislead. When the pipeline/auto-pack ran, it is present.
        if (!File.Exists(cliNupkg))
        {
            Assert.Inconclusive(
                $"CLI tool package not in the feed ({cliNupkg}); the pipeline / auto-pack must produce it first.");
            return;
        }

        // Throwaway install location so nothing leaks into the machine's tool set; cleaned in finally.
        string toolPath = Path.Combine(Path.GetTempPath(), "optunasharp-cli-parity-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(toolPath);
        // The install must resolve the CLI + Core from the feed and the rest from nuget.org — the project's
        // NuGet.config (one directory up from the feed) encodes exactly that mapping, so run install there.
        string workingDir = Directory.GetParent(feedDir)!.FullName;

        try
        {
            (int installExit, string installOut, string installErr) = Run(
                "dotnet",
                $"tool install OptunaSharp.Cli --version {version} --tool-path \"{toolPath}\"",
                workingDir);
            installExit.Should().Be(0, $"the shipped CLI tool must install from the feed.\nSTDOUT:\n{installOut}\nSTDERR:\n{installErr}");

            // The shim is named by ToolCommandName (optunasharp), with .exe on Windows.
            string exe = Path.Combine(toolPath,
                RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "optunasharp.exe" : "optunasharp");
            File.Exists(exe).Should().BeTrue($"the tool shim must be created at {exe}");

            (int runExit, string runOut, string runErr) = Run(exe, "--help", workingDir);
            runExit.Should().Be(0, $"the installed CLI must run.\nSTDOUT:\n{runOut}\nSTDERR:\n{runErr}");
            (runOut + runErr).Trim().Should().NotBeEmpty("`optunasharp --help` must print usage");
        }
        finally
        {
            // Best-effort cleanup — a leftover temp tool-path must never fail the test.
            try { Directory.Delete(toolPath, recursive: true); } catch { /* ignore */ }
        }
    }

    /// <summary>
    ///     Locates the local feed: an explicit <c>PACKAGE_PARITY_FEED_DIR</c> wins, else <c>feed/</c> next to
    ///     the project (three levels up from the test binary's <c>bin/&lt;cfg&gt;/&lt;tfm&gt;</c>), which is
    ///     where both the CI deploy and the MSBuild auto-pack place the packages.
    /// </summary>
    /// <returns>The absolute feed directory path (which may not exist yet).</returns>
    private static string ResolveFeedDir()
    {
        if (Environment.GetEnvironmentVariable("PACKAGE_PARITY_FEED_DIR") is { Length: > 0 } explicitDir)
            return Path.GetFullPath(explicitDir);
        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "feed"));
    }

    /// <summary>
    ///     Runs a process to completion capturing stdout/stderr, with a bounded wait so a hung child can never
    ///     stall the suite indefinitely.
    /// </summary>
    /// <param name="fileName">The executable to launch (a resolvable command or an absolute path).</param>
    /// <param name="arguments">The full argument string.</param>
    /// <param name="workingDir">Working directory — sets the NuGet.config scope for <c>dotnet tool install</c>.</param>
    /// <returns>The exit code (or -1 on timeout) with the captured standard output and error.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the process cannot be started.</exception>
    private static (int ExitCode, string StdOut, string StdErr) Run(string fileName, string arguments, string workingDir)
    {
        var psi = new ProcessStartInfo(fileName, arguments)
        {
            WorkingDirectory = workingDir,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        using var proc = Process.Start(psi)
            ?? throw new InvalidOperationException($"could not start '{fileName} {arguments}'");

        // Read both streams concurrently to avoid a pipe-buffer deadlock on chatty output.
        var stdout = new StringBuilder();
        var stderr = new StringBuilder();
        proc.OutputDataReceived += (_, e) => { if (e.Data is not null) stdout.AppendLine(e.Data); };
        proc.ErrorDataReceived += (_, e) => { if (e.Data is not null) stderr.AppendLine(e.Data); };
        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();

        // A tool install + a --help run are quick; 5 minutes is a generous ceiling for a slow restore.
        if (!proc.WaitForExit(milliseconds: 300_000))
        {
            try { proc.Kill(entireProcessTree: true); } catch { /* already gone */ }
            return (-1, stdout.ToString(), stderr.ToString() + "\n[timed out after 300s]");
        }
        // Flush any tail of the async reads after exit.
        proc.WaitForExit();
        return (proc.ExitCode, stdout.ToString(), stderr.ToString());
    }
}
