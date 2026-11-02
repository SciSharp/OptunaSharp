using System;
using System.Collections.Generic;
using System.IO;
using AwesomeAssertions;
using OptunaSharp;
using OptunaSharp.Cli;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Cli;

/// <summary>
///     Open parity bugs in the <c>optunasharp</c> CLI, found by the 2026-11-02 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-02.md</c>, D52: 50 commands run through both <c>optuna</c> and
///     <c>optunasharp</c> on fresh SQLite storages, comparing exit codes, stdout and the user-visible stderr lines).
///     Each test fails today and passes once fixed, at which point its <c>OpenBugs</c> category should be removed.
/// </summary>
[TestClass]
public class CliParityBugTests
{
    private readonly List<string> _tempFiles = new();

    /// <summary>Deletes the SQLite files the tests created; a file still locked by a pool is left behind.</summary>
    [TestCleanup]
    public void Cleanup()
    {
        foreach (var f in _tempFiles)
        {
            try
            {
                if (File.Exists(f))
                    File.Delete(f);
            }
            catch (IOException)
            {
                // SQLite's connection pool may still hold the file on Windows; the temp folder is cleaned later.
            }
        }
    }

    /// <summary>Returns a URL for a new, empty SQLite file under the temp folder and remembers it for cleanup.</summary>
    /// <returns>A <c>sqlite:///</c> URL with forward slashes, as the CLI expects.</returns>
    private string FreshDbUrl()
    {
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-cli-qa-{Guid.NewGuid():N}.db");
        _tempFiles.Add(path);
        return "sqlite:///" + path.Replace('\\', '/');
    }

    /// <summary>Runs the CLI in-process and captures what it writes to standard output.</summary>
    /// <param name="exitCode">The exit code <see cref="CliProgram.Run" /> returned.</param>
    /// <param name="args">The command line, without the executable name.</param>
    /// <returns>Everything written to <see cref="Console.Out" /> during the call.</returns>
    private static string Run(out int exitCode, params string[] args)
    {
        var original = Console.Out;
        var buffer = new StringWriter();
        Console.SetOut(buffer);
        try
        {
            exitCode = CliProgram.Run(args);
        }
        finally
        {
            Console.SetOut(original);
        }
        return buffer.ToString();
    }

    private const string FloatSpace =
        "{\"x\": {\"name\": \"FloatDistribution\", \"attributes\": {\"low\": 0.0, \"high\": 1.0, \"step\": null, \"log\": false}}}";

    /// <summary>
    ///     QA-35: <c>tell --values</c> must accept what Python's <c>float()</c> accepts, including <c>inf</c>; the port
    ///     parses the option as <see cref="double" /> with .NET's rules, which reject <c>inf</c>, so the command prints
    ///     its help and exits 1 without telling the trial.
    /// </summary>
    /// <remarks>
    ///     Optuna: <c>optuna tell --study-name m --trial-number 0 --values 0.25 inf</c> logs "Told trial 0 with values
    ///     [0.25, inf] and state None." and the trial is COMPLETE with <c>[0.25, inf]</c>. .NET accepts
    ///     <c>Infinity</c> and <c>NaN</c> but not <c>inf</c> (nor <c>1_000</c>). The same parser is behind the known
    ///     catalog item on <c>--values</c> following the locale (de-DE reads <c>0.5</c> as <c>5</c>); parsing with
    ///     <c>PyFloat</c> semantics would fix both. Python's argparse itself rejects <c>-inf</c> (it looks like an
    ///     option), so only positive spellings matter.
    /// </remarks>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void Tell_InfValue_AcceptedLikePythonFloat()
    {
        string url = FreshDbUrl();
        Run(out _, "create-study", "--study-name", "m", "--directions", "minimize", "maximize", "--storage", url);
        Run(out _, "ask", "--study-name", "m", "--storage", url, "--sampler", "RandomSampler",
            "--sampler-kwargs", "{\"seed\": 1}", "--search-space", FloatSpace);

        Run(out int exitCode, "tell", "--study-name", "m", "--storage", url, "--trial-number", "0", "--values", "0.25", "inf");

        exitCode.Should().Be(0);
        using var storage = new RDBStorage(url);
        var trial = Optuna.LoadStudy(studyName: "m", storage: storage).Trials[0];
        trial.State.Should().Be(TrialState.Complete);
        trial.Values.Should().Equal(0.25, double.PositiveInfinity);
    }

    /// <summary>
    ///     QA-36: YAML output must end its lines once; on Windows the port writes <c>CR CR LF</c>, because YamlDotNet's
    ///     text already carries <c>CR LF</c> and the console writer converts its line feeds again.
    /// </summary>
    /// <remarks>
    ///     The YAML format is documented as best-effort (PyYAML's exact bytes are not a target), but doubled carriage
    ///     returns are not a YAML difference: a file written with <c>&gt; out.yaml</c> shows a blank line after every
    ///     line in most editors, and tools that split on <c>\n</c> keep a stray <c>\r</c>. Linux (<c>\n</c> newlines)
    ///     is unaffected, so this test only fails on Windows.
    /// </remarks>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void TrialsYaml_LinesEndOnce()
    {
        string url = FreshDbUrl();
        Run(out _, "create-study", "--study-name", "y", "--storage", url);
        Run(out _, "ask", "--study-name", "y", "--storage", url, "--sampler", "RandomSampler",
            "--sampler-kwargs", "{\"seed\": 1}", "--search-space", FloatSpace);
        Run(out _, "tell", "--study-name", "y", "--storage", url, "--trial-number", "0", "--values", "0.5");

        string yaml = Run(out int exitCode, "trials", "--study-name", "y", "--storage", url, "-f", "yaml");

        exitCode.Should().Be(0);
        yaml.Should().NotContain("\r\r");
    }

    /// <summary>
    ///     QA-37: a command-line usage error must exit with code 2, as argparse does; the port exits 1 and prints the
    ///     command's help to standard output instead of a usage line and an error on standard error.
    /// </summary>
    /// <remarks>
    ///     Optuna: <c>optuna create-study --direction up</c> prints the usage and
    ///     <c>optuna create-study: error: argument --direction: invalid choice: 'up' (choose from minimize, maximize)</c>
    ///     to stderr and exits 2; the same for an invalid <c>tell --state running</c>, an unknown subcommand and a
    ///     missing required option. Scripts that tell usage errors (2) from failed operations (1) apart see them
    ///     differently.
    /// </remarks>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void UsageError_ExitsWith2LikeArgparse()
    {
        string url = FreshDbUrl();

        Run(out int exitCode, "create-study", "--study-name", "s", "--direction", "up", "--storage", url);

        exitCode.Should().Be(2);
    }
}
