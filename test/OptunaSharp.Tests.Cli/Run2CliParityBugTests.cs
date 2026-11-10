using System;
using System.Collections.Generic;
using System.IO;
using AwesomeAssertions;
using OptunaSharp;
using OptunaSharp.Cli;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Cli;

/// <summary>
///     Open parity bugs in the <c>optunasharp</c> CLI found by the 2026-11-10 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-10.md</c>). Each test fails today and passes once fixed, at which point its
///     <c>OpenBugs</c> category should be removed.
/// </summary>
[TestClass]
public class Run2CliParityBugTests
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

    /// <summary>Runs the CLI in-process and captures what it writes to standard output.</summary>
    /// <param name="exitCode">The exit code <see cref="CliProgram.Run" /> returned.</param>
    /// <param name="args">The command line, without the executable name.</param>
    /// <returns>Everything written to <see cref="Console.Out" /> during the call, line feeds only.</returns>
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
        // Compare lines, not line endings (QA-36 already covers the doubled carriage return).
        return buffer.ToString().Replace("\r", "");
    }

    /// <summary>
    ///     QA-61: <c>-f yaml</c> writes floats and datetimes in forms a YAML 1.1 reader (PyYAML, which Optuna's own
    ///     output targets) reads back as other types. PyYAML's <c>safe_dump</c> writes every float with a decimal point
    ///     (<c>2.0</c>, <c>1.0e-07</c>, <c>1.0e+16</c>) and quotes the formatted datetimes; YamlDotNet writes <c>2</c>,
    ///     <c>1E-07</c>, <c>10000000000000000</c> and bare datetimes.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Optuna 5.0.0, <c>optuna trials -f yaml</c> on a trial with <c>x=1e-07</c>, <c>y=2.0</c> and value
    ///         <c>1e16</c>: <c>x: 1.0e-07</c>, <c>y: 2.0</c>, <c>value: 1.0e+16</c>,
    ///         <c>datetime_start: '2026-11-10 22:51:17'</c>. The port: <c>x: 1E-07</c>, <c>"y": 2</c>,
    ///         <c>value: 10000000000000000</c>, <c>datetime_start: 2026-11-10 22:51:17</c>.
    ///     </para>
    ///     <para>
    ///         Read back with <c>yaml.safe_load</c>, the port's output gives the string <c>'1E-07'</c> (YAML 1.1's float
    ///         needs a decimal point), the int <c>2</c> and the int <c>10000000000000000</c> for the three floats, and a
    ///         <c>datetime</c> object where Optuna's output gives a string. <c>OutputFormat.cs</c> calls YAML byte parity
    ///         best-effort; these change the values' types, not only their bytes. The key quoting (<c>"y"</c>, YAML 1.1's
    ///         boolean <c>y</c>) and double instead of single quotes read back the same and are not asserted.
    ///     </para>
    /// </remarks>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void TrialsYaml_FloatsAndDatetimes_KeepTheirTypesLikePyYaml()
    {
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-cli-qa61-{Guid.NewGuid():N}.db");
        _tempFiles.Add(path);
        string url = "sqlite:///" + path.Replace('\\', '/');
        var study = Optuna.CreateStudy(studyName: "q", storage: new RDBStorage(url));
        study.AddTrial(FrozenTrial.CreateTrial(
            value: 1e16,
            @params: new OrderedDict<string, object?> { ["x"] = 1e-07, ["y"] = 2.0 },
            distributions: new OrderedDict<string, BaseDistribution>
            {
                ["x"] = new FloatDistribution(0, 1),
                ["y"] = new FloatDistribution(0, 5),
            }));

        string yaml = Run(out int exitCode, "trials", "--storage", url, "--study-name", "q", "-f", "yaml");

        exitCode.Should().Be(0);
        yaml.Should().Contain("    x: 1.0e-07\n");
        yaml.Should().Contain("    y: 2.0\n");
        yaml.Should().Contain("  value: 1.0e+16\n");
        yaml.Should().MatchRegex(@"datetime_start: '\d{4}-\d\d-\d\d \d\d:\d\d:\d\d'");
    }

    /// <summary>
    ///     QA-63: <c>tell --state fail</c> and <c>tell --state pruned</c> without <c>--values</c> must finish the trial in
    ///     that state, as Optuna's CLI does. The port's <c>--values</c> option yields an empty array (not null) when it
    ///     is omitted, so Core's <c>tell</c> sees values next to a PRUNED / FAIL state and refuses the call: the command
    ///     dies with <c>Unhandled exception: NumSharp.ValueError: Values were told. Values cannot be specified when state
    ///     is TrialState.PRUNED or TrialState.FAIL.</c>, exits 1, and the trial stays RUNNING.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Optuna 5.0.0: <c>optuna tell --storage … --study-name s --trial-number 0 --state fail</c> logs "Told
    ///         trial 0 with values None and state TrialState.FAIL." and exits 0 (argparse's <c>nargs="+"</c> option
    ///         defaults to <c>None</c>). Marking a failed or pruned trial from a shell script is the CLI's ask-and-tell
    ///         workflow, so this is the one finding of the run on a common path.
    ///     </para>
    ///     <para>
    ///         The same empty array changes a bare <c>tell --trial-number 2</c> (no values, no state): Optuna warns
    ///         "The value None could not be cast to float." and logs "Told trial 1 with values None and state None.";
    ///         the port warns "The number of the values 0 did not match the number of the objectives 1" and logs "with
    ///         values []". Both fail the trial. <c>CliParityTests</c>' setup runs these two commands too, but ignores
    ///         their exit codes, so its fixture silently kept two RUNNING trials.
    ///     </para>
    /// </remarks>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void Tell_StateWithoutValues_FinishesTheTrialLikeOptuna()
    {
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-cli-qa63-{Guid.NewGuid():N}.db");
        _tempFiles.Add(path);
        string url = "sqlite:///" + path.Replace('\\', '/');
        const string space =
            "{\"x\": {\"name\": \"FloatDistribution\", \"attributes\": {\"low\": 0.0, \"high\": 1.0, \"step\": null, \"log\": false}}}";

        Run(out _, "create-study", "--storage", url, "--study-name", "s");
        Run(out _, "ask", "--storage", url, "--study-name", "s", "--search-space", space);
        Run(out _, "ask", "--storage", url, "--study-name", "s", "--search-space", space);

        Run(out int failExit, "tell", "--storage", url, "--study-name", "s", "--trial-number", "0", "--state", "fail");
        Run(out int prunedExit, "tell", "--storage", url, "--study-name", "s", "--trial-number", "1", "--state", "pruned");

        failExit.Should().Be(0);
        prunedExit.Should().Be(0);
        var trials = Optuna.LoadStudy("s", new RDBStorage(url)).Trials;
        trials[0].State.Should().Be(TrialState.Fail);
        trials[1].State.Should().Be(TrialState.Pruned);
    }

    /// <summary>
    ///     QA-64: <c>--sampler-kwargs</c> values that are JSON objects or arrays must reach the sampler, as Optuna's
    ///     <c>getattr(optuna.samplers, name)(**kwargs)</c> passes them. The port's <c>SamplerFactory.TryCoerce</c> binds
    ///     only scalars (bool, string, int, long, double, float, enums), so <c>GridSampler</c>'s <c>search_space</c>,
    ///     <c>CmaEsSampler</c>'s <c>x0</c> and <c>NSGAIIISampler</c>'s <c>reference_points</c> make the command fail with
    ///     <c>Could not construct sampler 'GridSampler' from the given --sampler-kwargs ({search_space}).</c>.
    /// </summary>
    /// <remarks>
    ///     Optuna 5.0.0: <c>optuna ask … --search-space '{"x": FloatDistribution(0, 1)}' --sampler GridSampler
    ///     --sampler-kwargs '{"search_space": {"x": [0.5]}}'</c> prints <c>{"number": 0, "params": {"x": 0.5}}</c>;
    ///     <c>--sampler CmaEsSampler --sampler-kwargs '{"x0": {"x": 0.25}, "sigma0": 0.1, "seed": 1}'</c> and
    ///     <c>--sampler NSGAIIISampler --sampler-kwargs '{"reference_points": [[1.0]], "seed": 1}'</c> ask normally
    ///     (exit 0). The port exits 1 for all three. GridSampler is the natural CLI sampler for a shell-driven grid sweep.
    /// </remarks>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void Ask_StructuredSamplerKwargs_ConstructTheSamplerLikeOptuna()
    {
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-cli-qa64-{Guid.NewGuid():N}.db");
        _tempFiles.Add(path);
        string url = "sqlite:///" + path.Replace('\\', '/');
        const string space =
            "{\"x\": {\"name\": \"FloatDistribution\", \"attributes\": {\"low\": 0.0, \"high\": 1.0, \"step\": null, \"log\": false}}}";
        Run(out _, "create-study", "--storage", url, "--study-name", "s");

        string grid = Run(out int gridExit, "ask", "--storage", url, "--study-name", "s", "--search-space", space,
            "--sampler", "GridSampler", "--sampler-kwargs", "{\"search_space\": {\"x\": [0.5]}}");
        Run(out int cmaExit, "ask", "--storage", url, "--study-name", "s", "--search-space", space,
            "--sampler", "CmaEsSampler", "--sampler-kwargs", "{\"x0\": {\"x\": 0.25}, \"sigma0\": 0.1, \"seed\": 1}");

        gridExit.Should().Be(0);
        grid.Should().Be("{\"number\": 0, \"params\": {\"x\": 0.5}}\n");
        cmaExit.Should().Be(0);
    }

    /// <summary>
    ///     QA-65: Optuna's <c>--log-file</c> attaches a <c>logging.FileHandler</c> to Python's ROOT logger, but the
    ///     <c>optuna</c> logger is configured with <c>propagate = False</c>, so none of Optuna's records reach it and the
    ///     file stays empty; the port attaches its file handler to Optuna's own logger, so the file receives every log
    ///     line. An Optuna defect the port does not copy; the owner decides (as for QA-46).
    /// </summary>
    /// <remarks>
    ///     Measured with Optuna 5.0.0: <c>optuna --log-file f.log create-study …</c> leaves <c>f.log</c> at 0 bytes, also
    ///     with <c>-v</c> and with <c>--debug</c> on a failing <c>delete-study</c>; stderr still shows the log lines (the
    ///     <c>optuna</c> logger's own handler). <c>optunasharp --log-file f.log create-study …</c> writes
    ///     <c>[I …] A new study created in RDB with name: f</c>. The CLI runs as a separate process because its log-file
    ///     handler is never detached and would capture later tests' logs.
    /// </remarks>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void LogFile_ReceivesNoOptunaRecords_LikeOptuna()
    {
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-cli-qa65-{Guid.NewGuid():N}.db");
        string logFile = Path.Combine(Path.GetTempPath(), $"optunasharp-cli-qa65-{Guid.NewGuid():N}.log");
        _tempFiles.Add(path);
        _tempFiles.Add(logFile);
        string url = "sqlite:///" + path.Replace('\\', '/');

        string host = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH") is { Length: > 0 } h ? h : "dotnet";
        var psi = new System.Diagnostics.ProcessStartInfo(host)
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (string a in new[]
                 {
                     Path.Combine(AppContext.BaseDirectory, "OptunaSharp.Cli.dll"),
                     "--log-file", logFile, "create-study", "--storage", url, "--study-name", "f",
                 })
            psi.ArgumentList.Add(a);
        using (var process = System.Diagnostics.Process.Start(psi)!)
        {
            // Drain both pipes before waiting, so the child never blocks on a full pipe.
            var stdout = process.StandardOutput.ReadToEndAsync();
            var stderr = process.StandardError.ReadToEndAsync();
            process.WaitForExit(120_000).Should().BeTrue("the CLI finishes within two minutes");
            stdout.GetAwaiter().GetResult();
            stderr.GetAwaiter().GetResult();
            process.ExitCode.Should().Be(0);
        }

        File.Exists(logFile).Should().BeTrue("Optuna's FileHandler creates the file");
        File.ReadAllText(logFile).Should().BeEmpty("Optuna's records never propagate to the root logger's file handler");
    }
}
