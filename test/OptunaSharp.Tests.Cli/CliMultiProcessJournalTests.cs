using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using AwesomeAssertions;
using OptunaSharp;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Tests.Cli;

/// <summary>
///     Separate <c>optunasharp</c> processes asking and telling on one journal file at once: the CLI's flagship use in
///     parallel shells, and the real multi-process shape of the journal concurrency fixes. Every round starts its
///     processes together, so they contend for the journal lock and seed their samplers at nearly the same moment.
/// </summary>
/// <remarks>
///     <para>
///         Two fixes meet here. The file locks classify contention by the failed call's error code; they used to
///         re-check the lock path and let a release in between kill the waiting process. And an unseeded sampler seeds
///         itself from OS entropy; it took NumSharp's <c>Environment.TickCount</c> seed, so processes started within one
///         tick drew identical values (8 parallel shells repeated 11 and 15 of 80 draws in two runs).
///     </para>
///     <para>
///         The CLI runs as <c>dotnet OptunaSharp.Cli.dll</c> from this test's output folder, where the project reference
///         copies it with its runtime configuration; the case is inconclusive when that copy is missing.
///     </para>
/// </remarks>
[TestClass]
public class CliMultiProcessJournalTests
{
    private const int Processes = 6;
    private const int Rounds = 3;
    private const string SearchSpace =
        "{\"x\": {\"name\": \"FloatDistribution\", \"attributes\": {\"low\": -10.0, \"high\": 10.0, \"log\": false, \"step\": null}}}";

    private string _journal = null!;

    /// <summary>Creates an empty journal file under a unique temp name for this test.</summary>
    [TestInitialize]
    public void SetUp()
    {
        _journal = Path.Combine(Path.GetTempPath(), $"optunasharp-cli-mp-{Guid.NewGuid():N}.log");
        File.WriteAllText(_journal, "");
    }

    /// <summary>Deletes the journal and its lock; a file still held by a straggling process is left for temp cleanup.</summary>
    [TestCleanup]
    public void TearDown()
    {
        foreach (string f in new[] { _journal, _journal + ".lock" })
        {
            try
            {
                File.Delete(f);
            }
            catch (IOException)
            {
            }
        }
    }

    /// <summary>
    ///     Six processes per round, three rounds of <c>ask</c> then <c>tell</c>: every command succeeds, the study holds
    ///     18 complete trials numbered 0 to 17, and no two processes drew the same value.
    /// </summary>
    [TestMethod]
    public void ParallelAskTell_FromSeparateProcesses_IsAPerfectStudyWithoutRepeatedDraws()
    {
        string cli = Path.Combine(AppContext.BaseDirectory, "OptunaSharp.Cli.dll");
        if (!File.Exists(cli) || !File.Exists(Path.Combine(AppContext.BaseDirectory, "OptunaSharp.Cli.runtimeconfig.json")))
            Assert.Inconclusive("The optunasharp CLI is not in the test output folder.");

        RunAll(cli, new[] { new[] { "create-study", "--study-name", "par" } }).Single().ExitCode.Should().Be(0);

        for (int round = 0; round < Rounds; round++)
        {
            var asks = RunAll(cli, Enumerable.Range(0, Processes)
                .Select(_ => new[] { "ask", "--study-name", "par", "--search-space", SearchSpace, "-f", "json" }).ToList());
            asks.Should().OnlyContain(r => r.ExitCode == 0, "no process may die on the journal lock");
            var numbers = asks.Select(r => JsonDocument.Parse(r.Stdout).RootElement.GetProperty("number").GetInt32()).ToList();

            var tells = RunAll(cli, numbers
                .Select(n => new[] { "tell", "--study-name", "par", "--trial-number", n.ToString(), "--values", "1.5" }).ToList());
            tells.Should().OnlyContain(r => r.ExitCode == 0, "no process may die on the journal lock");
        }

        var trials = Optuna.LoadStudy("par", new JournalStorage(new JournalFileBackend(_journal))).Trials;
        trials.Select(t => t.Number).Should().Equal(Enumerable.Range(0, Processes * Rounds));
        trials.Should().OnlyContain(t => t.State == TrialState.Complete);
        trials.Select(t => Convert.ToDouble(t.Params["x"])).Should().OnlyHaveUniqueItems(
            "unseeded samplers in separate processes must not share a random stream");
    }

    /// <summary>One finished CLI process: its exit code, standard output and standard error.</summary>
    /// <param name="ExitCode">The process exit code; 0 on success.</param>
    /// <param name="Stdout">Everything the process wrote to standard output.</param>
    /// <param name="Stderr">Everything the process wrote to standard error.</param>
    private sealed record CliRun(int ExitCode, string Stdout, string Stderr);

    /// <summary>Starts one CLI process per command line, all at once, and waits for every one of them.</summary>
    /// <param name="cli">The path of <c>OptunaSharp.Cli.dll</c>.</param>
    /// <param name="commands">The command lines, each without the storage options, which are appended here.</param>
    /// <returns>Each process's result, in command order.</returns>
    /// <exception cref="TimeoutException">A process did not finish within two minutes.</exception>
    private List<CliRun> RunAll(string cli, IReadOnlyList<string[]> commands)
    {
        string host = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH") is { Length: > 0 } h ? h : "dotnet";
        var processes = commands.Select(args =>
        {
            var psi = new ProcessStartInfo(host)
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            psi.ArgumentList.Add(cli);
            foreach (string a in args)
                psi.ArgumentList.Add(a);
            psi.ArgumentList.Add("--storage-class");
            psi.ArgumentList.Add("JournalFileBackend");
            psi.ArgumentList.Add("--storage");
            psi.ArgumentList.Add(_journal);
            return Process.Start(psi)!;
        }).ToList();

        // Drain both pipes concurrently with the processes, so none blocks on a full pipe buffer.
        var outputs = processes.Select(p => (Out: p.StandardOutput.ReadToEndAsync(), Err: p.StandardError.ReadToEndAsync())).ToList();
        var results = new List<CliRun>();
        for (int i = 0; i < processes.Count; i++)
        {
            using var p = processes[i];
            if (!p.WaitForExit(TimeSpan.FromMinutes(2)))
                throw new TimeoutException($"optunasharp {string.Join(" ", commands[i])} did not finish");
            results.Add(new CliRun(p.ExitCode, outputs[i].Out.GetAwaiter().GetResult(), outputs[i].Err.GetAwaiter().GetResult()));
        }
        return results;
    }
}
