using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AwesomeAssertions;
using OptunaSharp.Cli;

namespace OptunaSharp.Tests.Cli;

/// <summary>
///     Differential parity of the CLI's stdout against the <b>live</b> <c>optuna</c> command. A study
///     is built in a SQLite DB by the OptunaSharp CLI (whose RDB schema is Optuna-compatible), then each
///     read command is run through both the real <c>optuna</c> CLI (a subprocess) and OptunaSharp's CLI
///     in-process; the two stdouts must be byte-identical for the <c>value</c>/<c>json</c>/<c>table</c>
///     formats. <c>yaml</c> is intentionally excluded (PyYAML's scalar quoting is not a parity target).
///     Self-skips (Inconclusive) when the <c>optuna</c> CLI is not discoverable.
/// </summary>
[TestClass]
public class CliParityTests
{
    private static string? _optunaCli;
    private static string _soUrl = "";
    private static string _moUrl = "";
    private static string _attrsUrl = "";
    private static string _emptyUrl = "";
    private static string _journalPath = "";
    private static readonly string[] TextFormats = { "value", "json", "table" };

    private const string FloatSpace =
        "{\"x\": {\"name\": \"FloatDistribution\", \"attributes\": {\"low\": 0.0, \"high\": 1.0, \"log\": false, \"step\": null}}}";

    [ClassInitialize]
    public static void ClassInitialize(TestContext _)
    {
        _optunaCli = FindOptunaCli();

        // Single-objective fixture: create-study + two seeded ask/tell trials (bit-exact RandomSampler).
        _soUrl = FreshDbUrl("so");
        RunInProcess("create-study", "--study-name", "demo", "--storage", _soUrl);
        for (int i = 0; i < 2; i++)
        {
            RunInProcess("ask", "--study-name", "demo", "--storage", _soUrl,
                "--sampler", "RandomSampler", "--sampler-kwargs", $"{{\"seed\": {i}}}", "--search-space", FloatSpace);
            RunInProcess("tell", "--study-name", "demo", "--storage", _soUrl, "--trial-number", i.ToString(), "--values", (0.5 + i).ToString("R"));
        }

        // Multi-objective fixture: exercises the integer `values` columns + the Pareto best-trials.
        _moUrl = FreshDbUrl("mo");
        RunInProcess("create-study", "--study-name", "mo", "--directions", "minimize", "maximize", "--storage", _moUrl);
        for (int i = 0; i < 3; i++)
        {
            RunInProcess("ask", "--study-name", "mo", "--storage", _moUrl,
                "--sampler", "RandomSampler", "--sampler-kwargs", $"{{\"seed\": {i}}}", "--search-space", FloatSpace);
            RunInProcess("tell", "--study-name", "mo", "--storage", _moUrl, "--trial-number", i.ToString(),
                "--values", (0.1 * i).ToString("R"), (0.9 - 0.1 * i).ToString("R"));
        }

        // Study-user-attr + mixed trial states (COMPLETE / PRUNED / FAIL): exercises the studies
        // user_attrs column and the None value of pruned/failed trials in `trials`.
        _attrsUrl = FreshDbUrl("attrs");
        RunInProcess("create-study", "--study-name", "attrs", "--storage", _attrsUrl);
        RunInProcess("study", "set-user-attr", "--study-name", "attrs", "--key", "owner", "--value", "alice", "--storage", _attrsUrl);
        AskAtSeed(_attrsUrl, "attrs", 0);
        RunInProcess("tell", "--study-name", "attrs", "--storage", _attrsUrl, "--trial-number", "0", "--values", "0.5");
        AskAtSeed(_attrsUrl, "attrs", 1);
        RunInProcess("tell", "--study-name", "attrs", "--storage", _attrsUrl, "--trial-number", "1", "--state", "pruned");
        AskAtSeed(_attrsUrl, "attrs", 2);
        RunInProcess("tell", "--study-name", "attrs", "--storage", _attrsUrl, "--trial-number", "2", "--state", "fail");

        // Empty study (zero trials).
        _emptyUrl = FreshDbUrl("empty");
        RunInProcess("create-study", "--study-name", "empty", "--storage", _emptyUrl);

        // Journal-file storage (a different backend, byte-compatible with Python's). Built with the
        // explicit class hint (the file does not exist yet); reads then guess journal from File.Exists.
        _journalPath = Path.Combine(Path.GetTempPath(), $"optunasharp-cli-journal-{Guid.NewGuid():N}.log");
        RunInProcess("create-study", "--study-name", "j", "--storage-class", "JournalFileBackend", "--storage", _journalPath);
        RunInProcess("ask", "--study-name", "j", "--storage-class", "JournalFileBackend", "--storage", _journalPath,
            "--sampler", "RandomSampler", "--sampler-kwargs", "{\"seed\": 0}", "--search-space", FloatSpace);
        RunInProcess("tell", "--study-name", "j", "--storage-class", "JournalFileBackend", "--storage", _journalPath,
            "--trial-number", "0", "--values", "0.5");

        // Release any leaked SQLite connections before the optuna subprocess reads the same files.
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }

    private static void AskAtSeed(string url, string study, int seed)
        => RunInProcess("ask", "--study-name", study, "--storage", url,
            "--sampler", "RandomSampler", "--sampler-kwargs", $"{{\"seed\": {seed}}}", "--search-space", FloatSpace);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void StudyNames_AllTextFormats()
    {
        var checks = new ParityBatch();
        foreach (var f in TextFormats)
            checks.Add(_soUrl, "study-names", "-f", f);
        checks.AssertAll();
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Studies_AllTextFormats_PlainAndFlatten()
    {
        var checks = new ParityBatch();
        foreach (var f in TextFormats)
        {
            checks.Add(_soUrl, "studies", "-f", f);
            checks.Add(_soUrl, "studies", "-f", f, "--flatten");
        }
        checks.AssertAll();
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Trials_SingleObjective_AllTextFormats_PlainAndFlatten()
    {
        var checks = new ParityBatch();
        foreach (var f in TextFormats)
        {
            checks.Add(_soUrl, "trials", "--study-name", "demo", "-f", f);
            checks.Add(_soUrl, "trials", "--study-name", "demo", "-f", f, "--flatten");
        }
        checks.AssertAll();
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void BestTrial_SingleObjective_AllTextFormats()
    {
        var checks = new ParityBatch();
        foreach (var f in TextFormats)
            checks.Add(_soUrl, "best-trial", "--study-name", "demo", "-f", f, "--flatten");
        checks.AssertAll();
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Trials_MultiObjective_ValuesColumns_AllTextFormats()
    {
        var checks = new ParityBatch();
        foreach (var f in TextFormats)
        {
            checks.Add(_moUrl, "trials", "--study-name", "mo", "-f", f);
            checks.Add(_moUrl, "trials", "--study-name", "mo", "-f", f, "--flatten");
        }
        checks.AssertAll();
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void BestTrials_MultiObjective_ParetoFront_AllTextFormats()
    {
        var checks = new ParityBatch();
        foreach (var f in TextFormats)
            checks.Add(_moUrl, "best-trials", "--study-name", "mo", "-f", f);
        checks.AssertAll();
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Studies_WithUserAttrs_AllTextFormats_PlainAndFlatten()
    {
        var checks = new ParityBatch();
        foreach (var f in TextFormats)
        {
            checks.Add(_attrsUrl, "studies", "-f", f);
            checks.Add(_attrsUrl, "studies", "-f", f, "--flatten");
        }
        checks.AssertAll();
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Trials_WithPrunedAndFailedTrials_AllTextFormats()
    {
        // The pruned/failed trials have a None objective value and no datetime_complete/duration.
        var checks = new ParityBatch();
        foreach (var f in TextFormats)
        {
            checks.Add(_attrsUrl, "trials", "--study-name", "attrs", "-f", f);
            checks.Add(_attrsUrl, "trials", "--study-name", "attrs", "-f", f, "--flatten");
        }
        checks.AssertAll();
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void EmptyStudy_Trials_AllTextFormats()
    {
        var checks = new ParityBatch();
        foreach (var f in TextFormats)
            checks.Add(_emptyUrl, "trials", "--study-name", "empty", "-f", f);
        checks.AssertAll();
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void JournalStorage_StudiesAndTrials_AllTextFormats()
    {
        // Reads omit --storage-class: both CLIs must guess journal-file from the existing file.
        var checks = new ParityBatch();
        foreach (var f in TextFormats)
        {
            checks.Add(_journalPath, "studies", "-f", f);
            checks.Add(_journalPath, "trials", "--study-name", "j", "-f", f);
        }
        checks.AssertAll();
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Ask_SingleObjective_SeededSuggestion_AllTextFormats()
    {
        // A fresh, identical study on each side + a seeded RandomSampler => the suggested parameter is
        // bit-identical, so `ask`'s stdout must match exactly.
        string[] create = { "create-study", "--study-name", "q" };
        var checks = new ParityBatch();
        foreach (var f in new[] { "json", "value", "table" })
            checks.AddFresh(create, new[]
            {
                "ask", "--study-name", "q", "--sampler", "RandomSampler", "--sampler-kwargs", "{\"seed\": 0}",
                "--search-space", FloatSpace, "-f", f,
            });
        checks.AddFresh(create, new[]
        {
            "ask", "--study-name", "q", "--sampler", "RandomSampler", "--sampler-kwargs", "{\"seed\": 0}",
            "--search-space", FloatSpace, "-f", "json", "--flatten",
        });
        checks.AssertAll();
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void CreateStudy_PrintsStudyName()
    {
        var checks = new ParityBatch();
        checks.AddFresh(Array.Empty<string>(), new[] { "create-study", "--study-name", "made" });
        checks.AddFresh(Array.Empty<string>(),
            new[] { "create-study", "--study-name", "mo2", "--directions", "minimize", "maximize" });
        checks.AssertAll();
    }

    /// <summary>
    ///     CLI recipe (<c>recipes/cli.md</c>): the shell workflow create-study → ask(--search-space) →
    ///     tell(--values --state). The recipe's exact −10..10 FloatDistribution search space (seeded for a
    ///     deterministic suggestion) must give byte-identical <c>ask</c> output to the real <c>optuna</c>
    ///     CLI, and after a <c>tell --values 0.7 --state complete</c> both CLIs must read the resulting
    ///     study identically in every text format.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Recipe_CliAskTellWorkflow_Parity()
    {
        const string recipeSpace =
            "{\"x\": {\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": -10.0, \"high\": 10.0, \"log\": false}}}";

        // (1) `ask` with the recipe's exact search space must match optuna byte-for-byte.
        var checks = new ParityBatch();
        checks.AddFresh(
            new[] { "create-study", "--study-name", "recipe-demo" },
            new[]
            {
                "ask", "--study-name", "recipe-demo", "--sampler", "RandomSampler", "--sampler-kwargs", "{\"seed\": 0}",
                "--search-space", recipeSpace, "-f", "json",
            });

        // (2) create → ask → tell(--values 0.7 --state complete): OptunaSharp writes it; both CLIs read it identically.
        // The writes finish before any reference process starts: AssertAll launches them, after finalizing the
        // connections these writes leaked.
        string url = FreshDbUrl("recipe-workflow");
        RunInProcess("create-study", "--study-name", "recipe-demo", "--storage", url);
        RunInProcess("ask", "--study-name", "recipe-demo", "--storage", url,
            "--sampler", "RandomSampler", "--sampler-kwargs", "{\"seed\": 0}", "--search-space", recipeSpace);
        RunInProcess("tell", "--study-name", "recipe-demo", "--storage", url,
            "--trial-number", "0", "--values", "0.7", "--state", "complete");

        foreach (var f in TextFormats)
        {
            checks.Add(url, "trials", "--study-name", "recipe-demo", "-f", f);
            checks.Add(url, "best-trial", "--study-name", "recipe-demo", "-f", f);
        }
        checks.AssertAll();
    }

    // ------------------------------------------------------------------ helpers

    /// <summary>
    ///     One test's parity checks, collected first and asserted together, so that their reference <c>optuna</c>
    ///     processes run at the same time instead of one after another.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Each check still runs one fresh <c>optuna</c> process, exactly as before; only the waiting overlaps.
    ///         Starting the interpreter and importing optuna costs about 1.3 s per command on a Windows runner, which
    ///         was most of this class's time (87 s for 13 tests there, against 36 s on Linux). The commands of one test
    ///         never depend on each other: each reads a fixture study, or writes a fresh database of its own.
    ///     </para>
    ///     <para>
    ///         OptunaSharp's side still runs in-process and one command at a time, because <see cref="RunInProcess" />
    ///         captures <see cref="Console.Out" />, which is process-wide. The checks are compared in the order they were
    ///         added, so the first mismatch reported is the one a sequential run reported.
    ///     </para>
    /// </remarks>
    private sealed class ParityBatch
    {
        /// <summary>Per check: the command as written (for the message), and each CLI's full arguments.</summary>
        private readonly List<(string[] Command, string[] Reference, string[] Ours)> _checks = new();

        /// <summary>Starts an empty batch, or skips the test when the reference CLI is missing.</summary>
        /// <exception cref="AssertInconclusiveException">The <c>optuna</c> CLI was not found.</exception>
        public ParityBatch()
        {
            // Before any setup work, as the per-check helpers this replaced skipped before touching a database.
            if (_optunaCli is null)
                Assert.Inconclusive("The `optuna` CLI was not found on PATH (set OPTUNASHARP_OPTUNA_CLI to enable).");
        }

        /// <summary>Adds a read command that both CLIs run against the same storage.</summary>
        /// <param name="storageUrl">The fixture storage both CLIs read.</param>
        /// <param name="command">The command and its options, without <c>--storage</c>.</param>
        public void Add(string storageUrl, params string[] command)
        {
            string[] full = WithStorage(command, storageUrl);
            _checks.Add((command, full, full));
        }

        /// <summary>
        ///     Adds a mutating command (ask, create-study), which cannot share a database: each CLI gets a fresh database,
        ///     prepared now by OptunaSharp with <paramref name="setup" />. The studies are identical and the samplers
        ///     seeded, so both CLIs print the same.
        /// </summary>
        /// <param name="setup">Commands run in-process on both fresh databases first; empty for none.</param>
        /// <param name="command">The command compared, without <c>--storage</c>.</param>
        public void AddFresh(string[] setup, string[] command)
        {
            string pyDb = FreshDbUrl("fresh-py");
            string csDb = FreshDbUrl("fresh-cs");
            if (setup.Length > 0)
            {
                RunInProcess(WithStorage(setup, pyDb));
                RunInProcess(WithStorage(setup, csDb));
            }
            _checks.Add((command, WithStorage(command, pyDb), WithStorage(command, csDb)));
        }

        /// <summary>Starts every reference process, then runs OptunaSharp's side and compares, check by check.</summary>
        /// <exception cref="AssertFailedException">A check's output differs from <c>optuna</c>'s.</exception>
        public void AssertAll()
        {
            // SQLite connections leaked by the in-process setup must be finalized (closed) before optuna opens the same
            // files, as the per-check helper did before each reference run.
            GC.Collect();
            GC.WaitForPendingFinalizers();

            var references = _checks.Select(c => RunOptunaAsync(c.Reference)).ToArray();
            for (int i = 0; i < _checks.Count; i++)
            {
                string ours = Normalize(RunInProcess(_checks[i].Ours));
                string optuna = Normalize(references[i].GetAwaiter().GetResult());
                ours.Should().Be(optuna, $"`optunasharp {string.Join(" ", _checks[i].Command)}` must match `optuna` exactly");
            }
        }
    }

    /// <summary>Appends <c>--storage <paramref name="storageUrl" /></c> to a command.</summary>
    /// <param name="args">The command and its options.</param>
    /// <param name="storageUrl">The storage URL or journal path.</param>
    /// <returns>A new array: <paramref name="args" />, then <c>--storage</c> and the URL.</returns>
    private static string[] WithStorage(string[] args, string storageUrl)
    {
        var full = new string[args.Length + 2];
        Array.Copy(args, full, args.Length);
        full[^2] = "--storage";
        full[^1] = storageUrl;
        return full;
    }

    private static string RunInProcess(params string[] args)
    {
        var original = Console.Out;
        var buffer = new StringWriter();
        Console.SetOut(buffer);
        try
        {
            CliProgram.Run(args);
        }
        finally
        {
            Console.SetOut(original);
        }
        return buffer.ToString();
    }

    /// <summary>
    ///     Runs the real <c>optuna</c> CLI as its own process and returns its stdout. Asynchronous so that a test's
    ///     reference commands run at the same time without holding a thread each (a few blocked pool threads would
    ///     have waited for the pool to grow on a 2-core runner).
    /// </summary>
    /// <param name="args">The full command line, <c>--storage</c> included.</param>
    /// <returns>Everything the process wrote to stdout; stderr is drained and ignored, as before.</returns>
    /// <exception cref="System.ComponentModel.Win32Exception">The <c>optuna</c> executable could not be started.</exception>
    private static async Task<string> RunOptunaAsync(string[] args)
    {
        var psi = new ProcessStartInfo(_optunaCli!)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            StandardOutputEncoding = Encoding.UTF8,
        };
        foreach (var a in args)
            psi.ArgumentList.Add(a);
        using var process = Process.Start(psi)!;
        // Both pipes drain at once: a child that fills the stderr pipe while only stdout is read would block forever.
        Task<string> stdout = process.StandardOutput.ReadToEndAsync();
        Task<string> stderr = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync().ConfigureAwait(false);
        await stderr.ConfigureAwait(false);
        return await stdout.ConfigureAwait(false);
    }

    private static string Normalize(string s) => s.Replace("\r\n", "\n").TrimEnd('\n');

    private static string FreshDbUrl(string tag)
    {
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-cli-{tag}-{Guid.NewGuid():N}.db");
        return "sqlite:///" + path.Replace("\\", "/");
    }

    private static string? FindOptunaCli()
    {
        string? overridePath = Environment.GetEnvironmentVariable("OPTUNASHARP_OPTUNA_CLI");
        if (!string.IsNullOrEmpty(overridePath))
            return overridePath;

        foreach (var candidate in new[] { "optuna", "optuna.exe" })
        {
            try
            {
                var psi = new ProcessStartInfo(candidate)
                {
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                };
                psi.ArgumentList.Add("--version");
                using var probe = Process.Start(psi);
                if (probe is null)
                    continue;
                probe.WaitForExit(15000);
                if (probe.HasExited && probe.ExitCode == 0)
                    return candidate;
            }
            catch
            {
                // not this candidate — try the next
            }
        }
        return null;
    }
}
