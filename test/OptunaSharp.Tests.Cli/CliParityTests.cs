using System;
using System.Diagnostics;
using System.IO;
using System.Text;
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
        foreach (var f in TextFormats)
            AssertParity(_soUrl, "study-names", "-f", f);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Studies_AllTextFormats_PlainAndFlatten()
    {
        foreach (var f in TextFormats)
        {
            AssertParity(_soUrl, "studies", "-f", f);
            AssertParity(_soUrl, "studies", "-f", f, "--flatten");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Trials_SingleObjective_AllTextFormats_PlainAndFlatten()
    {
        foreach (var f in TextFormats)
        {
            AssertParity(_soUrl, "trials", "--study-name", "demo", "-f", f);
            AssertParity(_soUrl, "trials", "--study-name", "demo", "-f", f, "--flatten");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void BestTrial_SingleObjective_AllTextFormats()
    {
        foreach (var f in TextFormats)
            AssertParity(_soUrl, "best-trial", "--study-name", "demo", "-f", f, "--flatten");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Trials_MultiObjective_ValuesColumns_AllTextFormats()
    {
        foreach (var f in TextFormats)
        {
            AssertParity(_moUrl, "trials", "--study-name", "mo", "-f", f);
            AssertParity(_moUrl, "trials", "--study-name", "mo", "-f", f, "--flatten");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void BestTrials_MultiObjective_ParetoFront_AllTextFormats()
    {
        foreach (var f in TextFormats)
            AssertParity(_moUrl, "best-trials", "--study-name", "mo", "-f", f);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Studies_WithUserAttrs_AllTextFormats_PlainAndFlatten()
    {
        foreach (var f in TextFormats)
        {
            AssertParity(_attrsUrl, "studies", "-f", f);
            AssertParity(_attrsUrl, "studies", "-f", f, "--flatten");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Trials_WithPrunedAndFailedTrials_AllTextFormats()
    {
        // The pruned/failed trials have a None objective value and no datetime_complete/duration.
        foreach (var f in TextFormats)
        {
            AssertParity(_attrsUrl, "trials", "--study-name", "attrs", "-f", f);
            AssertParity(_attrsUrl, "trials", "--study-name", "attrs", "-f", f, "--flatten");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void EmptyStudy_Trials_AllTextFormats()
    {
        foreach (var f in TextFormats)
            AssertParity(_emptyUrl, "trials", "--study-name", "empty", "-f", f);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void JournalStorage_StudiesAndTrials_AllTextFormats()
    {
        // Reads omit --storage-class: both CLIs must guess journal-file from the existing file.
        foreach (var f in TextFormats)
        {
            AssertParity(_journalPath, "studies", "-f", f);
            AssertParity(_journalPath, "trials", "--study-name", "j", "-f", f);
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Ask_SingleObjective_SeededSuggestion_AllTextFormats()
    {
        // A fresh, identical study on each side + a seeded RandomSampler => the suggested parameter is
        // bit-identical, so `ask`'s stdout must match exactly.
        string[] create = { "create-study", "--study-name", "q" };
        foreach (var f in new[] { "json", "value", "table" })
            AssertFreshStudyParity(create, new[]
            {
                "ask", "--study-name", "q", "--sampler", "RandomSampler", "--sampler-kwargs", "{\"seed\": 0}",
                "--search-space", FloatSpace, "-f", f,
            });
        AssertFreshStudyParity(create, new[]
        {
            "ask", "--study-name", "q", "--sampler", "RandomSampler", "--sampler-kwargs", "{\"seed\": 0}",
            "--search-space", FloatSpace, "-f", "json", "--flatten",
        });
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void CreateStudy_PrintsStudyName()
    {
        AssertFreshStudyParity(Array.Empty<string>(), new[] { "create-study", "--study-name", "made" });
        AssertFreshStudyParity(Array.Empty<string>(),
            new[] { "create-study", "--study-name", "mo2", "--directions", "minimize", "maximize" });
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
        AssertFreshStudyParity(
            new[] { "create-study", "--study-name", "recipe-demo" },
            new[]
            {
                "ask", "--study-name", "recipe-demo", "--sampler", "RandomSampler", "--sampler-kwargs", "{\"seed\": 0}",
                "--search-space", recipeSpace, "-f", "json",
            });

        // (2) create → ask → tell(--values 0.7 --state complete): OptunaSharp writes it; both CLIs read it identically.
        string url = FreshDbUrl("recipe-workflow");
        RunInProcess("create-study", "--study-name", "recipe-demo", "--storage", url);
        RunInProcess("ask", "--study-name", "recipe-demo", "--storage", url,
            "--sampler", "RandomSampler", "--sampler-kwargs", "{\"seed\": 0}", "--search-space", recipeSpace);
        RunInProcess("tell", "--study-name", "recipe-demo", "--storage", url,
            "--trial-number", "0", "--values", "0.7", "--state", "complete");
        GC.Collect();
        GC.WaitForPendingFinalizers();

        foreach (var f in TextFormats)
        {
            AssertParity(url, "trials", "--study-name", "recipe-demo", "-f", f);
            AssertParity(url, "best-trial", "--study-name", "recipe-demo", "-f", f);
        }
    }

    // ------------------------------------------------------------------ helpers

    // For mutating commands (ask/create-study) that cannot share a DB: run `setup` then `command` on a
    // fresh DB per engine, and compare the command's stdout. The engines produce identical results
    // because the study is fresh and identical and the RNG is seeded.
    private static void AssertFreshStudyParity(string[] setup, string[] command)
    {
        if (_optunaCli is null)
            Assert.Inconclusive("The `optuna` CLI was not found on PATH (set OPTUNASHARP_OPTUNA_CLI to enable).");

        string pyDb = FreshDbUrl("fresh-py");
        string csDb = FreshDbUrl("fresh-cs");
        if (setup.Length > 0)
        {
            RunInProcess(WithStorage(setup, pyDb));
            RunInProcess(WithStorage(setup, csDb));
        }
        GC.Collect();
        GC.WaitForPendingFinalizers();

        string optuna = Normalize(RunOptuna(WithStorage(command, pyDb)));
        string ours = Normalize(RunInProcess(WithStorage(command, csDb)));
        ours.Should().Be(optuna, $"`optunasharp {string.Join(" ", command)}` must match `optuna` exactly");
    }

    private static string[] WithStorage(string[] args, string storageUrl)
    {
        var full = new string[args.Length + 2];
        Array.Copy(args, full, args.Length);
        full[^2] = "--storage";
        full[^1] = storageUrl;
        return full;
    }

    private static void AssertParity(string storageUrl, params string[] command)
    {
        if (_optunaCli is null)
            Assert.Inconclusive("The `optuna` CLI was not found on PATH (set OPTUNASHARP_OPTUNA_CLI to enable).");

        var full = new string[command.Length + 2];
        Array.Copy(command, full, command.Length);
        full[^2] = "--storage";
        full[^1] = storageUrl;

        string optuna = Normalize(RunOptuna(full));
        string ours = Normalize(RunInProcess(full));
        ours.Should().Be(optuna, $"`optunasharp {string.Join(" ", command)}` must match `optuna` exactly");
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

    private static string RunOptuna(string[] args)
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
        string stdout = process.StandardOutput.ReadToEnd();
        process.StandardError.ReadToEnd();
        process.WaitForExit();
        return stdout;
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
