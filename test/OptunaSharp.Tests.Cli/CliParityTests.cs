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
    private static readonly string[] TextFormats = { "value", "json", "table" };

    [ClassInitialize]
    public static void ClassInitialize(TestContext _)
    {
        _optunaCli = FindOptunaCli();

        // Single-objective fixture: create-study + two seeded ask/tell trials (bit-exact RandomSampler).
        _soUrl = FreshDbUrl("so");
        const string floatSpace = "{\"x\": {\"name\": \"FloatDistribution\", \"attributes\": {\"low\": 0.0, \"high\": 1.0, \"log\": false, \"step\": null}}}";
        RunInProcess("create-study", "--study-name", "demo", "--storage", _soUrl);
        for (int i = 0; i < 2; i++)
        {
            RunInProcess("ask", "--study-name", "demo", "--storage", _soUrl,
                "--sampler", "RandomSampler", "--sampler-kwargs", $"{{\"seed\": {i}}}", "--search-space", floatSpace);
            RunInProcess("tell", "--study-name", "demo", "--storage", _soUrl, "--trial-number", i.ToString(), "--values", (0.5 + i).ToString("R"));
        }

        // Multi-objective fixture: exercises the integer `values` columns + the Pareto best-trials.
        _moUrl = FreshDbUrl("mo");
        RunInProcess("create-study", "--study-name", "mo", "--directions", "minimize", "maximize", "--storage", _moUrl);
        for (int i = 0; i < 3; i++)
        {
            RunInProcess("ask", "--study-name", "mo", "--storage", _moUrl,
                "--sampler", "RandomSampler", "--sampler-kwargs", $"{{\"seed\": {i}}}", "--search-space", floatSpace);
            RunInProcess("tell", "--study-name", "mo", "--storage", _moUrl, "--trial-number", i.ToString(),
                "--values", (0.1 * i).ToString("R"), (0.9 - 0.1 * i).ToString("R"));
        }

        // Release any leaked SQLite connections before the optuna subprocess reads the same files.
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }

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

    // ------------------------------------------------------------------ helpers

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
