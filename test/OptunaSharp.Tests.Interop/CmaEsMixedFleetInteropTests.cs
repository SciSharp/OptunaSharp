using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     One CMA-ES study optimized by an OptunaSharp worker and a Python Optuna worker taking turns on one journal file,
///     each reading the optimizer state the other stored (review item P3-2). Until 2026-10-29 OptunaSharp stored its
///     own JSON there: every Python trial after a C# worker failed with <c>ValueError: non-hexadecimal number found in
///     fromhex()</c>, and every C# trial after a Python worker with an <c>ArgumentOutOfRangeException</c>. Now both
///     store Optuna's pickle.
/// </summary>
/// <remarks>
///     The turns are sequential and every sampler is seeded, so the fleet is deterministic. The reference is the same
///     fleet with Python in both seats: for the bit-exact backends (CMA, separable CMA, learning-rate adaptation) the
///     mixed study must equal it bit for bit, which holds only if each language continues the other's optimizer
///     exactly as its own. With-margin CMA-ES rides SciPy's error function in its margin correction, which OptunaSharp
///     reproduces to a few ULP, so that fleet is checked for complete studies and the same generations instead.
/// </remarks>
[TestClass]
public class CmaEsMixedFleetInteropTests
{
    // 4-trial turns against a population of 6 (two dimensions): over 48 trials the 7 tells fall to both languages,
    // and the newest stored state changes hands in both directions.
    private const int TrialsPerTurn = 4;
    private const int Turns = 12;
    private const int SeedFirst = 11;
    private const int SeedSecond = 22;

    private string _dir = null!;

    /// <summary>Creates a fresh directory for the fleet's journal files.</summary>
    [TestInitialize]
    public void SetUp()
    {
        _dir = Path.Combine(Path.GetTempPath(), $"optunasharp-cma-fleet-{Guid.NewGuid():N}").Replace('\\', '/');
        Directory.CreateDirectory(_dir);
    }

    /// <summary>Deletes the journal files; one still held open is left for the OS temp cleanup.</summary>
    [TestCleanup]
    public void TearDown()
    {
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    /// <summary>
    ///     A C# and a Python worker take turns on one CMA-ES study, the C# one first or second: every trial completes,
    ///     both languages store optimizer states and continue the other's, and the study equals, bit for bit, the one
    ///     two Python workers produce from the same seeds.
    /// </summary>
    /// <param name="variant"><c>cma</c>, <c>sepcma</c> (separable) or <c>lr_adapt</c> (learning-rate adaptation).</param>
    /// <param name="csharpFirst">Whether the C# worker takes the first turn (and the study's startup trial).</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("cma", true)]
    [DataRow("cma", false)]
    [DataRow("sepcma", true)]
    [DataRow("sepcma", false)]
    [DataRow("lr_adapt", true)]
    [DataRow("lr_adapt", false)]
    public void CSharpAndPythonWorkers_TakingTurns_RunTheStudyPythonRunsAlone(string variant, bool csharpFirst)
    {
        PythonSession.EnsureOrInconclusive();

        var reference = RunFleet(variant, "reference", firstIsCSharp: false, secondIsCSharp: false);
        var mixed = RunFleet(variant, "mixed", firstIsCSharp: csharpFirst, secondIsCSharp: !csharpFirst);

        mixed.Should().HaveCount(Turns * TrialsPerTurn).And.OnlyContain(t => t.State == TrialState.Complete);
        AssertStatesChangedHands(mixed, Prefix(variant));
        Dump(mixed, variant).Should().Equal(Dump(reference, variant),
            "each language must continue the other's CMA-ES state exactly as its own");
    }

    /// <summary>
    ///     The same fleet with <c>with_margin=True</c> (a float and an integer): every trial completes, both languages
    ///     continue the other's CMAwM state, and the study passes through the same generations at the same trials as
    ///     the Python-only one, with the float parameter equal to it within a relative 1e-9.
    /// </summary>
    /// <param name="csharpFirst">Whether the C# worker takes the first turn.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(true)]
    [DataRow(false)]
    public void CSharpAndPythonWorkers_TakingTurns_WithMargin_CompleteTheStudy(bool csharpFirst)
    {
        PythonSession.EnsureOrInconclusive();

        var reference = RunFleet("with_margin", "reference", firstIsCSharp: false, secondIsCSharp: false);
        var mixed = RunFleet("with_margin", "mixed", firstIsCSharp: csharpFirst, secondIsCSharp: !csharpFirst);

        mixed.Should().HaveCount(Turns * TrialsPerTurn).And.OnlyContain(t => t.State == TrialState.Complete);
        AssertStatesChangedHands(mixed, "cmawm:");
        mixed.Select(t => Generation(t, "cmawm:")).Should().Equal(reference.Select(t => Generation(t, "cmawm:")));
        mixed.Select(t => Convert.ToInt64(t.Params["n"], CultureInfo.InvariantCulture))
            .Should().Equal(reference.Select(t => Convert.ToInt64(t.Params["n"], CultureInfo.InvariantCulture)));
        for (int i = 0; i < reference.Count; i++)
        {
            double expected = (double)reference[i].Params["x"]!, actual = (double)mixed[i].Params["x"]!;
            Math.Abs(actual - expected).Should().BeLessThanOrEqualTo(1e-9 * Math.Max(1, Math.Abs(expected)), $"trial {i}");
        }
    }

    /// <summary>
    ///     Runs a two-seat fleet on a new journal: the first seat (seed 11) takes the even turns, the second (seed 22)
    ///     the odd ones, each seat a C# or a Python worker whose study, storage and sampler live across its turns.
    /// </summary>
    /// <param name="variant">The sampler variant: <c>cma</c>, <c>sepcma</c>, <c>lr_adapt</c> or <c>with_margin</c>.</param>
    /// <param name="name">Names the journal file within the test's directory.</param>
    /// <param name="firstIsCSharp">Whether the first seat is an OptunaSharp worker.</param>
    /// <param name="secondIsCSharp">Whether the second seat is an OptunaSharp worker.</param>
    /// <returns>The finished study's trials, read back through a C# storage.</returns>
    private IReadOnlyList<FrozenTrial> RunFleet(string variant, string name, bool firstIsCSharp, bool secondIsCSharp)
    {
        string path = $"{_dir}/{name}.log";
        // Every C# storage gets its own backend, as a separate process would (the file backend holds no handle open).
        JournalStorage Open() => new(new JournalFileBackend(path, new JournalFileOpenLock(path)));
        Optuna.CreateStudy(storage: Open(), studyName: "fleet");

        // Both Python seats share one module; its workers keep their study (and sampler state) between turns. Like
        // every PyObject it is touched, and finally disposed, only while holding the GIL.
        PyModule fleet;
        using (Py.GIL())
        {
            fleet = Py.CreateScope("cma_fleet");
            fleet.Exec(
                "import warnings, optuna\n" +
                "from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock\n" +
                "warnings.simplefilter('ignore')\n" +
                "optuna.logging.set_verbosity(optuna.logging.ERROR)\n" +
                $"path = r'{path}'\n" +
                $"{PyObjective(variant)}" +
                "workers = {}\n" +
                "def open_worker(seat, seed):\n" +
                "    storage = optuna.storages.JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))\n" +
                $"    workers[seat] = optuna.load_study(study_name='fleet', storage=storage, sampler={PySampler(variant)})\n" +
                "def run(seat, n):\n" +
                "    workers[seat].optimize(objective, n_trials=n)\n");
        }

        try
        {
            var seats = new[] { (CSharp: firstIsCSharp, Seed: SeedFirst), (CSharp: secondIsCSharp, Seed: SeedSecond) };
            var csStudies = new Study?[2];
            for (int s = 0; s < 2; s++)
            {
                if (seats[s].CSharp)
                {
                    csStudies[s] = Optuna.LoadStudy("fleet", Open(), sampler: CsSampler(variant, seats[s].Seed));
                }
                else
                {
                    using (Py.GIL())
                        fleet.Exec($"open_worker({s}, {seats[s].Seed})");
                }
            }

            for (int turn = 0; turn < Turns; turn++)
            {
                int seat = turn % 2;
                if (csStudies[seat] is { } study)
                {
                    study.Optimize(t => CsObjective(variant, t), nTrials: TrialsPerTurn);
                }
                else
                {
                    using (Py.GIL())
                        fleet.Exec($"run({seat}, {TrialsPerTurn})");
                }
            }
        }
        finally
        {
            using (Py.GIL())
                fleet.Dispose();
        }

        return Optuna.LoadStudy("fleet", Open()).Trials;
    }

    /// <summary>The C# sampler of a variant.</summary>
    /// <param name="variant">The sampler variant.</param>
    /// <param name="seed">The seat's seed.</param>
    /// <returns>A seeded <see cref="CmaEsSampler" />.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The variant is unknown.</exception>
    private static CmaEsSampler CsSampler(string variant, int seed) => variant switch
    {
        "cma" => new CmaEsSampler(seed: seed),
        "sepcma" => new CmaEsSampler(seed: seed, useSeparableCma: true),
        "lr_adapt" => new CmaEsSampler(seed: seed, lrAdapt: true),
        "with_margin" => new CmaEsSampler(seed: seed, withMargin: true),
        _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, "Unknown CMA-ES variant."),
    };

    /// <summary>The Python sampler expression of a variant, using the <c>seed</c> argument of <c>open_worker</c>.</summary>
    /// <param name="variant">The sampler variant.</param>
    /// <returns>The expression.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The variant is unknown.</exception>
    private static string PySampler(string variant) => variant switch
    {
        "cma" => "optuna.samplers.CmaEsSampler(seed=seed)",
        "sepcma" => "optuna.samplers.CmaEsSampler(seed=seed, use_separable_cma=True)",
        "lr_adapt" => "optuna.samplers.CmaEsSampler(seed=seed, lr_adapt=True)",
        "with_margin" => "optuna.samplers.CmaEsSampler(seed=seed, with_margin=True)",
        _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, "Unknown CMA-ES variant."),
    };

    /// <summary>The attribute prefix the variant's sampler stores under.</summary>
    /// <param name="variant">The sampler variant.</param>
    /// <returns><c>cma:</c>, <c>sepcma:</c> or <c>cmawm:</c>.</returns>
    private static string Prefix(string variant) => variant switch
    {
        "sepcma" => "sepcma:",
        "with_margin" => "cmawm:",
        _ => "cma:",
    };

    /// <summary>
    ///     The Python objective: two floats, or a float and an integer for the margin variant, tagged with the worker's
    ///     language. Float <c>**</c> is the C library's pow, as is <see cref="Math.Pow" />, so both values agree to the bit.
    /// </summary>
    /// <param name="variant">The sampler variant.</param>
    /// <returns>The Python source of <c>objective</c>.</returns>
    private static string PyObjective(string variant) => variant == "with_margin"
        ? "def objective(t):\n" +
          "    x = t.suggest_float('x', -5, 5)\n" +
          "    n = t.suggest_int('n', -3, 3)\n" +
          "    t.set_user_attr('worker', 'py')\n" +
          "    return (x - 1) ** 2 + (n - 1) ** 2\n"
        : "def objective(t):\n" +
          "    x = t.suggest_float('x', -5, 5)\n" +
          "    y = t.suggest_float('y', -5, 5)\n" +
          "    t.set_user_attr('worker', 'py')\n" +
          "    return (x - 1) ** 2 + (y + 0.5) ** 2\n";

    /// <summary>The C# objective, operation for operation the Python one.</summary>
    /// <param name="variant">The sampler variant.</param>
    /// <param name="t">The running trial.</param>
    /// <returns>The objective value.</returns>
    private static double CsObjective(string variant, Trial t)
    {
        double x = t.SuggestFloat("x", -5, 5);
        if (variant == "with_margin")
        {
            long n = t.SuggestInt("n", -3, 3);
            t.SetUserAttr("worker", "cs");
            return Math.Pow(x - 1, 2) + (n - 1) * (n - 1);
        }
        double y = t.SuggestFloat("y", -5, 5);
        t.SetUserAttr("worker", "cs");
        return Math.Pow(x - 1, 2) + Math.Pow(y + 0.5, 2);
    }

    /// <summary>
    ///     Asserts that both languages stored optimizer states and that the newest state passed from C# to Python and
    ///     from Python to C#, so each language continued a state the other had written.
    /// </summary>
    /// <param name="trials">The mixed study's trials.</param>
    /// <param name="prefix">The sampler's attribute prefix.</param>
    private static void AssertStatesChangedHands(IReadOnlyList<FrozenTrial> trials, string prefix)
    {
        var writers = trials
            .Where(t => t.SystemAttrs.ContainsKey(prefix + "optimizer:0"))
            .Select(t => (string)t.UserAttrs["worker"]!)
            .ToList();
        var handovers = writers.Zip(writers.Skip(1), (a, b) => $"{a}->{b}").ToHashSet();

        writers.Should().Contain("cs").And.Contain("py", "both languages must have told the optimizer");
        handovers.Should().Contain("cs->py").And.Contain("py->cs", "each language must continue a state the other stored");
    }

    /// <summary>The generation a trial was sampled in, or -1 for the independent startup trial.</summary>
    /// <param name="t">A trial.</param>
    /// <param name="prefix">The sampler's attribute prefix.</param>
    /// <returns>The generation.</returns>
    private static long Generation(FrozenTrial t, string prefix)
        => t.SystemAttrs.TryGetValue(prefix + "generation", out var g) ? Convert.ToInt64(g, CultureInfo.InvariantCulture) : -1;

    /// <summary>One line per trial: number, state, parameters, value and generation, floats as their raw bits.</summary>
    /// <param name="trials">A finished fleet study's trials.</param>
    /// <param name="variant">The sampler variant, for its prefix.</param>
    /// <returns>The lines, in trial order.</returns>
    private static List<string> Dump(IReadOnlyList<FrozenTrial> trials, string variant)
    {
        string prefix = Prefix(variant);
        var lines = new List<string>();
        foreach (var t in trials)
        {
            var sb = new StringBuilder();
            sb.Append(CultureInfo.InvariantCulture, $"{t.Number}|{t.State}|{Generation(t, prefix)}|{BitConverter.DoubleToInt64Bits(t.Value!.Value):x16}");
            foreach (var (key, value) in t.Params.OrderBy(kv => kv.Key, StringComparer.Ordinal))
                sb.Append(CultureInfo.InvariantCulture, $"|{key}={BitConverter.DoubleToInt64Bits(Convert.ToDouble(value, CultureInfo.InvariantCulture)):x16}");
            lines.Add(sb.ToString());
        }
        return lines;
    }
}
