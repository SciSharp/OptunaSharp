using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Hand-picked adversarial persistence cases — the values most likely to break a port:
///     extreme float bit patterns, NaN/±inf intermediates, unicode everywhere, None/bool/mixed
///     categoricals, deep attr trees. Each writes through one side's journal, replays on the
///     other, and asserts full-state parity plus targeted spot checks.
/// </summary>
[TestClass]
public class EdgeCaseParityTests : OptunaInteropTestBase
{
    private string _path = null!;

    [TestInitialize]
    public void EdgeInit()
        => _path = Path.Combine(Path.GetTempPath(), $"optunasharp-edge-{Guid.NewGuid():N}.log").Replace('\\', '/');

    [TestCleanup]
    public void EdgeCleanup()
    {
        foreach (var f in Directory.GetFiles(Path.GetDirectoryName(_path)!, Path.GetFileName(_path) + "*"))
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

    private JournalStorage NewCSharpStorage()
        => new(new JournalFileBackend(_path, lockObj: new JournalFileOpenLock(_path)));

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Extreme_Float_BitPatterns_Survive_CSharp_To_Python()
    {
        // Every nasty double: min subnormal, max finite, epsilon neighborhoods, -0.0, exact
        // powers, values with 17 significant digits. Written by C# as user attrs and
        // intermediates; Python must read the identical 8 bytes (checked via struct.pack).
        double[] nasty =
        {
            5e-324, -5e-324, 1.7976931348623157e308, -1.7976931348623157e308,
            2.2250738585072014e-308, 0.1, 1.0 / 3.0, 9999999999999998.0,
            1e16, 1e15 + 1, -0.0, 0.0, 123456789012345.6, Math.PI, Math.E,
            6.62607015e-34, 1.0000000000000002,
        };

        var storage = NewCSharpStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(1), studyName: "bits");
        for (int i = 0; i < nasty.Length; i++)
        {
            var trial = study.Ask();
            trial.SuggestFloat("x", -1, 1);
            trial.SetUserAttr("payload", nasty[i]);
            trial.Report(nasty[i], 0);
            study.Tell(trial, 1.0);
        }

        PyExec(
            $"storage = {PyJournalStorageExpr(_path)}\n" +
            "study = optuna.load_study(study_name='bits', storage=storage)\n" +
            "import struct\n" +
            "attr_bits = [struct.pack('<d', float(t.user_attrs['payload'])).hex() for t in study.trials]\n" +
            "iv_bits = [struct.pack('<d', float(t.intermediate_values[0])).hex() for t in study.trials]\n");

        using (Gil())
        {
            using var attrBits = Scope.Eval("attr_bits");
            using var ivBits = Scope.Eval("iv_bits");
            for (int i = 0; i < nasty.Length; i++)
            {
                string expectedHex = BitConverter.GetBytes(nasty[i]).Aggregate("", (s, b) => s + b.ToString("x2"));
                attrBits[i].As<string>().Should().Be(expectedHex, $"user attr bits of index {i}");
                ivBits[i].As<string>().Should().Be(expectedHex, $"intermediate bits of index {i}");
            }
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void NaN_And_Infinity_Intermediates_RoundTrip_Both_Directions()
    {
        // Python writes NaN/±inf intermediates (json NaN/Infinity literals) → C# replays.
        PyExec(
            $"storage = {PyJournalStorageExpr(_path)}\n" +
            "study = optuna.create_study(study_name='nan', storage=storage, " +
            "sampler=optuna.samplers.RandomSampler(seed=2))\n" +
            "t = study.ask()\n" +
            "t.suggest_float('x', 0, 1)\n" +
            "t.report(float('nan'), 0)\n" +
            "t.report(float('inf'), 1)\n" +
            "t.report(float('-inf'), 2)\n" +
            "study.tell(t, 1.0)\n");

        var study = Optuna.LoadStudy("nan", NewCSharpStorage(), sampler: new RandomSampler(3));
        var iv = study.Trials[0].IntermediateValues;
        double.IsNaN(iv[0]).Should().BeTrue("NaN intermediate must replay from Python's journal");
        iv[1].Should().Be(double.PositiveInfinity, "+inf intermediate");
        iv[2].Should().Be(double.NegativeInfinity, "-inf intermediate");

        // And back: C# appends the same shapes; Python must read them.
        var trial = study.Ask();
        trial.SuggestFloat("x", 0, 1);
        trial.Report(double.NaN, 0);
        trial.Report(double.PositiveInfinity, 1);
        study.Tell(trial, 2.0);

        PyExec(
            $"fresh = optuna.load_study(study_name='nan', storage={PyJournalStorageExpr(_path)})\n" +
            "t1 = fresh.trials[1]\n" +
            "ok = math.isnan(t1.intermediate_values[0]) and t1.intermediate_values[1] == float('inf')\n");
        PyBool("ok").Should().BeTrue("Python must parse C#'s NaN/Infinity journal literals");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Unicode_Everywhere_Survives_Both_Directions()
    {
        // Unicode study name, param values, attr keys and values — through C#'s writer.
        var storage = NewCSharpStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(5), studyName: "研究-étude-😀");
        study.SetUserAttr("ключ", "значение-🦄");
        study.Optimize(t =>
        {
            string c = t.SuggestCategorical("выбор", new[] { "日本語", "ünïcode", "emoji😀" });
            t.SetUserAttr("choice-copy", c);
            return c.Length;
        }, nTrials: 3);
        var csState = StudySnapshot.Capture(study);

        PyExec(
            $"storage = {PyJournalStorageExpr(_path)}\n" +
            "study = optuna.load_study(study_name='研究-étude-😀', storage=storage)\n");
        var pyState = PyStudyState("study");

        StudySnapshot.AssertParity(csState, pyState, DatetimeComparison.Instant);

        // Python appends with the same unicode space; C# replays.
        PyExec(
            "study.sampler = optuna.samplers.RandomSampler(seed=6)\n" +
            "study.optimize(lambda t: len(t.suggest_categorical('выбор', ['日本語', 'ünïcode', 'emoji😀'])), n_trials=2)\n");
        var reloaded = Optuna.LoadStudy("研究-étude-😀", NewCSharpStorage(), sampler: new RandomSampler(7));
        reloaded.Trials.Count.Should().Be(5);
        reloaded.UserAttrs["ключ"].Should().Be("значение-🦄");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Mixed_Categorical_Choices_Preserve_Python_Types()
    {
        // None/bool/int/float/str in ONE distribution: the type of each stored external must
        // survive C#→journal→Python exactly (None stays None, True stays bool, 1 stays int...).
        var choices = new CategoricalChoice[] { CategoricalChoice.None, true, false, 1L, 2.5, "s" };
        var storage = NewCSharpStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(8), studyName: "mixed");
        study.Optimize(t => { t.SuggestCategorical("c", choices); return 1.0; }, nTrials: 12);

        PyExec(
            $"storage = {PyJournalStorageExpr(_path)}\n" +
            "study = optuna.load_study(study_name='mixed', storage=storage)\n" +
            "kinds = [type(t.params['c']).__name__ for t in study.trials]\n");

        using (Gil())
        {
            using var kinds = Scope.Eval("kinds");
            for (int i = 0; i < study.Trials.Count; i++)
            {
                string expectedKind = study.Trials[i].Params["c"] switch
                {
                    null => "NoneType",
                    bool => "bool",
                    long => "int",
                    double => "float",
                    _ => "str",
                };
                kinds[i].As<string>().Should().Be(expectedKind, $"trial {i} categorical python type");
            }
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Deep_Attr_Trees_And_None_Values_RoundTrip()
    {
        var storage = NewCSharpStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(9), studyName: "deep");
        study.SetUserAttr("tree", new OrderedDict<string, object?>
        {
            ["list"] = new List<object?>
            {
                1L, 2.5, null, true,
                new OrderedDict<string, object?> { ["inner"] = new List<object?> { "x", 0.1 } },
            },
            ["none"] = null,
            ["empty_list"] = new List<object?>(),
            ["empty_dict"] = new OrderedDict<string, object?>(),
        });
        var trial = study.Ask();
        trial.SuggestFloat("x", 0, 1);
        study.Tell(trial, 1.0);
        var csState = StudySnapshot.Capture(study);

        PyExec(
            $"storage = {PyJournalStorageExpr(_path)}\n" +
            "study = optuna.load_study(study_name='deep', storage=storage)\n");
        StudySnapshot.AssertParity(csState, PyStudyState("study"), DatetimeComparison.Instant);

        PyBool("study.user_attrs['tree']['none'] is None").Should().BeTrue("None survives as None");
        PyBool("study.user_attrs['tree']['list'][3] is True").Should().BeTrue("bool survives as bool");
        PyLong("len(study.user_attrs['tree']['empty_list'])").Should().Be(0);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Single_And_Adjusted_Distributions_Replay_Identically()
    {
        // Single-value dists short-circuit sampling; step grids adjust high — the persisted
        // distribution JSON and the params must agree after a Python→C# replay.
        PyExec(
            $"storage = {PyJournalStorageExpr(_path)}\n" +
            "import warnings\n" +
            "with warnings.catch_warnings():\n" +
            "    warnings.simplefilter('ignore')\n" +
            "    study = optuna.create_study(study_name='single', storage=storage, " +
            "sampler=optuna.samplers.RandomSampler(seed=10))\n" +
            "    def objective(t):\n" +
            "        a = t.suggest_float('a', 3.5, 3.5)\n" +
            "        b = t.suggest_int('b', 7, 7)\n" +
            "        c = t.suggest_categorical('c', ['only'])\n" +
            "        d = t.suggest_float('d', 0.0, 1.0, step=0.3)\n" +
            "        e = t.suggest_int('e', 0, 100, step=7)\n" +
            "        return a + b + d + e\n" +
            "    study.optimize(objective, n_trials=4)\n");
        var pyState = PyStudyState("study");

        var csStudy = Optuna.LoadStudy("single", NewCSharpStorage(), sampler: new RandomSampler(11));
        StudySnapshot.AssertParity(pyState, StudySnapshot.Capture(csStudy), DatetimeComparison.Instant);

        // The adjusted-high grids replay into the same distribution objects.
        csStudy.Trials[0].Distributions["d"].Should().Be(new FloatDistribution(0.0, 0.9, step: 0.3));
        csStudy.Trials[0].Distributions["e"].Should().Be(new IntDistribution(0, 98, step: 7));
    }
}
