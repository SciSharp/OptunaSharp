using System;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     The Phase 1 exit gate, live: cross-language journal-file persistence with zero external
///     deps on either side. Python Optuna writes a journal → OptunaSharp replays and continues
///     it → Python replays the C#-written operations back. Same file, both directions.
/// </summary>
[TestClass]
public class JournalInteropTests
{
    private string _path = null!;

    [TestInitialize]
    public void SetUp()
        => _path = Path.Combine(Path.GetTempPath(), $"optunasharp-interop-{Guid.NewGuid():N}.log").Replace('\\', '/');

    [TestCleanup]
    public void TearDown()
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
    public void Python_Writes_CSharp_Continues_Python_ReadsBack()
    {
        PythonSession.EnsureOrInconclusive();

        // --- 1. Python writes 5 trials + attrs into the journal. -------------------------
        double pythonBest;
        using (Py.GIL())
        {
            using PyModule scope = Py.CreateScope();
            scope.Exec(
                "import optuna\n" +
                "optuna.logging.set_verbosity(optuna.logging.WARNING)\n" +
                $"path = r'{_path}'\n" +
                "storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(" +
                "path, lock_obj=optuna.storages.journal.JournalFileOpenLock(path)))\n" +
                "study = optuna.create_study(study_name='interop', storage=storage, " +
                "sampler=optuna.samplers.RandomSampler(seed=1))\n" +
                "def objective(t):\n" +
                "    x = t.suggest_float('x', -10, 10)\n" +
                "    n = t.suggest_int('n', 1, 5)\n" +
                "    c = t.suggest_categorical('c', [None, True, 'sgd'])\n" +
                "    t.report(x * 0.5, 0)\n" +
                "    return (x - 2) ** 2 + n\n" +
                "study.optimize(objective, n_trials=5)\n" +
                "study.set_user_attr('writer', 'python')\n" +
                "best = float(study.best_value)\n");
            pythonBest = scope.Get("best").As<double>();
        }

        // --- 2. OptunaSharp replays the journal and continues the study. -----------------
        var storage = NewCSharpStorage();
        var study = Optuna.LoadStudy("interop", storage, sampler: new RandomSampler(seed: 7));

        study.Trials.Should().HaveCount(5, "all Python trials must replay");
        study.BestValue.Should().Be(pythonBest, "the replayed best value must be bit-identical");
        study.UserAttrs["writer"].Should().Be("python");
        study.Trials[0].IntermediateValues.Should().ContainKey(0);
        // Python categorical externals map onto the CategoricalChoice model.
        var categorical = study.Trials.Select(t => t.Params["c"]).ToList();
        categorical.Should().OnlyContain(v => v == null || v!.Equals(true) || v!.Equals("sgd"));

        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            long n = t.SuggestInt("n", 1, 5);
            t.SuggestCategorical("c", new OptunaSharp.Distributions.CategoricalChoice[] { OptunaSharp.Distributions.CategoricalChoice.None, true, "sgd" });
            return Math.Pow(x - 2, 2) + n;
        }, nTrials: 5);
        study.SetUserAttr("continued_by", "csharp");
        double csharpBest = study.BestValue;
        study.Trials.Should().HaveCount(10);

        // --- 3. Python replays the C#-continued journal and keeps optimizing. ------------
        using (Py.GIL())
        {
            using PyModule scope = Py.CreateScope();
            scope.Exec(
                "import optuna, math\n" +
                "optuna.logging.set_verbosity(optuna.logging.WARNING)\n" +
                $"path = r'{_path}'\n" +
                "storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(" +
                "path, lock_obj=optuna.storages.journal.JournalFileOpenLock(path)))\n" +
                "study = optuna.load_study(study_name='interop', storage=storage)\n" +
                "n_trials = len(study.trials)\n" +
                "best = float(study.best_value)\n" +
                "attrs = dict(study.user_attrs)\n" +
                "t5 = study.trials[5]\n" +
                "t5_x = float(t5.params['x'])\n" +
                "t5_state = t5.state.name\n" +
                "t5_start_ok = t5.datetime_start is not None\n" +
                "all_finite = all(t.value is not None and math.isfinite(t.value) for t in study.trials)\n");

            scope.Get("n_trials").As<int>().Should().Be(10, "Python must see the C# trials");
            scope.Get("best").As<double>().Should().Be(csharpBest, "best value agrees bit-exactly across languages");
            scope.Get("t5_x").As<double>().Should().Be((double)study.Trials[5].Params["x"]!,
                "a C#-written param must replay bit-identically in Python");
            scope.Get("t5_state").As<string>().Should().Be("COMPLETE");
            scope.Get("t5_start_ok").As<bool>().Should().BeTrue("C#-written datetimes must parse in Python");
            scope.Get("all_finite").As<bool>().Should().BeTrue();

            using var attrs = scope.Get("attrs");
            attrs["writer"].As<string>().Should().Be("python");
            attrs["continued_by"].As<string>().Should().Be("csharp");
        }

        // --- 4. And the C# storage sees nothing new to invalidate its state. -------------
        var reReader = NewCSharpStorage();
        Optuna.LoadStudy("interop", reReader, sampler: new RandomSampler(2))
            .Trials.Should().HaveCount(10);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void CSharp_Writes_First_Python_Continues()
    {
        PythonSession.EnsureOrInconclusive();

        // C# creates the study and writes trials incl. WAITING (enqueue), attrs, intermediates.
        var storage = NewCSharpStorage();
        var study = Optuna.CreateStudy(storage: storage, studyName: "cs-first",
            sampler: new RandomSampler(seed: 3), direction: StudyDirection.Maximize);
        study.EnqueueTrial(new OptunaSharp.Collections.OrderedDict<string, object?> { ["x"] = 4.5 });
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 10);
            t.Report(x / 2, 0);
            t.Report(x, 1);
            return x;
        }, nTrials: 3);

        using (Py.GIL())
        {
            using PyModule scope = Py.CreateScope();
            scope.Exec(
                "import optuna\n" +
                "optuna.logging.set_verbosity(optuna.logging.WARNING)\n" +
                $"path = r'{_path}'\n" +
                "storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(" +
                "path, lock_obj=optuna.storages.journal.JournalFileOpenLock(path)))\n" +
                "study = optuna.load_study(study_name='cs-first', storage=storage)\n" +
                "direction = study.direction.name\n" +
                "first_params_x = float(study.trials[0].params['x'])\n" +
                "iv = dict(study.trials[0].intermediate_values)\n" +
                "study.optimize(lambda t: t.suggest_float('x', 0, 10), n_trials=2)\n" +
                "n_trials = len(study.trials)\n");

            scope.Get("direction").As<string>().Should().Be("MAXIMIZE");
            scope.Get("first_params_x").As<double>().Should().Be(4.5, "the C# enqueued fixed param");
            scope.Get("n_trials").As<int>().Should().Be(5);
            using var iv = scope.Get("iv");
            iv[new PyInt(1)].As<double>().Should().Be((double)study.Trials[0].Params["x"]!);
        }

        // C# picks the Python-added trials back up.
        Optuna.LoadStudy("cs-first", NewCSharpStorage(), sampler: new RandomSampler(4))
            .Trials.Should().HaveCount(5);
    }

    /// <summary>
    ///     Plan 13, item S8: attributes C# writes in batches (a trial's user and system attributes, a study's user and
    ///     system attributes) are ordinary single-key journal records, so live Python Optuna replays the file without
    ///     tripping its <c>assert len(log["user_attr"]) == 1</c>, sees every attribute with its value, and continues the
    ///     study in the same file.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void CSharp_WritesAttributeBatches_Python_ReadsEveryOne()
    {
        PythonSession.EnsureOrInconclusive();

        var storage = NewCSharpStorage();
        var study = Optuna.CreateStudy(storage: storage, studyName: "batches", sampler: new RandomSampler(seed: 5));
        var batch = new OptunaSharp.Collections.OrderedDict<string, object?>
        {
            ["seed"] = 7L, ["ratio"] = 0.25, ["name"] = "run", ["flag"] = true, ["none"] = null,
            ["list"] = new System.Collections.Generic.List<object?> { 1L, "two" },
        };
        Trial trial = study.Ask();
        trial.SuggestFloat("x", 0, 1);
        trial.SetUserAttrs(batch);
        storage.SetTrialSystemAttrs(trial.TrialId, new OptunaSharp.Collections.OrderedDict<string, object?> { ["sys"] = "s" });
        study.SetUserAttrs(batch);
        storage.SetStudySystemAttrs(study.StudyId, new OptunaSharp.Collections.OrderedDict<string, object?> { ["ssys"] = 1L });
        study.Tell(trial, 0.5);

        using (Py.GIL())
        {
            using PyModule scope = Py.CreateScope();
            scope.Exec(
                "import json, optuna\n" +
                "optuna.logging.set_verbosity(optuna.logging.WARNING)\n" +
                $"path = r'{_path}'\n" +
                "storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(" +
                "path, lock_obj=optuna.storages.journal.JournalFileOpenLock(path)))\n" +
                "study = optuna.load_study(study_name='batches', storage=storage)\n" +
                "t0 = study.trials[0]\n" +
                "trial_user = json.dumps(t0.user_attrs, sort_keys=True)\n" +
                "trial_sys = json.dumps(storage.get_trial_system_attrs(t0._trial_id), sort_keys=True)\n" +
                "study_user = json.dumps(study.user_attrs, sort_keys=True)\n" +
                "study_sys = json.dumps(storage.get_study_system_attrs(study._study_id), sort_keys=True)\n" +
                "study.optimize(lambda t: t.suggest_float('x', 0, 1), n_trials=2)\n" +
                "n_trials = len(study.trials)\n");

            string expected = "{\"flag\": true, \"list\": [1, \"two\"], \"name\": \"run\", \"none\": null, \"ratio\": 0.25, \"seed\": 7}";
            scope.Get("trial_user").As<string>().Should().Be(expected);
            scope.Get("study_user").As<string>().Should().Be(expected);
            scope.Get("trial_sys").As<string>().Should().Contain("\"sys\": \"s\"");
            scope.Get("study_sys").As<string>().Should().Contain("\"ssys\": 1");
            scope.Get("n_trials").As<int>().Should().Be(3);
        }

        Optuna.LoadStudy("batches", NewCSharpStorage()).Trials.Should().HaveCount(3);
    }
}
