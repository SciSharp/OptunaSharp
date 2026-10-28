using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using AwesomeAssertions;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Python Optuna and OptunaSharp workers optimizing the same studies of ONE journal file at the same time, each
///     worker with its own storage, so the two languages exclude each other only through the journal's lock file.
///     The Python workers are threads of the embedded interpreter. They run while the C# workers do, because the
///     GIL is free between <c>Py.GIL()</c> blocks.
/// </summary>
/// <remarks>
///     The regression for the 2026-10 lock race. In a fleet of 6 Python and 6 C# processes on one file, every
///     Python worker finished while 4 of the 6 C# workers died of <c>IOException: … .lock already exists</c>:
///     Python reads contention from <c>errno</c>, and C# re-checked the path afterwards.
/// </remarks>
[TestClass]
public class JournalMixedFleetInteropTests
{
    // More C# workers than Python ones: the race this guards killed C# workers only, and Python threads share one GIL.
    private const int PythonWorkersPerStudy = 3;
    private const int CSharpWorkersPerStudy = 6;
    private const int TrialsPerWorker = 4;
    private static readonly string[] Studies = { "mixed-a", "mixed-b" };

    // A fresh journal per test; forward slashes keep the path a plain Python raw string on every OS.
    private string _path = null!;

    /// <summary>Picks a unique journal path in the temp directory.</summary>
    [TestInitialize]
    public void SetUp()
        => _path = Path.Combine(Path.GetTempPath(), $"optunasharp-mixed-{Guid.NewGuid():N}.log").Replace('\\', '/');

    /// <summary>Deletes the journal and its lock artifacts; a file still held open is left for the OS temp cleanup.</summary>
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

    /// <summary>
    ///     Two studies in one journal, each optimized at once by three Python and six C# workers: nobody fails, both
    ///     studies are perfect (contiguous numbers, every trial complete with its steps), both languages contributed
    ///     to each, and Python's replay is bit-identical to C#'s.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void PythonAndCSharpWorkers_OnOneJournalFile_ProduceIdenticalPerfectStudies()
    {
        PythonSession.EnsureOrInconclusive();
        // Optuna's default is the symlink lock; where symbolic links cannot be created (Windows without Developer
        // Mode) both languages use the open-file lock instead, which Python would need there too.
        bool symlinks = CanCreateSymlinks();
        Func<JournalFileBackend> backend = symlinks
            ? () => new JournalFileBackend(_path)
            : () => new JournalFileBackend(_path, new JournalFileOpenLock(_path));
        string pyLock = symlinks ? "None" : "optuna.storages.journal.JournalFileOpenLock(path)";

        var seed = new JournalStorage(backend());
        foreach (string name in Studies)
            Optuna.CreateStudy(storage: seed, studyName: name);

        // The module outlives the first GIL block: its threads keep running, and the check below joins them through
        // it. Like every PyObject it is touched, and finally disposed, only while holding the GIL.
        PyModule fleet;
        using (Py.GIL())
        {
            fleet = Py.CreateScope("mixed_fleet");
            fleet.Exec(
                "import struct, threading, optuna\n" +
                "optuna.logging.set_verbosity(optuna.logging.ERROR)\n" +
                $"path = r'{_path}'\n" +
                "errors = []\n" +
                "def objective(t):\n" +
                "    x = t.suggest_float('x', -10, 10)\n" +
                "    k = t.suggest_int('k', 0, 1000)\n" +
                "    for step in range(2):\n" +
                "        t.report(x * x + step, step)\n" +
                "    t.set_user_attr('worker', 'py')\n" +
                "    return x * x + k * 1e-3\n" +
                "def worker(study_name, seed):\n" +
                "    try:\n" +
                $"        backend = optuna.storages.journal.JournalFileBackend(path, lock_obj={pyLock})\n" +
                "        study = optuna.load_study(study_name=study_name, storage=optuna.storages.JournalStorage(backend),\n" +
                "            sampler=optuna.samplers.RandomSampler(seed=seed), pruner=optuna.pruners.NopPruner())\n" +
                $"        study.optimize(objective, n_trials={TrialsPerWorker})\n" +
                "    except BaseException as e:\n" +
                "        errors.append(repr(e))\n" +
                $"jobs = [(name, 500 + 10 * s + w) for s, name in enumerate({PyList(Studies)}) for w in range({PythonWorkersPerStudy})]\n" +
                "threads = [threading.Thread(target=worker, args=job) for job in jobs]\n" +
                "for t in threads:\n" +
                "    t.start()\n");
        }

        // The C# half of the fleet races the Python threads, which run whenever this process is not holding the GIL.
        var csErrors = new ConcurrentQueue<Exception>();
        var csThreads = Studies.SelectMany((name, s) => Enumerable.Range(0, CSharpWorkersPerStudy).Select(w => new Thread(() =>
        {
            try
            {
                var study = Optuna.LoadStudy(name, new JournalStorage(backend()),
                    sampler: new RandomSampler(700 + 10 * s + w), pruner: new NopPruner());
                study.Optimize(t =>
                {
                    double x = t.SuggestFloat("x", -10, 10);
                    long k = t.SuggestInt("k", 0, 1000);
                    for (int step = 0; step < 2; step++)
                        t.Report(x * x + step, step);
                    t.SetUserAttr("worker", "cs");
                    return x * x + k * 1e-3;
                }, nTrials: TrialsPerWorker);
            }
            catch (Exception e)
            {
                csErrors.Enqueue(e);
            }
        }))).ToList();
        csThreads.ForEach(t => t.Start());
        csThreads.ForEach(t => t.Join());

        string pyErrors;
        var pyDumps = new Dictionary<string, string>();
        using (Py.GIL())
        {
            try
            {
                // Join the Python workers (join releases the GIL while it waits), then replay the journal in Python.
                fleet.Exec(
                    "for t in threads:\n" +
                    "    t.join()\n" +
                    "def bits(v):\n" +
                    "    return struct.unpack('<q', struct.pack('<d', float(v)))[0]\n" +
                    "def dump(name):\n" +
                    "    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(path))\n" +
                    "    study = optuna.load_study(study_name=name, storage=storage)\n" +
                    "    return '\\n'.join(f\"{t.number}|{t.state.name}|{bits(t.value)}|{bits(t.params['x'])}|{t.params['k']}|{len(t.intermediate_values)}|{t.user_attrs['worker']}\" for t in sorted(study.trials, key=lambda t: t.number))\n");
                pyErrors = fleet.Eval("repr(errors)").As<string>();
                foreach (string name in Studies)
                    pyDumps[name] = fleet.Eval($"dump('{name}')").As<string>();
            }
            finally
            {
                fleet.Dispose();
            }
        }

        csErrors.Should().BeEmpty("no C# worker may die of lock contention");
        pyErrors.Should().Be("[]", "no Python worker may fail either");
        int perStudy = (PythonWorkersPerStudy + CSharpWorkersPerStudy) * TrialsPerWorker;
        foreach (string name in Studies)
        {
            var trials = Optuna.LoadStudy(name, new JournalStorage(backend())).Trials;
            trials.Select(t => t.Number).Should().Equal(Enumerable.Range(0, perStudy));
            trials.Should().OnlyContain(t => t.State == TrialState.Complete && t.IntermediateValues.Count == 2);
            trials.Count(t => (string)t.UserAttrs["worker"]! == "py").Should().Be(PythonWorkersPerStudy * TrialsPerWorker);
            trials.Count(t => (string)t.UserAttrs["worker"]! == "cs").Should().Be(CSharpWorkersPerStudy * TrialsPerWorker);
            Dump(trials).Should().Be(pyDumps[name], $"Python must replay study {name} exactly as C# does");
        }
    }

    /// <summary>The C# side of the canonical dump: one line per trial, floats as their raw 64-bit patterns.</summary>
    /// <param name="trials">Trials of one study, in number order.</param>
    /// <returns>The same text the Python <c>dump</c> function above produces for a replay of the same journal.</returns>
    private static string Dump(IEnumerable<FrozenTrial> trials)
    {
        var sb = new StringBuilder();
        foreach (var t in trials.OrderBy(t => t.Number))
        {
            if (sb.Length > 0)
                sb.Append('\n');
            sb.Append(CultureInfo.InvariantCulture,
                $"{t.Number}|{t.State.ToString().ToUpperInvariant()}|{BitConverter.DoubleToInt64Bits(t.Value!.Value)}|" +
                $"{BitConverter.DoubleToInt64Bits(Convert.ToDouble(t.Params["x"], CultureInfo.InvariantCulture))}|" +
                $"{Convert.ToInt64(t.Params["k"], CultureInfo.InvariantCulture)}|{t.IntermediateValues.Count}|{t.UserAttrs["worker"]}");
        }
        return sb.ToString();
    }

    /// <summary>A Python list literal of the given names.</summary>
    /// <param name="names">Plain identifiers without quotes.</param>
    /// <returns>For example <c>['a', 'b']</c>.</returns>
    private static string PyList(IEnumerable<string> names) => "[" + string.Join(", ", names.Select(n => $"'{n}'")) + "]";

    /// <summary>Whether this machine lets the test process create symbolic links next to the journal.</summary>
    /// <returns>False on Windows without Developer Mode or the symlink privilege, and on filesystems without links.</returns>
    private bool CanCreateSymlinks()
    {
        string probe = _path + ".symlink-probe";
        try
        {
            File.WriteAllText(_path + ".target-probe", "");
            File.CreateSymbolicLink(probe, _path + ".target-probe");
            File.Delete(probe);
            return true;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return false;
        }
        finally
        {
            File.Delete(_path + ".target-probe");
        }
    }
}
