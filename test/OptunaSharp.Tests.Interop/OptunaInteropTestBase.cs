using System;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Shared plumbing for engine-backed interop tests — the OptunaSharp analogue of
///     NumSharp's <c>InteropTestBase</c>. Each test gets a persistent <see cref="PyModule" />
///     scope with optuna imported and quieted, plus typed one-line helpers so tests read as
///     interleaved Python/C#:
///     <code>
///     PyExec("study = optuna.create_study(...)");
///     PyFloat("study.best_value").Should().Be(csStudy.BestValue);
///     </code>
///
///     <para><b>GIL discipline:</b> pythonnet 3.0.x requires the GIL for
///     <c>PyObject.Dispose</c>, so every helper creates AND disposes its PyObjects inside its
///     own <see cref="Py.GIL" /> scope; tests handle plain CLR values only unless they
///     explicitly open <see cref="Gil" />.</para>
/// </summary>
public abstract class OptunaInteropTestBase
{
    protected PyModule Scope = null!;

    [TestInitialize]
    public void InteropInit()
    {
        PythonSession.EnsureOrInconclusive();

        using (Py.GIL())
        {
            Scope = Py.CreateScope();
            Scope.Exec(
                "import optuna, json, math\n" +
                "optuna.logging.set_verbosity(optuna.logging.WARNING)\n");
        }
    }

    [TestCleanup]
    public void InteropCleanup()
    {
        if (Scope is null)
            return;  // Engine unavailable — the test was Inconclusive.

        using (Py.GIL())
            Scope.Dispose();
        Scope = null!;
    }

    // ---- python helpers (each opens/closes its own GIL scope) ----------------------------

    protected IDisposable Gil() => Py.GIL();

    protected void PyExec(string code)
    {
        using (Py.GIL())
            Scope.Exec(code);
    }

    protected long PyLong(string expr)
    {
        using (Py.GIL())
        {
            using var r = Scope.Eval(expr);
            return r.As<long>();
        }
    }

    protected double PyFloat(string expr)
    {
        using (Py.GIL())
        {
            using var r = Scope.Eval(expr);
            return r.As<double>();
        }
    }

    protected string PyStr(string expr)
    {
        using (Py.GIL())
        {
            using var r = Scope.Eval($"str({expr})");
            return r.As<string>();
        }
    }

    protected bool PyBool(string expr)
    {
        using (Py.GIL())
        {
            using var r = Scope.Eval($"bool({expr})");
            return r.As<bool>();
        }
    }

    /// <summary>Bind a CLR string into the scope (no source-escaping needed — the safe way to pass JSON).</summary>
    protected void PySet(string name, string value)
    {
        using (Py.GIL())
        {
            using var s = new PyString(value);
            Scope.Set(name, s);
        }
    }

    /// <summary>
    ///     Python source of <c>_oss_dump_state(study)</c> — serializes the COMPLETE study state
    ///     (directions, attrs and every trial field, floats via repr so bits survive) to a JSON
    ///     string the C# side parses with PyJson into a <see cref="StudySnapshot" />.
    /// </summary>
    protected const string DumpStateFunctionSource =
        "def _oss_dump_state(study):\n" +
        "    def dt(x):\n" +
        "        return None if x is None else x.isoformat(timespec='microseconds')\n" +
        "    trials = []\n" +
        "    for t in study.get_trials(deepcopy=False):\n" +
        "        trials.append({\n" +
        "            'number': t.number,\n" +
        "            'state': int(t.state),\n" +
        "            'values': None if t.values is None else [float(v) for v in t.values],\n" +
        "            'params': dict(t.params),\n" +
        "            'params_internal': {k: float(t.distributions[k].to_internal_repr(v)) for k, v in t.params.items()},\n" +
        "            'distributions': {k: optuna.distributions.distribution_to_json(d) for k, d in t.distributions.items()},\n" +
        "            'user_attrs': dict(t.user_attrs),\n" +
        "            'system_attrs': dict(t.system_attrs),\n" +
        "            'intermediate_values': {str(k): float(v) for k, v in t.intermediate_values.items()},\n" +
        "            'datetime_start': dt(t.datetime_start),\n" +
        "            'datetime_complete': dt(t.datetime_complete),\n" +
        "        })\n" +
        "    return json.dumps({\n" +
        "        'study_name': study.study_name,\n" +
        "        'directions': [int(d) for d in study.directions],\n" +
        "        'user_attrs': dict(study.user_attrs),\n" +
        "        'system_attrs': dict(study._storage.get_study_system_attrs(study._study_id)),\n" +
        "        'trials': trials,\n" +
        "    })\n";

    /// <summary>Dump the full state of the Python study bound to <paramref name="studyVar" />.</summary>
    protected StudySnapshot PyStudyState(string studyVar)
    {
        using (Py.GIL())
        {
            Scope.Exec(DumpStateFunctionSource);
            using var r = Scope.Eval($"_oss_dump_state({studyVar})");
            return StudySnapshot.ParsePythonJson(r.As<string>());
        }
    }

    /// <summary>The journal storage construction expression used on the Python side.</summary>
    protected static string PyJournalStorageExpr(string path)
        => "optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(" +
           $"r'{path}', lock_obj=optuna.storages.journal.JournalFileOpenLock(r'{path}')))";
}
