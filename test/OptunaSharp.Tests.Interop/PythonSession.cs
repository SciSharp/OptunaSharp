using System;
using System.Diagnostics;
using System.IO;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     One embedded CPython engine per test process, used to run <b>real Optuna</b> in-process so the
///     interop suite can assert OptunaSharp matches it. CPython+pythonnet cannot re-initialize after
///     <c>Py_Finalize</c>, so the engine lifecycle is assembly-scoped, not per-test.
///     <para>
///     Discovery is best-effort: it honors <c>PYTHONNET_PYDLL</c> if set, otherwise probes a
///     <c>python</c>/<c>python3</c> on PATH for its shared library. If no engine can be started (or
///     Optuna is not installed), the engine is marked unavailable and engine-backed tests report
///     <see cref="Assert.Inconclusive(string)" /> rather than failing — unless
///     <c>OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE</c> is truthy (CI sets this after installing Python),
///     in which case an absent engine is a hard failure so a discovery break goes red, not green.
///     </para>
/// </summary>
[TestClass]
public static class PythonSession
{
    /// <summary>True once the engine started AND <c>import optuna</c> succeeded.</summary>
    public static bool Available { get; private set; }

    /// <summary>Why the engine is unavailable (diagnostic shown on Inconclusive), or null.</summary>
    public static string? Unavailability { get; private set; }

    /// <summary>The <c>optuna.__version__</c> of the discovered install, when available.</summary>
    public static string? OptunaVersion { get; private set; }

    [AssemblyInitialize]
    public static void Start(TestContext _)
    {
        try
        {
            ResolvePythonDll();
            if (!PythonEngine.IsInitialized)
            {
                PythonEngine.Initialize();
                PythonEngine.BeginAllowThreads();
            }

            using (Py.GIL())
            {
                using PyObject optuna = Py.Import("optuna");
                OptunaVersion = optuna.GetAttr("__version__").ToString();
            }

            Available = true;
        }
        catch (Exception ex)
        {
            Available = false;
            Unavailability = ex.Message;
        }
    }

    /// <summary>
    ///     Call at the top of any engine-backed test. Skips (Inconclusive) when the engine is
    ///     unavailable, unless <c>OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE</c> is truthy, where it fails.
    /// </summary>
    public static void EnsureOrInconclusive()
    {
        if (Available)
            return;

        string reason = $"Python/Optuna engine unavailable: {Unavailability}";
        if (RequireEngine())
            Assert.Fail(reason + " (OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE is set, so this is a hard failure).");
        Assert.Inconclusive(reason);
    }

    private static bool RequireEngine()
    {
        string? v = Environment.GetEnvironmentVariable("OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE");
        return v is "1" or "true" or "TRUE" or "yes";
    }

    /// <summary>
    ///     Point pythonnet at a libpython. Honors <c>PYTHONNET_PYDLL</c> (pythonnet reads it directly),
    ///     otherwise probes a python executable for its shared-library path and sets it explicitly.
    /// </summary>
    private static void ResolvePythonDll()
    {
        if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("PYTHONNET_PYDLL")))
            return; // pythonnet honors this itself.

        foreach (string exe in new[] { Environment.GetEnvironmentVariable("PYTHON"), "python3", "python" })
        {
            if (string.IsNullOrEmpty(exe))
                continue;

            string? dll = ProbeSharedLibrary(exe!);
            if (dll is not null && File.Exists(dll))
            {
                Runtime.PythonDLL = dll;
                return;
            }
        }
        // Fall through: let PythonEngine.Initialize() try its own resolution; if it can't, Start() catches it.
    }

    private static string? ProbeSharedLibrary(string exe)
    {
        const string script =
            "import sys,sysconfig,os\n" +
            "if os.name=='nt':\n" +
            "    p=os.path.join(sys.base_prefix,'python%d%d.dll'%sys.version_info[:2])\n" +
            "else:\n" +
            "    d=sysconfig.get_config_var('LIBDIR') or ''\n" +
            "    n=sysconfig.get_config_var('INSTSONAME') or sysconfig.get_config_var('LDLIBRARY') or ''\n" +
            "    p=os.path.join(d,n)\n" +
            "print(p)";

        try
        {
            var psi = new ProcessStartInfo(exe, "-c \"" + script.Replace("\"", "\\\"") + "\"")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using Process? p = Process.Start(psi);
            if (p is null)
                return null;
            string outp = p.StandardOutput.ReadToEnd();
            p.WaitForExit(15_000);
            return outp.Trim() is { Length: > 0 } s ? s : null;
        }
        catch
        {
            return null; // exe not found / not runnable — try the next candidate.
        }
    }
}
