using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
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

    /// <summary>
    ///     NumPy's AVX-512 dispatch levels (NumPy 2.4 naming), disabled for the live oracle unless the caller
    ///     already set <c>NPY_DISABLE_CPU_FEATURES</c>. On an AVX-512 x86-64 Linux host, NumPy routes float64
    ///     <c>power</c>/<c>exp</c>/<c>log</c> through SVML / AVX-512 kernels that differ from the C runtime's
    ///     libm in the last bit — while .NET's <see cref="Math" /> always uses that libm — so the bit-exact
    ///     parity tests went red on such CI runners (pymoo SBX/PM, AGE-MOEA-2, CMA-ES, the NSGA crossovers;
    ///     1-6 ULP) and green everywhere else. Windows NumPy builds never dispatch these levels (they only have
    ///     <c>X86_V3</c>), so this aligns Linux with what the Windows oracle already computes. A level the
    ///     installed NumPy does not dispatch only triggers an (ignored-by-default) <c>ImportWarning</c>.
    /// </summary>
    internal const string DisabledNumpyCpuFeatures = "X86_V4 AVX512_ICL AVX512_SPR";

    /// <summary>
    ///     Starts the shared engine once per test run and imports Optuna; on any failure records why and marks
    ///     the engine unavailable (engine-backed tests then go Inconclusive, or fail under
    ///     <c>OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE</c>).
    /// </summary>
    /// <param name="_">The MSTest assembly context (unused).</param>
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
                // Must run before ANY `import numpy` (NumPy reads the variable once, at import). It goes
                // through os.environ, i.e. putenv, because NumPy's C code calls getenv: on Unix, .NET's
                // Environment.SetEnvironmentVariable only updates a managed copy that getenv never sees.
                using (PyModule scope = Py.CreateScope())
                {
                    scope.Set("_npy_disabled", DisabledNumpyCpuFeatures);
                    scope.Exec("import os\nos.environ.setdefault('NPY_DISABLE_CPU_FEATURES', _npy_disabled)\n");
                }

                // pymoo 0.6.1.3, the oracle the native AGE-MOEA-2 reproduces generation by generation and the version the
                // pymoo bridge tests pin, calls np.row_stack (NSGA-II's crowding distance), which NumPy 2.5 removed; the
                // oracle NumPy is 2.5.4 since NumSharp 0.81.0 bundles its OpenBLAS. NumPy 2.0-2.4 defined row_stack as an
                // alias of vstack, so restoring that alias changes no result (pymoo 0.6.2 no longer calls it, but it also
                // changed AGE-MOEA-2's arithmetic, so it cannot be the oracle). Test environment only: the product's
                // pythonnet runtime leaves the user's NumPy untouched.
                using (PyModule scope = Py.CreateScope())
                    scope.Exec("import numpy\nif not hasattr(numpy, 'row_stack'):\n    numpy.row_stack = numpy.vstack\n");

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

    /// <summary>
    ///     <c>true</c> only where byte-identical parity to Optuna is reproducible: x86/x64. OptunaSharp's
    ///     bit-exact claim relies on .NET's <c>Math.*</c> resolving to the SAME libm as NumPy — which
    ///     holds on x64 (both call the platform ucrt/glibc) but NOT on arm64, where .NET ships its own
    ///     libm and diverges by ~1 ULP. (BLAS-backed paths add a second per-CPU dependence via OpenBLAS
    ///     DYNAMIC_ARCH.) So the pinned host for bit-exact tests is x64.
    /// </summary>
    public static bool BitExactHost =>
        RuntimeInformation.ProcessArchitecture is Architecture.X64 or Architecture.X86;

    /// <summary>
    ///     Gate for BIT-EXACT parity tests: requires the engine AND a bit-exact host, else Inconclusive
    ///     — mirroring NumSharp's <c>matmul_parity</c>/<c>linalg_parity</c> tiers ("Inconclusive off the
    ///     pinned host"). Tolerance-based tests (TPE erf-tol, GP achieved-value, importance ranking) use
    ///     <see cref="EnsureOrInconclusive" /> instead and run on every architecture.
    /// </summary>
    public static void EnsureBitExactHostOrInconclusive()
    {
        EnsureOrInconclusive();
        if (!BitExactHost)
            Assert.Inconclusive(
                "bit-exact parity is host-pinned to x64 (.NET's arm64 libm differs from NumPy's by " +
                $"~1 ULP); ProcessArchitecture={RuntimeInformation.ProcessArchitecture} reports " +
                "Inconclusive, matching NumSharp's matmul_parity host-pin model. Runs on x64/x86.");
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
