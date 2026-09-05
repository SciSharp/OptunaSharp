using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Python.Runtime;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     Process-wide embedded-CPython session for the OptunaHub bridge — the analog of NumSharp's
///     <c>PythonRuntimeInterop</c>. Owns discovery of a libpython, one-time
///     <see cref="PythonEngine" /> initialization, the GIL entry point every Python call goes through,
///     and a per-session cache of the modules the bridge touches (<c>optuna</c>, <c>optunahub</c>,
///     <c>builtins</c>).
///     <para>
///     CPython+pythonnet cannot re-initialize after <c>Py_Finalize</c>, so the engine lifecycle is
///     process-scoped: <see cref="Initialize" /> is idempotent and there is no public shutdown.
///     </para>
///     <para><b>GIL discipline:</b> anything that touches a <see cref="PyObject" /> (including its
///     <c>Dispose</c>) must run under the GIL. Use <see cref="Run(Action)" /> / <see cref="Run{T}" />,
///     which acquire it for you; never hand a <see cref="PyObject" /> back out of a <c>Run</c> body.
///     </para>
/// </summary>
public static class PythonRuntime
{
    private static readonly object _gate = new();
    private static int _initialized;
    private static string? _sitePackages;   // discovered by ResolvePythonDll, injected after Initialize

    private static PyObject? _optuna, _optunahub, _builtins;

    /// <summary><c>true</c> once <see cref="PythonEngine" /> is running (started by us or the host).</summary>
    public static bool IsInitialized => PythonEngine.IsInitialized;

    /// <summary>
    ///     Ensure an embedded CPython engine is running, starting one if necessary. Idempotent and
    ///     thread-safe. Honors <c>PYTHONNET_PYDLL</c> (pythonnet reads it directly); otherwise probes a
    ///     <c>python</c>/<c>python3</c> (or <c>$PYTHON</c>) on PATH for its shared library. A host that
    ///     has already called <see cref="PythonEngine.Initialize()" /> is respected as-is.
    /// </summary>
    /// <exception cref="OptunaHubInteropException">No usable libpython could be located/started.</exception>
    public static void Initialize()
    {
        if (Volatile.Read(ref _initialized) == 1 && PythonEngine.IsInitialized)
            return;

        lock (_gate)
        {
            if (PythonEngine.IsInitialized)
            {
                Volatile.Write(ref _initialized, 1);
                return;
            }

            try
            {
                ResolvePythonDll();
                PythonEngine.Initialize();
                PythonEngine.BeginAllowThreads();
                InjectSitePackages();   // add the probed interpreter's site-packages to sys.path
                EnsureQuiet();
            }
            catch (Exception ex) when (ex is not OptunaHubInteropException)
            {
                throw new OptunaHubInteropException(
                    "Could not initialize an embedded CPython engine for the OptunaHub bridge. Install a " +
                    "CPython with `pip install optuna optunahub`, and if it cannot be found automatically " +
                    "set PYTHONNET_PYDLL to the libpython path (e.g. " +
                    "/usr/lib/x86_64-linux-gnu/libpython3.12.so, or C:\\...\\python312.dll).", ex);
            }

            Volatile.Write(ref _initialized, 1);
        }
    }

    /// <summary>
    ///     Lower Optuna's log verbosity to WARNING (the idiomatic quieting shown in Optuna's own docs).
    ///     The sampler bridge rebuilds an in-memory shadow study per trial, so INFO-level
    ///     "A new study created in memory" lines would otherwise flood the console. Idempotent and
    ///     best-effort; re-asserted right before each shadow study is created (Optuna re-configures its
    ///     root logger lazily on first real use, which can undo an early call). Opt out with
    ///     <c>OPTUNASHARP_OPTUNAHUB_QUIET=0</c>. Safe to call whether or not the GIL is already held
    ///     (<see cref="Py.GIL" /> is reentrant).
    /// </summary>
    internal static void EnsureQuiet()
    {
        string? v = Environment.GetEnvironmentVariable("OPTUNASHARP_OPTUNAHUB_QUIET");
        if (v is "0" or "false" or "FALSE" or "no" or "off")
            return;
        try
        {
            using (Py.GIL())
                PythonEngine.Exec("import optuna\noptuna.logging.set_verbosity(optuna.logging.WARNING)\n");
        }
        catch
        {
            // quieting is a convenience, never a reason to fail an operation.
        }
    }

    /// <summary>Run an action under the GIL (initializing the engine first if needed).</summary>
    public static void Run(Action body)
    {
        Initialize();
        using (Py.GIL())
            body();
    }

    /// <summary>Run a function under the GIL (initializing the engine first if needed) and return its result.</summary>
    public static T Run<T>(Func<T> body)
    {
        Initialize();
        using (Py.GIL())
            return body();
    }

    /// <summary>Cached <c>optuna</c> module (per process). Call under the GIL.</summary>
    public static PyObject Optuna => GetModule(ref _optuna, "optuna");

    /// <summary>Cached <c>optunahub</c> module (per process). Call under the GIL.</summary>
    public static PyObject OptunaHubModuleRef => GetModule(ref _optunahub, "optunahub");

    /// <summary>Cached <c>builtins</c> module (per process). Call under the GIL.</summary>
    public static PyObject Builtins => GetModule(ref _builtins, "builtins");

    /// <summary>The discovered <c>optuna.__version__</c> (or <c>null</c> before first use / on failure).</summary>
    public static string? OptunaVersion { get; private set; }

    /// <summary>The discovered <c>optunahub.__version__</c> (or <c>null</c> before first use / on failure).</summary>
    public static string? OptunaHubVersion { get; private set; }

    private static PyObject GetModule(ref PyObject? cache, string name)
    {
        var module = Volatile.Read(ref cache);
        if (module is not null)
            return module;

        try
        {
            module = Py.Import(name);
        }
        catch (PythonException ex)
        {
            throw new OptunaHubInteropException(
                $"Failed to `import {name}` in the embedded interpreter. Install it with " +
                $"`pip install {name}` into the Python that hosts this process.", ex);
        }

        var winner = Interlocked.CompareExchange(ref cache, module, null);
        if (winner is not null)
        {
            module.Dispose();
            return winner;
        }

        // Record versions opportunistically (cheap, under the GIL the caller already holds).
        try
        {
            if (name == "optuna")
                OptunaVersion = module.GetAttr("__version__").As<string>();
            else if (name == "optunahub")
                OptunaHubVersion = module.GetAttr("__version__").As<string>();
        }
        catch
        {
            // version is diagnostic only.
        }

        return module;
    }

    /// <summary>
    ///     Point pythonnet at a libpython AND at the interpreter's site-packages. Honors
    ///     <c>PYTHONNET_PYDLL</c> when set (the user pinned the interpreter); otherwise probes a
    ///     <c>python</c> executable for its shared library and the directories it imports packages from,
    ///     which are then inserted into the embedded <c>sys.path</c> right after startup (see
    ///     <see cref="InjectSitePackages"/>) so it can import whatever that python can — venv /
    ///     <c>--user</c> site / conda included. Without that step, pythonnet can start from the base DLL
    ///     and miss where <c>pip install optuna</c> actually landed — the classic "Failed to import
    ///     optuna in the embedded interpreter" despite a successful install. (Setting <c>PYTHONPATH</c>
    ///     doesn't help: an embeddable-distribution base — a <c>python3xx._pth</c> — ignores it, so the
    ///     paths must be pushed onto <c>sys.path</c> from inside the running interpreter.)
    /// </summary>
    private static void ResolvePythonDll()
    {
        if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("PYTHONNET_PYDLL")))
            return; // the user pinned the interpreter with PYTHONNET_PYDLL; honor it as-is.

        foreach (string? exe in new[] { Environment.GetEnvironmentVariable("PYTHON"), "python3", "python" })
        {
            if (string.IsNullOrEmpty(exe))
                continue;

            (string? dll, string? sitePackages) = ProbeEnvironment(exe!);
            if (dll is null || !File.Exists(dll))
                continue;

            Runtime.PythonDLL = dll;
            _sitePackages = sitePackages;   // pushed onto sys.path after Initialize (InjectSitePackages)
            return;
        }
        // Fall through: let PythonEngine.Initialize() try its own resolution.
    }

    /// <summary>
    ///     Push the probed interpreter's site-packages onto the LIVE <c>sys.path</c> (after
    ///     <see cref="PythonEngine.Initialize()"/>). This is what makes venv / <c>--user</c> / conda
    ///     packages importable when the loaded libpython is an embeddable base that ignores
    ///     <c>PYTHONPATH</c>. Best-effort and idempotent (skips entries already present).
    /// </summary>
    private static void InjectSitePackages()
    {
        if (string.IsNullOrEmpty(_sitePackages))
            return;
        try
        {
            using (Py.GIL())
            {
                using var sys = Py.Import("sys");
                using var sysPath = sys.GetAttr("path");
                string[] dirs = _sitePackages!.Split(Path.PathSeparator);
                // insert in reverse so the first probed dir ends up first (highest import priority)
                for (int i = dirs.Length - 1; i >= 0; i--)
                {
                    string dir = dirs[i];
                    if (string.IsNullOrEmpty(dir))
                        continue;
                    using var pyDir = new PyString(dir);
                    using var contains = sysPath.InvokeMethod("__contains__", pyDir);
                    if (contains.As<bool>())
                        continue; // already on sys.path — don't duplicate
                    using var zero = new PyInt(0);
                    using var _ = sysPath.InvokeMethod("insert", zero, pyDir);
                }
            }
        }
        catch
        {
            // best-effort — if we can't tweak sys.path, imports may still resolve from the base env.
        }
    }

    /// <summary>
    ///     Run <paramref name="exe"/> once to report (line 1) its libpython path and (line 2) its
    ///     site-packages directories joined by the platform path separator. Best-effort — a failure
    ///     just moves on to the next candidate.
    /// </summary>
    private static (string? Dll, string? SitePackages) ProbeEnvironment(string exe)
    {
        const string script =
            "import sys,sysconfig,os,site\n" +
            "if os.name=='nt':\n" +
            "    dll=os.path.join(sys.base_prefix,'python%d%d.dll'%sys.version_info[:2])\n" +
            "else:\n" +
            "    d=sysconfig.get_config_var('LIBDIR') or ''\n" +
            "    n=sysconfig.get_config_var('INSTSONAME') or sysconfig.get_config_var('LDLIBRARY') or ''\n" +
            "    dll=os.path.join(d,n)\n" +
            "paths=[]\n" +
            "try:\n" +
            "    paths+=list(site.getsitepackages())\n" +
            "except Exception:\n" +
            "    pass\n" +
            "try:\n" +
            "    if site.ENABLE_USER_SITE:\n" +
            "        paths.append(site.getusersitepackages())\n" +
            "except Exception:\n" +
            "    pass\n" +
            "uniq=[p for p in dict.fromkeys(paths) if p and os.path.isdir(p)]\n" +
            "print(dll)\n" +
            "print(os.pathsep.join(uniq))\n";

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
                return (null, null);
            string outp = p.StandardOutput.ReadToEnd();
            p.WaitForExit(15_000);

            string[] lines = outp.Replace("\r", "").Split('\n');
            string? dll = lines.Length > 0 && lines[0].Trim().Length > 0 ? lines[0].Trim() : null;
            string? sp = lines.Length > 1 && lines[1].Trim().Length > 0 ? lines[1].Trim() : null;
            return (dll, sp);
        }
        catch
        {
            return (null, null); // exe not found / not runnable — try the next candidate.
        }
    }
}

/// <summary>Raised when the embedded Python / OptunaHub bridge cannot be initialized or a Python call fails.</summary>
public sealed class OptunaHubInteropException : Exception
{
    public OptunaHubInteropException(string message) : base(message) { }
    public OptunaHubInteropException(string message, Exception inner) : base(message, inner) { }
}
