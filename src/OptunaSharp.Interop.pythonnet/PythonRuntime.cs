using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Python.Runtime;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     Process-wide embedded-CPython session — the <b>generic backend</b> every OptunaSharp pythonnet
///     bridge is built on (OptunaHub, BoTorch, …). Owns discovery of a libpython, one-time
///     <see cref="PythonEngine" /> initialization, the GIL entry point every Python call goes through,
///     and a per-process cache of imported modules (<see cref="Import" />). It is deliberately
///     library-agnostic: consumers layer their own module accessors / Pythonic helpers on top (e.g.
///     <c>OptunaHubRuntime</c>, the BoTorch <c>Python</c> layer).
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

    private static readonly ConcurrentDictionary<string, PyObject> _importCache = new();

    /// <summary><c>true</c> once <see cref="PythonEngine" /> is running (started by us or the host).</summary>
    public static bool IsInitialized => PythonEngine.IsInitialized;

    /// <summary>
    ///     Ensure an embedded CPython engine is running, starting one if necessary. Idempotent and
    ///     thread-safe. Honors <c>PYTHONNET_PYDLL</c> (pythonnet reads it directly); otherwise probes a
    ///     <c>python</c>/<c>python3</c> (or <c>$PYTHON</c>) on PATH for its shared library. A host that
    ///     has already called <see cref="PythonEngine.Initialize()" /> is respected as-is.
    /// </summary>
    /// <exception cref="PythonInteropException">No usable libpython could be located/started.</exception>
    public static void Initialize()
    {
        if (Volatile.Read(ref _initialized) == 1 && PythonEngine.IsInitialized)
            return;

        lock (_gate)
        {
            if (PythonEngine.IsInitialized)
            {
                // A host-started engine still needs the hook: objectives may hold the GIL while calling Trial (P3-1).
                GilAwareMonitor.Install();
                Volatile.Write(ref _initialized, 1);
                return;
            }

            try
            {
                ResolvePythonDll();
                PythonEngine.Initialize();
                PythonEngine.BeginAllowThreads();
                InjectSitePackages();   // add the probed interpreter's site-packages to sys.path
                // From now on a thread holding the GIL releases it while it waits for an OptunaSharp sampler, pruner
                // or tell monitor; without this an objective inside Py.GIL() deadlocked at nJobs > 1 (P3-1).
                GilAwareMonitor.Install();
            }
            catch (Exception ex) when (ex is not PythonInteropException)
            {
                throw new PythonInteropException(
                    "Could not initialize an embedded CPython engine. Install a CPython with the packages " +
                    "your bridge needs, and if it cannot be found automatically set PYTHONNET_PYDLL to the " +
                    "libpython path (e.g. /usr/lib/x86_64-linux-gnu/libpython3.12.so, or " +
                    "C:\\...\\python312.dll).", ex);
            }

            Volatile.Write(ref _initialized, 1);
        }
    }

    /// <summary>Run an action under the GIL (initializing the engine first if needed).</summary>
    /// <param name="body">The action to invoke while holding the GIL. Dispose temporary Python handles before it returns.</param>
    /// <exception cref="PythonInteropException">The embedded interpreter cannot be initialized.</exception>
    /// <exception cref="Exception">The supplied action throws; callback exceptions propagate unchanged.</exception>
    public static void Run(Action body)
    {
        Initialize();
        using (Py.GIL())
            body();
    }

    /// <summary>Run a function under the GIL (initializing the engine first if needed) and return its result.</summary>
    /// <typeparam name="T">The managed result type, whose use must not require the GIL after this call.</typeparam>
    /// <param name="body">The function to invoke while holding the GIL.</param>
    /// <returns>The function's result. Internal bridge callers may transfer an owned Python handle only to code that preserves GIL discipline.</returns>
    /// <exception cref="PythonInteropException">The embedded interpreter cannot be initialized.</exception>
    /// <exception cref="Exception">The supplied function throws; callback exceptions propagate unchanged.</exception>
    public static T Run<T>(Func<T> body)
    {
        Initialize();
        using (Py.GIL())
            return body();
    }

    /// <summary>
    ///     Import (once) and cache an arbitrary Python module by name — e.g. <c>optuna</c>, <c>torch</c>,
    ///     <c>botorch.acquisition.logei</c>. The one generic module accessor every bridge shares. The
    ///     returned handle is process-scoped: use it as a plain local and <b>do not dispose it</b> (only
    ///     fresh <c>GetAttr</c>/<c>Invoke</c> results are disposable). Call under the GIL (from a
    ///     <see cref="Run(Action)" /> body).
    /// </summary>
    /// <exception cref="PythonInteropException">The module could not be imported.</exception>
    /// <param name="moduleName">The absolute Python import name.</param>
    /// <returns>A borrowed, process-scoped module handle; do not dispose it.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="moduleName" /> is null.</exception>
    public static PyObject Import(string moduleName)
    {
        if (_importCache.TryGetValue(moduleName, out PyObject? cached))
            return cached;

        PyObject module;
        try
        {
            module = Py.Import(moduleName);
        }
        catch (PythonException ex)
        {
            throw new PythonInteropException(
                $"Failed to `import {moduleName}` in the embedded interpreter. Install the package that " +
                "provides it into the Python that hosts this process.", ex);
        }

        // The GIL serializes callers, but guard the race anyway: dispose the loser, keep the winner.
        if (_importCache.TryAdd(moduleName, module))
            return module;
        module.Dispose();
        return _importCache[moduleName];
    }

    /// <summary>Cached <c>builtins</c> module (per process). Call under the GIL.</summary>
    public static PyObject Builtins => Import("builtins");

    /// <summary>
    ///     Point pythonnet at a libpython AND at the interpreter's site-packages. Honors
    ///     <c>PYTHONNET_PYDLL</c> when set (the user pinned the interpreter); otherwise probes a
    ///     <c>python</c> executable for its shared library and the directories it imports packages from,
    ///     which are then inserted into the embedded <c>sys.path</c> right after startup (see
    ///     <see cref="InjectSitePackages"/>) so it can import whatever that python can — venv /
    ///     <c>--user</c> site / conda included. Without that step, pythonnet can start from the base DLL
    ///     and miss where <c>pip install …</c> actually landed — the classic "Failed to import
    ///     &lt;pkg&gt; in the embedded interpreter" despite a successful install. (Setting <c>PYTHONPATH</c>
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
    /// <param name="exe">The candidate interpreter executable or path.</param>
    /// <returns>The discovered shared-library path and import directories, or null entries when probing fails.</returns>
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

/// <summary>Raised when the embedded CPython engine cannot be initialized or a Python call fails.</summary>
public class PythonInteropException : Exception
{
    /// <summary>Creates an interop failure with a diagnostic message.</summary>
    /// <param name="message">The explanation of the failure.</param>
    public PythonInteropException(string message) : base(message) { }
    /// <summary>Creates an interop failure that preserves its underlying cause.</summary>
    /// <param name="message">The explanation of the failure.</param>
    /// <param name="inner">The exception raised by Python or interpreter initialization.</param>
    public PythonInteropException(string message, Exception inner) : base(message, inner) { }
}
