using Python.Runtime;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     The OptunaHub-specific layer over the generic <see cref="PythonRuntime" /> backend — the cached
///     <c>optuna</c> / <c>optunahub</c> module handles, their discovered versions, and the log-quieting
///     the OptunaHub bridge needs. Kept OUT of <see cref="PythonRuntime" /> so that backend stays
///     library-agnostic (the BoTorch bridge layers its own accessors the same way). The module
///     accessors and <see cref="EnsureQuiet" /> are called under the GIL; the cached
///     <see cref="OptunaVersion" /> / <see cref="OptunaHubVersion" /> strings are readable without it
///     once their module has been touched.
/// </summary>
public static class OptunaHubRuntime
{
    /// <summary>Cached <c>optuna</c> module (per process). Records <see cref="OptunaVersion" />. Call under the GIL.</summary>
    public static PyObject Optuna
    {
        get
        {
            PyObject module = PythonRuntime.Import("optuna");
            OptunaVersion ??= ReadVersion(module);
            return module;
        }
    }

    /// <summary>Cached <c>optunahub</c> module (per process). Records <see cref="OptunaHubVersion" />. Call under the GIL.</summary>
    public static PyObject OptunaHub
    {
        get
        {
            PyObject module = PythonRuntime.Import("optunahub");
            OptunaHubVersion ??= ReadVersion(module);
            return module;
        }
    }

    /// <summary>The discovered <c>optuna.__version__</c> (or <c>null</c> before <see cref="Optuna" /> is first touched).</summary>
    public static string? OptunaVersion { get; private set; }

    /// <summary>The discovered <c>optunahub.__version__</c> (or <c>null</c> before <see cref="OptunaHub" /> is first touched).</summary>
    public static string? OptunaHubVersion { get; private set; }

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
        string? v = System.Environment.GetEnvironmentVariable("OPTUNASHARP_OPTUNAHUB_QUIET");
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

    private static string? ReadVersion(PyObject module)
    {
        try
        {
            using PyObject version = module.GetAttr("__version__");
            return version.As<string>();
        }
        catch
        {
            return null; // version is diagnostic only.
        }
    }
}

/// <summary>Raised when the OptunaHub bridge cannot be initialized or an OptunaHub Python call fails.</summary>
public sealed class OptunaHubInteropException : PythonInteropException
{
    public OptunaHubInteropException(string message) : base(message) { }
    public OptunaHubInteropException(string message, System.Exception inner) : base(message, inner) { }
}
