using System;
using System.Collections.Generic;
using Python.Runtime;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     A loaded OptunaHub package (the Python module returned by <see cref="OptunaHub.LoadModule" />).
///     Use it to instantiate the package's samplers and benchmarks and wrap them for OptunaSharp:
///     <see cref="CreateSampler" /> yields an OptunaSharp <see cref="Samplers.BaseSampler" /> and
///     <see cref="CreateProblem" /> an OptunaSharp objective (via <see cref="OptunaHubProblem" />).
/// </summary>
public sealed class OptunaHubModule : IDisposable
{
    private PyObject? _module;

    /// <summary>Takes ownership of a successfully loaded Python package handle.</summary>
    /// <param name="package">The registry-relative package name used for diagnostics.</param>
    /// <param name="module">The owned module reference, released by <see cref="Dispose" /> under the GIL.</param>
    internal OptunaHubModule(string package, PyObject module)
    {
        Package = package;
        _module = module;
    }

    /// <summary>The package name this module was loaded from (e.g. <c>"benchmarks/bbob"</c>).</summary>
    public string Package { get; }

    private PyObject Module => _module ?? throw new ObjectDisposedException(nameof(OptunaHubModule));

    /// <summary>
    ///     Access the raw Python module for advanced use. The action runs under the GIL; do not let any
    ///     <see cref="PyObject" /> escape it.
    /// </summary>
    /// <param name="body">The action receiving a borrowed module handle; neither dispose nor retain that handle.</param>
    /// <exception cref="ObjectDisposedException">The wrapper has been disposed.</exception>
    /// <exception cref="PythonInteropException">The interpreter cannot be initialized.</exception>
    /// <exception cref="Exception">The supplied action throws; its exception propagates unchanged.</exception>
    public void WithRaw(Action<PyObject> body) => PythonRuntime.Run(() => body(Module));

    /// <summary>Whether the module defines an attribute (class/function) of the given name.</summary>
    /// <param name="name">The Python attribute name to test.</param>
    /// <returns>Whether the live module exposes the attribute.</returns>
    /// <exception cref="ObjectDisposedException">The wrapper has been disposed.</exception>
    /// <exception cref="PythonInteropException">The interpreter cannot be initialized.</exception>
    public bool HasAttr(string name) => PythonRuntime.Run(() => Module.HasAttr(name));

    /// <summary>
    ///     Instantiate a class from the package (e.g. a sampler or problem) with keyword arguments of
    ///     primitive CLR values, and wrap it as a live Python instance. Called under the GIL.
    /// </summary>
    /// <param name="className">The callable attribute to construct.</param>
    /// <param name="kwargs">Managed keyword arguments, or null for a parameterless construction.</param>
    /// <returns>A new owned Python reference; the caller must dispose it under the GIL.</returns>
    /// <exception cref="ObjectDisposedException">The wrapper has been disposed.</exception>
    /// <exception cref="PythonException">Attribute lookup, argument conversion, or the Python constructor fails.</exception>
    internal PyObject Instantiate(string className, Dictionary<string, object?>? kwargs)
    {
        using var cls = Module.GetAttr(className);
        if (kwargs is null || kwargs.Count == 0)
            return cls.Invoke();

        using var pyKwargs = new PyDict();
        foreach (var (key, value) in kwargs)
            pyKwargs[key] = PyConvert.ToPy(value);
        return cls.Invoke(Array.Empty<PyObject>(), pyKwargs);
    }

    /// <summary>
    ///     Instantiate a benchmark (<c>optunahub.benchmarks.BaseProblem</c> subclass) and wrap it as an
    ///     OptunaSharp objective. The default class name <c>"Problem"</c> matches most hub benchmarks.
    /// </summary>
    /// <param name="className">The problem class name in the package (default <c>"Problem"</c>).</param>
    /// <param name="kwargs">Constructor keyword arguments, e.g. <c>{ ["function_id"]=1, ["dimension"]=2 }</c>.</param>
    /// <returns>An owned benchmark wrapper; dispose it after its objectives are no longer used.</returns>
    /// <exception cref="ObjectDisposedException">The module has been disposed.</exception>
    /// <exception cref="PythonException">The class cannot be instantiated or its benchmark metadata cannot be read.</exception>
    /// <exception cref="PythonInteropException">The interpreter cannot initialize or benchmark metadata is unsupported.</exception>
    public OptunaHubProblem CreateProblem(
        string className = "Problem",
        Dictionary<string, object?>? kwargs = null)
        => PythonRuntime.Run(() => new OptunaHubProblem(Package, Instantiate(className, kwargs)));

    /// <summary>
    ///     Instantiate a sampler (<c>optuna.samplers.BaseSampler</c> / <c>optunahub.samplers.SimpleBaseSampler</c>
    ///     subclass) and wrap it as an OptunaSharp <see cref="Samplers.BaseSampler" /> that drives the
    ///     real Python sampler over a synchronized shadow study.
    /// </summary>
    /// <param name="className">The sampler class name in the package (e.g. <c>"AutoSampler"</c>, <c>"NelderMeadSampler"</c>).</param>
    /// <param name="kwargs">Constructor keyword arguments (e.g. <c>{ ["seed"]=0 }</c>).</param>
    /// <returns>An owned sampler wrapper, intended for a single-worker study.</returns>
    /// <exception cref="ObjectDisposedException">The module has been disposed.</exception>
    /// <exception cref="PythonException">The sampler class cannot be instantiated.</exception>
    /// <exception cref="PythonInteropException">The interpreter cannot be initialized.</exception>
    public OptunaHubSampler CreateSampler(
        string className,
        Dictionary<string, object?>? kwargs = null)
        => PythonRuntime.Run(() => new OptunaHubSampler($"{Package}:{className}", Instantiate(className, kwargs)));

    /// <summary>
    ///     Instantiate a pruner (<c>optuna.pruners.BasePruner</c> subclass) and wrap it as an OptunaSharp
    ///     <see cref="Pruners.BasePruner" /> that drives the real Python pruner over a synchronized
    ///     shadow study (single-objective).
    /// </summary>
    /// <param name="className">The pruner class name in the package (e.g. <c>"MedianPruner"</c>, <c>"HyperbandPruner"</c>).</param>
    /// <param name="kwargs">Constructor keyword arguments (e.g. <c>{ ["n_warmup_steps"]=5 }</c>).</param>
    /// <returns>An owned pruner wrapper, intended for a single-objective, single-worker study.</returns>
    /// <exception cref="ObjectDisposedException">The module has been disposed.</exception>
    /// <exception cref="PythonException">The pruner class cannot be instantiated.</exception>
    /// <exception cref="PythonInteropException">The interpreter cannot be initialized.</exception>
    public OptunaHubPruner CreatePruner(
        string className,
        Dictionary<string, object?>? kwargs = null)
        => PythonRuntime.Run(() => new OptunaHubPruner($"{Package}:{className}", Instantiate(className, kwargs)));

    /// <summary>Releases this wrapper's module reference under the GIL; repeated calls have no effect.</summary>
    /// <exception cref="PythonInteropException">The interpreter is unavailable when the owned reference is released.</exception>
    public void Dispose()
    {
        var module = _module;
        if (module is null)
            return;
        // Clear the managed ownership marker before releasing the native reference so repeated
        // disposal cannot release the same wrapper-owned reference again.
        _module = null;
        PythonRuntime.Run(module.Dispose);
    }
}
