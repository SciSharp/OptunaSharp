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
    public void WithRaw(Action<PyObject> body) => PythonRuntime.Run(() => body(Module));

    /// <summary>Whether the module defines an attribute (class/function) of the given name.</summary>
    public bool HasAttr(string name) => PythonRuntime.Run(() => Module.HasAttr(name));

    /// <summary>
    ///     Instantiate a class from the package (e.g. a sampler or problem) with keyword arguments of
    ///     primitive CLR values, and wrap it as a live Python instance. Called under the GIL.
    /// </summary>
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
    public OptunaHubSampler CreateSampler(
        string className,
        Dictionary<string, object?>? kwargs = null)
        => PythonRuntime.Run(() => new OptunaHubSampler($"{Package}:{className}", Instantiate(className, kwargs)));

    public void Dispose()
    {
        var module = _module;
        if (module is null)
            return;
        _module = null;
        PythonRuntime.Run(module.Dispose);
    }
}
