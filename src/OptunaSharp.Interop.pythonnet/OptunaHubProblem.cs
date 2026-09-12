using System;
using System.Collections.Generic;
using Python.Runtime;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     Wraps a live OptunaHub benchmark (an <c>optunahub.benchmarks.BaseProblem</c> instance) as an
///     OptunaSharp objective. It exposes the problem's optimization <see cref="Directions" /> and
///     <see cref="SearchSpace" /> as OptunaSharp types, and an <see cref="Objective" /> delegate that an
///     OptunaSharp <see cref="Study" /> can optimize directly: the C# trial suggests each parameter from
///     the search space, then the Python <c>evaluate(params)</c> is called to score it.
///     <para>This mirrors what <c>BaseProblem.__call__</c> does in Python, but keeps the C# study in
///     control so any OptunaSharp sampler can be benchmarked against real hub problems.</para>
/// </summary>
public sealed class OptunaHubProblem : IDisposable
{
    private PyObject? _problem;

    /// <summary>Reads directions + search space eagerly (called under the GIL by <see cref="OptunaHubModule.CreateProblem" />).</summary>
    /// <param name="source">The registry package name retained for diagnostics.</param>
    /// <param name="problem">The borrowed Python problem or benchmark instance.</param>
    internal OptunaHubProblem(string source, PyObject problem)
    {
        Source = source;
        _problem = problem;
        Directions = ReadDirections(problem);
        SearchSpace = ReadSearchSpace(problem);
    }

    /// <summary>Where this problem came from (e.g. <c>"benchmarks/bbob"</c>).</summary>
    public string Source { get; }

    /// <summary>The problem's optimization directions (length &gt; 1 ⇒ multi-objective).</summary>
    public IReadOnlyList<StudyDirection> Directions { get; }

    /// <summary>The problem's search space, in Python declaration order.</summary>
    public OrderedDict<string, BaseDistribution> SearchSpace { get; }

    private PyObject Problem => _problem ?? throw new ObjectDisposedException(nameof(OptunaHubProblem));

    /// <summary>
    ///     A multi-objective objective for <see cref="Study.Optimize(Func{Trial, IReadOnlyList{double}}, int?, double?, int, Type[], IEnumerable{Action{Study, FrozenTrial}}, bool, bool)" />:
    ///     suggests each search-space parameter on the trial, then calls the Python <c>evaluate</c>.
    /// </summary>
    public Func<Trial, IReadOnlyList<double>> Objective => Evaluate;

    /// <summary>
    ///     A single-objective objective for <see cref="Study.Optimize(Func{Trial, double}, int?, double?, int, Type[], IEnumerable{Action{Study, FrozenTrial}}, bool, bool)" />.
    ///     Throws if the problem is multi-objective.
    /// </summary>
    /// <exception cref="OptunaSharp.Interop.PythonNet.OptunaHubInteropException">The benchmark declares more than one objective; use Objective to retain all values.</exception>
    public Func<Trial, double> SingleObjective
    {
        get
        {
            if (Directions.Count != 1)
                throw new OptunaHubInteropException(
                    $"'{Source}' is multi-objective ({Directions.Count} directions); use Objective, not SingleObjective.");
            return trial => Evaluate(trial)[0];
        }
    }

    /// <summary>Suggests each declared parameter in C# and evaluates the resulting configuration with the Python benchmark.</summary>
    /// <param name="trial">The trial being mirrored, sampled, or checked for optimizer compatibility.</param>
    /// <returns>One managed value per benchmark objective in its declared order.</returns>
    /// <exception cref="System.ObjectDisposedException">The benchmark wrapper has already been disposed.</exception>
    /// <exception cref="global::Python.Runtime.PythonException">The benchmark evaluation fails in Python.</exception>
    private IReadOnlyList<double> Evaluate(Trial trial)
    {
        // C# side first: suggest every parameter on the trial (no GIL needed here).
        var values = new OrderedDict<string, object?>();
        foreach (var (name, dist) in SearchSpace)
            values[name] = Suggest(trial, name, dist);

        // Then cross into Python to score.
        return PythonRuntime.Run(() =>
        {
            using var pyParams = new PyDict();
            foreach (var (name, value) in values)
                pyParams[name] = PyConvert.ToPy(value);

            using PyObject result = Python.optunahub.benchmarks.BaseProblem.evaluate(Problem, pyParams);
            return ReadValues(result);
        });
    }

    /// <summary>Dispatches a benchmark distribution to the matching typed OptunaSharp suggestion method.</summary>
    /// <param name="trial">The trial being mirrored, sampled, or checked for optimizer compatibility.</param>
    /// <param name="name">The Python attribute, parameter, or algorithm name required by the operation.</param>
    /// <param name="dist">The distribution defining value conversion or the borrowed Python distribution object.</param>
    /// <returns>The trial's suggested external value for the declared parameter.</returns>
    /// <exception cref="OptunaSharp.Interop.PythonNet.OptunaHubInteropException">The benchmark search space contains an unsupported distribution type.</exception>
    private static object? Suggest(Trial trial, string name, BaseDistribution dist) => dist switch
    {
        FloatDistribution f => trial.SuggestFloat(name, f.Low, f.High, f.Step, f.Log),
        IntDistribution i => trial.SuggestInt(name, i.Low, i.High, i.Step, i.Log),
        CategoricalDistribution c => trial.SuggestCategorical(name, c.Choices).Value,
        _ => throw new OptunaHubInteropException($"Unsupported distribution {dist.GetType().Name} in '{name}'."),
    };

    /// <summary>Copies the Python benchmark's ordered objective directions into managed values.</summary>
    /// <param name="problem">The borrowed Python problem or benchmark instance.</param>
    /// <returns>The benchmark's managed directions in objective-index order.</returns>
    private static IReadOnlyList<StudyDirection> ReadDirections(PyObject problem)
    {
        using var directions = Python.optunahub.benchmarks.BaseProblem.directions(problem);
        long n = directions.Length();
        var result = new List<StudyDirection>((int)n);
        for (long k = 0; k < n; k++)
        {
            using var d = directions[new PyInt(k)];
            result.Add(ParseDirection(d));
        }
        return result;
    }

    /// <summary>Recognizes a Python direction enum or textual name as an OptunaSharp direction.</summary>
    /// <param name="d">The borrowed Python optimization-direction enum or name.</param>
    /// <returns>The recognized managed optimization direction.</returns>
    /// <exception cref="OptunaSharp.Interop.PythonNet.OptunaHubInteropException">The Python direction is neither minimize nor maximize.</exception>
    private static StudyDirection ParseDirection(PyObject d)
    {
        // optuna.study.StudyDirection is an IntEnum; read its .name, falling back to str().
        string text;
        if (d.HasAttr("name"))
            using (var name = d.GetAttr("name"))
                text = name.As<string>() ?? "";
        else
            text = d.ToString() ?? "";

        text = text.ToUpperInvariant();
        if (text.Contains("MAXIMIZE"))
            return StudyDirection.Maximize;
        if (text.Contains("MINIMIZE"))
            return StudyDirection.Minimize;
        throw new OptunaHubInteropException($"Unrecognized optimization direction '{text}'.");
    }

    /// <summary>Copies the Python benchmark's ordered distribution mapping into OptunaSharp distributions.</summary>
    /// <param name="problem">The borrowed Python problem or benchmark instance.</param>
    /// <returns>The ordered managed search-space dictionary.</returns>
    private static OrderedDict<string, BaseDistribution> ReadSearchSpace(PyObject problem)
    {
        using var searchSpace = Python.optunahub.benchmarks.BaseProblem.search_space(problem);
        using var items = searchSpace.GetAttr("items").Invoke();
        using var iter = PyIter.GetIter(items);
        var result = new OrderedDict<string, BaseDistribution>();
        while (iter.MoveNext())
        {
            using PyObject kv = iter.Current;
            using var key = kv[new PyInt(0)];
            using var val = kv[new PyInt(1)];
            result[key.As<string>()!] = PyConvert.ToClrDistribution(val);
        }
        return result;
    }

    /// <summary>Converts a Python scalar or sequence into an ordered managed objective-value list.</summary>
    /// <param name="result">The borrowed Python scalar or objective-value sequence returned by evaluation.</param>
    /// <returns>A one-element list for a scalar, or the converted objective sequence.</returns>
    private static IReadOnlyList<double> ReadValues(PyObject result)
    {
        // A sequence (list/tuple/ndarray) has __len__; a python/numpy scalar does not.
        bool isSequence = result.HasAttr("__len__") && !IsString(result);
        if (!isSequence)
            return new[] { result.As<double>() };

        long n = result.Length();
        var values = new double[n];
        for (long k = 0; k < n; k++)
            using (var item = result[new PyInt(k)])
                values[k] = item.As<double>();
        return values;
    }

    /// <summary>Recognizes Python and NumPy strings so they are not treated as objective sequences.</summary>
    /// <param name="o">The borrowed Python object whose type is inspected.</param>
    /// <returns>Whether the object type is Python str or NumPy str_.</returns>
    private static bool IsString(PyObject o)
    {
        using var type = o.GetPythonType();
        using var name = type.GetAttr("__name__");
        return name.As<string>() is "str" or "str_";
    }

    /// <summary>Releases owned Python references while holding the GIL.</summary>
    public void Dispose()
    {
        var problem = _problem;
        if (problem is null)
            return;
        // Mark the wrapper unusable before native release; its objective delegates must not retain
        // permission to call a Python object whose owned reference has been released.
        _problem = null;
        PythonRuntime.Run(problem.Dispose);
    }
}
