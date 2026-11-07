using System;
using System.Collections.Generic;
using System.Globalization;
using Python.Runtime;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     Wraps a live OptunaHub benchmark (an <c>optunahub.benchmarks.BaseProblem</c> instance) as an
///     OptunaSharp objective. It exposes the problem's optimization <see cref="Directions" /> and
///     <see cref="SearchSpace" /> as OptunaSharp types, and an <see cref="Objective" /> delegate that an
///     OptunaSharp <see cref="Study" /> can optimize directly: the C# trial suggests each parameter from
///     the search space, the problem's constraints are recorded on the trial, then the Python
///     <c>evaluate(params)</c> is called to score it.
///     <para>This mirrors what <c>BaseProblem.__call__</c> does in Python (OptunaHub 0.5.0), but keeps the
///     C# study in control so any OptunaSharp sampler can be benchmarked against real hub problems.</para>
///     <para><b>Constraints (OptunaHub ≥ 0.5.0):</b> the problem's <c>evaluate_constraints(params)</c> is
///     applied through <see cref="Trial.SetConstraint" /> <i>before</i> the problem is evaluated — so a
///     constrained-aware sampler sees feasibility, and a failing <c>evaluate</c> still leaves the
///     constraints recorded — exactly the order of <c>__call__</c>. A legacy sequence return gets the
///     constraint names <c>"0"</c>, <c>"1"</c>, … and OptunaHub's own <c>FutureWarning</c>. With an older
///     OptunaHub installed (no <c>BaseProblem.evaluate_constraints</c>), nothing is applied, matching that
///     version's <c>__call__</c>.</para>
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
    ///     A multi-objective objective for <see cref="Study.Optimize(Func{Trial, IReadOnlyList{double}}, int?, double?, int, Type[], IEnumerable{Action{Study, FrozenTrial}}, bool, bool, TrialExecutorOptions, System.Threading.CancellationToken)" />:
    ///     suggests each search-space parameter on the trial, records the problem's constraints on it, then
    ///     calls the Python <c>evaluate</c>.
    /// </summary>
    public Func<Trial, IReadOnlyList<double>> Objective => Evaluate;

    /// <summary>
    ///     A single-objective objective for <see cref="Study.Optimize(Func{Trial, double}, int?, double?, int, Type[], IEnumerable{Action{Study, FrozenTrial}}, bool, bool, TrialExecutorOptions, System.Threading.CancellationToken)" />.
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

    /// <summary>
    ///     Suggests each declared parameter in C#, records the problem's constraints on the trial, and
    ///     evaluates the resulting configuration with the Python benchmark — OptunaHub 0.5.0's
    ///     <c>BaseProblem.__call__</c>, step for step.
    /// </summary>
    /// <param name="trial">The trial being mirrored, sampled, or checked for optimizer compatibility.</param>
    /// <returns>One managed value per benchmark objective in its declared order.</returns>
    /// <exception cref="System.ObjectDisposedException">The benchmark wrapper has already been disposed.</exception>
    /// <exception cref="global::Python.Runtime.PythonException">The benchmark's constraint evaluation or evaluation fails in Python, or a warnings filter escalates the legacy-sequence <c>FutureWarning</c>.</exception>
    /// <exception cref="NumSharp.TypeError">A constraint value cannot be converted to <c>float</c> (Optuna's <c>set_constraint</c> message).</exception>
    /// <exception cref="NumSharp.ValueError">A constraint value is NaN, which <see cref="Trial.SetConstraint" /> rejects.</exception>
    private IReadOnlyList<double> Evaluate(Trial trial)
    {
        // C# side first: suggest every parameter on the trial (no GIL needed here).
        var values = new OrderedDict<string, object?>();
        foreach (var (name, dist) in SearchSpace)
            values[name] = Suggest(trial, name, dist);

        // One params dict is shared by evaluate_constraints and evaluate, exactly as in __call__, so a
        // problem that reads (or even mutates) it sees the same object in both calls. It is created and
        // disposed under the GIL; the constraints are applied in between, on the C# side.
        PyDict? pyParams = null;
        try
        {
            IReadOnlyList<KeyValuePair<string, double>> constraints = PythonRuntime.Run(() =>
            {
                pyParams = new PyDict();
                foreach (var (name, value) in values)
                    pyParams[name] = PyConvert.ToPy(value);
                return EvaluateConstraints(Problem, pyParams);
            });

            // __call__ applies the constraints BEFORE evaluate, so they stay recorded even when the
            // evaluation raises. Trial.SetConstraint is the port of Optuna's set_constraint (NaN check,
            // already-set warning, storage write); it never needs the GIL.
            foreach (var (key, value) in constraints)
                trial.SetConstraint(key, value);

            // Then cross into Python to score.
            return PythonRuntime.Run(() =>
            {
                using PyObject result = Python.optunahub.benchmarks.BaseProblem.evaluate(Problem, pyParams!);
                return ReadValues(result);
            });
        }
        finally
        {
            // A PyObject must be released under the GIL; the dict is ours, so dispose it even on failure.
            if (pyParams is not null)
                PythonRuntime.Run(pyParams.Dispose);
        }
    }

    /// <summary>
    ///     The verbatim <c>FutureWarning</c> text OptunaHub 0.5.0's <c>_as_constraints_dict</c> raises when
    ///     <c>evaluate_constraints</c> returns a sequence instead of a <c>dict</c>.
    /// </summary>
    private const string LegacySequenceWarning =
        "Returning a sequence from `evaluate_constraints` is deprecated in v0.5.0. " +
        "The benchmark package cached in your local directory is outdated. " +
        "Please update it by `optunahub.load_module(..., force_reload=True)`.";

    /// <summary>
    ///     OptunaHub 0.5.0's constraint step of <c>BaseProblem.__call__</c>:
    ///     <c>_as_constraints_dict(self.evaluate_constraints(params))</c>, followed by the <c>float(value)</c>
    ///     cast Optuna's <c>set_constraint</c> performs. Must run under the GIL.
    /// </summary>
    /// <remarks>
    ///     Returns no constraints when the installed OptunaHub predates 0.5.0 (its <c>BaseProblem</c> has no
    ///     <c>evaluate_constraints</c>), because that version's <c>__call__</c> never applied any. Python's
    ///     guard <c>if constraints and not hasattr(trial, "set_constraint")</c> has no C# counterpart: every
    ///     OptunaSharp <see cref="Trial" /> implements <see cref="Trial.SetConstraint" />.
    /// </remarks>
    /// <param name="problem">The borrowed Python problem instance.</param>
    /// <param name="pyParams">The borrowed params dict, the same object later passed to <c>evaluate</c>.</param>
    /// <returns>The constraint names and float values in the problem's order (dict insertion order, or sequence index).</returns>
    /// <exception cref="global::Python.Runtime.PythonException">The problem's <c>evaluate_constraints</c> raises, the result is neither a <c>dict</c> nor iterable, or a warnings filter escalates the legacy-sequence warning.</exception>
    /// <exception cref="NumSharp.TypeError">A value cannot be converted to <c>float</c>.</exception>
    private static IReadOnlyList<KeyValuePair<string, double>> EvaluateConstraints(PyObject problem, PyObject pyParams)
    {
        // Version probe on the CLASS, not the instance: an old ConstrainedMixin problem defines its own
        // evaluate_constraints, but only OptunaHub ≥ 0.5.0's BaseProblem.__call__ applies constraints.
        using PyObject benchmarks = OptunaHubRuntime.OptunaHub.GetAttr("benchmarks");
        using PyObject baseProblem = benchmarks.GetAttr("BaseProblem");
        if (!baseProblem.HasAttr("evaluate_constraints"))
            return Array.Empty<KeyValuePair<string, double>>();

        using PyObject raw = Python.optunahub.benchmarks.BaseProblem.evaluate_constraints(problem, pyParams);
        var constraints = new List<KeyValuePair<string, double>>();
        if (PyDict.IsDictType(raw))
        {
            // isinstance(constraints, dict): use it as is — insertion order, keys formatted the way
            // set_constraint's f"constraints:{key}" formats them (str()).
            using PyObject itemsFn = raw.GetAttr("items");
            using PyObject items = itemsFn.Invoke();
            using var iter = PyIter.GetIter(items);
            while (iter.MoveNext())
            {
                using PyObject kv = iter.Current;
                // The int indexer creates and releases its own index object (no leaked PyInt temporaries).
                using PyObject key = kv[0];
                using PyObject value = kv[1];
                constraints.Add(new KeyValuePair<string, double>(key.ToString() ?? "", ToConstraintFloat(value)));
            }
            return constraints;
        }

        // A sequence from a benchmark package written for an older OptunaHub: warn exactly like
        // _as_constraints_dict, then name each value by its index ({str(i): value for i, value in ...}).
        using (PyObject futureWarning = Python.builtins.FutureWarning)
            Python.warnings.warn(LegacySequenceWarning, futureWarning);
        using (var iter = PyIter.GetIter(raw))
        {
            int index = 0;
            while (iter.MoveNext())
            {
                using PyObject value = iter.Current;
                constraints.Add(new KeyValuePair<string, double>(
                    index.ToString(CultureInfo.InvariantCulture), ToConstraintFloat(value)));
                index++;
            }
        }
        return constraints;
    }

    /// <summary>
    ///     Optuna's <c>set_constraint</c> cast: <c>float(value)</c>, re-raising a <c>TypeError</c> or
    ///     <c>ValueError</c> as Optuna's own <c>TypeError</c> message. Must run under the GIL.
    /// </summary>
    /// <param name="value">The borrowed Python constraint value (a float, int, numpy scalar, …).</param>
    /// <returns>The value as a managed double (NaN passes through; <see cref="Trial.SetConstraint" /> rejects it).</returns>
    /// <exception cref="NumSharp.TypeError">The value cannot be converted to <c>float</c>.</exception>
    /// <exception cref="global::Python.Runtime.PythonException">The conversion fails with a Python error other than <c>TypeError</c>/<c>ValueError</c>.</exception>
    private static double ToConstraintFloat(PyObject value)
    {
        try
        {
            // Qualified: Core's internal OptunaSharp.Distributions.PyFloat is visible to this friend assembly too.
            using global::Python.Runtime.PyFloat converted = global::Python.Runtime.PyFloat.AsFloat(value);
            return converted.As<double>();
        }
        catch (PythonException ex) when (IsTypeOrValueError(ex))
        {
            // Optuna: f"The `value` argument is of type '{type(value)}' but supposed to be a float."
            using PyObject type = value.GetPythonType();
            throw new NumSharp.TypeError($"The `value` argument is of type '{type}' but supposed to be a float.");
        }
    }

    /// <summary>
    ///     Whether a Python error is a <c>TypeError</c> or <c>ValueError</c> (subclasses included), i.e. what
    ///     <c>except (TypeError, ValueError)</c> would catch. Must run under the GIL.
    /// </summary>
    /// <param name="ex">The Python exception raised by the conversion.</param>
    /// <returns><c>true</c> when the exception value is an instance of either built-in class.</returns>
    private static bool IsTypeOrValueError(PythonException ex)
    {
        ex.Normalize();   // make Value the exception instance before the isinstance checks
        if (ex.Value is null)
            return false;
        using PyObject typeError = Python.builtins.TypeError;
        using PyObject valueError = Python.builtins.ValueError;
        return ex.Value.IsInstance(typeError) || ex.Value.IsInstance(valueError);
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
