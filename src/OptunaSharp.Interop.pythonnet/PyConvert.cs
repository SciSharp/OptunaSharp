using System;
using System.Collections.Generic;
using Python.Runtime;
using OptunaSharp.Distributions;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     Marshals the small set of Optuna objects the OptunaHub bridge exchanges with Python — parameter
///     <b>values</b> (Python's <c>None</c>/<c>bool</c>/<c>int</c>/<c>float</c>/<c>str</c>, incl. the
///     numpy scalar spellings a sampler may return) and <b>distributions</b> — between CLR types and
///     live <c>optuna.distributions.*</c> objects.
///     <para>Every method must be called under the GIL (see <see cref="PythonRuntime.Run(Action)" />).</para>
/// </summary>
public static class PyConvert
{
    // ---- values ---------------------------------------------------------------------------------

    /// <summary>CLR parameter value (from a suggest call / <see cref="CategoricalChoice" />) → Python object.</summary>
    /// <param name="value">The parameter value or borrowed Python scalar to convert.</param>
    /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
    public static PyObject ToPy(object? value)
    {
        switch (value)
        {
            case null:
                return ((object?)null).ToPython(); // Python None
            case CategoricalChoice choice:
                return ToPy(choice.Value);
            case bool b:
                return b.ToPython();
            case string s:
                return s.ToPython();
            case double d:
                return d.ToPython();
            case float f:
                return ((double)f).ToPython();
            case sbyte or byte or short or ushort or int or uint or long:
                return Convert.ToInt64(value).ToPython();
            default:
                return value.ToPython();
        }
    }

    /// <summary>
    ///     Python value → CLR (<c>null</c>, <see cref="bool" />, <see cref="long" />,
    ///     <see cref="double" /> or <see cref="string" />), recognizing numpy scalar types too.
    /// </summary>
    /// <param name="value">The parameter value or borrowed Python scalar to convert.</param>
    /// <returns>A managed scalar, string, or null, detached from Python reference lifetime.</returns>
    public static object? ToClr(PyObject value)
    {
        if (value is null || value.IsNone())
            return null;

        string kind;
        using (var type = value.GetPythonType())
        using (var nameObj = type.GetAttr("__name__"))
            kind = nameObj.As<string>() ?? "";

        return kind switch
        {
            "bool" or "bool_" => value.As<bool>(),
            "int" or "int64" or "int32" or "int16" or "int8"
                or "uint64" or "uint32" or "uint16" or "uint8" or "long" => value.As<long>(),
            "float" or "float64" or "float32" or "float16" or "double" => value.As<double>(),
            "str" or "str_" => value.As<string>(),
            "NoneType" => null,
            // Fallback: try float, then string.
            _ => TryFloatElseString(value),
        };
    }

    /// <summary>Python value → a typed <see cref="CategoricalChoice" /> (Python-equality semantics apply downstream).</summary>
    /// <param name="value">The parameter value or borrowed Python scalar to convert.</param>
    /// <returns>A typed categorical value using OptunaSharp's Python-compatible equality rules.</returns>
    public static CategoricalChoice ToChoice(PyObject value) => CategoricalChoice.From(ToClr(value));

    /// <summary>Attempts numeric conversion before falling back to a Python string conversion.</summary>
    /// <param name="value">The parameter value or borrowed Python scalar to convert.</param>
    /// <returns>The converted floating-point value, or the fallback managed string.</returns>
    private static object? TryFloatElseString(PyObject value)
    {
        try { return value.As<double>(); }
        catch { return value.As<string>(); }
    }

    // ---- distributions --------------------------------------------------------------------------

    /// <summary>OptunaSharp distribution → a live <c>optuna.distributions.*</c> object (constructed
    /// through the <c>Python.optuna.distributions</c> facade).</summary>
    /// <param name="dist">The distribution defining value conversion or the borrowed Python distribution object.</param>
    /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
    /// <exception cref="OptunaSharp.Interop.PythonNet.OptunaHubInteropException">The distribution is not a supported float, integer, or categorical distribution.</exception>
    public static PyObject ToPy(BaseDistribution dist) => dist switch
    {
        FloatDistribution f => Python.optuna.distributions.FloatDistribution(f.Low, f.High, f.Log, f.Step),
        IntDistribution i => Python.optuna.distributions.IntDistribution(i.Low, i.High, i.Log, i.Step),
        CategoricalDistribution c => Python.optuna.distributions.CategoricalDistribution(c.Choices),
        _ => throw new OptunaHubInteropException(
            $"Cannot marshal distribution of type {dist.GetType().Name} to Python (supported: " +
            "FloatDistribution, IntDistribution, CategoricalDistribution)."),
    };

    /// <summary>A live <c>optuna.distributions.*</c> object → the matching OptunaSharp distribution.</summary>
    /// <param name="dist">The distribution defining value conversion or the borrowed Python distribution object.</param>
    /// <returns>A managed float, integer, or categorical distribution corresponding to the Python object.</returns>
    /// <exception cref="OptunaSharp.Interop.PythonNet.OptunaHubInteropException">The Python distribution type is unsupported.</exception>
    public static BaseDistribution ToClrDistribution(PyObject dist)
    {
        string name;
        using (var type = dist.GetPythonType())
        using (var nameObj = type.GetAttr("__name__"))
            name = nameObj.As<string>() ?? "";

        switch (name)
        {
            case "FloatDistribution":
            case "UniformDistribution":
            case "LogUniformDistribution":
            case "DiscreteUniformDistribution":
            {
                double low = dist.GetAttr("low").As<double>();
                double high = dist.GetAttr("high").As<double>();
                bool log = HasTrue(dist, "log");
                double? step = ReadNullableDouble(dist, "step");
                return new FloatDistribution(low, high, log, step);
            }
            case "IntDistribution":
            case "IntUniformDistribution":
            case "IntLogUniformDistribution":
            {
                long low = dist.GetAttr("low").As<long>();
                long high = dist.GetAttr("high").As<long>();
                bool log = HasTrue(dist, "log");
                long step = ReadNullableLong(dist, "step") ?? 1L;
                return new IntDistribution(low, high, log, step);
            }
            case "CategoricalDistribution":
            {
                using var choices = dist.GetAttr("choices");
                long n = choices.Length();
                var list = new List<CategoricalChoice>((int)n);
                for (long k = 0; k < n; k++)
                {
                    using var item = choices[new PyInt(k)];
                    list.Add(ToChoice(item));
                }
                return new CategoricalDistribution(list);
            }
            default:
                throw new OptunaHubInteropException(
                    $"Cannot marshal Python distribution '{name}' to OptunaSharp (supported: Float/Int/Categorical).");
        }
    }

    /// <summary>Reads an optional Boolean attribute, treating a missing attribute or None as false.</summary>
    /// <param name="obj">The borrowed Python object whose optional attribute is inspected.</param>
    /// <param name="attr">The case-sensitive Python attribute name.</param>
    /// <returns>Whether the attribute exists, is not None, and converts to true.</returns>
    private static bool HasTrue(PyObject obj, string attr)
    {
        if (!obj.HasAttr(attr))
            return false;
        using var a = obj.GetAttr(attr);
        return !a.IsNone() && a.As<bool>();
    }

    /// <summary>Reads an optional floating-point attribute, treating a missing attribute or None as null.</summary>
    /// <param name="obj">The borrowed Python object whose optional attribute is inspected.</param>
    /// <param name="attr">The case-sensitive Python attribute name.</param>
    /// <returns>The floating-point attribute value, or null for a missing attribute or None.</returns>
    private static double? ReadNullableDouble(PyObject obj, string attr)
    {
        if (!obj.HasAttr(attr))
            return null;
        using var a = obj.GetAttr(attr);
        return a.IsNone() ? null : a.As<double>();
    }

    /// <summary>Reads an optional integer attribute, treating a missing attribute or None as null.</summary>
    /// <param name="obj">The borrowed Python object whose optional attribute is inspected.</param>
    /// <param name="attr">The case-sensitive Python attribute name.</param>
    /// <returns>The integer attribute value, or null for a missing attribute or None.</returns>
    private static long? ReadNullableLong(PyObject obj, string attr)
    {
        if (!obj.HasAttr(attr))
            return null;
        using var a = obj.GetAttr(attr);
        return a.IsNone() ? null : a.As<long>();
    }
}
